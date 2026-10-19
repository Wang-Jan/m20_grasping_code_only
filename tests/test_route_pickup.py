import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m20_rebot_moving_grasp.machine_dog.route_pickup import (
    MapPose, PickupOutcome, RouteDetectionState, RoutePickupController, RouteStopRequested, load_route_definition,
)
from m20_rebot_moving_grasp.machine_dog import ApproachResult, M20Config
from m20_rebot_moving_grasp.machine_dog.d455_vision import D455Detection


def bottle() -> D455Detection:
    return D455Detection(
        "bottle", 0.9, (320.0, 240.0), 1.2, (0.0, 0.0, 1.2),
        relative_xy_m=(1.2, -0.1), range_m=1.2,
    )


def points() -> dict:
    return {
        "start": {"point_info": 1, "x": 0, "y": 0, "z": 0, "yaw": 0},
        "route_a": {"point_id": 11, "point_info": 1, "x": 1, "y": 2, "z": 0, "yaw": 0},
        "bin_point": {"point_id": 22, "point_info": 1, "x": 3, "y": 4, "z": 0, "yaw": 0},
    }


def route_config(tmp: Path) -> tuple[Path, Path]:
    routes = tmp / "routes.json"
    saved_points = tmp / "points.json"
    routes.write_text(json.dumps({"daily": ["route_a"]}), encoding="utf-8")
    saved_points.write_text(json.dumps(points()), encoding="utf-8")
    return routes, saved_points


class FakeD455:
    def __init__(self, result=ApproachResult.PRONE_AT_TARGET):
        self.result = result
        self.approach = Mock(return_value=result)
        self.scan_start = Mock()
        self.scan_stop = Mock()
        self.scan_handler = Mock(return_value=())


class FakeArm:
    def __init__(self, *, prepare=True, grasp=True, hold=True, drop=True, home=True, recoverable_no_grasp=False):
        self.init_reset = Mock(return_value=True)
        self.prepare_grasp = Mock(return_value=prepare)
        self.grasp_current_target = Mock(return_value=grasp)
        self.hold = Mock(return_value=hold)
        self.drop = Mock(return_value=drop)
        self.safe_home = Mock(return_value=home)
        self.release_vision = Mock()
        self.shutdown = Mock()
        self.is_recoverable_detection_failure = recoverable_no_grasp
        self.last_failure_reason = "D435 中心帧及六视角补偿均未获得可行抓取位姿" if recoverable_no_grasp else ""


class RoutePickupTest(unittest.TestCase):
    def test_project_routes_do_not_embed_a_fixed_scan_point(self):
        root = Path(__file__).resolve().parents[1]
        routes = json.loads((root / "config/m20/routes.json").read_text(encoding="utf-8"))
        self.assertTrue(routes)
        self.assertTrue(all("scan_point" not in names for names in routes.values()))

    def test_coordinate_only_route_point_does_not_require_point_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "routes.json").write_text(json.dumps({"r": ["origin"]}), encoding="utf-8")
            (root / "points.json").write_text(
                json.dumps({"origin": {"x": 0, "y": 0, "z": 0, "yaw": 0}}),
                encoding="utf-8",
            )
            route, saved_points = load_route_definition(root / "routes.json", root / "points.json", "r")
        self.assertEqual(route, ["origin"])
        self.assertEqual(saved_points["origin"]["point_info"], 1)
        self.assertNotIn("point_id", saved_points["origin"])

    def test_missing_route_point_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "routes.json").write_text(json.dumps({"r": ["missing"]}), encoding="utf-8")
            (root / "points.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不存在"):
                load_route_definition(root / "routes.json", root / "points.json", "r")

    def test_detection_state_accepts_only_first_detection_with_pose(self):
        class Detection:
            relative_xy_m = (1.0, 0.0)
            range_m = 1.0
            depth_m = 1.0

        state = RouteDetectionState()
        pose = MapPose(1.0, 2.0, 0.0, 0.1)
        self.assertTrue(state.accept([Detection()], pose))
        self.assertFalse(state.accept([Detection()], pose))
        value = state.take()
        self.assertIsNotNone(value)
        self.assertEqual(value.pose, pose)
        self.assertFalse(state.event.is_set())

    def test_detection_without_pose_is_ignored(self):
        class Detection:
            relative_xy_m = (1.0, 0.0)
            range_m = 1.0
            depth_m = 1.0

        state = RouteDetectionState()
        self.assertFalse(state.accept([Detection()], None))
        self.assertIsNone(state.take())

    def controller(self, directory: Path, transport=object(), d455=None):
        routes, saved_points = route_config(directory)
        return RoutePickupController(
            {
                "route_loops": 1,
                "route_nav_timeout_sec": 0.2,
                "pose_interval_sec": 0.1,
                "pickup_settle_seconds": 0.0,
                "bin_settle_seconds": 0.0,
                "pose_log": str(directory / "poses.jsonl"),
                "resume_log": str(directory / "resume.json"),
                "missed_detection_log": str(directory / "d435_misses.jsonl"),
                "d435_miss_suppression_radius_m": 0.75,
                "resume_point_id": 902,
                "current_loop": 1,
                "current_index": 0,
                "current_point": "route_a",
            },
            M20Config(command_timeout_sec=0.1, cancel_confirm_timeout_sec=0.1).validate(),
            transport,
            d455 or FakeD455(),
            routes_path=routes,
            points_path=saved_points,
            route_name="daily",
        )

    def test_stop_interrupts_navigation_wait_and_cleans_up(self):
        for immediate_ack in (False, True):
            with self.subTest(immediate_ack=immediate_ack), tempfile.TemporaryDirectory() as directory:
                controller = self.controller(Path(directory))
                requested, release = threading.Event(), threading.Event()
                controller.stop_requested = requested.is_set
                controller.config["route_nav_timeout_sec"] = 30
                def request(payload, **kwargs):
                    command = json.loads(payload)["PatrolDevice"]
                    if command["Type"] == 1003:
                        if not immediate_ack:
                            requested.set()
                            release.wait(2)
                        return {"PatrolDevice": {"Items": {"ErrorCode": 0}}}
                    requested.set()
                    return {"PatrolDevice": {"Items": {"Status": 1}}}
                dog, arm = Mock(), FakeArm()
                began = time.monotonic()
                try:
                    with patch.object(controller, "_request", side_effect=request), \
                         patch.object(controller, "_start_publisher"), \
                         patch.object(controller, "_start_scanner"), \
                         patch.object(controller, "_stop_scanner"), \
                         patch.object(controller, "_stop_publisher"), \
                         patch.object(controller, "_cancel_and_lie", return_value=True) as cancel:
                        self.assertFalse(controller.run(dog, arm))
                    self.assertLess(time.monotonic() - began, 2)
                    cancel.assert_called_once()
                    arm.shutdown.assert_called_once()
                    self.assertTrue(controller.safe_stop_completed)
                    dog.goto_home.assert_not_called()
                finally:
                    release.set()

    def test_stop_at_arm_boundary_does_not_begin_grasp(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(Path(directory))
            requested = threading.Event()
            controller.stop_requested = requested.is_set
            controller.detection_state.accept([bottle()], MapPose(1, 2, 0, 0))
            arm = FakeArm()
            def prepare():
                requested.set()
                return True
            arm.prepare_grasp.side_effect = prepare
            with self.assertRaises(RouteStopRequested):
                controller._handle_detection(arm, points()["bin_point"], MapPose(1, 2, 0, 0), Path(directory)/"resume.json")
            arm.grasp_current_target.assert_not_called()

    def test_failed_stop_cleanup_is_not_reported_as_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(Path(directory))
            controller.stop_requested = lambda: True
            arm = FakeArm()
            with patch.object(controller, "_cancel_and_lie", return_value=False):
                self.assertFalse(controller.run(Mock(), arm))
            self.assertFalse(controller.safe_stop_completed)
            arm.init_reset.assert_not_called()
            arm.shutdown.assert_called_once()

    def test_detection_cancels_long_route_request_without_waiting_for_navigation_timeout(self):
        class Transport:
            def __init__(self):
                self.started = threading.Event()
                self.requests = []
            def send_request(self, payload, timeout, **_kwargs):
                item = json.loads(payload)["PatrolDevice"]
                self.requests.append(item)
                if item["Type"] == 1003:
                    self.started.set()
                    time.sleep(0.08)
                    return None
                if item["Type"] == 1004:
                    return {"PatrolDevice": {"Items": {"ErrorCode": 0}}}
                if item["Type"] == 1007:
                    return {"PatrolDevice": {"Items": {"ErrorCode": 0x2302}}}
                raise AssertionError("unexpected request")

        with tempfile.TemporaryDirectory() as directory:
            transport = Transport()
            controller = self.controller(Path(directory), transport=transport)
            trigger = threading.Thread(
                target=lambda: (transport.started.wait(0.2), controller.detection_state.accept([bottle()], MapPose(0, 0, 0, 0))),
            )
            trigger.start()
            self.assertFalse(controller._navigate(controller.points["route_a"], point_id=None, point_info=1,
                                                  label="route_a", interruptible=True))
            trigger.join()
            self.assertTrue(any(request["Type"] == 1004 for request in transport.requests))

    def test_route_and_bin_use_coordinate_navigation_value_zero(self):
        class Transport:
            def __init__(self): self.payloads = []
            def send_request(self, payload, timeout, **_kwargs):
                self.payloads.append(json.loads(payload))
                return {"PatrolDevice": {"Items": {"ErrorCode": 0x2300}}}

        with tempfile.TemporaryDirectory() as directory:
            transport = Transport()
            controller = self.controller(Path(directory), transport=transport)
            self.assertTrue(controller._navigate(controller.points["route_a"], point_id=None,
                                                 point_info=1, label="route_a"))
            self.assertTrue(controller._navigate(controller.points["bin_point"], point_id=None,
                                                 point_info=1, label="bin_point"))
        items = [payload["PatrolDevice"]["Items"] for payload in transport.payloads]
        self.assertEqual([item["Value"] for item in items], [0, 0])
        self.assertEqual((items[0]["PosX"], items[0]["PosY"]), (1.0, 2.0))
        self.assertEqual((items[1]["PosX"], items[1]["PosY"]), (3.0, 4.0))

    def test_route_abort_cancels_navigation_then_lies_in_place(self):
        class Transport:
            def __init__(self):
                self.payloads = []
                self.responses = [
                    {"PatrolDevice": {"Items": {"ErrorCode": 0}}},
                    {"PatrolDevice": {"Items": {"ErrorCode": 0x2302}}},
                    {"PatrolDevice": {"Items": {"ErrorCode": 0}}},
                ]

            def send_request(self, payload, timeout, **_kwargs):
                self.payloads.append(json.loads(payload))
                return self.responses.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            transport = Transport()
            controller = self.controller(Path(directory), transport=transport)
            self.assertTrue(controller._cancel_and_lie("test"))
        requests = [payload["PatrolDevice"] for payload in transport.payloads]
        self.assertEqual([(request["Type"], request["Command"]) for request in requests], [(1004, 1), (1007, 1), (2, 22)])
        self.assertEqual(requests[-1]["Items"]["MotionParam"], 4)

    def test_route_cancel_accepts_completed_navigation_without_waiting(self):
        class Transport:
            def __init__(self):
                self.payloads = []
                self.responses = [
                    {"PatrolDevice": {"Items": {"ErrorCode": 0}}},
                    {"PatrolDevice": {"Items": {"Value": 902, "Status": 0, "ErrorCode": 0x2300}}},
                ]

            def send_request(self, payload, timeout, **_kwargs):
                self.payloads.append(json.loads(payload))
                return self.responses.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            transport = Transport()
            controller = self.controller(Path(directory), transport=transport)
            self.assertTrue(controller._cancel_navigation())
        self.assertEqual(len(transport.payloads), 2)

    def test_pickup_passes_detection_pose_to_approach_then_bin_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            d455 = FakeD455()
            controller = self.controller(root, d455=d455)
            controller.detection_state.accept([bottle()], MapPose(1.0, 2.0, 0.0, 0.2))
            arm = FakeArm()
            resume = root / "resume.json"
            with patch.object(controller, "_motion", return_value=True), \
                 patch.object(controller, "_navigate", return_value=True) as navigate, \
                 patch("m20_rebot_moving_grasp.machine_dog.route_pickup.time.sleep"):
                self.assertIs(controller._handle_detection(arm, controller.points["bin_point"],
                                                           MapPose(1.2, 2.1, 0.0, 0.3), resume),
                              PickupOutcome.COMPLETED)
            d455.approach.assert_called_once_with(map_pose=(1.0, 2.0, 0.0, 0.2))
            self.assertEqual([call.kwargs["point_id"] for call in navigate.call_args_list], [None, 902])
            self.assertEqual([call.kwargs["label"] for call in navigate.call_args_list], ["bin_point", "恢复中断位姿"])
            arm.prepare_grasp.assert_called_once()
            arm.grasp_current_target.assert_called_once()
            arm.hold.assert_called_once()
            arm.drop.assert_called_once()
            arm.safe_home.assert_called_once()
            self.assertFalse(resume.exists())

    def test_failed_drop_preserves_resume_record_and_does_not_resume_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.controller(root)
            controller.detection_state.accept([bottle()], MapPose(1.0, 2.0, 0.0, 0.2))
            arm = FakeArm(drop=False)
            resume = root / "resume.json"
            with patch.object(controller, "_motion", return_value=True), \
                 patch.object(controller, "_navigate", return_value=True) as navigate, \
                 patch("m20_rebot_moving_grasp.machine_dog.route_pickup.time.sleep"):
                self.assertIs(controller._handle_detection(arm, controller.points["bin_point"],
                                                           MapPose(1.2, 2.1, 0.0, 0.3), resume),
                              PickupOutcome.FAILED)
            self.assertTrue(resume.exists())
            self.assertEqual(navigate.call_count, 1)

    def test_no_feasible_grasp_recovers_to_902_records_and_skips_bin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.controller(root)
            detection_pose = MapPose(1.0, 2.0, 0.0, 0.2)
            controller.detection_state.accept([bottle()], detection_pose)
            arm = FakeArm(prepare=False, recoverable_no_grasp=True)
            resume = root / "resume.json"
            with patch.object(controller, "_motion", return_value=True) as motion, \
                 patch.object(controller, "_navigate", return_value=True) as navigate, \
                 patch("m20_rebot_moving_grasp.machine_dog.route_pickup.time.sleep"):
                result = controller._handle_detection(arm, controller.points["bin_point"],
                                                      MapPose(1.2, 2.1, 0.0, 0.3), resume)
            self.assertIs(result, PickupOutcome.NO_FEASIBLE_GRASP_RECOVERED)
            self.assertEqual([call.kwargs["label"] for call in navigate.call_args_list], ["D435 无可行抓取位姿恢复中断位姿"])
            self.assertEqual(motion.call_args_list[0].args, (1,))
            arm.grasp_current_target.assert_not_called()
            arm.hold.assert_not_called()
            arm.drop.assert_not_called()
            arm.safe_home.assert_called_once()
            arm.release_vision.assert_called_once()
            self.assertFalse(resume.exists())
            missed = root / "d435_misses.jsonl"
            payload = json.loads(missed.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "d435_no_feasible_grasp_recovered")
            self.assertEqual(payload["detection_pose"], detection_pose.__dict__)

    def test_no_feasible_grasp_resume_failure_keeps_resume_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.controller(root)
            controller.detection_state.accept([bottle()], MapPose(1.0, 2.0, 0.0, 0.2))
            arm = FakeArm(prepare=False, recoverable_no_grasp=True)
            resume = root / "resume.json"
            with patch.object(controller, "_motion", return_value=True), \
                 patch.object(controller, "_navigate", return_value=False), \
                 patch("m20_rebot_moving_grasp.machine_dog.route_pickup.time.sleep"):
                result = controller._handle_detection(arm, controller.points["bin_point"],
                                                      MapPose(1.2, 2.1, 0.0, 0.3), resume)
            self.assertIs(result, PickupOutcome.FAILED)
            self.assertTrue(resume.exists())
            self.assertEqual(json.loads(resume.read_text(encoding="utf-8"))["status"], "d435_no_feasible_grasp_recovery_pending")

    def test_missed_target_is_suppressed_until_robot_leaves_radius(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(Path(directory))
            controller._suppressed_detection_pose = MapPose(1.0, 2.0, 0.0, 0.0)
            self.assertFalse(controller._accept_route_detection([bottle()], MapPose(1.5, 2.0, 0.0, 0.0)))
            self.assertFalse(controller.detection_state.event.is_set())
            self.assertTrue(controller._accept_route_detection([bottle()], MapPose(1.8, 2.0, 0.0, 0.0)))
            self.assertTrue(controller.detection_state.event.is_set())

    def test_arm_motion_failure_does_not_resume_or_navigate_to_bin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.controller(root)
            controller.detection_state.accept([bottle()], MapPose(1.0, 2.0, 0.0, 0.2))
            arm = FakeArm(prepare=False, recoverable_no_grasp=False)
            arm.last_failure_reason = "D435 补偿1/6 关节轨迹未完成"
            resume = root / "resume.json"
            with patch.object(controller, "_motion", return_value=True), \
                 patch.object(controller, "_navigate", return_value=True) as navigate, \
                 patch("m20_rebot_moving_grasp.machine_dog.route_pickup.time.sleep"):
                result = controller._handle_detection(
                    arm,
                    controller.points["bin_point"],
                    MapPose(1.2, 2.1, 0.0, 0.3),
                    resume,
                )
            self.assertIs(result, PickupOutcome.FAILED)
            navigate.assert_not_called()
            arm.safe_home.assert_called_once()
            arm.release_vision.assert_called_once()
            self.assertTrue(resume.exists())

    def test_route_skips_interrupted_point_after_recovered_no_feasible_grasp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.controller(root)
            controller.route = ["route_a", "route_a"]
            controller.detection_state.event.set()
            dog = type("Dog", (), {
                "init_reset": Mock(return_value=True),
                "stand": Mock(return_value=True),
                "goto_home": Mock(return_value=True),
            })()
            arm = FakeArm()

            def consume(*_args):
                controller.detection_state.event.clear()
                return PickupOutcome.NO_FEASIBLE_GRASP_RECOVERED

            with patch.object(controller, "_start_publisher"), \
                 patch.object(controller, "_start_scanner"), \
                 patch.object(controller, "_stop_publisher"), \
                 patch.object(controller, "_stop_scanner"), \
                 patch.object(controller, "_consume_detection", side_effect=consume), \
                 patch.object(controller, "_navigate", return_value=True) as navigate, \
                 patch.object(controller, "_motion", return_value=True):
                self.assertTrue(controller.run(dog, arm))
            self.assertEqual(navigate.call_count, 2)
            self.assertEqual(navigate.call_args_list[0].kwargs["label"], "route_a")
            self.assertEqual(navigate.call_args_list[1].kwargs["label"], "Home")


if __name__ == "__main__":
    unittest.main()
