"""D455 YOLOE 与内置 M20 接近流程的无硬件单元测试。"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from m20_rebot_moving_grasp.common.config import ConfigError, load_config
from m20_rebot_moving_grasp.machine_dog import ApproachResult, M20Config
from m20_rebot_moving_grasp.machine_dog.d455_approach import D455ApproachController
from m20_rebot_moving_grasp.machine_dog.d455_vision import (
    D455Config, D455Detection, D455Vision, camera_point_to_robot_aligned_xy,
    choose_nearest_detection, depth_patch_values, mask_bottom_contact_pixel, valid_depth_median,
)


def config_mapping() -> dict:
    return {
        "camera": {"serial": "d455-test", "color_width": 640, "color_height": 480,
                   "depth_width": 640, "depth_height": 480, "fps": 30},
        "detection": {"model_name": "yoloe-26l-seg.pt", "device": "0",
                      "confidence_threshold": 0.25, "iou_threshold": 0.45,
                      "custom_classes": ["water bottle", "bottle"]},
        "target": {"min_depth_m": 0.2, "max_depth_m": 8.0,
                   "safe_stop_distance_m": 0.2, "approach_trigger_max_range_m": 1.25,
                   "max_navigation_sec": 30.0},
        "navigation": {"base_link_offset_x_m": 0.0, "base_link_offset_y_m": 0.0,
                       "pre_lie_forward_m": 0.10, "perpendicular_side": "left",
                       "approach_point_id": 901, "gait": 0x3002, "speed": 1,
                       "obs_mode": 0, "nav_mode": 0},
    }


def detection(depth: float, relative_xy: tuple[float, float] | None = None) -> D455Detection:
    return D455Detection(
        "bottle", 0.9, (320.0, 240.0), depth, (0.0, 0.0, depth),
        relative_xy_m=relative_xy, range_m=depth,
    )


def response(items: dict) -> dict:
    return {"PatrolDevice": {"Items": items}}


def pose(x: float, y: float, yaw: float, z: float = 0.0) -> dict:
    return response({"ErrorCode": 0, "Location": 0, "PosX": x, "PosY": y, "PosZ": z, "Yaw": yaw})


def perception(*, location: int = 0, obstacle: int = 0) -> dict:
    return response({"ErrorCode": 0, "Location": location, "ObsState": obstacle})


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.timeouts: list[float] = []

    def send_request(self, payload, timeout):
        self.requests.append(json.loads(payload))
        self.timeouts.append(timeout)
        if not self.responses:
            raise AssertionError("unexpected M20 request")
        return self.responses.pop(0)


class VisionTest(unittest.TestCase):
    def test_project_d455_uses_15cm_safe_stop_distance(self):
        config = D455Config.from_mapping(load_config(PROJECT_ROOT / "config/d455.yaml"))
        self.assertAlmostEqual(config.safe_stop_distance_m, 0.15)
        self.assertAlmostEqual(config.approach_trigger_max_range_m, 1.25)
        self.assertAlmostEqual(config.pre_lie_forward_m, 0.10)
        self.assertFalse(hasattr(config, "goal_tolerance_m"))
        self.assertFalse(hasattr(config, "yaw_tolerance_rad"))
        self.assertEqual(config.perpendicular_side, "left")
        self.assertAlmostEqual(config.max_navigation_sec, 300.0)

    def test_config_rejects_missing_serial_and_invalid_depth_range(self):
        cfg = config_mapping(); cfg["camera"]["serial"] = ""
        with self.assertRaisesRegex(ConfigError, "serial"):
            D455Config.from_mapping(cfg)
        cfg = config_mapping(); cfg["target"]["max_depth_m"] = 0.1
        with self.assertRaisesRegex(ConfigError, "深度范围"):
            D455Config.from_mapping(cfg)
        cfg = config_mapping(); cfg["camera"]["frame_timeout_sec"] = 0
        with self.assertRaisesRegex(ConfigError, "frame_timeout_sec"):
            D455Config.from_mapping(cfg)
        cfg = config_mapping(); cfg["navigation"]["pre_lie_forward_m"] = -0.01
        with self.assertRaisesRegex(ConfigError, "接近距离"):
            D455Config.from_mapping(cfg)
        cfg = config_mapping(); cfg["target"]["approach_trigger_max_range_m"] = 0.2
        with self.assertRaisesRegex(ConfigError, "近距触发距离"):
            D455Config.from_mapping(cfg)
        cfg = config_mapping(); cfg["target"]["approach_trigger_max_range_m"] = float("inf")
        with self.assertRaisesRegex(ConfigError, "近距触发距离"):
            D455Config.from_mapping(cfg)

    def test_warmup_uses_configured_frame_timeout(self):
        cfg = config_mapping(); cfg["camera"]["frame_timeout_sec"] = 12.5

        class Pipeline:
            def __init__(self): self.timeouts = []
            def wait_for_frames(self, timeout_ms): self.timeouts.append(timeout_ms); return object()

        pipeline = Pipeline()
        vision = D455Vision(D455Config.from_mapping(cfg), model=object(), pipeline=pipeline)
        vision.warm_up(2)
        self.assertEqual(pipeline.timeouts, [12500, 12500])

    def test_valid_depth_filters_invalid_and_out_of_range_values(self):
        cfg = D455Config.from_mapping(config_mapping())
        values = np.array([np.nan, 0.1, 0.4, 0.6, 9.0], dtype=np.float32)
        self.assertAlmostEqual(valid_depth_median(values, cfg), 0.5)
        self.assertIsNone(valid_depth_median(np.array([0.0, 10.0]), cfg))

    def test_nearest_target_is_selected(self):
        self.assertEqual(choose_nearest_detection([detection(2.0), detection(0.5)]).depth_m, 0.5)
        self.assertIsNone(choose_nearest_detection([]))

    def test_nearest_target_max_range_uses_3d_range_without_depth_fallback(self):
        at_boundary = detection(1.25)
        beyond_boundary = detection(1.26)
        missing_range = D455Detection(
            "bottle", 0.9, (320.0, 240.0), 0.5, (0.0, 0.0, 0.5), range_m=None,
        )
        self.assertIs(choose_nearest_detection([at_boundary, beyond_boundary], max_range_m=1.25), at_boundary)
        self.assertIsNone(choose_nearest_detection([beyond_boundary], max_range_m=1.25))
        self.assertIsNone(choose_nearest_detection([missing_range], max_range_m=1.25))

    def test_d455_45_degree_horizontal_geometry(self):
        x, y = camera_point_to_robot_aligned_xy(0.2, 0.0, 2.0, 45.0)
        self.assertAlmostEqual(x, 2.0 / 2**0.5, places=6)
        self.assertAlmostEqual(y, -0.2, places=6)

    def test_mask_bottom_contact_and_depth_patch(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:7, 3:6] = 1
        self.assertEqual(mask_bottom_contact_pixel(mask, (1, 1, 7, 7)), (4.0, 6.0))
        depth = np.arange(64, dtype=np.float32).reshape(8, 8)
        self.assertEqual(depth_patch_values(depth, (0, 0), 1).tolist(), [0.0, 1.0, 8.0, 9.0])

    def test_injected_model_receives_open_vocabulary_classes(self):
        class Model:
            def __init__(self): self.classes = None
            def set_classes(self, classes): self.classes = classes
        model = Model()
        D455Vision(D455Config.from_mapping(config_mapping()), model=model, pipeline=object()).open()
        self.assertEqual(model.classes, ["water bottle", "bottle"])


class ControllerTest(unittest.TestCase):
    def controller(self, responses):
        cfg = D455Config.from_mapping(config_mapping())
        m20 = M20Config(command_timeout_sec=0.1, cancel_confirm_timeout_sec=0.1).validate()
        transport = FakeTransport(responses)
        controller = D455ApproachController(cfg, m20, transport, sleep=lambda _: None)
        return controller, transport

    def normal_responses(self):
        yaw = math.pi / 2.0
        return [
            pose(0.0, 0.0, 0.0), perception(), response({"ErrorCode": 0}),
            response({"Value": 901, "Status": 4, "ErrorCode": 0}),
            pose(1.8, 0.0, yaw), perception(), response({"ErrorCode": 0}),
            response({"Value": 902, "Status": 4, "ErrorCode": 0}),
            pose(1.8, 0.10, yaw), perception(), response({"ErrorCode": 0}),
        ]

    def test_approach_uses_verified_standoff_forward_segment_and_prone(self):
        controller, transport = self.controller(self.normal_responses())
        controller._last_detections = (detection(1.2, (2.0, 0.0)),)
        self.assertIs(controller.approach(), ApproachResult.PRONE_AT_TARGET)
        navs = [request["PatrolDevice"]["Items"] for request in transport.requests
                if request["PatrolDevice"]["Type"] == 1003]
        self.assertEqual([item["Value"] for item in navs], [901, 902])
        self.assertAlmostEqual(navs[0]["PosX"], 1.8)
        self.assertAlmostEqual(navs[0]["PosY"], 0.0)
        # Type=1003 intentionally uses the verified CLI's %f encoding.
        self.assertAlmostEqual(navs[0]["AngleYaw"], math.pi / 2.0, places=6)
        self.assertAlmostEqual(navs[1]["PosX"], 1.8)
        self.assertAlmostEqual(navs[1]["PosY"], 0.10)
        self.assertAlmostEqual(navs[1]["AngleYaw"], math.pi / 2.0, places=6)
        self.assertEqual((navs[0]["Gait"], navs[0]["Speed"], navs[0]["ObsMode"], navs[0]["NavMode"]),
                         (0x3002, 1, 0, 0))
        motions = [request["PatrolDevice"]["Items"]["MotionParam"] for request in transport.requests
                   if request["PatrolDevice"]["Type"] == 2]
        self.assertEqual(motions, [4])

    def test_approach_uses_injected_detection_pose_without_requerying_initial_location(self):
        # Route pickup pairs the first D455 detection with the cached map pose
        # before cancelling cruise navigation.  That pose must determine 901;
        # a later location read is only for the 902 forward segment/final check.
        responses = self.normal_responses()[1:]
        controller, transport = self.controller(responses)
        controller._last_detections = (detection(1.2, (2.0, 0.0)),)
        self.assertIs(
            controller.approach(map_pose=(5.0, 6.0, 0.0, 0.0)),
            ApproachResult.PRONE_AT_TARGET,
        )
        self.assertEqual(transport.requests[0]["PatrolDevice"]["Type"], 2002)
        first_nav = next(request["PatrolDevice"]["Items"] for request in transport.requests
                         if request["PatrolDevice"]["Type"] == 1003)
        self.assertAlmostEqual(first_nav["PosX"], 6.8)
        self.assertAlmostEqual(first_nav["PosY"], 6.0)

    def test_no_target_or_invalid_target_never_sends_navigation(self):
        controller, transport = self.controller([])
        self.assertIs(controller.approach(), ApproachResult.NO_TARGET)
        self.assertEqual(transport.requests, [])
        controller._last_detections = (detection(0.1, (-0.1, 0.0)),)
        self.assertIs(controller.approach(), ApproachResult.INVALID_TARGET)
        self.assertEqual(transport.requests, [])

    def test_approach_rejects_cached_target_beyond_trigger_range_without_navigation(self):
        controller, transport = self.controller([])
        controller._last_detections = (detection(1.26, (1.26, 0.0)),)
        self.assertIs(controller.approach(), ApproachResult.NO_TARGET)
        self.assertEqual(transport.requests, [])

    def test_obstacle_blocks_navigation_before_command(self):
        controller, transport = self.controller([pose(0.0, 0.0, 0.0), perception(obstacle=1)])
        controller._last_detections = (detection(1.2, (2.0, 0.0)),)
        self.assertIs(controller.approach(), ApproachResult.OBSTACLE_DETECTED)
        self.assertFalse(any(request["PatrolDevice"]["Type"] == 1003 for request in transport.requests))

    def test_navigation_rejection_cancels_and_lies_without_grasp(self):
        controller, transport = self.controller([
            pose(0.0, 0.0, 0.0), perception(), response({"ErrorCode": 1}),
            response({"ErrorCode": 0}), response({"ErrorCode": 0x2302}), response({"ErrorCode": 0}),
        ])
        controller._last_detections = (detection(1.2, (2.0, 0.0)),)
        self.assertIs(controller.approach(), ApproachResult.MOTION_FAILED)
        types = [(request["PatrolDevice"]["Type"], request["PatrolDevice"]["Command"])
                 for request in transport.requests]
        self.assertIn((1004, 1), types)
        motions = [request["PatrolDevice"]["Items"]["MotionParam"] for request in transport.requests
                   if request["PatrolDevice"]["Type"] == 2]
        self.assertEqual(motions, [4])

    def test_initial_navigation_response_timeout_is_reported_as_timeout(self):
        controller, transport = self.controller([
            pose(0.0, 0.0, 0.0), perception(), None,
            response({"ErrorCode": 0}), response({"ErrorCode": 0x2302}), response({"ErrorCode": 0}),
        ])
        controller._last_detections = (detection(1.2, (2.0, 0.0)),)
        self.assertIs(controller.approach(), ApproachResult.TIMEOUT)
        motions = [request["PatrolDevice"]["Items"]["MotionParam"] for request in transport.requests
                   if request["PatrolDevice"]["Type"] == 2]
        self.assertEqual(motions, [4])

    def test_terminal_navigation_response_uses_motion_timeout_without_status_poll(self):
        controller, transport = self.controller([response({"Value": 901, "Status": 4, "ErrorCode": 0x2300})])
        self.assertTrue(controller._submit_navigation(901, 1.0, 2.0, 0.0, 0.5))
        self.assertEqual(transport.timeouts, [controller.config.max_navigation_sec])
        self.assertIsNone(controller._wait_navigation(901))
        self.assertEqual(len(transport.requests), 1)

    def test_cancel_accepts_completed_navigation_without_waiting(self):
        controller, transport = self.controller([
            response({"ErrorCode": 0}),
            response({"Value": 902, "Status": 0, "ErrorCode": 0x2300}),
        ])
        self.assertTrue(controller._cancel_and_confirm())
        self.assertEqual(len(transport.requests), 2)

    def test_navigation_timeout_cancels_and_lies_without_grasp(self):
        cfg_map = config_mapping()
        cfg_map["target"]["max_navigation_sec"] = 0.01
        cfg = D455Config.from_mapping(cfg_map)
        m20 = M20Config(command_timeout_sec=0.1, cancel_confirm_timeout_sec=0.1).validate()
        transport = FakeTransport([
            pose(0.0, 0.0, 0.0), perception(), response({"ErrorCode": 0}),
            response({"ErrorCode": 0}), response({"ErrorCode": 0x2302}), response({"ErrorCode": 0}),
        ])
        ticks = iter((0.0, 0.02, 0.0, 0.0))
        controller = D455ApproachController(cfg, m20, transport, sleep=lambda _: None, monotonic=lambda: next(ticks))
        controller._last_detections = (detection(1.2, (2.0, 0.0)),)
        self.assertIs(controller.approach(), ApproachResult.TIMEOUT)
        motions = [request["PatrolDevice"]["Items"]["MotionParam"] for request in transport.requests
                   if request["PatrolDevice"]["Type"] == 2]
        self.assertEqual(motions, [4])

    def test_stop_request_cancels_then_lies(self):
        cfg = D455Config.from_mapping(config_mapping())
        m20 = M20Config(command_timeout_sec=0.1, cancel_confirm_timeout_sec=0.1).validate()
        transport = FakeTransport([
            response({"ErrorCode": 0}), response({"ErrorCode": 0x2302}), response({"ErrorCode": 0}),
        ])
        controller = D455ApproachController(
            cfg, m20, transport, sleep=lambda _: None, stop_requested=lambda: True,
        )
        self.assertIs(controller.approach(), ApproachResult.MOTION_FAILED)
        motions = [request["PatrolDevice"]["Items"]["MotionParam"] for request in transport.requests
                   if request["PatrolDevice"]["Type"] == 2]
        self.assertEqual(motions, [4])

    def test_final_position_and_yaw_error_are_diagnostic_only(self):
        responses = self.normal_responses()
        responses[8] = pose(3.0, 0.10, math.pi / 2.0 + math.radians(6.0))
        controller, transport = self.controller(responses)
        controller._last_detections = (detection(1.2, (2.0, 0.0)),)
        self.assertIs(controller.approach(), ApproachResult.PRONE_AT_TARGET)
        motions = [request["PatrolDevice"]["Items"]["MotionParam"] for request in transport.requests
                   if request["PatrolDevice"]["Type"] == 2]
        self.assertEqual(motions, [4])
        self.assertFalse(any(request["PatrolDevice"]["Type"] == 1004 for request in transport.requests))

    def test_forward_segment_preserves_left_side_goal_yaw_after_901_drift(self):
        drifted_yaw = 0.08
        final_yaw = math.pi / 2.0
        forward_x = 1.8 + 0.10 * math.cos(drifted_yaw)
        forward_y = 0.10 * math.sin(drifted_yaw)
        responses = [
            pose(0.0, 0.0, 0.0), perception(), response({"ErrorCode": 0}),
            response({"Value": 901, "Status": 4, "ErrorCode": 0}),
            pose(1.8, 0.0, drifted_yaw), perception(), response({"ErrorCode": 0}),
            response({"Value": 902, "Status": 4, "ErrorCode": 0}),
            pose(forward_x, forward_y, final_yaw), perception(), response({"ErrorCode": 0}),
        ]
        controller, transport = self.controller(responses)
        controller._last_detections = (detection(1.2, (2.0, 0.0)),)
        self.assertIs(controller.approach(), ApproachResult.PRONE_AT_TARGET)
        navs = [request["PatrolDevice"]["Items"] for request in transport.requests
                if request["PatrolDevice"]["Type"] == 1003]
        self.assertAlmostEqual(navs[0]["AngleYaw"], final_yaw, places=6)
        self.assertAlmostEqual(navs[1]["PosX"], forward_x, places=6)
        self.assertAlmostEqual(navs[1]["PosY"], forward_y, places=6)
        self.assertAlmostEqual(navs[1]["AngleYaw"], final_yaw, places=6)

    def test_scan_lifecycle_caches_target_then_closes(self):
        class Vision:
            def __init__(self): self.opened = self.closed = False
            def open(self): self.opened = True
            def detect_once(self): return [detection(1.0, (1.0, 0.0))]
            def close(self): self.closed = True
        vision = Vision()
        cfg = D455Config.from_mapping(config_mapping())
        controller = D455ApproachController(cfg, M20Config().validate(), FakeTransport([]), vision=vision)
        controller.scan_start()
        self.assertEqual(len(controller.scan_handler()), 1)
        controller.scan_stop()
        self.assertTrue(vision.opened and vision.closed)
        self.assertEqual(controller._last_detections, ())

    def test_scan_handler_discards_far_frames_and_keeps_only_near_candidates(self):
        class Vision:
            def __init__(self):
                self.frames = [
                    [detection(1.26, (1.26, 0.0))],
                    [detection(1.24, (1.24, 0.0)), detection(0.8, (0.8, 0.0))],
                ]

            def detect_once(self):
                return self.frames.pop(0)

        controller = D455ApproachController(
            D455Config.from_mapping(config_mapping()), M20Config().validate(), FakeTransport([]), vision=Vision(),
        )
        self.assertEqual(controller.scan_handler(), ())
        self.assertEqual(controller._last_detections, ())
        near_detections = controller.scan_handler()
        self.assertEqual([item.range_m for item in near_detections], [1.24, 0.8])
        self.assertEqual(choose_nearest_detection(near_detections).range_m, 0.8)


if __name__ == "__main__":
    unittest.main()
