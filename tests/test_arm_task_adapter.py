import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m20_rebot_moving_grasp.arm.arm_runtime import ArmRuntime
from m20_rebot_moving_grasp.arm.arm_task_adapter import ArmTaskAdapter
from m20_rebot_moving_grasp.arm.vision_2d import BaseGraspPoses, DetectionBatch, IKCheckResult


class _Controller:
    _running = True
    _n = 6
    _model = type("Model", (), {
        "lowerPositionLimit": np.full(6, -3.0),
        "upperPositionLimit": np.full(6, 3.0),
    })()
    def __init__(self):
        self._q_target = np.zeros(6)
    def safe_home(self): self.home = True


class _Driver:
    def get_tcp_pose(self): return np.eye(4)


class _Camera:
    def close(self): self.closed = True


RECOVERY_POSES = {
    "d435_recovery_scan": {
        "views": [
            {"joint1_offset_rad": 0.0, "joint5_offset_rad": -0.174533},
            {"joint1_offset_rad": 0.0, "joint5_offset_rad": 0.174533},
            {"joint1_offset_rad": -0.174533, "joint5_offset_rad": -0.174533},
            {"joint1_offset_rad": -0.174533, "joint5_offset_rad": 0.174533},
            {"joint1_offset_rad": 0.174533, "joint5_offset_rad": -0.174533},
            {"joint1_offset_rad": 0.174533, "joint5_offset_rad": 0.174533},
        ],
        "duration_sec": 0.01,
        "joint_speed_limit_rad_s": 1000.0,
        "settle_sec": 0.0,
        "joint_limit_margin_rad": 0.05,
        "joint_feedback_tolerance_rad": 0.10,
        "feedback_timeout_sec": 0.01,
        "feedback_retry_count": 1,
    }
}


class ArmTaskAdapterTest(unittest.TestCase):
    def runtime(self):
        return ArmRuntime(object(), _Controller(), _Driver(), Path("."), Path("."), "posvel")

    def test_init_starts_runtime_and_keeps_arm_at_safe_home(self):
        rt = self.runtime()
        adapter = ArmTaskAdapter({}, {}, RECOVERY_POSES, vision_runtime_factory=lambda *args: object())
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.create_arm_runtime", return_value=rt), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_arm_runtime"), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_d435") as start_d435, \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.safe_home", return_value=True) as home, \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_ready_pose") as ready:
            self.assertTrue(adapter.init_reset())
            self.assertIs(adapter.runtime, rt)
            self.assertIsNone(adapter.d435)
            start_d435.assert_not_called()
            home.assert_called_once_with(rt)
            ready.assert_not_called()
            self.assertTrue(adapter.arm_at_safe_home)

    def test_init_fails_and_cleans_up_when_safe_home_fails(self):
        rt = self.runtime()
        adapter = ArmTaskAdapter({}, {}, RECOVERY_POSES, vision_runtime_factory=lambda *args: object())
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.create_arm_runtime", return_value=rt), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_arm_runtime"), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_d435") as start_d435, \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.safe_home", return_value=False), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.stop_d435") as stop_d435, \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.stop_arm_runtime") as stop_arm:
            self.assertFalse(adapter.init_reset())
            start_d435.assert_not_called()
            stop_d435.assert_not_called()
            stop_arm.assert_called_once_with(rt)
            self.assertFalse(adapter.arm_at_safe_home)

    def test_prepare_grasp_starts_d435_lazily_after_approach(self):
        rt = self.runtime()
        d435 = type("D435", (), {"camera": _Camera()})()
        adapter = ArmTaskAdapter({}, {}, RECOVERY_POSES, vision_runtime_factory=lambda *args: object())
        adapter.runtime = rt
        events = []
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_ready_pose", side_effect=lambda *_: events.append("ready") or True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_scan_pose", side_effect=lambda *_: events.append("scan") or True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_d435", side_effect=lambda *_args, **_kwargs: events.append("d435") or d435) as start_d435, \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.capture_frame", side_effect=lambda *_: events.append("capture") or object()), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.detect_2d_targets", return_value=[]), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.estimate_2d_grasps", return_value=[]), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_d435_recovery_view", return_value=True) as move_view, \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.time.sleep"):
            self.assertFalse(adapter.prepare_grasp())
        start_d435.assert_called_once_with(rt, {}, camera_factory=None)
        self.assertIs(adapter.d435, d435)
        self.assertEqual(events, ["ready", "scan", "d435", "capture"] + ["capture"] * 6)
        self.assertEqual(move_view.call_count, 7)
        self.assertEqual((move_view.call_args.args[2].joint1_offset_rad, move_view.call_args.args[2].joint5_offset_rad), (0.0, 0.0))
        self.assertTrue(adapter.is_recoverable_detection_failure)
        self.assertIn("六视角补偿均未获得", adapter.last_failure_reason)

    def test_prepare_grasp_does_not_start_d435_when_scan_pose_fails(self):
        rt = self.runtime()
        adapter = ArmTaskAdapter({}, {}, {})
        adapter.runtime = rt
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_ready_pose", return_value=True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_scan_pose", return_value=False), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_d435") as start_d435:
            self.assertFalse(adapter.prepare_grasp())
        start_d435.assert_not_called()

    def test_prepare_grasp_does_not_start_d435_when_ready_pose_fails(self):
        rt = self.runtime()
        adapter = ArmTaskAdapter({}, {}, {})
        adapter.runtime = rt
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_ready_pose", return_value=False), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_scan_pose") as scan_pose, \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_d435") as start_d435:
            self.assertFalse(adapter.prepare_grasp())
        scan_pose.assert_not_called()
        start_d435.assert_not_called()

    def test_prepare_grasp_reports_coordinate_transform_failure(self):
        rt = self.runtime()
        d435 = type("D435", (), {"camera": _Camera()})()
        adapter = ArmTaskAdapter({}, {}, RECOVERY_POSES, vision_runtime_factory=lambda *args: object())
        adapter.runtime = rt
        candidate = object()
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_ready_pose", return_value=True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_scan_pose", return_value=True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_d435", return_value=d435), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.capture_frame", return_value=object()), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.detect_2d_targets",
                   return_value=DetectionBatch([], [object()])), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.estimate_2d_grasps", return_value=[candidate]), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.select_best_grasp", return_value=candidate), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.transform_grasp_to_base",
                   side_effect=ValueError("手眼矩阵无效")), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_d435_recovery_view") as move_view:
            self.assertFalse(adapter.prepare_grasp())
        self.assertEqual(adapter.last_failure_reason, "抓取位姿坐标变换失败: 手眼矩阵无效")
        move_view.assert_not_called()

    def test_prepare_grasp_ik_unreachable_runs_all_recovery_views(self):
        rt = self.runtime()
        d435 = type("D435", (), {"camera": _Camera()})()
        adapter = ArmTaskAdapter(
            {}, {}, RECOVERY_POSES,
            vision_runtime_factory=lambda *args: object(),
            ik_checker_factory=lambda _runtime: object(),
        )
        adapter.runtime = rt
        candidate = object()
        poses = BaseGraspPoses((0.0,) * 6, (0.0,) * 6, (0.0,) * 6, np.eye(4))
        ik_result = IKCheckResult(False, 0.01, 0.02, 0.03)
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_ready_pose", return_value=True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_scan_pose", return_value=True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_d435", return_value=d435), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.capture_frame", return_value=object()), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.detect_2d_targets",
                   return_value=DetectionBatch([], [object()])), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.estimate_2d_grasps", return_value=[candidate]), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.select_best_grasp", return_value=candidate), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.transform_grasp_to_base", return_value=poses), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.check_grasp_poses_ik", return_value=ik_result), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_d435_recovery_view", return_value=True) as move_view, \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.time.sleep"):
            self.assertFalse(adapter.prepare_grasp())
        self.assertEqual(move_view.call_count, 7)
        self.assertTrue(adapter.is_recoverable_detection_failure)
        self.assertIn("均未获得可行抓取位姿", adapter.last_failure_reason)
        self.assertIn("抓取位姿 IK 不可达", adapter.last_failure_reason)

    def test_prepare_grasp_ik_unreachable_then_recovery_view_succeeds(self):
        rt = self.runtime()
        d435 = type("D435", (), {"camera": _Camera()})()
        candidate = object()
        poses = BaseGraspPoses((0.0,) * 6, (0.0,) * 6, (0.0,) * 6, np.eye(4))
        adapter = ArmTaskAdapter(
            {}, {}, RECOVERY_POSES,
            vision_runtime_factory=lambda *args: object(),
            ik_checker_factory=lambda _runtime: object(),
        )
        adapter.runtime = rt
        target_batch = DetectionBatch([], [object()])
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_ready_pose", return_value=True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_scan_pose", return_value=True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_d435", return_value=d435), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.capture_frame", return_value=object()), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.detect_2d_targets", return_value=target_batch), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.estimate_2d_grasps", return_value=[candidate]), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.select_best_grasp", return_value=candidate), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.transform_grasp_to_base", return_value=poses), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.check_grasp_poses_ik", side_effect=[
                 IKCheckResult(False, 0.01, 0.02, 0.03), IKCheckResult(True, 0.0, 0.0, 0.0),
             ]), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_d435_recovery_view", return_value=True) as move_view, \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.time.sleep"):
            self.assertTrue(adapter.prepare_grasp())
        self.assertEqual(move_view.call_count, 1)
        self.assertIs(adapter.current_poses, poses)

    def test_prepare_grasp_center_success_does_not_run_recovery_views(self):
        rt = self.runtime()
        d435 = type("D435", (), {"camera": _Camera()})()
        candidate = object()
        poses = BaseGraspPoses((0.0,) * 6, (0.0,) * 6, (0.0,) * 6, np.eye(4))
        adapter = ArmTaskAdapter(
            {}, {}, RECOVERY_POSES,
            vision_runtime_factory=lambda *args: object(),
            ik_checker_factory=lambda _runtime: object(),
        )
        adapter.runtime = rt
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_ready_pose", return_value=True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_scan_pose", return_value=True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_d435", return_value=d435), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.capture_frame", return_value=object()), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.detect_2d_targets",
                   return_value=DetectionBatch([], [object()])), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.estimate_2d_grasps", return_value=[candidate]), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.select_best_grasp", return_value=candidate), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.transform_grasp_to_base", return_value=poses), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.check_grasp_poses_ik",
                   return_value=IKCheckResult(True, 0.0, 0.0, 0.0)), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_d435_recovery_view") as move_view:
            self.assertTrue(adapter.prepare_grasp())
        move_view.assert_not_called()
        self.assertIs(adapter.current_poses, poses)

    def test_prepare_grasp_uses_sixth_absolute_view_and_keeps_it_on_success(self):
        rt = self.runtime()
        d435 = type("D435", (), {"camera": _Camera()})()
        candidate = object()
        poses = BaseGraspPoses((0.0,) * 6, (0.0,) * 6, (0.0,) * 6, np.eye(4))
        adapter = ArmTaskAdapter(
            {}, {}, RECOVERY_POSES,
            vision_runtime_factory=lambda *args: object(),
            ik_checker_factory=lambda _runtime: object(),
        )
        adapter.runtime = rt
        empty_batch = DetectionBatch([], [])
        target_batch = DetectionBatch([], [object()])
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_ready_pose", return_value=True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_scan_pose", return_value=True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.start_d435", return_value=d435), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.capture_frame", return_value=object()), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.detect_2d_targets",
                   side_effect=[empty_batch] * 6 + [target_batch]), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.estimate_2d_grasps",
                   side_effect=[[]] * 6 + [[candidate]]), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.select_best_grasp", return_value=candidate), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.transform_grasp_to_base", return_value=poses), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.check_grasp_poses_ik",
                   return_value=IKCheckResult(True, 0.0, 0.0, 0.0)), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_d435_recovery_view", return_value=True) as move_view, \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.time.sleep"):
            self.assertTrue(adapter.prepare_grasp())
        self.assertEqual(move_view.call_count, 6)
        final_view = move_view.call_args.args[2]
        self.assertAlmostEqual(final_view.joint1_offset_rad, 0.174533)
        self.assertAlmostEqual(final_view.joint5_offset_rad, 0.174533)
        self.assertIs(adapter.current_poses, poses)

    def test_return_to_detection_repeats_ready_and_scan_pose(self):
        rt = self.runtime()
        adapter = ArmTaskAdapter({}, {}, RECOVERY_POSES)
        adapter.runtime = rt
        events = []
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_ready_pose", side_effect=lambda *_: events.append("ready") or True), \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.move_to_scan_pose", side_effect=lambda *_: events.append("scan") or True):
            self.assertTrue(adapter._return_to_detection())
        self.assertEqual(events, ["ready", "scan"])

    def test_release_vision_closes_d435_and_clears_runtime_vision(self):
        adapter = ArmTaskAdapter({}, {}, {})
        d435 = type("D435", (), {"camera": _Camera()})()
        adapter.d435 = d435
        adapter.vision = object()
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.stop_d435") as stop_d435:
            adapter.release_vision()
        stop_d435.assert_called_once_with(d435)
        self.assertIsNone(adapter.d435)
        self.assertIsNone(adapter.vision)

    def test_grasp_uses_current_pose_and_configured_retry_count(self):
        adapter = ArmTaskAdapter({}, {}, {}, max_attempts=2)
        adapter.runtime = self.runtime()
        pose = (0.1, 0.0, 0.3, 0.0, 0.7, 0.0)
        adapter.current_poses = BaseGraspPoses(pose, pose, pose, np.eye(4))
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.grasp_with_retry") as grasp:
            grasp.return_value.success = True
            self.assertTrue(adapter.grasp_current_target())
            self.assertEqual(grasp.call_args.kwargs["max_attempts"], 2)

    def test_shutdown_closes_camera_and_arm(self):
        adapter = ArmTaskAdapter({}, {}, {})
        adapter.runtime = self.runtime()
        adapter.d435 = type("D435", (), {"camera": _Camera()})()
        with patch("m20_rebot_moving_grasp.arm.arm_task_adapter.stop_d435") as stop_camera, \
             patch("m20_rebot_moving_grasp.arm.arm_task_adapter.stop_arm_runtime") as stop_arm:
            adapter.shutdown()
            stop_camera.assert_called_once()
            stop_arm.assert_called_once()
            self.assertIsNone(adapter.runtime)


if __name__ == "__main__":
    unittest.main()
