"""create_arm_runtime 的无硬件配置检查。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from m20_rebot_moving_grasp.arm.arm_runtime import (
    ArmRuntime,
    create_arm_runtime,
    start_arm_runtime,
    stop_arm_runtime,
)
from m20_rebot_moving_grasp.arm.arm_motion import (
    d435_recovery_views, freeze_d435_scan_center, move_to_d435_recovery_view,
    move_to_ready_pose, move_to_scan_pose, open_gripper, validate_d435_recovery_targets,
)
from m20_rebot_moving_grasp.arm.arm_camera import (
    FrameCaptureError,
    capture_frame,
    start_d435,
)


class _FakeArm:
    def __init__(self) -> None:
        self._connected = False
        self.feedback = [0.0, 0.0]
        self.feedback_controller = None
        self.follow_command_target = True
        self.feedback_requests = 0
        self.stop_loop_calls = 0
        self.disconnect_calls = 0

    def get_state(self, *, request_feedback: bool = False):
        if request_feedback:
            self.feedback_requests += 1
            controller = self.feedback_controller
            if self.follow_command_target and controller is not None:
                self.feedback = np.asarray(controller._q_target, dtype=np.float64).tolist()
        return self.feedback, [], []

    def stop_control_loop(self) -> None:
        self.stop_loop_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False


class _FakeController:
    def __init__(self) -> None:
        self._running = False


class _FakeMotionController(_FakeController):
    def __init__(self, arm: _FakeArm, *, accepted: bool = True) -> None:
        super().__init__()
        self._running = True
        self._n = 2
        self._q_target = [0.0, 0.0]
        self._send_thread = None
        self.arm = arm
        self.accepted = accepted
        self.calls = []

    def _arm_motion_start(self):
        return np.asarray(self.arm.feedback, dtype=np.float64)

    def move_to_traj(self, *pose: float, duration: float) -> bool:
        self.calls.append((pose, duration))
        if not self.accepted:
            return False
        self._q_target = [pose[0], pose[1]]
        self.arm.feedback = [pose[0], pose[1]]
        return True


class _FakeRecoveryMotionController(_FakeMotionController):
    def __init__(self, arm: _FakeArm) -> None:
        super().__init__(arm)
        self._n = 6
        self._q_target = np.zeros(6)
        self._model = type("Model", (), {
            "lowerPositionLimit": np.full(6, -1.57),
            "upperPositionLimit": np.full(6, 1.57),
        })()


class _FakeGripperDriver:
    def __init__(self, arm: _FakeArm, controller: _FakeController) -> None:
        self.arm = arm
        self.controller = controller
        self.start_calls = 0
        self.MAX_DISTANCE_M = 0.09
        self._angle_open = -5.2
        self._open_lo = -5.096
        self._open_hi = 0.0
        self.position = 0.0
        self.open_calls = []

    def start(self) -> None:
        self.start_calls += 1
        self.arm._connected = True
        self.controller._running = True

    def open_gripper(self, *, distance_m: float, timeout: float) -> None:
        self.open_calls.append((distance_m, timeout))
        raw_target = (distance_m / self.MAX_DISTANCE_M) * self._angle_open
        self.position = min(max(raw_target, self._open_lo), self._open_hi)

    def get_gripper_state(self):
        return self.position, 0.0, 0.0


class _BrokenGripperDriver:
    def start(self) -> None:
        return


class _FakeIntrinsics:
    shape = (3, 3)


class _FakeCamera:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.open_calls = 0
        self.warmup_calls = []
        self.close_calls = 0
        self.K = _FakeIntrinsics()
        self.color_frame = np.zeros((kwargs["height"], kwargs["width"], 3), dtype=np.uint8)
        self.depth_frame = np.ones((kwargs["height"], kwargs["width"]), dtype=np.uint16)

    def open(self) -> None:
        self.open_calls += 1

    def warm_up(self, frames: int) -> None:
        self.warmup_calls.append(frames)

    def close(self) -> None:
        self.close_calls += 1

    def get_frame(self):
        return self.color_frame, self.depth_frame


class _FallbackProfileCamera(_FakeCamera):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._w = 848
        self._h = 480
        self._fps = 30


def _fake_runtime(driver: object | None = None) -> ArmRuntime:
    arm = _FakeArm()
    controller = _FakeController()
    actual_driver = driver or _FakeGripperDriver(arm, controller)
    return ArmRuntime(
        rebotarm=arm,
        controller=controller,
        gripper_driver=actual_driver,
        sdk_root=PROJECT_ROOT,
        grasp_root=PROJECT_ROOT,
        arm_control_mode="posvel",
    )


def _motion_runtime(*, accepted: bool = True) -> tuple[ArmRuntime, _FakeMotionController]:
    arm = _FakeArm()
    controller = _FakeMotionController(arm, accepted=accepted)
    runtime = ArmRuntime(
        rebotarm=arm,
        controller=controller,
        gripper_driver=object(),
        sdk_root=PROJECT_ROOT,
        grasp_root=PROJECT_ROOT,
        arm_control_mode="posvel",
    )
    return runtime, controller


def _recovery_motion_runtime() -> tuple[ArmRuntime, _FakeRecoveryMotionController]:
    arm = _FakeArm()
    arm.feedback = [0.0] * 6
    controller = _FakeRecoveryMotionController(arm)
    arm.feedback_controller = controller
    runtime = ArmRuntime(
        rebotarm=arm,
        controller=controller,
        gripper_driver=object(),
        sdk_root=PROJECT_ROOT,
        grasp_root=PROJECT_ROOT,
        arm_control_mode="posvel",
    )
    return runtime, controller


def _gripper_runtime() -> tuple[ArmRuntime, _FakeGripperDriver]:
    arm = _FakeArm()
    controller = _FakeController()
    controller._running = True
    driver = _FakeGripperDriver(arm, controller)
    runtime = ArmRuntime(
        rebotarm=arm,
        controller=controller,
        gripper_driver=driver,
        sdk_root=PROJECT_ROOT,
        grasp_root=PROJECT_ROOT,
        arm_control_mode="posvel",
    )
    return runtime, driver


READY_POSE = {
    "ready_pose": {
        "x": 0.3,
        "y": 0.0,
        "z": 0.25,
        "roll": 0.0,
        "pitch": 0.75,
        "yaw": 0.0,
        "duration_sec": 0.01,
    }
}

SCAN_POSE = {
    "scan_pose0": {
        "joint1_offset_rad": -1.5708,
        "duration_sec": 0.01,
        "joint1_speed_limit_rad_s": 1000.0,
    }
}

RECOVERY_SCAN = {
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


VISION_CONFIG = {
    "camera": {
        "type": "realsense_d435cb",
        "serial": "335522070264",
        "color_width": 640,
        "color_height": 480,
        "depth_width": 640,
        "depth_height": 480,
        "fps": 30,
    },
    "calibration": {
        "hand_eye_path": "config/calibration/realsense_d435cb/hand_eye.npz",
    },
}


class CreateArmRuntimeTest(unittest.TestCase):
    def test_missing_sdk_path_is_reported_before_sdk_import(self) -> None:
        config = {
            "robot": {
                "repo_root": "/tmp/does-not-exist-m20-sdk",
                "grasp_repo_root": str(PROJECT_ROOT),
            }
        }
        with self.assertRaisesRegex(FileNotFoundError, "DM SDK 目录不存在"):
            create_arm_runtime(config)

    def test_missing_grasp_path_is_reported(self) -> None:
        config = {
            "robot": {
                "repo_root": str(PROJECT_ROOT),
                "grasp_repo_root": "/tmp/does-not-exist-m20-grasp",
            }
        }
        with self.assertRaisesRegex(FileNotFoundError, "已验证抓取代码目录不存在"):
            create_arm_runtime(config)

    def test_start_starts_one_control_loop(self) -> None:
        runtime = _fake_runtime()
        start_arm_runtime(runtime)

        self.assertTrue(runtime.rebotarm._connected)
        self.assertTrue(runtime.controller._running)
        self.assertEqual(runtime.gripper_driver.start_calls, 1)

    def test_start_is_idempotent(self) -> None:
        runtime = _fake_runtime()
        start_arm_runtime(runtime)
        start_arm_runtime(runtime)

        self.assertEqual(runtime.gripper_driver.start_calls, 1)

    def test_start_rejects_driver_without_connection(self) -> None:
        runtime = _fake_runtime(driver=_BrokenGripperDriver())
        with self.assertRaisesRegex(RuntimeError, "SDK 未建立连接"):
            start_arm_runtime(runtime)

    def test_ready_pose_sends_configured_cartesian_target(self) -> None:
        runtime, controller = _motion_runtime()
        self.assertTrue(move_to_ready_pose(runtime, READY_POSE, wait_extra_sec=0.0))
        self.assertEqual(controller.calls, [((0.3, 0.0, 0.25, 0.0, 0.75, 0.0), 0.01)])

    def test_ready_pose_returns_false_when_trajectory_is_rejected(self) -> None:
        runtime, _ = _motion_runtime(accepted=False)
        self.assertFalse(move_to_ready_pose(runtime, READY_POSE, wait_extra_sec=0.0))

    def test_ready_pose_accepts_completed_trajectory_with_stale_feedback(self) -> None:
        runtime, controller = _motion_runtime()

        def move_without_arrival(*pose: float, duration: float) -> bool:
            controller.calls.append((pose, duration))
            controller._q_target = [pose[0], pose[1]]
            runtime.rebotarm.feedback = [0.0, 0.0]
            return True

        controller.move_to_traj = move_without_arrival
        self.assertTrue(move_to_ready_pose(runtime, READY_POSE, wait_extra_sec=0.0))

    def test_ready_pose_returns_false_when_trajectory_sender_times_out(self) -> None:
        runtime, _ = _motion_runtime()
        with patch("m20_rebot_moving_grasp.arm.arm_motion._wait_for_motion", return_value=False):
            self.assertFalse(move_to_ready_pose(runtime, READY_POSE, wait_extra_sec=0.0))

    def test_scan_pose_rotates_j1_by_configured_offset_after_ready_pose(self) -> None:
        runtime, controller = _motion_runtime()
        runtime.rebotarm.feedback = [0.3, -0.1]
        with patch("m20_rebot_moving_grasp.arm.arm_motion.time.sleep"):
            self.assertTrue(move_to_scan_pose(runtime, SCAN_POSE))
        self.assertAlmostEqual(controller._q_target[0], 0.3 - 1.5708)
        self.assertAlmostEqual(controller._q_target[1], -0.1)

    def test_d435_recovery_views_are_absolute_offsets_from_one_center(self) -> None:
        runtime, controller = _recovery_motion_runtime()
        controller._q_target[:] = [0.4, -0.2, 0.1, -0.3, 0.25, 0.05]
        center = freeze_d435_scan_center(runtime)
        views = d435_recovery_views(RECOVERY_SCAN)
        validate_d435_recovery_targets(runtime, center, views, RECOVERY_SCAN)
        targets = []
        with patch("m20_rebot_moving_grasp.arm.arm_motion.time.sleep"):
            for view in views:
                self.assertTrue(move_to_d435_recovery_view(runtime, center, view, RECOVERY_SCAN))
                targets.append(controller._q_target.copy())
        expected_offsets = [(0.0, -0.174533), (0.0, 0.174533), (-0.174533, -0.174533),
                            (-0.174533, 0.174533), (0.174533, -0.174533), (0.174533, 0.174533)]
        for target, (j1_offset, j5_offset) in zip(targets, expected_offsets):
            self.assertAlmostEqual(target[0], center[0] + j1_offset)
            self.assertAlmostEqual(target[4], center[4] + j5_offset)
            np.testing.assert_allclose(target[[1, 2, 3, 5]], center[[1, 2, 3, 5]])

    def test_d435_recovery_rejects_any_target_outside_joint_limit_margin(self) -> None:
        runtime, controller = _recovery_motion_runtime()
        controller._q_target[4] = 1.5
        center = freeze_d435_scan_center(runtime)
        with self.assertRaisesRegex(ValueError, "J5.*安全限位"):
            validate_d435_recovery_targets(runtime, center, d435_recovery_views(RECOVERY_SCAN), RECOVERY_SCAN)

    def test_d435_recovery_retries_once_then_rejects_missing_joint_arrival(self) -> None:
        runtime, controller = _recovery_motion_runtime()
        runtime.rebotarm.follow_command_target = False
        controller._q_target[:] = [0.4, -0.2, 0.1, -0.3, 0.25, 0.05]
        center = freeze_d435_scan_center(runtime)
        view = d435_recovery_views(RECOVERY_SCAN)[0]
        with patch("m20_rebot_moving_grasp.arm.arm_motion.time.sleep"):
            self.assertFalse(move_to_d435_recovery_view(runtime, center, view, RECOVERY_SCAN))
        self.assertGreaterEqual(runtime.rebotarm.feedback_requests, 2)
        self.assertAlmostEqual(controller._q_target[0], center[0])
        self.assertAlmostEqual(controller._q_target[4], center[4] - 0.174533)

    def test_open_gripper_uses_verified_driver_and_confirms_position(self) -> None:
        runtime, driver = _gripper_runtime()

        self.assertTrue(open_gripper(runtime))
        self.assertEqual(driver.open_calls, [(0.09, 3.0)])
        self.assertAlmostEqual(driver.position, -5.096)

    def test_open_gripper_returns_false_when_feedback_is_not_at_target(self) -> None:
        runtime, driver = _gripper_runtime()

        def open_without_arrival(*, distance_m: float, timeout: float) -> None:
            driver.open_calls.append((distance_m, timeout))
            driver.position = 0.0

        driver.open_gripper = open_without_arrival
        self.assertFalse(open_gripper(runtime))

    def test_stop_disconnects_without_calling_safe_home(self) -> None:
        runtime = _fake_runtime()
        runtime.rebotarm._connected = True
        runtime.controller._running = True

        stop_arm_runtime(runtime)

        self.assertEqual(runtime.rebotarm.stop_loop_calls, 1)
        self.assertEqual(runtime.rebotarm.disconnect_calls, 1)
        self.assertFalse(runtime.rebotarm._connected)
        self.assertFalse(runtime.controller._running)

    def test_start_d435_opens_and_warms_up_configured_camera(self) -> None:
        runtime = _fake_runtime()
        holder = {}

        def factory(**kwargs):
            holder["camera"] = _FakeCamera(**kwargs)
            return holder["camera"]

        d435 = start_d435(runtime, VISION_CONFIG, warmup_frames=3, camera_factory=factory)
        camera = holder["camera"]

        self.assertEqual(camera.open_calls, 1)
        self.assertEqual(camera.warmup_calls, [3])
        self.assertEqual(camera.kwargs["serial"], "335522070264")
        self.assertEqual((d435.width, d435.height, d435.fps), (640, 480, 30))

    def test_start_d435_rejects_unaligned_config(self) -> None:
        runtime = _fake_runtime()
        invalid_config = {
            **VISION_CONFIG,
            "camera": {**VISION_CONFIG["camera"], "depth_width": 848},
        }
        with self.assertRaisesRegex(ValueError, "彩色与深度分辨率必须相同"):
            start_d435(runtime, invalid_config, camera_factory=_FakeCamera)

    def test_start_d435_rejects_fallback_profile(self) -> None:
        runtime = _fake_runtime()
        holder = {}

        def factory(**kwargs):
            holder["camera"] = _FallbackProfileCamera(**kwargs)
            return holder["camera"]

        with self.assertRaisesRegex(RuntimeError, "D435 启动、预热或内参检查失败"):
            start_d435(runtime, VISION_CONFIG, camera_factory=factory)
        self.assertEqual(holder["camera"].close_calls, 1)

    def test_capture_frame_validates_aligned_bgr_and_depth(self) -> None:
        runtime = _fake_runtime()
        d435 = start_d435(runtime, VISION_CONFIG, warmup_frames=0, camera_factory=_FakeCamera)
        frame = capture_frame(d435)

        self.assertEqual(frame.color_bgr.shape, (480, 640, 3))
        self.assertEqual(frame.depth_mm.shape, (480, 640))
        self.assertEqual(frame.color_bgr.dtype, np.uint8)
        self.assertEqual(frame.depth_mm.dtype, np.uint16)

    def test_capture_frame_rejects_empty_depth(self) -> None:
        runtime = _fake_runtime()
        d435 = start_d435(runtime, VISION_CONFIG, warmup_frames=0, camera_factory=_FakeCamera)
        d435.camera.depth_frame.fill(0)

        with self.assertRaisesRegex(FrameCaptureError, "深度帧全部为零"):
            capture_frame(d435)

    def test_capture_frame_rejects_wrong_shape(self) -> None:
        runtime = _fake_runtime()
        d435 = start_d435(runtime, VISION_CONFIG, warmup_frames=0, camera_factory=_FakeCamera)
        d435.camera.color_frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with self.assertRaisesRegex(FrameCaptureError, "彩色帧尺寸错误"):
            capture_frame(d435)
