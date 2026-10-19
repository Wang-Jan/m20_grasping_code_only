import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m20_rebot_moving_grasp.arm.arm_actions import (
    GraspAttemptResult,
    drop_object,
    emergency_stop,
    execute_grasp_attempt,
    grasp_with_retry,
    move_to_hold_pose,
)
from m20_rebot_moving_grasp.arm.arm_runtime import ArmRuntime
from m20_rebot_moving_grasp.arm.vision_2d import BaseGraspPoses


class Controller:
    _running = True
    _n = 2
    _q_target = np.zeros(2)
    _stop_send = type("Stop", (), {"set": lambda self: None, "clear": lambda self: None})()
    def _arm_motion_start(self): return np.zeros(2)
    def move_to_traj(self, *pose, duration, **kwargs):
        self.last_move = (pose, duration, kwargs)
        return True
    def safe_home(self): self.safe_home_called = True


class Driver:
    MAX_DISTANCE_M = 0.09
    _angle_open = 5.2
    _open_lo = -5.096
    _open_hi = 0.0
    def __init__(self, results): self.results = iter(results); self.calls = []
    def open_gripper(self, **kwargs): self.calls.append("open")
    def get_gripper_state(self): return (-5.096, 0.0, 0.0)
    def grasp(self, **kwargs): self.calls.append("grasp"); return next(self.results)
    def release_gripper(self): self.calls.append("release")


def runtime(driver):
    arm = type("Arm", (), {"disable_all": lambda self: setattr(self, "disabled", True)})()
    return ArmRuntime(arm, Controller(), driver, None, None, "posvel")


def poses():
    p = (0.1, 0.0, 0.3, 0.0, 0.7, 0.0)
    return BaseGraspPoses(p, p, p, np.eye(4))


CFG = {"arm_motion": {"pregrasp_duration_sec": 0.001, "grasp_duration_sec": 0.001, "retreat_duration_sec": 0.001},
       "transport_pose": {"joint_positions_rad": [0.1, -0.1], "duration_sec": 0.001},
       "pre_drop_pose": {"x": 0.1, "y": 0.0, "z": 0.3, "roll": 0.0, "pitch": 0.7, "yaw": 0.0, "duration": 0.001,
                         "ik_seed_joint_positions_rad": [0.1, -0.1]}}


class ArmActionsTest(unittest.TestCase):
    @patch("m20_rebot_moving_grasp.arm.arm_actions.move_to_cartesian_pose", return_value=True)
    @patch("m20_rebot_moving_grasp.arm.arm_actions.open_gripper", return_value=True)
    def test_attempt_uses_driver_grasp_and_retreats(self, open_fn, move_fn):
        d = Driver([True])
        self.assertTrue(execute_grasp_attempt(runtime(d), poses(), CFG))
        self.assertEqual(d.calls, ["grasp"])
        self.assertEqual(move_fn.call_count, 3)

    @patch("m20_rebot_moving_grasp.arm.arm_actions.execute_grasp_attempt", side_effect=[False, True])
    def test_retry_reacquires_after_failed_hard_close(self, execute):
        d = Driver([])
        reacquire = lambda: poses()
        returned = []
        result = grasp_with_retry(runtime(d), poses(), CFG, max_attempts=2,
                                  reacquire_poses=reacquire,
                                  return_to_detection=lambda: returned.append(True) or True)
        self.assertEqual(result, GraspAttemptResult(True, 2))
        self.assertEqual(len(returned), 1)

    def test_hold_pose_writes_joint_target(self):
        rt = runtime(Driver([]))
        self.assertTrue(move_to_hold_pose(rt, CFG))
        np.testing.assert_allclose(rt.controller._q_target, [0.1, -0.1])

    @patch("m20_rebot_moving_grasp.arm.arm_actions.move_to_pre_drop_pose", return_value=True)
    def test_drop_releases_only_after_reaching_pre_drop(self, move):
        d = Driver([])
        self.assertTrue(drop_object(runtime(d), CFG))
        self.assertEqual(d.calls, ["release"])

    def test_pre_drop_uses_verified_duration_and_ik_seed(self):
        from m20_rebot_moving_grasp.arm.arm_actions import move_to_pre_drop_pose

        rt = runtime(Driver([]))
        self.assertTrue(move_to_pre_drop_pose(rt, CFG))
        pose, duration, kwargs = rt.controller.last_move
        self.assertEqual(pose, (0.1, 0.0, 0.3, 0.0, 0.7, 0.0))
        self.assertEqual(duration, 0.001)
        np.testing.assert_allclose(kwargs["ik_seed"], [0.1, -0.1])

    @patch("m20_rebot_moving_grasp.arm.arm_actions.open_gripper", return_value=True)
    def test_emergency_stop_order_and_disable(self, open_fn):
        d = Driver([False])
        rt = runtime(d)
        self.assertTrue(emergency_stop(rt))  # grasp 返回 False 仍不打断急停顺序
        self.assertEqual(d.calls, ["grasp"])
        self.assertTrue(getattr(rt.controller, "safe_home_called", False))
        self.assertTrue(getattr(rt.rebotarm, "disabled", False))


if __name__ == "__main__":
    unittest.main()
