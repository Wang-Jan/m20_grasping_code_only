import unittest
from pathlib import Path

import numpy as np

from m20_rebot_moving_grasp.arm.arm_camera import D435Runtime, RGBDFrame
from m20_rebot_moving_grasp.arm.vision_2d import (
    BaseGraspPoses,
    DetectionBatch,
    Vision2DRuntime,
    check_grasp_poses_ik,
    create_2d_vision_runtime,
    detect_2d_targets,
    estimate_2d_grasps,
    select_best_grasp,
    transform_grasp_to_base,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASP_ROOT = next(
    candidate
    for candidate in (
        PROJECT_ROOT / "third_party/reBot-DevArm-Grasp",
        PROJECT_ROOT.parent / "third_party/reBot-DevArm-Grasp",
    )
    if candidate.is_dir()
)


class _Camera:
    K = np.eye(3)


class _Grasp:
    def __init__(self, name, conf, valid=True):
        self.class_name = name
        self.conf = conf
        self.is_valid = valid
        self.position = np.array([0.1, 0.2, 0.4])
        self.tcp_rotation = np.eye(3)


def _vision(**kwargs):
    cfg = {
        "camera": {"type": "realsense_d435cb"},
        "calibration": {"hand_eye_path": "unused.npz", "hand_eye_mode": "eye_in_hand"},
        "detection": {
            "model_name": "fake.pt", "device": "cpu", "confidence_threshold": 0.05,
            "iou_threshold": 0.45, "infer_every_live": 3, "custom_classes": [],
        },
        "grasp_2d": {
            "depth_quantile": 0.2, "pregrasp_offset_m": 0.08,
            "retreat_offset_m": 0.08, "insertion_depth_m": 0.063, "min_base_z_m": -0.2,
        },
    }
    return Vision2DRuntime(
        model=object(), yolo_options={"device": "cpu"}, config={
            "camera": cfg["camera"], "calibration": cfg["calibration"],
            "detection": {"conf_threshold": 0.05, "iou_threshold": 0.45, "infer_every_live": 3},
            "grasp_pipeline": {"infer_every_live": 3, "grasp": {
                "depth_quantile": 0.2, "pregrasp_offset_m": 0.08,
                "retreat_offset_m": 0.08, "insertion_depth_m": 0.063, "min_base_z_m": -0.2,
            }}, "yolo": {"model_name": "fake.pt", "device": "cpu", "custom_classes": []},
        }, camera_intrinsics=np.eye(3), hand_eye=np.eye(4),
        grasp_root=GRASP_ROOT, **kwargs,
    )


class Vision2DTest(unittest.TestCase):
    def setUp(self):
        self.frame = RGBDFrame(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.uint16))

    def test_create_runtime_uses_injected_model_and_hand_eye(self):
        d435 = D435Runtime(_Camera(), "x", 2, 2, 30, Path("."))
        runtime = create_2d_vision_runtime(
            d435, {
                "camera": {"type": "realsense_d435cb"},
                "calibration": {"hand_eye_path": "missing.npz", "hand_eye_mode": "eye_in_hand"},
                "detection": {"model_name": "fake.pt", "device": "cpu", "confidence_threshold": 0.05,
                              "iou_threshold": 0.45, "infer_every_live": 3},
                "grasp_2d": {"depth_quantile": 0.2, "pregrasp_offset_m": 0.08,
                             "retreat_offset_m": 0.08, "insertion_depth_m": 0.063, "min_base_z_m": -0.2},
            }, GRASP_ROOT,
            model=object(), hand_eye=np.eye(4), load_model=False,
        )
        self.assertEqual(runtime.camera_intrinsics.shape, (3, 3))
        self.assertEqual(
            runtime.config["grasp_pipeline"]["grasp"],
            {
                "depth_quantile": 0.2,
                "pregrasp_offset_m": 0.08,
                "retreat_offset_m": 0.08,
                "insertion_depth_m": 0.063,
                "min_base_z_m": -0.2,
            },
        )
        self.assertEqual(runtime.config["grasp_pipeline"]["grasp"]["depth_quantile"], 0.2)

    def test_detect_and_estimate_use_injected_functions(self):
        calls = {}
        def detect(model, image, options):
            calls["detect"] = options
            return ["raw"], ["det"]
        def estimate(raw, depth, K, depth_quantile):
            calls["quantile"] = depth_quantile
            return [_Grasp("bottle", 0.8)]
        vision = _vision(detect_objects_fn=detect, estimate_grasps_fn=estimate)
        batch = detect_2d_targets(vision, self.frame)
        grasps = estimate_2d_grasps(vision, self.frame, batch)
        self.assertEqual(batch.raw_results, ["raw"])
        self.assertEqual(batch.detections, ["det"])
        self.assertEqual(calls["quantile"], 0.2)
        self.assertEqual(len(grasps), 1)

    def test_select_best_grasp_filters_class_and_invalid(self):
        grasps = [_Grasp("bottle", 0.5), _Grasp("can", 0.95), _Grasp("bottle", 0.8, False)]
        self.assertEqual(select_best_grasp(grasps, "bottle").conf, 0.5)
        self.assertIsNone(select_best_grasp(grasps, "tissue"))

    def test_transform_and_ik_check(self):
        vision = _vision()
        poses = transform_grasp_to_base(vision, _Grasp("bottle", 0.8), np.eye(4))
        self.assertIsInstance(poses, BaseGraspPoses)
        self.assertEqual(poses.T_cam_to_base.shape, (4, 4))

        class Checker:
            def __init__(self): self.seeds = []
            def solve_pose(self, pose, q_seed=None):
                self.seeds.append(q_seed)
                return True, 0.01, np.array([1.0, 2.0])
        checker = Checker()
        result = check_grasp_poses_ik(vision, poses, checker)
        self.assertTrue(result.feasible)
        self.assertIsNone(checker.seeds[0])
        self.assertIsNotNone(checker.seeds[1])
        self.assertIsNotNone(checker.seeds[2])

    def test_ik_rejects_low_base_z_without_solving(self):
        vision = _vision()
        poses = BaseGraspPoses((0, 0, -0.3, 0, 0, 0), (0, 0, -0.3, 0, 0, 0), (0, 0, -0.3, 0, 0, 0), np.eye(4))
        class Checker:
            def solve_pose(self, *args, **kwargs): raise AssertionError("must not solve")
        self.assertFalse(check_grasp_poses_ik(vision, poses, Checker()).feasible)


if __name__ == "__main__":
    unittest.main()
