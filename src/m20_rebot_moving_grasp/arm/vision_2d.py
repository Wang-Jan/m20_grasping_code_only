"""复用 mobile_2d_grasp.py 的二维 OBB/RGB-D 视觉链路。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .arm_camera import D435Runtime, RGBDFrame
from ..common.model_assets import require_mobileclip_weight
from ..common.torch_numpy import install_torch_numpy_compat


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class DetectionBatch:
    raw_results: list[Any]
    detections: list[Any]


@dataclass
class BaseGraspPoses:
    grasp_pose: tuple[float, ...]
    pregrasp_pose: tuple[float, ...]
    retreat_pose: tuple[float, ...]
    T_cam_to_base: np.ndarray


@dataclass
class IKCheckResult:
    feasible: bool
    pre_error: float
    grasp_error: float
    retreat_error: float
    q_pre: Any = None
    q_grasp: Any = None


@dataclass
class Vision2DRuntime:
    """二维视觉推理和坐标变换运行时。"""

    model: Any
    yolo_options: dict[str, Any]
    config: dict[str, Any]
    camera_intrinsics: np.ndarray
    hand_eye: np.ndarray
    grasp_root: Path
    detect_objects_fn: Optional[Callable[..., Any]] = None
    estimate_grasps_fn: Optional[Callable[..., Any]] = None


def _legacy_vision_config(config: dict[str, Any]) -> dict[str, Any]:
    """将新工程配置转换为原项目工具所需的字段结构。"""

    detection = config["detection"]
    grasp = config["grasp_2d"]
    return {
        "camera": config["camera"],
        "calibration": config["calibration"],
        "detection": {
            "conf_threshold": detection["confidence_threshold"],
            "iou_threshold": detection["iou_threshold"],
        },
        "grasp_pipeline": {
            "infer_every_live": detection["infer_every_live"],
            "grasp": {
                "depth_quantile": grasp["depth_quantile"],
                "pregrasp_offset_m": grasp["pregrasp_offset_m"],
                "retreat_offset_m": grasp["retreat_offset_m"],
                "insertion_depth_m": grasp["insertion_depth_m"],
                "min_base_z_m": grasp["min_base_z_m"],
            },
        },
        "yolo": {
            "model_name": detection["model_name"],
            "device": detection["device"],
            "use_world": True,
            "custom_classes": detection.get("custom_classes", []),
        },
    }


def create_2d_vision_runtime(
    d435: D435Runtime,
    vision_config: dict[str, Any],
    grasp_root: Path,
    *,
    model: Any = None,
    hand_eye: np.ndarray | None = None,
    load_model: bool = True,
) -> Vision2DRuntime:
    """创建二维视觉运行时，兼容新配置和原项目工具函数。

    ``model`` 和 ``hand_eye`` 可在测试或离线模式中注入；真机运行时省略它们，
    函数会从配置加载已经训练好的 YOLO 模型和手眼矩阵。
    """

    if not isinstance(d435, D435Runtime):
        raise TypeError("d435 必须是 D435Runtime")
    if not isinstance(vision_config, dict):
        raise TypeError("vision_config 必须是 dict")
    if not isinstance(grasp_root, Path):
        grasp_root = Path(grasp_root).expanduser().resolve()
    if not grasp_root.is_dir():
        raise FileNotFoundError(f"二维视觉工具目录不存在: {grasp_root}")

    legacy_config = _legacy_vision_config(vision_config)
    if hand_eye is None:
        hand_eye_path = Path(str(vision_config["calibration"]["hand_eye_path"])).expanduser()
        if not hand_eye_path.is_absolute():
            hand_eye_path = PROJECT_ROOT / hand_eye_path
        if not hand_eye_path.is_file():
            raise FileNotFoundError(f"D435 手眼标定文件不存在: {hand_eye_path}")
        data = np.load(str(hand_eye_path), allow_pickle=False)
        if "T_result" not in data:
            raise ValueError("手眼标定文件缺少 T_result")
        hand_eye = np.asarray(data["T_result"], dtype=np.float64)

    hand_eye = np.asarray(hand_eye, dtype=np.float64)
    if hand_eye.shape != (4, 4) or not np.all(np.isfinite(hand_eye)):
        raise ValueError("手眼变换必须是有限的4x4矩阵")

    if model is None and load_model:
        install_torch_numpy_compat()
        require_mobileclip_weight()
        import_path = str(grasp_root)
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
        try:
            from utils.yolo_utils import load_yolo  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("无法导入原项目 YOLO 工具") from exc
        model, yolo_options = load_yolo(
            legacy_config,
            project_root=PROJECT_ROOT,
            extra_classes=[],
        )
    else:
        yolo_options = {
            "device": legacy_config["yolo"]["device"],
            "conf": legacy_config["detection"]["conf_threshold"],
            "iou": legacy_config["detection"]["iou_threshold"],
            "infer_every": legacy_config["grasp_pipeline"]["infer_every_live"],
        }

    if model is None:
        raise RuntimeError("YOLO 模型未创建；请检查模型文件或传入 model")

    return Vision2DRuntime(
        model=model,
        yolo_options=yolo_options,
        config=legacy_config,
        camera_intrinsics=np.asarray(d435.camera.K, dtype=np.float64),
        hand_eye=hand_eye,
        grasp_root=grasp_root,
    )


def detect_2d_targets(vision: Vision2DRuntime, frame: RGBDFrame) -> DetectionBatch:
    """对一帧 BGR 图像执行 YOLO 检测并返回原始结果和目标列表。"""

    if not isinstance(vision, Vision2DRuntime) or not isinstance(frame, RGBDFrame):
        raise TypeError("vision 和 frame 类型不正确")
    if vision.detect_objects_fn is None:
        install_torch_numpy_compat()
        import_path = str(vision.grasp_root)
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
        from utils.yolo_utils import detect_objects  # type: ignore[import-not-found]

        detect_fn = detect_objects
    else:
        detect_fn = vision.detect_objects_fn

    raw_results, detections = detect_fn(
        vision.model,
        frame.color_bgr,
        vision.yolo_options,
    )
    return DetectionBatch(list(raw_results), list(detections))


def estimate_2d_grasps(
    vision: Vision2DRuntime,
    frame: RGBDFrame,
    detections: DetectionBatch,
) -> list[Any]:
    """基于 YOLO 结果、深度图和内参估计二维 OBB/RGB-D 抓取候选。"""

    if vision.estimate_grasps_fn is None:
        import_path = str(vision.grasp_root)
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
        from utils.ordinary_grasp import estimate_grasps  # type: ignore[import-not-found]

        estimate_fn = estimate_grasps
    else:
        estimate_fn = vision.estimate_grasps_fn

    depth_quantile = float(vision.config["grasp_pipeline"]["grasp"]["depth_quantile"])
    return list(
        estimate_fn(
            detections.raw_results,
            frame.depth_mm,
            vision.camera_intrinsics,
            depth_quantile=depth_quantile,
        )
    )


def select_best_grasp(grasps: list[Any], target_class: str | None = None) -> Any | None:
    """过滤无效候选和类别后，选择置信度最高的二维抓取候选。"""

    def matches(grasp: Any) -> bool:
        if not target_class:
            return True
        actual = str(getattr(grasp, "class_name", "")).casefold().strip()
        target = target_class.casefold().strip()
        return actual == target or target in actual

    valid = [grasp for grasp in grasps if bool(getattr(grasp, "is_valid", False)) and matches(grasp)]
    return max(valid, key=lambda grasp: float(getattr(grasp, "conf", 0.0))) if valid else None


def transform_grasp_to_base(
    vision: Vision2DRuntime,
    grasp: Any,
    tcp_to_base: np.ndarray,
) -> BaseGraspPoses:
    """使用 D435 手眼矩阵将候选抓取位姿转换到机械臂基座。"""

    if grasp is None or not bool(getattr(grasp, "is_valid", False)):
        raise ValueError("不能转换无效抓取候选")
    tcp_to_base = np.asarray(tcp_to_base, dtype=np.float64)
    if tcp_to_base.shape != (4, 4):
        raise ValueError("tcp_to_base 必须是4x4矩阵")
    if not np.all(np.isfinite(tcp_to_base)):
        raise ValueError("tcp_to_base 包含非有限值")

    import_path = str(vision.grasp_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)
    from utils.camera_utils import compose_cam_to_base_transform  # type: ignore[import-not-found]
    from utils.transforms import transform_grasp_pose_to_base_with_retreat  # type: ignore[import-not-found]

    T_cam_to_base = compose_cam_to_base_transform(tcp_to_base, vision.hand_eye, vision.config)
    grasp_cfg = vision.config["grasp_pipeline"]["grasp"]
    grasp_pose, pregrasp_pose, retreat_pose = transform_grasp_pose_to_base_with_retreat(
        grasp.position,
        grasp.tcp_rotation,
        T_cam_to_base,
        float(grasp_cfg["pregrasp_offset_m"]),
        float(grasp_cfg["retreat_offset_m"]),
        float(grasp_cfg["insertion_depth_m"]),
    )
    return BaseGraspPoses(grasp_pose, pregrasp_pose, retreat_pose, T_cam_to_base)


def check_grasp_poses_ik(
    vision: Vision2DRuntime,
    poses: BaseGraspPoses,
    ik_checker: Any,
) -> IKCheckResult:
    """依次检查 pregrasp、grasp、retreat 三个位姿的 IK 可达性。"""

    min_base_z = float(vision.config["grasp_pipeline"]["grasp"]["min_base_z_m"])
    if min(pose[2] for pose in (poses.pregrasp_pose, poses.grasp_pose, poses.retreat_pose)) < min_base_z:
        return IKCheckResult(False, float("inf"), float("inf"), float("inf"))

    pre_ok, pre_error, q_pre = ik_checker.solve_pose(poses.pregrasp_pose)
    grasp_ok, grasp_error, q_grasp = ik_checker.solve_pose(poses.grasp_pose, q_seed=q_pre)
    retreat_ok, retreat_error, _ = ik_checker.solve_pose(poses.retreat_pose, q_seed=q_grasp)
    return IKCheckResult(
        bool(pre_ok and grasp_ok and retreat_ok),
        float(pre_error),
        float(grasp_error),
        float(retreat_error),
        q_pre=q_pre,
        q_grasp=q_grasp,
    )
