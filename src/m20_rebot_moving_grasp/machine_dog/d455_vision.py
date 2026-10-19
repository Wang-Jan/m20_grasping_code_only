"""D455 YOLOE 目标检测与相机坐标计算。

本模块只负责相机和视觉，不发送机器狗或机械臂控制命令。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ..common.config import ConfigError
from ..common.model_assets import require_mobileclip_weight
from ..common.torch_numpy import install_torch_numpy_compat, tensor_to_numpy

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class D455Config:
    serial: str
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    fps: int
    model_name: str
    device: str
    confidence_threshold: float
    iou_threshold: float
    custom_classes: tuple[str, ...]
    min_depth_m: float
    max_depth_m: float
    safe_stop_distance_m: float
    approach_trigger_max_range_m: float
    max_navigation_sec: float
    pitch_down_deg: float = 45.0
    depth_patch_radius_px: int = 2
    min_valid_depth_samples: int = 3
    inference_size: int = 640
    base_link_offset_x_m: float = 0.0
    base_link_offset_y_m: float = 0.0
    pre_lie_forward_m: float = 0.10
    perpendicular_side: str = "right"
    approach_point_id: int = 901
    gait: int = 0x3002
    speed: int = 1
    obs_mode: int = 0
    nav_mode: int = 0
    frame_timeout_sec: float = 15.0

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "D455Config":
        try:
            camera = mapping["camera"]
            detection = mapping["detection"]
            target = mapping["target"]
            navigation = mapping.get("navigation", {})
            values = cls(
                serial=str(camera.get("serial", "")).strip(),
                color_width=int(camera["color_width"]), color_height=int(camera["color_height"]),
                depth_width=int(camera["depth_width"]), depth_height=int(camera["depth_height"]),
                fps=int(camera["fps"]), model_name=str(detection["model_name"]),
                device=str(detection.get("device", "0")),
                confidence_threshold=float(detection["confidence_threshold"]),
                iou_threshold=float(detection["iou_threshold"]),
                custom_classes=tuple(str(x).strip() for x in detection.get("custom_classes", ()) if str(x).strip()),
                min_depth_m=float(target["min_depth_m"]), max_depth_m=float(target["max_depth_m"]),
                safe_stop_distance_m=float(target["safe_stop_distance_m"]),
                approach_trigger_max_range_m=float(target["approach_trigger_max_range_m"]),
                max_navigation_sec=float(target["max_navigation_sec"]),
                pitch_down_deg=float(target.get("pitch_down_deg", 45.0)),
                depth_patch_radius_px=int(target.get("depth_patch_radius_px", 2)),
                min_valid_depth_samples=int(target.get("min_valid_depth_samples", 3)),
                inference_size=int(detection.get("inference_size", 640)),
                base_link_offset_x_m=float(navigation.get("base_link_offset_x_m", 0.0)),
                base_link_offset_y_m=float(navigation.get("base_link_offset_y_m", 0.0)),
                pre_lie_forward_m=float(navigation.get("pre_lie_forward_m", 0.10)),
                perpendicular_side=str(navigation.get("perpendicular_side", "right")),
                approach_point_id=int(navigation.get("approach_point_id", 901)),
                gait=int(navigation.get("gait", 0x3002)),
                speed=int(navigation.get("speed", 1)),
                obs_mode=int(navigation.get("obs_mode", 0)),
                nav_mode=int(navigation.get("nav_mode", 0)),
                frame_timeout_sec=float(camera.get("frame_timeout_sec", 15.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"D455 配置字段无效: {exc}") from exc
        if not values.serial:
            raise ConfigError("d455.camera.serial 不能为空；请填写 D455 序列号")
        if min(values.color_width, values.color_height, values.depth_width, values.depth_height, values.fps) <= 0:
            raise ConfigError("D455 分辨率和 fps 必须大于0")
        if not 0 <= values.confidence_threshold <= 1 or not 0 <= values.iou_threshold <= 1:
            raise ConfigError("D455 YOLO 阈值必须位于[0,1]")
        if not (0 < values.min_depth_m < values.max_depth_m):
            raise ConfigError("D455 深度范围无效")
        if not (0 < values.safe_stop_distance_m < values.max_depth_m):
            raise ConfigError("D455 safe_stop_distance_m 无效")
        if (
            not math.isfinite(values.approach_trigger_max_range_m)
            or values.approach_trigger_max_range_m <= values.safe_stop_distance_m
        ):
            raise ConfigError("D455 近距触发距离必须为大于停靠距离的有限数值")
        if values.max_navigation_sec <= 0:
            raise ConfigError("D455 max_navigation_sec 必须大于0")
        if not -90.0 <= values.pitch_down_deg <= 90.0:
            raise ConfigError("D455 pitch_down_deg 必须位于[-90,90]")
        if values.depth_patch_radius_px < 0 or values.min_valid_depth_samples <= 0:
            raise ConfigError("D455 深度 patch 参数无效")
        if values.inference_size <= 0:
            raise ConfigError("D455 inference_size 必须大于0")
        if not values.model_name.strip():
            raise ConfigError("D455 model_name 不能为空")
        if not all(math.isfinite(number) for number in (
            values.base_link_offset_x_m, values.base_link_offset_y_m, values.pre_lie_forward_m,
        )):
            raise ConfigError("D455 接近参数必须为有限数值")
        if values.pre_lie_forward_m < 0:
            raise ConfigError("D455 接近距离无效")
        if values.perpendicular_side not in ("left", "right"):
            raise ConfigError("D455 perpendicular_side 必须为 left 或 right")
        if values.approach_point_id < 1 or values.gait < 0 or values.speed not in (0, 1, 2):
            raise ConfigError("D455 导航参数无效")
        if values.obs_mode not in (0, 1) or values.nav_mode not in (0, 1):
            raise ConfigError("D455 导航模式无效")
        if not math.isfinite(values.frame_timeout_sec) or not 0 < values.frame_timeout_sec <= 60:
            raise ConfigError("D455 frame_timeout_sec 必须位于(0, 60]秒")
        return values


@dataclass(frozen=True)
class D455Detection:
    class_name: str
    confidence: float
    center_px: tuple[float, float]
    depth_m: float
    point_camera_m: tuple[float, float, float]
    contact_px: tuple[float, float] | None = None
    relative_xy_m: tuple[float, float] | None = None
    range_m: float | None = None


def valid_depth_median(values: np.ndarray, config: D455Config) -> float | None:
    """返回位于配置距离范围内的中位深度（米）。"""
    valid = np.asarray(values, dtype=np.float32)
    valid = valid[np.isfinite(valid) & (valid >= config.min_depth_m) & (valid <= config.max_depth_m)]
    return float(np.median(valid)) if valid.size else None


def _finite_detection_range_m(detection: D455Detection) -> float | None:
    """Return a usable three-dimensional color-lens range, if present."""

    if detection.range_m is None:
        return None
    try:
        range_m = float(detection.range_m)
    except (TypeError, ValueError):
        return None
    return range_m if math.isfinite(range_m) and range_m > 0.0 else None


def choose_nearest_detection(
    detections: Iterable[D455Detection],
    *,
    max_range_m: float | None = None,
) -> D455Detection | None:
    """在已通过类别、置信度和深度过滤的候选中选最近目标。

    ``max_range_m`` 仅接受 D455 彩色镜头到接触点的三维 ``range_m``；
    缺失或无效的 ``range_m`` 不会回退使用光轴 ``depth_m``。
    """

    items = list(detections)
    if max_range_m is not None:
        try:
            maximum = float(max_range_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_range_m 必须为有限非负数") from exc
        if not math.isfinite(maximum) or maximum < 0.0:
            raise ValueError("max_range_m 必须为有限非负数")
        items = [
            item for item in items
            if (range_m := _finite_detection_range_m(item)) is not None and range_m <= maximum
        ]
        return min(items, key=lambda item: float(item.range_m)) if items else None
    return min(items, key=lambda item: item.range_m if item.range_m is not None else item.depth_m) if items else None


def camera_point_to_robot_aligned_xy(
    right_m: float, down_m: float, forward_m: float, pitch_down_deg: float,
) -> tuple[float, float]:
    """Convert RealSense optical coordinates to forward/left horizontal axes."""

    pitch = np.deg2rad(float(pitch_down_deg))
    x_forward = np.cos(pitch) * float(forward_m) - np.sin(pitch) * float(down_m)
    y_left = -float(right_m)
    return float(x_forward), float(y_left)


def mask_bottom_contact_pixel(
    mask: np.ndarray | None, box_xyxy: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Estimate the lower contact pixel from a mask, with box-bottom fallback."""

    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    fallback = ((x1 + x2) / 2.0, y2)
    if mask is None:
        return fallback
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        return fallback
    ys, xs = np.nonzero(mask_array > 0.5)
    if ys.size < 3:
        return fallback
    max_y = float(np.max(ys))
    min_y = float(np.min(ys))
    band_height = max(2.0, min(20.0, (max_y - min_y) * 0.08))
    bottom_xs = xs[ys >= max_y - band_height]
    if bottom_xs.size == 0:
        return fallback
    return float(np.median(bottom_xs)), max_y


def depth_patch_values(depth_m: np.ndarray, pixel: tuple[float, float], radius_px: int) -> np.ndarray:
    """Return a bounded depth patch around a color-aligned pixel."""

    center_u, center_v = (int(round(value)) for value in pixel)
    height, width = depth_m.shape[:2]
    u1, u2 = max(0, center_u - radius_px), min(width, center_u + radius_px + 1)
    v1, v2 = max(0, center_v - radius_px), min(height, center_v + radius_px + 1)
    return depth_m[v1:v2, u1:u2].reshape(-1)


def _model_path(name: str) -> Path:
    path = Path(name).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / "models" / path


def _result_items(result: Any) -> Iterable[
    tuple[str, float, np.ndarray | None, tuple[float, float], tuple[float, float, float, float]]
]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return
    names = getattr(result, "names", {})
    masks = getattr(result, "masks", None)
    mask_data = getattr(masks, "data", None)
    for index in range(len(boxes)):
        try:
            cls_id = int(boxes.cls[index].item())
            confidence = float(boxes.conf[index].item())
            xyxy = tensor_to_numpy(boxes.xyxy[index]).astype(float, copy=False).reshape(4)
            center = ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)
            mask = None
            if mask_data is not None and index < len(mask_data):
                mask = tensor_to_numpy(mask_data[index]).astype(np.float32, copy=False)
            name = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(cls_id)
            yield name, confidence, mask, center, tuple(float(value) for value in xyxy)
        except (AttributeError, IndexError, TypeError, ValueError):
            continue


class D455Vision:
    def __init__(self, config: D455Config, *, model: Any = None, pipeline: Any = None) -> None:
        self.config = config
        self.model = model
        self.pipeline = pipeline
        self.align = None
        self.profile = None
        self._depth_scale = 0.001
        self.last_color_bgr: np.ndarray | None = None
        self.last_depth_m: np.ndarray | None = None
        self.last_result: Any = None

    def open(self) -> None:
        if self.model is None:
            install_torch_numpy_compat()
            from ultralytics import YOLO
            path = _model_path(self.config.model_name)
            if not path.is_file():
                raise FileNotFoundError(f"D455 YOLO 模型不存在: {path}")
            self.model = YOLO(str(path))
        if self.config.custom_classes and hasattr(self.model, "set_classes"):
            if self.model.__class__.__module__.startswith("ultralytics"):
                require_mobileclip_weight()
            self.model.set_classes(list(self.config.custom_classes))
        if self.pipeline is None:
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            camera = rs.config()
            camera.enable_device(self.config.serial)
            camera.enable_stream(rs.stream.color, self.config.color_width, self.config.color_height, rs.format.bgr8, self.config.fps)
            camera.enable_stream(rs.stream.depth, self.config.depth_width, self.config.depth_height, rs.format.z16, self.config.fps)
            self.profile = self.pipeline.start(camera)
            self.align = rs.align(rs.stream.color)
            self._depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())

    def close(self) -> None:
        if self.pipeline is not None and hasattr(self.pipeline, "stop"):
            self.pipeline.stop()
        self.pipeline = None
        self.profile = None

    def warm_up(self, frame_count: int = 30) -> None:
        """丢弃启动帧，使自动曝光和深度流稳定；不执行 YOLO 推理。"""
        if self.pipeline is None:
            raise RuntimeError("D455Vision 尚未 open")
        if frame_count < 0:
            raise ValueError("frame_count 不能小于0")
        for _ in range(frame_count):
            self._wait_for_frames()

    def _wait_for_frames(self) -> Any:
        if self.pipeline is None:
            raise RuntimeError("D455Vision 尚未 open")
        timeout_ms = max(1, int(round(self.config.frame_timeout_sec * 1000.0)))
        return self.pipeline.wait_for_frames(timeout_ms)

    def detect_once(self) -> list[D455Detection]:
        if self.pipeline is None or self.model is None:
            raise RuntimeError("D455Vision 尚未 open")
        frames = self._wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return []
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale
        install_torch_numpy_compat()
        results = self.model.predict(
            color,
            device=self.config.device,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.inference_size,
            verbose=False,
        )
        self.last_color_bgr = color.copy()
        self.last_depth_m = depth.copy()
        self.last_result = results[0] if results else None
        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        detections: list[D455Detection] = []
        for name, confidence, mask, center, box_xyxy in _result_items(self.last_result):
            if self.config.custom_classes and name not in self.config.custom_classes:
                continue
            if mask is None:
                full_mask = None
            else:
                full_mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            contact_px = mask_bottom_contact_pixel(full_mask, box_xyxy)
            patch = depth_patch_values(depth, contact_px, self.config.depth_patch_radius_px)
            if full_mask is None:
                # A small center fallback keeps box-only models usable while
                # still applying the same robust depth filtering.
                patch = np.concatenate((patch, depth_patch_values(depth, center, self.config.depth_patch_radius_px)))
            valid_values = patch[
                np.isfinite(patch)
                & (patch >= self.config.min_depth_m)
                & (patch <= self.config.max_depth_m)
            ]
            if valid_values.size < self.config.min_valid_depth_samples:
                continue
            depth_m = valid_depth_median(valid_values, self.config)
            if depth_m is None:
                continue
            point = rs_deproject(intr, contact_px, depth_m)
            relative_xy = camera_point_to_robot_aligned_xy(*point, self.config.pitch_down_deg)
            distance = float(np.linalg.norm(np.asarray(point, dtype=np.float64)))
            detections.append(D455Detection(
                name, confidence, center, depth_m, point,
                contact_px=contact_px, relative_xy_m=relative_xy, range_m=distance,
            ))
        return detections


def rs_deproject(intrinsics: Any, pixel: tuple[float, float], depth_m: float) -> tuple[float, float, float]:
    import pyrealsense2 as rs
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(pixel[0]), float(pixel[1])], float(depth_m))
    result = tuple(float(v) for v in point)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise RuntimeError("D455 深度反投影结果无效")
    return result


__all__ = [
    "D455Config", "D455Detection", "D455Vision", "choose_nearest_detection",
    "camera_point_to_robot_aligned_xy", "depth_patch_values", "mask_bottom_contact_pixel",
    "rs_deproject", "valid_depth_median",
]
