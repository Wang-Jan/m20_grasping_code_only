"""D435 RGB-D 相机运行时。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .arm_runtime import ArmRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class D435Runtime:
    """已经打开且预热完成的 D435 数据流。"""

    camera: Any
    serial: str
    width: int
    height: int
    fps: int
    calibration_dir: Path


@dataclass
class RGBDFrame:
    """一帧已经完成彩色-深度对齐的 D435 数据。"""

    color_bgr: np.ndarray
    depth_mm: np.ndarray


class FrameCaptureError(RuntimeError):
    """D435 返回空帧、格式错误或无有效深度时抛出。"""


def start_d435(
    runtime: ArmRuntime,
    vision_config: dict[str, Any],
    *,
    warmup_frames: int = 20,
    camera_factory: Callable[..., Any] | None = None,
) -> D435Runtime:
    """按 ``vision.yaml`` 打开、预热并基础校验 D435 RGB-D 数据流。

    默认复用原项目 ``drivers.camera.realsense.RealsenseCamera``；该驱动在
    打开时根据配置序列号选择设备，启动彩色/深度对齐流，并读取相机内参。
    本函数不读取图像用于检测，图像获取由下一阶段 ``capture_frame`` 负责。
    """

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")
    if not isinstance(vision_config, dict):
        raise TypeError("vision 配置必须是 dict")
    if warmup_frames < 0:
        raise ValueError("warmup_frames 不能小于 0")

    camera_config = vision_config.get("camera")
    calibration_config = vision_config.get("calibration")
    if not isinstance(camera_config, dict):
        raise ValueError("vision 配置缺少 camera mapping")
    if not isinstance(calibration_config, dict):
        raise ValueError("vision 配置缺少 calibration mapping")

    camera_type = str(camera_config.get("type", "")).lower()
    if camera_type != "realsense_d435cb":
        raise ValueError(
            f"当前自动抓取工程仅支持 realsense_d435cb，实际为: {camera_type!r}"
        )

    serial = str(camera_config.get("serial", "")).strip()
    if not serial:
        raise ValueError("camera.serial 未配置")

    width = int(camera_config.get("color_width", 0))
    height = int(camera_config.get("color_height", 0))
    depth_width = int(camera_config.get("depth_width", 0))
    depth_height = int(camera_config.get("depth_height", 0))
    fps = int(camera_config.get("fps", 0))
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("camera 的 color_width、color_height 和 fps 必须大于 0")
    if (depth_width, depth_height) != (width, height):
        raise ValueError("D435 彩色与深度分辨率必须相同，才能使用已有对齐驱动")

    hand_eye_value = calibration_config.get("hand_eye_path")
    if not hand_eye_value:
        raise ValueError("calibration.hand_eye_path 未配置")
    hand_eye_path = Path(str(hand_eye_value)).expanduser()
    if not hand_eye_path.is_absolute():
        hand_eye_path = PROJECT_ROOT / hand_eye_path
    calibration_dir = hand_eye_path.parent

    if camera_factory is None:
        grasp_import_path = str(runtime.grasp_root)
        if grasp_import_path not in sys.path:
            sys.path.insert(0, grasp_import_path)
        try:
            from drivers.camera.realsense import RealsenseCamera  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "无法导入原项目 RealSense 驱动；请检查 arm.robot.grasp_repo_root"
            ) from exc
        camera_factory = RealsenseCamera

    camera = camera_factory(
        width=width,
        height=height,
        fps=fps,
        calib_dir=str(calibration_dir),
        serial=serial,
    )

    try:
        camera.open()
        actual_profile = (
            int(getattr(camera, "_w", width)),
            int(getattr(camera, "_h", height)),
            int(getattr(camera, "_fps", fps)),
        )
        if actual_profile != (width, height, fps):
            raise RuntimeError(
                "D435 实际 profile 与配置不一致: "
                f"实际 {actual_profile[0]}x{actual_profile[1]}@{actual_profile[2]}，"
                f"配置 {width}x{height}@{fps}"
            )
        camera.warm_up(warmup_frames)
        intrinsics = camera.K
        if tuple(getattr(intrinsics, "shape", ())) != (3, 3):
            raise RuntimeError("D435 内参矩阵必须是 3x3")
    except Exception as exc:
        try:
            camera.close()
        except Exception:
            pass
        raise RuntimeError("D435 启动、预热或内参检查失败") from exc

    return D435Runtime(
        camera=camera,
        serial=serial,
        width=width,
        height=height,
        fps=fps,
        calibration_dir=calibration_dir,
    )


def capture_frame(d435: D435Runtime) -> RGBDFrame:
    """获取并验证一帧 D435 BGR/深度数据。

    原 RealSense 驱动返回 BGR ``uint8`` 彩色图和以毫米为单位的
    ``uint16`` 深度图。本函数只做数据边界检查，不对图像进行检测、滤波或
    坐标变换；后续视觉函数将基于这个稳定的数据契约工作。
    """

    if not isinstance(d435, D435Runtime):
        raise TypeError("d435 必须是 D435Runtime")

    color, depth = d435.camera.get_frame()
    if color is None or depth is None:
        raise FrameCaptureError("D435 返回空的彩色帧或深度帧")

    color = np.asarray(color)
    depth = np.asarray(depth)
    expected_color_shape = (d435.height, d435.width, 3)
    expected_depth_shape = (d435.height, d435.width)

    if color.size == 0 or depth.size == 0:
        raise FrameCaptureError("D435 返回空数组")
    if color.shape != expected_color_shape:
        raise FrameCaptureError(
            f"彩色帧尺寸错误: 实际 {color.shape}，期望 {expected_color_shape}"
        )
    if depth.shape != expected_depth_shape:
        raise FrameCaptureError(
            f"深度帧尺寸错误: 实际 {depth.shape}，期望 {expected_depth_shape}"
        )
    if color.dtype != np.uint8:
        raise FrameCaptureError(f"彩色帧类型错误: 实际 {color.dtype}，期望 uint8")
    if depth.dtype != np.uint16:
        raise FrameCaptureError(f"深度帧类型错误: 实际 {depth.dtype}，期望 uint16")
    if not np.any(depth > 0):
        raise FrameCaptureError("深度帧全部为零，没有可用深度数据")

    return RGBDFrame(color_bgr=color, depth_mm=depth)


def stop_d435(d435: D435Runtime) -> None:
    """关闭 D435 数据流；重复调用安全。"""

    if not isinstance(d435, D435Runtime):
        raise TypeError("d435 必须是 D435Runtime")
    close = getattr(d435.camera, "close", None)
    if callable(close):
        close()
