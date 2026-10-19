"""旧版外部桥接 D455 回调模块。

主入口已改用 :class:`D455ApproachController`，不再加载此模块。本文件仅保留给
曾依赖外部地图变换/桥接模块的手动工具调用：
``--d455-config``、``--d455-transform`` (module:function)、``--m20-bridge-module``。
外部桥接模块须提供 get_map_pose/navigate_to/cancel_navigation/stop。
"""

from __future__ import annotations

import importlib
import math
import threading
from pathlib import Path

import yaml

from .d455_bridge import ModuleMotionBridge, load_callable, parse_pose, transform_point
from .d455_vision import D455Config, D455Detection, D455Vision

_vision: D455Vision | None = None
_bridge: ModuleMotionBridge | None = None
_transform = None
_last_detections: tuple[D455Detection, ...] = ()
_config_path = "config/d455.yaml"
_transform_spec = ""
_bridge_module = ""


def configure(*, d455_config: str | None = "config/d455.yaml", transform_spec: str | None = None,
              bridge_module: str | None = None) -> None:
    """由主入口在真机启动前注入外部桥接位置，且不打开任何硬件。"""
    global _config_path, _transform_spec, _bridge_module, _bridge, _transform
    _config_path = d455_config or "config/d455.yaml"
    _transform_spec = transform_spec or ""
    _bridge_module = bridge_module or ""
    _bridge = None
    _transform = None


def _config() -> D455Config:
    path = Path(_config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as stream:
        return D455Config.from_mapping(yaml.safe_load(stream))


def scan_start() -> None:
    global _vision
    if _vision is None:
        _vision = D455Vision(_config())
    _vision.open()


def scan_handler():
    global _last_detections
    if _vision is None:
        raise RuntimeError("D455 尚未启动")
    detections = _vision.detect_once()
    # 只缓存检测帧。相机光轴深度在 D455 下俯安装时并非水平距离，真正的
    # 目标选择必须在转成地图坐标后，以机器人 x/y 平面距离完成。
    _last_detections = tuple(detections)
    return detections


def _bridge_instance() -> ModuleMotionBridge:
    global _bridge
    if _bridge is None:
        module_name = _bridge_module.strip()
        if not module_name:
            raise RuntimeError("未设置 M20_BRIDGE_MODULE；拒绝发送机器狗导航命令")
        _bridge = ModuleMotionBridge(importlib.import_module(module_name))
    return _bridge


def _navigate_with_timeout(bridge: ModuleMotionBridge, x: float, y: float, yaw: float,
                           timeout_sec: float) -> bool:
    """限制外部导航调用；超时后主动取消并停车。

    厂商 SDK 调用运行在守护线程中，避免一个无响应的 SDK 调用阻塞主任务。
    桥接实现仍应自行保证 ``cancel_navigation`` 能中断底层导航。
    """
    done = threading.Event()
    result: dict[str, object] = {}

    def call() -> None:
        try:
            result["value"] = bridge.navigate_to(x, y, yaw)
        except BaseException as exc:  # 需要将厂商 SDK 异常传回主任务
            result["error"] = exc
        finally:
            done.set()

    threading.Thread(target=call, name="d455-m20-navigation", daemon=True).start()
    if not done.wait(timeout_sec):
        bridge.cancel_navigation()
        bridge.stop()
        return False
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result.get("value") is True


def approach_handler() -> bool:
    if not _last_detections:
        return False
    transform_spec = _transform_spec.strip()
    if not transform_spec:
        raise RuntimeError("未设置 D455_TRANSFORM=module:function；拒绝使用未标定坐标")
    cfg = _config()
    bridge = _bridge_instance()
    robot_x, robot_y, _ = parse_pose(bridge.get_map_pose())
    transform = load_callable(transform_spec, "D455_TRANSFORM")
    map_candidates = [
        transform_point(transform, detection.point_camera_m)
        for detection in _last_detections
    ]
    # ``z`` 为光轴深度，仅用于深度有效性过滤。此处专门依据地图 x/y
    # 选择最近目标，能正确处理相机下俯 45 度安装。
    point_map = min(map_candidates, key=lambda point: math.hypot(point[0] - robot_x, point[1] - robot_y))
    distance = math.hypot(point_map[0] - robot_x, point_map[1] - robot_y)
    if distance <= cfg.safe_stop_distance_m:
        return True
    ratio = (distance - cfg.safe_stop_distance_m) / distance
    target_x = robot_x + (point_map[0] - robot_x) * ratio
    target_y = robot_y + (point_map[1] - robot_y) * ratio
    yaw = math.atan2(point_map[1] - robot_y, point_map[0] - robot_x)
    try:
        if not _navigate_with_timeout(bridge, target_x, target_y, yaw, cfg.max_navigation_sec):
            bridge.cancel_navigation()
            bridge.stop()
            return False
        return True
    except Exception:
        try:
            bridge.cancel_navigation()
        finally:
            bridge.stop()
        raise


def scan_stop() -> None:
    global _vision, _last_detections
    if _vision is not None:
        _vision.close()
    _vision = None
    _last_detections = ()


__all__ = ["configure", "scan_start", "scan_handler", "approach_handler", "scan_stop"]
