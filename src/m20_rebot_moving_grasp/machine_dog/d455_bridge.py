"""D455 坐标变换和 M20 导航桥接的稳定接口。"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class D455Transform(Protocol):
    def __call__(self, point_camera_m: tuple[float, float, float]) -> Any: ...


class M20MotionBridge(Protocol):
    """厂商桥接模块接口。

    ``navigate_to`` 必须阻塞到导航完成或失败后返回 bool，并应在其自身的
    厂商 SDK 超时策略内返回；本工程绝不会猜测或发送私有 M20 运动协议。
    """
    def get_map_pose(self) -> Any: ...
    def navigate_to(self, x: float, y: float, yaw: float) -> Any: ...
    def cancel_navigation(self) -> Any: ...
    def stop(self) -> Any: ...


def load_callable(spec: str, name: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise RuntimeError(f"{name} 必须使用 module:function 格式")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise RuntimeError(f"{name} 不存在可调用函数: {spec}")
    return function


def transform_point(transform: Callable[..., Any], point: tuple[float, float, float]) -> tuple[float, float, float]:
    value = transform(point)
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"), value.get("z", 0.0))
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("坐标变换必须返回长度为3的序列或 x/y/z mapping") from exc
    if len(result) != 3 or not all(math.isfinite(v) for v in result):
        raise RuntimeError("坐标变换结果必须是有限的三维坐标")
    return result


@dataclass
class ModuleMotionBridge:
    module: Any

    def _call(self, name: str, *args: Any) -> Any:
        fn = getattr(self.module, name, None)
        if not callable(fn):
            raise RuntimeError(f"M20 bridge 缺少 {name}()")
        return fn(*args)

    def get_map_pose(self) -> Any:
        return self._call("get_map_pose")

    def navigate_to(self, x: float, y: float, yaw: float) -> Any:
        return self._call("navigate_to", float(x), float(y), float(yaw))

    def cancel_navigation(self) -> Any:
        return self._call("cancel_navigation")

    def stop(self) -> Any:
        return self._call("stop")


def parse_pose(value: Any) -> tuple[float, float, float]:
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"), value.get("yaw"))
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("get_map_pose() 必须返回 (x, y, yaw) 或 mapping") from exc
    if len(result) != 3 or not all(math.isfinite(v) for v in result):
        raise RuntimeError("机器狗地图位姿必须是有限的 (x, y, yaw)")
    return result


__all__ = ["D455Transform", "M20MotionBridge", "ModuleMotionBridge", "load_callable", "transform_point", "parse_pose"]
