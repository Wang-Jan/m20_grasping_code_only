"""M20 Pro 的最小任务接口。

这里不实现 AOS/NOS/GOS 协议；同伴的机器狗模块只需实现这个接口，主流程
即可在不改变抓取逻辑的情况下替换真实 TCP/SDK 调用。
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Protocol


class ApproachResult(Enum):
    """D455 接近结果；只有 ``PRONE_AT_TARGET`` 可进入抓取状态。"""

    PRONE_AT_TARGET = auto()
    NO_TARGET = auto()
    INVALID_TARGET = auto()
    LOCALIZATION_FAILED = auto()
    OBSTACLE_DETECTED = auto()
    TIMEOUT = auto()
    MOTION_FAILED = auto()
    COMM_ERROR = auto()


class MachineDogInterface(Protocol):
    def init_reset(self) -> bool: ...
    def goto_target(self) -> bool: ...
    def approach(self) -> bool:
        """Return true only after the D455 approach has been verified and prone is acknowledged."""
        ...
    def is_clean(self) -> bool: ...
    def sit(self) -> bool: ...
    def stand(self) -> bool: ...
    def goto_bin(self) -> bool: ...
    def goto_home(self) -> bool: ...


__all__ = ["ApproachResult", "MachineDogInterface"]
