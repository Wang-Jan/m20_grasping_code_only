"""M20 Pro 任务适配器：导航、姿态、扫描和 D455 接近回调。"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .interface import ApproachResult
from .m20_config import M20Config
from .m20_protocol import (
    UdpJsonTransport, cancel_nav_request, map_location_request, motion_state_request,
    nav_status_request, response_items, response_ok, single_point_nav_request,
)


LOGGER = logging.getLogger(__name__)


class MachineDogCommunicationError(RuntimeError):
    pass


class M20DogAdapter:
    """可注入传输层和视觉回调的 M20 适配器。"""

    def __init__(
        self,
        config: M20Config | None = None,
        *,
        transport: UdpJsonTransport | None = None,
        approach_handler: Optional[Callable[[], Any]] = None,
        scan_handler: Optional[Callable[[], Any]] = None,
        scan_start: Optional[Callable[[], Any]] = None,
        scan_stop: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = (config or M20Config.from_env()).validate()
        self.transport = transport or UdpJsonTransport(self.config.aos_ip, self.config.udp_port)
        self.approach_handler = approach_handler
        self.scan_handler = scan_handler
        self.scan_start = scan_start
        self.scan_stop = scan_stop

    def _request(
        self,
        payload: str,
        *,
        timeout: float | None = None,
        allow_response_id_mismatch: bool = False,
    ) -> dict[str, Any] | None:
        request_timeout = self.config.command_timeout_sec if timeout is None else timeout
        if allow_response_id_mismatch and isinstance(self.transport, UdpJsonTransport):
            return self.transport.send_request(
                payload,
                timeout=request_timeout,
                allow_response_id_mismatch=True,
            )
        return self.transport.send_request(payload, timeout=request_timeout)

    def _motion(self, value: int) -> bool:
        return response_ok(self._request(motion_state_request(value)))

    def init_reset(self) -> bool:
        result = self.sit()
        if not result:
            LOGGER.error("M20 初始趴下命令未获确认（AOS %s:%s）", self.config.aos_ip, self.config.udp_port)
        return result

    def stand(self) -> bool:
        return self._motion(1)

    def sit(self) -> bool:
        return self._motion(4)

    def _load_point(self, name: str) -> dict[str, float | int] | None:
        path = self.config.resolved_points_file()
        try:
            with path.open("r", encoding="utf-8") as stream:
                points = json.load(stream)
        except (OSError, json.JSONDecodeError):
            return None
        point = points.get(name) if isinstance(points, dict) else None
        if not isinstance(point, dict):
            return None
        required = ("x", "y", "z", "yaw")
        if any(key not in point for key in required):
            return None
        try:
            normalized = {"x": float(point["x"]),
                          "y": float(point["y"]), "z": float(point["z"]), "yaw": float(point["yaw"])}
            if "point_id" in point:
                normalized["point_id"] = int(point["point_id"])
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(float(normalized[key])) for key in ("x", "y", "z", "yaw")):
            return None
        return normalized

    def _location_valid(self) -> bool:
        response = self._request(map_location_request())
        items = response_items(response)
        valid = response_ok(response) and items.get("Location") == 0
        if not valid:
            LOGGER.error("M20 定位不可用: response=%s", response)
        return valid

    def _cancel_navigation(self) -> bool:
        response = self._request(cancel_nav_request(), allow_response_id_mismatch=True)
        if not response or response_items(response).get("ErrorCode") != 0:
            return False
        deadline = time.monotonic() + self.config.cancel_confirm_timeout_sec
        while time.monotonic() < deadline:
            status = response_items(self._request(nav_status_request(), allow_response_id_mismatch=True))
            if status.get("ErrorCode") in (0x2300, 0x2302):
                return True
            if status.get("Status") == 0 and status.get("ErrorCode") in (None, 0):
                return True
            time.sleep(0.1)
        return False

    def _cancel_and_lie(self, reason: str) -> bool:
        """Best-effort M20 abort: cancel motion first, then request prone in place."""

        LOGGER.warning("M20 任务中止: %s；请求取消导航并原地趴下", reason)
        cancel_ok = self._cancel_navigation()
        if not cancel_ok:
            LOGGER.error("M20 取消导航未确认: %s；仍发送原地趴下命令", reason)
        prone_ok = self.sit()
        if not prone_ok:
            LOGGER.error("M20 原地趴下命令未获确认: %s", reason)
        return cancel_ok and prone_ok

    def _navigate(self, name: str) -> bool:
        point = self._load_point(name)
        if point is None:
            LOGGER.error("M20 导航点 %s 无有效 x/y/z/yaw 坐标: %s", name, self.config.resolved_points_file())
            return False
        if not self._location_valid():
            return False
        # Static task points are sent as coordinate navigation.  Their
        # coordinates remain in the configured point JSON, but its point_id is
        # deliberately ignored so AOS does not resolve a saved map point.
        response = self._request(
            single_point_nav_request(point, point_id=None),
            # The verified CLI waits for Type=1003 using the navigation
            # timeout, rather than the 5-second ordinary command timeout.
            # On this M20 firmware the response can arrive only after the
            # robot reaches the terminal navigation state.
            timeout=self.config.nav_timeout_sec,
            allow_response_id_mismatch=True,
        )
        submit_items = response_items(response)
        submit_error = submit_items.get("ErrorCode")
        LOGGER.info("M20 坐标导航提交: name=%s target=%s response=%s", name, point, response)
        # AOS may return 0x2300 directly when the coordinate task has already
        # completed. This is a successful terminal status, not a rejected
        # request, and no further status request is needed.
        if submit_error == 0x2300:
            LOGGER.info("M20 坐标导航已完成: name=%s", name)
            return True
        if not response_ok(response):
            LOGGER.error("M20 坐标导航提交失败: name=%s items=%s", name, submit_items)
            self._cancel_and_lie(f"坐标导航提交失败: {name}")
            return False
        deadline = time.monotonic() + self.config.nav_timeout_sec
        while time.monotonic() < deadline:
            items = response_items(self._request(nav_status_request(), allow_response_id_mismatch=True))
            # Coordinate navigation is not correlated with a saved map point;
            # firmware may report Value=0 or an internal task number. The UDP
            # response is already matched by message ID, so accept either.
            if items.get("Status") == 4 or items.get("ErrorCode") == 0x2300:
                LOGGER.info("M20 坐标导航已完成: name=%s status=%s", name, items)
                return True
            if items.get("ErrorCode") not in (None, 0):
                LOGGER.error("M20 坐标导航失败: name=%s status=%s", name, items)
                self._cancel_and_lie(f"坐标导航执行失败: {name}")
                return False
            time.sleep(0.1)
        self._cancel_and_lie(f"坐标导航超时: {name}")
        LOGGER.error("M20 坐标导航超时: name=%s timeout_sec=%s", name, self.config.nav_timeout_sec)
        return False

    def goto_target(self) -> bool:
        return self._navigate(self.config.scan_point_name)

    def goto_bin(self) -> bool:
        return self._navigate(self.config.bin_point_name)

    def goto_home(self) -> bool:
        return self._navigate(self.config.home_point_name) and self.sit()

    @staticmethod
    def _has_detections(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict) and "detections" in value:
            value = value["detections"]
        count = getattr(value, "count", None)
        if isinstance(count, (int, float)):
            return count > 0
        try:
            return len(value) > 0
        except TypeError:
            return bool(value)

    def is_clean(self) -> bool:
        if self.scan_handler is None:
            raise MachineDogCommunicationError("未配置 D455 scan_handler")
        started = False
        target_found = False
        try:
            started = True
            try:
                if self.scan_start is not None:
                    self.scan_start()
            except (OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
                raise MachineDogCommunicationError(f"D455 扫描启动失败: {exc}") from exc
            deadline = time.monotonic() + self.config.clean_scan_sec
            while time.monotonic() < deadline:
                try:
                    detections = self.scan_handler()
                except (OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
                    # pyrealsense2 reports a missing frame as RuntimeError,
                    # not TimeoutError.  Preserve that context for the
                    # operator before the state machine enters its fail-safe
                    # emergency-stop path.
                    raise MachineDogCommunicationError(f"D455 扫描失败: {exc}") from exc
                if self._has_detections(detections):
                    # 主状态机需要在发现目标后立即进入 approach；若一直
                    # 等到“连续10秒无目标”，将永远无法进入抓取分支。
                    target_found = True
                    return False
                time.sleep(0.1)
            return True
        finally:
            if started and self.scan_stop is not None and not target_found:
                self.scan_stop()

    @staticmethod
    def _map_approach_result(result: Any) -> ApproachResult:
        return result if isinstance(result, ApproachResult) else ApproachResult.COMM_ERROR

    def approach(self) -> bool:
        if self.approach_handler is None:
            if self.scan_stop is not None:
                self.scan_stop()
            return False
        try:
            return self._map_approach_result(self.approach_handler()) is ApproachResult.PRONE_AT_TARGET
        except (TimeoutError, OSError, ConnectionError):
            return False
        finally:
            if self.scan_stop is not None:
                self.scan_stop()

    def emergency_stop(self) -> bool:
        """Compatibility entry for task aborts; M20 stops by lying down, not soft e-stop."""

        return self._cancel_and_lie("程序异常退出")


__all__ = ["M20DogAdapter", "M20Config", "ApproachResult", "MachineDogCommunicationError"]
