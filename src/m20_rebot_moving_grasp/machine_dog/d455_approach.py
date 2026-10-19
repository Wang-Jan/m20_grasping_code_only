"""D455 目标到 M20 停靠与趴下的已验证接近流程。

本模块按 ``/home/g/Downloads/m20_cli.py`` 的 D455 接近路径实现，但通过
工程内的消息 ID 校验 UDP 传输层发包。它不依赖外部地图变换或厂商桥接模块：
D455 输出的相对水平坐标直接按已验证的 base_link 约定参与导航计算。
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable

from .d455_vision import D455Config, D455Detection, D455Vision, choose_nearest_detection
from .interface import ApproachResult
from .m20_config import M20Config
from .m20_protocol import (
    UdpJsonTransport,
    cancel_nav_request,
    map_location_request,
    motion_state_request,
    nav_status_request,
    perception_state_request,
    response_items,
    single_point_nav_request,
)


LOGGER = logging.getLogger(__name__)


def _is_finite_positive(value: object) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


class D455ApproachController:
    """Own D455 scan lifecycle and execute one verified M20 approach."""

    def __init__(
        self,
        config: D455Config,
        m20_config: M20Config,
        transport: UdpJsonTransport,
        *,
        vision: D455Vision | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.m20_config = m20_config
        self.transport = transport
        self._vision = vision or D455Vision(config)
        self._sleep = sleep
        self._monotonic = monotonic
        self._stop_requested = stop_requested or (lambda: False)
        self._last_detections: tuple[D455Detection, ...] = ()
        # AOS may acknowledge Type=1003 with the final 0x2300 response instead
        # of an immediate acceptance.  Remember that terminal response so the
        # following status poll does not mistake an already-completed command
        # for an idle task and time out.
        self._completed_navigation_ids: set[int] = set()
        self._last_navigation_submit_failure: ApproachResult | None = None

    def scan_start(self) -> None:
        self._vision.open()
        # Match the standalone D455 verification path before attempting the
        # first YOLO inference.  This also gives auto-exposure/depth streams
        # time to stabilize after the device is selected by serial number.
        warm_up = getattr(self._vision, "warm_up", None)
        if callable(warm_up):
            warm_up(30)

    def scan_handler(self) -> tuple[D455Detection, ...]:
        detections = tuple(self._vision.detect_once())
        trigger_range_m = self.config.approach_trigger_max_range_m
        eligible = tuple(
            detection for detection in detections
            if choose_nearest_detection((detection,), max_range_m=trigger_range_m) is not None
        )
        self._last_detections = eligible

        finite_ranges = [
            float(detection.range_m)
            for detection in detections
            if detection.range_m is not None and _is_finite_positive(detection.range_m)
        ]
        far_ranges = [range_m for range_m in finite_ranges if range_m > trigger_range_m]
        if far_ranges:
            LOGGER.info(
                "D455 超距目标不触发接近: nearest_far_range_m=%.3f trigger_max_range_m=%.3f ignored=%s",
                min(far_ranges), trigger_range_m, len(far_ranges),
            )
        elif detections and not eligible:
            LOGGER.info(
                "D455 检测结果缺少有效三维镜头距离，当前帧不触发接近: trigger_max_range_m=%.3f",
                trigger_range_m,
            )
        return self._last_detections

    def scan_stop(self, *, preserve_detections: bool = False) -> None:
        try:
            self._vision.close()
        finally:
            if not preserve_detections:
                self._last_detections = ()

    def _request(
        self,
        payload: str,
        *,
        timeout: float | None = None,
        allow_response_id_mismatch: bool = False,
    ):
        request_timeout = self.m20_config.command_timeout_sec if timeout is None else timeout
        if allow_response_id_mismatch and isinstance(self.transport, UdpJsonTransport):
            return self.transport.send_request(
                payload,
                timeout=request_timeout,
                allow_response_id_mismatch=True,
            )
        return self.transport.send_request(payload, timeout=request_timeout)

    @staticmethod
    def _finite_pose(items: dict) -> tuple[float, float, float, float] | None:
        try:
            pose = (
                float(items["PosX"]),
                float(items["PosY"]),
                float(items.get("PosZ", 0.0)),
                float(items["Yaw"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        return pose if all(math.isfinite(value) for value in pose) else None

    def _map_pose(self) -> tuple[float, float, float, float] | None:
        response = self._request(map_location_request())
        items = response_items(response)
        if not response or items.get("Location") != 0:
            LOGGER.error("D455 接近地图定位查询失败: response=%s", self._response_summary(response))
            return None
        pose = self._finite_pose(items)
        if pose is None:
            LOGGER.error("D455 接近地图定位数据无效: response=%s", self._response_summary(response))
        return pose

    def _perception_result(self) -> ApproachResult | None:
        response = self._request(perception_state_request())
        if not response:
            LOGGER.error("D455 接近感知状态查询无响应")
            return ApproachResult.COMM_ERROR
        items = response_items(response)
        if items.get("Location") != 0:
            LOGGER.error("D455 接近定位状态异常: response=%s", self._response_summary(response))
            return ApproachResult.LOCALIZATION_FAILED
        if items.get("ObsState") == 1:
            LOGGER.warning("D455 接近检测到避障状态: response=%s", self._response_summary(response))
            return ApproachResult.OBSTACLE_DETECTED
        return None

    @staticmethod
    def _response_summary(response: object) -> dict[str, object]:
        """Keep AOS diagnostics useful without logging entire protocol packets."""

        items = response_items(response)
        if not items:
            return {}
        keys = ("ErrorCode", "Status", "Value", "Location", "ObsState", "PosX", "PosY", "PosZ", "Yaw")
        return {key: items[key] for key in keys if key in items}

    def _submit_navigation(self, point_id: int, x: float, y: float, z: float, yaw: float) -> bool:
        self._completed_navigation_ids.discard(point_id)
        self._last_navigation_submit_failure = None
        point = {"point_id": point_id, "x": x, "y": y, "z": z, "yaw": yaw}
        LOGGER.info(
            "D455 动态导航提交: point_id=%s target={x=%.3f,y=%.3f,z=%.3f,yaw=%.3f} timeout_sec=%.1f",
            point_id, x, y, z, yaw, self.config.max_navigation_sec,
        )
        response = self._request(
            single_point_nav_request(
                point,
                point_id=point_id,
                point_info=1,
                gait=self.config.gait,
                speed=self.config.speed,
                manner=0,
                obs_mode=self.config.obs_mode,
                nav_mode=self.config.nav_mode,
            ),
            # Match the verified CLI: Type=1003 may return only once the
            # navigation task finishes, so it must be allowed the full D455
            # motion budget instead of the ordinary command ACK timeout.
            timeout=self.config.max_navigation_sec,
            allow_response_id_mismatch=True,
        )
        if not response:
            self._last_navigation_submit_failure = ApproachResult.TIMEOUT
            LOGGER.error(
                "D455 动态导航无终态响应: point_id=%s timeout_sec=%.1f",
                point_id, self.config.max_navigation_sec,
            )
            return False
        items = response_items(response)
        error_code = items.get("ErrorCode")
        LOGGER.info("D455 动态导航响应: point_id=%s response=%s", point_id, self._response_summary(response))
        if error_code == 0x2300:
            self._completed_navigation_ids.add(point_id)
            return True
        if error_code not in (None, 0):
            self._last_navigation_submit_failure = ApproachResult.MOTION_FAILED
            LOGGER.error("D455 动态导航被 M20 拒绝: point_id=%s response=%s", point_id, self._response_summary(response))
            return False
        return True

    def _wait_navigation(self, point_id: int) -> ApproachResult | None:
        if point_id in self._completed_navigation_ids:
            self._completed_navigation_ids.discard(point_id)
            LOGGER.info("D455 动态导航已在提交响应中完成: point_id=%s", point_id)
            return None
        deadline = self._monotonic() + self.config.max_navigation_sec
        last_status: tuple[object, object, object] | None = None
        while self._monotonic() < deadline:
            if self._stop_requested():
                raise KeyboardInterrupt
            remaining = max(0.1, deadline - self._monotonic())
            response = self._request(
                nav_status_request(),
                timeout=min(10.0, remaining),
                allow_response_id_mismatch=True,
            )
            if not response:
                LOGGER.warning("D455 动态导航状态查询无响应: point_id=%s", point_id)
                self._sleep(1.0)
                continue
            items = response_items(response)
            if items.get("Value") != point_id:
                LOGGER.info("D455 动态导航状态关联号不匹配: expected=%s response=%s", point_id, self._response_summary(response))
                self._sleep(1.0)
                continue
            status = items.get("Status")
            error_code = items.get("ErrorCode", 0)
            status_summary = (items.get("Value"), status, error_code)
            if status_summary != last_status:
                LOGGER.info("D455 动态导航状态: point_id=%s response=%s", point_id, self._response_summary(response))
                last_status = status_summary
            if status == 4 or error_code == 0x2300:
                LOGGER.info("D455 动态导航完成: point_id=%s", point_id)
                return None
            if error_code not in (None, 0):
                LOGGER.error("D455 动态导航执行失败: point_id=%s response=%s", point_id, self._response_summary(response))
                return ApproachResult.MOTION_FAILED
            self._sleep(1.0)
        LOGGER.error("D455 动态导航状态超时: point_id=%s timeout_sec=%.1f", point_id, self.config.max_navigation_sec)
        return ApproachResult.TIMEOUT

    def _cancel_and_confirm(self) -> bool:
        LOGGER.warning("D455 接近安全清理: 请求取消当前导航")
        response = self._request(cancel_nav_request())
        if not response or response_items(response).get("ErrorCode") != 0:
            LOGGER.error("D455 接近取消导航失败: response=%s", self._response_summary(response))
            return False
        LOGGER.info("D455 接近取消导航已确认: response=%s", self._response_summary(response))
        deadline = self._monotonic() + self.m20_config.cancel_confirm_timeout_sec
        while self._monotonic() < deadline:
            response = self._request(nav_status_request(), timeout=min(5.0, self.m20_config.command_timeout_sec))
            if response:
                items = response_items(response)
                LOGGER.info("D455 接近取消导航状态: response=%s", self._response_summary(response))
                if items.get("ErrorCode") in (0x2300, 0x2302):
                    return True
                if items.get("Status") == 0 and items.get("ErrorCode") in (None, 0):
                    return True
            self._sleep(0.5)
        LOGGER.error("D455 接近取消导航确认超时: timeout_sec=%.1f", self.m20_config.cancel_confirm_timeout_sec)
        return False

    def _motion(self, motion_param: int) -> bool:
        response = self._request(motion_state_request(motion_param))
        success = bool(response) and response_items(response).get("ErrorCode") in (None, 0)
        action = {1: "站立", 2: "软急停", 4: "趴下"}.get(motion_param, str(motion_param))
        level = LOGGER.info if success else LOGGER.error
        level("D455 接近姿态命令: action=%s success=%s response=%s", action, success, self._response_summary(response))
        return success

    def _fail_after_navigation(self, result: ApproachResult) -> ApproachResult:
        # A command may have reached the robot despite a lost acknowledgement.
        LOGGER.error("D455 接近失败: result=%s；开始取消导航并原地趴下", result.name)
        self._cancel_and_lie()
        return result

    def _cancel_and_lie(self) -> bool:
        """Cancel navigation where possible, then always request prone in place."""

        cancel_ok = self._cancel_and_confirm()
        if not cancel_ok:
            LOGGER.error("D455 接近取消导航未确认；仍发送原地趴下命令")
        prone_ok = self._motion(4)
        return cancel_ok and prone_ok

    def _move_forward_before_lie(self, final_yaw: float) -> tuple[float, float, float] | ApproachResult:
        """Move forward along the current body heading while correcting to ``final_yaw``."""

        if not math.isfinite(final_yaw):
            return ApproachResult.INVALID_TARGET

        pose = self._map_pose()
        if pose is None:
            return ApproachResult.LOCALIZATION_FAILED
        perception = self._perception_result()
        if perception is not None:
            return perception
        robot_x, robot_y, robot_z, robot_yaw = pose
        goal_x = robot_x + self.config.pre_lie_forward_m * math.cos(robot_yaw)
        goal_y = robot_y + self.config.pre_lie_forward_m * math.sin(robot_yaw)
        point_id = self.config.approach_point_id + 1
        if not self._submit_navigation(point_id, goal_x, goal_y, robot_z, final_yaw):
            return self._last_navigation_submit_failure or ApproachResult.MOTION_FAILED
        complete = self._wait_navigation(point_id)
        if complete is not None:
            return complete
        return goal_x, goal_y, final_yaw

    def approach(self, *, map_pose: tuple[float, float, float, float] | None = None) -> ApproachResult:
        """Approach cached D455 target and return prone success only after confirmation."""

        navigation_submitted = False
        try:
            if self._stop_requested():
                raise KeyboardInterrupt
            target = choose_nearest_detection(
                self._last_detections,
                max_range_m=self.config.approach_trigger_max_range_m,
            )
            if target is None:
                return ApproachResult.NO_TARGET
            if target.relative_xy_m is None:
                return ApproachResult.INVALID_TARGET
            target_x = float(target.relative_xy_m[0]) + self.config.base_link_offset_x_m
            target_y = float(target.relative_xy_m[1]) + self.config.base_link_offset_y_m
            if not all(math.isfinite(value) for value in (target_x, target_y)) or target_x <= 0:
                return ApproachResult.INVALID_TARGET
            target_range = math.hypot(target_x, target_y)
            if target_range <= self.config.safe_stop_distance_m:
                return ApproachResult.INVALID_TARGET

            pose = map_pose if map_pose is not None else self._map_pose()
            if pose is None or len(pose) != 4 or not all(math.isfinite(float(value)) for value in pose):
                return ApproachResult.LOCALIZATION_FAILED
            perception = self._perception_result()
            if perception is not None:
                return perception
            robot_x, robot_y, robot_z, robot_yaw = pose

            LOGGER.info(
                "D455 接近目标: relative_xy={x=%.3f,y=%.3f} range=%.3f robot_pose={x=%.3f,y=%.3f,z=%.3f,yaw=%.3f}",
                target_x, target_y, target_range, robot_x, robot_y, robot_z, robot_yaw,
            )

            scale = (target_range - self.config.safe_stop_distance_m) / target_range
            approach_x = target_x * scale
            approach_y = target_y * scale
            goal_x = robot_x + math.cos(robot_yaw) * approach_x - math.sin(robot_yaw) * approach_y
            goal_y = robot_y + math.sin(robot_yaw) * approach_x + math.cos(robot_yaw) * approach_y
            target_bearing = math.atan2(target_y, target_x)
            side = math.pi / 2.0 if self.config.perpendicular_side == "left" else -math.pi / 2.0
            goal_yaw = math.atan2(
                math.sin(robot_yaw + target_bearing + side),
                math.cos(robot_yaw + target_bearing + side),
            )

            navigation_submitted = True
            if not self._submit_navigation(self.config.approach_point_id, goal_x, goal_y, robot_z, goal_yaw):
                return self._fail_after_navigation(self._last_navigation_submit_failure or ApproachResult.MOTION_FAILED)
            complete = self._wait_navigation(self.config.approach_point_id)
            if complete is not None:
                return self._fail_after_navigation(complete)

            final_goal = (goal_x, goal_y, goal_yaw)
            if self.config.pre_lie_forward_m > 0:
                moved = self._move_forward_before_lie(goal_yaw)
                if isinstance(moved, ApproachResult):
                    return self._fail_after_navigation(moved)
                final_goal = moved

            final_pose = self._map_pose()
            if final_pose is None:
                return self._fail_after_navigation(ApproachResult.LOCALIZATION_FAILED)
            position_error = math.hypot(final_pose[0] - final_goal[0], final_pose[1] - final_goal[1])
            yaw_error = math.atan2(
                math.sin(final_pose[3] - final_goal[2]),
                math.cos(final_pose[3] - final_goal[2]),
            )
            LOGGER.info(
                "D455 接近末端位姿诊断: target={x=%.3f,y=%.3f,yaw=%.3f} actual={x=%.3f,y=%.3f,yaw=%.3f} position_error=%.3f yaw_error=%.3f",
                final_goal[0], final_goal[1], final_goal[2], final_pose[0], final_pose[1], final_pose[3],
                position_error, yaw_error,
            )
            perception = self._perception_result()
            if perception is not None:
                return self._fail_after_navigation(perception)
            if not self._motion(4):
                return self._fail_after_navigation(ApproachResult.MOTION_FAILED)
            LOGGER.info("D455 接近成功：901/902、定位、避障复核和趴下命令均已确认")
            return ApproachResult.PRONE_AT_TARGET
        except KeyboardInterrupt:
            self._cancel_and_lie()
            return ApproachResult.MOTION_FAILED
        except (ConnectionError, OSError, TimeoutError, TypeError, ValueError):
            LOGGER.exception("D455 接近通信或数据异常")
            return self._fail_after_navigation(ApproachResult.COMM_ERROR) if navigation_submitted else ApproachResult.COMM_ERROR


__all__ = ["D455ApproachController"]
