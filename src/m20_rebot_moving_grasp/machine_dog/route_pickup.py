"""巡航中 D455 检测、机械臂抓取和路线恢复控制器。"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .d455_approach import D455ApproachController
from .d455_vision import D455Detection, choose_nearest_detection
from .interface import ApproachResult
from .m20_config import M20Config
from .m20_protocol import (
    UdpJsonTransport,
    cancel_nav_request,
    map_location_request,
    motion_state_request,
    nav_status_request,
    response_items,
    single_point_nav_request,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
POINT_INFO_VALUES = {0, 1, 3}
ROUTE_STATUS_POLL_SEC = 0.25


@dataclass(frozen=True)
class MapPose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class RouteDetection:
    detection: D455Detection
    pose: MapPose


class PickupOutcome(str, Enum):
    COMPLETED = "completed"
    NO_FEASIBLE_GRASP_RECOVERED = "no_feasible_grasp_recovered"
    FAILED = "failed"


class RouteDetectionState:
    """Accept only the first valid detection and pair it with a map pose."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._value: RouteDetection | None = None

    def accept(self, detections: Any, pose: MapPose | None) -> bool:
        if pose is None:
            return False
        selected = choose_nearest_detection(detections or ())
        if selected is None:
            return False
        with self._lock:
            if self._value is not None:
                return False
            self._value = RouteDetection(selected, pose)
            self.event.set()
            return True

    def take(self) -> RouteDetection | None:
        with self._lock:
            value = self._value
            self._value = None
            self.event.clear()
            return value


class RoutePosePublisher:
    """Poll and persist the latest valid M20 map pose."""

    def __init__(self, transport: UdpJsonTransport, *, timeout: float, interval: float, log_path: Path) -> None:
        self.transport = transport
        self.timeout = float(timeout)
        self.interval = float(interval)
        self.log_path = log_path
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: MapPose | None = None
        self._thread = threading.Thread(target=self._run, name="m20-route-pose", daemon=True)

    @staticmethod
    def _parse(response: dict[str, Any] | None) -> MapPose | None:
        items = response_items(response)
        if not response or items.get("Location") != 0:
            return None
        try:
            pose = MapPose(float(items["PosX"]), float(items["PosY"]),
                           float(items.get("PosZ", 0.0)), float(items["Yaw"]))
        except (KeyError, TypeError, ValueError):
            return None
        return pose if all(math.isfinite(v) for v in (pose.x, pose.y, pose.z, pose.yaw)) else None

    def read(self) -> MapPose | None:
        return self._parse(self.transport.send_request(map_location_request(), timeout=self.timeout))

    def _record(self, pose: MapPose) -> None:
        with self._lock:
            self._latest = pose
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            json.dump({"time": time.time(), **pose.__dict__}, stream, ensure_ascii=False)
            stream.write("\n")

    def snapshot(self, *, force: bool = False) -> MapPose | None:
        if not force:
            with self._lock:
                return self._latest
        pose = self.read()
        if pose is not None:
            self._record(pose)
        return pose

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        # Let an in-flight location query finish before a D455 approach starts
        # issuing its own map/navigation requests on the same AOS endpoint.
        self._thread.join(timeout=max(2.0, self.interval + 1.0, self.timeout + 1.0))

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                pose = self.read()
                if pose is not None:
                    self._record(pose)
            except (OSError, RuntimeError, TimeoutError):
                LOGGER.warning("巡航定位查询失败", exc_info=True)
            self.stop_event.wait(self.interval)


def point_info_from_saved_point(point: dict[str, Any]) -> int | None:
    value = point.get("point_info")
    if value is None:
        point_type = point.get("point_type", "task")
        value = {"transition": 0, "task": 1, "charge": 3}.get(point_type)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value in POINT_INFO_VALUES else None


def load_route_definition(
    routes_path: str | Path,
    points_path: str | Path,
    route_name: str,
    *,
    required_point_names: tuple[str, ...] = (),
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load and validate a project route and its referenced points."""

    routes_file = Path(routes_path).expanduser()
    points_file = Path(points_path).expanduser()
    if not routes_file.is_absolute():
        routes_file = PROJECT_ROOT / routes_file
    if not points_file.is_absolute():
        points_file = PROJECT_ROOT / points_file
    try:
        routes = json.loads(routes_file.read_text(encoding="utf-8"))
        points = json.loads(points_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"路线或点位文件不可读: {exc}") from exc
    route = routes.get(route_name) if isinstance(routes, dict) else None
    if not isinstance(route, list) or not route:
        raise ValueError(f"未找到有效路线: {route_name}")
    if not isinstance(points, dict):
        raise ValueError("点位文件顶层必须是 mapping")
    selected: dict[str, dict[str, Any]] = {}

    def normalize(name: str) -> None:
        if not isinstance(name, str) or not isinstance(points.get(name), dict):
            raise ValueError(f"路线引用了不存在的点位: {name}")
        point = points[name]
        try:
            normalized = {key: float(point[key]) for key in ("x", "y", "z", "yaw")}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"路线点 {name} 坐标无效") from exc
        if not all(math.isfinite(v) for v in normalized.values()):
            raise ValueError(f"路线点 {name} 包含非法数值")
        info = point_info_from_saved_point(point)
        # Route legs are coordinate navigation (Value=0), so ``point_id`` is
        # optional metadata and never resolves a point stored in the map.
        # PointInfo remains part of the verified AOS payload and defaults to
        # task-point semantics when the source point omits it (for example
        # the map origin named ``start``).
        if info is None and "point_info" in point:
            raise ValueError(f"路线点 {name} 的 PointInfo 无效")
        normalized["point_info"] = 1 if info is None else info
        if "point_id" in point:
            try:
                point_id = int(point["point_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"路线点 {name} 的 point_id 无效") from exc
            if point_id < 0:
                raise ValueError(f"路线点 {name} 的 point_id 无效")
            normalized["point_id"] = point_id
        selected[name] = normalized

    for name in route:
        normalize(name)
    for name in required_point_names:
        if not isinstance(name, str) or not name:
            raise ValueError("必需点位名称无效")
        if name not in selected:
            normalize(name)
    return list(route), selected


class RouteStopRequested(Exception):
    """Cooperative stop at a route/action boundary."""


class RoutePickupController:
    """Run the companion's route interruption flow with the existing arm path."""

    def __init__(self, config: dict[str, Any], m20: M20Config, transport: UdpJsonTransport,
                 d455: D455ApproachController, *, routes_path: str | Path,
                 points_path: str | Path, route_name: str,
                 stop_requested: Callable[[], bool] | None = None) -> None:
        self.stop_requested = stop_requested or (lambda: False)
        self.safe_stop_completed = False
        self.config = config
        self.m20 = m20
        self.transport = transport
        self.d455 = d455
        self.routes_path = routes_path
        self.points_path = points_path
        self.route_name = route_name
        self.route, self.points = load_route_definition(
            routes_path,
            points_path,
            route_name,
            required_point_names=(m20.bin_point_name, m20.home_point_name),
        )
        self.detection_state = RouteDetectionState()
        self.scanner_stop = threading.Event()
        self.scanner_ready = threading.Event()
        self.scanner_error: Exception | None = None
        self.scanner_thread: threading.Thread | None = None
        self.publisher: RoutePosePublisher | None = None
        self.last_failure_reason = ""
        self._suppressed_detection_pose: MapPose | None = None
        self._suppression_logged = False

    def _check_stop(self) -> None:
        if self.stop_requested():
            raise RouteStopRequested("收到巡检停止请求")

    def _settle(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            self._check_stop()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def _fail(self, reason: str) -> bool:
        self.last_failure_reason = reason
        LOGGER.error("巡航任务失败: %s", reason)
        return False

    def _d435_miss_suppression_radius_m(self) -> float:
        try:
            radius = float(self.config.get("d435_miss_suppression_radius_m", 0.75))
        except (TypeError, ValueError) as exc:
            raise ValueError("D435 漏检目标抑制距离无效") from exc
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("D435 漏检目标抑制距离必须大于0")
        return radius

    def _accept_route_detection(self, detections: Any, pose: MapPose | None) -> bool:
        """Ignore the recently missed target until the robot has left its area."""

        if pose is None:
            return False
        suppressed = self._suppressed_detection_pose
        if suppressed is not None:
            distance = math.hypot(pose.x - suppressed.x, pose.y - suppressed.y)
            radius = self._d435_miss_suppression_radius_m()
            if distance <= radius:
                if not self._suppression_logged:
                    LOGGER.info("D455 目标触发暂时抑制：距 D435 漏检位置 %.2fm，阈值 %.2fm", distance, radius)
                    self._suppression_logged = True
                return False
            LOGGER.info("D455 漏检目标抑制解除：已离开 %.2fm", radius)
            self._suppressed_detection_pose = None
            self._suppression_logged = False
        return self.detection_state.accept(detections, pose)

    def _request(self, payload: str, *, timeout: float | None = None, mismatch: bool = False):
        return self.transport.send_request(payload, timeout=timeout or self.m20.command_timeout_sec,
                                           allow_response_id_mismatch=mismatch)

    def _motion(self, value: int) -> bool:
        if value != 4:  # Prone must remain available during stop cleanup.
            self._check_stop()
        response = self._request(motion_state_request(value))
        return response_items(response).get("ErrorCode") in (None, 0) and bool(response)

    def _cancel_navigation(self) -> bool:
        response = self._request(cancel_nav_request(), mismatch=True)
        if response_items(response).get("ErrorCode") not in (None, 0):
            return False
        deadline = time.monotonic() + self.m20.cancel_confirm_timeout_sec
        while time.monotonic() < deadline:
            items = response_items(self._request(nav_status_request(), mismatch=True))
            if items.get("ErrorCode") in (0x2300, 0x2302) or (
                items.get("Status") == 0 and items.get("ErrorCode") in (None, 0)
            ):
                return True
            time.sleep(0.5)
        return False

    def _cancel_and_lie(self, reason: str) -> bool:
        """Terminate route motion by cancelling navigation and requesting prone."""

        LOGGER.warning("巡航任务中止: %s；请求取消导航并原地趴下", reason)
        try:
            cancel_ok = self._cancel_navigation()
        except Exception:
            LOGGER.exception("取消导航通信异常；仍尝试原地趴下")
            cancel_ok = False
        if not cancel_ok:
            LOGGER.error("巡航取消导航未确认: %s；仍发送原地趴下命令", reason)
        prone_ok = self._motion(4)
        if not prone_ok:
            LOGGER.error("巡航原地趴下命令未获确认: %s", reason)
        return cancel_ok and prone_ok

    def _navigate(self, point: dict[str, Any], *, point_id: int | None, point_info: int, label: str,
                  interruptible: bool = False) -> bool:
        """Run a route navigation request while allowing D455 to interrupt it.

        Some M20 firmware revisions keep the Type=1003 UDP request open until
        the point is reached.  Run that request in a worker so a D455 event can
        still cancel the movement immediately instead of waiting for its full
        300-second navigation budget.
        """
        self._check_stop()
        result: dict[str, Any] = {}
        submitted = threading.Event()

        def submit() -> None:
            try:
                result["response"] = self._request(
                    single_point_nav_request(point, point_id=point_id, point_info=point_info,
                                             gait=0x3002, speed=1, manner=0, obs_mode=0, nav_mode=1),
                    timeout=float(self.config["route_nav_timeout_sec"]),
                    mismatch=True,
                )
            except Exception as exc:  # Converted to a controlled route failure below.
                result["error"] = exc
            finally:
                submitted.set()

        navigation_name = point_id if point_id is not None else "coordinates"
        worker = threading.Thread(target=submit, name=f"m20-route-nav-{navigation_name}", daemon=True)
        worker.start()
        deadline = time.monotonic() + float(self.config["route_nav_timeout_sec"])
        while not submitted.wait(ROUTE_STATUS_POLL_SEC):
            self._check_stop()
            if self.scanner_error is not None:
                self._cancel_navigation()
                raise RuntimeError(f"D455 巡航扫描失败: {self.scanner_error}")
            if interruptible and self.detection_state.event.is_set():
                if not self._cancel_navigation():
                    LOGGER.error("检测触发后取消巡航导航未确认；发送原地趴下命令")
                    self._motion(4)
                    raise RuntimeError("检测触发后取消巡航导航失败")
                return False
            if time.monotonic() >= deadline:
                self._cancel_navigation()
                return False

        self._check_stop()
        if "error" in result:
            LOGGER.error("巡航导航通信异常: %s", result["error"])
            self._cancel_navigation()
            return False
        response = result.get("response")
        items = response_items(response)
        if items.get("ErrorCode") not in (None, 0, 0x2300):
            LOGGER.error("巡航导航提交失败: %s items=%s", label, items)
            self._cancel_navigation()
            return False
        if items.get("ErrorCode") == 0x2300:
            return True
        deadline = time.monotonic() + float(self.config["route_nav_timeout_sec"])
        while time.monotonic() < deadline:
            self._check_stop()
            if self.scanner_error is not None:
                self._cancel_navigation()
                raise RuntimeError(f"D455 巡航扫描失败: {self.scanner_error}")
            if interruptible and self.detection_state.event.is_set():
                if not self._cancel_navigation():
                    LOGGER.error("检测触发后取消巡航导航未确认；发送原地趴下命令")
                    self._motion(4)
                    raise RuntimeError("检测触发后取消巡航导航失败")
                return False
            status = response_items(self._request(nav_status_request(), timeout=ROUTE_STATUS_POLL_SEC, mismatch=True))
            # Coordinate navigation reports Value=0 or a firmware-generated
            # task number.  Only temporary 902 recovery keeps an explicit
            # correlation id, and even then terminal status is authoritative.
            if point_id is not None and status.get("Value") != point_id:
                time.sleep(ROUTE_STATUS_POLL_SEC)
                continue
            if status.get("Status") == 4 or status.get("ErrorCode") == 0x2300:
                return True
            if status.get("ErrorCode") not in (None, 0):
                self._cancel_navigation()
                return False
            time.sleep(ROUTE_STATUS_POLL_SEC)
        self._cancel_navigation()
        return False

    def _scanner_loop(self) -> None:
        try:
            self.d455.scan_start()
            self.scanner_ready.set()
            while not self.scanner_stop.is_set():
                detections = self.d455.scan_handler()
                if self.publisher is not None and self._accept_route_detection(detections, self.publisher.snapshot()):
                    return
        except Exception as exc:
            self.scanner_error = exc
            self.scanner_ready.set()

    def _start_scanner(self) -> None:
        self.scanner_stop = threading.Event()
        self.scanner_ready = threading.Event()
        self.scanner_error = None
        self.scanner_thread = threading.Thread(target=self._scanner_loop, name="m20-route-d455", daemon=True)
        self.scanner_thread.start()
        deadline = time.monotonic() + 60.0
        while not self.scanner_ready.wait(0.1):
            self._check_stop()
            if time.monotonic() >= deadline:
                raise RuntimeError("D455 巡航扫描启动超时")
        self._check_stop()
        if self.scanner_error:
            raise RuntimeError(f"D455 巡航扫描失败: {self.scanner_error}")

    def _stop_scanner(self, *, preserve: bool = False) -> None:
        self.scanner_stop.set()
        if self.scanner_thread is not None:
            self.scanner_thread.join(timeout=8.0)
            self.scanner_thread = None
        self.d455.scan_stop(preserve_detections=preserve)
        if self.scanner_error:
            raise RuntimeError(f"D455 巡航扫描失败: {self.scanner_error}")

    def _save_resume(
        self,
        path: Path,
        loop: int,
        index: int,
        point_name: str,
        detection: RouteDetection,
        exit_pose: MapPose,
        *,
        status: str = "pickup_pending",
        failure_reason: str = "",
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"route_name": self.route_name, "loop": loop, "route_index": index,
                   "point_name": point_name,
                   "relative_target": {"x": detection.detection.relative_xy_m[0], "y": detection.detection.relative_xy_m[1]},
                   "detection_pose": detection.pose.__dict__, "exit_pose": exit_pose.__dict__,
                   "status": status}
        if failure_reason:
            payload["failure_reason"] = failure_reason
        fd, temporary = tempfile.mkstemp(prefix="route-resume-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            Path(temporary).replace(path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise

    @staticmethod
    def _clear_resume(path: Path) -> None:
        path.unlink(missing_ok=True)

    def _append_d435_no_feasible_grasp(
        self,
        path: Path,
        detection: RouteDetection,
        exit_pose: MapPose,
        reason: str,
    ) -> None:
        """Persist a D435 recovery that found no feasible grasp pose."""

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": time.time(),
            "route_name": self.route_name,
            "loop": int(self.config["current_loop"]),
            "route_index": int(self.config["current_index"]),
            "point_name": str(self.config["current_point"]),
            "relative_target": {
                "x": detection.detection.relative_xy_m[0],
                "y": detection.detection.relative_xy_m[1],
            },
            "detection_pose": detection.pose.__dict__,
            "exit_pose": exit_pose.__dict__,
            "status": "d435_no_feasible_grasp_recovered",
            "failure_reason": reason,
        }
        with path.open("a", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
            stream.write("\n")
            stream.flush()

    def _recover_from_d435_no_feasible_grasp(
        self,
        arm: Any,
        detection: RouteDetection,
        exit_pose: MapPose,
        resume_path: Path,
        reason: str,
    ) -> PickupOutcome:
        """Safely return to the interrupted route after all D435 views fail."""

        self._save_resume(
            resume_path,
            int(self.config["current_loop"]),
            int(self.config["current_index"]),
            str(self.config["current_point"]),
            detection,
            exit_pose,
            status="d435_no_feasible_grasp_recovery_pending",
            failure_reason=reason,
        )
        if not arm.safe_home():
            getattr(arm, "release_vision", lambda: None)()
            self._fail("D435 无可行抓取位姿后机械臂无法安全回位")
            return PickupOutcome.FAILED
        getattr(arm, "release_vision", lambda: None)()
        if not self._motion(1):
            self._fail("D435 无可行抓取位姿后机器狗站立失败")
            return PickupOutcome.FAILED
        resume = {"x": exit_pose.x, "y": exit_pose.y, "z": exit_pose.z, "yaw": exit_pose.yaw}
        if not self._navigate(resume, point_id=int(self.config["resume_point_id"]), point_info=0,
                              label="D435 无可行抓取位姿恢复中断位姿"):
            self._fail("D435 无可行抓取位姿后返回中断位姿失败或超时")
            return PickupOutcome.FAILED
        missed_path = Path(self.config.get("missed_detection_log", resume_path.with_name("d435_misses.jsonl"))).expanduser()
        if not missed_path.is_absolute():
            missed_path = PROJECT_ROOT / missed_path
        try:
            self._append_d435_no_feasible_grasp(missed_path, detection, exit_pose, reason)
        except OSError as exc:
            self._fail(f"D435 无可行抓取位姿记录写入失败: {exc}")
            return PickupOutcome.FAILED
        self._suppressed_detection_pose = detection.pose
        self._suppression_logged = False
        self._clear_resume(resume_path)
        LOGGER.warning("D435 六视角补偿耗尽，已记录无可行抓取位姿并恢复巡航: %s", reason)
        return PickupOutcome.NO_FEASIBLE_GRASP_RECOVERED

    def _handle_detection(
        self,
        arm: Any,
        bin_point: dict[str, Any],
        exit_pose: MapPose,
        resume_path: Path,
    ) -> PickupOutcome:
        detection = self.detection_state.take()
        if detection is None or detection.detection.relative_xy_m is None:
            self._fail("未获得有效的 D455 检测缓存")
            return PickupOutcome.FAILED
        self._save_resume(resume_path, int(self.config["current_loop"]), int(self.config["current_index"]),
                          str(self.config["current_point"]), detection, exit_pose)
        self._check_stop()
        approach_result = self.d455.approach(map_pose=(detection.pose.x, detection.pose.y, detection.pose.z, detection.pose.yaw))
        if approach_result is not ApproachResult.PRONE_AT_TARGET:
            self._fail(f"D455 接近未成功: {approach_result.name}")
            return PickupOutcome.FAILED
        self._settle(float(self.config["pickup_settle_seconds"]))
        if not arm.prepare_grasp():
            reason = getattr(arm, "last_failure_reason", "") or "未知原因"
            if bool(getattr(arm, "is_recoverable_detection_failure", False)):
                return self._recover_from_d435_no_feasible_grasp(arm, detection, exit_pose, resume_path, str(reason))
            arm.safe_home()
            getattr(arm, "release_vision", lambda: None)()
            self._fail(f"D435 抓取准备失败: {reason}")
            return PickupOutcome.FAILED
        self._check_stop()
        if not arm.grasp_current_target():
            arm.safe_home()
            getattr(arm, "release_vision", lambda: None)()
            self._fail(f"机械臂抓取动作失败: {getattr(arm, 'last_failure_reason', '') or '未知原因'}")
            return PickupOutcome.FAILED
        self._check_stop()
        if not arm.hold():
            self._fail("机械臂未能进入运输保持姿态")
            return PickupOutcome.FAILED
        if not self._motion(1):
            self._fail("机械臂抓取后机器狗站立失败")
            return PickupOutcome.FAILED
        if not self._navigate(bin_point, point_id=None,
                              point_info=point_info_from_saved_point(bin_point) or 1, label="bin_point"):
            self._fail("导航到 bin_point 失败或超时")
            return PickupOutcome.FAILED
        if not self._motion(4):
            self._fail("到达 bin_point 后机器狗趴下失败")
            return PickupOutcome.FAILED
        self._settle(float(self.config["bin_settle_seconds"]))
        self._check_stop()
        if not arm.drop():
            self._fail("机械臂投放动作失败")
            return PickupOutcome.FAILED
        if not arm.safe_home():
            self._fail("投放后机械臂安全回位失败")
            return PickupOutcome.FAILED
        getattr(arm, "release_vision", lambda: None)()
        if not self._motion(1):
            self._fail("投放后机器狗站立失败")
            return PickupOutcome.FAILED
        resume = {"x": exit_pose.x, "y": exit_pose.y, "z": exit_pose.z, "yaw": exit_pose.yaw}
        if not self._navigate(resume, point_id=int(self.config["resume_point_id"]), point_info=0, label="恢复中断位姿"):
            self._fail("返回巡航中断位姿失败或超时")
            return PickupOutcome.FAILED
        self._clear_resume(resume_path)
        return PickupOutcome.COMPLETED

    def _start_publisher(self) -> None:
        self.publisher = RoutePosePublisher(
            self.transport,
            timeout=self.m20.command_timeout_sec,
            interval=float(self.config["pose_interval_sec"]),
            log_path=Path(self.config["pose_log"]),
        )
        if self.publisher.snapshot(force=True) is None:
            self.publisher = None
            raise RuntimeError("M20 巡航启动前未取得有效地图定位")
        self.publisher.start()

    def _stop_publisher(self) -> None:
        if self.publisher is not None:
            self.publisher.stop()
            self.publisher = None

    def _consume_detection(self, arm: Any, bin_point: dict[str, Any], resume_path: Path) -> PickupOutcome:
        """Stop parallel readers, pick up once, then restore them on success."""
        self._stop_scanner(preserve=True)
        if self.publisher is None:
            self._fail("巡航定位发布器未启动")
            return PickupOutcome.FAILED
        exit_pose = self.publisher.snapshot(force=True)
        self._stop_publisher()
        if exit_pose is None:
            self._fail("检测后未取得有效中断地图位姿")
            return PickupOutcome.FAILED
        outcome = self._handle_detection(arm, bin_point, exit_pose, resume_path)
        if outcome is PickupOutcome.FAILED:
            return outcome
        self._check_stop()
        self._start_publisher()
        self._start_scanner()
        return outcome

    def run(self, dog: Any, arm: Any) -> bool:
        resume_path = Path(self.config["resume_log"]).expanduser()
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        points = self.points
        bin_name = self.m20.bin_point_name
        if bin_name not in points:
            raise ValueError(f"路线模式缺少 bin 点位: {bin_name}")
        bin_point = points[bin_name]
        completed = False
        try:
            self._check_stop()
            if not arm.init_reset():
                return self._fail("机械臂初始化失败")
            self._check_stop()
            if not dog.init_reset():
                return self._fail("机器狗初始化失败")
            self._check_stop()
            if not dog.stand():
                return self._fail("机器狗站立失败")
            self._start_publisher()
            self._start_scanner()
            loops = int(self.config["route_loops"])
            for loop in range(1, loops + 1):
                index = 0
                while index < len(self.route):
                    self._check_stop()
                    name = self.route[index]
                    self.config.update({"current_loop": loop, "current_index": index, "current_point": name})
                    if self.detection_state.event.is_set():
                        detected_index = int(self.config["current_index"])
                        outcome = self._consume_detection(arm, bin_point, resume_path)
                        if outcome is PickupOutcome.FAILED:
                            return False
                        if outcome is PickupOutcome.NO_FEASIBLE_GRASP_RECOVERED:
                            index = max(index, detected_index + 1)
                        continue
                    navigated = self._navigate(self.points[name], point_id=None,
                                               point_info=int(self.points[name]["point_info"]), label=name,
                                               interruptible=True)
                    if navigated:
                        index += 1
                    if self.detection_state.event.is_set():
                        detected_index = int(self.config["current_index"])
                        outcome = self._consume_detection(arm, bin_point, resume_path)
                        if outcome is PickupOutcome.FAILED:
                            return False
                        if outcome is PickupOutcome.NO_FEASIBLE_GRASP_RECOVERED:
                            index = max(index, detected_index + 1)
                        continue
                    if not navigated:
                        return self._fail(f"巡航点 {name} 导航失败或超时")
            self._stop_scanner()
            self._stop_publisher()
            self._check_stop()
            # Use the interruptible route navigator also for the final return.
            home = self.points[self.m20.home_point_name]
            completed = self._navigate(home, point_id=None, point_info=int(home["point_info"]),
                                       label="Home") and self._motion(4)
            self._check_stop()
            if not completed:
                self._fail("巡航完成后返回 Home 失败")
            return completed
        except RouteStopRequested:
            LOGGER.warning("巡航停止请求已处理，开始取消导航并关闭设备")
            return False
        finally:
            cleanup_ok = True
            # Cancel motion before waiting for camera/publisher threads to exit.
            if not completed or self.stop_requested():
                try:
                    cleanup_ok = bool(self._cancel_and_lie("巡航停止或异常退出"))
                except Exception:
                    cleanup_ok = False
                    LOGGER.exception("巡航退出后的机器狗趴下处理异常")
            try:
                if self.scanner_thread is not None:
                    self._stop_scanner()
            except Exception:
                cleanup_ok = False
                LOGGER.exception("停止巡航 D455 失败")
            try:
                self._stop_publisher()
            except Exception:
                cleanup_ok = False
                LOGGER.exception("停止巡航定位发布器失败")
            try:
                arm.shutdown()
            except Exception:
                cleanup_ok = False
                LOGGER.exception("巡航模式机械臂关闭失败")
            self.safe_stop_completed = self.stop_requested() and cleanup_ok
            if self.stop_requested() and not cleanup_ok:
                self.last_failure_reason = "停止清理失败：取消导航、趴下或设备关闭未确认"


__all__ = ["MapPose", "PickupOutcome", "RouteDetectionState", "RoutePosePublisher", "RoutePickupController", "load_route_definition", "point_info_from_saved_point"]
