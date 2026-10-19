"""DM-B601 的显式姿态运动函数。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .arm_runtime import ArmRuntime


GRIPPER_OPEN_POSITION_TOLERANCE_RAD = 0.12
D435_SCAN_JOINT1_INDEX = 0
D435_SCAN_JOINT5_INDEX = 4
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class D435RecoveryView:
    """A D435 compensation view relative to the frozen scan-pose centre."""

    joint1_offset_rad: float
    joint5_offset_rad: float


def d435_recovery_views(poses_config: dict[str, Any]) -> tuple[D435RecoveryView, ...]:
    """Read the six absolute-offset D435 compensation views from configuration."""

    if not isinstance(poses_config, dict):
        raise TypeError("poses 配置必须是 dict")
    recovery = poses_config.get("d435_recovery_scan")
    if not isinstance(recovery, dict):
        raise ValueError("poses 配置缺少 d435_recovery_scan")
    raw_views = recovery.get("views")
    if not isinstance(raw_views, list) or len(raw_views) != 6:
        raise ValueError("d435_recovery_scan.views 必须包含6个视角")
    views: list[D435RecoveryView] = []
    for index, raw_view in enumerate(raw_views):
        if not isinstance(raw_view, dict):
            raise ValueError(f"d435_recovery_scan.views[{index}] 必须是 mapping")
        try:
            view = D435RecoveryView(
                float(raw_view["joint1_offset_rad"]),
                float(raw_view["joint5_offset_rad"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"d435_recovery_scan.views[{index}] 偏移无效") from exc
        if not all(isfinite(value) for value in (view.joint1_offset_rad, view.joint5_offset_rad)):
            raise ValueError(f"d435_recovery_scan.views[{index}] 偏移必须是有限数字")
        views.append(view)
    if len({(view.joint1_offset_rad, view.joint5_offset_rad) for view in views}) != 6:
        raise ValueError("d435_recovery_scan.views 不能包含重复视角")
    return tuple(views)


def _d435_recovery_parameters(poses_config: dict[str, Any]) -> tuple[float, float, float, float]:
    recovery = poses_config.get("d435_recovery_scan")
    if not isinstance(recovery, dict):
        raise ValueError("poses 配置缺少 d435_recovery_scan")
    try:
        duration, speed_limit, settle_sec, margin = (
            float(recovery["duration_sec"]),
            float(recovery["joint_speed_limit_rad_s"]),
            float(recovery["settle_sec"]),
            float(recovery["joint_limit_margin_rad"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("d435_recovery_scan 运动参数无效") from exc
    if not all(isfinite(value) for value in (duration, speed_limit, settle_sec, margin)):
        raise ValueError("d435_recovery_scan 运动参数必须是有限数字")
    if duration <= 0.0 or speed_limit <= 0.0 or settle_sec < 0.0 or margin <= 0.0:
        raise ValueError("d435_recovery_scan 运动参数范围无效")
    return duration, speed_limit, settle_sec, margin


def _d435_recovery_feedback_parameters(poses_config: dict[str, Any]) -> tuple[float, float, int]:
    """Read the required physical-arrival check settings for recovery views."""

    recovery = poses_config.get("d435_recovery_scan")
    if not isinstance(recovery, dict):
        raise ValueError("poses 配置缺少 d435_recovery_scan")
    try:
        tolerance = float(recovery["joint_feedback_tolerance_rad"])
        timeout = float(recovery["feedback_timeout_sec"])
        retries_raw = float(recovery["feedback_retry_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("d435_recovery_scan 关节反馈确认参数无效") from exc
    retries = int(retries_raw)
    if (
        not all(isfinite(value) for value in (tolerance, timeout, retries_raw))
        or tolerance <= 0.0
        or timeout <= 0.0
        or retries_raw != retries
        or retries < 0
        or retries > 1
    ):
        raise ValueError("d435_recovery_scan 关节反馈确认参数范围无效")
    return tolerance, timeout, retries


def d435_recovery_settle_sec(poses_config: dict[str, Any]) -> float:
    """Return the camera settling time after a D435 recovery joint move."""

    return _d435_recovery_parameters(poses_config)[2]


def _read_recovery_joint_feedback(runtime: ArmRuntime, joint_count: int) -> np.ndarray:
    get_state = getattr(runtime.rebotarm, "get_state", None)
    if not callable(get_state):
        raise RuntimeError("机械臂 SDK 不支持读取关节反馈")
    state = get_state(request_feedback=True)
    try:
        values = np.asarray(state[0], dtype=np.float64)[:joint_count]
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("D435 补偿扫描关节反馈格式无效") from exc
    if values.shape != (joint_count,) or not np.all(np.isfinite(values)):
        raise RuntimeError("D435 补偿扫描关节反馈无效")
    return values


def _wait_for_recovery_feedback(
    runtime: ArmRuntime,
    target: np.ndarray,
    *,
    timeout_sec: float,
    tolerance_rad: float,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Wait for real J1/J5 feedback without accepting a command-only result."""

    deadline = time.monotonic() + timeout_sec
    actual: np.ndarray | None = None
    error: np.ndarray | None = None
    while True:
        try:
            actual = _read_recovery_joint_feedback(runtime, target.size)
            error = np.abs(actual[[D435_SCAN_JOINT1_INDEX, D435_SCAN_JOINT5_INDEX]] - target[
                [D435_SCAN_JOINT1_INDEX, D435_SCAN_JOINT5_INDEX]
            ])
            if float(np.max(error)) <= tolerance_rad:
                return True, actual, error
        except RuntimeError:
            actual = None
            error = None
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False, actual, error
        time.sleep(min(0.05, remaining))


def _send_recovery_target(
    runtime: ArmRuntime,
    target: np.ndarray,
    *,
    min_duration: float,
    speed_limit: float,
) -> bool:
    """Send one minimum-jerk command toward an absolute recovery target."""

    controller = runtime.controller
    joint_count = int(getattr(controller, "_n", 0))
    start_reader = getattr(controller, "_arm_motion_start", None)
    if not callable(start_reader):
        raise RuntimeError("机械臂控制器不支持读取补偿扫描姿势起点")
    q_start = np.asarray(start_reader(), dtype=np.float64)[:joint_count].copy()
    if q_start.shape != (joint_count,) or not np.all(np.isfinite(q_start)):
        raise RuntimeError("D435 补偿扫描起始关节反馈无效")

    sender = getattr(controller, "_send_thread", None)
    stop_sender = getattr(controller, "_stop_send", None)
    if sender is not None and sender.is_alive():
        if stop_sender is not None:
            stop_sender.set()
        sender.join(timeout=max(2.0, min_duration + 2.0))
        if sender.is_alive():
            return False
        if stop_sender is not None:
            stop_sender.clear()

    max_delta = float(np.max(np.abs(target - q_start)))
    duration = max(min_duration, 1.875 * max_delta / speed_limit)
    steps = max(2, int(round(duration * 50.0)))
    controller._moving = True
    try:
        for step in range(steps + 1):
            progress = step / steps
            blend = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
            controller._q_target[:] = q_start + (target - q_start) * blend
            if step < steps:
                time.sleep(duration / steps)
        controller._q_target[:] = target
    finally:
        controller._moving = False
    return True


def freeze_d435_scan_center(runtime: ArmRuntime) -> np.ndarray:
    """Freeze the commanded scan centre without consulting potentially stale feedback."""

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")
    controller = runtime.controller
    joint_count = int(getattr(controller, "_n", 0))
    if joint_count <= D435_SCAN_JOINT5_INDEX:
        raise RuntimeError("D435 补偿扫描要求至少包含 J1 至 J5")
    target = np.asarray(getattr(controller, "_q_target", ()), dtype=np.float64)[:joint_count].copy()
    if target.shape != (joint_count,) or not np.all(np.isfinite(target)):
        raise RuntimeError("scan_pose0 中心关节目标无效")
    return target


def _recovery_target(center_q: np.ndarray, view: D435RecoveryView) -> np.ndarray:
    target = np.asarray(center_q, dtype=np.float64).copy()
    if target.ndim != 1 or target.size <= D435_SCAN_JOINT5_INDEX or not np.all(np.isfinite(target)):
        raise ValueError("D435 扫描中心关节目标无效")
    target[D435_SCAN_JOINT1_INDEX] = center_q[D435_SCAN_JOINT1_INDEX] + view.joint1_offset_rad
    target[D435_SCAN_JOINT5_INDEX] = center_q[D435_SCAN_JOINT5_INDEX] + view.joint5_offset_rad
    return target


def validate_d435_recovery_targets(
    runtime: ArmRuntime,
    center_q: np.ndarray,
    views: tuple[D435RecoveryView, ...],
    poses_config: dict[str, Any],
) -> None:
    """Reject all compensation motion if any J1/J5 target violates URDF limits."""

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")
    if len(views) != 6:
        raise ValueError("D435 补偿扫描必须包含6个视角")
    _, _, _, margin = _d435_recovery_parameters(poses_config)
    controller = runtime.controller
    model = getattr(controller, "_model", None)
    lower = np.asarray(getattr(model, "lowerPositionLimit", ()), dtype=np.float64)
    upper = np.asarray(getattr(model, "upperPositionLimit", ()), dtype=np.float64)
    if lower.size <= D435_SCAN_JOINT5_INDEX or upper.size <= D435_SCAN_JOINT5_INDEX:
        raise RuntimeError("机械臂控制器未提供 J1/J5 的 URDF 限位")
    for joint_index, joint_name in ((D435_SCAN_JOINT1_INDEX, "J1"), (D435_SCAN_JOINT5_INDEX, "J5")):
        if not (isfinite(float(lower[joint_index])) and isfinite(float(upper[joint_index]))):
            raise RuntimeError(f"机械臂 URDF 未提供 {joint_name} 有效限位")
        if lower[joint_index] + margin >= upper[joint_index] - margin:
            raise RuntimeError(f"{joint_name} 的补偿扫描安全余量无效")
    for index, view in enumerate(views, start=1):
        target = _recovery_target(center_q, view)
        for joint_index, joint_name in ((D435_SCAN_JOINT1_INDEX, "J1"), (D435_SCAN_JOINT5_INDEX, "J5")):
            if not lower[joint_index] + margin <= target[joint_index] <= upper[joint_index] - margin:
                raise ValueError(
                    f"D435 补偿视角{index}的 {joint_name} 目标 {target[joint_index]:.6f} 超出 URDF 安全限位"
                )


def move_to_d435_recovery_view(
    runtime: ArmRuntime,
    center_q: np.ndarray,
    view: D435RecoveryView,
    poses_config: dict[str, Any],
    *,
    label: str = "",
) -> bool:
    """Move J1/J5 to one absolute-offset D435 compensation view.

    The target is always derived from ``center_q``.  It deliberately never
    accumulates offsets from the previous visual viewpoint.
    """

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")
    controller = runtime.controller
    if not getattr(controller, "_running", False):
        raise RuntimeError("机械臂控制循环尚未启动，不能执行 D435 补偿扫描")
    joint_count = int(getattr(controller, "_n", 0))
    if joint_count <= D435_SCAN_JOINT5_INDEX:
        raise RuntimeError("D435 补偿扫描要求至少包含 J1 至 J5")
    min_duration, speed_limit, _, _ = _d435_recovery_parameters(poses_config)
    feedback_tolerance, feedback_timeout, feedback_retries = _d435_recovery_feedback_parameters(poses_config)
    target = _recovery_target(center_q, view)
    if target.shape != (joint_count,):
        raise ValueError("D435 补偿扫描中心关节数量不匹配")
    view_name = label or "D435 补偿视角"
    for attempt in range(feedback_retries + 1):
        if not _send_recovery_target(
            runtime,
            target,
            min_duration=min_duration,
            speed_limit=speed_limit,
        ):
            return False
        arrived, actual, error = _wait_for_recovery_feedback(
            runtime,
            target,
            timeout_sec=feedback_timeout,
            tolerance_rad=feedback_tolerance,
        )
        if actual is None or error is None:
            LOGGER.warning(
                "%s 关节反馈无效: command_attempt=%s/%s",
                view_name,
                attempt + 1,
                feedback_retries + 1,
            )
        else:
            LOGGER.info(
                "%s 关节反馈: command_attempt=%s/%s target={J1=%.4f,J5=%.4f} actual={J1=%.4f,J5=%.4f} error={J1=%.4f,J5=%.4f}",
                view_name,
                attempt + 1,
                feedback_retries + 1,
                target[D435_SCAN_JOINT1_INDEX],
                target[D435_SCAN_JOINT5_INDEX],
                actual[D435_SCAN_JOINT1_INDEX],
                actual[D435_SCAN_JOINT5_INDEX],
                error[0],
                error[1],
            )
        if arrived:
            return True
        if attempt < feedback_retries:
            LOGGER.warning("%s 未在 %.2fs 内到位；重发一次绝对补偿目标", view_name, feedback_timeout)
    LOGGER.error("%s 关节反馈未在容差 %.3frad 内到位", view_name, feedback_tolerance)
    return False


def _wait_for_motion(controller: Any, duration_sec: float, extra_sec: float) -> bool:
    """等待 SDK 的末端轨迹发送线程完成。"""

    sender = getattr(controller, "_send_thread", None)
    if sender is not None and sender.is_alive():
        sender.join(timeout=max(2.0, duration_sec + extra_sec + 2.0))
        return not sender.is_alive()

    # SDK 在未创建发送线程时，沿用原 mobile_2d_grasp.py 的等待语义。
    time.sleep(max(0.0, duration_sec + extra_sec))
    return True


def move_to_ready_pose(
    runtime: ArmRuntime,
    poses_config: dict[str, Any],
    *,
    wait_extra_sec: float = 0.6,
) -> bool:
    """移动至 ``poses.yaml`` 的 ready_pose，并等待控制器轨迹完成。

    该函数复用已验证抓取流程的控制顺序：调用
    ``RebotArmEndPose.move_to_traj`` 后等待发送线程完成。POS_VEL 模式下的
    缓存反馈可能滞后或暂时为零，不能据此否定已经完成的控制器轨迹。返回
    ``False`` 表示 IK/轨迹创建失败或发送超时；配置错误和运行时未启动属于
    调用错误，会抛出异常。
    """

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")
    if not getattr(runtime.controller, "_running", False):
        raise RuntimeError("机械臂控制循环尚未启动，不能执行 ready_pose")
    if not isinstance(poses_config, dict):
        raise TypeError("poses 配置必须是 dict")

    ready = poses_config.get("ready_pose")
    if not isinstance(ready, dict):
        raise ValueError("poses 配置缺少 ready_pose mapping")

    required = ("x", "y", "z", "roll", "pitch", "yaw", "duration_sec")
    missing = [key for key in required if key not in ready]
    if missing:
        raise ValueError(f"ready_pose 缺少字段: {', '.join(missing)}")

    pose = tuple(float(ready[key]) for key in ("x", "y", "z", "roll", "pitch", "yaw"))
    duration_sec = float(ready["duration_sec"])
    if duration_sec <= 0.0:
        raise ValueError("ready_pose.duration_sec 必须大于 0")

    if not runtime.controller.move_to_traj(*pose, duration=duration_sec):
        return False
    return _wait_for_motion(runtime.controller, duration_sec, wait_extra_sec)


def move_to_scan_pose(
    runtime: ArmRuntime,
    poses_config: dict[str, Any],
) -> bool:
    """From ``ready_pose``, rotate J1 into the configured D435 scan pose.

    ``scan_pose0.joint1_offset_rad`` is relative to the joint feedback after
    the ready-pose trajectory, rather than a global joint target.  This keeps
    the existing verified ready pose intact while applying the configured
    clockwise scan rotation.  The minimum-jerk trajectory will be extended if
    needed to remain within ``joint1_speed_limit_rad_s``.
    """

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")
    controller = runtime.controller
    if not getattr(controller, "_running", False):
        raise RuntimeError("机械臂控制循环尚未启动，不能执行 scan_pose0")
    if not isinstance(poses_config, dict):
        raise TypeError("poses 配置必须是 dict")

    scan = poses_config.get("scan_pose0")
    if not isinstance(scan, dict):
        raise ValueError("poses 配置缺少 scan_pose0 mapping")
    required = ("joint1_offset_rad", "duration_sec", "joint1_speed_limit_rad_s")
    missing = [key for key in required if key not in scan]
    if missing:
        raise ValueError(f"scan_pose0 缺少字段: {', '.join(missing)}")

    offset = float(scan["joint1_offset_rad"])
    configured_duration = float(scan["duration_sec"])
    speed_limit = float(scan["joint1_speed_limit_rad_s"])
    if not all(isfinite(value) for value in (offset, configured_duration, speed_limit)):
        raise ValueError("scan_pose0 参数必须是有限数字")
    if configured_duration <= 0.0 or speed_limit <= 0.0:
        raise ValueError("scan_pose0.duration_sec 和 joint1_speed_limit_rad_s 必须大于0")

    joint_count = int(getattr(controller, "_n", 0))
    if joint_count < 1:
        raise RuntimeError("机械臂控制器没有可用的 J1")
    start_reader = getattr(controller, "_arm_motion_start", None)
    if not callable(start_reader):
        raise RuntimeError("机械臂控制器不支持读取扫描姿势起点")
    q_start = np.asarray(start_reader(), dtype=np.float64)[:joint_count].copy()
    if q_start.shape != (joint_count,) or not np.all(np.isfinite(q_start)):
        raise RuntimeError("scan_pose0 起始关节反馈无效")
    q_target = q_start.copy()
    q_target[0] += offset

    # A minimum-jerk segment has a peak velocity of 1.875 * delta / duration.
    duration = max(configured_duration, 1.875 * abs(offset) / speed_limit)
    sender = getattr(controller, "_send_thread", None)
    stop_sender = getattr(controller, "_stop_send", None)
    if sender is not None and sender.is_alive():
        if stop_sender is not None:
            stop_sender.set()
        sender.join(timeout=max(2.0, configured_duration + 2.0))
        if sender.is_alive():
            return False
        if stop_sender is not None:
            stop_sender.clear()

    steps = max(2, int(round(duration * 50.0)))
    controller._moving = True
    try:
        for step in range(steps + 1):
            progress = step / steps
            blend = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
            controller._q_target[:] = q_start + (q_target - q_start) * blend
            if step < steps:
                time.sleep(duration / steps)
        controller._q_target[:] = q_target
    finally:
        controller._moving = False
    return True


def open_gripper(
    runtime: ArmRuntime,
    *,
    distance_m: float | None = None,
    timeout_sec: float = 3.0,
    position_tolerance_rad: float = GRIPPER_OPEN_POSITION_TOLERANCE_RAD,
) -> bool:
    """使用已验证的 ``GraspDriver`` 打开 J7 夹爪并确认位置到位。

    本函数不直接发送 MIT 命令。它只调用 ``GraspDriver.open_gripper()``，
    再使用同一驱动器的开口角、软限位和状态反馈确认实际位置。这样 J7 的
    位置控制、限位裁剪与控制循环都保持为 ``mobile_2d_grasp.py`` 的已验证逻辑。
    """

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")
    if not getattr(runtime.controller, "_running", False):
        raise RuntimeError("机械臂控制循环尚未启动，不能打开夹爪")
    if timeout_sec <= 0.0:
        raise ValueError("timeout_sec 必须大于 0")
    if position_tolerance_rad <= 0.0:
        raise ValueError("position_tolerance_rad 必须大于 0")

    driver = runtime.gripper_driver
    max_distance_m = float(driver.MAX_DISTANCE_M)
    requested_distance_m = max_distance_m if distance_m is None else float(distance_m)
    if not isfinite(requested_distance_m):
        raise ValueError("distance_m 必须是有限数字")

    # 与 GraspDriver.open_gripper() 一致：开口距离裁剪到 [0, MAX_DISTANCE_M]。
    requested_distance_m = min(max(requested_distance_m, 0.0), max_distance_m)
    driver.open_gripper(distance_m=requested_distance_m, timeout=timeout_sec)

    raw_target = (requested_distance_m / max_distance_m) * float(driver._angle_open)
    target_position = min(max(raw_target, float(driver._open_lo)), float(driver._open_hi))
    actual_position, _, _ = driver.get_gripper_state()

    return abs(float(actual_position) - target_position) <= position_tolerance_rad
