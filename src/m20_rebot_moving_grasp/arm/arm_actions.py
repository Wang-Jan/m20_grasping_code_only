"""机械臂单次抓取、保持、投放和安全停机动作。

夹爪只通过已经在 ``mobile_2d_grasp.py`` 真机验证的 ``GraspDriver`` 控制：
``open_gripper`` 打开，``grasp`` 以 J7 扭矩/堵转逻辑判断接触，
``release_gripper`` 释放。这里不再用 D435 判断抓取结果。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .arm_motion import _wait_for_motion, open_gripper
from .arm_runtime import ArmRuntime, stop_arm_runtime
from .vision_2d import BaseGraspPoses


@dataclass(frozen=True)
class GraspAttemptResult:
    success: bool
    attempts: int
    last_error: str = ""


def _pose_values(pose: Any, name: str) -> tuple[float, ...]:
    if not isinstance(pose, (list, tuple)) or len(pose) != 6:
        raise ValueError(f"{name} 必须是长度为6的(x,y,z,roll,pitch,yaw)")
    values = tuple(float(v) for v in pose)
    if not all(np.isfinite(values)):
        raise ValueError(f"{name} 包含非有限数字")
    return values


def move_to_cartesian_pose(
    runtime: ArmRuntime,
    pose: tuple[float, ...],
    duration_sec: float,
    *,
    wait_extra_sec: float = 0.6,
) -> bool:
    """通过 SDK 的 ``move_to_traj`` 移动，并等待发送线程结束。"""

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")
    if not getattr(runtime.controller, "_running", False):
        raise RuntimeError("机械臂控制循环尚未启动")
    pose = _pose_values(pose, "pose")
    duration_sec = float(duration_sec)
    if duration_sec <= 0.0:
        raise ValueError("duration_sec 必须大于0")
    if not runtime.controller.move_to_traj(*pose, duration=duration_sec):
        return False
    return _wait_for_motion(runtime.controller, duration_sec, wait_extra_sec)


def _pose_duration_sec(pose: dict[str, Any], name: str) -> float:
    """Read a pose duration while accepting the verified YAML field name."""

    value = pose.get("duration", pose.get("duration_sec"))
    if value is None:
        raise ValueError(f"{name} 缺少 duration")
    duration_sec = float(value)
    if not np.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError(f"{name}.duration 必须大于0且为有限数字")
    return duration_sec


def execute_grasp_attempt(
    runtime: ArmRuntime,
    poses: BaseGraspPoses,
    poses_config: dict[str, Any],
    *,
    force: Optional[float] = None,
) -> bool:
    """执行一次 pregrasp→grasp→J7 闭合→retreat。

    ``GraspDriver.grasp`` 返回 True 仅表示达到扭矩并停住；夹爪完全闭合的
    硬限位路径返回 False。无论结果如何，函数都会先退回 retreat 位姿，
    由上层决定是否重新检测和重试。
    """

    if not isinstance(poses, BaseGraspPoses):
        raise TypeError("poses 必须是 BaseGraspPoses")
    if not isinstance(poses_config, dict):
        raise TypeError("poses_config 必须是 dict")
    motion = poses_config.get("arm_motion", {})
    pre_d = float(motion.get("pregrasp_duration_sec", 2.0))
    grasp_d = float(motion.get("grasp_duration_sec", 1.5))
    retreat_d = float(motion.get("retreat_duration_sec", 1.5))

    # 夹爪先打开；这是进入目标上方检测位姿后的固定动作。
    if not open_gripper(runtime):
        return False
    if not move_to_cartesian_pose(runtime, poses.pregrasp_pose, pre_d):
        return False
    if not move_to_cartesian_pose(runtime, poses.grasp_pose, grasp_d):
        return False

    driver = runtime.gripper_driver
    grasp_ok = bool(driver.grasp(force=force) if force is not None else driver.grasp())
    # 即使 J7 完全闭合导致失败，也必须撤离，避免继续压在物体/地面上。
    retreat_ok = move_to_cartesian_pose(runtime, poses.retreat_pose, retreat_d)
    return grasp_ok and retreat_ok


def grasp_with_retry(
    runtime: ArmRuntime,
    first_poses: BaseGraspPoses,
    poses_config: dict[str, Any],
    *,
    max_attempts: int = 2,
    reacquire_poses: Optional[Callable[[], Optional[BaseGraspPoses]]] = None,
    return_to_detection: Optional[Callable[[], bool]] = None,
) -> GraspAttemptResult:
    """抓取失败后回到检测姿势重新识别，最多执行 ``max_attempts`` 次。"""

    if max_attempts < 1:
        raise ValueError("max_attempts 必须至少为1")
    poses: Optional[BaseGraspPoses] = first_poses
    last_error = ""
    for attempt in range(1, int(max_attempts) + 1):
        if poses is None:
            last_error = "重新检测未得到可用抓取位姿"
            break
        try:
            if execute_grasp_attempt(runtime, poses, poses_config):
                return GraspAttemptResult(True, attempt)
            last_error = "J7 未达到抓取扭矩或撤退失败"
        except Exception as exc:
            last_error = str(exc)
        if attempt == max_attempts:
            break
        if return_to_detection is not None and not return_to_detection():
            last_error = "返回检测姿势失败"
            break
        poses = reacquire_poses() if reacquire_poses is not None else poses
    return GraspAttemptResult(False, min(int(max_attempts), attempt), last_error)


def move_to_hold_pose(runtime: ArmRuntime, poses_config: dict[str, Any]) -> bool:
    """移动到投放前保持姿态；保留 J7 当前扭矩保持状态。"""

    pose = poses_config.get("transport_pose")
    if not isinstance(pose, dict):
        raise ValueError("poses 配置缺少 transport_pose")
    target = pose.get("joint_positions_rad")
    if not isinstance(target, (list, tuple)) or len(target) != int(runtime.controller._n):
        raise ValueError("transport_pose.joint_positions_rad 关节数量不匹配")
    target = np.asarray([float(v) for v in target], dtype=np.float64)
    if not np.all(np.isfinite(target)):
        raise ValueError("transport_pose 关节目标包含非有限值")
    duration = float(pose.get("duration_sec", 0.0))
    if duration <= 0.0:
        raise ValueError("transport_pose.duration_sec 必须大于0")

    controller = runtime.controller
    sender = getattr(controller, "_send_thread", None)
    if sender is not None and sender.is_alive():
        controller._stop_send.set()
        sender.join(timeout=10.0)
        controller._stop_send.clear()
    q_start = np.asarray(controller._arm_motion_start(), dtype=np.float64)[: controller._n]
    steps = max(2, int(round(duration * 50.0)))
    controller._moving = True
    for step in range(steps + 1):
        s = step / steps
        blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        controller._q_target[:] = q_start + (target - q_start) * blend
        if step < steps:
            time.sleep(duration / steps)
    controller._q_target[:] = target
    controller._moving = False
    # 与 mobile_grasp.py 的关节轨迹保持一致：反馈偏差过大时不能宣称已到保持位。
    get_state = getattr(runtime.rebotarm, "get_state", None)
    if callable(get_state):
        q_now = np.asarray(get_state(request_feedback=False)[0], dtype=np.float64)[: controller._n]
        if q_now.shape != target.shape or np.max(np.abs(q_now - target)) > 0.08:
            return False
    return True


def move_to_pre_drop_pose(runtime: ArmRuntime, poses_config: dict[str, Any]) -> bool:
    """以已验证的 Cartesian 投放位姿移动，必要时使用确定性 IK 初值。"""

    pose = poses_config.get("pre_drop_pose")
    if not isinstance(pose, dict):
        raise ValueError("poses 配置缺少 pre_drop_pose")
    values = tuple(float(pose[k]) for k in ("x", "y", "z", "roll", "pitch", "yaw"))
    duration_sec = _pose_duration_sec(pose, "pre_drop_pose")
    seed = pose.get("ik_seed_joint_positions_rad")
    if seed is None:
        return move_to_cartesian_pose(runtime, values, duration_sec)

    ik_seed = np.asarray(seed, dtype=np.float64)
    if ik_seed.shape != (int(runtime.controller._n),) or not np.all(np.isfinite(ik_seed)):
        raise ValueError("pre_drop_pose.ik_seed_joint_positions_rad 与机械臂关节数不匹配")
    if not runtime.controller.move_to_traj(
        *values,
        duration=duration_sec,
        ik_seed=ik_seed,
    ):
        return False
    return _wait_for_motion(runtime.controller, duration_sec, 0.6)


def drop_object(runtime: ArmRuntime, poses_config: dict[str, Any]) -> bool:
    """在预投放位姿释放物体；释放后夹爪回到关闭静止位。"""

    if not move_to_pre_drop_pose(runtime, poses_config):
        return False
    runtime.gripper_driver.release_gripper()
    return True


def safe_home(runtime: ArmRuntime) -> bool:
    """回到机械臂安全位，不改变夹爪当前状态。"""

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")
    if not getattr(runtime.controller, "_running", False):
        raise RuntimeError("机械臂控制循环尚未启动")
    try:
        runtime.controller.safe_home()
        return True
    except Exception:
        return False


def emergency_stop(runtime: ArmRuntime) -> bool:
    """受控急停序列：张开→闭合→safe_home→所有电机失能。"""

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")
    ok = True
    try:
        open_gripper(runtime)
    except Exception:
        ok = False
    try:
        runtime.gripper_driver.grasp()
    except Exception:
        ok = False
    try:
        runtime.controller.safe_home()
    except Exception:
        ok = False
    try:
        stop_loop = getattr(runtime.rebotarm, "stop_control_loop", None)
        if callable(stop_loop):
            stop_loop()
        runtime.rebotarm.disable_all()
        runtime.controller._running = False
    except Exception:
        ok = False
    return ok


__all__ = [
    "GraspAttemptResult", "move_to_cartesian_pose", "execute_grasp_attempt",
    "grasp_with_retry", "move_to_hold_pose", "move_to_pre_drop_pose",
    "drop_object", "emergency_stop",
    "safe_home",
]
