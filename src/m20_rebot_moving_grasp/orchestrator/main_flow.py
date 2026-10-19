"""M20 Pro + DM-B601 联合任务主状态机。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Callable, Protocol

from ..machine_dog.interface import MachineDogInterface


LOGGER = logging.getLogger(__name__)


def _release_grasp_vision(arm: ArmTaskInterface) -> None:
    """Release D435 between grasp phases without constraining other adapters."""

    release = getattr(arm, "release_vision", None)
    if callable(release):
        release()


def _arm_failure_reason(arm: ArmTaskInterface, fallback: str) -> str:
    reason = str(getattr(arm, "last_failure_reason", "") or "").strip()
    return reason or fallback


def _request_dog_prone_on_abort(dog: MachineDogInterface) -> None:
    """Request the adapter's cancel-and-prone abort path without masking the cause."""

    dog_abort = getattr(dog, "emergency_stop", None)
    try:
        if callable(dog_abort):
            dog_abort()
            return
        dog.sit()
    except Exception:
        LOGGER.exception("任务异常退出后的机器狗原地趴下请求失败")


def _abort_task(
    dog: MachineDogInterface,
    cycles: int,
    history: list[TaskState],
    error: str,
    *,
    append_error_state: bool = False,
) -> TaskResult:
    """Return a failed task result only after attempting the M20 prone path."""

    _request_dog_prone_on_abort(dog)
    if append_error_state:
        history.append(TaskState.ERROR)
    return TaskResult(False, TaskState.ERROR, cycles, history, error)


class ArmTaskInterface(Protocol):
    def init_reset(self) -> bool: ...
    def prepare_grasp(self) -> bool: ...
    def grasp_current_target(self) -> bool: ...
    def hold(self) -> bool: ...
    def drop(self) -> bool: ...
    def safe_home(self) -> bool: ...
    def emergency_stop(self) -> bool: ...
    def shutdown(self) -> None: ...


class TaskState(str, Enum):
    INIT = "INIT"
    SCAN_NAV = "SCAN_NAV"
    SCANNING = "SCANNING"
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    TRANSPORT = "TRANSPORT"
    BIN_NAV = "BIN_NAV"
    DROP = "DROP"
    CLEAN_RETURN = "CLEAN_RETURN"
    DONE = "DONE"
    WAIT_OPERATOR = "WAIT_OPERATOR"
    ERROR = "ERROR"


@dataclass
class TaskResult:
    success: bool
    state: TaskState
    cycles: int
    history: list[TaskState] = field(default_factory=list)
    error: str = ""


def run_task(
    dog: MachineDogInterface,
    arm: ArmTaskInterface,
    *,
    max_cycles: int | None = None,
    operator_wait: Callable[[], str] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> TaskResult:
    """执行完整任务；抓取失败后等待人工 ``resume`` 或 ``abort``。"""
    state = TaskState.INIT
    history: list[TaskState] = [state]
    cycles = 0
    try:
        # Keep the two checks separate.  The previous short-circuit expression
        # hid which device failed and could skip the M20 check entirely when
        # the arm initialization returned False.
        if not arm.init_reset():
            LOGGER.error("机械臂初始化失败")
            return _abort_task(dog, cycles, history, "机械臂初始化失败")
        if not dog.init_reset():
            LOGGER.error("M20 机器狗初始化失败")
            return _abort_task(dog, cycles, history, "M20 机器狗初始化失败")
        if not dog.stand():
            return _abort_task(dog, cycles, history, "机器狗站立失败")
        while max_cycles is None or cycles < max_cycles:
            if stop_requested is not None and stop_requested():
                arm.emergency_stop()
                return _abort_task(dog, cycles, history, "收到停止请求")
            if cycles > 0 and not dog.stand():
                return _abort_task(dog, cycles, history, "下一轮扫描前机器狗站立失败")
            state = TaskState.SCAN_NAV; history.append(state)
            if not dog.goto_target():
                return _abort_task(dog, cycles, history, "导航到扫描点失败")
            state = TaskState.SCANNING; history.append(state)
            if dog.is_clean():
                state = TaskState.CLEAN_RETURN; history.append(state)
                if not dog.goto_home():
                    return _abort_task(dog, cycles, history, "返回Home失败")
                state = TaskState.DONE; history.append(state)
                return TaskResult(True, state, cycles, history)
            state = TaskState.APPROACH; history.append(state)
            # ``approach`` only succeeds once M20 has acknowledged the final
            # prone command; sending a second motion-state command here would
            # race the verified D455 approach sequence.
            if not dog.approach():
                return _abort_task(dog, cycles, history, "接近、复核或趴下失败")
            state = TaskState.GRASP; history.append(state)
            if not arm.prepare_grasp():
                arm.safe_home()
                _release_grasp_vision(arm)
                state = TaskState.WAIT_OPERATOR; history.append(state)
                action = operator_wait() if operator_wait is not None else "abort"
                if str(action).lower() == "resume":
                    if not dog.stand():
                        return _abort_task(dog, cycles, history, "人工恢复后机器狗站立失败")
                    continue
                return TaskResult(False, TaskState.WAIT_OPERATOR, cycles, history,
                                  f"抓取准备失败: {_arm_failure_reason(arm, '未知原因')}；等待人工处理")
            if not arm.grasp_current_target():
                arm.safe_home()
                _release_grasp_vision(arm)
                state = TaskState.WAIT_OPERATOR; history.append(state)
                action = operator_wait() if operator_wait is not None else "abort"
                if str(action).lower() == "resume":
                    if not dog.stand():
                        return _abort_task(dog, cycles, history, "人工恢复后机器狗站立失败")
                    continue
                return TaskResult(False, TaskState.WAIT_OPERATOR, cycles, history,
                                  f"抓取动作失败: {_arm_failure_reason(arm, '未知原因')}；等待人工处理")
            if not arm.hold():
                return _abort_task(dog, cycles, history, "进入保持姿态失败")
            state = TaskState.TRANSPORT; history.append(state)
            if not dog.stand():
                return _abort_task(dog, cycles, history, "运输前站立失败")
            state = TaskState.BIN_NAV; history.append(state)
            if not dog.goto_bin() or not dog.sit():
                return _abort_task(dog, cycles, history, "投放点导航或趴下失败")
            state = TaskState.DROP; history.append(state)
            if not arm.drop() or not arm.safe_home():
                return _abort_task(dog, cycles, history, "投放或安全回位失败")
            _release_grasp_vision(arm)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                state = TaskState.CLEAN_RETURN; history.append(state)
                if not dog.stand() or not dog.goto_home():
                    return _abort_task(dog, cycles, history, "达到轮数上限后返回Home失败")
                state = TaskState.DONE; history.append(state)
                return TaskResult(True, state, cycles, history)
        return TaskResult(True, TaskState.DONE, cycles, history)
    except Exception as exc:
        try:
            arm.emergency_stop()
        except Exception:
            pass
        try:
            return _abort_task(dog, cycles, history, str(exc), append_error_state=True)
        except Exception:
            pass
        history.append(TaskState.ERROR)
        return TaskResult(False, TaskState.ERROR, cycles, history, str(exc))
    finally:
        try:
            arm.shutdown()
        except Exception:
            pass


__all__ = ["ArmTaskInterface", "TaskState", "TaskResult", "run_task"]
