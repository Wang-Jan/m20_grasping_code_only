"""将 DM-B601 运行时、D435 视觉和抓取动作封装成任务接口。"""

from __future__ import annotations

import sys
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .arm_actions import (
    emergency_stop, grasp_with_retry, move_to_hold_pose,
    safe_home, drop_object,
)
from .arm_camera import D435Runtime, capture_frame, start_d435, stop_d435
from .arm_motion import (
    D435RecoveryView, d435_recovery_settle_sec, d435_recovery_views,
    freeze_d435_scan_center, move_to_d435_recovery_view,
    move_to_ready_pose, move_to_scan_pose, validate_d435_recovery_targets,
)
from .arm_runtime import ArmRuntime, create_arm_runtime, start_arm_runtime, stop_arm_runtime
from .vision_2d import (
    BaseGraspPoses, check_grasp_poses_ik, create_2d_vision_runtime,
    detect_2d_targets, estimate_2d_grasps, select_best_grasp, transform_grasp_to_base,
)


LOGGER = logging.getLogger(__name__)


class GraspPreparationFailureKind(str, Enum):
    """Structured reason category for a failed ``prepare_grasp`` call."""

    NONE = "none"
    VISUAL_MISS = "visual_miss"
    RECOVERABLE_NO_GRASP = "recoverable_no_grasp"
    ARM_MOTION = "arm_motion"
    VISION_RUNTIME = "vision_runtime"
    TRANSFORM = "transform"
    IK = "ik"


class _ViewOutcome(str, Enum):
    SUCCESS = "success"
    RECOVERABLE_NO_GRASP = "recoverable_no_grasp"
    FAILURE = "failure"


@dataclass(frozen=True)
class _ViewAttempt:
    outcome: _ViewOutcome
    reason: str = ""


class _IkChecker:
    """复用 DM SDK 的 IK 求解器，并以当前反馈作为默认种子。"""

    def __init__(self, runtime: ArmRuntime):
        sdk_root = str(runtime.sdk_root)
        if sdk_root not in sys.path:
            sys.path.insert(0, sdk_root)
        from reBotArm_control_py.kinematics import (  # type: ignore[import-not-found]
            get_end_effector_frame_id, load_robot_model, pad_q_for_model, pos_rot_to_se3, solve_ik,
        )
        from reBotArm_control_py.kinematics.inverse_kinematics import IKParams  # type: ignore[import-not-found]
        self._arm = runtime.rebotarm
        self._n = int(runtime.controller._n)
        self._model = load_robot_model()
        self._data = self._model.createData()
        self._frame = get_end_effector_frame_id(self._model)
        self._pad = pad_q_for_model
        self._pose_to_se3 = pos_rot_to_se3
        self._solve_ik = solve_ik
        self._params = IKParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)

    def _current_q(self) -> np.ndarray:
        return np.asarray(self._arm.get_state(request_feedback=False)[0][: self._n], dtype=np.float64)

    def solve_pose(self, pose: tuple[float, ...], q_seed: Any = None):
        seed = self._current_q() if q_seed is None else np.asarray(q_seed, dtype=np.float64)
        q_init = self._pad(self._model, seed, self._n)
        target = self._pose_to_se3(np.asarray(pose[:3]), roll=pose[3], pitch=pose[4], yaw=pose[5])
        result = self._solve_ik(self._model, self._data, self._frame, target, q_init, self._params, controlled_joints=self._n)
        return bool(result.success), float(result.error), np.asarray(result.q[: self._n], dtype=np.float64)


class ArmTaskAdapter:
    """主状态机使用的机械臂任务外观。"""

    def __init__(
        self,
        arm_config: dict[str, Any],
        vision_config: dict[str, Any],
        poses_config: dict[str, Any],
        *,
        max_attempts: int = 2,
        d435_factory: Optional[Callable[..., Any]] = None,
        vision_runtime_factory: Optional[Callable[..., Any]] = None,
        ik_checker_factory: Optional[Callable[[ArmRuntime], Any]] = None,
    ) -> None:
        self.arm_config = arm_config
        self.vision_config = vision_config
        self.poses_config = poses_config
        self.max_attempts = int(max_attempts)
        self.d435_factory = d435_factory
        self.vision_runtime_factory = vision_runtime_factory
        self.ik_checker_factory = ik_checker_factory
        self.runtime: ArmRuntime | None = None
        self.d435: D435Runtime | None = None
        self.vision: Any = None
        self.current_poses: BaseGraspPoses | None = None
        self.arm_at_safe_home = False
        self.arm_at_detection_pose = False
        self.last_failure_reason = ""
        self.last_failure_kind = GraspPreparationFailureKind.NONE
        self._scan_center_q: np.ndarray | None = None

    @property
    def is_recoverable_detection_failure(self) -> bool:
        """Whether scanning exhausted views without a feasible grasp pose."""

        return self.last_failure_kind in (
            GraspPreparationFailureKind.VISUAL_MISS,
            GraspPreparationFailureKind.RECOVERABLE_NO_GRASP,
        )

    def _grasp_failure(
        self,
        reason: str,
        *,
        kind: GraspPreparationFailureKind,
        exc_info: bool = False,
    ) -> bool:
        """Persist a human-readable failure reason for both task entrypoints."""

        self.last_failure_reason = reason
        self.last_failure_kind = kind
        LOGGER.error("机械臂抓取准备失败: %s", reason, exc_info=exc_info)
        return False

    def init_reset(self) -> bool:
        if self.runtime is not None:
            return True
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须至少为1")
        try:
            self.runtime = create_arm_runtime(self.arm_config)
            start_arm_runtime(self.runtime)
            # The ready pose is a pre-grasp pose and must not be entered while
            # the dog is still navigating/scanning.  Keep the arm at the
            # controller's verified safe-home (zero-joint) pose until the
            # D455 approach has completed and the dog is prone.
            if not safe_home(self.runtime):
                self.shutdown()
                return False
            self.arm_at_safe_home = True
            self.arm_at_detection_pose = False
            self._scan_center_q = None
            return True
        except Exception:
            LOGGER.exception("DM-B601 初始化异常；将释放已创建的设备资源")
            self.shutdown()
            return False

    def _start_grasp_vision(self) -> bool:
        """Start D435 only after D455 approach has released its camera stream.

        D455 owns navigation-time perception and D435 is required only for
        the subsequent arm grasp.  Keeping both RGB-D streams open during
        navigation is unnecessary and can exhaust the Jetson USB 2.0 bus.
        """

        if self.d435 is not None and self.vision is not None:
            return True
        if self.runtime is None:
            raise RuntimeError("机械臂适配器尚未初始化")
        try:
            self.d435 = start_d435(self.runtime, self.vision_config, camera_factory=self.d435_factory)
            factory = self.vision_runtime_factory or create_2d_vision_runtime
            self.vision = factory(self.d435, self.vision_config, self.runtime.grasp_root)
            return True
        except Exception:
            LOGGER.exception("D435 抓取视觉初始化异常；不会开始检测")
            self.release_vision()
            return False

    def _move_to_detection_pose(self) -> bool:
        """Move safe-home arm to the verified D435 scanning viewpoint."""

        if self.runtime is None:
            raise RuntimeError("机械臂适配器尚未初始化")
        if self.arm_at_detection_pose:
            return True
        LOGGER.info("机器狗已趴下；机械臂开始进入 D435 扫描姿势：ready_pose")
        if not move_to_ready_pose(self.runtime, self.poses_config):
            LOGGER.error("机械臂 ready_pose 失败；不会打开 D435 或开始检测")
            self.last_failure_reason = "机械臂未能进入 ready_pose"
            self.last_failure_kind = GraspPreparationFailureKind.ARM_MOTION
            return False
        LOGGER.info("机械臂 ready_pose 控制器轨迹完成；开始 J1 scan_pose0")
        if not move_to_scan_pose(self.runtime, self.poses_config):
            LOGGER.error("机械臂 scan_pose0 失败；不会打开 D435 或开始检测")
            self.last_failure_reason = "机械臂未能进入 scan_pose0"
            self.last_failure_kind = GraspPreparationFailureKind.ARM_MOTION
            return False
        try:
            self._scan_center_q = freeze_d435_scan_center(self.runtime)
        except Exception as exc:
            self.last_failure_reason = f"无法冻结 scan_pose0 中心关节目标: {exc}"
            self.last_failure_kind = GraspPreparationFailureKind.ARM_MOTION
            LOGGER.exception("机械臂 scan_pose0 中心目标无效；不会打开 D435")
            return False
        LOGGER.info("机械臂 J1 scan_pose0 控制器轨迹完成；允许启动 D435")
        self.arm_at_safe_home = False
        self.arm_at_detection_pose = True
        return True

    def _detect_current_view(self, label: str) -> _ViewAttempt:
        """Capture and evaluate one D435 viewpoint without moving the arm."""

        if self.runtime is None or self.d435 is None or self.vision is None:
            self._grasp_failure(
                "D435 抓取视觉初始化后未获得可用运行时",
                kind=GraspPreparationFailureKind.VISION_RUNTIME,
            )
            return _ViewAttempt(_ViewOutcome.FAILURE, self.last_failure_reason)
        try:
            frame = capture_frame(self.d435)
            detections = detect_2d_targets(self.vision, frame)
            grasps = estimate_2d_grasps(self.vision, frame, detections)
        except Exception as exc:
            reason = f"D435 取帧、YOLO 检测或二维抓取估计异常: {exc}"
            self._grasp_failure(reason, kind=GraspPreparationFailureKind.VISION_RUNTIME, exc_info=True)
            return _ViewAttempt(_ViewOutcome.FAILURE, reason)
        detected_items = list(getattr(detections, "detections", ()) or ())
        LOGGER.info("D435 检测结果: view=%s YOLO目标=%s 二维抓取候选=%s", label, len(detected_items), len(grasps))
        if not detected_items:
            return _ViewAttempt(_ViewOutcome.RECOVERABLE_NO_GRASP, "D435 YOLO 未检测到目标")
        if not grasps:
            return _ViewAttempt(_ViewOutcome.RECOVERABLE_NO_GRASP, "D435 检测到目标，但未生成有效二维抓取候选")
        target_class = self.vision_config.get("detection", {}).get("target_class")
        best = select_best_grasp(grasps, target_class)
        if best is None:
            self.current_poses = None
            class_hint = f"，目标类别过滤为 {target_class}" if target_class else ""
            return _ViewAttempt(_ViewOutcome.RECOVERABLE_NO_GRASP, f"二维抓取候选均无效或不匹配{class_hint}")
        try:
            tcp_to_base = self.runtime.gripper_driver.get_tcp_pose()
            poses = transform_grasp_to_base(self.vision, best, tcp_to_base)
        except Exception as exc:
            reason = f"抓取位姿坐标变换失败: {exc}"
            self._grasp_failure(reason, kind=GraspPreparationFailureKind.TRANSFORM, exc_info=True)
            return _ViewAttempt(_ViewOutcome.FAILURE, reason)
        try:
            checker = self.ik_checker_factory(self.runtime) if self.ik_checker_factory else _IkChecker(self.runtime)
            ik_result = check_grasp_poses_ik(self.vision, poses, checker)
        except Exception as exc:
            reason = f"抓取位姿 IK 检查异常: {exc}"
            self._grasp_failure(reason, kind=GraspPreparationFailureKind.IK, exc_info=True)
            return _ViewAttempt(_ViewOutcome.FAILURE, reason)
        if not ik_result.feasible:
            self.current_poses = None
            errors = (ik_result.pre_error, ik_result.grasp_error, ik_result.retreat_error)
            if any(math.isinf(value) for value in errors):
                reason = "抓取位姿低于基座安全高度或 IK 前置检查拒绝"
            else:
                reason = (
                    "抓取位姿 IK 不可达: "
                    f"pregrasp误差={ik_result.pre_error:.6f}, "
                    f"grasp误差={ik_result.grasp_error:.6f}, "
                    f"retreat误差={ik_result.retreat_error:.6f}"
                )
            LOGGER.info("D435 视角无可行 IK，继续补偿扫描: view=%s reason=%s", label, reason)
            return _ViewAttempt(_ViewOutcome.RECOVERABLE_NO_GRASP, reason)
        self.current_poses = poses
        LOGGER.info("D435 抓取准备通过：view=%s，检测、坐标变换与 IK 均成功", label)
        return _ViewAttempt(_ViewOutcome.SUCCESS)

    @staticmethod
    def _view_label(index: int, view: D435RecoveryView) -> str:
        return (
            f"补偿{index}/6 "
            f"J1={math.degrees(view.joint1_offset_rad):+.1f}deg "
            f"J5={math.degrees(view.joint5_offset_rad):+.1f}deg"
        )

    def _run_d435_recovery_scan(self, first_rejection: str) -> bool:
        """Try six J1/J5 views after the centre view has no feasible grasp."""

        if self.runtime is None or self._scan_center_q is None:
            return self._grasp_failure(
                "D435 补偿扫描缺少 scan_pose0 中心关节目标",
                kind=GraspPreparationFailureKind.ARM_MOTION,
            )
        try:
            views = d435_recovery_views(self.poses_config)
            validate_d435_recovery_targets(self.runtime, self._scan_center_q, views, self.poses_config)
            settle_sec = d435_recovery_settle_sec(self.poses_config)
        except Exception as exc:
            return self._grasp_failure(
                f"D435 六视角补偿配置或关节限位校验失败: {exc}",
                kind=GraspPreparationFailureKind.ARM_MOTION,
                exc_info=True,
            )

        reasons = [f"中心视角: {first_rejection}"]
        for index, view in enumerate(views, start=1):
            label = self._view_label(index, view)
            LOGGER.info("D435 开始%s", label)
            try:
                moved = move_to_d435_recovery_view(
                    self.runtime,
                    self._scan_center_q,
                    view,
                    self.poses_config,
                    label=label,
                )
            except Exception as exc:
                return self._grasp_failure(
                    f"D435 {label} 关节运动异常: {exc}",
                    kind=GraspPreparationFailureKind.ARM_MOTION,
                    exc_info=True,
                )
            if not moved:
                return self._grasp_failure(
                    f"D435 {label} 关节轨迹未完成",
                    kind=GraspPreparationFailureKind.ARM_MOTION,
                )
            if settle_sec:
                time.sleep(settle_sec)
            attempt = self._detect_current_view(label)
            if attempt.outcome is _ViewOutcome.SUCCESS:
                return True
            if attempt.outcome is _ViewOutcome.FAILURE:
                return False
            reasons.append(f"{label}: {attempt.reason}")

        center = D435RecoveryView(0.0, 0.0)
        try:
            returned = move_to_d435_recovery_view(
                self.runtime,
                self._scan_center_q,
                center,
                self.poses_config,
                label="D435 补偿扫描结束回中心",
            )
        except Exception as exc:
            return self._grasp_failure(
                f"D435 六视角补偿结束后无法返回扫描中心: {exc}",
                kind=GraspPreparationFailureKind.ARM_MOTION,
                exc_info=True,
            )
        if not returned:
            return self._grasp_failure(
                "D435 六视角补偿结束后未能返回扫描中心",
                kind=GraspPreparationFailureKind.ARM_MOTION,
            )
        self.current_poses = None
        return self._grasp_failure(
            "D435 中心帧及六视角补偿均未获得可行抓取位姿: " + "; ".join(reasons),
            kind=GraspPreparationFailureKind.RECOVERABLE_NO_GRASP,
        )

    def prepare_grasp(self) -> bool:
        if self.runtime is None:
            raise RuntimeError("机械臂适配器尚未初始化")
        self.last_failure_reason = ""
        self.last_failure_kind = GraspPreparationFailureKind.NONE
        # D435 is eye-in-hand.  Its first frame must be captured from the
        # configured scan viewpoint, never from safe-home.
        if not self._move_to_detection_pose():
            return False
        LOGGER.info("D435 将从扫描姿势启动并采集中心首帧")
        if not self._start_grasp_vision():
            return self._grasp_failure(
                "D435 或 YOLO 抓取视觉初始化失败",
                kind=GraspPreparationFailureKind.VISION_RUNTIME,
            )
        initial = self._detect_current_view("中心 J1=+0.0deg J5=+0.0deg")
        if initial.outcome is _ViewOutcome.SUCCESS:
            return True
        if initial.outcome is _ViewOutcome.FAILURE:
            return False
        LOGGER.info("D435 中心视角未得到可行抓取位姿；开始六视角补偿: %s", initial.reason)
        return self._run_d435_recovery_scan(initial.reason)

    def _return_to_detection(self) -> bool:
        self.arm_at_detection_pose = False
        self._scan_center_q = None
        return self._move_to_detection_pose()

    def _reacquire(self) -> BaseGraspPoses | None:
        return self.current_poses if self.prepare_grasp() else None

    def grasp_current_target(self) -> bool:
        if self.runtime is None or self.current_poses is None:
            self.last_failure_reason = "未获得已通过 IK 的抓取位姿"
            self.last_failure_kind = GraspPreparationFailureKind.NONE
            return False
        self.arm_at_safe_home = False
        self.arm_at_detection_pose = False
        result = grasp_with_retry(
            self.runtime, self.current_poses, self.poses_config,
            max_attempts=self.max_attempts,
            reacquire_poses=self._reacquire,
            return_to_detection=self._return_to_detection,
        )
        if not result.success:
            self.last_failure_reason = f"抓取动作失败（尝试{result.attempts}次）: {result.last_error or '未知执行错误'}"
            LOGGER.error("%s", self.last_failure_reason)
        return result.success

    def hold(self) -> bool:
        self.arm_at_safe_home = False
        self.arm_at_detection_pose = False
        return self.runtime is not None and move_to_hold_pose(self.runtime, self.poses_config)

    def drop(self) -> bool:
        self.arm_at_safe_home = False
        self.arm_at_detection_pose = False
        return self.runtime is not None and drop_object(self.runtime, self.poses_config)

    def safe_home(self) -> bool:
        result = self.runtime is not None and safe_home(self.runtime)
        self.arm_at_safe_home = bool(result)
        if result:
            self.arm_at_detection_pose = False
            self._scan_center_q = None
        return result

    def emergency_stop(self) -> bool:
        return self.runtime is not None and emergency_stop(self.runtime)

    def release_vision(self) -> None:
        """Close D435 once a grasp phase is over, freeing the RGB-D bus."""

        if self.d435 is not None:
            try:
                stop_d435(self.d435)
            except Exception:
                LOGGER.exception("关闭 D435 失败")
            self.d435 = None
        self.vision = None
        self.current_poses = None
        self.arm_at_detection_pose = False
        self._scan_center_q = None

    def shutdown(self) -> None:
        self.release_vision()
        if self.runtime is not None:
            try:
                stop_arm_runtime(self.runtime)
            except Exception:
                pass
            self.runtime = None
        self.vision = None
        self.current_poses = None
        self.arm_at_safe_home = False
        self.arm_at_detection_pose = False
        self._scan_center_q = None


__all__ = ["ArmTaskAdapter"]
