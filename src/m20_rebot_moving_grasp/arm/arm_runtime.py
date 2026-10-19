"""DM-B601 运行时对象的创建。

本模块当前只负责构造 SDK 对象，不连接串口、不使能电机、不启动控制循环。
后续的 ``start_arm_runtime`` 函数才会执行真实硬件初始化。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_external_root(value: str | Path) -> Path:
    """解析外部源码根目录，支持两种受支持的工程布局。"""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Jetson: <project>/third_party；开发机: <workspace>/third_party。
    for candidate in (PROJECT_ROOT / path, PROJECT_ROOT.parent / path):
        if candidate.is_dir():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


@dataclass
class ArmRuntime:
    """机械臂运行时的对象句柄。

    ``rebotarm``、``controller`` 和 ``gripper_driver`` 分别对应原项目中
    ``RebotArm``、``RebotArmEndPose`` 和已经真机验证过的 ``GraspDriver``。
    创建对象不等于已经连接硬件；当前阶段三者都尚未启动。
    """

    rebotarm: Any
    controller: Any
    gripper_driver: Any
    sdk_root: Path
    grasp_root: Path
    arm_control_mode: str


def create_arm_runtime(config: dict[str, Any]) -> ArmRuntime:
    """根据 ``arm.yaml`` 创建 DM-B601 的 SDK 运行时对象。

    所执行的步骤：

    1. 检查 SDK 和已验证抓取代码目录；
    2. 将两个外部源码根目录加入当前进程的导入路径；
    3. 根据 SDK 硬件配置确认当前是 DM 机械臂；
    4. 创建 ``RebotArm``、``RebotArmEndPose`` 和 ``GraspDriver``；
    5. 返回对象句柄，不调用 ``connect``、``enable``、``start`` 或任何运动函数。

    这样可以在不碰真机的情况下先验证工程配置和对象依赖关系。
    """

    if not isinstance(config, dict):
        raise TypeError("arm 配置必须是 dict")

    robot_config = config.get("robot")
    if not isinstance(robot_config, dict):
        raise ValueError("arm 配置缺少 robot mapping")

    sdk_value = robot_config.get("repo_root")
    grasp_value = robot_config.get("grasp_repo_root")
    if not sdk_value:
        raise ValueError("arm.robot.repo_root 未配置")
    if not grasp_value:
        raise ValueError("arm.robot.grasp_repo_root 未配置")

    sdk_root = _resolve_external_root(str(sdk_value))
    grasp_root = _resolve_external_root(str(grasp_value))

    if not sdk_root.is_dir():
        raise FileNotFoundError(f"DM SDK 目录不存在: {sdk_root}")
    if not grasp_root.is_dir():
        raise FileNotFoundError(f"已验证抓取代码目录不存在: {grasp_root}")

    # mobile_2d_grasp.py 也是通过源码根目录复用这两个包。
    # 插入前移除重复项，避免同一进程中导入到不同版本。
    for import_root in (grasp_root, sdk_root):
        import_path = str(import_root)
        while import_path in sys.path:
            sys.path.remove(import_path)
        sys.path.insert(0, import_path)

    try:
        from drivers.robot.grasp_driver import (  # type: ignore[import-not-found]
            GraspDriver,
            selected_arm_config,
            selected_hardware_yaml,
        )
        from reBotArm_control_py.actuator import RebotArm  # type: ignore[import-not-found]
        from reBotArm_control_py.controllers import RebotArmEndPose  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "无法导入 DM SDK 或 GraspDriver。请在当前 conda 环境安装原项目依赖，"
            "尤其是 motorbridge；本函数尚未连接任何硬件。"
        ) from exc

    selected = selected_arm_config(str(sdk_root))
    if selected.arm_type != "dm":
        raise ValueError(
            f"当前工程要求 DM 机械臂，检测到硬件类型: {selected.arm_type}"
        )

    hardware_yaml = selected_hardware_yaml(str(sdk_root))
    rebotarm = RebotArm(hw_yaml=str(hardware_yaml))
    controller = RebotArmEndPose(
        rebotarm,
        arm_control_mode=selected.controller_mode,
    )
    gripper_driver = GraspDriver(
        rebotarm,
        controller,
        gripper_config=robot_config.get("gripper", {}),
        repo_root=str(sdk_root),
    )

    return ArmRuntime(
        rebotarm=rebotarm,
        controller=controller,
        gripper_driver=gripper_driver,
        sdk_root=sdk_root,
        grasp_root=grasp_root,
        arm_control_mode=selected.controller_mode,
    )


def start_arm_runtime(runtime: ArmRuntime) -> None:
    """连接并启动已经构造的 DM-B601 运行时。

    本函数直接复用 ``GraspDriver.start()``，因此启动次序与已经真机验证的
    ``mobile_2d_grasp.py`` 一致：连接 DM SDK、使 J1--J6 进入 ``pos_vel``、
    使 J7 进入 MIT、使能两个 group，并启动共享控制循环。

    它不会移动机械臂、不会打开或闭合夹爪。SDK 控制循环会首先以当前反馈关节角
    作为保持目标，防止启动时突跳。

    调用已经启动的运行时是幂等的，不会启动第二个控制循环。
    """

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")

    if getattr(runtime.controller, "_running", False):
        return

    try:
        runtime.gripper_driver.start()
    except Exception as exc:
        raise RuntimeError("DM-B601 启动失败：GraspDriver.start() 未完成") from exc

    if not getattr(runtime.rebotarm, "_connected", False):
        raise RuntimeError("DM-B601 启动失败：SDK 未建立连接")

    if not getattr(runtime.controller, "_running", False):
        raise RuntimeError("DM-B601 启动失败：控制循环未运行")


def stop_arm_runtime(runtime: ArmRuntime) -> None:
    """停止控制循环并断开 SDK，但不调用 ``safe_home``。

    ``RebotArmEndPose.end()`` 会先执行 ``safe_home``，因此不能用于普通的
    资源清理或初始化回滚。本函数先停止末端轨迹发送线程，再停止 SDK 共享
    控制循环，最后由 ``RebotArm.disconnect()`` 完成失能和串口释放。
    """

    if not isinstance(runtime, ArmRuntime):
        raise TypeError("runtime 必须是 ArmRuntime")

    controller = runtime.controller
    rebotarm = runtime.rebotarm
    errors: list[Exception] = []

    # 末端轨迹线程与SDK共享控制循环是两个不同的线程，必须先停止前者。
    sender = getattr(controller, "_send_thread", None)
    stop_event = getattr(controller, "_stop_send", None)
    if sender is not None and sender.is_alive():
        if stop_event is not None:
            stop_event.set()
        try:
            sender.join(timeout=5.0)
            if sender.is_alive():
                errors.append(RuntimeError("末端轨迹发送线程未在5秒内停止"))
        except Exception as exc:
            errors.append(exc)

    try:
        stop_control_loop = getattr(rebotarm, "stop_control_loop", None)
        if callable(stop_control_loop):
            stop_control_loop()
    except Exception as exc:
        errors.append(exc)

    try:
        disconnect = getattr(rebotarm, "disconnect", None)
        if callable(disconnect):
            disconnect()
    except Exception as exc:
        errors.append(exc)
    finally:
        # 防止后续代码把已经断开的运行时误认为仍在运行。
        controller._running = False
        if stop_event is not None:
            stop_event.clear()

    if errors:
        raise RuntimeError("DM-B601 停止时出现错误") from errors[0]
