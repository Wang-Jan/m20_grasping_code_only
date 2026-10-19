"""YAML 配置文件读取工具。

本模块刻意只负责“读取并保证基础结构正确”。
例如 D435 序列号是否存在、手眼标定文件是否存在、机械臂位姿是否可达，
会在对应的硬件初始化或运动函数中检查，而不是在这里触发硬件访问。
"""

from __future__ import annotations

import ipaddress
import math
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """配置文件存在、格式或顶层结构不符合要求时抛出。"""


def load_config(path: str | Path) -> dict[str, Any]:
    """读取一个 YAML 配置文件并返回顶层字典。

    参数:
        path: YAML 文件路径；支持 ``~``，可以是相对路径或绝对路径。

    返回:
        YAML 顶层 mapping 对应的普通 Python 字典。

    异常:
        FileNotFoundError: 指定文件不存在或不是普通文件。
        ConfigError: YAML 语法错误、文件为空，或顶层不是 mapping。

    说明:
        本函数不合并多份配置，也不验证具体业务字段。这样每个后续模块
        都能明确声明自己需要的字段，并将硬件相关错误留在恰当的阶段处理。
    """

    config_path = Path(path).expanduser()

    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在或不是普通文件: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 配置语法错误: {config_path}") from exc

    if data is None:
        raise ConfigError(f"配置文件为空: {config_path}")

    if not isinstance(data, dict):
        raise ConfigError(
            f"配置文件顶层必须是 mapping/dict，实际为: {type(data).__name__}"
        )

    return data


def validate_config(config: dict[str, Any], config_kind: str) -> dict[str, Any]:
    """验证项目配置的字段、类型和数值范围并返回原字典。

    ``config_kind`` 可取 ``system``、``arm``、``vision`` 或 ``poses``。
    本函数不检查硬件是否在线，也不检查模型、SDK 或手眼标定文件是否存在；
    这些属于对应运行时的职责。
    """

    if not isinstance(config, dict):
        raise ConfigError("待验证配置必须是 mapping/dict")
    if config_kind not in {"system", "arm", "vision", "poses"}:
        raise ConfigError(f"未知配置类型: {config_kind}")

    def mapping(parent: dict[str, Any], key: str, path: str) -> dict[str, Any]:
        value = parent.get(key)
        if not isinstance(value, dict):
            raise ConfigError(f"{path} 必须是 mapping")
        return value

    def required(parent: dict[str, Any], keys: tuple[str, ...], path: str) -> None:
        missing = [key for key in keys if key not in parent]
        if missing:
            raise ConfigError(f"{path} 缺少字段: {', '.join(missing)}")

    def positive(value: Any, path: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path} 必须是数字") from exc
        if not math.isfinite(number) or number <= 0.0:
            raise ConfigError(f"{path} 必须是大于0的有限数字")
        return number

    def nonnegative(value: Any, path: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path} 必须是数字") from exc
        if not math.isfinite(number) or number < 0.0:
            raise ConfigError(f"{path} 必须是非负有限数字")
        return number

    def ratio(value: Any, path: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path} 必须是数字") from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ConfigError(f"{path} 必须位于[0, 1]")
        return number

    if config_kind == "system":
        network = mapping(config, "network", "network")
        m20 = mapping(config, "m20", "m20")
        task = mapping(config, "task", "task")
        required(network, ("subnet", "router_management_ip", "orin_ip"), "network")
        required(m20, ("aos_ip", "gos_ip", "nos_ip", "basic_server_tcp_port"), "m20")
        required(task, ("detect_timeout_sec", "clean_scan_continuous_no_target_sec", "max_grasp_attempts",
                        "d435_miss_suppression_radius_m"), "task")
        try:
            ipaddress.ip_network(str(network["subnet"]), strict=False)
            for key in ("router_management_ip", "orin_ip"):
                ipaddress.ip_address(str(network[key]))
            for key in ("aos_ip", "gos_ip", "nos_ip"):
                ipaddress.ip_address(str(m20[key]))
        except ValueError as exc:
            raise ConfigError(f"system 配置包含非法IP或网段: {exc}") from exc
        port = int(m20["basic_server_tcp_port"])
        if not 1 <= port <= 65535:
            raise ConfigError("m20.basic_server_tcp_port 必须位于1~65535")
        positive(task["detect_timeout_sec"], "task.detect_timeout_sec")
        positive(task["clean_scan_continuous_no_target_sec"], "task.clean_scan_continuous_no_target_sec")
        positive(task["d435_miss_suppression_radius_m"], "task.d435_miss_suppression_radius_m")
        if int(task["max_grasp_attempts"]) < 1:
            raise ConfigError("task.max_grasp_attempts 必须至少为1")

    elif config_kind == "arm":
        robot = mapping(config, "robot", "robot")
        motion = mapping(config, "arm_motion", "arm_motion")
        required(robot, ("repo_root", "grasp_repo_root", "gripper"), "robot")
        required(motion, ("pregrasp_duration_sec", "grasp_duration_sec", "retreat_duration_sec", "default_motion_timeout_sec"), "arm_motion")
        for key in ("repo_root", "grasp_repo_root"):
            if not str(robot[key]).strip():
                raise ConfigError(f"robot.{key} 不能为空")
        gripper = mapping(robot, "gripper", "robot.gripper")
        dm = mapping(gripper, "dm", "robot.gripper.dm")
        required(dm, ("angle_open", "counterclockwise", "tau_max", "close_torque", "default_force"), "robot.gripper.dm")
        positive(dm["angle_open"], "robot.gripper.dm.angle_open")
        tau_max = positive(dm["tau_max"], "robot.gripper.dm.tau_max")
        close_torque = positive(dm["close_torque"], "robot.gripper.dm.close_torque")
        default_force = positive(dm["default_force"], "robot.gripper.dm.default_force")
        if close_torque > tau_max or default_force > tau_max:
            raise ConfigError("robot.gripper.dm 的闭合/默认力矩不能超过 tau_max")
        for key in ("pregrasp_duration_sec", "grasp_duration_sec", "retreat_duration_sec", "default_motion_timeout_sec"):
            positive(motion[key], f"arm_motion.{key}")

    elif config_kind == "vision":
        camera = mapping(config, "camera", "camera")
        calibration = mapping(config, "calibration", "calibration")
        detection = mapping(config, "detection", "detection")
        grasp = mapping(config, "grasp_2d", "grasp_2d")
        required(camera, ("type", "serial", "color_width", "color_height", "depth_width", "depth_height", "fps"), "camera")
        required(calibration, ("hand_eye_path", "hand_eye_mode"), "calibration")
        required(detection, ("model_name", "device", "confidence_threshold", "iou_threshold", "infer_every_live"), "detection")
        required(grasp, ("depth_quantile", "pregrasp_offset_m", "retreat_offset_m", "insertion_depth_m", "min_base_z_m"), "grasp_2d")
        if str(camera["type"]).lower() != "realsense_d435cb":
            raise ConfigError("当前 vision.camera.type 必须为 realsense_d435cb")
        if not str(camera["serial"]).strip():
            raise ConfigError("camera.serial 不能为空")
        for key in ("color_width", "color_height", "depth_width", "depth_height", "fps"):
            if int(camera[key]) <= 0:
                raise ConfigError(f"camera.{key} 必须大于0")
        if (int(camera["color_width"]), int(camera["color_height"])) != (int(camera["depth_width"]), int(camera["depth_height"])):
            raise ConfigError("camera 彩色和深度分辨率必须相同")
        if not str(calibration["hand_eye_path"]).strip():
            raise ConfigError("calibration.hand_eye_path 不能为空")
        ratio(detection["confidence_threshold"], "detection.confidence_threshold")
        ratio(detection["iou_threshold"], "detection.iou_threshold")
        if int(detection["infer_every_live"]) < 1:
            raise ConfigError("detection.infer_every_live 必须至少为1")
        ratio(grasp["depth_quantile"], "grasp_2d.depth_quantile")
        nonnegative(grasp["pregrasp_offset_m"], "grasp_2d.pregrasp_offset_m")
        nonnegative(grasp["retreat_offset_m"], "grasp_2d.retreat_offset_m")
        nonnegative(grasp["insertion_depth_m"], "grasp_2d.insertion_depth_m")
        try:
            if not math.isfinite(float(grasp["min_base_z_m"])):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ConfigError("grasp_2d.min_base_z_m 必须是有限数字") from exc

    elif config_kind == "poses":
        ready = mapping(config, "ready_pose", "ready_pose")
        scan = mapping(config, "scan_pose0", "scan_pose0")
        recovery = mapping(config, "d435_recovery_scan", "d435_recovery_scan")
        transport = mapping(config, "transport_pose", "transport_pose")
        drop = mapping(config, "pre_drop_pose", "pre_drop_pose")
        required(ready, ("x", "y", "z", "roll", "pitch", "yaw", "duration_sec"), "ready_pose")
        required(scan, ("joint1_offset_rad", "duration_sec", "joint1_speed_limit_rad_s"), "scan_pose0")
        required(recovery, ("views", "duration_sec", "joint_speed_limit_rad_s", "settle_sec", "joint_limit_margin_rad",
                            "joint_feedback_tolerance_rad", "feedback_timeout_sec", "feedback_retry_count"),
                 "d435_recovery_scan")
        required(transport, ("joint_positions_rad", "duration_sec"), "transport_pose")
        required(drop, ("x", "y", "z", "roll", "pitch", "yaw"), "pre_drop_pose")
        if "duration" not in drop and "duration_sec" not in drop:
            raise ConfigError("pre_drop_pose 缺少字段: duration")
        positive(ready["duration_sec"], "ready_pose.duration_sec")
        positive(scan["duration_sec"], "scan_pose0.duration_sec")
        positive(scan["joint1_speed_limit_rad_s"], "scan_pose0.joint1_speed_limit_rad_s")
        positive(recovery["duration_sec"], "d435_recovery_scan.duration_sec")
        positive(recovery["joint_speed_limit_rad_s"], "d435_recovery_scan.joint_speed_limit_rad_s")
        nonnegative(recovery["settle_sec"], "d435_recovery_scan.settle_sec")
        positive(recovery["joint_limit_margin_rad"], "d435_recovery_scan.joint_limit_margin_rad")
        positive(recovery["joint_feedback_tolerance_rad"], "d435_recovery_scan.joint_feedback_tolerance_rad")
        positive(recovery["feedback_timeout_sec"], "d435_recovery_scan.feedback_timeout_sec")
        try:
            retry_count_raw = float(recovery["feedback_retry_count"])
            retry_count = int(retry_count_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigError("d435_recovery_scan.feedback_retry_count 必须是整数") from exc
        if not math.isfinite(retry_count_raw) or retry_count_raw != retry_count or retry_count not in (0, 1):
            raise ConfigError("d435_recovery_scan.feedback_retry_count 必须为 0 或 1")
        views = recovery["views"]
        if not isinstance(views, list) or len(views) != 6:
            raise ConfigError("d435_recovery_scan.views 必须包含6个视角")
        unique_views: set[tuple[float, float]] = set()
        for index, view in enumerate(views):
            if not isinstance(view, dict):
                raise ConfigError(f"d435_recovery_scan.views[{index}] 必须是 mapping")
            required(view, ("joint1_offset_rad", "joint5_offset_rad"), f"d435_recovery_scan.views[{index}]")
            try:
                pair = (float(view["joint1_offset_rad"]), float(view["joint5_offset_rad"]))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"d435_recovery_scan.views[{index}] 偏移必须是数字") from exc
            if not all(math.isfinite(value) for value in pair):
                raise ConfigError(f"d435_recovery_scan.views[{index}] 偏移必须是有限数字")
            unique_views.add(pair)
        if len(unique_views) != 6:
            raise ConfigError("d435_recovery_scan.views 不能包含重复视角")
        positive(transport["duration_sec"], "transport_pose.duration_sec")
        # ``duration`` 是真机验证版 poses.yaml 的原始字段名；保留
        # ``duration_sec`` 以兼容本工程此前生成的配置文件。
        drop_duration_key = "duration" if "duration" in drop else "duration_sec"
        positive(drop[drop_duration_key], f"pre_drop_pose.{drop_duration_key}")
        if not isinstance(transport["joint_positions_rad"], list) or len(transport["joint_positions_rad"]) != 6:
            raise ConfigError("transport_pose.joint_positions_rad 必须包含6个关节角")
        for pose_name, pose in (("ready_pose", ready), ("pre_drop_pose", drop)):
            for key in ("x", "y", "z", "roll", "pitch", "yaw"):
                try:
                    if not math.isfinite(float(pose[key])):
                        raise ValueError
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"{pose_name}.{key} 必须是有限数字") from exc
        seed = drop.get("ik_seed_joint_positions_rad")
        if seed is not None:
            if not isinstance(seed, list) or len(seed) != 6:
                raise ConfigError("pre_drop_pose.ik_seed_joint_positions_rad 必须包含6个关节角")
            for index, value in enumerate(seed):
                try:
                    if not math.isfinite(float(value)):
                        raise ValueError
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"pre_drop_pose.ik_seed_joint_positions_rad[{index}] 必须是有限数字"
                    ) from exc

    return config
