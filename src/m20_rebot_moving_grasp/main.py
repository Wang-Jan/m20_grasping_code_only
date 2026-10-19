"""M20 Pro + DM-B601 联合任务入口。"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import math
import signal
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .arm.arm_task_adapter import ArmTaskAdapter
from .common.config import load_config, validate_config
from .machine_dog import M20Config, M20DogAdapter
from .machine_dog.d455_approach import D455ApproachController
from .machine_dog.d455_vision import D455Config
from .machine_dog.m20_protocol import UdpJsonTransport
from .orchestrator.main_flow import TaskState, run_task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("m20_rebot_moving_grasp")


def _resolve_project_path(value: str | Path) -> Path:
    """解析工程路径，兼容开发机和 Jetson 的 third_party 布局。"""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # Jetson 部署时 third_party 位于工程内；开发机保留在工程同级。
    for candidate in (PROJECT_ROOT / path, PROJECT_ROOT.parent / path):
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M20 Pro + DM-B601 自动移动抓取")
    parser.add_argument("--run-real", action="store_true", help="允许连接并控制真实设备")
    parser.add_argument("--dry-run", action="store_true", help="只执行配置和依赖检查")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--system-config", default=str(PROJECT_ROOT / "config/system.yaml"))
    parser.add_argument("--arm-config", default=str(PROJECT_ROOT / "config/arm.yaml"))
    parser.add_argument("--vision-config", default=str(PROJECT_ROOT / "config/vision.yaml"))
    parser.add_argument("--poses-config", default=str(PROJECT_ROOT / "config/poses.yaml"))
    parser.add_argument("--points-file", default=None)
    parser.add_argument("--d455-config", default=str(PROJECT_ROOT / "config/d455.yaml"), help="D455 视觉配置")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _load_configs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    system = validate_config(load_config(args.system_config), "system")
    arm = validate_config(load_config(args.arm_config), "arm")
    vision = validate_config(load_config(args.vision_config), "vision")
    poses = validate_config(load_config(args.poses_config), "poses")
    return system, arm, vision, poses


def _preflight(
    system: dict[str, Any],
    arm: dict[str, Any],
    vision: dict[str, Any],
    *,
    points_override: str | None = None,
    required_point_names: tuple[str, ...] | None = None,
) -> list[str]:
    errors: list[str] = []
    network = system["network"]
    try:
        ipaddress.ip_address(str(network["orin_ip"]))
        ipaddress.ip_address(str(system["m20"]["aos_ip"]))
    except ValueError as exc:
        errors.append(f"网络地址无效: {exc}")
    robot = arm["robot"]
    for key in ("repo_root", "grasp_repo_root"):
        if not _resolve_project_path(str(robot[key])).is_dir():
            errors.append(f"目录不存在: {robot[key]}")
    points_path = Path(points_override or system["m20"].get("points_file", "")).expanduser()
    if not points_path.is_absolute():
        points_path = PROJECT_ROOT / points_path
    if not points_path.is_file():
        errors.append(f"点位文件不存在: {points_path}")
    else:
        try:
            import json
            points = json.loads(points_path.read_text(encoding="utf-8"))
            if required_point_names is None:
                m20 = system["m20"]
                required_point_names = (
                    str(m20.get("scan_point_name", "scan_point")),
                    str(m20.get("bin_point_name", "bin_point")),
                    str(m20.get("home_point_name", "start")),
                )
            for name in required_point_names:
                point = points.get(name) if isinstance(points, dict) else None
                if not isinstance(point, dict):
                    errors.append(f"点位文件缺少命名点: {name}")
                    continue
                try:
                    coordinates = tuple(float(point[key]) for key in ("x", "y", "z", "yaw"))
                except (KeyError, TypeError, ValueError):
                    errors.append(f"点位 {name} 缺少有效 x/y/z/yaw 坐标")
                    continue
                if not all(math.isfinite(value) for value in coordinates):
                    errors.append(f"点位 {name} 包含非有限坐标")
        except (OSError, ValueError) as exc:
            errors.append(f"点位文件不可读: {exc}")
    hand_eye = Path(str(vision["calibration"]["hand_eye_path"])).expanduser()
    if not hand_eye.is_absolute():
        hand_eye = PROJECT_ROOT / hand_eye
    if not hand_eye.is_file():
        errors.append(f"D435 手眼标定文件不存在: {hand_eye}")
    model_name = str(vision["detection"].get("model_name", ""))
    model_candidates = [Path(model_name).expanduser()]
    if not Path(model_name).is_absolute():
        grasp_root = _resolve_project_path(str(robot["grasp_repo_root"]))
        model_candidates.extend((
            PROJECT_ROOT / model_name,
            PROJECT_ROOT / "models" / model_name,
            grasp_root / "models" / model_name,
        ))
    if model_name and not any(path.is_file() for path in model_candidates):
        errors.append(f"YOLO 模型文件不存在: {model_name}")
    return errors


def _validate_builtin_d455(args: argparse.Namespace) -> list[str]:
    """真机启动前校验内置 D455 回调的静态依赖，不触碰硬件。"""
    errors: list[str] = []
    try:
        cfg = D455Config.from_mapping(load_config(args.d455_config))
        model_path = Path(cfg.model_name).expanduser()
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / "models" / model_path
        if not model_path.is_file():
            errors.append(f"D455 YOLO 模型文件不存在: {model_path}")
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"D455 配置无效: {exc}")
    return errors


def _operator_wait(stop_requested: dict[str, bool]) -> str:
    while not stop_requested["value"]:
        try:
            command = input("抓取失败：输入 resume 重新扫描，输入 abort 退出：").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "abort"
        if command in ("resume", "abort"):
            return command
    return "abort"


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    system, arm_cfg, vision_cfg, poses_cfg = _load_configs(args)
    errors = _preflight(system, arm_cfg, vision_cfg, points_override=args.points_file)
    if errors:
        for error in errors:
            LOGGER.error(error)
        return 2
    if args.dry_run or not args.run_real:
        LOGGER.info("配置和依赖检查通过；未连接真实设备。使用 --run-real 才会执行任务。")
        return 0

    d455_errors = _validate_builtin_d455(args)
    if d455_errors:
        for error in d455_errors:
            LOGGER.error(error)
        return 2

    # ``from_env`` accepts the loaded system mapping as a keyword-only
    # argument; passing it positionally causes real-mode startup to fail
    # before any device is initialized.
    m20_cfg = M20Config.from_env(system_config=system).validate()
    if args.points_file:
        m20_cfg = replace(m20_cfg, points_file=args.points_file).validate()
    d455_cfg = D455Config.from_mapping(load_config(args.d455_config))
    stop_requested = {"value": False}
    transport = UdpJsonTransport(m20_cfg.aos_ip, m20_cfg.udp_port)
    d455 = D455ApproachController(
        d455_cfg,
        m20_cfg,
        transport,
        stop_requested=lambda: stop_requested["value"],
    )
    dog = M20DogAdapter(
        m20_cfg,
        transport=transport,
        approach_handler=d455.approach,
        scan_handler=d455.scan_handler,
        scan_start=d455.scan_start,
        scan_stop=d455.scan_stop,
    )
    arm = ArmTaskAdapter(arm_cfg, vision_cfg, poses_cfg, max_attempts=int(system["task"]["max_grasp_attempts"]))

    def request_stop(signum, _frame):
        LOGGER.warning("收到信号 %s，任务将在当前安全边界停止", signum)
        stop_requested["value"] = True

    old_int = signal.signal(signal.SIGINT, request_stop)
    old_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        result = run_task(
            dog,
            arm,
            max_cycles=args.max_cycles,
            operator_wait=lambda: _operator_wait(stop_requested),
            stop_requested=lambda: stop_requested["value"],
        )
        LOGGER.info("任务结束: state=%s cycles=%s success=%s error=%s", result.state, result.cycles, result.success, result.error)
        return 0 if result.success else 1
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run_real and args.dry_run:
        _parser().error("--run-real 与 --dry-run 不能同时使用")
    if args.max_cycles is not None and args.max_cycles < 1:
        _parser().error("--max-cycles 必须大于0")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
