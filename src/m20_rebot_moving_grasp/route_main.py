"""路线巡航拾取入口，与定点扫描闭环 Demo 完全分离。"""

from __future__ import annotations

import argparse
import logging
import math
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import main as demo_main
from .arm.arm_task_adapter import ArmTaskAdapter
from .common.config import load_config
from .machine_dog import M20Config, M20DogAdapter
from .machine_dog.d455_approach import D455ApproachController
from .machine_dog.d455_vision import D455Config
from .machine_dog.m20_protocol import UdpJsonTransport
from .machine_dog.route_pickup import RoutePickupController, load_route_definition


PROJECT_ROOT = demo_main.PROJECT_ROOT
LOGGER = logging.getLogger("m20_rebot_moving_grasp.route")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M20 Pro 路线巡航中检测、抓取与恢复")
    parser.add_argument("--run-real", action="store_true", help="允许连接并控制真实设备")
    parser.add_argument("--dry-run", action="store_true", help="只执行路线与依赖检查")
    parser.add_argument("--route-name", required=True, help="config/m20/routes.json 中的路线名称")
    parser.add_argument("--route-config", default=str(PROJECT_ROOT / "config/m20/routes.json"))
    parser.add_argument("--route-points-file", default=str(PROJECT_ROOT / "config/m20/route_points.json"))
    parser.add_argument("--route-loops", type=int, default=1)
    parser.add_argument("--route-nav-timeout", type=float, default=300.0)
    parser.add_argument("--pose-interval", type=float, default=1.0)
    parser.add_argument("--pickup-settle-seconds", type=float, default=5.0)
    parser.add_argument("--bin-settle-seconds", type=float, default=5.0)
    parser.add_argument("--pose-log", default=None)
    parser.add_argument("--resume-log", default=None)
    parser.add_argument("--system-config", default=str(PROJECT_ROOT / "config/system.yaml"))
    parser.add_argument("--arm-config", default=str(PROJECT_ROOT / "config/arm.yaml"))
    parser.add_argument("--vision-config", default=str(PROJECT_ROOT / "config/vision.yaml"))
    parser.add_argument("--poses-config", default=str(PROJECT_ROOT / "config/poses.yaml"))
    parser.add_argument("--d455-config", default=str(PROJECT_ROOT / "config/d455.yaml"))
    parser.add_argument("--log-level", default="INFO")
    return parser


def _validate_route_args(args: argparse.Namespace, m20: M20Config) -> list[str]:
    errors: list[str] = []
    if args.route_loops < 1:
        errors.append("--route-loops 必须至少为1")
    for name, value in (("--route-nav-timeout", args.route_nav_timeout),
                        ("--pose-interval", args.pose_interval),
                        ("--pickup-settle-seconds", args.pickup_settle_seconds),
                        ("--bin-settle-seconds", args.bin_settle_seconds)):
        settle = name in ("--pickup-settle-seconds", "--bin-settle-seconds")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0 or (not settle and float(value) <= 0.0):
            errors.append(f"{name} 必须是大于0的有限数字")
    try:
        load_route_definition(
            args.route_config,
            args.route_points_file,
            args.route_name,
            required_point_names=(m20.bin_point_name, m20.home_point_name),
        )
    except (OSError, ValueError) as exc:
        errors.append(f"巡航路线配置无效: {exc}")
    return errors


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    system, arm_cfg, vision_cfg, poses_cfg = demo_main._load_configs(args)
    m20_cfg = replace(M20Config.from_env(system_config=system).validate(), points_file=args.route_points_file).validate()
    errors = demo_main._preflight(
        system,
        arm_cfg,
        vision_cfg,
        points_override=args.route_points_file,
        required_point_names=(),
    )
    errors.extend(_validate_route_args(args, m20_cfg))
    if errors:
        for error in errors:
            LOGGER.error(error)
        return 2
    if args.dry_run or not args.run_real:
        LOGGER.info("巡航路线、配置和依赖检查通过；未连接真实设备。使用 --run-real 才会执行任务。")
        return 0

    d455_errors = demo_main._validate_builtin_d455(args)
    if d455_errors:
        for error in d455_errors:
            LOGGER.error(error)
        return 2

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
        LOGGER.warning("收到信号 %s，巡航将在当前安全边界停止", signum)
        stop_requested["value"] = True

    old_int = signal.signal(signal.SIGINT, request_stop)
    old_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        route_config: dict[str, Any] = {
            "route_loops": args.route_loops,
            "route_nav_timeout_sec": args.route_nav_timeout,
            "pose_interval_sec": args.pose_interval,
            "pickup_settle_seconds": args.pickup_settle_seconds,
            "bin_settle_seconds": args.bin_settle_seconds,
            "pose_log": args.pose_log or str(PROJECT_ROOT / "runs" / "route" / f"{args.route_name}_poses.jsonl"),
            "resume_log": args.resume_log or str(PROJECT_ROOT / "runs" / "route" / f"{args.route_name}_resume.json"),
            "missed_detection_log": str(PROJECT_ROOT / "runs" / "route" / f"{args.route_name}_d435_misses.jsonl"),
            "d435_miss_suppression_radius_m": float(system["task"]["d435_miss_suppression_radius_m"]),
            "resume_point_id": 902,
        }
        route_runner = RoutePickupController(
            route_config,
            m20_cfg,
            transport,
            d455,
            routes_path=args.route_config,
            points_path=args.route_points_file,
            route_name=args.route_name,
            stop_requested=lambda: stop_requested["value"],
        )
        success = route_runner.run(dog, arm)
        LOGGER.info("巡航任务结束: route=%s success=%s error=%s", args.route_name, success,
                    route_runner.last_failure_reason or "无")
        if stop_requested["value"]:
            return 0 if route_runner.safe_stop_completed else 1
        return 0 if success else 1
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run_real and args.dry_run:
        _parser().error("--run-real 与 --dry-run 不能同时使用")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
