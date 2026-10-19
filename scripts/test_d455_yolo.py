#!/usr/bin/env python3
"""仅测试 D455 + YOLOE，不连接 M20、机械臂或 D435。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from m20_rebot_moving_grasp.common.config import load_config
from m20_rebot_moving_grasp.machine_dog.d455_bridge import load_callable, transform_point
from m20_rebot_moving_grasp.machine_dog.d455_vision import D455Config, D455Vision, choose_nearest_detection


def main() -> int:
    parser = argparse.ArgumentParser(description="D455 单帧 YOLOE 检测测试（不控制机器狗）")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/d455.yaml"))
    parser.add_argument("--transform", default=None, help="可选：D455 到地图坐标函数，module:function")
    args = parser.parse_args()

    config = D455Config.from_mapping(load_config(args.config))
    vision = D455Vision(config)
    try:
        vision.open()
        detections = vision.detect_once()
    finally:
        vision.close()

    target = choose_nearest_detection(detections)
    if target is None:
        print("未检测到满足类别、置信度和深度范围的目标")
        return 1
    print(f"目标: {target.class_name}; 置信度: {target.confidence:.3f}; 深度: {target.depth_m:.3f} m")
    print(f"D455 相机坐标 (m): {target.point_camera_m}")
    if args.transform:
        map_point = transform_point(load_callable(args.transform, "--transform"), target.point_camera_m)
        print(f"地图坐标 (m): {map_point}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
