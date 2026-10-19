#!/usr/bin/env python3
"""一键检查 D455 彩色/深度流和 YOLOE。

这是手动真机脚本，刻意不以 ``test_`` 开头，避免 unittest 自动发现时打开相机。
它绝不连接 M20、D435 或机械臂，也不会发送任何导航命令。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from m20_rebot_moving_grasp.common.config import ConfigError, load_config
from m20_rebot_moving_grasp.machine_dog.d455_vision import D455Config, D455Vision


def main() -> int:
    try:
        config = D455Config.from_mapping(load_config(PROJECT_ROOT / "config/d455.yaml"))
    except (FileNotFoundError, ConfigError) as exc:
        print(f"D455 配置无效：{exc}", file=sys.stderr)
        print("请先在 config/d455.yaml 的 camera.serial 填写 D455 序列号。", file=sys.stderr)
        return 2
    output_dir = PROJECT_ROOT / "tests/output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"d455_yolo_{stamp}.jpg"
    report_path = output_dir / f"d455_yolo_{stamp}.json"

    print("仅测试 D455 + YOLOE；不会连接或控制机器狗、机械臂、D435。")
    print(f"D455 serial: {config.serial}")
    print(f"模型: {config.model_name}; 类别: {list(config.custom_classes)}")

    vision = D455Vision(config)
    try:
        vision.open()
        vision.warm_up(30)
        detections = vision.detect_once()
        if vision.last_color_bgr is None or vision.last_depth_m is None:
            raise RuntimeError("未获得有效的彩色和深度帧")
        valid_depth = vision.last_depth_m[vision.last_depth_m > 0]
        result = vision.last_result
        annotated = result.plot() if result is not None and hasattr(result, "plot") else vision.last_color_bgr
        if not cv2.imwrite(str(image_path), annotated):
            raise RuntimeError(f"无法写入结果图: {image_path}")
    finally:
        vision.close()

    report = {
        "color_shape": list(vision.last_color_bgr.shape),
        "depth_shape": list(vision.last_depth_m.shape),
        "depth_median_m": float(np.median(valid_depth)) if valid_depth.size else None,
        "detections": [
            {"class": item.class_name, "confidence": item.confidence,
             "depth_m": item.depth_m, "point_camera_m": item.point_camera_m,
             "contact_px": item.contact_px, "relative_xy_m": item.relative_xy_m,
             "range_m": item.range_m}
            for item in detections
        ],
        "annotated_image": str(image_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"彩色帧: {tuple(report['color_shape'])}; 深度帧: {tuple(report['depth_shape'])}")
    print(f"有效深度中位数: {report['depth_median_m']!s} m")
    print(f"检测到 {len(detections)} 个有效目标")
    for item in detections:
        print(f"  {item.class_name}: conf={item.confidence:.3f}, optical_depth={item.depth_m:.3f} m, "
              f"camera={item.point_camera_m}, horizontal_xy={item.relative_xy_m}")
    print(f"标注图: {image_path}")
    print(f"报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
