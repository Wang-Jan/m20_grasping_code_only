#!/usr/bin/env python3
"""一键检查 D455 彩色和对齐深度流，不导入 YOLO/Torch。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from m20_rebot_moving_grasp.common.config import ConfigError, load_config
from m20_rebot_moving_grasp.machine_dog.d455_vision import D455Config


def main() -> int:
    try:
        config = D455Config.from_mapping(load_config(PROJECT_ROOT / "config/d455.yaml"))
    except (FileNotFoundError, ConfigError) as exc:
        print(f"D455 配置无效：{exc}", file=sys.stderr)
        return 2

    output_dir = PROJECT_ROOT / "tests/output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(config.serial)
    rs_config.enable_stream(rs.stream.color, config.color_width, config.color_height, rs.format.bgr8, config.fps)
    rs_config.enable_stream(rs.stream.depth, config.depth_width, config.depth_height, rs.format.z16, config.fps)

    print("仅测试 D455 彩色/深度流；不会连接或控制机器狗、机械臂、D435。")
    try:
        profile = pipeline.start(rs_config)
        align = rs.align(rs.stream.color)
        scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
        frame_timeout_ms = max(1, int(round(config.frame_timeout_sec * 1000.0)))
        for _ in range(30):
            pipeline.wait_for_frames(frame_timeout_ms)
        frames = align.process(pipeline.wait_for_frames(frame_timeout_ms))
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth_m = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale_m
    finally:
        pipeline.stop()

    valid = depth_m[depth_m > 0]
    color_path = output_dir / f"d455_color_{stamp}.jpg"
    depth_path = output_dir / f"d455_depth_{stamp}.png"
    report_path = output_dir / f"d455_camera_{stamp}.json"
    depth_preview = cv2.applyColorMap(
        cv2.convertScaleAbs(np.clip(depth_m, 0.0, config.max_depth_m), alpha=255.0 / config.max_depth_m),
        cv2.COLORMAP_TURBO,
    )
    if not cv2.imwrite(str(color_path), color) or not cv2.imwrite(str(depth_path), depth_preview):
        raise RuntimeError("无法保存 D455 检查图像")
    report = {
        "serial": config.serial,
        "color_shape": list(color.shape), "depth_shape": list(depth_m.shape),
        "depth_scale_m": scale_m, "valid_depth_pixels": int(valid.size),
        "median_depth_m": float(np.median(valid)) if valid.size else None,
        "color_image": str(color_path), "depth_preview": str(depth_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
