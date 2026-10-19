"""load_config 的独立单元测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from m20_rebot_moving_grasp.common.config import ConfigError, load_config, validate_config


class LoadConfigTest(unittest.TestCase):
    def _write(self, content: str) -> Path:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        handle.write(content)
        handle.close()
        return Path(handle.name)

    def test_reads_mapping(self) -> None:
        config_path = self._write("camera:\n  fps: 30\ntask:\n  retry: 2\n")
        self.assertEqual(load_config(config_path), {"camera": {"fps": 30}, "task": {"retry": 2}})

    def test_missing_file_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config("/tmp/m20_rebot_missing_config.yaml")

    def test_empty_file_raises_config_error(self) -> None:
        config_path = self._write("")
        with self.assertRaises(ConfigError):
            load_config(config_path)

    def test_list_root_raises_config_error(self) -> None:
        config_path = self._write("- one\n- two\n")
        with self.assertRaises(ConfigError):
            load_config(config_path)

    def test_project_configs_validate(self) -> None:
        for kind in ("system", "arm", "vision", "poses"):
            config = load_config(PROJECT_ROOT / "config" / f"{kind}.yaml")
            self.assertIs(validate_config(config, kind), config)

    def test_transport_pose_keeps_measured_j4_hold_angle(self) -> None:
        poses = load_config(PROJECT_ROOT / "config" / "poses.yaml")
        target = poses["transport_pose"]["joint_positions_rad"]
        self.assertEqual(len(target), 6)
        self.assertAlmostEqual(float(target[3]), -1.629091, places=6)

    def test_validate_rejects_invalid_d435_recovery_feedback_retry_count(self) -> None:
        poses = load_config(PROJECT_ROOT / "config" / "poses.yaml")
        poses["d435_recovery_scan"]["feedback_retry_count"] = 2
        with self.assertRaisesRegex(ConfigError, "feedback_retry_count"):
            validate_config(poses, "poses")

    def test_validate_rejects_unsupported_camera(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "vision.yaml")
        config["camera"]["type"] = "realsense_d435i"
        with self.assertRaisesRegex(ConfigError, "camera.type"):
            validate_config(config, "vision")

    def test_validate_rejects_gripper_torque_above_limit(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "arm.yaml")
        config["robot"]["gripper"]["dm"]["close_torque"] = 2.0
        with self.assertRaisesRegex(ConfigError, "不能超过 tau_max"):
            validate_config(config, "arm")
