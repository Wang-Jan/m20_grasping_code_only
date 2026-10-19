import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m20_rebot_moving_grasp import main as entry


class MainEntryTest(unittest.TestCase):
    def test_default_entry_does_not_create_hardware(self):
        configs = ({"network": {"orin_ip": "10.21.31.50"}, "m20": {"aos_ip": "10.21.31.103"}}, {}, {}, {})
        with patch.object(entry, "_load_configs", return_value=configs), \
             patch.object(entry, "_preflight", return_value=[]), \
             patch.object(entry, "M20DogAdapter") as dog, \
             patch.object(entry, "ArmTaskAdapter") as arm:
            self.assertEqual(entry.main([]), 0)
            dog.assert_not_called()
            arm.assert_not_called()

    def test_run_real_and_dry_run_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            entry.main(["--run-real", "--dry-run"])

    def test_max_cycles_must_be_positive(self):
        with self.assertRaises(SystemExit):
            entry.main(["--max-cycles", "0"])

    def test_point_scan_entry_rejects_route_arguments(self):
        with self.assertRaises(SystemExit):
            entry.main(["--route-name", "daily"])


if __name__ == "__main__":
    unittest.main()
