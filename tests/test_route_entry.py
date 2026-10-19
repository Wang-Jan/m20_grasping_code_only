import sys
import signal
from unittest.mock import Mock, patch
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m20_rebot_moving_grasp import route_main


class RouteEntryTest(unittest.TestCase):
    def test_route_entry_requires_a_persisted_route_name(self):
        with self.assertRaises(SystemExit):
            route_main.main([])

    def test_route_entry_uses_separate_route_points_file(self):
        args = route_main._parser().parse_args(["--route-name", "daily"])
        self.assertTrue(args.route_points_file.endswith("config/m20/route_points.json"))
        self.assertTrue(args.route_config.endswith("config/m20/routes.json"))

    def test_signal_callback_is_shared_with_route_and_exit_reflects_cleanup(self):
        for cleanup_ok in (True, False):
            with self.subTest(cleanup_ok=cleanup_ok):
                args = route_main._parser().parse_args(["--route-name", "route1", "--run-real"])
                runner = Mock(safe_stop_completed=cleanup_ok, last_failure_reason="")
                def run(*_args):
                    signal.raise_signal(signal.SIGINT)
                    self.assertTrue(factory.call_args.kwargs["stop_requested"]())
                    self.assertTrue(d455_factory.call_args.kwargs["stop_requested"]())
                    return False
                runner.run.side_effect = run
                old_handler = signal.getsignal(signal.SIGINT)
                with patch.object(route_main.demo_main, "_load_configs", return_value=({"task": {"max_grasp_attempts": 1, "d435_miss_suppression_radius_m": .75}}, {}, {}, {})), \
                     patch.object(route_main.demo_main, "_preflight", return_value=[]), \
                     patch.object(route_main.demo_main, "_validate_builtin_d455", return_value=[]), \
                     patch.object(route_main, "_validate_route_args", return_value=[]), \
                     patch.object(route_main, "load_config", return_value={}), \
                     patch.object(route_main.D455Config, "from_mapping"), \
                     patch.object(route_main, "D455ApproachController") as d455_factory, \
                     patch.object(route_main, "ArmTaskAdapter"), \
                     patch.object(route_main, "M20DogAdapter"), \
                     patch.object(route_main, "RoutePickupController", return_value=runner) as factory:
                    self.assertEqual(route_main.run(args), 0 if cleanup_ok else 1)
                self.assertIs(signal.getsignal(signal.SIGINT), old_handler)


if __name__ == "__main__":
    unittest.main()
