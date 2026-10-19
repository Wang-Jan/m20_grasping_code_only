import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m20_rebot_moving_grasp.orchestrator import TaskState, run_task


class Fake:
    def __init__(self, clean_after=False, grasp=True):
        self.calls = []
        self.clean_after = clean_after
        self.grasp_ok = grasp
        self.scan_count = 0
    def _ok(self, name): self.calls.append(name); return True
    def init_reset(self): return self._ok("init_reset")
    def stand(self): return self._ok("stand")
    def sit(self): return self._ok("sit")
    def goto_target(self): return self._ok("goto_target")
    def goto_bin(self): return self._ok("goto_bin")
    def goto_home(self):
        self._ok("goto_home")
        return self.sit()
    def approach(self): return self._ok("approach")
    def is_clean(self): self.calls.append("is_clean"); return self.clean_after
    def prepare_grasp(self): return self._ok("prepare_grasp")
    def grasp_current_target(self): self.calls.append("grasp_current_target"); return self.grasp_ok
    def hold(self): return self._ok("hold")
    def drop(self): return self._ok("drop")
    def safe_home(self): return self._ok("safe_home")
    def release_vision(self): return self._ok("release_vision")
    def emergency_stop(self): return self._ok("emergency_stop")
    def shutdown(self): self.calls.append("shutdown")


class MainFlowTest(unittest.TestCase):
    def test_successful_cycle_then_clean_return(self):
        dog, arm = Fake(), Fake()
        # First scan has an object, second scan is clean.
        dog.is_clean = lambda: (dog.calls.append("is_clean") or dog.calls.count("goto_target") >= 2)
        result = run_task(dog, arm, max_cycles=2)
        self.assertTrue(result.success)
        self.assertEqual(result.state, TaskState.DONE)
        self.assertIn("goto_bin", dog.calls)
        self.assertIn("drop", arm.calls)
        self.assertEqual(arm.calls[-1], "shutdown")

    def test_max_cycles_returns_home_prone_and_succeeds(self):
        dog, arm = Fake(), Fake()
        result = run_task(dog, arm, max_cycles=1)
        self.assertTrue(result.success)
        self.assertEqual((result.state, result.cycles), (TaskState.DONE, 1))
        self.assertEqual(dog.calls[-3:], ["stand", "goto_home", "sit"])

    def test_grasp_failure_keeps_dog_prone_and_waits_operator(self):
        dog, arm = Fake(grasp=False), Fake(grasp=False)
        result = run_task(dog, arm, max_cycles=1)
        self.assertFalse(result.success)
        self.assertEqual(result.state, TaskState.WAIT_OPERATOR)
        self.assertIn("safe_home", arm.calls)
        self.assertNotIn("goto_bin", dog.calls)

    def test_grasp_failure_calls_operator_abort_callback(self):
        dog, arm = Fake(grasp=False), Fake(grasp=False)
        actions = []
        result = run_task(dog, arm, max_cycles=1, operator_wait=lambda: actions.append("wait") or "abort")
        self.assertEqual(result.state, TaskState.WAIT_OPERATOR)
        self.assertEqual(actions, ["wait"])

    def test_approach_failure_never_starts_grasp(self):
        dog, arm = Fake(), Fake()
        dog.approach = lambda: dog.calls.append("approach") or False
        result = run_task(dog, arm, max_cycles=1)
        self.assertEqual(result.state, TaskState.ERROR)
        self.assertNotIn("prepare_grasp", arm.calls)
        self.assertNotIn("grasp_current_target", arm.calls)
        self.assertIn("emergency_stop", dog.calls)

    def test_grasp_starts_without_a_second_sit_after_approach(self):
        dog, arm = Fake(), Fake()
        original_prepare = arm.prepare_grasp

        def prepare():
            approach_index = dog.calls.index("approach")
            self.assertNotIn("sit", dog.calls[approach_index + 1:])
            return original_prepare()

        arm.prepare_grasp = prepare
        run_task(dog, arm, max_cycles=1)

    def test_prepare_grasp_runs_only_after_approach(self):
        dog, arm = Fake(), Fake()
        events = []
        dog.approach = lambda: events.append("approach") or True
        arm.prepare_grasp = lambda: events.append("prepare_grasp") or True
        run_task(dog, arm, max_cycles=1)
        self.assertLess(events.index("approach"), events.index("prepare_grasp"))

    def test_grasp_vision_is_released_before_a_second_d455_scan(self):
        dog, arm = Fake(), Fake()
        events = []
        goto_target = dog.goto_target
        release_vision = arm.release_vision
        dog.goto_target = lambda: events.append("goto_target") or goto_target()
        arm.release_vision = lambda: events.append("release_vision") or release_vision()
        result = run_task(dog, arm, max_cycles=2)
        self.assertTrue(result.success)
        self.assertEqual(events[:3], ["goto_target", "release_vision", "goto_target"])

    def test_exception_requests_emergency_stop(self):
        dog, arm = Fake(), Fake()
        dog.goto_target = lambda: (_ for _ in ()).throw(RuntimeError("network down"))
        result = run_task(dog, arm)
        self.assertEqual(result.state, TaskState.ERROR)
        self.assertIn("emergency_stop", arm.calls)
        self.assertIn("emergency_stop", dog.calls)


if __name__ == "__main__":
    unittest.main()
