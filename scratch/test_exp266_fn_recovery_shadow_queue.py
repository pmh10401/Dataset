from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "run_exp266_fn_recovery_shadow",
    Path(__file__).with_name("run_exp266_fn_recovery_shadow.py"),
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class QueueRunnerTests(unittest.TestCase):
    def test_claim_reaps_dead_owner_before_claiming_slot(self):
        state = {
            "completed": [],
            "history": [],
            "queue": [runner.EXPERIMENT_ID],
            "running": {"id": "other", "pid": 99999999, "child_pid": 99999998},
        }

        claimed = runner._claim(
            state,
            "owner",
            Path("/tmp/exp266"),
            1117,
            pid_checker=lambda _pid: False,
        )

        self.assertEqual(claimed["running"]["owner_token"], "owner")
        self.assertEqual(claimed["history"][-1]["return_code"], "stale_owner_reaped")

    def test_startup_failure_releases_only_its_claim(self):
        state = runner._claim({"completed": [], "history": [], "queue": [], "running": None}, "owner", Path("/tmp/exp266"), 1117)

        released = runner._release_claim(state, "owner", return_code=1, reason="startup failed")

        self.assertIsNone(released["running"])
        self.assertNotIn(runner.EXPERIMENT_ID, released["queue"])
        self.assertEqual(released["history"][-1]["return_code"], 1)

    def test_termination_helper_forwards_to_live_child(self):
        class Child:
            def __init__(self):
                self.terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

        child = Child()
        runner._terminate_child(child)
        self.assertTrue(child.terminated)


if __name__ == "__main__":
    unittest.main()
