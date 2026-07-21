from __future__ import annotations

import importlib.util
import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "run_exp266_fn_recovery_shadow",
    Path(__file__).with_name("run_exp266_fn_recovery_shadow.py"),
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class QueueRunnerTests(unittest.TestCase):
    def _signal_installer(self, handlers: dict[int, object]):
        def install(signum, handler):
            previous = handlers.get(signum, signal.SIG_DFL)
            handlers[signum] = handler
            return previous

        return install

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

    def test_sigterm_while_queued_cancels_without_starting_a_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "state.json"
            global_lock = root / "state.lock"
            handlers: dict[int, object] = {}
            installer = self._signal_installer(handlers)

            def signal_on_install(signum, handler):
                previous = installer(signum, handler)
                if signum == signal.SIGTERM and callable(handler):
                    handler(signum, None)
                return previous

            with patch.object(runner, "GLOBAL_STATE_LOCK_PATH", global_lock), \
                 patch.object(runner.signal, "signal", side_effect=signal_on_install), \
                 patch.object(runner.subprocess, "Popen") as popen:
                result = runner.main([
                    "--output-root", str(root / "outputs"),
                    "--state-path", str(state_path),
                    "--poll-seconds", "0.01",
                ])

            state = json.loads(state_path.read_text(encoding="utf-8"))
            status = json.loads((root / "outputs" / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 128 + signal.SIGTERM)
        popen.assert_not_called()
        self.assertIsNone(state["running"])
        self.assertNotIn(runner.EXPERIMENT_ID, state["queue"])
        self.assertEqual(status["status"], "failed")
        self.assertTrue(status["errors"])

    def test_sigint_after_claim_cancels_before_child_start_and_releases_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "state.json"
            global_lock = root / "state.lock"
            handlers: dict[int, object] = {}
            real_claim = runner._claim

            def claim_then_signal(*args, **kwargs):
                claimed = real_claim(*args, **kwargs)
                handler = handlers[signal.SIGINT]
                assert callable(handler)
                handler(signal.SIGINT, None)
                return claimed

            with patch.object(runner, "GLOBAL_STATE_LOCK_PATH", global_lock), \
                 patch.object(runner.signal, "signal", side_effect=self._signal_installer(handlers)), \
                 patch.object(runner, "_claim", side_effect=claim_then_signal), \
                 patch.object(runner.subprocess, "Popen") as popen:
                result = runner.main([
                    "--output-root", str(root / "outputs"),
                    "--state-path", str(state_path),
                    "--poll-seconds", "0.01",
                ])

            state = json.loads(state_path.read_text(encoding="utf-8"))
            status = json.loads((root / "outputs" / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 128 + signal.SIGINT)
        popen.assert_not_called()
        self.assertIsNone(state["running"])
        self.assertNotIn(runner.EXPERIMENT_ID, state["queue"])
        self.assertEqual(status["status"], "failed")
        self.assertTrue(status["errors"])

    def test_signal_during_popen_terminates_child_before_running_state_is_published(self):
        class SpawnedChild:
            pid = 4321

            def __init__(self):
                self.terminated = False
                self.waited = False

            def poll(self):
                return -signal.SIGTERM if self.terminated else None

            def terminate(self):
                self.terminated = True

            def wait(self):
                self.waited = True
                return -signal.SIGTERM if self.terminated else 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "state.json"
            global_lock = root / "state.lock"
            handlers: dict[int, object] = {}
            child = SpawnedChild()

            def spawn_during_signal(*_args, **_kwargs):
                handler = handlers[signal.SIGTERM]
                assert callable(handler)
                handler(signal.SIGTERM, None)
                return child

            with patch.object(runner, "GLOBAL_STATE_LOCK_PATH", global_lock), \
                 patch.object(runner.signal, "signal", side_effect=self._signal_installer(handlers)), \
                 patch.object(runner.subprocess, "Popen", side_effect=spawn_during_signal), \
                 patch.object(runner, "completed_output_is_valid", return_value=True):
                result = runner.main([
                    "--output-root", str(root / "outputs"),
                    "--state-path", str(state_path),
                    "--poll-seconds", "0.01",
                ])

            state = json.loads(state_path.read_text(encoding="utf-8"))
            status = json.loads((root / "outputs" / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 128 + signal.SIGTERM)
        self.assertTrue(child.terminated)
        self.assertTrue(child.waited)
        self.assertIsNone(state["running"])
        self.assertNotIn(runner.EXPERIMENT_ID, state["completed"])
        self.assertEqual(state["history"][-1]["return_code"], 128 + signal.SIGTERM)
        self.assertEqual(status["status"], "failed")


if __name__ == "__main__":
    unittest.main()
