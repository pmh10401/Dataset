from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "task7_runner", ROOT / "scratch/run_exp263_264_full_missing_aware_integration.py"
    )
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    return runner


class RunnerStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_prepare_queue_removes_stale_and_duplicate_task_entries(self):
        state = {
            "completed": [self.runner.EXP263_ID],
            "history": [],
            "queue": [
                "unrelated",
                self.runner.EXP263_ID,
                self.runner.EXP263_ID,
                self.runner.EXP264_ID,
            ],
            "running": None,
        }

        prepared = self.runner.prepare_queue(state)

        self.assertEqual(prepared["queue"], ["unrelated", self.runner.EXP264_ID])

    def test_finish_phase_removes_queue_entry_and_records_return_code(self):
        state = {
            "completed": [],
            "history": [],
            "queue": [self.runner.EXP263_ID, self.runner.EXP264_ID],
            "running": {"id": self.runner.EXP263_ID, "phase": "predict"},
        }

        finished = self.runner.finish_phase(
            state,
            experiment_id=self.runner.EXP263_ID,
            phase="predict",
            return_code=7,
            started_unix=100.0,
            finished_unix=160.0,
        )

        self.assertIsNone(finished["running"])
        self.assertNotIn(self.runner.EXP263_ID, finished["queue"])
        self.assertEqual(finished["history"][-1]["return_code"], 7)
        self.assertEqual(finished["history"][-1]["phase"], "predict")
        self.assertEqual(finished["history"][-1]["elapsed_minutes"], 1.0)
        self.assertNotIn(self.runner.EXP263_ID, finished["completed"])

    def test_successful_terminal_phase_marks_experiment_completed(self):
        state = {
            "completed": [],
            "history": [],
            "queue": [self.runner.EXP263_ID, self.runner.EXP264_ID],
            "running": {"id": self.runner.EXP263_ID, "phase": "predict"},
        }

        finished = self.runner.finish_phase(
            state,
            experiment_id=self.runner.EXP263_ID,
            phase="predict",
            return_code=0,
            started_unix=100.0,
            finished_unix=100.0,
        )

        self.assertIn(self.runner.EXP263_ID, finished["completed"])
        self.assertEqual(finished["queue"], [self.runner.EXP264_ID])

    def test_state_round_trip_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            payload = self.runner.default_state()
            self.runner.write_state(payload, path)

            loaded = self.runner.load_state(path)
            leftovers = list(path.parent.glob("*.tmp"))

        self.assertEqual(loaded, payload)
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
