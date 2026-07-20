from __future__ import annotations

import importlib.util
import argparse
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
            "running": {
                "id": self.runner.EXP263_ID,
                "phase": "predict",
                "owner_token": "owner-a",
            },
        }

        finished = self.runner.finish_phase(
            state,
            experiment_id=self.runner.EXP263_ID,
            phase="predict",
            owner_token="owner-a",
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
            "running": {
                "id": self.runner.EXP263_ID,
                "phase": "predict",
                "owner_token": "owner-a",
            },
        }

        finished = self.runner.finish_phase(
            state,
            experiment_id=self.runner.EXP263_ID,
            phase="predict",
            owner_token="owner-a",
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

    def test_runner_rejects_production_anomaly_override(self):
        args = argparse.Namespace(
            dataset_limit=None,
            expected_total_anomalies=10,
        )
        with self.assertRaisesRegex(ValueError, "dataset-limit"):
            self.runner.validate_args(args)

    def test_runner_rejects_full_population_anomaly_override(self):
        args = argparse.Namespace(
            dataset_limit=1117,
            expected_total_anomalies=10,
        )
        with self.assertRaisesRegex(ValueError, "strict subset"):
            self.runner.validate_args(args)

    def test_claim_run_slot_refuses_to_replace_unrelated_active_run(self):
        state = {
            "completed": [],
            "history": [],
            "queue": [self.runner.EXP263_ID],
            "running": {"id": "unrelated", "owner_token": "other-owner"},
        }

        claimed = self.runner.claim_run_slot(
            state,
            experiment_id=self.runner.EXP263_ID,
            phase="audit",
            owner_token="owner-a",
            expected_datasets=1117,
            output_root=Path("/tmp/output"),
        )

        self.assertIsNone(claimed)
        self.assertEqual(state["running"]["id"], "unrelated")

    def test_finish_phase_does_not_clear_different_owner(self):
        state = {
            "completed": [],
            "history": [],
            "queue": [self.runner.EXP263_ID],
            "running": {
                "id": "unrelated",
                "phase": "audit",
                "owner_token": "other-owner",
            },
        }

        finished = self.runner.finish_phase(
            state,
            experiment_id=self.runner.EXP263_ID,
            phase="audit",
            owner_token="owner-a",
            return_code=1,
            started_unix=100.0,
            finished_unix=101.0,
        )

        self.assertEqual(finished, state)

    def test_claim_phase_is_serialized_and_second_claim_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.json"
            lock_path = root / "state.lock"
            self.runner.write_state(self.runner.default_state(), state_path)

            first = self.runner.claim_phase(
                state_path=state_path,
                lock_path=lock_path,
                experiment_id=self.runner.EXP263_ID,
                phase="audit",
                owner_token="owner-a",
                expected_datasets=10,
                output_root=root / "output",
            )
            second = self.runner.claim_phase(
                state_path=state_path,
                lock_path=lock_path,
                experiment_id=self.runner.EXP263_ID,
                phase="audit",
                owner_token="owner-b",
                expected_datasets=10,
                output_root=root / "output",
            )
            state = self.runner.load_state(state_path)

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(state["running"]["owner_token"], "owner-a")

    def test_production_state_uses_canonical_global_lock(self):
        self.assertEqual(
            self.runner.state_lock_path(self.runner.STATE_PATH),
            self.runner.GLOBAL_STATE_LOCK_PATH,
        )


if __name__ == "__main__":
    unittest.main()
