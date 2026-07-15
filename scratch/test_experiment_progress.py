from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment_progress import ExperimentProgress


class ExperimentProgressTests(unittest.TestCase):
    def test_progress_lifecycle_is_atomic_and_preserves_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.json"
            progress = ExperimentProgress(
                experiment_id="experiment_165_small_sample_calibration",
                label="Exp165",
                expected_datasets=316,
                detail_csv="detail.csv",
                stdout_log="run.log",
                state_path=path,
            )

            progress.start(phase="audit")
            progress.update(done=7, current_dataset="Demo_normal_1", rows=56, errors=0, phase="calibrate")
            running = json.loads(path.read_text())
            self.assertEqual(running["status"], "running")
            self.assertEqual(running["datasets_done"], 7)
            self.assertEqual(running["current_dataset"], "Demo_normal_1")
            self.assertEqual(running["phase"], "calibrate")
            self.assertFalse(path.with_suffix(".tmp").exists())

            progress.finish(rows=2528, result_summary={"errors": 0})
            completed = json.loads(path.read_text())
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["datasets_done"], 316)
            self.assertEqual(completed["result_summary"], {"errors": 0})


if __name__ == "__main__":
    unittest.main()
