from __future__ import annotations

import inspect
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.exp290_train_raw_exact_normal_guard import prediction


class StructureContractTest(unittest.TestCase):
    def test_prediction_blob_query_does_not_select_labels(self) -> None:
        source = inspect.getsource(prediction.load_blob_bundle_for_prediction)
        select_clause = source.split("FROM instances", 1)[0]
        self.assertNotIn("i.label", select_clause)
        self.assertNotIn("labels_blob", select_clause)

    def test_scratch_wrapper_runs_from_external_working_directory(self) -> None:
        root = Path(__file__).resolve().parents[3]
        script = root / "scratch" / "run_exp290_train_raw_exact_normal_guard.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd="/tmp",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--phase", completed.stdout)


if __name__ == "__main__":
    unittest.main()
