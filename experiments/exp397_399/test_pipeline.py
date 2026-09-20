from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.exp397_399.independent_reaggregate_exp397_399 import (
    compare_summary,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "exp397_399"
    / "run_experiment_397_399_pipeline.py"
)


class PipelineTests(unittest.TestCase):
    def test_direct_script_imports_from_repository_root(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--stage", result.stdout)

    def test_independent_reaggregate_detects_modified_summary(self):
        candidate_rows = [
            {
                "variant": "exp292_baseline",
                "alpha": "0.005",
                "rank": "1",
                "dataset": "A",
                "is_true_anomaly": "True",
            }
        ]
        summary_rows = [
            {
                "variant": "exp292_baseline",
                "alpha": "0.005",
                "workload_type": "global_k",
                "workload_k": "1",
                "reviewed": "1",
                "tp_at_k": "0",
                "fp_at_k": "1",
                "precision_at_k": "0.0",
            }
        ]

        mismatches = compare_summary(candidate_rows, summary_rows)

        self.assertTrue(mismatches)

    def test_independent_reaggregate_detects_modified_precision_lift(self):
        candidate_rows = [
            {
                "variant": "exp292_baseline",
                "alpha": "0.005",
                "rank": "1",
                "dataset": "A",
                "is_true_anomaly": "True",
            }
        ]
        summary_rows = [
            {
                "variant": "exp292_baseline",
                "alpha": "0.005",
                "workload_type": "global_k",
                "workload_k": "1",
                "reviewed": "1",
                "tp_at_k": "1",
                "fp_at_k": "0",
                "precision_at_k": "1.0",
                "total_standard_review_candidates": "1",
                "total_standard_review_tp": "1",
                "total_standard_review_fp": "0",
                "review_tp_capture_rate": "1.0",
                "baseline_precision_at_k": "1.0",
                "precision_lift_vs_exp292": "0.5",
            }
        ]

        mismatches = compare_summary(candidate_rows, summary_rows)

        self.assertTrue(mismatches)

    def test_queue_registry_contains_exp397_399_once(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "run_rank_experiments_sequential.py"
        ).read_text(encoding="utf-8")

        self.assertEqual(
            source.count('"id": "experiment_397_399_review_ranking"'),
            1,
        )
        self.assertIn("exp399_workload_summary.csv", source)
        self.assertIn("exp398_dataset_diagnostics.csv", source)

    def test_pipeline_preserves_zero_candidate_datasets(self):
        pipeline_source = SCRIPT.read_text(encoding="utf-8")
        evaluation_source = (
            Path(__file__).resolve().parent / "evaluation.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"dataset_names": names', pipeline_source)
        self.assertIn('prediction_manifest.get("dataset_names"', evaluation_source)


if __name__ == "__main__":
    unittest.main()
