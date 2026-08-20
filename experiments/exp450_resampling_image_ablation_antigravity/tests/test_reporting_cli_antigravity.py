"""Task 8 report and closed-command contracts on fixture-only evaluation data."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from experiments.exp450_resampling_image_ablation_antigravity.config_antigravity import (
    EVALUATION_ROOT,
    OFFICIAL_ROOT,
    REPORT_ROOT,
    SMOKE_ROOT,
)
from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import (
    Exp450EvaluationAntigravity,
    METHOD_ORDER,
    curve_tables_exp450_antigravity,
    metric_tables_exp450_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.tests.test_evaluation_antigravity import (
    ALL_SCORES,
    LABELS,
)


REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPT = REPOSITORY / "scratch" / "run_exp450_resampling_image_ablation_antigravity.py"
PYTHON = sys.executable


def _evaluation() -> Exp450EvaluationAntigravity:
    return Exp450EvaluationAntigravity(
        manifests={},
        metric_tables=metric_tables_exp450_antigravity(LABELS, ALL_SCORES),
        curve_tables=curve_tables_exp450_antigravity(LABELS, ALL_SCORES),
    )


class Exp450ReportingCliTests(unittest.TestCase):
    def test_report_writes_canonical_tables_figures_and_exact_curve_axes(self) -> None:
        """The paper artifact retains all ten methods and unswapped coordinates."""
        from experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity import (
            write_exp450_report_antigravity,
        )

        with TemporaryDirectory(prefix="exp450_report_antigravity_") as directory:
            artifact = write_exp450_report_antigravity(_evaluation(), Path(directory))
            self.assertTrue(artifact.metric_csv_path.is_file())
            self.assertTrue(artifact.metric_json_path.is_file())
            self.assertTrue(artifact.curve_csv_path.is_file())
            self.assertTrue(artifact.curve_json_path.is_file())
            self.assertTrue(artifact.paired_csv_path.is_file())
            self.assertTrue(artifact.png_path.is_file())
            self.assertTrue(artifact.svg_path.is_file())
            html = artifact.html_path.read_text(encoding="utf-8")
            self.assertIn("False-positive rate", html)
            self.assertIn("True-positive rate", html)
            self.assertIn("Recall", html)
            self.assertIn("Precision", html)
            self.assertIn("no automatic promotion", html)
            for method in METHOD_ORDER:
                self.assertIn(method, html)
            curve_payload = json.loads(artifact.curve_json_path.read_text(encoding="utf-8"))
            self.assertEqual(curve_payload["roc"]["x_name"], "false_positive_rate")
            self.assertEqual(curve_payload["roc"]["y_name"], "true_positive_rate")
            self.assertEqual(curve_payload["precision_recall"]["x_name"], "recall")
            self.assertEqual(curve_payload["precision_recall"]["y_name"], "precision")

    def test_status_and_verify_are_scorer_and_label_free_fresh_processes(self) -> None:
        """Read-only inspection must not load scientific scoring or label capability."""
        for command in ("status", "verify"):
            result = subprocess.run(
                [PYTHON, "-B", str(SCRIPT), command],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPOSITORY,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["experiment"], "Exp450")
            self.assertEqual(payload["science_modules_loaded"], [])
        self.assertTrue(all(not root.exists() for root in (SMOKE_ROOT, OFFICIAL_ROOT, EVALUATION_ROOT, REPORT_ROOT)))
    def test_progress_schema_and_evaluation_gate_do_not_start_science(self) -> None:
        """Progress is a stable dataset-condition record and evaluation is gated first."""
        from scratch.run_exp450_resampling_image_ablation_antigravity import (
            progress_record_antigravity,
        )

        record = progress_record_antigravity(
            dataset="fixture_antigravity", condition="A", status="success",
            elapsed_seconds=2.0, completed=1, total=4,
        )
        self.assertEqual(
            set(record),
            {"experiment", "dataset", "condition", "status", "elapsed_seconds", "eta_seconds", "completed", "total"},
        )
        self.assertEqual(record["eta_seconds"], 6.0)
        result = subprocess.run(
            [PYTHON, "-B", str(SCRIPT), "evaluate"],
            capture_output=True,
            text=True,
            cwd=REPOSITORY,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("all four full score manifests", result.stderr)
        self.assertTrue(all(not root.exists() for root in (SMOKE_ROOT, OFFICIAL_ROOT, EVALUATION_ROOT, REPORT_ROOT)))
