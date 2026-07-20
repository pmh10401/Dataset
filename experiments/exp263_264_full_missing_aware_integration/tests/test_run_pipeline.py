from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from experiments.exp263_264_full_missing_aware_integration import run_pipeline


ROOT = Path(__file__).resolve().parents[3]


class RunPipelineTest(unittest.TestCase):
    def test_completed_status_contains_dashboard_progress_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with mock.patch.object(
                run_pipeline,
                "run_phase",
                return_value={"done": 10, "current_dataset": "Dataset_010"},
            ):
                return_code = run_pipeline.main(
                    [
                        "--phase",
                        "predict",
                        "--dataset-limit",
                        "10",
                        "--output-root",
                        str(output_root),
                    ]
                )

            status = json.loads((output_root / "status.json").read_text())

        self.assertEqual(return_code, 0)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["phase"], "predict")
        self.assertEqual(status["done"], 10)
        self.assertEqual(status["datasets_done"], 10)
        self.assertEqual(status["expected"], 10)
        self.assertEqual(status["expected_datasets"], 10)
        self.assertEqual(status["current_dataset"], "Dataset_010")
        self.assertEqual(status["errors"], [])
        self.assertEqual(status["error_count"], 0)
        self.assertIsInstance(status["pid"], int)
        resolved_root = output_root.resolve()
        self.assertEqual(status["output_root"], str(resolved_root))
        self.assertEqual(status["output_paths"]["status"], str(resolved_root / "status.json"))
        self.assertEqual(status["detail_csv"], str(resolved_root / "results.csv"))

    def test_exception_writes_failed_status_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            retained = output_root / "retained.txt"
            retained.write_text("keep", encoding="utf-8")
            with mock.patch.object(
                run_pipeline, "run_phase", side_effect=RuntimeError("deliberate failure")
            ):
                with redirect_stderr(io.StringIO()):
                    return_code = run_pipeline.main(
                        ["--phase", "audit", "--output-root", str(output_root)]
                    )

            status = json.loads((output_root / "status.json").read_text())
            retained_text = retained.read_text(encoding="utf-8")

        self.assertNotEqual(return_code, 0)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["phase"], "audit")
        self.assertEqual(status["error_count"], 1)
        self.assertIn("RuntimeError: deliberate failure", status["errors"][0])
        self.assertEqual(retained_text, "keep")

    def test_smoke_evaluation_requires_explicit_anomaly_override(self):
        parser = run_pipeline.build_parser()
        args = parser.parse_args(
            ["--phase", "evaluate", "--dataset-limit", "10"]
        )
        with self.assertRaisesRegex(ValueError, "expected-total-anomalies"):
            run_pipeline.validate_args(args)

    def test_status_write_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            reporter = run_pipeline.StatusReporter(
                phase="audit", output_root=output_root, expected=10
            )
            reporter.update(status="running", done=3, current_dataset="Dataset_003")

            status = json.loads((output_root / "status.json").read_text())
            leftovers = list(output_root.glob("*.tmp"))

        self.assertEqual(status["done"], 3)
        self.assertEqual(leftovers, [])

    def test_dashboard_catalog_registers_exp263_and_exp264_at_full_population(self):
        spec = importlib.util.spec_from_file_location(
            "task7_dashboard", ROOT / "serve_rank_dashboard.py"
        )
        dashboard = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(dashboard)

        for experiment_id in (
            "experiment_263_full_missing_aware_integration",
            "experiment_264_full_missing_aware_posthoc_evaluation",
        ):
            self.assertEqual(
                dashboard.EXPERIMENTS[experiment_id]["expected_datasets"], 1117
            )


if __name__ == "__main__":
    unittest.main()
