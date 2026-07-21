from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.exp266_fn_recovery_shadow import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_cli_exposes_every_stage(self):
        parser = run_pipeline.build_parser()
        for stage in ("audit", "predict", "risk", "evaluate", "report", "all"):
            self.assertEqual(parser.parse_args(["--stage", stage]).stage, stage)

    def test_run_all_resumes_after_completed_stages_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            (output_root / "00_contract_audit.json").write_text("{}\n", encoding="utf-8")
            (output_root / "01_source_manifest.csv").write_text("dataset\n", encoding="utf-8")
            calls: list[str] = []
            with patch.object(
                run_pipeline,
                "run_stage",
                side_effect=lambda stage, *_args: calls.append(stage),
            ):
                run_pipeline.run_all(output_root, dataset_limit=None, workers=7)

        self.assertEqual(calls, ["predict", "risk", "evaluate", "report"])

    def test_status_records_running_progress_and_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            run_pipeline.write_status(
                status_path,
                status="running",
                stage="predict",
                datasets_done=8,
                expected_datasets=10,
                current_dataset="demo",
                workers=7,
                output_root=Path(temporary),
                stdout_log=Path(temporary) / "full.log",
                started_unix=100.0,
                warnings=["resume"],
                errors=[],
            )
            running = json.loads(status_path.read_text(encoding="utf-8"))
            run_pipeline.write_status(
                status_path,
                status="failed",
                stage="predict",
                datasets_done=8,
                expected_datasets=10,
                current_dataset="demo",
                workers=7,
                output_root=Path(temporary),
                stdout_log=Path(temporary) / "full.log",
                started_unix=100.0,
                warnings=[],
                errors=["missing dataset"],
            )
            failed = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(running["status"], "running")
        self.assertEqual(running["datasets_done"], 8)
        self.assertEqual(running["expected_datasets"], 10)
        self.assertEqual(running["current_dataset"], "demo")
        self.assertIn("recent_datasets_per_minute", running)
        self.assertIn("eta_seconds", running)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["errors"], ["missing dataset"])


if __name__ == "__main__":
    unittest.main()
