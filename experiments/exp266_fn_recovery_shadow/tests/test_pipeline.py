from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from experiments.exp266_fn_recovery_shadow import run_pipeline
from experiments.exp266_fn_recovery_shadow import prediction
from experiments.exp266_fn_recovery_shadow.contracts import FrozenDatasetInputs, FrozenRecoveryInputs


class PipelineTests(unittest.TestCase):
    def test_cli_exposes_every_stage(self):
        parser = run_pipeline.build_parser()
        for stage in ("audit", "predict", "risk", "evaluate", "report", "all"):
            self.assertEqual(parser.parse_args(["--stage", stage]).stage, stage)

    def test_run_all_resumes_after_completed_stages_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            calls: list[str] = []
            with patch.object(
                run_pipeline,
                "run_stage",
                side_effect=lambda stage, *_args: calls.append(stage),
            ), patch.object(run_pipeline, "_stage_complete", side_effect=[True, False]):
                run_pipeline.run_all(output_root, dataset_limit=None, workers=7)

        self.assertEqual(calls, ["predict", "risk", "evaluate", "report"])

    def test_run_all_smoke_skips_full_population_train_risk(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            calls: list[str] = []
            with patch.object(
                run_pipeline,
                "run_stage",
                side_effect=lambda stage, *_args: calls.append(stage),
            ), patch.object(run_pipeline, "first_incomplete_stage", return_value="audit"):
                run_pipeline.run_all(output_root, dataset_limit=10, workers=2)

        self.assertEqual(calls, ["audit", "predict", "evaluate", "report"])

    def test_smoke_audit_completion_accepts_full_frozen_input_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "00_contract_audit.json").write_text(
                json.dumps(
                    {
                        "experiment_id": run_pipeline.EXPERIMENT_ID,
                        "datasets": run_pipeline.EXPECTED_DATASETS,
                        "lane_totals": run_pipeline.EXPECTED_LANE_TOTALS,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(run_pipeline, "_full_population_matches", return_value=True):
                complete = run_pipeline._stage_complete("audit", root, dataset_limit=10)

        self.assertTrue(complete)

    def test_smoke_completion_validates_partial_predict_evaluate_and_report_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "smoke" / "dataset_limit_10"
            root.mkdir(parents=True)
            names = [f"dataset-{index}" for index in range(10)]
            (root / "03_frozen_predictions.jsonl").write_text(
                "".join(json.dumps({"dataset": name}) + "\n" for name in names),
                encoding="utf-8",
            )
            (root / "04_prediction_manifest.json").write_text("{}\n", encoding="utf-8")

            for filename in (
                "05_dataset_results.csv",
                "07_source_agreement_diagnostics.csv",
            ):
                with (root / filename).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["dataset"])
                    writer.writeheader()
                    writer.writerows({"dataset": name} for name in names)
            (root / "06_global_summary.csv").write_text(
                "datasets\n10\n", encoding="utf-8"
            )

            for filename in (
                "00_contract_audit.json",
                "01_source_manifest.csv",
                "08_invariance_report.md",
                "09_result_report.md",
            ):
                (root / filename).write_text("smoke\n", encoding="utf-8")
            (root / "10_independent_reaggregation.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "missing_datasets": 0,
                        "index_errors": 0,
                        "lane_overlaps": 0,
                        "hard_parity_mismatches": 0,
                        "summary_verification": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            manifest_files = (
                "00_contract_audit.json",
                "01_source_manifest.csv",
                "03_frozen_predictions.jsonl",
                "04_prediction_manifest.json",
                "05_dataset_results.csv",
                "06_global_summary.csv",
                "07_source_agreement_diagnostics.csv",
                "08_invariance_report.md",
                "09_result_report.md",
                "10_independent_reaggregation.json",
            )
            with (root / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file", "sha256"])
                writer.writeheader()
                for filename in manifest_files:
                    writer.writerow(
                        {
                            "file": filename,
                            "sha256": hashlib.sha256((root / filename).read_bytes()).hexdigest(),
                        }
                    )

            prediction_manifest = {
                "datasets": 10,
                "rows": 10,
                "complete_population": False,
                "smoke": True,
                "coverage": {
                    "expected_datasets": run_pipeline.EXPECTED_DATASETS,
                    "frozen_population": run_pipeline.EXPECTED_DATASETS,
                    "predicted_datasets": 10,
                },
            }
            with patch.object(
                run_pipeline, "verify_prediction_checkpoint", return_value=prediction_manifest
            ), patch.object(run_pipeline, "verify_summary", return_value={"passed": True}):
                predict_complete = run_pipeline._stage_complete("predict", root, dataset_limit=10)
                evaluate_complete = run_pipeline._stage_complete("evaluate", root, dataset_limit=10)
                report_complete = run_pipeline._stage_complete("report", root, dataset_limit=10)

        self.assertTrue(predict_complete)
        self.assertTrue(evaluate_complete)
        self.assertTrue(report_complete)

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

    def test_periodic_heartbeat_keeps_live_prediction_status_current(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heartbeat = run_pipeline.StatusHeartbeat(
                root / "status.json",
                output_root=root,
                stdout_log=root / "full.log",
                workers=7,
                expected_datasets=3,
                interval_seconds=0.01,
            )
            heartbeat.start("predict")
            heartbeat.progress("first", 1)
            time.sleep(0.01)
            heartbeat.progress("demo", 2)
            time.sleep(0.03)
            heartbeat.close()
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(status["status"], "running")
        self.assertEqual(status["stage"], "predict")
        self.assertEqual(status["datasets_done"], 2)
        self.assertEqual(status["current_dataset"], "demo")
        self.assertGreater(status["recent_datasets_per_minute"], 0)
        self.assertGreater(status["average_datasets_per_minute"], 0)
        self.assertGreater(status["updated_unix"], 0)

    def test_failed_heartbeat_preserves_last_progress_and_adds_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heartbeat = run_pipeline.StatusHeartbeat(
                root / "status.json",
                output_root=root,
                stdout_log=root / "full.log",
                workers=7,
                expected_datasets=10,
                interval_seconds=1.0,
            )
            heartbeat.start("evaluate")
            time.sleep(0.01)
            heartbeat.progress("latest-dataset", 4)
            heartbeat.fail(["RuntimeError: intentional failure"])
            heartbeat.close()
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["stage"], "evaluate")
        self.assertEqual(status["datasets_done"], 4)
        self.assertEqual(status["current_dataset"], "latest-dataset")
        self.assertGreater(status["recent_datasets_per_minute"], 0)
        self.assertEqual(status["errors"], ["RuntimeError: intentional failure"])

    def test_new_stage_resets_rate_sample_until_its_own_progress_arrives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heartbeat = run_pipeline.StatusHeartbeat(
                root / "status.json", output_root=root, stdout_log=root / "full.log",
                workers=7, expected_datasets=10, interval_seconds=1.0,
            )
            heartbeat.start("predict")
            time.sleep(0.01)
            heartbeat.progress("predict-dataset", 1)
            heartbeat.start("risk")
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            heartbeat.close()

        self.assertEqual(status["stage"], "risk")
        self.assertEqual(status["datasets_done"], 0)
        self.assertEqual(status["recent_datasets_per_minute"], 0.0)

    def test_run_stage_passes_progress_callback_to_each_dataset_stage(self):
        callback = lambda *_args: None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls: dict[str, object] = {}

            def record(stage):
                def action(*_args, **kwargs):
                    calls[stage] = kwargs.get("progress_callback")

                return action

            with patch.object(run_pipeline, "load_frozen_inputs", record("audit")), \
                 patch.object(run_pipeline, "predict_all", record("predict")), \
                 patch.object(run_pipeline, "audit_train_risk", record("risk")), \
                 patch.object(run_pipeline, "evaluate_predictions", record("evaluate")), \
                 patch.object(run_pipeline, "write_reports", record("report")), \
                 patch.object(run_pipeline, "_stage_complete", return_value=True):
                for stage in run_pipeline.STAGE_ORDER:
                    run_pipeline.run_stage(stage, root, None, progress_callback=callback)

        self.assertEqual(calls, {stage: callback for stage in run_pipeline.STAGE_ORDER})

    def test_completion_rejects_truncated_or_inconsistent_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "03_frozen_predictions.jsonl").write_text('{"dataset":"only"}\n', encoding="utf-8")
            with patch.object(
                run_pipeline,
                "verify_prediction_checkpoint",
                return_value={"datasets": 1117, "complete_population": True},
            ):
                self.assertFalse(run_pipeline._stage_complete("predict", root, None))

            (root / "02_train_normal_risk_audit.csv").write_text("dataset\nonly\n", encoding="utf-8")
            self.assertFalse(run_pipeline._stage_complete("risk", root, None))

            (root / "05_dataset_results.csv").write_text("dataset\nonly\n", encoding="utf-8")
            (root / "06_global_summary.csv").write_text("datasets\n1117\n", encoding="utf-8")
            (root / "07_source_agreement_diagnostics.csv").write_text("dataset\nonly\n", encoding="utf-8")
            with patch.object(run_pipeline, "verify_summary", return_value={"passed": False}):
                self.assertFalse(run_pipeline._stage_complete("evaluate", root, None))

            for name in ("08_invariance_report.md", "09_result_report.md", "MANIFEST.csv"):
                (root / name).write_text("ok\n", encoding="utf-8")
            (root / "10_independent_reaggregation.json").write_text('{"passed": false}\n', encoding="utf-8")
            self.assertFalse(run_pipeline._stage_complete("report", root, None))

            (root / "status.json").write_text('{"error_count": 1, "errors": ["boom"]}\n', encoding="utf-8")
            with patch.object(run_pipeline, "first_incomplete_stage", return_value=None):
                self.assertFalse(run_pipeline.completed_output_is_valid(root, None))


class PredictionWorkerTests(unittest.TestCase):
    def test_prediction_uses_bounded_workers_in_size_target_name_order_with_sorted_output(self):
        def item(name, n_train, n_test, target_length):
            return FrozenDatasetInputs(
                dataset=name,
                n_train=n_train,
                n_test=n_test,
                target_length=target_length,
                hard=frozenset(), shadow=frozenset(), priority=frozenset(), standard=frozenset(),
                evidence=MappingProxyType({}), available_groups=frozenset(), quality_flags=frozenset(),
            )

        datasets = {
            "gamma": item("gamma", 3, 3, 10),
            "alpha": item("alpha", 3, 2, 100),
            "beta": item("beta", 2, 3, 90),
        }
        recovery = FrozenRecoveryInputs(
            population=("alpha", "beta", "gamma"),
            datasets=MappingProxyType(datasets),
            input_sha256=MappingProxyType({}), lane_totals=MappingProxyType({}),
        )

        class RecordingExecutor:
            max_workers_seen = None
            submitted: list[str] = []

            def __init__(self, *, max_workers):
                type(self).max_workers_seen = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, function, *args):
                type(self).submitted.append(args[0].dataset)
                future = Future()
                try:
                    future.set_result(function(*args))
                except BaseException as exc:
                    future.set_exception(exc)
                return future

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(prediction, "load_frozen_inputs", return_value=recovery), \
                 patch.object(prediction, "_validate_complete_population"), \
                 patch.object(prediction, "route_recovery", side_effect=lambda value: value.dataset), \
                 patch.object(prediction, "_prediction_row", side_effect=lambda value, _row, _recovery: {"dataset": value.dataset}), \
                 patch.object(prediction, "ThreadPoolExecutor", RecordingExecutor):
                path = prediction.predict_all(root, workers=2)
            output_names = [json.loads(line)["dataset"] for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(RecordingExecutor.max_workers_seen, 2)
        self.assertEqual(RecordingExecutor.submitted, ["beta", "alpha", "gamma"])
        self.assertEqual(output_names, ["alpha", "beta", "gamma"])


if __name__ == "__main__":
    unittest.main()
