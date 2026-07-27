from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np


def process_safe_smoke_processor(dataset: str) -> dict[str, object]:
    return {"dataset": dataset, "kind": "multiprocess-smoke"}


class PipelineTests(unittest.TestCase):
    def _evaluation_hash_fixture(self, pipeline, output: Path) -> dict[str, Path]:
        prediction = output / "exp404_hard_confidence.jsonl"
        feature_manifest = output / "exp403_feature_manifest.csv"
        contract = output / "exp403_404_provenance_contract.json"
        provenance_file = output / "referenced-provenance.bin"
        database = output / "labels.sqlite"
        exp291 = output / "exp291.jsonl"
        exp291_manifest = output / "exp291_manifest.json"
        exp292 = output / "exp292.jsonl"
        exp292_manifest = output / "exp292_manifest.json"
        for path, contents in (
            (prediction, "prediction-v1\n"),
            (feature_manifest, "feature-v1\n"),
            (provenance_file, "provenance-v1\n"),
            (database, "database-v1\n"),
            (exp291, "exp291-v1\n"),
            (exp291_manifest, "exp291-manifest-v1\n"),
            (exp292, "exp292-v1\n"),
            (exp292_manifest, "exp292-manifest-v1\n"),
        ):
            path.write_text(contents, encoding="utf-8")
        contract.write_text(
            json.dumps({
                "schema": "fixture",
                "provenance": {
                    "feature_artifacts": {},
                    "input_artifacts": {"fixture": {"path": str(provenance_file)}},
                    "baseline_artifacts": {},
                    "config_artifacts": {},
                    "code_artifacts": {},
                },
            }),
            encoding="utf-8",
        )
        (output / "exp404_prediction_manifest.json").write_text(
            json.dumps({
                "hard_confidence_file": prediction.name,
                "feature_manifest_file": feature_manifest.name,
                "provenance_contract_sha256": "a" * 64,
            }),
            encoding="utf-8",
        )
        return {
            "prediction": prediction,
            "provenance": provenance_file,
            "database": database,
            "exp291": exp291,
            "exp291_manifest": exp291_manifest,
            "exp292": exp292,
            "exp292_manifest": exp292_manifest,
        }

    def _evaluation_hash_patches(self, pipeline, fixture: dict[str, Path]) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(pipeline, "DB_PATH", fixture["database"]))
        stack.enter_context(patch.object(pipeline, "EXP291_PREDICTION_PATH", fixture["exp291"]))
        stack.enter_context(patch.object(pipeline, "EXP291_MANIFEST_PATH", fixture["exp291_manifest"]))
        stack.enter_context(patch.object(pipeline, "EXP292_PREDICTION_PATH", fixture["exp292"]))
        stack.enter_context(patch.object(pipeline, "EXP292_MANIFEST_PATH", fixture["exp292_manifest"]))
        return stack

    def test_resume_skips_completed_dataset_hashes(self) -> None:
        from experiments.exp403_408.pipeline import run_smoke

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            first = run_smoke(output, limit=3, stop_after=2)
            second = run_smoke(output, limit=3, resume=True)

        self.assertEqual(first.completed, 2)
        self.assertEqual(second.recomputed_datasets, 1)
        self.assertEqual(second.completed, 3)

    def test_status_has_dashboard_fields(self) -> None:
        from experiments.exp403_408.pipeline import read_status, run_smoke

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_smoke(output, limit=1)
            status = read_status(output)

        self.assertTrue(
            {
                "stage",
                "current_dataset",
                "completed",
                "total",
                "progress_percent",
                "eta_seconds",
                "throughput_per_minute",
                "errors",
                "device",
                "checkpoint_sha256",
            }
            <= set(status)
        )
        self.assertEqual(status["status"], "completed")

    def test_failed_dataset_is_checkpointed_as_explicit_abstention(self) -> None:
        from experiments.exp403_408.pipeline import run_stage

        def failing_processor(_dataset: str) -> dict[str, object]:
            raise RuntimeError("fixture failure")

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = run_stage(
                "audit-features",
                ["small", "large"],
                output,
                processor=failing_processor,
            )
            checkpoint_path = next(
                (output / "checkpoints" / "audit-features").glob("small-*.json")
            )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        self.assertEqual(result.errors, 2)
        self.assertEqual(checkpoint["outcome"], "abstained")
        self.assertEqual(checkpoint["abstention_reason"], "stage_error")
        self.assertIn("fixture failure", checkpoint["error"])

    def test_stage_preserves_small_to_large_input_order(self) -> None:
        from experiments.exp403_408.pipeline import run_stage

        seen: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            run_stage(
                "report",
                ["small", "medium", "large"],
                Path(tmp),
                processor=lambda dataset: seen.append(dataset) or {"dataset": dataset},
                workers=1,
            )

        self.assertEqual(seen, ["small", "medium", "large"])

    def test_cpu_stages_cap_outer_workers_and_learned_stages_are_serial(self) -> None:
        from experiments.exp403_408.pipeline import stage_worker_count

        self.assertEqual(stage_worker_count("classical-predict", 99), 7)
        self.assertEqual(stage_worker_count("classical-predict", 1), 7)
        self.assertEqual(stage_worker_count("ssl-predict", 7), 1)

    def test_changed_input_identity_invalidates_a_checkpoint(self) -> None:
        from experiments.exp403_408.pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_stage(
                "report", ["dataset"], output,
                processor=process_safe_smoke_processor,
                input_hash_for_dataset=lambda _dataset: "old-input",
            )
            rerun = run_stage(
                "report", ["dataset"], output,
                processor=process_safe_smoke_processor,
                input_hash_for_dataset=lambda _dataset: "new-input",
            )

        self.assertEqual(rerun.recomputed_datasets, 1)

    def test_stage_hash_changes_for_actual_values_and_upstream_checkpoint(self) -> None:
        import experiments.exp403_408.pipeline as pipeline

        first = MagicMock(
            train_normal=np.array([[1.0, 2.0]]), test_values=np.array([[3.0, 4.0]]),
            hard_indices=frozenset({0}), standard_indices=frozenset(),
            priority_indices=frozenset(), target_length=512,
        )
        changed_values = MagicMock(
            train_normal=np.array([[1.0, 2.0]]), test_values=np.array([[3.0, 5.0]]),
            hard_indices=frozenset({0}), standard_indices=frozenset(),
            priority_indices=frozenset(), target_length=512,
        )
        inputs = iter((first, changed_values))

        def next_input(_dataset: str) -> MagicMock:
            return next(inputs)

        def first_input(_dataset: str) -> MagicMock:
            return first

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(pipeline, "load_dataset_input", next_input),
                patch.object(pipeline, "_read_stage_checkpoint", return_value={"checkpoint_sha256": "a" * 64}),
            ):
                original = pipeline._stage_input_hash("classical-predict", "toy", Path(tmp), "cpu")
                values_changed = pipeline._stage_input_hash("classical-predict", "toy", Path(tmp), "cpu")
            with (
                patch.object(pipeline, "load_dataset_input", first_input),
                patch.object(pipeline, "_read_stage_checkpoint", return_value={"checkpoint_sha256": "b" * 64}),
            ):
                upstream_changed = pipeline._stage_input_hash("classical-predict", "toy", Path(tmp), "cpu")

        self.assertNotEqual(original, values_changed)
        self.assertNotEqual(original, upstream_changed)

    def test_evaluation_hash_invalidates_prediction_and_referenced_provenance(self) -> None:
        import experiments.exp403_408.pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            fixture = self._evaluation_hash_fixture(pipeline, output)
            with self._evaluation_hash_patches(pipeline, fixture):
                initial = pipeline._stage_input_hash("classical-evaluate", "__aggregate__", output, "cpu")
                fixture["prediction"].write_text("prediction-v2\n", encoding="utf-8")
                prediction_changed = pipeline._stage_input_hash("classical-evaluate", "__aggregate__", output, "cpu")
                fixture["provenance"].write_text("provenance-v2\n", encoding="utf-8")
                provenance_changed = pipeline._stage_input_hash("classical-evaluate", "__aggregate__", output, "cpu")
                fixture["database"].write_text("database-v2\n", encoding="utf-8")
                database_changed = pipeline._stage_input_hash("classical-evaluate", "__aggregate__", output, "cpu")

        self.assertNotEqual(initial, prediction_changed)
        self.assertNotEqual(prediction_changed, provenance_changed)
        self.assertNotEqual(provenance_changed, database_changed)

    def test_report_resume_excludes_own_checkpoint_and_invalidates_upstream(self) -> None:
        import experiments.exp403_408.pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name in (
                pipeline.AUDIT_ARTIFACT,
                "exp404_prediction_manifest.json",
                "exp404_hard_confidence.jsonl",
                pipeline.EVALUATION_ARTIFACT,
                pipeline.CONSENSUS_ARTIFACT,
                pipeline.CONSENSUS_SUMMARY,
            ):
                (output / name).write_text("{}\n", encoding="utf-8")
            upstream = output / "checkpoints" / "audit-features" / "toy.json"
            upstream.parent.mkdir(parents=True)
            upstream.write_text('{"checkpoint":"v1"}\n', encoding="utf-8")
            report_hash = lambda _dataset: pipeline._stage_input_hash(
                "report", "__aggregate__", output, "cpu"
            )
            first = pipeline.run_stage(
                "report", ["__aggregate__"], output,
                processor=process_safe_smoke_processor,
                input_hash_for_dataset=report_hash,
            )
            resumed = pipeline.run_stage(
                "report", ["__aggregate__"], output,
                processor=process_safe_smoke_processor,
                input_hash_for_dataset=report_hash,
                resume=True,
            )
            upstream.write_text('{"checkpoint":"v2"}\n', encoding="utf-8")
            invalidated = pipeline.run_stage(
                "report", ["__aggregate__"], output,
                processor=process_safe_smoke_processor,
                input_hash_for_dataset=report_hash,
                resume=True,
            )

        self.assertEqual(first.recomputed_datasets, 1)
        self.assertEqual(resumed.recomputed_datasets, 0)
        self.assertEqual(invalidated.recomputed_datasets, 1)

    def test_input_hash_failure_is_checkpointed_and_stage_continues(self) -> None:
        from experiments.exp403_408.pipeline import read_status, run_stage

        def hash_for(dataset: str) -> str:
            if dataset == "bad":
                raise ValueError("fixture hash failure")
            return f"hash-{dataset}"

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = run_stage(
                "audit-features", ["bad", "good"], output,
                processor=process_safe_smoke_processor,
                input_hash_for_dataset=hash_for,
            )
            bad = next((output / "checkpoints" / "audit-features").glob("bad-*.json"))
            checkpoint = json.loads(bad.read_text(encoding="utf-8"))
            status = read_status(output)

        self.assertEqual(result.completed, 2)
        self.assertEqual(result.errors, 1)
        self.assertEqual(checkpoint["outcome"], "abstained")
        self.assertEqual(checkpoint["abstention_reason"], "stage_error")
        self.assertIn("fixture hash failure", checkpoint["error"])
        self.assertEqual(status["current_dataset"], "good")

    def test_learned_stage_heartbeats_are_run_bound_while_processor_runs(self) -> None:
        from experiments.exp403_408.pipeline import read_status, run_stage

        entered = threading.Event()
        release = threading.Event()

        def slow_processor(dataset: str) -> dict[str, object]:
            entered.set()
            release.wait(timeout=2.0)
            return {"dataset": dataset}

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            worker = threading.Thread(
                target=run_stage,
                kwargs={
                    "stage": "ssl-predict", "datasets": ["slow"], "output_dir": output,
                    "processor": slow_processor, "run_id": "run-fixture",
                    "run_started_unix": 123.0,
                    "heartbeat_interval_seconds": 0.01,
                },
            )
            worker.start()
            self.assertTrue(entered.wait(timeout=1.0))
            first = read_status(output)
            time.sleep(0.04)
            second = read_status(output)
            release.set()
            worker.join(timeout=2.0)

        self.assertEqual(first["run_id"], "run-fixture")
        self.assertEqual(first["run_started_unix"], 123.0)
        self.assertEqual(first["current_dataset"], "slow")
        self.assertGreater(second["updated_unix"], first["updated_unix"])

    def test_resume_rate_uses_only_this_invocations_new_work(self) -> None:
        from experiments.exp403_408.pipeline import read_status, run_stage

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_stage(
                "report", ["one", "two"], output,
                processor=process_safe_smoke_processor,
                stop_after=1,
            )
            with patch("experiments.exp403_408.pipeline.time.monotonic", side_effect=(10.0, 14.0, 14.0, 14.0, 14.0)):
                run_stage(
                    "report", ["one", "two"], output,
                    processor=process_safe_smoke_processor,
                    resume=True,
                )
            status = read_status(output)

        self.assertEqual(status["completed"], 2)
        self.assertEqual(status["newly_processed"], 1)
        self.assertEqual(status["throughput_per_minute"], 15.0)
        self.assertEqual(status["eta_seconds"], 0.0)

    def test_multiprocess_stop_after_does_not_submit_extra_datasets(self) -> None:
        from experiments.exp403_408.pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            result = run_stage(
                "audit-features",
                ["one", "two", "three"],
                Path(tmp),
                processor=process_safe_smoke_processor,
                workers=2,
                stop_after=1,
            )

        self.assertEqual(result.recomputed_datasets, 1)
        self.assertEqual(result.completed, 1)

    def test_non_prefix_resume_reports_actual_last_processed_dataset(self) -> None:
        from experiments.exp403_408.pipeline import read_status, run_stage

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_stage(
                "report", ["large"], output,
                processor=process_safe_smoke_processor,
            )
            run_stage(
                "report", ["small", "medium", "large"], output,
                processor=process_safe_smoke_processor,
                resume=True,
            )
            status = read_status(output)

        self.assertEqual(status["current_dataset"], "medium")

    def test_classical_finalizer_invokes_task6_publication(self) -> None:
        import experiments.exp403_408.pipeline as pipeline

        def toy_input(_dataset: str) -> MagicMock:
            return MagicMock(target_length=512)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / pipeline.AUDIT_ARTIFACT).parent.mkdir(parents=True, exist_ok=True)
            (output / pipeline.AUDIT_ARTIFACT).write_text("{}\n", encoding="utf-8")
            checkpoint = output / "checkpoints" / "classical-predict" / "toy-000000000000.json"
            checkpoint.parent.mkdir(parents=True)
            pipeline._write_checkpoint(
                checkpoint,
                {"dataset": "toy", "outcome": "completed", "prediction_rows": []},
            )
            with (
                patch.object(pipeline, "_safe_name", return_value="toy-000000000000"),
                patch.object(pipeline, "freeze_provenance_contract", return_value="a" * 64),
                patch.object(pipeline, "load_expected_provenance", return_value=MagicMock()),
                patch.object(pipeline, "load_frozen_lanes", return_value=MagicMock()),
                patch.object(pipeline, "load_dataset_input", toy_input),
                patch.object(pipeline, "write_prediction") as publish,
            ):
                pipeline._publish_classical_prediction(output, ["toy"])

        publish.assert_called_once()

    def test_classical_predict_records_explicit_no_hard_candidate_result(self) -> None:
        import experiments.exp403_408.pipeline as pipeline

        bundle = MagicMock(hard_indices=frozenset())
        with patch.object(pipeline, "load_dataset_input", return_value=bundle), patch.object(
            pipeline, "predict_dataset"
        ) as predict:
            result = pipeline._classical_predict("toy")

        self.assertEqual(result["result_kind"], "no_frozen_hard_candidates")
        self.assertEqual(result["prediction_rows"], [])
        predict.assert_not_called()

    def test_aggregate_evaluation_invokes_task7_hash_gate(self) -> None:
        import experiments.exp403_408.pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "exp404_prediction_manifest.json").write_text(
                json.dumps({"provenance_contract_sha256": "a" * 64}), encoding="utf-8"
            )
            with (
                patch.object(pipeline, "evaluate_exp405", return_value={"hash_gated": True}),
                patch.object(pipeline, "load_frozen_lanes", return_value=MagicMock()),
                patch.object(pipeline, "_evaluation_labels", return_value=MagicMock()),
            ):
                result = pipeline._classical_evaluate(output)

        self.assertEqual(result["outcome"], "completed")

    def test_consensus_stage_invokes_task10(self) -> None:
        import experiments.exp403_408.pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(pipeline, "load_dataset_input", return_value=MagicMock()),
                patch.object(pipeline, "_consensus_sources", return_value=(MagicMock(),)),
                patch.object(pipeline, "load_frozen_lanes", return_value=MagicMock()),
                patch.object(pipeline, "build_consensus", return_value={"real": "task10"}) as build,
            ):
                result = pipeline._consensus_evaluate("toy", Path(tmp))

        self.assertEqual(result["consensus"], {"real": "task10"})
        build.assert_called_once()

    def test_report_emits_registered_result_and_summary_artifacts(self) -> None:
        import experiments.exp403_408.pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name in (
                pipeline.AUDIT_ARTIFACT,
                "exp404_prediction_manifest.json",
                "exp404_hard_confidence.jsonl",
                pipeline.EVALUATION_ARTIFACT,
                pipeline.CONSENSUS_ARTIFACT,
                pipeline.CONSENSUS_SUMMARY,
            ):
                (output / name).write_text("{}\n", encoding="utf-8")
            def write_tables(_output: Path) -> None:
                (output / pipeline.EXP405_DATASET_METRICS).write_text("fixture\n", encoding="utf-8")
                (output / pipeline.EXP405_CANDIDATE_METRICS).write_text("fixture\n", encoding="utf-8")
                (output / pipeline.EXP408_VARIANT_METRICS).write_text("fixture\n", encoding="utf-8")
                (output / pipeline.EXP408_CANDIDATE_METRICS).write_text("fixture\n", encoding="utf-8")

            with (
                patch.object(pipeline, "_write_published_evaluation_tables", side_effect=write_tables),
                patch.object(pipeline, "write_reports") as write_reports,
            ):
                result = pipeline._write_report(output)

            self.assertEqual(result["outcome"], "completed")
            self.assertGreater((output / pipeline.RESULT_ARTIFACT).stat().st_size, 0)
            self.assertGreater((output / pipeline.SUMMARY_ARTIFACT).stat().st_size, 0)
            write_reports.assert_called_once()

    def test_report_separates_autonomous_and_human_assisted_metrics(self) -> None:
        from experiments.exp403_408.reporting import build_report, write_reports
        from experiments.exp403_408.tests.test_evaluation import reporting_fixture

        with tempfile.TemporaryDirectory() as tmp:
            output = reporting_fixture(Path(tmp))
            text = build_report(output)
            artifacts = write_reports(output, command="python3 scratch/run_exp403_408_feature_ml_validation.py --stage report")

            self.assertIn("자동 경보는 변경하지 않음", text)
            self.assertIn("human-assisted", text)
            self.assertNotIn("배포 검증 완료", text)
            self.assertTrue(artifacts.manifest_json.is_file())
            self.assertTrue(artifacts.manifest_markdown.is_file())

    def test_report_freezes_official_evaluation_rows_before_independent_report(self) -> None:
        import experiments.exp403_408.pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name in (
                pipeline.AUDIT_ARTIFACT,
                "exp404_prediction_manifest.json",
                "exp404_hard_confidence.jsonl",
                pipeline.EVALUATION_ARTIFACT,
                pipeline.CONSENSUS_ARTIFACT,
                pipeline.CONSENSUS_SUMMARY,
            ):
                (output / name).write_text("{}\n", encoding="utf-8")
            def write_tables_side_effect(_output: Path) -> None:
                (output / pipeline.EXP405_DATASET_METRICS).write_text("fixture\n", encoding="utf-8")
                (output / pipeline.EXP405_CANDIDATE_METRICS).write_text("fixture\n", encoding="utf-8")
                (output / pipeline.EXP408_VARIANT_METRICS).write_text("fixture\n", encoding="utf-8")
                (output / pipeline.EXP408_CANDIDATE_METRICS).write_text("fixture\n", encoding="utf-8")

            with (
                patch.object(pipeline, "_write_published_evaluation_tables", side_effect=write_tables_side_effect) as write_tables,
                patch.object(pipeline, "write_reports"),
            ):
                pipeline._write_report(output)

        write_tables.assert_called_once_with(output)

    def test_all_marks_status_failed_when_registered_outputs_are_absent(self) -> None:
        import experiments.exp403_408.pipeline as pipeline

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            completed = pipeline.StageResult(
                stage="audit-features", completed=1, total=1,
                recomputed_datasets=0, errors=0, output_dir=output,
            )
            with (
                patch.object(pipeline, "list_datasets_small_to_large", return_value=["toy"]),
                patch.object(pipeline, "run_stage", return_value=completed),
                patch.object(pipeline, "_finalize_stage"),
            ):
                with self.assertRaisesRegex(ValueError, "required Exp403-408 outputs"):
                    pipeline.run_pipeline(stage="all", output_dir=output)
            status = pipeline.read_status(output)

        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["stage"], "report")


if __name__ == "__main__":
    unittest.main()
