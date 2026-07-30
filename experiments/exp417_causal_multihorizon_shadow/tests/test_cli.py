"""Behavioral tests for the Exp417 official and non-official CLIs."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest
from unittest.mock import patch

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG


class ReachedLaterStage(RuntimeError):
    """Controlled sentinel proving argparse reached the post-validation path."""


def _input_files(root: Path) -> tuple[Path, ...]:
    paths = tuple(root / name for name in ("db", "p291", "m291", "p292", "m292"))
    for path in paths:
        path.write_bytes(b"x")
    return paths


def _official_argv(
    root: Path, *, checkpoint: Path | None = None, output: Path | None = None
) -> list[str]:
    files = _input_files(root)
    return [
        "--database", str(files[0]),
        "--exp291-prediction", str(files[1]),
        "--exp291-manifest", str(files[2]),
        "--exp292-prediction", str(files[3]),
        "--exp292-manifest", str(files[4]),
        "--checkpoint-dir", str(checkpoint or root / "checkpoint"),
        "--output-dir", str(output or root / "output"),
    ]


def _smoke_argv(
    root: Path, *, count: str = "3", output: Path | None = None
) -> list[str]:
    files = _input_files(root)
    return [
        "--database", str(files[0]),
        "--exp291-prediction", str(files[1]),
        "--exp291-manifest", str(files[2]),
        "--exp292-prediction", str(files[3]),
        "--exp292-manifest", str(files[4]),
        "--smoke-datasets", count,
        "--output-dir", str(output or root / "output"),
    ]


class OfficialCliTests(unittest.TestCase):
    def setUp(self) -> None:
        from scratch import run_exp417_causal_multihorizon_shadow as official

        self.official = official

    def test_model_contract_hash_covers_runner_routing_and_seed_loops(self) -> None:
        """Changing qualification routing in runner.py must invalidate resume provenance."""
        from experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints import (
            _context,
        )

        calls: list[tuple[str, ...]] = []
        with patch.object(
            self.official,
            "_source_digest",
            side_effect=lambda *paths: calls.append(paths) or "f" * 64,
        ):
            self.official._provenance(_context())

        model_call = next(
            paths
            for paths in calls
            if any(path.endswith("scoring.py") for path in paths)
        )
        self.assertIn(
            "experiments/exp417_causal_multihorizon_shadow/runner.py",
            model_call,
        )

    def test_parser_accepts_only_the_seven_frozen_path_arguments(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = _input_files(root)
            argv = _official_argv(root)
            args = self.official.build_parser().parse_args(argv)
            self.assertEqual(args.database, files[0].resolve())
            for forbidden in (
                "--dataset", "--max-datasets", "--family", "--alpha",
                "--horizon", "--threshold", "--seed", "--budget",
                "--backend", "--label-path", "--data",
            ):
                with self.subTest(forbidden=forbidden):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        self.official.build_parser().parse_args(
                            argv + [forbidden, "x"]
                        )

    def test_existing_official_output_is_rejected_only_by_new_output_validator(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.official.main(_official_argv(root, output=output))
            with (
                patch.object(
                    self.official,
                    "_new_output_dir",
                    side_effect=lambda value: Path(value).resolve(),
                ),
                patch.object(
                    self.official,
                    "execute_official",
                    side_effect=ReachedLaterStage("official output validator bypassed"),
                ),
                self.assertRaisesRegex(ReachedLaterStage, "output validator"),
            ):
                self.official.main(_official_argv(root, output=output))

    def test_missing_checkpoint_parent_is_rejected_only_by_checkpoint_validator(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "missing" / "checkpoint"
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.official.main(_official_argv(root, checkpoint=checkpoint))
            with (
                patch.object(
                    self.official,
                    "_checkpoint_dir",
                    side_effect=lambda value: Path(value).resolve(),
                ),
                patch.object(
                    self.official,
                    "execute_official",
                    side_effect=ReachedLaterStage("checkpoint validator bypassed"),
                ),
                self.assertRaisesRegex(ReachedLaterStage, "checkpoint validator"),
            ):
                self.official.main(_official_argv(root, checkpoint=checkpoint))

    def test_checkpoint_and_output_ancestry_must_be_disjoint(self) -> None:
        """An official output nested in the checkpoint root corrupts exact checkpoint inventory."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            cases = (
                (checkpoint, checkpoint / "official-output"),
                (root / "same-new-path", root / "same-new-path"),
            )
            for checkpoint_path, output in cases:
                with (
                    self.subTest(output=output),
                    patch.object(
                        self.official,
                        "execute_official",
                        side_effect=ReachedLaterStage("ancestry validator bypassed"),
                    ),
                    self.assertRaisesRegex(ValueError, "disjoint"),
                ):
                    self.official.main(
                        _official_argv(
                            root, checkpoint=checkpoint_path, output=output
                        )
                    )

    def test_dirty_git_is_rejected_by_the_real_cleanliness_guard(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Exp417 Test"],
                check=True,
            )
            (root / "tracked").write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "fixture"], check=True
            )
            self.official.require_clean_git(root)
            (root / "tracked").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean"):
                self.official.require_clean_git(root)

    def test_official_pipeline_has_strict_order_and_stops_after_failure(self) -> None:
        events: list[str] = []
        verified_head = "d" * 40
        stages = self.official.OfficialStages(
            clean_git=lambda: events.append("clean_git") or verified_head,
            label_guard=lambda: events.append("label_guard"),
            freeze_inputs=lambda head: events.append(
                f"freeze_inputs:{head}"
            ) or "context",
            open_checkpoint=lambda context: events.append("checkpoint") or "store",
            run=lambda context, store: events.append("runner") or "summary",
            revalidate=lambda: events.append("revalidate"),
            render=lambda summary, store: events.append("render") or {"payload": b"x"},
            verify=lambda payloads, store: events.append("verify"),
            publish=lambda payloads, store: events.append("publish"),
        )
        self.assertEqual(self.official.execute_official(stages), 0)
        self.assertEqual(
            events,
            [
                "clean_git", "label_guard", f"freeze_inputs:{verified_head}", "checkpoint",
                "runner", "revalidate", "render", "verify", "publish",
            ],
        )

        stage_names = (
            "clean_git", "label_guard", "freeze_inputs", "checkpoint",
            "runner", "revalidate", "render", "verify", "publish",
        )
        for failed_stage in stage_names:
            observed: list[str] = []

            def stage(name: str, value=None):
                def invoke(*_args):
                    observed.append(name)
                    if name == failed_stage:
                        raise RuntimeError(name)
                    return value

                return invoke

            failing = self.official.OfficialStages(
                clean_git=stage("clean_git", verified_head),
                label_guard=stage("label_guard"),
                freeze_inputs=stage("freeze_inputs", "context"),
                open_checkpoint=stage("checkpoint", "store"),
                run=stage("runner", "summary"),
                revalidate=stage("revalidate"),
                render=stage("render", {"payload": b"x"}),
                verify=stage("verify"),
                publish=stage("publish"),
            )
            with self.subTest(failed_stage=failed_stage):
                with self.assertRaisesRegex(RuntimeError, failed_stage):
                    self.official.execute_official(failing)
                self.assertEqual(observed[-1], failed_stage)
                self.assertEqual(
                    observed,
                    list(stage_names[: stage_names.index(failed_stage) + 1]),
                )


class SmokeCliTests(unittest.TestCase):
    def setUp(self) -> None:
        from scratch import smoke_exp417_causal_multihorizon_shadow as smoke

        self.smoke = smoke

    def test_smoke_context_is_irreversibly_nonofficial(self) -> None:
        context = self.smoke.SmokeContext(
            selected_datasets=("A", "B", "C"),
            output_dir=Path("/private/tmp/exp417-smoke"),
        )
        self.assertFalse(context.official_prediction)
        self.assertFalse(context.publication_eligible)
        self.assertFalse(context.contains_test_labels)
        self.assertFalse(context.official_checkpoint_reusable)
        self.assertEqual(context.smoke_datasets, 3)

    def test_smoke_count_is_exactly_three_and_explicit_names_are_forbidden(self) -> None:
        parser = self.smoke.build_parser()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            argv = _smoke_argv(root)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(argv + ["--dataset", "A"])

    def test_smoke_count_is_rejected_only_by_exact_three_validator(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            argv = _smoke_argv(root, count="2")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.smoke.main(argv)
            with (
                patch.object(self.smoke, "_exact_three", return_value=3),
                patch.object(
                    self.smoke,
                    "require_clean_git",
                    side_effect=ReachedLaterStage("smoke count validator bypassed"),
                ),
                self.assertRaisesRegex(ReachedLaterStage, "count validator"),
            ):
                self.smoke.main(argv)

    def test_existing_smoke_output_is_rejected_only_by_new_output_validator(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            argv = _smoke_argv(root, output=output)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.smoke.main(argv)
            with (
                patch.object(
                    self.smoke,
                    "_new_output_dir",
                    side_effect=lambda value: Path(value).resolve(),
                ),
                patch.object(
                    self.smoke,
                    "require_clean_git",
                    side_effect=ReachedLaterStage("smoke output validator bypassed"),
                ),
                self.assertRaisesRegex(ReachedLaterStage, "output validator"),
            ):
                self.smoke.main(argv)

    def test_train_strata_selection_is_deterministic_and_uses_three_regions(self) -> None:
        rows = (
            self.smoke.TrainMetadata("Zulu-low", 10, 20, 20),
            self.smoke.TrainMetadata("Alpha-low", 10, 20, 20),
            self.smoke.TrainMetadata("Middle", 50, 100, 100),
            self.smoke.TrainMetadata("Zulu-high", 100, 500, 500),
            self.smoke.TrainMetadata("Alpha-high", 100, 500, 500),
        )
        expected = ("Alpha-high", "Alpha-low", "Middle")
        self.assertEqual(self.smoke.select_train_strata(rows, 3), expected)
        self.assertEqual(
            self.smoke.select_train_strata(tuple(reversed(rows)), 3), expected
        )

    def test_smoke_payload_cannot_be_mistaken_for_official_publication(self) -> None:
        context = self.smoke.SmokeContext(
            selected_datasets=("A", "B", "C"),
            output_dir=Path("/private/tmp/exp417-smoke"),
        )
        projection = self.smoke.DiskProjection(
            measured_smoke_artifact_bytes=1024,
            measured_sample_checkpoint_bytes=2048,
            failed_terminal_projection_bytes=1_000_000,
            candidate_bearing_conservative_upper_bytes=9_000_000,
            frozen_dataset_count=1117,
            frozen_test_instance_count=5000,
            normal_train_series_count=6000,
            sample_basis="actual_canonical_smoke_results",
        )
        payload = self.smoke.render_smoke_context(
            context,
            statuses={"A": "qualified", "B": "failed", "C": "qualified"},
            backend={
                "backend": "cpu",
                "candidate_identity": True,
                "max_normalized_error": 0.0,
                "spearman": 1.0,
                "speedup": 0.67,
                "fallback": False,
                "fallback_reason": None,
            },
            elapsed_seconds=10.0,
            estimated_full_seconds=3723.3333333333335,
            disk_projection=projection,
        )
        self.assertEqual(payload["experiment_id"], FROZEN_CONFIG.experiment_id)
        self.assertFalse(payload["official_prediction"])
        self.assertFalse(payload["publication_eligible"])
        self.assertFalse(payload["official_checkpoint_reusable"])
        self.assertNotIn("precision", payload)
        self.assertNotIn("recall", payload)
        self.assertNotIn("f1", payload)

    def test_renderer_and_canonical_artifact_use_exact_fifteen_key_schema(self) -> None:
        context = self.smoke.SmokeContext(
            selected_datasets=("A", "B", "C"),
            output_dir=Path("/private/tmp/exp417-smoke-schema"),
        )
        projection = self.smoke.DiskProjection(
            measured_smoke_artifact_bytes=100,
            measured_sample_checkpoint_bytes=200,
            failed_terminal_projection_bytes=300,
            candidate_bearing_conservative_upper_bytes=400,
            frozen_dataset_count=1117,
            frozen_test_instance_count=2222,
            normal_train_series_count=3333,
            sample_basis="actual_canonical_smoke_results",
        )
        payload = self.smoke.render_smoke_context(
            context,
            statuses={"A": "failed", "B": "failed", "C": "failed"},
            backend={
                "backend": "cpu",
                "candidate_identity": True,
                "max_normalized_error": 0.0,
                "spearman": 1.0,
                "speedup": 0.67,
                "fallback": False,
                "fallback_reason": None,
            },
            elapsed_seconds=1.0,
            estimated_full_seconds=372.0,
            disk_projection=projection,
        )
        literal = {
            "experiment_id", "artifact_kind", "official_prediction",
            "publication_eligible", "contains_test_labels",
            "official_checkpoint_reusable", "selection_source",
            "selection_rule", "selected_datasets", "statuses",
            "backend_audit", "elapsed_seconds", "estimated_full_seconds",
            "disk_projection", "estimate_method",
        }
        self.assertEqual(len(literal), 15)
        self.assertEqual(self.smoke.SMOKE_SCHEMA_KEYS, frozenset(literal))
        self.assertEqual(set(payload), literal)
        canonical = json.loads(
            self.smoke.canonical_json_bytes(payload)
        )
        self.assertEqual(set(canonical), literal)
        self.assertEqual(
            set(canonical["disk_projection"]),
            {
                "measured_smoke_artifact_bytes",
                "measured_sample_checkpoint_bytes",
                "failed_terminal_projection_bytes",
                "candidate_bearing_conservative_upper_bytes",
                "frozen_dataset_count",
                "frozen_test_instance_count",
                "normal_train_series_count",
                "sample_basis",
            },
        )

    def test_disk_projection_uses_canonical_bytes_and_exact_population_counts(self) -> None:
        projection = self.smoke.project_disk_bounds(
            measured_smoke_artifact_bytes=895,
            measured_sample_checkpoint_bytes=12_000,
            sample_dataset_count=3,
            frozen_dataset_count=1117,
            frozen_test_instance_count=10_000,
            normal_train_series_count=20_000,
            worst_decision_bytes=900,
            worst_group_bytes=600,
        )
        self.assertEqual(
            projection.failed_terminal_projection_bytes,
            895 + 4_468_000,
        )
        self.assertEqual(
            projection.candidate_bearing_conservative_upper_bytes,
            projection.failed_terminal_projection_bytes
            + 10_000 * 900
            + 20_000 * 600,
        )

    def test_real_checkpoint_record_and_context_units_drive_projection(self) -> None:
        """Corrupting either canonical measurement source changes or stops projection."""
        from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
            Exp417CheckpointStore,
        )
        from experiments.exp417_causal_multihorizon_shadow.runner import (
            CheckpointEvidence,
            _result_evidence_digest,
        )
        from experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints import (
            _context,
            _provenance,
            _result,
        )

        context = _context()
        result = _result(context)
        contract = result.decision_evidence.frozen_train_contract
        material = CheckpointEvidence(
            raw_train_identity_sha256="0" * 64,
            repaired_train_identity_sha256="1" * 64,
            length_group_identity_sha256={100: "2" * 64},
            canonical_model_hashes={
                100: {seed: "3" * 64 for seed in FROZEN_CONFIG.model_seeds}
            },
            score_calibration_hashes={100: "4" * 64},
            elapsed_seconds=0.0,
            dataset=result.dataset,
            database_sha256=context.database_sha256,
            train_contract_identity_sha256=contract.train_identity_sha256,
            result_evidence_sha256=_result_evidence_digest(result),
        )
        with TemporaryDirectory() as directory:
            store = Exp417CheckpointStore(
                Path(directory) / "checkpoint", _provenance(), context.datasets
            )
            store.write_dataset(
                result,
                raw_train_identity_sha256=material.raw_train_identity_sha256,
                repaired_train_identity_sha256=material.repaired_train_identity_sha256,
                length_group_identity_sha256=material.length_group_identity_sha256,
                canonical_model_hashes=material.canonical_model_hashes,
                score_calibration_hashes=material.score_calibration_hashes,
                backend_audit=contract.backend,
                elapsed_seconds=material.elapsed_seconds,
            )
            actual_record_bytes = store._dataset_path(result.dataset).stat().st_size

        with (
            patch.object(
                self.smoke,
                "_checkpoint_record_size",
                wraps=self.smoke._checkpoint_record_size,
            ) as checkpoint_measurement,
            patch.object(
                self.smoke,
                "_schema_upper_units",
                wraps=self.smoke._schema_upper_units,
            ) as context_units,
        ):
            projection = self.smoke.estimate_disk_projection(
                context,
                (result,),
                normal_train_series_count=100,
                measured_smoke_artifact_bytes=10,
            )

        self.assertEqual(actual_record_bytes, 160_843)
        self.assertEqual(projection.measured_sample_checkpoint_bytes, actual_record_bytes)
        self.assertEqual(projection.failed_terminal_projection_bytes, 179_661_641)
        self.assertEqual(
            projection.candidate_bearing_conservative_upper_bytes, 187_863_577
        )
        self.assertEqual(projection.frozen_test_instance_count, 2_234)
        self.assertEqual(checkpoint_measurement.call_count, 1)
        self.assertEqual(context_units.call_count, 1)

        with patch.object(
            self.smoke,
            "_checkpoint_record_size",
            side_effect=ValueError("corrupt checkpoint measurement"),
        ):
            with self.assertRaisesRegex(ValueError, "corrupt checkpoint"):
                self.smoke.estimate_disk_projection(
                    context,
                    (result,),
                    normal_train_series_count=100,
                    measured_smoke_artifact_bytes=10,
                )
        with patch.object(
            self.smoke,
            "_schema_upper_units",
            side_effect=ValueError("corrupt context units"),
        ):
            with self.assertRaisesRegex(ValueError, "corrupt context"):
                self.smoke.estimate_disk_projection(
                    context,
                    (result,),
                    normal_train_series_count=100,
                    measured_smoke_artifact_bytes=10,
                )
        with patch.object(self.smoke, "_checkpoint_record_size", return_value=1):
            altered_record = self.smoke.estimate_disk_projection(
                context,
                (result,),
                normal_train_series_count=100,
                measured_smoke_artifact_bytes=10,
            )
        with patch.object(self.smoke, "_schema_upper_units", return_value=(1, 1)):
            altered_units = self.smoke.estimate_disk_projection(
                context,
                (result,),
                normal_train_series_count=100,
                measured_smoke_artifact_bytes=10,
            )
        self.assertNotEqual(
            altered_record.failed_terminal_projection_bytes,
            projection.failed_terminal_projection_bytes,
        )
        self.assertNotEqual(
            altered_units.candidate_bearing_conservative_upper_bytes,
            projection.candidate_bearing_conservative_upper_bytes,
        )

    def test_smoke_write_handles_partial_writes_and_cleans_failed_output(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            real_write = __import__("os").write

            def one_byte(descriptor: int, payload: bytes) -> int:
                return real_write(descriptor, payload[:1])

            with patch.object(self.smoke.os, "write", side_effect=one_byte):
                self.smoke._write_smoke(output, {"value": 1})
            self.assertEqual(
                (output / self.smoke.SMOKE_FILE).read_text(encoding="utf-8"),
                '{"value":1}\n',
            )

        with TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            with patch.object(
                self.smoke.os, "write", side_effect=OSError("injected failure")
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    self.smoke._write_smoke(output, {"value": 1})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
