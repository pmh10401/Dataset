"""TDD coverage for the hash-gated Exp435 retrospective evaluator."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import io
import json
import os
import sqlite3
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from hashlib import sha256
from unittest.mock import patch

from experiments.exp435_train_only_consensus_recovery_antigravity.evaluation_antigravity import (
    BootstrapResult,
    ConfidenceInterval,
    ConfusionCounts,
    DatasetComparison,
    SuccessDecision,
    decide_exp435_success,
    paired_dataset_bootstrap,
)
from experiments.exp435_train_only_consensus_recovery_antigravity import (
    cli_antigravity as cli_module,
    evaluation_antigravity as evaluation_module,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.cli_antigravity import (
    _CliDependencies,
    main,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.reporting_antigravity import (
    PublicationExpectations,
    publish_exp435,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_reporting_antigravity import (
    _runtime,
)


class Exp435EvaluationTests(unittest.TestCase):
    def _rows(self) -> tuple[DatasetComparison, ...]:
        # D0 is rescued, D1 keeps a true positive and adds no FP, and D2 is a
        # strong-baseline regression.  These rows make the paired nature of
        # the bootstrap visible without involving TEST labels.
        return (
            DatasetComparison(
                dataset="D0",
                test_size=4,
                baseline=ConfusionCounts(tp=0, fp=0, fn=1, tn=3),
                combined=ConfusionCounts(tp=1, fp=0, fn=0, tn=3),
            ),
            DatasetComparison(
                dataset="D1",
                test_size=4,
                baseline=ConfusionCounts(tp=1, fp=0, fn=0, tn=3),
                combined=ConfusionCounts(tp=1, fp=0, fn=0, tn=3),
            ),
            DatasetComparison(
                dataset="D2",
                test_size=8,
                baseline=ConfusionCounts(tp=4, fp=0, fn=0, tn=4),
                combined=ConfusionCounts(tp=4, fp=1, fn=0, tn=3),
            ),
        )

    def test_hand_counts_rescued_and_regression_properties(self) -> None:
        rows = self._rows()
        self.assertEqual(rows[0].baseline.f1, 0.0)
        self.assertEqual(rows[0].combined.f1, 1.0)
        self.assertTrue(rows[0].rescued)
        self.assertTrue(rows[2].regressed)
        self.assertTrue(rows[2].strong_baseline)

    def test_dataset_comparison_rejects_lost_baseline_true_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "monotonic"):
            DatasetComparison(
                dataset="lost",
                test_size=8,
                baseline=ConfusionCounts(tp=4, fp=0, fn=0, tn=4),
                combined=ConfusionCounts(tp=3, fp=1, fn=1, tn=3),
            )

    def _evaluation(self) -> object:
        rows = self._rows()
        baseline = ConfusionCounts(
            tp=sum(row.baseline.tp for row in rows),
            fp=sum(row.baseline.fp for row in rows),
            fn=sum(row.baseline.fn for row in rows),
            tn=sum(row.baseline.tn for row in rows),
        )
        combined = ConfusionCounts(
            tp=sum(row.combined.tp for row in rows),
            fp=sum(row.combined.fp for row in rows),
            fn=sum(row.combined.fn for row in rows),
            tn=sum(row.combined.tn for row in rows),
        )
        bootstrap = paired_dataset_bootstrap(rows, repetitions=8, seed=123)
        return evaluation_module.Exp435Evaluation(
            dataset_count=len(rows),
            candidate_count=3,
            added_fp=combined.fp - baseline.fp,
            dataset_rows=rows,
            baseline_confusion=baseline,
            combined_confusion=combined,
            micro_f1_delta=combined.f1 - baseline.f1,
            macro_f1_delta=sum(row.combined_f1 - row.baseline_f1 for row in rows) / len(rows),
            micro_f1_ci=bootstrap.micro_f1_ci,
            macro_f1_ci=bootstrap.macro_f1_ci,
            rescued_datasets=tuple(row.dataset for row in rows if row.rescued),
            regressed_datasets=tuple(row.dataset for row in rows if row.regressed),
            strong_baseline_regressions=tuple(row.dataset for row in rows if row.regressed and row.strong_baseline),
            prediction_sha256="a" * 64,
            manifest_sha256="b" * 64,
            bootstrap=bootstrap,
        )

    def test_evaluation_payload_has_stable_evaluated_state(self) -> None:
        payload = self._evaluation().to_payload()
        self.assertEqual(payload["state"], "evaluated")

    def test_injected_cli_returns_zero_for_evaluation_payload(self) -> None:
        evaluation = self._evaluation()
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "output"
            checkpoint = parent / "checkpoint"
            output.mkdir()
            checkpoint.mkdir()
            dependencies = _CliDependencies(
                output=output,
                checkpoint=checkpoint,
                readiness=parent / "readiness.json",
                current_head=lambda: "a" * 40,
                readiness_call=lambda: {},
                execute=lambda _mode: {},
                inspect=lambda: {},
                evaluate=evaluation.to_payload,
            )
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["evaluate"], _dependencies=dependencies)
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["state"], "evaluated")

    def test_bootstrap_is_paired_deterministic_and_frozen(self) -> None:
        first = paired_dataset_bootstrap(self._rows())
        second = paired_dataset_bootstrap(self._rows())
        self.assertIsInstance(first, BootstrapResult)
        self.assertEqual(first.repetitions, 20_000)
        self.assertEqual(first.seed, 43_520_260_806)
        self.assertEqual(first.to_payload(), second.to_payload())
        self.assertEqual(first.micro_f1_delta_ci, first.micro_f1_ci)
        self.assertEqual(first.macro_f1_delta_ci, first.macro_f1_ci)
        self.assertLessEqual(first.micro_f1_ci.lower, first.micro_f1_ci.upper)
        self.assertLessEqual(first.macro_f1_ci.lower, first.macro_f1_ci.upper)

    def test_decision_requires_every_gate(self) -> None:
        rows = self._rows()
        bootstrap = paired_dataset_bootstrap(rows, repetitions=16, seed=123)
        decision = decide_exp435_success(
            dataset_count=1117,
            candidate_count=3,
            rescued_datasets=tuple("R" + str(index) for index in range(10)),
            added_fp=0,
            micro_f1_delta=0.1,
            macro_f1_delta=0.1,
            micro_f1_ci=ConfidenceInterval(0.01, 0.2),
            macro_f1_ci=ConfidenceInterval(0.01, 0.2),
            prediction_label_access_count=0,
        )
        self.assertIsInstance(decision, SuccessDecision)
        self.assertTrue(decision.passed)
        failed = decide_exp435_success(
            dataset_count=1117,
            candidate_count=3,
            rescued_datasets=tuple("R" + str(index) for index in range(10)),
            added_fp=31,
            micro_f1_delta=0.1,
            macro_f1_delta=0.1,
            micro_f1_ci=ConfidenceInterval(0.01, 0.2),
            macro_f1_ci=ConfidenceInterval(0.01, 0.2),
            prediction_label_access_count=0,
        )
        self.assertFalse(failed.passed)
        self.assertIn("added_false_positive_gate", failed.reasons)

    def test_nonpositive_point_estimates_are_explicit_failures(self) -> None:
        common = dict(
            dataset_count=1117,
            candidate_count=0,
            rescued_datasets=tuple("D" + str(i) for i in range(10)),
            added_fp=0,
            micro_f1_delta=0.0,
            macro_f1_delta=-0.1,
            micro_f1_ci=ConfidenceInterval(0.01, 0.2),
            macro_f1_ci=ConfidenceInterval(0.01, 0.2),
            prediction_label_access_count=0,
        )
        decision = decide_exp435_success(**common)
        self.assertFalse(decision.passed)
        self.assertIn("micro_f1_point_gate", decision.reasons)
        self.assertIn("macro_f1_point_gate", decision.reasons)

    def test_each_success_gate_fails_independently(self) -> None:
        common = dict(
            dataset_count=1117,
            candidate_count=100,
            rescued_datasets=tuple("R" + str(index) for index in range(10)),
            added_fp=30,
            micro_f1_delta=0.1,
            macro_f1_delta=0.1,
            micro_f1_ci=ConfidenceInterval(0.01, 0.2),
            macro_f1_ci=ConfidenceInterval(0.01, 0.2),
            prediction_label_access_count=0,
        )
        cases = {
            "rescued_dataset_gate": {"rescued_datasets": tuple("R" + str(index) for index in range(9))},
            "added_false_positive_gate": {"added_fp": 31},
            "micro_f1_point_gate": {"micro_f1_delta": 0.0},
            "macro_f1_point_gate": {"macro_f1_delta": 0.0},
            "micro_f1_ci_gate": {"micro_f1_ci": ConfidenceInterval(0.0, 0.2)},
            "macro_f1_ci_gate": {"macro_f1_ci": ConfidenceInterval(0.0, 0.2)},
            "candidate_budget_gate": {"candidate_count": 101},
            "dataset_count_gate": {"dataset_count": 1116},
            "prediction_label_access_gate": {"prediction_label_access_count": 1},
        }
        for reason, override in cases.items():
            with self.subTest(reason=reason):
                decision = decide_exp435_success(**{**common, **override})
                self.assertFalse(decision.passed)
                self.assertEqual(decision.reasons, (reason,))

    def test_evaluation_payload_bytes_are_deterministic(self) -> None:
        first = evaluation_module._evaluation_payloads(self._evaluation())
        second = evaluation_module._evaluation_payloads(self._evaluation())
        self.assertEqual(first, second)

    def test_label_loader_schema_rejects_noncanonical_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary"):
            evaluation_module._label_payload({"instance_indices": [0], "labels": [2]}, dataset="Demo")
        with self.assertRaisesRegex(ValueError, "canonical"):
            evaluation_module._label_payload({"instance_indices": [1, 0], "labels": [0, 1]}, dataset="Demo")

    def test_prediction_hash_failure_precedes_test_label_loader(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _runtime(root / "output", root / "checkpoint")
            publication = publish_exp435(
                store,
                root / "output",
                _expectations=PublicationExpectations(total=1),
            )
            prediction = publication.publication / "prediction_antigravity.jsonl"
            prediction.write_bytes(b'{"dataset":"tampered","instance_index":0}\n')
            calls: list[str] = []
            with self.assertRaisesRegex(ValueError, "prediction SHA256"):
                # Import locally to keep the test's high-level API obvious.
                from experiments.exp435_train_only_consensus_recovery_antigravity.evaluation_antigravity import evaluate_frozen_prediction

                evaluate_frozen_prediction(
                    publication=publication,
                    hard_coordinates={"Demo": ()},
                    label_loader=lambda _dataset: calls.append("opened"),
                    completed_store=store,
                    expected_datasets=1,
                )
            self.assertEqual(calls, [])

    def _fixture_labels(self, _dataset: str) -> dict[str, object]:
        return {
            "instance_indices": list(range(101)),
            "labels": [1] + [0] * 100,
        }

    def _fixture_publication(self, root: Path):
        output = root / "output"
        checkpoint = root / "checkpoint"
        store = _runtime(output, checkpoint, candidates=1)
        publication = publish_exp435(
            store,
            output,
            _expectations=PublicationExpectations(total=1),
        )
        return output, checkpoint, store, publication

    def _tree_signature(self, root: Path) -> dict[str, tuple[str, bytes | str | None]]:
        signature: dict[str, tuple[str, bytes | str | None]] = {}
        for path in sorted((root, *root.rglob("*"))):
            relative = "." if path == root else path.relative_to(root).as_posix()
            if path.is_symlink():
                signature[relative] = ("symlink", os.readlink(path))
            elif path.is_dir():
                signature[relative] = ("directory", None)
            elif path.is_file():
                signature[relative] = ("file", path.read_bytes())
            else:
                signature[relative] = ("other", None)
        return signature

    def test_end_to_end_evaluation_writes_eight_artifacts_and_preserves_source(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output, _checkpoint, store, publication = self._fixture_publication(root)
            before = self._tree_signature(output)
            evaluation_root = root / "evaluation"
            with patch.object(evaluation_module, "_exclusive_rename", os.rename):
                result = evaluation_module.evaluate_frozen_prediction(
                    publication,
                    hard_coordinates={"Demo": ()},
                    label_loader=self._fixture_labels,
                    completed_store=store,
                    expected_datasets=("Demo",),
                    evaluation_root=evaluation_root,
                )
            self.assertEqual(result.dataset_count, 1)
            self.assertEqual(result.rescued_datasets, ("Demo",))
            self.assertEqual(result.prediction_label_access_count, 0)
            self.assertEqual(
                {item.name for item in evaluation_root.iterdir()},
                set(evaluation_module._EVALUATION_FILES),
            )
            hashes = json.loads((evaluation_root / evaluation_module.EVALUATION_HASH_FILE).read_bytes())
            for name, digest in hashes["artifact_sha256"].items():
                self.assertEqual(sha256((evaluation_root / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(before, self._tree_signature(output))

    def test_source_tamper_after_label_access_fails_before_evaluation_publish(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output, _checkpoint, store, publication = self._fixture_publication(root)
            evaluation_root = root / "evaluation"

            def tampering_loader(dataset: str) -> dict[str, object]:
                progress = output / "progress_antigravity.json"
                progress.write_bytes(progress.read_bytes() + b"tampered")
                return self._fixture_labels(dataset)

            with patch.object(evaluation_module, "_exclusive_rename", os.rename):
                with self.assertRaisesRegex(ValueError, "frozen|source|publication"):
                    evaluation_module.evaluate_frozen_prediction(
                        publication,
                        hard_coordinates={"Demo": ()},
                        label_loader=tampering_loader,
                        completed_store=store,
                        expected_datasets=("Demo",),
                        evaluation_root=evaluation_root,
                    )
            self.assertFalse(evaluation_root.exists())

    def test_evaluation_root_is_no_clobber(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output, _checkpoint, store, publication = self._fixture_publication(root)
            evaluation_root = root / "evaluation"
            with patch.object(evaluation_module, "_exclusive_rename", os.rename):
                evaluation_module.evaluate_frozen_prediction(
                    publication,
                    hard_coordinates={"Demo": ()},
                    label_loader=self._fixture_labels,
                    completed_store=store,
                    expected_datasets=("Demo",),
                    evaluation_root=evaluation_root,
                )
            original = self._tree_signature(evaluation_root)
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluation_module.evaluate_frozen_prediction(
                    publication,
                    hard_coordinates={"Demo": ()},
                    label_loader=self._fixture_labels,
                    completed_store=store,
                    expected_datasets=("Demo",),
                    evaluation_root=evaluation_root,
                )
            self.assertEqual(original, self._tree_signature(evaluation_root))

    def test_helper_rejects_evaluation_root_inside_frozen_output_or_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output, checkpoint, store, publication = self._fixture_publication(root)
            for destination in (
                output / "nested-evaluation",
                output / evaluation_module.PUBLICATION_DIRECTORY / "nested-evaluation",
                checkpoint / "nested-evaluation",
            ):
                with self.subTest(destination=destination):
                    with self.assertRaisesRegex(ValueError, "sibling|nested"):
                        evaluation_module.evaluate_frozen_prediction(
                            publication,
                            hard_coordinates={"Demo": ()},
                            label_loader=lambda _dataset: self._fixture_labels("Demo"),
                            completed_store=store,
                            expected_datasets=("Demo",),
                            evaluation_root=destination,
                        )

    def test_production_entrypoint_accepts_only_fixed_sibling_paths(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            fixed_output = parent / "output"
            fixed_evaluation = parent / "evaluation"
            with patch.object(cli_module, "RUN_PARENT", parent), patch.object(cli_module, "FROZEN_OUTPUT", fixed_output), patch.object(cli_module, "FROZEN_EVALUATION", fixed_evaluation):
                with self.assertRaisesRegex(ValueError, "fixed"):
                    evaluation_module.evaluate_exp435(parent / "alternate", fixed_evaluation)
                with self.assertRaisesRegex(ValueError, "sibling|fixed"):
                    evaluation_module.evaluate_exp435(fixed_output, fixed_output)

    def test_frozen_exp404_and_exp291_maps_match(self) -> None:
        hard = evaluation_module._load_frozen_hard_coordinates()
        self.assertEqual(len(hard), 1117)
        self.assertEqual(sum(bool(indices) for indices in hard.values()), 888)
        self.assertEqual(sum(len(indices) for indices in hard.values()), 2218)

    def test_exp404_hash_mismatch_is_rejected_before_labels(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "exp404.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with patch.object(evaluation_module, "EXP404_HARD_PATH", path):
                with self.assertRaisesRegex(ValueError, "Exp404 Hard prediction SHA256"):
                    evaluation_module._load_frozen_hard_coordinates()

    def test_exp291_hash_mismatch_is_rejected_after_exp404_pin(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "exp291.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with patch.object(evaluation_module, "EXP291_HARD_PATH", path):
                with self.assertRaisesRegex(ValueError, "Exp291 Hard prediction SHA256"):
                    evaluation_module._load_frozen_hard_coordinates()

    def test_pinned_artifact_reader_keeps_opened_bytes_after_pathname_aba_replacement(self) -> None:
        """Replacing the name after ``open`` cannot change the captured bytes."""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "exp291.jsonl"
            original = b'{"dataset":"Original","automatic_hard_indices":[],"contains_test_labels":false}\n'
            replacement = b'{"dataset":"Replacement","automatic_hard_indices":[9],"contains_test_labels":false}\n'
            path.write_bytes(original)
            replacement_path = Path(directory) / "replacement.jsonl"
            replacement_path.write_bytes(replacement)
            real_open = evaluation_module.os.open

            def replace_after_open(target: os.PathLike[str] | str, flags: int, *args: object) -> int:
                descriptor = real_open(target, flags, *args)
                if Path(target) == path:
                    os.replace(replacement_path, path)
                return descriptor

            with patch.object(evaluation_module.os, "open", replace_after_open):
                try:
                    payload, digest = evaluation_module._read_pinned_bytes(path)
                except ValueError as error:
                    # Losing the only directory entry changes st_nlink and is
                    # deliberately rejected rather than silently reopening.
                    self.assertIn("linked regular file", str(error))
                else:
                    self.assertEqual(payload, original)
                    self.assertEqual(digest, sha256(original).hexdigest())
            self.assertEqual(path.read_bytes(), replacement)

    def test_hard_parser_consumes_captured_bytes_instead_of_reopening_pathname(self) -> None:
        """The Exp291 parser must receive the pinned payload, not read its path again."""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "exp291.jsonl"
            original = b'{"dataset":"Original","automatic_hard_indices":[],"contains_test_labels":false}\n'
            replacement = b'{"dataset":"Replacement","automatic_hard_indices":[9],"contains_test_labels":false}\n'
            path.write_bytes(original)

            def captured_reader(target: Path) -> tuple[bytes, str]:
                self.assertEqual(Path(target), path)
                path.write_bytes(replacement)
                return original, sha256(original).hexdigest()

            with patch.object(evaluation_module, "_read_pinned_bytes", captured_reader):
                self.assertEqual(
                    evaluation_module._parse_hard_file(path),
                    {"Original": ()},
                )
            self.assertEqual(path.read_bytes(), replacement)

    def test_sqlite_label_loader_uses_verified_session_and_selects_only_labels(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "labels.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT UNIQUE);
                CREATE TABLE instances (dataset_id INTEGER, split TEXT, instance_index INTEGER,
                    label TEXT, values_blob BLOB, labels_blob BLOB);
                INSERT INTO datasets VALUES (1, 'Demo');
                INSERT INTO instances VALUES (1, 'TEST', 0, '1', X'00', X'01');
                INSERT INTO instances VALUES (1, 'TEST', 1, '0', X'02', X'03');
                """
            )
            connection.commit()
            connection.close()
            calls: list[tuple[Path, str]] = []
            traces: list[str] = []
            original_connect = sqlite3.connect

            @contextmanager
            def verified_session(path: Path, expected_sha256: str):
                calls.append((Path(path), expected_sha256))
                opened = original_connect(path)
                opened.set_trace_callback(traces.append)
                try:
                    yield opened
                finally:
                    opened.close()

            with patch.object(evaluation_module, "verified_read_only_connection", verified_session):
                result = evaluation_module._load_test_labels(
                    "Demo",
                    database_path=database,
                    expected_sha256="d" * 64,
                )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls, [(database, "d" * 64)])
            self.assertEqual(result["labels"], [1, 0])
            sql = " ".join(traces).lower()
            self.assertIn("select i.instance_index", sql)
            self.assertIn("label", sql)
            self.assertNotIn("values_blob", sql)
            self.assertNotIn("labels_blob", sql)

    def test_private_evaluation_uses_one_pinned_database_session_after_all_artifact_gates(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "labels.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT UNIQUE);
                CREATE TABLE instances (dataset_id INTEGER, split TEXT, instance_index INTEGER,
                    label TEXT, values_blob BLOB, labels_blob BLOB);
                INSERT INTO datasets VALUES (1, 'Demo');
                """
            )
            connection.executemany(
                "INSERT INTO instances VALUES (1, 'TEST', ?, ?, X'00', X'01')",
                ((index, "1" if index == 0 else "0") for index in range(101)),
            )
            connection.commit()
            connection.close()
            output, _checkpoint, store, publication = self._fixture_publication(root)
            calls: list[tuple[Path, str]] = []
            closed: list[bool] = []
            original_connect = sqlite3.connect

            class ClosingConnection:
                def __init__(self, connection: sqlite3.Connection) -> None:
                    self.connection = connection

                def execute(self, *args: object, **kwargs: object):
                    return self.connection.execute(*args, **kwargs)

                def close(self) -> None:
                    self.connection.close()
                    closed.append(True)

            @contextmanager
            def verified_session(path: Path, expected_sha256: str):
                calls.append((Path(path), expected_sha256))
                wrapped = ClosingConnection(original_connect(path))
                try:
                    yield wrapped
                finally:
                    wrapped.close()

            with patch.object(evaluation_module, "verified_read_only_connection", verified_session), patch.object(sqlite3, "connect", side_effect=AssertionError("evaluator reopened SQLite by pathname")):
                evaluation_module.evaluate_frozen_prediction(
                    publication,
                    hard_coordinates={"Demo": ()},
                    completed_store=store,
                    expected_datasets=("Demo",),
                    database_path=database,
                )
            self.assertEqual(calls, [(database, evaluation_module.FROZEN_CONFIG.database_sha256)])
            self.assertEqual(closed, [True])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
