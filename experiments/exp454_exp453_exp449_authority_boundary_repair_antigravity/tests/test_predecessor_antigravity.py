"""Label-free real-authority and mutation tests for the Exp454 normalize boundary."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import ast
import inspect
import sqlite3
import tempfile
import unittest
from unittest.mock import patch


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PACKAGE_ANTIGRAVITY = ROOT_ANTIGRAVITY / "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity"
EXP453_STOPPED_SHA256_ANTIGRAVITY = "1b6b2ea01c0c4dde9128051e57c764d32cbdf8bc1af7fa0115f19c7a10181d97"
EXP453_READINESS_SHA256_ANTIGRAVITY = "38607b6c8cad785ca8cd0830dacd42775d43f9175a67ed01d487657a3c44764e"
EXP453_EVALUATE_SHA256_ANTIGRAVITY = "caa705be90ddd56a00c53d5eb94a90604768dac0a80274846adce2061349b3b6"
EXP453_FAILURE_AUDIT_SHA256_ANTIGRAVITY = "97e7bea79bd1a830b925e7ffa2234e4e34725ac6734e080216f409688cb58276"
EXP449_PREDICTION_SHA256_ANTIGRAVITY = "a6054a44fc4e78f354dcf61d2e003284804680cd4197405364fcb5ef8d822e0d"


def _block_test_labels_antigravity(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("TEST labels were queried")


class PredecessorAntigravityTest(unittest.TestCase):
    def test_real_exp449_authority_normalizes_without_test_fixture_conversion_antigravity(self) -> None:
        """Break caught: production conversion is missing or depends on a test fixture."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import (
            capture_exp448_authority_antigravity,
        )
        from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import (
            DatasetBaselineAntigravity as ForeignBaselineAntigravity,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.contracts_antigravity import (
            ConfusionAntigravity,
            DatasetBaselineAntigravity,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.predecessor_antigravity import (
            NormalizedTargetAuthorityAntigravity,
            normalize_exp449_target_authority_antigravity,
        )

        source = inspect.getsource(normalize_exp449_target_authority_antigravity)
        self.assertNotIn("tests.", source)
        self.assertNotIn("fixture", source.casefold())
        self.assertNotIn("TEST", source)

        with patch.object(sqlite3, "connect", _block_test_labels_antigravity):
            raw = capture_exp448_authority_antigravity()
            normalized = normalize_exp449_target_authority_antigravity(raw)

        self.assertIs(type(normalized), NormalizedTargetAuthorityAntigravity)
        self.assertEqual(len(normalized.target_dataset_names), 48)
        self.assertEqual(len(normalized.target_baseline_rows), 48)
        self.assertIs(type(normalized.target_baseline_rows[0]), DatasetBaselineAntigravity)
        self.assertIs(type(normalized.baseline_confusion), ConfusionAntigravity)
        self.assertIsNot(type(normalized.target_baseline_rows[0]), ForeignBaselineAntigravity)
        self.assertEqual(normalized.target_dataset_names, raw.target_dataset_names)
        self.assertEqual(normalized.target_baseline_rows[0].dataset_order, raw.target_baseline_rows[0].dataset_order)
        self.assertEqual(normalized.target_baseline_rows[0].family, raw.target_baseline_rows[0].family)
        self.assertEqual(normalized.target_baseline_rows[0].stable_candidates, raw.target_baseline_rows[0].stable_candidates)
        self.assertEqual(
            (
                normalized.baseline_confusion.tp,
                normalized.baseline_confusion.fp,
                normalized.baseline_confusion.fn,
                normalized.baseline_confusion.tn,
            ),
            (
                raw.baseline_confusion.tp,
                raw.baseline_confusion.fp,
                raw.baseline_confusion.fn,
                raw.baseline_confusion.tn,
            ),
        )
        self.assertEqual(
            (
                normalized.target_baseline_rows[0].confusion.tp,
                normalized.target_baseline_rows[0].confusion.fp,
                normalized.target_baseline_rows[0].confusion.fn,
                normalized.target_baseline_rows[0].confusion.tn,
            ),
            (
                raw.target_baseline_rows[0].confusion.tp,
                raw.target_baseline_rows[0].confusion.fp,
                raw.target_baseline_rows[0].confusion.fn,
                raw.target_baseline_rows[0].confusion.tn,
            ),
        )

    def test_mutated_raw_authority_fields_fail_normalization_antigravity(self) -> None:
        """Break caught: dataset order, family, candidates, or confusion drift is copied through."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import (
            capture_exp448_authority_antigravity,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.predecessor_antigravity import (
            normalize_exp449_target_authority_antigravity,
        )

        with patch.object(sqlite3, "connect", _block_test_labels_antigravity):
            raw = capture_exp448_authority_antigravity()
        first = raw.target_baseline_rows[0]
        other_family = "Yoga" if first.family != "Yoga" else "Crop"
        cases = (
            (
                "dataset order",
                replace(
                    raw,
                    target_dataset_names=(raw.target_dataset_names[1], raw.target_dataset_names[0], *raw.target_dataset_names[2:]),
                    target_baseline_rows=(raw.target_baseline_rows[1], raw.target_baseline_rows[0], *raw.target_baseline_rows[2:]),
                ),
            ),
            (
                "family",
                replace(raw, target_baseline_rows=(replace(first, family=other_family), *raw.target_baseline_rows[1:])),
            ),
            (
                "stable candidates",
                replace(
                    raw,
                    target_baseline_rows=(replace(first, stable_candidates=frozenset({999999})), *raw.target_baseline_rows[1:]),
                ),
            ),
            (
                "baseline confusion",
                replace(
                    raw,
                    target_baseline_rows=(
                        replace(first, confusion=replace(first.confusion, fn=first.confusion.fn + 1)),
                        *raw.target_baseline_rows[1:],
                    ),
                ),
            ),
            (
                "global confusion",
                replace(raw, baseline_confusion=replace(raw.baseline_confusion, fn=raw.baseline_confusion.fn + 1)),
            ),
        )
        for label, mutated in cases:
            with self.subTest(label=label), patch.object(sqlite3, "connect", _block_test_labels_antigravity):
                with self.assertRaises(ValueError):
                    normalize_exp449_target_authority_antigravity(mutated)

    def test_capture_normalized_uses_real_exp449_capture_antigravity(self) -> None:
        """Break caught: production capture is fixture-only or skips the single normalize boundary."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.contracts_antigravity import (
            DatasetBaselineAntigravity,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.predecessor_antigravity import (
            capture_normalized_target_authority_antigravity,
            normalize_exp449_target_authority_antigravity,
        )

        source = inspect.getsource(capture_normalized_target_authority_antigravity)
        self.assertIn("capture_exp448_authority_antigravity", source)
        self.assertIn("normalize_exp449_target_authority_antigravity", source)
        self.assertNotIn("tests.", source)
        self.assertNotIn("fixture", source.casefold())
        with patch.object(sqlite3, "connect", _block_test_labels_antigravity):
            normalized = capture_normalized_target_authority_antigravity()
        self.assertEqual(len(normalized.target_dataset_names), 48)
        self.assertEqual(len(normalized.target_baseline_rows), 48)
        self.assertIs(type(normalized.target_baseline_rows[0]), DatasetBaselineAntigravity)
        with patch.object(sqlite3, "connect", _block_test_labels_antigravity):
            again = normalize_exp449_target_authority_antigravity(
                __import__(
                    "experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity",
                    fromlist=["capture_exp448_authority_antigravity"],
                ).capture_exp448_authority_antigravity()
            )
        self.assertEqual(normalized, again)

    def test_normalize_function_exists_only_in_predecessor_antigravity(self) -> None:
        """Break caught: a second conversion is added outside the predecessor boundary."""
        hits: list[str] = []
        for path in sorted(PACKAGE_ANTIGRAVITY.rglob("*_antigravity.py")):
            if "tests" in path.parts:
                continue
            if "def normalize_exp449_target_authority_antigravity" in path.read_text(encoding="utf-8"):
                hits.append(path.name)
        self.assertEqual(hits, ["predecessor_antigravity.py"])

    def test_predecessor_binds_exp453_and_exp449_hashes_antigravity(self) -> None:
        """Break caught: predecessor forgets the immutable Exp453 failure or Exp449 prediction hashes."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.predecessor_antigravity import (
            EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY,
            EXP453_EVALUATE_RECEIPT_SHA256_ANTIGRAVITY,
            EXP453_FAILURE_AUDIT_SHA256_ANTIGRAVITY,
            EXP453_READINESS_RECEIPT_SHA256_ANTIGRAVITY,
            EXP453_STOPPED_RECEIPT_SHA256_ANTIGRAVITY,
        )

        self.assertEqual(EXP453_STOPPED_RECEIPT_SHA256_ANTIGRAVITY, EXP453_STOPPED_SHA256_ANTIGRAVITY)
        self.assertEqual(EXP453_READINESS_RECEIPT_SHA256_ANTIGRAVITY, EXP453_READINESS_SHA256_ANTIGRAVITY)
        self.assertEqual(EXP453_EVALUATE_RECEIPT_SHA256_ANTIGRAVITY, EXP453_EVALUATE_SHA256_ANTIGRAVITY)
        self.assertEqual(EXP453_FAILURE_AUDIT_SHA256_ANTIGRAVITY, EXP453_FAILURE_AUDIT_SHA256_ANTIGRAVITY)
        self.assertEqual(EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY, EXP449_PREDICTION_SHA256_ANTIGRAVITY)

    def test_predecessor_source_does_not_import_test_or_label_modules_antigravity(self) -> None:
        """Break caught: the normalize boundary imports TEST labels or a fixture converter."""
        tree = ast.parse((PACKAGE_ANTIGRAVITY / "predecessor_antigravity.py").read_text(encoding="utf-8"))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        forbidden = {"sqlite3", "tests", "mlx", "torch"}
        self.assertFalse(any(name == blocked or f".{blocked}" in name or name.endswith(".tests") for name in imported for blocked in forbidden))
        self.assertFalse(any("test_fixture" in name or "tests." in name for name in imported))


_COMMIT_ANTIGRAVITY = "a" * 40


def _canonical_antigravity(value: object) -> bytes:
    from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.artifacts_antigravity import (
        canonical_json_bytes_antigravity,
    )

    return canonical_json_bytes_antigravity(value)


class SyntheticExp453PredecessorFixtureAntigravity:
    """Build label-free Exp453 terminal evidence, Exp449 prediction bytes, and normalized target."""

    def __init__(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import (
            prediction_antigravity as old_prediction,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.contracts_antigravity import (
            ConfusionAntigravity,
            DatasetBaselineAntigravity,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.predecessor_antigravity import (
            NormalizedTargetAuthorityAntigravity,
        )

        self.temporary = tempfile.TemporaryDirectory(prefix="exp454_predecessor_antigravity_")
        self.parent = Path(self.temporary.name)
        self.prediction = self.parent / "prediction_antigravity"
        self.receipts = self.parent / "receipts_antigravity"
        self.audit = self.parent / "failure_audit_antigravity.json"
        self.published_evaluation = self.parent / "published_evaluation_antigravity"
        old_prediction.run_synthetic_lifecycle_antigravity(self.prediction, _COMMIT_ANTIGRAVITY)
        capture = old_prediction.capture_frozen_prediction_authority_antigravity(root=self.prediction)
        self.prediction_aggregate = capture.authority.aggregate_sha256
        self.receipts.mkdir(mode=0o700)
        readiness = {
            "aeon_version": "1.4.0",
            "evaluation_root_identity": None,
            "event": "complete",
            "executable": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
            "experiment_id": "Exp453",
            "numpy_version": "2.3.5",
            "ordinal": 0,
            "permanent_stop": False,
            "phase": "readiness",
            "prefix": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314",
            "previous_leaf_sha256": None,
            "python_version": "3.14.6",
            "reviewed_commit": _COMMIT_ANTIGRAVITY,
            "run_identity": "exp453-exp452-fixed-interpreter-binding-repair-antigravity-20260813",
            "schema": "exp453_lifecycle_receipt",
            "sklearn_version": "1.7.2",
        }
        evaluate = {
            **readiness,
            "event": "started",
            "ordinal": 1,
            "phase": "evaluate",
            "previous_leaf_sha256": hashlib.sha256(_canonical_antigravity(readiness)).hexdigest(),
        }
        stopped = {
            **evaluate,
            "event": "stopped",
            "ordinal": 2,
            "permanent_stop": True,
            "previous_leaf_sha256": hashlib.sha256(_canonical_antigravity(evaluate)).hexdigest(),
        }
        (self.receipts / "readiness_complete_antigravity.json").write_bytes(_canonical_antigravity(readiness))
        (self.receipts / "evaluate_started_antigravity.json").write_bytes(_canonical_antigravity(evaluate))
        (self.receipts / "stopped_antigravity.json").write_bytes(_canonical_antigravity(stopped))
        self.readiness_sha = hashlib.sha256((self.receipts / "readiness_complete_antigravity.json").read_bytes()).hexdigest()
        self.evaluate_sha = hashlib.sha256((self.receipts / "evaluate_started_antigravity.json").read_bytes()).hexdigest()
        self.receipt_sha = hashlib.sha256((self.receipts / "stopped_antigravity.json").read_bytes()).hexdigest()
        audit = {
            "evaluation_root": str(self.published_evaluation),
            "permanent_stop": True,
            "phase": "evaluate",
            "primary_error": "ValueError: Exp448 target identity differs",
            "recovery_state": "stopped",
            "reviewed_commit": _COMMIT_ANTIGRAVITY,
            "schema": "exp453.lifecycle.failure",
        }
        self.audit.write_bytes(_canonical_antigravity(audit))
        self.audit_sha = hashlib.sha256(self.audit.read_bytes()).hexdigest()
        names = tuple(f"dataset_{index}" for index in range(48))
        rows = tuple(
            DatasetBaselineAntigravity(index, "Crop", frozenset({0}), ConfusionAntigravity(0, 0, 0, 1))
            for index in range(48)
        )
        self.normalized = NormalizedTargetAuthorityAntigravity(names, rows, ConfusionAntigravity(0, 0, 0, 48))
        self.events: list[str] = []

    def capture_normalized_target_antigravity(self):  # type: ignore[no-untyped-def]
        self.events.append("raw_target_capture")
        self.events.append("normalization")
        return self.normalized

    def dependencies_antigravity(self):  # type: ignore[no-untyped-def]
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.predecessor_antigravity import (
            _PredecessorDependenciesAntigravity,
        )

        return _PredecessorDependenciesAntigravity(
            prediction_root=self.prediction,
            exp453_receipt_directory=self.receipts,
            exp453_failure_audit=self.audit,
            exp453_evaluation_root=self.published_evaluation,
            expected_prediction_aggregate_sha256=self.prediction_aggregate,
            expected_exp453_stopped_receipt_sha256=self.receipt_sha,
            expected_exp453_readiness_receipt_sha256=self.readiness_sha,
            expected_exp453_evaluate_receipt_sha256=self.evaluate_sha,
            expected_exp453_audit_sha256=self.audit_sha,
            capture_normalized_target=self.capture_normalized_target_antigravity,
        )

    def close_antigravity(self) -> None:
        self.temporary.cleanup()


class PredecessorCaptureAntigravityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SyntheticExp453PredecessorFixtureAntigravity()

    def tearDown(self) -> None:
        self.fixture.close_antigravity()

    def test_two_complete_captures_issue_normalized_nonmetric_authority_antigravity(self) -> None:
        """Break caught: missing normalized target or qualified metrics can authorize labels."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )

        authority = predecessor._capture_predecessors_with_dependencies_antigravity(
            self.fixture.dependencies_antigravity()
        )
        self.assertEqual(authority.dataset_count, 48)
        self.assertEqual(authority.prediction_count, 96)
        self.assertEqual(authority.exp453_stopped_receipt_sha256, self.fixture.receipt_sha)
        self.assertEqual(authority.exp453_failure_audit_sha256, self.fixture.audit_sha)
        self.assertTrue(authority.exp453_evaluation_root_absent)
        self.assertEqual(authority.prediction_aggregate_sha256, self.fixture.prediction_aggregate)
        self.assertEqual(len(authority.dataset_order), 48)
        self.assertEqual(len(authority.normalized_target.target_dataset_names), 48)
        self.assertFalse(any("qualified" in name.casefold() for name in dir(authority)))
        self.assertEqual(
            predecessor.validate_predecessors_antigravity(authority),
            authority.dataset_order,
        )

    def test_exp453_hash_or_present_root_fails_without_prediction_confusion_antigravity(self) -> None:
        """Break caught: Exp453 evidence drift is reported as an Exp449 prediction failure."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )

        with self.subTest(case="receipt"), self.assertRaisesRegex(ValueError, "Exp453 stopped receipt"):
            predecessor._capture_predecessors_with_dependencies_antigravity(
                replace(self.fixture.dependencies_antigravity(), expected_exp453_stopped_receipt_sha256="0" * 64)
            )
        with self.subTest(case="audit"), self.assertRaisesRegex(ValueError, "Exp453 failure audit"):
            predecessor._capture_predecessors_with_dependencies_antigravity(
                replace(self.fixture.dependencies_antigravity(), expected_exp453_audit_sha256="0" * 64)
            )
        self.fixture.published_evaluation.mkdir()
        with self.subTest(case="root"), self.assertRaisesRegex(ValueError, "evaluation root"):
            predecessor._capture_predecessors_with_dependencies_antigravity(self.fixture.dependencies_antigravity())

    def test_exp449_prediction_bytes_fail_separately_antigravity(self) -> None:
        """Break caught: prediction drift is reported as Exp453 failure evidence."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )

        dependencies = replace(
            self.fixture.dependencies_antigravity(),
            expected_prediction_aggregate_sha256="1" * 64,
        )
        with self.assertRaisesRegex(ValueError, "Exp449 prediction"):
            predecessor._capture_predecessors_with_dependencies_antigravity(dependencies)

    def test_second_capture_drift_fails_without_issuing_authority_antigravity(self) -> None:
        """Break caught: the first snapshot alone authorizes labels after predecessor drift."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )

        def drift() -> None:
            self.fixture.audit.write_bytes(self.fixture.audit.read_bytes() + b" ")

        dependencies = replace(self.fixture.dependencies_antigravity(), between_captures=drift)
        with self.assertRaisesRegex(ValueError, "independent captures"):
            predecessor._capture_predecessors_with_dependencies_antigravity(dependencies)


class PredecessorLabelBoundaryAntigravityTest(unittest.TestCase):
    def test_label_opener_is_unreachable_before_evidence_prediction_and_normalization_antigravity(self) -> None:
        """Break caught: label opener can be called before Exp453 evidence, Exp449 prediction, raw capture, and normalize succeed."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            sources_antigravity as sources,
        )

        fixture = SyntheticExp453PredecessorFixtureAntigravity()
        try:
            reached: list[str] = []
            dependencies = sources._LabelDependenciesAntigravity(
                verify_database=lambda: reached.append("database"),  # type: ignore[arg-type,return-value]
                open_connection=lambda _descriptor: reached.append("connection"),  # type: ignore[arg-type,return-value]
                assert_database_unchanged=lambda _descriptor, _identity: None,
                close_descriptor=lambda _descriptor: None,
                audit_sink=lambda _audit: None,
            )
            with self.assertRaisesRegex(ValueError, "authority"):
                sources._open_test_labels_once_with_dependencies_antigravity(object(), dependencies)
            self.assertEqual(reached, [])
            self.assertEqual(fixture.events, [])

            with self.assertRaisesRegex(ValueError, "Exp453"):
                predecessor._capture_predecessors_with_dependencies_antigravity(
                    replace(fixture.dependencies_antigravity(), expected_exp453_stopped_receipt_sha256="0" * 64)
                )
            self.assertEqual(fixture.events, [])
            with self.assertRaisesRegex(ValueError, "authority"):
                sources._open_test_labels_once_with_dependencies_antigravity(object(), dependencies)
            self.assertEqual(reached, [])

            with self.assertRaisesRegex(ValueError, "Exp449 prediction"):
                predecessor._capture_predecessors_with_dependencies_antigravity(
                    replace(fixture.dependencies_antigravity(), expected_prediction_aggregate_sha256="1" * 64)
                )
            self.assertEqual(fixture.events, [])
            with self.assertRaisesRegex(ValueError, "authority"):
                sources._open_test_labels_once_with_dependencies_antigravity(object(), dependencies)
            self.assertEqual(reached, [])

            def fail_normalize() -> object:
                fixture.events.append("raw_target_capture")
                fixture.events.append("normalization")
                raise ValueError("Exp449 target authority differs")

            with self.assertRaisesRegex(ValueError, "target authority"):
                predecessor._capture_predecessors_with_dependencies_antigravity(
                    replace(fixture.dependencies_antigravity(), capture_normalized_target=fail_normalize)
                )
            self.assertEqual(fixture.events, ["raw_target_capture", "normalization"])
            with self.assertRaisesRegex(ValueError, "authority"):
                sources._open_test_labels_once_with_dependencies_antigravity(object(), dependencies)
            self.assertEqual(reached, [])

            fixture.events.clear()
            authority = predecessor._capture_predecessors_with_dependencies_antigravity(
                fixture.dependencies_antigravity()
            )
            self.assertEqual(fixture.events, ["raw_target_capture", "normalization", "raw_target_capture", "normalization"])
            with self.assertRaises(Exception):
                sources._open_test_labels_once_with_dependencies_antigravity(
                    authority,
                    sources._LabelDependenciesAntigravity(
                        verify_database=lambda: (_ for _ in ()).throw(RuntimeError("synthetic setup")),
                        open_connection=lambda _descriptor: None,
                        assert_database_unchanged=lambda _descriptor, _identity: None,
                        close_descriptor=lambda _descriptor: None,
                    ),
                )
            self.assertEqual(len(authority.normalized_target.target_dataset_names), 48)
        finally:
            fixture.close_antigravity()


if __name__ == "__main__":
    unittest.main()
