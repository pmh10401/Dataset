"""Synthetic dual-predecessor tests for Exp453 Task 3."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest


_COMMIT_ANTIGRAVITY = "a" * 40


def _canonical_antigravity(value: object) -> bytes:
    from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.artifacts_antigravity import (
        canonical_json_bytes_antigravity,
    )

    return canonical_json_bytes_antigravity(value)


class SyntheticDualPredecessorFixtureAntigravity:
    """Build label-free Exp452 failure evidence and Exp449 prediction bytes."""

    def __init__(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import (
            prediction_antigravity as old_prediction,
        )

        self.temporary = tempfile.TemporaryDirectory(prefix="exp453_predecessor_antigravity_")
        self.parent = Path(self.temporary.name)
        self.prediction = self.parent / "prediction_antigravity"
        self.receipts = self.parent / "receipts_antigravity"
        self.audit = self.parent / "failure_audit_antigravity.json"
        self.published_evaluation = self.parent / "published_evaluation_antigravity"
        old_prediction.run_synthetic_lifecycle_antigravity(self.prediction, _COMMIT_ANTIGRAVITY)
        capture = old_prediction.capture_frozen_prediction_authority_antigravity(root=self.prediction)
        self.prediction_aggregate = capture.authority.aggregate_sha256
        self.receipts.mkdir(mode=0o700)
        receipt = {
            "evaluation_root_identity": None,
            "event": "stopped",
            "experiment_id": "Exp452",
            "ordinal": 0,
            "permanent_stop": True,
            "phase": "readiness",
            "previous_leaf_sha256": None,
            "reviewed_commit": _COMMIT_ANTIGRAVITY,
            "run_identity": "exp452-exp449-index-order-repair-antigravity-20260813",
            "schema": "exp452_lifecycle_receipt",
        }
        (self.receipts / "stopped_antigravity.json").write_bytes(_canonical_antigravity(receipt))
        self.receipt_sha = hashlib.sha256((self.receipts / "stopped_antigravity.json").read_bytes()).hexdigest()
        audit = {
            "evaluation_root": str(self.published_evaluation),
            "permanent_stop": True,
            "phase": "readiness",
            "primary_error": "ValueError: Exp452 requires the exact fixed Python and NumPy runtime",
            "recovery_state": "stopped",
            "reviewed_commit": _COMMIT_ANTIGRAVITY,
            "schema": "exp452.lifecycle.failure",
        }
        self.audit.write_bytes(_canonical_antigravity(audit))
        self.audit_sha = hashlib.sha256(self.audit.read_bytes()).hexdigest()

    def dependencies_antigravity(self):  # type: ignore[no-untyped-def]
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.predecessor_antigravity import (
            _PredecessorDependenciesAntigravity,
        )

        return _PredecessorDependenciesAntigravity(
            prediction_root=self.prediction,
            exp452_receipt_directory=self.receipts,
            exp452_failure_audit=self.audit,
            exp452_evaluation_root=self.published_evaluation,
            expected_prediction_aggregate_sha256=self.prediction_aggregate,
            expected_exp452_receipt_sha256=self.receipt_sha,
            expected_exp452_audit_sha256=self.audit_sha,
        )

    def close_antigravity(self) -> None:
        self.temporary.cleanup()


class PredecessorAntigravityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SyntheticDualPredecessorFixtureAntigravity()

    def tearDown(self) -> None:
        self.fixture.close_antigravity()

    def test_two_complete_captures_issue_dual_nonmetric_authority_antigravity(self) -> None:
        """Break caught: one predecessor or qualified metrics can authorize labels."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )

        authority = predecessor._capture_predecessors_with_dependencies_antigravity(
            self.fixture.dependencies_antigravity()
        )
        self.assertEqual(authority.dataset_count, 48)
        self.assertEqual(authority.prediction_count, 96)
        self.assertEqual(authority.exp452_stopped_receipt_sha256, self.fixture.receipt_sha)
        self.assertEqual(authority.exp452_failure_audit_sha256, self.fixture.audit_sha)
        self.assertTrue(authority.exp452_evaluation_root_absent)
        self.assertEqual(authority.prediction_aggregate_sha256, self.fixture.prediction_aggregate)
        self.assertEqual(len(authority.dataset_order), 48)
        self.assertFalse(any("qualified" in name.casefold() for name in dir(authority)))
        self.assertFalse(hasattr(authority, "order_mismatch_count"))
        self.assertEqual(
            predecessor.validate_predecessors_antigravity(authority),
            authority.dataset_order,
        )

    def test_exp452_hash_or_present_root_fails_without_prediction_confusion_antigravity(self) -> None:
        """Break caught: Exp452 evidence drift is reported as an Exp449 prediction failure."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )

        with self.subTest(case="receipt"), self.assertRaisesRegex(ValueError, "Exp452 stopped receipt"):
            predecessor._capture_predecessors_with_dependencies_antigravity(
                replace(self.fixture.dependencies_antigravity(), expected_exp452_receipt_sha256="0" * 64)
            )
        with self.subTest(case="audit"), self.assertRaisesRegex(ValueError, "Exp452 failure audit"):
            predecessor._capture_predecessors_with_dependencies_antigravity(
                replace(self.fixture.dependencies_antigravity(), expected_exp452_audit_sha256="0" * 64)
            )
        self.fixture.published_evaluation.mkdir()
        with self.subTest(case="root"), self.assertRaisesRegex(ValueError, "evaluation root"):
            predecessor._capture_predecessors_with_dependencies_antigravity(self.fixture.dependencies_antigravity())

    def test_exp449_prediction_bytes_fail_separately_antigravity(self) -> None:
        """Break caught: prediction drift is reported as Exp452 failure evidence."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
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
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )

        def drift() -> None:
            self.fixture.audit.write_bytes(self.fixture.audit.read_bytes() + b" ")

        dependencies = replace(self.fixture.dependencies_antigravity(), between_captures=drift)
        with self.assertRaisesRegex(ValueError, "independent captures"):
            predecessor._capture_predecessors_with_dependencies_antigravity(dependencies)

    def test_label_opener_is_unreachable_before_two_equal_captures_antigravity(self) -> None:
        """Break caught: label opener can be called before two equal predecessor captures."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            sources_antigravity as sources,
        )

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

        original_audit = self.fixture.audit.read_bytes()

        def drift() -> None:
            self.fixture.audit.write_bytes(original_audit + b" ")

        with self.assertRaisesRegex(ValueError, "independent captures"):
            predecessor._capture_predecessors_with_dependencies_antigravity(
                replace(self.fixture.dependencies_antigravity(), between_captures=drift)
            )
        with self.assertRaisesRegex(ValueError, "authority"):
            sources._open_test_labels_once_with_dependencies_antigravity(object(), dependencies)
        self.assertEqual(reached, [])
        self.fixture.audit.write_bytes(original_audit)

        authority = predecessor._capture_predecessors_with_dependencies_antigravity(
            self.fixture.dependencies_antigravity()
        )
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

    def test_missing_symlink_multilink_and_publication_fail_closed_antigravity(self) -> None:
        """Break caught: one unsafe Exp452 evidence namespace survives to capability issuance."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )

        cases = ("missing", "symlink", "multilink", "published")
        for case in cases:
            with self.subTest(case=case):
                if case != cases[0]:
                    self.fixture.close_antigravity()
                    self.fixture = SyntheticDualPredecessorFixtureAntigravity()
                dependencies = self.fixture.dependencies_antigravity()
                if case == "missing":
                    self.fixture.audit.unlink()
                elif case == "symlink":
                    payload = self.fixture.parent / "audit_payload_antigravity.json"
                    self.fixture.audit.rename(payload)
                    self.fixture.audit.symlink_to(payload)
                elif case == "multilink":
                    os.link(self.fixture.audit, self.fixture.parent / "audit_hardlink_antigravity.json")
                else:
                    self.fixture.published_evaluation.mkdir()
                with self.assertRaises(ValueError):
                    predecessor._capture_predecessors_with_dependencies_antigravity(dependencies)


if __name__ == "__main__":
    unittest.main()
