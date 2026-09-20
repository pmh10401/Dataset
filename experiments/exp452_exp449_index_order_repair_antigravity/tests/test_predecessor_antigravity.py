"""Synthetic descriptor-authority tests for the stopped Exp449 predecessor."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


_COMMIT_ANTIGRAVITY = "a" * 40
_MISMATCHES_ANTIGRAVITY = 75


def _canonical_antigravity(value: object) -> bytes:
    from experiments.exp452_exp449_index_order_repair_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity

    return canonical_json_bytes_antigravity(value)


class SyntheticPredecessorFixtureAntigravity:
    """Build one label-free Exp449-shaped authority using only temporary files."""

    def __init__(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as old_lifecycle
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as old_prediction

        self.temporary = tempfile.TemporaryDirectory(prefix="exp452_predecessor_antigravity_")
        self.parent = Path(self.temporary.name)
        self.prediction = self.parent / "prediction_antigravity"
        self.receipts = self.parent / "receipts_antigravity"
        self.audit = self.parent / "failure_audit_antigravity.json"
        self.staging = self.parent / "staging_antigravity"
        self.published_evaluation = self.parent / "published_evaluation_antigravity"

        old_prediction.run_synthetic_lifecycle_antigravity(self.prediction, _COMMIT_ANTIGRAVITY)
        manifest_path = self.prediction / "prediction_manifest_antigravity.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for sequence, relative in enumerate(manifest["inventory"]):
            path = self.prediction / relative
            row = json.loads(path.read_text(encoding="utf-8"))
            row["indices"] = [0, 1] if sequence < _MISMATCHES_ANTIGRAVITY else []
            path.write_bytes(_canonical_antigravity(row))
        digests = {
            relative: hashlib.sha256((self.prediction / relative).read_bytes()).hexdigest()
            for relative in manifest["inventory"]
        }
        manifest["leaf_sha256"] = digests
        manifest["aggregate_sha256"] = hashlib.sha256("".join(digests[relative] for relative in manifest["inventory"]).encode("ascii")).hexdigest()
        manifest_path.write_bytes(_canonical_antigravity(manifest))
        capture = old_prediction.capture_frozen_prediction_authority_antigravity(root=self.prediction)
        self.prediction_aggregate = capture.authority.aggregate_sha256

        old_lifecycle.append_receipt_antigravity(
            self.receipts, "readiness", "complete", _COMMIT_ANTIGRAVITY,
            prediction_root=self.prediction, evaluation_root=self.published_evaluation,
        )
        old_lifecycle.append_receipt_antigravity(
            self.receipts, "predict", "started", _COMMIT_ANTIGRAVITY,
            prediction_root=self.prediction, evaluation_root=self.published_evaluation,
        )
        old_lifecycle.append_receipt_antigravity(
            self.receipts, "predict", "complete", _COMMIT_ANTIGRAVITY,
            prediction_root=self.prediction, evaluation_root=self.published_evaluation,
        )
        self.failure_root_destinations = (
            "/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_antigravity_20260812",
            "/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_evaluation_antigravity_20260812",
        )
        try:
            old_lifecycle.run_phase_antigravity(
                "evaluate",
                _COMMIT_ANTIGRAVITY,
                lambda: (_ for _ in ()).throw(
                    ValueError("frozen prediction indices differ from evidence")
                ),
                receipt_dir=self.receipts,
                audit_path=self.audit,
                prediction_root=self.prediction,
                evaluation_root=self.published_evaluation,
            )
        except ValueError as error:
            if str(error) != "frozen prediction indices differ from evidence":
                raise
        else:
            raise AssertionError("synthetic Exp449 evaluate failure did not occur")
        self._write_staging_antigravity(capture.canonical_rows)

    def _write_staging_antigravity(self, frozen_rows: bytes) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.evaluation_antigravity import _LEAVES_ANTIGRAVITY
        from experiments.exp452_exp449_index_order_repair_antigravity.config_antigravity import CANDIDATE_MATRIX_ANTIGRAVITY

        rows = [json.loads(line) for line in frozen_rows.splitlines()]
        by_dataset: dict[str, dict[str, object]] = {}
        methods = dict(CANDIDATE_MATRIX_ANTIGRAVITY)
        for sequence, row in enumerate(rows):
            dataset = row["dataset"]
            record = by_dataset.setdefault(dataset, {
                "dataset": dataset,
                "family": row["family"],
                "target_length": row["target_length"],
                "baseline_confusion": {"tp": 0, "fp": 0, "fn": 0, "tn": 1},
                "candidates": {},
            })
            evidence = list(row["indices"])
            if sequence < _MISMATCHES_ANTIGRAVITY:
                evidence.reverse()
            record["candidates"][row["method"]] = {
                "indices": evidence,
                "confusion": {"tp": 0, "fp": 0, "fn": 0, "tn": 1},
            }
        self.staging.mkdir(mode=0o700)
        evaluation = {
            "schema": "exp449",
            "reviewed_commit": _COMMIT_ANTIGRAVITY,
            "candidate_count": 16,
            "qualified_count": 0,
            "label_session_count": 1,
            "label_query_count": 48,
            "label_row_count": 72300,
            "routing_research_eligible": False,
            "dataset_evidence": list(by_dataset.values()),
        }
        payloads: dict[str, bytes] = {
            "context_antigravity.json": _canonical_antigravity({"schema": "exp449", "label_access_count": 1}),
            "compatibility_antigravity.json": _canonical_antigravity({"schema": "exp449.task6"}),
            "candidate_registry_antigravity.json": _canonical_antigravity([[family, list(family_methods)] for family, family_methods in CANDIDATE_MATRIX_ANTIGRAVITY]),
            "frozen_predictions_antigravity.jsonl": frozen_rows,
            "prediction_manifest_antigravity.json": _canonical_antigravity({"schema": "synthetic_exp449_manifest"}),
            "evaluation_antigravity.json": _canonical_antigravity(evaluation),
            "candidate_metrics_antigravity.jsonl": b"",
            "family_metrics_antigravity.jsonl": b"",
            "bootstrap_antigravity.json": _canonical_antigravity({"schema": "synthetic"}),
            "decision_antigravity.json": _canonical_antigravity({"schema": "synthetic"}),
            "dashboard_snapshot_antigravity.json": _canonical_antigravity({"schema": "synthetic", "status": "synthetic_complete"}),
        }
        for name in _LEAVES_ANTIGRAVITY[:-1]:
            (self.staging / name).write_bytes(payloads[name])
        artifact_map = {name: hashlib.sha256(payloads[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
        (self.staging / _LEAVES_ANTIGRAVITY[-1]).write_bytes(_canonical_antigravity(artifact_map))
        self.staging_artifact_sha = hashlib.sha256((self.staging / "artifact_sha256_antigravity.json").read_bytes()).hexdigest()
        self.staging_evaluation_sha = hashlib.sha256((self.staging / "evaluation_antigravity.json").read_bytes()).hexdigest()
        self.staging_dashboard_sha = hashlib.sha256((self.staging / "dashboard_snapshot_antigravity.json").read_bytes()).hexdigest()

    def dependencies_antigravity(self):  # type: ignore[no-untyped-def]
        from experiments.exp452_exp449_index_order_repair_antigravity.predecessor_antigravity import _PredecessorDependenciesAntigravity

        return _PredecessorDependenciesAntigravity(
            prediction_root=self.prediction,
            receipt_directory=self.receipts,
            failure_audit=self.audit,
            staging_root=self.staging,
            published_evaluation_root=self.published_evaluation,
            expected_prediction_aggregate_sha256=self.prediction_aggregate,
            expected_failure_audit_sha256=hashlib.sha256(self.audit.read_bytes()).hexdigest(),
            expected_staging_inode=self.staging.stat().st_ino,
            expected_staging_artifact_map_sha256=self.staging_artifact_sha,
            expected_staging_evaluation_sha256=self.staging_evaluation_sha,
            expected_staging_dashboard_snapshot_sha256=self.staging_dashboard_sha,
            expected_failure_root_destinations=self.failure_root_destinations,
        )

    def close_antigravity(self) -> None:
        self.temporary.cleanup()


class PredecessorAntigravityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SyntheticPredecessorFixtureAntigravity()

    def tearDown(self) -> None:
        self.fixture.close_antigravity()

    def test_two_complete_captures_issue_only_frozen_nonmetric_authority_antigravity(self) -> None:
        """Break caught: labels can be reached from one/partial capture or failed qualified metrics leak."""
        from experiments.exp452_exp449_index_order_repair_antigravity import predecessor_antigravity as predecessor

        authority = predecessor._capture_exp449_stopped_authority_with_dependencies_antigravity(
            self.fixture.dependencies_antigravity()
        )
        self.assertEqual(authority.dataset_count, 48)
        self.assertEqual(authority.prediction_count, 96)
        self.assertEqual(authority.order_mismatch_count, 75)
        self.assertEqual(authority.order_equal_set_count, 75)
        self.assertEqual((authority.label_session_count, authority.label_query_count, authority.label_row_count), (1, 48, 72300))
        self.assertEqual(authority.prediction_aggregate_sha256, self.fixture.prediction_aggregate)
        self.assertEqual(len(authority.dataset_order), 48)
        self.assertFalse(any("qualified" in name.casefold() for name in dir(authority)))
        self.assertEqual(
            predecessor.validate_exp449_stopped_authority_antigravity(authority),
            authority.dataset_order,
        )

    def test_second_capture_drift_fails_without_issuing_authority_antigravity(self) -> None:
        """Break caught: the first snapshot alone authorizes labels after predecessor drift."""
        from experiments.exp452_exp449_index_order_repair_antigravity import predecessor_antigravity as predecessor

        def drift() -> None:
            self.fixture.audit.write_bytes(self.fixture.audit.read_bytes() + b" ")

        dependencies = replace(self.fixture.dependencies_antigravity(), between_captures=drift)
        with self.assertRaisesRegex(ValueError, "independent captures"):
            predecessor._capture_exp449_stopped_authority_with_dependencies_antigravity(dependencies)

    def test_actual_operation_failure_receipt_has_bound_started_then_unbound_stopped_antigravity(self) -> None:
        """Break caught: a synthetic direct append hides Exp449's actual failure-path receipt format."""
        from experiments.exp452_exp449_index_order_repair_antigravity import predecessor_antigravity as predecessor

        started = json.loads(
            (self.fixture.receipts / "evaluate_started_antigravity.json").read_text(encoding="utf-8")
        )
        stopped_path = self.fixture.receipts / "stopped_antigravity.json"
        stopped = json.loads(stopped_path.read_text(encoding="utf-8"))
        self.assertRegex(started["prediction_root_identity"], r"^[0-9a-f]{64}$")
        self.assertIsNone(stopped["prediction_root_identity"])
        self.assertIsNone(stopped["evaluation_root_identity"])
        predecessor._capture_exp449_stopped_authority_with_dependencies_antigravity(
            self.fixture.dependencies_antigravity()
        )

        stopped["prediction_root_identity"] = started["prediction_root_identity"]
        stopped_path.write_bytes(_canonical_antigravity(stopped))
        with self.assertRaisesRegex(ValueError, "prediction binding"):
            predecessor._capture_exp449_stopped_authority_with_dependencies_antigravity(
                self.fixture.dependencies_antigravity()
            )

    def test_missing_symlink_multilink_inventory_digest_and_publication_fail_closed_antigravity(self) -> None:
        """Break caught: one unsafe predecessor namespace survives to capability issuance."""
        from experiments.exp452_exp449_index_order_repair_antigravity import predecessor_antigravity as predecessor

        cases = ("missing", "symlink", "multilink", "inventory", "digest", "published")
        for case in cases:
            with self.subTest(case=case):
                # Rebuild after every destructive case so expected bytes stay independent.
                if case != cases[0]:
                    self.fixture.close_antigravity()
                    self.fixture = SyntheticPredecessorFixtureAntigravity()
                dependencies = self.fixture.dependencies_antigravity()
                if case == "missing":
                    self.fixture.audit.unlink()
                elif case == "symlink":
                    payload = self.fixture.parent / "audit_payload_antigravity.json"
                    self.fixture.audit.rename(payload)
                    self.fixture.audit.symlink_to(payload)
                elif case == "multilink":
                    os.link(self.fixture.audit, self.fixture.parent / "audit_hardlink_antigravity.json")
                elif case == "inventory":
                    (self.fixture.staging / "extra_antigravity.json").write_bytes(b"{}\n")
                elif case == "digest":
                    dependencies = replace(dependencies, expected_staging_evaluation_sha256="0" * 64)
                else:
                    self.fixture.published_evaluation.mkdir()
                with self.assertRaises(ValueError):
                    predecessor._capture_exp449_stopped_authority_with_dependencies_antigravity(dependencies)

    def test_prediction_leaf_hardlink_and_postread_replacement_are_rejected_antigravity(self) -> None:
        """Break caught: a linked prediction leaf or same-byte namespace replacement attests."""
        from experiments.exp452_exp449_index_order_repair_antigravity import predecessor_antigravity as predecessor

        candidate = next((self.fixture.prediction / "candidates_antigravity").iterdir())
        os.link(candidate, self.fixture.parent / "candidate_hardlink_antigravity.json")
        with self.assertRaises(ValueError):
            predecessor._capture_exp449_stopped_authority_with_dependencies_antigravity(
                self.fixture.dependencies_antigravity()
            )

        self.fixture.close_antigravity()
        self.fixture = SyntheticPredecessorFixtureAntigravity()
        target = self.fixture.staging / "bootstrap_antigravity.json"

        def replace_after_read(_root: Path) -> None:
            if _root != self.fixture.staging:
                return
            replacement = self.fixture.parent / "replacement_antigravity.json"
            replacement.write_bytes(target.read_bytes())
            os.replace(replacement, target)

        with patch.object(predecessor, "_after_leaf_reads_antigravity", side_effect=replace_after_read):
            with self.assertRaisesRegex(ValueError, "changed"):
                predecessor._capture_exp449_stopped_authority_with_dependencies_antigravity(
                    self.fixture.dependencies_antigravity()
                )


if __name__ == "__main__":
    unittest.main()
