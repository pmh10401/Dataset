"""Behavior tests for Exp441's label-free six-file publication boundary."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import CandidateEvidenceAntigravity, CompletedAttemptAntigravity, FrozenSourceSnapshotAntigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity import _selection_payload_antigravity, _snapshot_payload_antigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.selector_antigravity import select_hard_empty_single_shot_antigravity


class ReportingContractAntigravityTests(unittest.TestCase):
    @staticmethod
    def _fixture_snapshot_antigravity() -> FrozenSourceSnapshotAntigravity:
        datasets = tuple(f"D{index:03d}" for index in range(49))
        candidates = tuple(CandidateEvidenceAntigravity(
            dataset=dataset, instance_index=index, raw_length=16, depths=(2, 4), agreement_count=2,
            second_excess_ratio=100.0 - index, second_empirical_rank=0.1,
        ) for index, dataset in enumerate(datasets))
        return FrozenSourceSnapshotAntigravity(
            context_datasets=datasets, test_row_count_by_dataset={dataset: 64 for dataset in datasets},
            record_digest_by_dataset={dataset: "a" * 64 for dataset in datasets}, hard_coordinates=frozenset(),
            provisional_candidates=candidates, context_sha256="b" * 64, state_sha256="c" * 64,
            retained_inventory_sha256="d" * 64, publication_prediction_sha256="e" * 64,
            publication_manifest_sha256="f" * 64, publication_artifact_map_sha256="1" * 64,
            hard_prediction_sha256="2" * 64, hard_manifest_sha256="3" * 64,
            hard_coordinate_identity_sha256="4" * 64,
        )

    def _write_terminal_attempt_antigravity(self, root: Path, snapshot: FrozenSourceSnapshotAntigravity) -> CompletedAttemptAntigravity:
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        attempt_root = root / "attempts_antigravity" / "attempt_0000_antigravity"
        attempt_root.mkdir(parents=True)
        snapshot_bytes = canonical_json_bytes_antigravity(_snapshot_payload_antigravity(snapshot))
        snapshot_sha = sha256(snapshot_bytes).hexdigest()
        selection_payload = _selection_payload_antigravity(selection, reviewed_commit="a" * 40, source_snapshot_sha256=snapshot_sha)
        selection_bytes = canonical_json_bytes_antigravity(selection_payload)
        (attempt_root / "source_snapshot_antigravity.json").write_bytes(snapshot_bytes)
        (attempt_root / "selection_antigravity.json").write_bytes(selection_bytes)
        (attempt_root / "progress_antigravity.json").write_bytes(canonical_json_bytes_antigravity({
            "stage": "complete", "active": False, "completed": 1117, "total": 1117, "error": 0,
            "provisional_candidates": 840, "outside_hard_candidates": 394, "eligible_candidates": 73, "selected_candidates": 30,
        }))
        (attempt_root / "pid_antigravity").write_bytes(canonical_json_bytes_antigravity({"pid": 1, "active": False}))
        for entry in attempt_root.iterdir():
            os.chmod(entry, 0o600)
        return CompletedAttemptAntigravity(0, "a" * 40, snapshot_sha, selection, sha256(selection_bytes).hexdigest())

    def test_reporting_public_api_exposes_fixed_publish_and_inspect_boundaries(self) -> None:
        # Break caught: a caller cannot freeze or replay the fixed label-free publication boundary.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity

        self.assertTrue(callable(reporting_antigravity.publish_exp441_antigravity))
        self.assertTrue(callable(reporting_antigravity.inspect_exp441_publication_antigravity))

    def test_publish_writes_exact_six_files_and_rejects_prediction_tamper(self) -> None:
        # Break caught: publication could silently accept a changed prediction after its hash was frozen.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename):
                frozen = reporting_antigravity.publish_exp441_antigravity(attempt)
                publication = root / "publication_antigravity"
                self.assertEqual(sorted(item.name for item in publication.iterdir()), sorted((
                    "prediction_antigravity.jsonl", "candidate_ranking_antigravity.jsonl", "source_audit_antigravity.json",
                    "manifest_antigravity.json", "status_antigravity.json", "artifact_sha256_antigravity.json",
                )))
                self.assertEqual(len((publication / "prediction_antigravity.jsonl").read_text().splitlines()), 30)
                self.assertEqual(len((publication / "candidate_ranking_antigravity.jsonl").read_text().splitlines()), 49)
                self.assertEqual(frozen.prediction_sha256, sha256((publication / "prediction_antigravity.jsonl").read_bytes()).hexdigest())
                (publication / "prediction_antigravity.jsonl").write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "hash"):
                    reporting_antigravity.inspect_exp441_publication_antigravity()

    def test_publish_rename_race_preserves_governed_failed_staging_without_destination(self) -> None:
        # Break caught: a rename conflict could discard evidence or adopt a concurrent publication winner.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=FileExistsError("race")):
                with self.assertRaises(FileExistsError):
                    reporting_antigravity.publish_exp441_antigravity(attempt)
            self.assertFalse((root / "publication_antigravity").exists())
            failed = list(root.glob("failed_staging_*_antigravity"))
            self.assertEqual(len(failed), 1)
            failure = json.loads((failed[0] / "failure_antigravity.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual({key: failure[key] for key in reporting_antigravity.governance_payload_antigravity()}, reporting_antigravity.governance_payload_antigravity())

    def test_existing_destination_is_replayed_not_replaced(self) -> None:
        # Break caught: a second publisher overwrites a previously frozen publication instead of inspecting it.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename):
                first = reporting_antigravity.publish_exp441_antigravity(attempt)
                before = (root / "publication_antigravity" / "prediction_antigravity.jsonl").read_bytes()
                second = reporting_antigravity.publish_exp441_antigravity(attempt)
            self.assertEqual(first.publication_sha256, second.publication_sha256)
            self.assertEqual((root / "publication_antigravity" / "prediction_antigravity.jsonl").read_bytes(), before)

    def test_failed_staging_governance_rejects_truthy_substitutions(self) -> None:
        # Break caught: failed evidence accepts a truthy substitute and falsely claims the fixed review-only boundary.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        attempt = CompletedAttemptAntigravity(0, "a" * 40, "b" * 64, selection, "c" * 64)
        payload = reporting_antigravity._failure_payload_antigravity(attempt, ValueError("race"))
        reporting_antigravity._validate_failure_payload_antigravity(payload)
        tampered = dict(payload)
        tampered["review_only"] = 1
        with self.assertRaisesRegex(ValueError, "governance"):
            reporting_antigravity._validate_failure_payload_antigravity(tampered)


if __name__ == "__main__":
    unittest.main()
