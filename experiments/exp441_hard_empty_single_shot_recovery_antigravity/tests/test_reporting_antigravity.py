"""Behavior tests for Exp441's label-free six-file publication boundary."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import (
    CandidateEvidenceAntigravity, CompletedAttemptAntigravity, FrozenSourceSnapshotAntigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity import (
    _selection_payload_antigravity, _snapshot_payload_antigravity,
    terminal_attempt_authority_sha256_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.selector_antigravity import select_hard_empty_single_shot_antigravity


class ReportingContractAntigravityTests(unittest.TestCase):
    @staticmethod
    def _fixture_snapshot_antigravity() -> FrozenSourceSnapshotAntigravity:
        datasets = tuple(f"D{index:04d}" for index in range(1117))
        candidates = tuple(CandidateEvidenceAntigravity(
            dataset=dataset, instance_index=index, raw_length=16, depths=(2, 4), agreement_count=2,
            second_excess_ratio=100.0 - index, second_empirical_rank=0.1,
        ) for index, dataset in enumerate(datasets[:49]))
        return FrozenSourceSnapshotAntigravity(
            context_datasets=datasets, test_row_count_by_dataset={dataset: 64 for dataset in datasets},
            record_digest_by_dataset={dataset: "a" * 64 for dataset in datasets}, hard_coordinates=frozenset(),
            provisional_candidates=candidates, context_sha256="b" * 64, state_sha256="c" * 64,
            retained_inventory_sha256="d" * 64, publication_prediction_sha256="e" * 64,
            publication_manifest_sha256="f" * 64, publication_artifact_map_sha256="1" * 64,
            hard_prediction_sha256="2" * 64, hard_manifest_sha256="3" * 64,
            hard_coordinate_identity_sha256="4" * 64,
        )

    def _write_terminal_attempt_antigravity(self, root: Path, snapshot: FrozenSourceSnapshotAntigravity, *, pid_value: int = 1) -> CompletedAttemptAntigravity:
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        attempt_root = root / "attempts_antigravity" / "attempt_0000_antigravity"
        attempt_root.mkdir(parents=True)
        os.chmod(attempt_root, 0o700)
        snapshot_bytes = canonical_json_bytes_antigravity(_snapshot_payload_antigravity(snapshot))
        snapshot_sha = sha256(snapshot_bytes).hexdigest()
        selection_payload = _selection_payload_antigravity(selection, reviewed_commit="a" * 40, source_snapshot_sha256=snapshot_sha)
        selection_bytes = canonical_json_bytes_antigravity(selection_payload)
        (attempt_root / "source_snapshot_antigravity.json").write_bytes(snapshot_bytes)
        (attempt_root / "selection_antigravity.json").write_bytes(selection_bytes)
        (attempt_root / "progress_antigravity.json").write_bytes(canonical_json_bytes_antigravity({
            "pid": pid_value, "active": False, "current_dataset": None, "current_record_name": None, "active_datasets": [],
            "train_row_count": None, "test_row_count": None, "raw_length": None,
            "dimension_status": "not_available_in_frozen_selector_source", "stage": "complete", "backend": "cpu",
            "completed": 1117, "total": 1117, "imported": 1117, "recomputed": 0, "success": 1117, "error": 0,
            "provisional_candidates": 840, "outside_hard_candidates": 394, "eligible_candidates": 73, "selected_candidates": 30,
            "throughput": 1.0, "elapsed_seconds": 1117.0, "eta_seconds": None,
            "source_checkpoint": "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity/records_antigravity",
            "heartbeat": "2026-08-08T00:00:00Z", "latest_error": None,
        }))
        (attempt_root / "pid_antigravity").write_bytes(canonical_json_bytes_antigravity({"pid": pid_value, "active": False, "heartbeat": "2026-08-08T00:00:01Z"}))
        events = b"".join(canonical_json_bytes_antigravity({
            "sequence": index, "dataset": dataset, "record_name": f"{index:04d}_record_antigravity.json",
            "test_row_count": 64, "raw_length": 16 if index < 49 else None, "accepted": True,
        }) for index, dataset in enumerate(snapshot.context_datasets))
        (attempt_root / "events_antigravity.jsonl").write_bytes(events)
        (attempt_root / "text_antigravity.log").write_text("".join(
            f"accepted dataset={dataset} record={index:04d}_record_antigravity.json test_row_count=64 raw_length={16 if index < 49 else None}\n"
            for index, dataset in enumerate(snapshot.context_datasets)
        ))
        for entry in attempt_root.iterdir():
            os.chmod(entry, 0o600)
        return CompletedAttemptAntigravity(
            0, "a" * 40, snapshot_sha, selection, sha256(selection_bytes).hexdigest(),
            terminal_attempt_authority_sha256_antigravity(attempt_root),
        )

    def test_reporting_public_api_exposes_fixed_publish_and_inspect_boundaries(self) -> None:
        # Break caught: a caller cannot freeze or replay the fixed label-free publication boundary.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity

        self.assertTrue(callable(reporting_antigravity.publish_exp441_antigravity))
        self.assertTrue(callable(reporting_antigravity.inspect_exp441_publication_antigravity))

    def test_completed_attempt_carries_immutable_six_file_authority_identity(self) -> None:
        # Break caught: coordinated pre-entry edits can change every semantic field while the attempt object has no byte identity.
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            attempt = self._write_terminal_attempt_antigravity(Path(directory), snapshot)
            self.assertRegex(attempt.terminal_authority_sha256, r"^[0-9a-f]{64}$")

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
        attempt = CompletedAttemptAntigravity(0, "a" * 40, "b" * 64, selection, "c" * 64, "d" * 64)
        payload = reporting_antigravity._failure_payload_antigravity(attempt, ValueError("race"))
        reporting_antigravity._validate_failure_payload_antigravity(payload)
        tampered = dict(payload)
        tampered["review_only"] = 1
        with self.assertRaisesRegex(ValueError, "governance"):
            reporting_antigravity._validate_failure_payload_antigravity(tampered)

    def test_publish_rejects_inactive_negative_attempt_pid(self) -> None:
        # Break caught: a shape-valid but impossible terminal PID can authorize a publication.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            pid = root / "attempts_antigravity" / "attempt_0000_antigravity" / "pid_antigravity"
            pid.write_bytes(canonical_json_bytes_antigravity({"pid": -1, "active": False, "heartbeat": "2026-08-08T00:00:01Z"}))
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename):
                with self.assertRaisesRegex(ValueError, "authority|PID"):
                    reporting_antigravity.publish_exp441_antigravity(attempt)

    def test_publish_accepts_realistic_positive_nonone_terminal_pid(self) -> None:
        # Break caught: a production PID is rejected merely because it is not a fixture's constant value.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = self._write_terminal_attempt_antigravity(root, snapshot, pid_value=43210)
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename):
                frozen = reporting_antigravity.publish_exp441_antigravity(attempt)
            self.assertEqual(frozen.selection, attempt.selection)

    def test_publish_rejects_zero_boolean_and_mismatched_pid_authority(self) -> None:
        # Break caught: invalid PID types or cross-file disagreement still authorize the completed attempt.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        cases = ((0, 0), (True, True), (2, 1))
        for progress_pid, pid_pid in cases:
            with self.subTest(progress_pid=progress_pid, pid_pid=pid_pid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                attempt = self._write_terminal_attempt_antigravity(root, snapshot)
                progress_path = root / "attempts_antigravity" / "attempt_0000_antigravity" / "progress_antigravity.json"
                pid_path = root / "attempts_antigravity" / "attempt_0000_antigravity" / "pid_antigravity"
                progress = json.loads(progress_path.read_text()); progress["pid"] = progress_pid
                pid = json.loads(pid_path.read_text()); pid["pid"] = pid_pid
                progress_path.write_bytes(canonical_json_bytes_antigravity(progress)); pid_path.write_bytes(canonical_json_bytes_antigravity(pid))
                with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                     patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                     patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                     patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename):
                    with self.assertRaises(ValueError):
                        reporting_antigravity.publish_exp441_antigravity(attempt)

    def test_inspect_rejects_nonprivate_publication_root(self) -> None:
        # Break caught: an otherwise valid publication becomes world-readable without replay rejection.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename):
                reporting_antigravity.publish_exp441_antigravity(attempt)
                os.chmod(root / "publication_antigravity", 0o755)
                with self.assertRaisesRegex(ValueError, "root"):
                    reporting_antigravity.inspect_exp441_publication_antigravity()

    def test_publish_final_gate_reopens_attempt_authority_after_staging(self) -> None:
        # Break caught: a source/selection authority file changes after staging but before rename.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            calls = {"count": 0}
            def replay():
                calls["count"] += 1
                if calls["count"] == 2:
                    selection_path = root / "attempts_antigravity" / "attempt_0000_antigravity" / "selection_antigravity.json"
                    payload = json.loads(selection_path.read_text())
                    payload["review_only"] = False
                    selection_path.write_bytes(canonical_json_bytes_antigravity(payload))
                return snapshot
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", side_effect=replay), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename):
                with self.assertRaisesRegex(ValueError, "selection hash|authority"):
                    reporting_antigravity.publish_exp441_antigravity(attempt)
            self.assertFalse((root / "publication_antigravity").exists())

    def test_publish_rejects_authority_mutated_during_last_source_replay(self) -> None:
        # Break caught: the final source replay completes after an attempt file changes, leaving an unchecked rename window.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            calls = {"count": 0}
            def replay():
                calls["count"] += 1
                if calls["count"] == 3:
                    progress = root / "attempts_antigravity" / "attempt_0000_antigravity" / "progress_antigravity.json"
                    payload = json.loads(progress.read_text())
                    payload["throughput"] = 999.0
                    progress.write_bytes(canonical_json_bytes_antigravity(payload))
                return snapshot
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", side_effect=replay), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename):
                with self.assertRaisesRegex(ValueError, "authority|progress"):
                    reporting_antigravity.publish_exp441_antigravity(attempt)
            self.assertFalse((root / "publication_antigravity").exists())

    def test_publish_rejects_preentry_progress_and_text_log_tampering(self) -> None:
        # Break caught: elapsed/throughput or parent log bytes can drift before publication without being bound to source replay.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        for filename, mutator in (
            ("progress_antigravity.json", lambda payload: payload.update({"throughput": 2.0})),
            ("text_antigravity.log", lambda payload: payload + b"extra\n"),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); attempt = self._write_terminal_attempt_antigravity(root, snapshot)
                target = root / "attempts_antigravity" / "attempt_0000_antigravity" / filename
                if filename.endswith(".json"):
                    payload = json.loads(target.read_text()); mutator(payload); target.write_bytes(canonical_json_bytes_antigravity(payload))
                else:
                    target.write_bytes(mutator(target.read_bytes()))
                with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                     patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                     patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                     patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename):
                    with self.assertRaises(ValueError):
                        reporting_antigravity.publish_exp441_antigravity(attempt)

    def test_publish_rejects_coordinated_preentry_terminal_metadata_tampering(self) -> None:
        # Break caught: progress and PID can be changed together into a valid-looking terminal state before the first replay.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); attempt = self._write_terminal_attempt_antigravity(root, snapshot, pid_value=43210)
            attempt_root = root / "attempts_antigravity" / "attempt_0000_antigravity"
            progress_path = attempt_root / "progress_antigravity.json"; pid_path = attempt_root / "pid_antigravity"
            progress = json.loads(progress_path.read_text()); progress["heartbeat"] = "2026-08-08T00:00:05Z"
            pid = json.loads(pid_path.read_text()); pid["heartbeat"] = "2026-08-08T00:00:06Z"
            progress_path.write_bytes(canonical_json_bytes_antigravity(progress)); pid_path.write_bytes(canonical_json_bytes_antigravity(pid))
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root):
                with self.assertRaisesRegex(ValueError, "authority identity"):
                    reporting_antigravity.publish_exp441_antigravity(attempt)

    def test_publish_rejects_noncanonical_terminal_heartbeat_even_with_recomputed_identity(self) -> None:
        # Break caught: a fractional or shifted timestamp survives byte sealing but is not the canonical UTC-second heartbeat.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            attempt_root = root / "attempts_antigravity" / "attempt_0000_antigravity"
            pid_path = attempt_root / "pid_antigravity"
            pid = json.loads(pid_path.read_text()); pid["heartbeat"] = "2026-08-08T00:00:01.0Z"
            pid_path.write_bytes(canonical_json_bytes_antigravity(pid))
            attempt = replace(attempt, terminal_authority_sha256=terminal_attempt_authority_sha256_antigravity(attempt_root))
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root):
                with self.assertRaisesRegex(ValueError, "heartbeat"):
                    reporting_antigravity.publish_exp441_antigravity(attempt)

    def test_publish_rejects_attempt_root_with_nonprivate_mode_or_symlink(self) -> None:
        # Break caught: a terminal authority can be read from a world-readable or redirected root.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        for shape in ("mode", "symlink"):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); attempt = self._write_terminal_attempt_antigravity(root, snapshot)
                attempt_root = root / "attempts_antigravity" / "attempt_0000_antigravity"
                if shape == "mode":
                    os.chmod(attempt_root, 0o755)
                else:
                    target = root / "redirected_attempt_antigravity"
                    attempt_root.rename(target)
                    attempt_root.symlink_to(target, target_is_directory=True)
                with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root):
                    with self.assertRaisesRegex(ValueError, "attempt root"):
                        reporting_antigravity.publish_exp441_antigravity(attempt)

    def test_publish_final_authority_gate_is_adjacent_to_exclusive_rename(self) -> None:
        # Break caught: an evidence callback or mutable validation hook runs after the final source replay and before rename.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            order: list[str] = []; runtime_validation_calls = {"count": 0}
            real_parse = reporting_antigravity._parse_attempt_antigravity
            real_replay = reporting_antigravity._reverified_selection_antigravity
            real_runtime_validation = reporting_antigravity._validate_runtime_attempt_evidence_antigravity
            def parsed(value: CompletedAttemptAntigravity):
                order.append("parse")
                return real_parse(value)
            def replay(*args: object):
                order.append("source")
                return real_replay(*args)
            def validate(*args: object) -> None:
                runtime_validation_calls["count"] += 1
                real_runtime_validation(*args)
            def renamed(source: Path, destination: Path) -> None:
                order.append("rename")
                os.rename(source, destination)
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "_parse_attempt_antigravity", side_effect=parsed), \
                 patch.object(reporting_antigravity, "_reverified_selection_antigravity", side_effect=replay), \
                 patch.object(reporting_antigravity, "_validate_runtime_attempt_evidence_antigravity", side_effect=validate), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=renamed):
                reporting_antigravity.publish_exp441_antigravity(attempt)
            self.assertEqual(order[-3:], ["source", "parse", "rename"])
            self.assertEqual(runtime_validation_calls["count"], 1)

    def test_inspect_rejects_status_commit_that_differs_from_manifest(self) -> None:
        # Break caught: status can point at a different implementation commit while its hashes still look valid.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename):
                reporting_antigravity.publish_exp441_antigravity(attempt)
                publication = root / "publication_antigravity"
                status = json.loads((publication / "status_antigravity.json").read_text())
                status["reviewed_commit"] = "b" * 40
                status_bytes = canonical_json_bytes_antigravity(status)
                (publication / "status_antigravity.json").write_bytes(status_bytes)
                artifacts = json.loads((publication / "artifact_sha256_antigravity.json").read_text())
                artifacts["files"]["status_antigravity.json"] = sha256(status_bytes).hexdigest()
                (publication / "artifact_sha256_antigravity.json").write_bytes(canonical_json_bytes_antigravity(artifacts))
                with self.assertRaisesRegex(ValueError, "commit"):
                    reporting_antigravity.inspect_exp441_publication_antigravity()

    def test_inspect_rejects_tampering_of_each_six_file_inventory_member(self) -> None:
        # Break caught: one of the six immutable publication artifacts can be changed without replay failure.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename):
                reporting_antigravity.publish_exp441_antigravity(attempt)
                publication = root / "publication_antigravity"
                for name in (
                    "prediction_antigravity.jsonl", "candidate_ranking_antigravity.jsonl", "source_audit_antigravity.json",
                    "manifest_antigravity.json", "status_antigravity.json", "artifact_sha256_antigravity.json",
                ):
                    with self.subTest(name=name):
                        target = publication / name; original = target.read_bytes(); target.write_bytes(original + b"x")
                        with self.assertRaises(ValueError):
                            reporting_antigravity.inspect_exp441_publication_antigravity()
                        target.write_bytes(original)

    def test_publish_fsyncs_populated_private_staging_before_rename(self) -> None:
        # Break caught: directory metadata is renamed before the six create-only entries are durable.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            fsynced: list[Path] = []
            real_fsync = reporting_antigravity.fsync_directory_antigravity
            def record_fsync(path: Path) -> None:
                fsynced.append(Path(path))
                real_fsync(path)
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=os.rename), \
                 patch.object(reporting_antigravity, "fsync_directory_antigravity", side_effect=record_fsync):
                reporting_antigravity.publish_exp441_antigravity(attempt)
            self.assertTrue(any(path.name.startswith(".publication_") and path.name.endswith("_staging_antigravity") for path in fsynced))

    def test_failed_staging_preservation_fault_chains_without_masking_primary_error(self) -> None:
        # Break caught: a failure while preserving evidence hides the original no-clobber publication failure.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity
        snapshot = self._fixture_snapshot_antigravity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = self._write_terminal_attempt_antigravity(root, snapshot)
            with patch.object(reporting_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting_antigravity, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting_antigravity, "selected_coordinate_identity_sha256_antigravity", return_value=reporting_antigravity.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting_antigravity, "exclusive_rename_antigravity", side_effect=FileExistsError("primary race")), \
                 patch.object(reporting_antigravity, "_preserve_failed_staging_antigravity", side_effect=OSError("preservation fault")):
                with self.assertRaisesRegex(FileExistsError, "primary race") as caught:
                    reporting_antigravity.publish_exp441_antigravity(attempt)
            self.assertIsInstance(caught.exception.__cause__, OSError)


if __name__ == "__main__":
    unittest.main()
