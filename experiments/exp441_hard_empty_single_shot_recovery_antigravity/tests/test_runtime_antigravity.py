"""RED/GREEN behavior tests for Exp441 immutable attempts and runtime ownership."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.config_antigravity import governance_payload_antigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import FrozenSourceSnapshotAntigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.sources_antigravity import hard_coordinate_identity_sha256_antigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity import (
    allocate_attempt_antigravity,
    run_exp441_new_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.selector_antigravity import (
    select_hard_empty_single_shot_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_selector_antigravity import _snapshot, _fixture_candidates


class RuntimeAntigravityTests(unittest.TestCase):
    @staticmethod
    def _prefix(snapshot):
        return snapshot.context_datasets, {}, snapshot.hard_coordinates, {}, {}

    @staticmethod
    def _replay(snapshot):
        def replay(observer):
            for index, dataset in enumerate(snapshot.context_datasets):
                values = {item.raw_length for item in snapshot.provisional_candidates if item.dataset == dataset}
                observer({"sequence": index, "dataset": dataset, "test_row_count": snapshot.test_row_count_by_dataset[dataset], "raw_length": next(iter(values)) if len(values) == 1 else None})
            return snapshot
        return replay

    def test_attempt_allocation_uses_smallest_gap_and_rejects_non_directory_claim(self) -> None:
        # Break caught: allocating monotonically or treating a symlink/file as a completed immutable attempt.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempts = root / "attempts_antigravity"
            attempts.mkdir()
            (attempts / "attempt_0000_antigravity").mkdir()
            (attempts / "attempt_0002_antigravity").mkdir()
            allocated = allocate_attempt_antigravity(root)
            self.assertEqual(allocated.name, "attempt_0001_antigravity")
            link = attempts / "attempt_0003_antigravity"
            link.symlink_to(attempts / "attempt_0000_antigravity", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "attempt"):
                allocate_attempt_antigravity(root)
            link.unlink()
            bad = attempts / "attempt_0003_antigravity"
            bad.write_text("not a directory")
            with self.assertRaisesRegex(ValueError, "attempt"):
                allocate_attempt_antigravity(root)

    def test_completed_attempt_is_create_only_and_contains_exact_governance(self) -> None:
        # Break caught: a later new could repair an old attempt, or selection evidence could dilute governance values.
        snapshot = _snapshot(_fixture_candidates())
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.FROZEN_CONFIG", SimpleNamespace(dataset_count=len(snapshot.context_datasets))), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=self._replay(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._reverify_sources_after_artifacts_antigravity", return_value=snapshot), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.select_hard_empty_single_shot_antigravity", return_value=selection), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.validate_production_selection_antigravity", side_effect=lambda value: value):
                first = run_exp441_new_antigravity(reviewed_commit="a" * 40)
                first_bytes = (root / "attempts_antigravity" / "attempt_0000_antigravity" / "selection_antigravity.json").read_bytes()
                second = run_exp441_new_antigravity(reviewed_commit="a" * 40)
            self.assertEqual((first.attempt_sequence, second.attempt_sequence), (0, 1))
            selection_payload = json.loads((root / "attempts_antigravity" / "attempt_0000_antigravity" / "selection_antigravity.json").read_text())
            governance = governance_payload_antigravity()
            self.assertEqual({name: selection_payload[name] for name in governance}, governance)
            self.assertEqual(set(selection_payload) & set(governance), set(governance))
            self.assertEqual(len(list((root / "attempts_antigravity").iterdir())), 2)
            self.assertEqual((root / "attempts_antigravity" / "attempt_0000_antigravity" / "selection_antigravity.json").read_bytes(), first_bytes)

    def test_source_failure_is_immutable_failed_attempt_without_publication(self) -> None:
        # Break caught: source/schema failure escapes without preserved evidence or accidentally publishes a prediction.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", side_effect=ValueError("bad source")), \
                patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=ValueError("bad source")):
                with self.assertRaisesRegex(ValueError, "bad source"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
                with self.assertRaisesRegex(ValueError, "bad source"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            failure = json.loads((root / "attempts_antigravity" / "attempt_0000_antigravity" / "failure_antigravity.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual({name: failure[name] for name in governance_payload_antigravity()}, governance_payload_antigravity())
            self.assertFalse((root / "publication_antigravity").exists())
            self.assertTrue((root / "attempts_antigravity" / "attempt_0001_antigravity" / "failure_antigravity.json").exists())

    def test_ordered_parent_observer_rejects_a_later_context_record(self) -> None:
        # Break caught: committing a worker-completed B before the parent has accepted ordered A.
        snapshot = _snapshot(_fixture_candidates())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def out_of_order(observer):
                observer({"sequence": 1, "dataset": "B", "test_row_count": 8, "raw_length": None})
                return snapshot
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=out_of_order):
                with self.assertRaisesRegex(ValueError, "context order"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            progress = json.loads((root / "attempts_antigravity" / "attempt_0000_antigravity" / "progress_antigravity.json").read_text())
            self.assertEqual(progress["stage"], "failed")
            self.assertEqual(progress["completed"], 0)

    def test_sigterm_is_recorded_as_interrupted_not_schema_failure(self) -> None:
        # Break caught: SIGTERM follows the generic failure path and loses the distinct interruption meaning.
        snapshot = _snapshot(_fixture_candidates())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def interrupt(_observer):
                os.kill(os.getpid(), 15)
                return snapshot
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "SIGTERM"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            failure = json.loads((root / "attempts_antigravity" / "attempt_0000_antigravity" / "failure_antigravity.json").read_text())
            self.assertEqual(failure["status"], "interrupted")

    def test_governance_validator_rejects_missing_extra_wrong_and_nested_values(self) -> None:
        # Break caught: a governing terminal artifact accepts a partial alias, bool-like substitute, or nested replacement.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        valid = governance_payload_antigravity()
        runtime._validate_governance_payload_antigravity(valid)
        for name in valid:
            missing = dict(valid)
            missing.pop(name)
            with self.assertRaisesRegex(ValueError, "governance"):
                runtime._validate_governance_payload_antigravity(missing)
            wrong = dict(valid)
            wrong[name] = {"value": valid[name]}
            with self.assertRaisesRegex(ValueError, "governance"):
                runtime._validate_governance_payload_antigravity(wrong)
            wrong_value = dict(valid)
            wrong_value[name] = (not valid[name]) if type(valid[name]) is bool else "changed"
            with self.assertRaisesRegex(ValueError, "governance"):
                runtime._validate_governance_payload_antigravity(wrong_value)
        extra = dict(valid)
        extra["review_status"] = "alias"
        with self.assertRaisesRegex(ValueError, "governance"):
            runtime._validate_governance_payload_antigravity(extra)

    def test_real_terminal_artifact_shapes_reject_each_governance_tamper(self) -> None:
        # Break caught: selection/failure JSON accepts a governance omission, alias, wrong primitive, or nested object.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        snapshot = _snapshot(_fixture_candidates())
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        artifacts = (
            runtime._selection_payload_antigravity(selection, reviewed_commit="a" * 40, source_snapshot_sha256="b" * 64),
            runtime._failure_payload_antigravity(sequence=0, reviewed_commit="a" * 40, status="failed", error="bad source"),
        )
        governance = governance_payload_antigravity()
        for artifact in artifacts:
            runtime._validate_terminal_artifact_governance_antigravity(artifact)
            for name in governance:
                missing = dict(artifact)
                missing.pop(name)
                with self.assertRaisesRegex(ValueError, "governance"):
                    runtime._validate_terminal_artifact_governance_antigravity(missing)
                wrong = dict(artifact)
                wrong[name] = {"nested": governance[name]}
                with self.assertRaisesRegex(ValueError, "governance"):
                    runtime._validate_terminal_artifact_governance_antigravity(wrong)
            alias = dict(artifact)
            alias["review_only_alias"] = True
            with self.assertRaisesRegex(ValueError, "alias"):
                runtime._validate_terminal_artifact_governance_antigravity(alias)

    def test_production_shaped_parent_observer_reaches_1117_terminal_counts_and_rechecks_sources(self) -> None:
        # Break caught: a mocked successful source verifier skips parent events, misses the full prefix, or lacks the second authority check.
        datasets = tuple(f"D{index:04d}" for index in range(1117))
        snapshot = FrozenSourceSnapshotAntigravity(
            context_datasets=datasets, test_row_count_by_dataset={name: 8 for name in datasets},
            record_digest_by_dataset={name: f"{index:064x}" for index, name in enumerate(datasets)},
            hard_coordinates=frozenset(), provisional_candidates=(), context_sha256="a" * 64,
            state_sha256="b" * 64, retained_inventory_sha256="c" * 64,
            publication_prediction_sha256="d" * 64, publication_manifest_sha256="e" * 64,
            publication_artifact_map_sha256="f" * 64, hard_prediction_sha256="0" * 64,
            hard_manifest_sha256="1" * 64,
            hard_coordinate_identity_sha256=hard_coordinate_identity_sha256_antigravity(()),
        )
        sparse_selection = select_hard_empty_single_shot_antigravity(_snapshot(_fixture_candidates()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def replay(observer):
                for index, dataset in enumerate(datasets):
                    observer({"sequence": index, "dataset": dataset, "test_row_count": 8, "raw_length": None})
                return snapshot
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity") as clean, \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=replay), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._reverify_sources_after_artifacts_antigravity", return_value=snapshot), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.select_hard_empty_single_shot_antigravity", return_value=sparse_selection), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.validate_production_selection_antigravity", side_effect=lambda value: value):
                run_exp441_new_antigravity(reviewed_commit="a" * 40)
            progress = json.loads((root / "attempts_antigravity" / "attempt_0000_antigravity" / "progress_antigravity.json").read_text())
            self.assertEqual((progress["completed"], progress["imported"], progress["success"]), (1117, 1117, 1117))
            self.assertEqual((progress["current_dataset"], progress["current_record_name"], progress["active_datasets"]), (None, None, []))
            self.assertEqual((progress["stage"], progress["provisional_candidates"], progress["outside_hard_candidates"], progress["eligible_candidates"], progress["selected_candidates"]), ("complete", 5, 5, 5, 4))
            self.assertEqual(clean.call_count, 2)

    def test_reverification_detects_source_identity_change_after_attempt_artifacts(self) -> None:
        # Break caught: an input mutation after selection is accepted because only the first frozen-source replay is checked.
        snapshot = _snapshot(_fixture_candidates())
        changed = _snapshot(_fixture_candidates())
        object.__setattr__(changed, "context_sha256", "9" * 64)
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.FROZEN_CONFIG", SimpleNamespace(dataset_count=len(snapshot.context_datasets))), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=self._replay(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._reverify_sources_after_artifacts_antigravity", return_value=changed), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.select_hard_empty_single_shot_antigravity", return_value=selection), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.validate_production_selection_antigravity", side_effect=lambda value: value):
                with self.assertRaisesRegex(ValueError, "changed"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            self.assertTrue((root / "attempts_antigravity" / "attempt_0000_antigravity" / "source_snapshot_antigravity.json").exists())
            self.assertTrue((root / "attempts_antigravity" / "attempt_0000_antigravity" / "failure_antigravity.json").exists())

    def test_selection_failure_writes_no_snapshot_or_selection_artifact(self) -> None:
        # Break caught: source evidence is frozen before the pure selector has successfully completed.
        snapshot = _snapshot(_fixture_candidates())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.FROZEN_CONFIG", SimpleNamespace(dataset_count=len(snapshot.context_datasets))), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=self._replay(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.select_hard_empty_single_shot_antigravity", side_effect=ValueError("selection failed")):
                with self.assertRaisesRegex(ValueError, "selection failed"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
            self.assertFalse((attempt / "source_snapshot_antigravity.json").exists())
            self.assertFalse((attempt / "selection_antigravity.json").exists())

    def test_source_snapshot_records_only_authority_identity_counts_and_ranking_coordinates(self) -> None:
        # Break caught: an attempt snapshot stores raw source rows, TEST sizes, or any field beyond allowed replay evidence.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        payload = runtime._snapshot_payload_antigravity(_snapshot(_fixture_candidates()))
        self.assertEqual(set(payload), {
            "context_dataset_count", "record_digest_count", "record_digest_map_identity_sha256",
            "hard_coordinate_count", "hard_coordinate_identity_sha256", "provisional_candidate_count",
            "ranking_evidence", "source_hashes", "label_access_count",
        })
        self.assertTrue(all(set(row) == {"dataset", "instance_index", "second_excess_ratio", "second_empirical_rank"}
                            for row in payload["ranking_evidence"]))
        self.assertNotIn("test_row_count_by_dataset", payload)
        self.assertNotIn("provisional_candidates", payload)

    def test_no_resume_api_is_exported(self) -> None:
        # Break caught: exposing a resume entrypoint allows mutation of immutable failure evidence.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        self.assertFalse(any("resume" in name.lower() for name in dir(runtime)))
