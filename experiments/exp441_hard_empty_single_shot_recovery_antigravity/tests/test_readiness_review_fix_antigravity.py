"""Adversarial Task 7 review-fix probes for readiness ownership and handoff."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.config_antigravity import FROZEN_CONFIG


def _source_hashes_antigravity() -> dict[str, str]:
    return {
        "context": FROZEN_CONFIG.checkpoint_context_sha256, "state": FROZEN_CONFIG.checkpoint_state_sha256,
        "retained_inventory": FROZEN_CONFIG.checkpoint_retained_inventory_sha256,
        "publication_prediction": FROZEN_CONFIG.publication_prediction_sha256,
        "publication_manifest": FROZEN_CONFIG.publication_manifest_sha256,
        "publication_artifact_map": FROZEN_CONFIG.publication_artifact_map_sha256,
        "hard_prediction": FROZEN_CONFIG.hard_prediction_sha256, "hard_manifest": FROZEN_CONFIG.hard_manifest_sha256,
        "hard_coordinate_identity": FROZEN_CONFIG.hard_coordinate_identity_sha256,
    }


class ReadinessReviewFixAntigravityTests(unittest.TestCase):
    """Each test names the lifecycle mutation that must now fail closed."""

    def _patch_successful_replay_antigravity(self, stack: ExitStack, readiness: object, root: Path, commit: str) -> None:
        stack.enter_context(patch.object(readiness, "_RUN_ROOT_ANTIGRAVITY", root))
        stack.enter_context(patch.object(readiness, "_current_clean_head_antigravity", return_value=commit))
        stack.enter_context(patch.object(readiness, "verify_frozen_sources_antigravity", return_value=object()))
        stack.enter_context(patch.object(readiness, "_source_payload_antigravity", return_value={
            "context_dataset_count": 1117, "record_digest_count": 1117, "hard_coordinate_count": 2218,
            "provisional_candidate_count": 840, "label_access_count": 0, "source_hashes": _source_hashes_antigravity(),
        }))
        stack.enter_context(patch.object(readiness, "_selection_payload_antigravity", return_value={
            "provisional_candidate_count": 840, "outside_hard_candidate_count": 394,
            "eligible_candidate_count": 73, "winner_count": 49, "selected_candidate_count": 30,
        }))
        stack.enter_context(patch.object(readiness, "_environment_antigravity", return_value={
            "python_version": "3.14.0", "numpy_version": "2.4.6", "pytorch_version": "2.8.0",
            "mps_built": True, "mps_available": True, "mlx_version": "0.30.0",
            "mlx_metal_version": "0.30.0", "mlx_default_device": "gpu", "acceleration_reason": "selector_has_no_tensor_stage",
        }))

    def test_concurrent_winner_preserves_our_staging_with_canonical_failure(self) -> None:
        """A winner appearing before rename must not strand a raw staging directory."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            def winner(source: Path, destination: Path) -> None:
                destination.mkdir(mode=0o700)
                raise FileExistsError("concurrent winner")
            with ExitStack() as stack:
                self._patch_successful_replay_antigravity(stack, readiness, root, commit)
                stack.enter_context(patch.object(readiness, "exclusive_rename_antigravity", side_effect=winner))
                with self.assertRaises(FileExistsError): readiness.run_readiness_antigravity(commit)
            self.assertTrue(root.is_dir())  # concurrent owner remains untouched
            self.assertEqual(list(Path(temporary).glob(".run_antigravity_staging_*_antigravity")), [])
            failed = list(Path(temporary).glob(".run_antigravity_failed_*_antigravity"))
            self.assertEqual(len(failed), 1)
            failure = json.loads((failed[0] / "failure_antigravity.json").read_text())
            self.assertEqual(failure["schema"], "exp441.readiness_failure.v1")
            self.assertEqual(failure["reviewed_commit"], commit)

    def test_post_rename_move_aside_replacement_cannot_be_adopted_as_ours(self) -> None:
        """Capturing ownership from destination after rename must not bless a racer's empty root."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            def move_aside(source: Path, destination: Path) -> None:
                os.rename(source, destination)
                os.rename(destination, destination.with_name("owned_moved_aside_antigravity"))
                destination.mkdir(mode=0o700)
            with ExitStack() as stack:
                self._patch_successful_replay_antigravity(stack, readiness, root, commit)
                stack.enter_context(patch.object(readiness, "exclusive_rename_antigravity", side_effect=move_aside))
                with self.assertRaises(ValueError): readiness.run_readiness_antigravity(commit)
            self.assertTrue(root.is_dir())  # replacement remains racer's namespace, never adopted or moved
            self.assertTrue((Path(temporary) / "owned_moved_aside_antigravity" / "readiness_antigravity.json").is_file())

    def test_post_rename_parent_fsync_rolls_back_our_owned_parent(self) -> None:
        """A durability failure after rename must leave no fixed parent behind."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            original_fsync = readiness.fsync_directory_antigravity
            failed_once = {"value": False}
            def fsync(path: Path) -> None:
                if path == root.parent and root.exists() and not failed_once["value"]:
                    failed_once["value"] = True
                    raise OSError("parent fsync failed")
                original_fsync(path)
            with ExitStack() as stack:
                self._patch_successful_replay_antigravity(stack, readiness, root, commit)
                stack.enter_context(patch.object(readiness, "fsync_directory_antigravity", side_effect=fsync))
                with self.assertRaisesRegex(OSError, "parent fsync failed"):
                    readiness.run_readiness_antigravity(commit)
            self.assertFalse(root.exists())
            failed = list(Path(temporary).glob(".run_antigravity_failed_*_antigravity"))
            self.assertEqual(len(failed), 1)
            self.assertEqual(set(item.name for item in failed[0].iterdir()), {"readiness_antigravity.json", "failure_antigravity.json"})

    def test_validation_rejects_output_added_during_replay_before_handoff(self) -> None:
        """An added output during replay must be caught by a retained readiness-root pin."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"; root.mkdir(mode=0o700)
            backend = readiness.BackendAssessmentAntigravity("cpu", 11, False, "not_run_inapplicable", "inapplicable_no_tensor_stage", "inapplicable_no_tensor_stage", "inapplicable_no_tensor_stage", {"numpy_version": "2.4.6"})
            source = {"context_dataset_count": 1117, "record_digest_count": 1117, "hard_coordinate_count": 2218, "provisional_candidate_count": 840, "label_access_count": 0, "source_hashes": _source_hashes_antigravity()}
            selection = {"provisional_candidate_count": 840, "outside_hard_candidate_count": 394, "eligible_candidate_count": 73, "winner_count": 49, "selected_candidate_count": 30}
            payload = readiness._readiness_payload_antigravity(commit, source, selection, backend)
            artifact = root / "readiness_antigravity.json"; artifact.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"); artifact.chmod(0o600)
            def mutate() -> object:
                (root / "checkpoint_antigravity").mkdir()
                return object()
            with ExitStack() as stack:
                stack.enter_context(patch.object(readiness, "_RUN_ROOT_ANTIGRAVITY", root))
                stack.enter_context(patch.object(readiness, "_current_clean_head_antigravity", return_value=commit))
                stack.enter_context(patch.object(readiness, "verify_frozen_sources_antigravity", side_effect=mutate))
                stack.enter_context(patch.object(readiness, "_source_payload_antigravity", return_value=source))
                stack.enter_context(patch.object(readiness, "_selection_payload_antigravity", return_value=selection))
                stack.enter_context(patch.object(readiness, "_backend_antigravity", return_value=backend))
                with self.assertRaises(ValueError): readiness.validate_readiness_for_run_antigravity(commit)

    def test_preservation_write_failure_chains_without_masking_primary_readiness_error(self) -> None:
        """A secondary failure must be visible while callers still receive the source failure."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            with ExitStack() as stack:
                stack.enter_context(patch.object(readiness, "_RUN_ROOT_ANTIGRAVITY", root))
                stack.enter_context(patch.object(readiness, "_current_clean_head_antigravity", return_value=commit))
                stack.enter_context(patch.object(readiness, "verify_frozen_sources_antigravity", side_effect=ValueError("primary source failure")))
                stack.enter_context(patch.object(readiness, "write_create_only_antigravity", side_effect=OSError("preservation write failure")))
                with self.assertRaisesRegex(ValueError, "primary source failure") as caught:
                    readiness.run_readiness_antigravity(commit)
            self.assertIsInstance(caught.exception.__cause__, OSError)
            self.assertIn("preservation also failed", "\n".join(caught.exception.__notes__))
            self.assertFalse(root.exists())

    def test_validation_rejects_readiness_root_replacement_during_replay(self) -> None:
        """Replacing the directory with an identical-looking sibling cannot redirect handoff."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"; root.mkdir(mode=0o700)
            backend = readiness.BackendAssessmentAntigravity("cpu", 11, False, "not_run_inapplicable", "inapplicable_no_tensor_stage", "inapplicable_no_tensor_stage", "inapplicable_no_tensor_stage", {"numpy_version": "2.4.6"})
            source = {"context_dataset_count": 1117, "record_digest_count": 1117, "hard_coordinate_count": 2218, "provisional_candidate_count": 840, "label_access_count": 0, "source_hashes": _source_hashes_antigravity()}
            selection = {"provisional_candidate_count": 840, "outside_hard_candidate_count": 394, "eligible_candidate_count": 73, "winner_count": 49, "selected_candidate_count": 30}
            encoded = json.dumps(readiness._readiness_payload_antigravity(commit, source, selection, backend), sort_keys=True, separators=(",", ":")) + "\n"
            (root / "readiness_antigravity.json").write_text(encoded); (root / "readiness_antigravity.json").chmod(0o600)
            def replace() -> object:
                old = root.with_name("old_antigravity"); os.rename(root, old); root.mkdir(mode=0o700)
                replacement = root / "readiness_antigravity.json"; replacement.write_text(encoded); replacement.chmod(0o600)
                return object()
            with ExitStack() as stack:
                stack.enter_context(patch.object(readiness, "_RUN_ROOT_ANTIGRAVITY", root))
                stack.enter_context(patch.object(readiness, "_current_clean_head_antigravity", return_value=commit))
                stack.enter_context(patch.object(readiness, "verify_frozen_sources_antigravity", side_effect=replace))
                stack.enter_context(patch.object(readiness, "_source_payload_antigravity", return_value=source))
                stack.enter_context(patch.object(readiness, "_selection_payload_antigravity", return_value=selection))
                stack.enter_context(patch.object(readiness, "_backend_antigravity", return_value=backend))
                with self.assertRaises(ValueError): readiness.validate_readiness_for_run_antigravity(commit)
