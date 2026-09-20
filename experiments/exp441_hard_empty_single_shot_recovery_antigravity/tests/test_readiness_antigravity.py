"""Behavior tests for Exp441's pre-run ownership boundary."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
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


class ReadinessAntigravityTests(unittest.TestCase):
    """The production change caught here is accepting a non-frozen launch."""

    def test_public_readiness_module_is_available(self) -> None:
        """A missing readiness boundary would leave the fixed lifecycle unbound."""
        module = importlib.import_module(
            "experiments.exp441_hard_empty_single_shot_recovery_antigravity.readiness_antigravity"
        )
        self.assertTrue(callable(module.run_readiness_antigravity))
        self.assertTrue(callable(module.validate_readiness_for_run_antigravity))

    def test_readiness_owns_only_an_absent_parent_with_canonical_evidence(self) -> None:
        """Replacing an existing parent must fail rather than overwrite its evidence."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            snapshot = object()
            with ExitStack() as stack:
                stack.enter_context(patch.object(readiness, "_RUN_ROOT_ANTIGRAVITY", root))
                stack.enter_context(patch.object(readiness, "_current_clean_head_antigravity", return_value=commit))
                stack.enter_context(patch.object(readiness, "_environment_antigravity", return_value={
                    "python_version": "3.14.0", "numpy_version": "2.4.6",
                    "pytorch_version": "2.8.0", "mps_built": True, "mps_available": True,
                    "mlx_version": "0.30.0", "mlx_metal_version": "0.30.0", "mlx_default_device": "gpu",
                }))
                stack.enter_context(patch.object(readiness, "verify_frozen_sources_antigravity", return_value=snapshot))
                stack.enter_context(patch.object(readiness, "_source_payload_antigravity", return_value={
                    "context_dataset_count": 1117, "record_digest_count": 1117,
                    "hard_coordinate_count": 2218, "provisional_candidate_count": 840,
                    "label_access_count": 0, "source_hashes": _source_hashes_antigravity(),
                }))
                stack.enter_context(patch.object(readiness, "_selection_payload_antigravity", return_value={
                    "provisional_candidate_count": 840, "outside_hard_candidate_count": 394,
                    "eligible_candidate_count": 73, "winner_count": 49, "selected_candidate_count": 30,
                }))
                binding = readiness.run_readiness_antigravity(commit)
            artifact = root / "readiness_antigravity.json"
            payload = artifact.read_bytes()
            self.assertEqual(payload, json.dumps(json.loads(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode() + b"\n")
            self.assertEqual(binding.reviewed_commit, commit)
            self.assertEqual(binding.backend.workers, 11)
            self.assertEqual(binding.readiness_sha256, __import__("hashlib").sha256(payload).hexdigest())
            with self.assertRaises(ValueError):
                readiness.run_readiness_antigravity(commit)

    def test_validation_replays_canonical_bytes_and_rejects_governance_tamper(self) -> None:
        """A changed governance bit must not launch a new immutable attempt."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            snapshot = object()
            with ExitStack() as stack:
                stack.enter_context(patch.object(readiness, "_RUN_ROOT_ANTIGRAVITY", root))
                stack.enter_context(patch.object(readiness, "_current_clean_head_antigravity", return_value=commit))
                stack.enter_context(patch.object(readiness, "_environment_antigravity", return_value={
                    "python_version": "3.14.0", "numpy_version": "2.4.6", "pytorch_version": "2.8.0",
                    "mps_built": True, "mps_available": True, "mlx_version": "0.30.0",
                    "mlx_metal_version": "0.30.0", "mlx_default_device": "gpu",
                }))
                stack.enter_context(patch.object(readiness, "verify_frozen_sources_antigravity", return_value=snapshot))
                stack.enter_context(patch.object(readiness, "_source_payload_antigravity", return_value={
                    "context_dataset_count": 1117, "record_digest_count": 1117,
                    "hard_coordinate_count": 2218, "provisional_candidate_count": 840,
                    "label_access_count": 0, "source_hashes": _source_hashes_antigravity(),
                }))
                stack.enter_context(patch.object(readiness, "_selection_payload_antigravity", return_value={
                    "provisional_candidate_count": 840, "outside_hard_candidate_count": 394,
                    "eligible_candidate_count": 73, "winner_count": 49, "selected_candidate_count": 30,
                }))
                readiness.run_readiness_antigravity(commit)
                self.assertEqual(readiness.validate_readiness_for_run_antigravity(commit).reviewed_commit, commit)
                artifact = root / "readiness_antigravity.json"
                tampered = json.loads(artifact.read_text())
                tampered["review_only"] = False
                artifact.write_text(json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n")
                with self.assertRaises(ValueError):
                    readiness.validate_readiness_for_run_antigravity(commit)

    def test_payload_rejects_every_governance_key_type_value_and_replay_tamper(self) -> None:
        """Truthiness, nesting, source identity, and selector-count drift must not launch."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        backend = readiness.BackendAssessmentAntigravity(
            "cpu", 11, False, "not_run_inapplicable", "inapplicable_no_tensor_stage",
            "inapplicable_no_tensor_stage", "inapplicable_no_tensor_stage", {"numpy_version": "2.4.6"},
        )
        source = {"context_dataset_count": 1117, "record_digest_count": 1117, "hard_coordinate_count": 2218,
                  "provisional_candidate_count": 840, "label_access_count": 0, "source_hashes": _source_hashes_antigravity()}
        selection = {"provisional_candidate_count": 840, "outside_hard_candidate_count": 394,
                     "eligible_candidate_count": 73, "winner_count": 49, "selected_candidate_count": 30}
        payload = readiness._readiness_payload_antigravity("a" * 40, source, selection, backend)
        for key, value in list(readiness.governance_payload_antigravity().items()):
            altered = dict(payload); altered[key] = {"nested": value}
            with self.assertRaises(ValueError, msg=key): readiness._validate_readiness_payload_antigravity(altered)
            altered = dict(payload); altered[key] = (not value) if type(value) is bool else value + "_wrong"
            with self.assertRaises(ValueError, msg=key): readiness._validate_readiness_payload_antigravity(altered)
        for key in ("source_snapshot_sha256", "database_sha256"):
            altered = dict(payload); altered[key] = "0" * 64
            with self.assertRaises(ValueError, msg=key): readiness._validate_readiness_payload_antigravity(altered)
        altered = dict(payload); altered["selection"] = dict(selection, selected_candidate_count=True)
        with self.assertRaises(ValueError): readiness._validate_readiness_payload_antigravity(altered)
        altered = dict(payload); altered["source"] = dict(source, source_hashes={})
        with self.assertRaises(ValueError): readiness._validate_readiness_payload_antigravity(altered)

    def test_precondition_and_staging_failures_never_create_fixed_parent(self) -> None:
        """Existing, symlink, write, fsync, and rename winners preserve the fixed absence."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            root.mkdir()
            with patch.object(readiness, "_RUN_ROOT_ANTIGRAVITY", root), patch.object(readiness, "_current_clean_head_antigravity", return_value=commit):
                with self.assertRaises(ValueError): readiness.run_readiness_antigravity(commit)
            root.rmdir()
            root.symlink_to(Path(temporary) / "target")
            with patch.object(readiness, "_RUN_ROOT_ANTIGRAVITY", root), patch.object(readiness, "_current_clean_head_antigravity", return_value=commit):
                with self.assertRaises(ValueError): readiness.run_readiness_antigravity(commit)
            root.unlink()
            with patch.object(readiness, "_RUN_ROOT_ANTIGRAVITY", root), patch.object(readiness, "_current_clean_head_antigravity", return_value=commit), patch.object(readiness, "verify_frozen_sources_antigravity", side_effect=OSError("source changed")):
                with self.assertRaises(OSError): readiness.run_readiness_antigravity(commit)
            self.assertFalse(root.exists())
            preserved = list(Path(temporary).glob(".run_antigravity_failed_*_antigravity"))
            self.assertEqual(len(preserved), 1)
            self.assertTrue((preserved[0] / "failure_antigravity.json").is_file())

    def test_dirty_mismatched_and_double_head_are_rejected_before_ownership(self) -> None:
        """Any HEAD drift around replay must fail closed rather than bind the wrong attempt."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            with patch.object(readiness, "_RUN_ROOT_ANTIGRAVITY", root), patch.object(readiness, "_current_clean_head_antigravity", return_value="b" * 40):
                with self.assertRaises(ValueError): readiness.run_readiness_antigravity(commit)
            self.assertFalse(root.exists())
