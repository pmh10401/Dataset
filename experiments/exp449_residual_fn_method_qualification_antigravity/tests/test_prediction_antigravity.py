"""Synthetic Task 5 lifecycle tests; no database or official root is touched."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _BaselineAntigravity:
    family: str


@dataclass(frozen=True)
class _Exp448Antigravity:
    target_baseline_rows: tuple[_BaselineAntigravity, ...]


class PredictionAntigravityTests(unittest.TestCase):
    def test_private_validator_reconstructs_exact_dataset_order_and_rejects_tampered_leaf_antigravity(self) -> None:
        """Break caught: accepting a rehashed/tampered manifest or deriving order without canonical leaves."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            run_synthetic_lifecycle_antigravity,
            validate_prediction_root_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_validator_antigravity_") as temporary:
            root = Path(temporary) / "run"
            authority = run_synthetic_lifecycle_antigravity(root, "f" * 40)
            self.assertEqual(validate_prediction_root_antigravity(root, authority), tuple(f"synthetic_dataset_{index:03d}" for index in range(48)))
            leaf = root / authority.inventory[0]
            leaf.write_bytes(b'{"sequence":0}\n')
            with self.assertRaises(ValueError):
                validate_prediction_root_antigravity(root, authority)

    def test_private_validator_rejects_candidate_symlink_and_root_extra_antigravity(self) -> None:
        """Break caught: traversing symlinks or accepting an unpinned prediction namespace."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            run_synthetic_lifecycle_antigravity,
            validate_prediction_root_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_validator_antigravity_") as temporary:
            root = Path(temporary) / "run"
            authority = run_synthetic_lifecycle_antigravity(root, "a" * 40)
            leaf = root / authority.inventory[0]
            replacement = root / "replacement_antigravity.json"
            replacement.write_bytes(leaf.read_bytes())
            leaf.unlink(); leaf.symlink_to(replacement.name)
            with self.assertRaises(ValueError):
                validate_prediction_root_antigravity(root, authority)

    def test_private_validator_rejects_hardlinked_status_channel_antigravity(self) -> None:
        """Break caught: accepting a status channel that can be mutated through another name."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            run_synthetic_lifecycle_antigravity,
            validate_prediction_root_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_validator_antigravity_") as temporary:
            root = Path(temporary) / "run"
            authority = run_synthetic_lifecycle_antigravity(root, "c" * 40)
            os.link(root / "progress_antigravity.txt", Path(temporary) / "linked_progress_antigravity.txt")
            with self.assertRaises(ValueError):
                validate_prediction_root_antigravity(root, authority)

    def test_private_validator_rejects_rewritten_terminal_text_antigravity(self) -> None:
        """Break caught: accepting a terminal text channel that disagrees with verified leaf prefix."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            run_synthetic_lifecycle_antigravity,
            validate_prediction_root_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_validator_antigravity_") as temporary:
            root = Path(temporary) / "run"
            authority = run_synthetic_lifecycle_antigravity(root, "d" * 40)
            (root / "progress_antigravity.txt").write_text("96\n")
            with self.assertRaises(ValueError):
                validate_prediction_root_antigravity(root, authority)

    def test_private_validator_rejects_rehashed_semantic_context_attack_antigravity(self) -> None:
        """Break caught: accepting a manifest rehash after frozen candidate semantics are changed."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            run_synthetic_lifecycle_antigravity,
            validate_prediction_root_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_validator_antigravity_") as temporary:
            root = Path(temporary) / "run"
            authority = run_synthetic_lifecycle_antigravity(root, "e" * 40)
            context_path = root / "context_antigravity.json"
            context = json.loads(context_path.read_text())
            context["candidate_matrix"] = []
            context_path.write_bytes(canonical_json_bytes_antigravity(context))
            manifest_path = root / "prediction_manifest_antigravity.json"
            manifest = json.loads(manifest_path.read_text())
            import hashlib
            manifest["context_sha256"] = hashlib.sha256(context_path.read_bytes()).hexdigest()
            manifest_path.write_bytes(canonical_json_bytes_antigravity(manifest))
            with self.assertRaises(ValueError):
                validate_prediction_root_antigravity(root, authority)
        with tempfile.TemporaryDirectory(prefix="exp449_validator_antigravity_") as temporary:
            root = Path(temporary) / "run"
            authority = run_synthetic_lifecycle_antigravity(root, "b" * 40)
            (root / "extra_antigravity.json").write_text("{}\n")
            with self.assertRaises(ValueError):
                validate_prediction_root_antigravity(root, authority)

    def test_attempt_preflight_rejects_existing_partial_and_symlink_roots_antigravity(self) -> None:
        """Break caught: beginning work in a preexisting, partial, or symlinked attempt root."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import prepare_attempt_root_antigravity  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_root_antigravity_") as temporary:
            parent = Path(temporary)
            partial = parent / "partial"
            partial.mkdir()
            (partial / "context_antigravity.json").write_text("{}\n")
            with self.assertRaises(ValueError):
                prepare_attempt_root_antigravity(partial)
            target = parent / "target"
            target.mkdir()
            link = parent / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                prepare_attempt_root_antigravity(link)
            fresh = parent / "fresh"
            prepare_attempt_root_antigravity(fresh)
            self.assertTrue(fresh.is_dir())

    def test_private_lifecycle_commits_exact_prefix_then_audits_next_sequence_antigravity(self) -> None:
        """Break caught: committing progress ahead of leaves or returning authority after a post-leaf failure."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_antigravity_") as temporary:
            parent = Path(temporary)
            root = parent / "run"
            audit = parent / "audit_antigravity.json"
            with self.assertRaisesRegex(RuntimeError, "exact_next_sequence=48"):
                run_synthetic_lifecycle_antigravity(root, "d" * 40, fail_after_leaf=47, audit_path=audit)
            leaves = tuple(sorted((root / "candidates_antigravity").glob("*.json")))
            self.assertEqual(len(leaves), 48)
            self.assertEqual(json.loads((root / "progress_antigravity.json").read_text())["dataset_unit_completed"], 48)
            self.assertFalse(json.loads((root / "pid_antigravity.json").read_text())["active"])
            self.assertFalse((root / "prediction_manifest_antigravity.json").exists())
            self.assertEqual(json.loads(audit.read_text())["exact_next_sequence"], 48)

    def test_private_lifecycle_preserves_prefix_at_first_and_last_leaf_failure_antigravity(self) -> None:
        """Break caught: deleting or advancing immutable completed leaves across terminal failures."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity  # noqa: PLC0415

        for failed_sequence, expected_count in ((0, 1), (95, 96)):
            with self.subTest(failed_sequence=failed_sequence), tempfile.TemporaryDirectory(prefix="exp449_lifecycle_antigravity_") as temporary:
                parent = Path(temporary); root = parent / "run"; audit = parent / "audit_antigravity.json"
                with self.assertRaisesRegex(RuntimeError, f"exact_next_sequence={expected_count}"):
                    run_synthetic_lifecycle_antigravity(root, "b" * 40, fail_after_leaf=failed_sequence, audit_path=audit)
                self.assertEqual(len(tuple((root / "candidates_antigravity").glob("*.json"))), expected_count)
                self.assertEqual(json.loads(audit.read_text())["exact_next_sequence"], expected_count)
                self.assertFalse(json.loads((root / "pid_antigravity.json").read_text())["active"])
                self.assertFalse((root / "prediction_manifest_antigravity.json").exists())

    def test_private_lifecycle_is_byte_identical_after_real_process_completion_antigravity(self) -> None:
        """Break caught: allowing out-of-order worker completion to alter durable leaf bytes."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_antigravity_") as temporary:
            parent = Path(temporary)
            first = run_synthetic_lifecycle_antigravity(parent / "first", "e" * 40)
            second = run_synthetic_lifecycle_antigravity(parent / "second", "e" * 40)
            self.assertEqual(first, second)
            first_bytes = tuple((parent / "first" / item).read_bytes() for item in first.inventory)
            second_bytes = tuple((parent / "second" / item).read_bytes() for item in second.inventory)
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first.dataset_units, 96)

    def test_real_eleven_worker_scheduler_commits_reversed_completion_in_sequence_antigravity(self) -> None:
        """Break caught: allowing ProcessPool completion order to determine leaf order."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            schedule_synthetic_units_antigravity,
        )

        ordered, completed, provenance = schedule_synthetic_units_antigravity(tuple(range(96)))
        self.assertEqual(ordered, tuple(range(96)))
        self.assertEqual(len(completed), 96)
        self.assertNotEqual(completed, tuple(range(96)))
        self.assertEqual(provenance, {"logical_cpus": 12, "workers": 11})

    def test_prediction_inventory_commits_ordered_96_label_free_units_antigravity(self) -> None:
        """Break caught: unordered, duplicate, non-96, or label-bearing candidate commitments."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            PredictionDependenciesAntigravity,
            predict_with_dependencies_antigravity,
        )

        families = ("Crop", "ElectricDevices", "FordA", "FordB", "Yoga", "ECG5000", "ChlorineConcentration", "StarLightCurves")
        rows = tuple(_BaselineAntigravity(family) for family, count in zip(families, (24, 7, 2, 2, 2, 5, 3, 3), strict=True) for _ in range(count))
        def load(sequence: int, family: str):
            return family + "_" + str(sequence), np.arange(10.0).reshape(5, 2), np.arange(6.0).reshape(3, 2)
        def score(train, test, candidate, seed):
            return np.arange(len(train), dtype=float) + seed / 10_000_000, np.arange(len(test), dtype=float) + seed / 10_000_000
        with tempfile.TemporaryDirectory(prefix="exp449_prediction_antigravity_") as temporary:
            dependencies = PredictionDependenciesAntigravity(
                root=Path(temporary), exp448_authority=_Exp448Antigravity(rows), load_bundle=load,
                score=score, method_audit=lambda: (), mlx_decision=lambda: {"backend": "cpu"}, executor_factory=None,
            )
            authority = predict_with_dependencies_antigravity(dependencies, reviewed_commit="b" * 40)
            self.assertEqual(authority.family_method_pairs, 16)
            self.assertEqual(authority.dataset_units, 96)
            self.assertEqual(authority.label_access_count, 0)
            self.assertEqual(len(authority.candidate_names), 96)
            self.assertEqual(len(set(authority.candidate_names)), 96)
            self.assertEqual(authority.candidate_names[0], "Crop_0__minirocket_local_gap")
            self.assertEqual(authority.candidate_names[-1], "StarLightCurves_47__hydra_local_gap")

    def test_validator_fails_closed_without_official_root_antigravity(self) -> None:
        """Break caught: Task 3 can open a database before descriptor validation fails."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            FrozenPredictionAuthorityAntigravity,
            validate_frozen_prediction_authority_antigravity,
        )

        authority = FrozenPredictionAuthorityAntigravity("c" * 40, 16, 96, (), (), (), "d" * 64, 0)
        with self.assertRaises(ValueError):
            validate_frozen_prediction_authority_antigravity(authority)
