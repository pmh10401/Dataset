"""Synthetic Task 5 lifecycle tests; no database or official root is touched."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _BaselineAntigravity:
    family: str


@dataclass(frozen=True)
class _Exp448Antigravity:
    target_baseline_rows: tuple[_BaselineAntigravity, ...]


class PredictionAntigravityTests(unittest.TestCase):
    def test_manifest_durable_boundary_fault_matrix_leaves_no_manifest_or_authority_antigravity(self) -> None:
        """Break caught: treating a manifest syscall failure as a successful terminal authority."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity  # noqa: PLC0415
        for tag in ("manifest_cache_create", "manifest_write", "manifest_fsync_leaf", "manifest_cache_fsync", "manifest_cache_reread", "manifest_cache_sha_mismatch", "manifest_fstat_read", "manifest_stat_name"):
            with self.subTest(tag=tag), tempfile.TemporaryDirectory(prefix="exp449_manifest_fault_antigravity_") as temporary:
                root = Path(temporary) / "run"; audit = Path(temporary) / "audit_antigravity.json"
                def fault(current, sequence):
                    if current == tag and sequence == 96:
                        raise OSError(tag)
                with self.assertRaisesRegex(OSError, tag):
                    run_synthetic_lifecycle_antigravity(root, "a" * 40, audit_path=audit, fault_hook=fault)
                self.assertFalse((root / "prediction_manifest_antigravity.json").exists())
                self.assertEqual(len(tuple((root / "candidates_antigravity").iterdir())), 96)
                self.assertEqual(json.loads(audit.read_text())["exact_next_sequence"], 96)

    def test_validator_rejects_exact_context_source_mlx_type_and_rehash_attacks_antigravity(self) -> None:
        """Break caught: accepting rehashed semantic context aliases, bool-as-int, or source order drift."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity, validate_prediction_root_antigravity  # noqa: PLC0415
        import hashlib
        for mutate in ("extra", "bool", "source_order", "mlx_extra"):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory(prefix="exp449_schema_antigravity_") as temporary:
                root = Path(temporary) / "run"; authority = run_synthetic_lifecycle_antigravity(root, "b" * 40)
                context_path = root / "context_antigravity.json"; context = json.loads(context_path.read_text())
                if mutate == "extra": context["unexpected"] = 1
                elif mutate == "bool": context["workers"] = True
                elif mutate == "source_order": context["source_identity"]["target_dataset_names"] = list(reversed(context["source_identity"]["target_dataset_names"]))
                else: context["mlx"]["unexpected"] = False
                context_path.write_bytes(canonical_json_bytes_antigravity(context))
                manifest_path = root / "prediction_manifest_antigravity.json"; manifest = json.loads(manifest_path.read_text())
                manifest["context_sha256"] = hashlib.sha256(context_path.read_bytes()).hexdigest(); manifest_path.write_bytes(canonical_json_bytes_antigravity(manifest))
                with self.assertRaises(ValueError): validate_prediction_root_antigravity(root, authority)
    def test_dependency_lifecycle_revalidates_predecessor_before_commit_antigravity(self) -> None:
        """Break caught: the injectable production-loop seam writes the first leaf after predecessor drift."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import PredictionDependenciesAntigravity, predict_with_dependencies_antigravity  # noqa: PLC0415
        families = ("Crop", "ElectricDevices", "FordA", "FordB", "Yoga", "ECG5000", "ChlorineConcentration", "StarLightCurves")
        rows = tuple(_BaselineAntigravity(family) for family, count in zip(families, (24, 7, 2, 2, 2, 5, 3, 3), strict=True) for _ in range(count))
        authority = _Exp448Antigravity(rows)
        with tempfile.TemporaryDirectory(prefix="exp449_dep_drift_antigravity_") as temporary:
            dependencies = PredictionDependenciesAntigravity(Path(temporary), authority, lambda sequence, family: (f"{family}_{sequence}", np.ones((5, 2)), np.ones((3, 2))), lambda train, test, method, seed: (np.arange(5.0), np.arange(3.0)), lambda: (), lambda: {}, None, lambda: _Exp448Antigravity(rows + (_BaselineAntigravity("Crop"),)))
            with self.assertRaisesRegex(ValueError, "predecessor drift"):
                predict_with_dependencies_antigravity(dependencies, reviewed_commit="a" * 40)
            self.assertFalse((Path(temporary) / "candidates_antigravity").exists())
    def test_dependency_pending_commit_loop_rejects_drift_at_0_47_95_antigravity(self) -> None:
        """Break caught: the real dependency lifecycle commits its exact next unit after a fresh predecessor drift."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import PredictionDependenciesAntigravity, predict_with_dependencies_antigravity  # noqa: PLC0415
        families = ("Crop", "ElectricDevices", "FordA", "FordB", "Yoga", "ECG5000", "ChlorineConcentration", "StarLightCurves")
        rows = tuple(_BaselineAntigravity(family) for family, count in zip(families, (24, 7, 2, 2, 2, 5, 3, 3), strict=True) for _ in range(count))
        original = _Exp448Antigravity(rows); changed = _Exp448Antigravity(rows + (_BaselineAntigravity("Crop"),))
        for drift_at in (0, 47, 95):
            with self.subTest(drift_at=drift_at), tempfile.TemporaryDirectory(prefix="exp449_dependency_drift_antigravity_") as temporary:
                captures = 0
                def capture():
                    nonlocal captures
                    captures += 1
                    return changed if captures == drift_at + 1 else original
                dependencies = PredictionDependenciesAntigravity(Path(temporary), original, lambda sequence, family: (f"{family}_{sequence}", np.ones((5, 2)), np.ones((3, 2))), lambda train, test, method, seed: (np.arange(5.0), np.arange(3.0)), lambda: (), lambda: {}, None, capture)
                with self.assertRaisesRegex(ValueError, "predecessor drift"):
                    predict_with_dependencies_antigravity(dependencies, reviewed_commit="a" * 40)
                directory = Path(temporary) / "candidates_antigravity"
                self.assertEqual(0 if not directory.exists() else len(tuple(directory.iterdir())), drift_at)
    def test_descriptor_validator_rejects_same_size_leaf_replacement_during_capture_antigravity(self) -> None:
        """Break caught: trusting bytes from a pinned descriptor after its namespace leaf is replaced."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="exp449_descriptor_race_antigravity_") as temporary:
            root = Path(temporary) / "run"; authority = prediction.run_synthetic_lifecycle_antigravity(root, "a" * 40)
            original = prediction._read_pinned_antigravity; fired = False
            def replace(fd):
                nonlocal fired
                result = original(fd)
                if not fired:
                    fired = True; target = root / authority.inventory[0]; replacement = root / "replacement_antigravity.json"; replacement.write_bytes(target.read_bytes()); os.replace(replacement, target)
                return result
            with patch.object(prediction, "_read_pinned_antigravity", side_effect=replace), self.assertRaises(ValueError):
                prediction.validate_prediction_root_antigravity(root, authority)
    def test_descriptor_validator_fstat_fault_does_not_leak_retained_descriptor_antigravity(self) -> None:
        """Break caught: a pin fstat exception leaks the descriptor opened before identity validation."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="exp449_fd_antigravity_") as temporary:
            root = Path(temporary) / "run"; authority = prediction.run_synthetic_lifecycle_antigravity(root, "a" * 40)
            before = len(os.listdir("/dev/fd")); original = prediction.os.fstat; calls = 0
            def fail_once(fd):
                nonlocal calls
                calls += 1
                if calls == 3: raise OSError("fstat injected")
                return original(fd)
            with patch.object(prediction.os, "fstat", side_effect=fail_once), self.assertRaisesRegex(OSError, "fstat injected"):
                prediction.validate_prediction_root_antigravity(root, authority)
            self.assertEqual(len(os.listdir("/dev/fd")), before)
    def test_production_precommit_revalidation_rejects_drift_at_0_47_95_antigravity(self) -> None:
        """Break caught: production commits a candidate after fresh Exp448 authority differs."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import _revalidate_predecessor_before_commit_antigravity  # noqa: PLC0415
        for sequence in (0, 47, 95):
            with self.subTest(sequence=sequence):
                original = ("authority", sequence)
                with self.assertRaisesRegex(ValueError, "predecessor drift"):
                    _revalidate_predecessor_before_commit_antigravity(original, lambda: ("authority", sequence + 1))
    def test_table_driven_pre_and_post_commit_faults_preserve_exact_prefix_antigravity(self) -> None:
        """Break caught: deriving failure progress from scheduled work instead of durable leaves."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity, validate_prediction_root_antigravity  # noqa: PLC0415

        for stage, sequence, expected_prefix in (("before_commit", 0, 0), ("before_commit", 47, 47), ("before_commit", 95, 95), ("after_commit", 0, 1), ("after_commit", 47, 48), ("after_commit", 95, 96)):
            with self.subTest(stage=stage, sequence=sequence), tempfile.TemporaryDirectory(prefix="exp449_fault_antigravity_") as temporary:
                parent = Path(temporary); root = parent / "run"; audit = parent / "audit_antigravity.json"
                def fault(current_stage, current_sequence):
                    if (current_stage, current_sequence) == (stage, sequence):
                        raise RuntimeError(f"{stage} fault")
                with self.assertRaisesRegex(RuntimeError, f"{stage} fault"):
                    run_synthetic_lifecycle_antigravity(root, "3" * 40, audit_path=audit, fault_hook=fault)
                leaves = tuple(sorted((root / "candidates_antigravity").glob("*.json")))
                self.assertEqual(len(leaves), expected_prefix)
                self.assertEqual(json.loads(audit.read_text())["exact_next_sequence"], expected_prefix)
                self.assertFalse((root / "prediction_manifest_antigravity.json").exists())
                self.assertFalse(json.loads((root / "pid_antigravity.json").read_text())["active"])

    def test_named_fault_matrix_preserves_boundary_specific_prefixes_antigravity(self) -> None:
        """Break caught: treating cache and channel failures as one generic scheduling failure."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity  # noqa: PLC0415

        pre_tags = ("worker_exception", "scorer_exception", "nonfinite_scores", "cache_create", "cache_fsync", "cache_reread", "cache_sha_mismatch")
        post_tags = ("progress_json", "progress_jsonl", "progress_text")
        for tag in pre_tags + post_tags:
            sequence, expected_prefix = (0, 0) if tag in pre_tags else (47, 48)
            with self.subTest(tag=tag), tempfile.TemporaryDirectory(prefix="exp449_fault_antigravity_") as temporary:
                parent = Path(temporary); root = parent / "run"; audit = parent / "audit_antigravity.json"
                def fault(current_tag, current_sequence):
                    if (current_tag, current_sequence) == (tag, sequence):
                        raise RuntimeError(f"{tag} injected")
                with self.assertRaisesRegex(RuntimeError, f"{tag} injected"):
                    run_synthetic_lifecycle_antigravity(root, "4" * 40, audit_path=audit, fault_hook=fault)
                from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import verified_prefix_antigravity  # noqa: PLC0415
                self.assertEqual(verified_prefix_antigravity(root / "candidates_antigravity", tuple(f"{item:03d}_synthetic_{item:03d}.json" for item in range(96))), expected_prefix)
                evidence = json.loads(audit.read_text())
                self.assertEqual(evidence["exact_next_sequence"], expected_prefix)
                self.assertIn(tag, evidence["primary_error"])
                self.assertFalse((root / "prediction_manifest_antigravity.json").exists())
                self.assertFalse(json.loads((root / "pid_antigravity.json").read_text())["active"])

    def test_predecessor_drift_matrix_rejects_next_leaf_at_0_47_95_antigravity(self) -> None:
        """Break caught: committing after the immutable predecessor authority has drifted."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity  # noqa: PLC0415

        for sequence in (0, 47, 95):
            with self.subTest(sequence=sequence), tempfile.TemporaryDirectory(prefix="exp449_drift_antigravity_") as temporary:
                parent = Path(temporary); root = parent / "run"; audit = parent / "audit_antigravity.json"
                def fault(tag, current):
                    if tag == "predecessor_drift" and current == sequence:
                        raise RuntimeError("predecessor_drift injected")
                with self.assertRaisesRegex(RuntimeError, "predecessor_drift"):
                    run_synthetic_lifecycle_antigravity(root, "5" * 40, audit_path=audit, fault_hook=fault)
                self.assertEqual(len(tuple((root / "candidates_antigravity").glob("*.json"))), sequence)
                self.assertEqual(json.loads(audit.read_text())["exact_next_sequence"], sequence)

    def test_pid_and_audit_faults_keep_primary_truth_antigravity(self) -> None:
        """Break caught: returning authority after PID close failure or masking a primary failure with audit failure."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_pid_antigravity_") as temporary:
            parent = Path(temporary); root = parent / "run"; audit = parent / "audit_antigravity.json"
            def pid_fault(tag, sequence):
                if tag == "pid_close": raise OSError("pid_close injected")
            with self.assertRaisesRegex(OSError, "pid_close injected"):
                run_synthetic_lifecycle_antigravity(root, "6" * 40, audit_path=audit, fault_hook=pid_fault)
            self.assertEqual(len(tuple((root / "candidates_antigravity").glob("*.json"))), 96)
            self.assertFalse(json.loads((root / "pid_antigravity.json").read_text())["active"])
            self.assertEqual(json.loads(audit.read_text())["exact_next_sequence"], 96)
        with tempfile.TemporaryDirectory(prefix="exp449_audit_antigravity_") as temporary:
            root = Path(temporary) / "run"
            def audit_fault(tag, sequence):
                if tag == "cache_create" and sequence == 0: raise RuntimeError("cache_create injected")
                if tag == "failure_audit": raise OSError("failure_audit injected")
            with self.assertRaisesRegex(RuntimeError, "cache_create injected"):
                run_synthetic_lifecycle_antigravity(root, "7" * 40, fault_hook=audit_fault)
            self.assertEqual(len(tuple((root / "candidates_antigravity").glob("*.json"))), 0)
    def test_progress_channel_failure_preserves_committed_leaf_and_audits_exact_next_antigravity(self) -> None:
        """Break caught: discarding a committed candidate or trusting failed progress after a channel error."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_failure_antigravity_") as temporary:
            parent = Path(temporary); root = parent / "run"; audit = parent / "audit_antigravity.json"
            original = prediction.record_verified_progress_antigravity
            def fail_at_48(path, inventory, completed, **_kwargs):
                if completed == 48:
                    raise OSError("persistent progress channel failure")
                return original(path, inventory, completed, **_kwargs)
            with patch.object(prediction, "record_verified_progress_antigravity", side_effect=fail_at_48), self.assertRaisesRegex(OSError, "persistent progress"):
                prediction.run_synthetic_lifecycle_antigravity(root, "1" * 40, audit_path=audit)
            self.assertEqual(len(tuple((root / "candidates_antigravity").glob("*.json"))), 48)
            self.assertEqual(json.loads(audit.read_text())["exact_next_sequence"], 48)
            self.assertFalse(json.loads((root / "pid_antigravity.json").read_text())["active"])

    def test_audit_write_failure_keeps_primary_failure_fact_antigravity(self) -> None:
        """Break caught: replacing the primary worker failure when independent audit persistence also fails."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_failure_antigravity_") as temporary:
            root = Path(temporary) / "run"
            with patch.object(prediction, "write_failure_audit_antigravity", side_effect=OSError("audit unavailable")):
                with self.assertRaisesRegex(RuntimeError, "synthetic post-leaf failure exact_next_sequence=1"):
                    prediction.run_synthetic_lifecycle_antigravity(root, "2" * 40, fail_after_leaf=0)
            self.assertEqual(len(tuple((root / "candidates_antigravity").glob("*.json"))), 1)
            self.assertFalse(json.loads((root / "pid_antigravity.json").read_text())["active"])
    def test_lifecycle_catch_preserves_primary_pid_cleanup_and_audit_failures_antigravity(self) -> None:
        """Break caught: terminal cleanup or audit persistence masks the original production-path failure."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="exp449_catch_antigravity_") as temporary:
            root = Path(temporary) / "run"; audit = Path(temporary) / "audit_antigravity.json"; original_pid = prediction._set_pid_inactive_antigravity; calls = 0
            def fault(tag, sequence):
                if tag == "cache_create" and sequence == 0: raise RuntimeError("primary cache failure")
            def bad_pid(path):
                nonlocal calls
                calls += 1
                if calls == 1: raise OSError("pid cleanup failure")
                return original_pid(path)
            with patch.object(prediction, "_set_pid_inactive_antigravity", side_effect=bad_pid), patch.object(prediction, "write_failure_audit_antigravity", side_effect=OSError("audit persistence failure")):
                with self.assertRaisesRegex(RuntimeError, "primary cache failure") as raised:
                    prediction.run_synthetic_lifecycle_antigravity(root, "c" * 40, audit_path=audit, fault_hook=fault)
            notes = "\n".join(getattr(raised.exception, "__notes__", ()))
            self.assertIn("pid_cleanup_error=OSError: pid cleanup failure", notes)
            self.assertIn("failure_audit_error=OSError: audit persistence failure", notes)
    def test_private_validator_reconstructs_exact_dataset_order_and_rejects_tampered_leaf_antigravity(self) -> None:
        """Break caught: accepting a rehashed/tampered manifest or deriving order without canonical leaves."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            run_synthetic_lifecycle_antigravity,
            validate_prediction_root_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_validator_antigravity_") as temporary:
            root = Path(temporary) / "run"
            authority = run_synthetic_lifecycle_antigravity(root, "f" * 40)
            self.assertEqual(len(validate_prediction_root_antigravity(root, authority)), 48)
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

    def test_private_validator_subprocess_firewall_never_imports_source_or_sqlite_antigravity(self) -> None:
        """Break caught: validator importing a DB-capable Task-3 source or sqlite module."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity, validate_prediction_root_antigravity  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_firewall_antigravity_") as temporary:
            root = Path(temporary) / "run"; authority = run_synthetic_lifecycle_antigravity(root, "8" * 40)
            program = """import builtins,sys
orig=builtins.__import__
def deny(name,*a,**k):
    if name == 'sqlite3' or name.endswith('.sources_antigravity'): raise AssertionError(name)
    return orig(name,*a,**k)
builtins.__import__=deny
from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import FrozenPredictionAuthorityAntigravity,validate_prediction_root_antigravity
v=__import__('json').loads(sys.argv[2]); a=FrozenPredictionAuthorityAntigravity(v[0],v[1],v[2],tuple(v[3]),tuple(v[4]),tuple(tuple(p) for p in v[5]),v[6],v[7])
print(len(validate_prediction_root_antigravity(__import__('pathlib').Path(sys.argv[1]),a)))
"""
            encoded = json.dumps([authority.reviewed_commit, authority.family_method_pairs, authority.dataset_units, list(authority.candidate_names), list(authority.inventory), [list(pair) for pair in authority.leaf_sha256], authority.aggregate_sha256, authority.label_access_count])
            result = subprocess.run([sys.executable, "-c", program, str(root), encoded], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "48")
        with tempfile.TemporaryDirectory(prefix="exp449_validator_antigravity_") as temporary:
            root = Path(temporary) / "run"
            authority = run_synthetic_lifecycle_antigravity(root, "b" * 40)
            (root / "extra_antigravity.json").write_text("{}\n")
            with self.assertRaises(ValueError):
                validate_prediction_root_antigravity(root, authority)

    def test_private_validator_rejects_candidates_extra_gap_and_directory_replacement_antigravity(self) -> None:
        """Break caught: accepting extra, missing, or replaced candidates namespaces."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity, validate_prediction_root_antigravity  # noqa: PLC0415
        for mode in ("extra", "gap", "replace"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="exp449_candidates_antigravity_") as temporary:
                root = Path(temporary) / "run"; authority = run_synthetic_lifecycle_antigravity(root, "9" * 40); candidates = root / "candidates_antigravity"
                if mode == "extra": (candidates / "extra_antigravity.json").write_text("{}\n")
                elif mode == "gap": (candidates / Path(authority.inventory[0]).name).unlink()
                else:
                    replacement = Path(temporary) / "replacement_antigravity"; replacement.mkdir(); os.replace(candidates, replacement); candidates.mkdir()
                with self.assertRaises(ValueError): validate_prediction_root_antigravity(root, authority)

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
