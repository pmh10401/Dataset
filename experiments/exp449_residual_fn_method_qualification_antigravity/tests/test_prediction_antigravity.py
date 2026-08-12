"""Synthetic Task 5 lifecycle tests; no database or official root is touched."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import sys
import tempfile
import traceback
import unittest
import inspect
from types import SimpleNamespace
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
    def test_unexpected_failure_time_prefix_exception_fails_closed_after_stale_progress_proof_antigravity(self) -> None:
        """Break caught: stale progress-era proof advances audit after validation machinery cannot recheck a tampered leaf."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity import progress_antigravity as progress  # noqa: PLC0415

        for failure_sequence in (1, 47, 95):
            with self.subTest(failure_sequence=failure_sequence), tempfile.TemporaryDirectory(prefix="exp449_stale_prefix_antigravity_") as temporary:
                root = Path(temporary) / "run"; audit = Path(temporary) / "audit_antigravity.json"
                primary = RuntimeError(f"primary failure at {failure_sequence}")
                original_validate = progress._CommitSessionAntigravity.validate_prefix
                active_traces: list[object] = []
                original_handler = prediction._handle_lifecycle_failure_antigravity
                def force_after_tamper(session):
                    if len(session._commits) == failure_sequence + 1:
                        raise OSError("unexpected prefix machinery failure")
                    return original_validate(session)
                def observe_handler(*args, **kwargs):
                    active_traces.append(sys.exception().__traceback__)
                    return original_handler(*args, **kwargs)
                def fault(tag, sequence):
                    if (tag, sequence) == ("after_commit", failure_sequence):
                        target = root / "candidates_antigravity" / "000_synthetic_000.json"
                        replacement = root / "replacement_antigravity.json"
                        replacement.write_bytes(target.read_bytes())
                        os.replace(replacement, target)
                        raise primary
                with patch.object(progress._CommitSessionAntigravity, "validate_prefix", autospec=True, side_effect=force_after_tamper), patch.object(prediction, "_handle_lifecycle_failure_antigravity", side_effect=observe_handler):
                    with self.assertRaisesRegex(RuntimeError, f"primary failure at {failure_sequence}") as raised:
                        prediction.run_synthetic_lifecycle_antigravity(root, "z" * 40, audit_path=audit, fault_hook=fault)
                self.assertIs(raised.exception, primary)
                self.assertIn("fault", tuple(frame.name for frame in traceback.extract_tb(active_traces[0])))
                evidence = json.loads(audit.read_text())
                self.assertEqual(evidence["exact_next_sequence"], 0)
                self.assertEqual(evidence["prefix_validation_error"], "OSError: unexpected prefix machinery failure")

    def test_production_and_synthetic_failure_boundary_preserves_primary_when_prefix_and_audit_fail_antigravity(self) -> None:
        """Break caught: either lifecycle lets prefix/audit failure replace its original raised object."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity import progress_antigravity as progress  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity import predecessor_antigravity as predecessor  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity import method_audit_antigravity as method_audit  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity import mlx_gate_antigravity as mlx_gate  # noqa: PLC0415

        def forced_prefix(_session):
            raise OSError("forced prefix validator failure")

        def assert_synthetic(audit_fails: bool) -> None:
            with tempfile.TemporaryDirectory(prefix="exp449_shared_synthetic_antigravity_") as temporary:
                root = Path(temporary) / "run"; audit = Path(temporary) / "audit_antigravity.json"
                primary = RuntimeError("synthetic primary truth")
                def fault(tag, sequence):
                    if (tag, sequence) == ("worker_exception", 0):
                        raise primary
                audit_calls: list[object] = []
                active_traces: list[object] = []
                def write_audit(destination, *args, **kwargs):
                    audit_calls.append((destination, args, kwargs))
                    if audit_fails:
                        raise OSError("forced audit persistence failure")
                    return progress.write_failure_audit_antigravity(audit, *args, **kwargs)
                original_handler = prediction._handle_lifecycle_failure_antigravity
                def observe_handler(*args, **kwargs):
                    active_traces.append(sys.exception().__traceback__)
                    return original_handler(*args, **kwargs)
                with patch.object(progress._CommitSessionAntigravity, "validate_prefix", autospec=True, side_effect=forced_prefix), patch.object(prediction, "write_failure_audit_antigravity", side_effect=write_audit), patch.object(prediction, "_handle_lifecycle_failure_antigravity", side_effect=observe_handler):
                    with self.assertRaisesRegex(RuntimeError, "synthetic primary truth") as raised:
                        prediction.run_synthetic_lifecycle_antigravity(root, "s" * 40, audit_path=audit, fault_hook=fault)
                self.assertIs(raised.exception, primary)
                self.assertIn("fault", tuple(frame.name for frame in traceback.extract_tb(active_traces[0])))
                self.assertEqual(len(audit_calls), 1)
                notes = "\n".join(getattr(raised.exception, "__notes__", ()))
                if audit_fails:
                    self.assertIn("failure_audit_error=OSError: forced audit persistence failure", notes)
                else:
                    evidence = json.loads(audit.read_text())
                    self.assertEqual(evidence["exact_next_sequence"], 0)
                    self.assertEqual(evidence["prefix_validation_error"], "OSError: forced prefix validator failure")

        def assert_production(audit_fails: bool) -> None:
            with tempfile.TemporaryDirectory(prefix="exp449_shared_production_antigravity_") as temporary:
                root = Path(temporary) / "run"; audit = Path(temporary) / "audit_antigravity.json"
                families = tuple(family for family, count in zip((name for name, _ in prediction.CANDIDATE_MATRIX_ANTIGRAVITY), (24, 7, 2, 2, 2, 5, 3, 3), strict=True) for _ in range(count))
                authority = SimpleNamespace(
                    target_baseline_rows=tuple(_BaselineAntigravity(family) for family in families),
                    target_dataset_names=tuple(f"{family}_production_{index:03d}" for index, family in enumerate(families)),
                    artifact_map_sha256="a" * 64, evaluation_sha256="b" * 64, decision_sha256="c" * 64,
                )
                mlx = mlx_gate.MlxDecisionAntigravity("cpu", "test", False, None, None, None, None, None, None, False, None, None, False, None)
                primary = RuntimeError("production primary truth")
                class AbortPool:
                    def __init__(self, *args, **kwargs): pass
                    def __enter__(self): raise primary
                    def __exit__(self, *args): return False
                audit_calls: list[object] = []
                active_traces: list[object] = []
                def write_audit(destination, *args, **kwargs):
                    audit_calls.append((destination, args, kwargs))
                    if audit_fails:
                        raise OSError("forced audit persistence failure")
                    return None
                original_handler = prediction._handle_lifecycle_failure_antigravity
                def observe_handler(*args, **kwargs):
                    active_traces.append(sys.exception().__traceback__)
                    return original_handler(*args, **kwargs)
                with patch.object(prediction, "_OFFICIAL_ROOT_ANTIGRAVITY", root), patch.object(predecessor, "capture_exp448_authority_antigravity", return_value=authority), patch.object(method_audit, "audit_methods_antigravity", return_value=()), patch.object(mlx_gate, "assess_mlx_antigravity", return_value=mlx), patch.object(prediction, "ProcessPoolExecutor", AbortPool), patch.object(progress._CommitSessionAntigravity, "validate_prefix", autospec=True, side_effect=forced_prefix), patch.object(prediction, "write_failure_audit_antigravity", side_effect=write_audit), patch.object(prediction, "_handle_lifecycle_failure_antigravity", side_effect=observe_handler):
                    with self.assertRaisesRegex(RuntimeError, "production primary truth") as raised:
                        prediction.predict_antigravity("p" * 40)
                self.assertIs(raised.exception, primary)
                self.assertIn("__enter__", tuple(frame.name for frame in traceback.extract_tb(active_traces[0])))
                self.assertEqual(len(audit_calls), 1)
                notes = "\n".join(getattr(raised.exception, "__notes__", ()))
                if audit_fails:
                    self.assertIn("failure_audit_error=OSError: forced audit persistence failure", notes)
                else:
                    self.assertEqual(audit_calls[0][2]["exact_next_sequence"], 0)
                    self.assertEqual(audit_calls[0][2]["prefix_validation_error"], "OSError: forced prefix validator failure")

        for lifecycle in (assert_synthetic, assert_production):
            with self.subTest(lifecycle=lifecycle.__name__, audit_fails=False):
                lifecycle(False)
            with self.subTest(lifecycle=lifecycle.__name__, audit_fails=True):
                lifecycle(True)
        source = inspect.getsource(prediction)
        self.assertEqual(source.count("_handle_lifecycle_failure_antigravity("), 3)

    def test_first_fstat_failure_with_official_shaped_replacement_never_advances_receipt_prefix_antigravity(self) -> None:
        """Break caught: pathname-valid next leaves advance failure evidence without a writer receipt."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity import progress_antigravity as progress  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity  # noqa: PLC0415

        for failed_sequence in (0, 47, 95):
            with self.subTest(failed_sequence=failed_sequence), tempfile.TemporaryDirectory(prefix="exp449_receipt_fstat_antigravity_") as temporary:
                parent = Path(temporary); root = parent / "run"; audit = parent / "audit_antigravity.json"
                leaves = root / "candidates_antigravity"
                destination = leaves / f"{failed_sequence:03d}_synthetic_{failed_sequence:03d}.json"
                family_order = tuple(name for name, _ in prediction.CANDIDATE_MATRIX_ANTIGRAVITY)
                family = tuple(item for item, count in zip(family_order, (24, 7, 2, 2, 2, 5, 3, 3), strict=True) for _ in range(count))[failed_sequence // 2]
                method = dict(prediction.CANDIDATE_MATRIX_ANTIGRAVITY)[family][failed_sequence % 2]
                replacement_payload = prediction._candidate_payload_antigravity(
                    failed_sequence, f"{family}_synthetic_{failed_sequence // 2:03d}", family, method,
                    [], 3, 5,
                    [{"seed": 20260721, "feature_dimension": 1}, {"seed": 20261721, "feature_dimension": 1}],
                )
                returned: list[tuple[str, object]] = []; fired = False
                original_fstat = progress.os.fstat; original_commit = prediction.commit_candidate_leaf_antigravity

                def replace_then_fail(fd):
                    nonlocal fired
                    if not fired and destination.exists():
                        fired = True
                        replacement = parent / "official_shaped_replacement_antigravity.json"
                        replacement.write_bytes(canonical_json_bytes_antigravity(replacement_payload))
                        os.replace(replacement, destination)
                        raise OSError("actual first fstat failure")
                    return original_fstat(fd)

                def track_return(directory, name, value, **kwargs):
                    result = original_commit(directory, name, value, **kwargs)
                    returned.append((name, result))
                    return result

                with patch.object(progress.os, "fstat", side_effect=replace_then_fail), patch.object(prediction, "commit_candidate_leaf_antigravity", side_effect=track_return):
                    with self.assertRaisesRegex(OSError, "actual first fstat failure"):
                        prediction.run_synthetic_lifecycle_antigravity(root, "r" * 40, audit_path=audit)
                self.assertTrue(fired)
                self.assertNotIn(destination.name, tuple(name for name, _result in returned))
                self.assertEqual(json.loads(destination.read_text()), replacement_payload)
                self.assertEqual(json.loads((root / "progress_antigravity.json").read_text())["dataset_unit_completed"], failed_sequence)
                evidence = json.loads(audit.read_text())
                self.assertEqual(evidence["exact_next_sequence"], failed_sequence)
                self.assertIn("candidate_cleanup_error=OSError: unable to prove created inode after fstat failure", evidence["primary_error_notes"])
                self.assertFalse((root / "prediction_manifest_antigravity.json").exists())
                self.assertFalse(json.loads((root / "pid_antigravity.json").read_text())["active"])

    def test_actual_processpool_pending_commit_loop_rejects_predecessor_drift_0_47_95_antigravity(self) -> None:
        """Break caught: the ProcessPool/as-completed pending loop commits after predecessor drift."""
        from concurrent.futures import ProcessPoolExecutor
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import _consume_pending_executor_antigravity, _synthetic_worker_antigravity  # noqa: PLC0415
        for drift_at in (0, 47, 95):
            with self.subTest(drift_at=drift_at):
                original = ("frozen",); calls = 0; committed: list[int] = []
                def capture():
                    nonlocal calls
                    calls += 1
                    return ("drift",) if calls == drift_at + 1 else original
                with ProcessPoolExecutor(max_workers=11) as executor:
                    futures = {executor.submit(_synthetic_worker_antigravity, item): item for item in range(96)}
                    with self.assertRaisesRegex(ValueError, "predecessor drift"):
                        _consume_pending_executor_antigravity(futures, original, capture, lambda sequence, result: committed.append(sequence), lambda result: result)
                self.assertEqual(committed, list(range(drift_at)))
    def test_canonical_production_candidate_builder_has_exact_validator_schema_antigravity(self) -> None:
        """Break caught: a real worker payload omits dataset/seeds or adds unvalidated fields."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import _candidate_payload_antigravity  # noqa: PLC0415
        payload = _candidate_payload_antigravity(0, "Crop_normal_1", "Crop", "minirocket_local_gap", [0, 2], 3, 5, [{"seed": 20260721, "feature_dimension": 1}, {"seed": 20261721, "feature_dimension": 1}])
        self.assertEqual(set(payload), {"sequence", "candidate_name", "dataset", "family", "method", "indices", "contains_test_labels", "target_length", "train_count", "test_count", "seeds", "score_identities", "alpha"})
        self.assertEqual(payload["dataset"], "Crop_normal_1")
        self.assertEqual(payload["seeds"], [20260721, 20261721])
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

    def test_post_manifest_validation_failure_removes_durable_manifest_antigravity(self) -> None:
        """Break caught: a manifest survives when full post-commit descriptor validation fails."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import run_synthetic_lifecycle_antigravity  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="exp449_post_manifest_antigravity_") as temporary:
            root = Path(temporary) / "run"; audit = Path(temporary) / "audit_antigravity.json"
            def fault(tag, sequence):
                if (tag, sequence) == ("post_manifest_validation", 96): raise ValueError("post manifest invalid")
            with self.assertRaisesRegex(ValueError, "post manifest invalid"):
                run_synthetic_lifecycle_antigravity(root, "a" * 40, audit_path=audit, fault_hook=fault)
            self.assertFalse((root / "prediction_manifest_antigravity.json").exists())
            self.assertEqual(json.loads(audit.read_text())["exact_next_sequence"], 96)

    def test_manifest_path_stat_failure_cannot_escape_its_descriptor_commit_antigravity(self) -> None:
        """Break caught: a post-commit pathname stat failure leaves a manifest and masks validation."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="exp449_manifest_stat_antigravity_") as temporary:
            root = Path(temporary) / "run"; audit = Path(temporary) / "audit_antigravity.json"; original_stat = prediction.os.stat
            def fail_legacy_manifest_stat(path, *args, **kwargs):
                if path == root / "prediction_manifest_antigravity.json":
                    raise OSError("legacy manifest pathname stat")
                return original_stat(path, *args, **kwargs)
            def fault(tag, sequence):
                if (tag, sequence) == ("post_manifest_validation", 96):
                    raise RuntimeError("post manifest validation truth")
            with patch.object(prediction.os, "stat", side_effect=fail_legacy_manifest_stat):
                with self.assertRaisesRegex(RuntimeError, "post manifest validation truth"):
                    prediction.run_synthetic_lifecycle_antigravity(root, "a" * 40, audit_path=audit, fault_hook=fault)
            self.assertFalse((root / "prediction_manifest_antigravity.json").exists())

    def test_manifest_replacement_before_legacy_identity_stat_is_never_deleted_antigravity(self) -> None:
        """Break caught: a manifest replacement becomes cleanup authority through a second pathname stat."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="exp449_manifest_replace_antigravity_") as temporary:
            root = Path(temporary) / "run"; audit = Path(temporary) / "audit_antigravity.json"; destination = root / "prediction_manifest_antigravity.json"; original_commit = prediction.commit_candidate_leaf_antigravity
            def commit_then_replace(directory, name, value, **kwargs):
                result = original_commit(directory, name, value, **kwargs)
                if name == destination.name:
                    replacement = directory / "replacement_antigravity.json"; replacement.write_text('{"replacement":true}\n')
                    os.replace(replacement, destination)
                return result
            def fault(tag, sequence):
                if (tag, sequence) == ("post_manifest_validation", 96):
                    raise RuntimeError("post manifest validation truth")
            with patch.object(prediction, "commit_candidate_leaf_antigravity", side_effect=commit_then_replace):
                with self.assertRaisesRegex(RuntimeError, "post manifest validation truth"):
                    prediction.run_synthetic_lifecycle_antigravity(root, "a" * 40, audit_path=audit, fault_hook=fault)
            self.assertEqual(json.loads(destination.read_text()), {"replacement": True})

    def test_manifest_cleanup_pid_and_audit_errors_do_not_replace_validation_truth_antigravity(self) -> None:
        """Break caught: manifest cleanup escaping the handler replaces the original validation failure."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="exp449_manifest_multi_error_antigravity_") as temporary:
            root = Path(temporary) / "run"; audit = Path(temporary) / "audit_antigravity.json"; original_pid = prediction._set_pid_inactive_antigravity; pid_calls = 0
            def fault(tag, sequence):
                if (tag, sequence) == ("post_manifest_validation", 96):
                    raise RuntimeError("post manifest validation truth")
            def fail_cleanup_pid(path):
                nonlocal pid_calls
                pid_calls += 1
                if pid_calls == 1:
                    return original_pid(path)
                raise OSError("pid cleanup failure")
            with patch.object(prediction, "_remove_exact_manifest_antigravity", side_effect=OSError("manifest cleanup failure")), patch.object(prediction, "_set_pid_inactive_antigravity", side_effect=fail_cleanup_pid), patch.object(prediction, "write_failure_audit_antigravity", side_effect=OSError("audit persistence failure")):
                with self.assertRaisesRegex(RuntimeError, "post manifest validation truth") as raised:
                    prediction.run_synthetic_lifecycle_antigravity(root, "a" * 40, audit_path=audit, fault_hook=fault)
            notes = "\n".join(getattr(raised.exception, "__notes__", ()))
            self.assertIn("manifest_cleanup_error=OSError: manifest cleanup failure", notes)
            self.assertIn("pid_cleanup_error=OSError: pid cleanup failure", notes)
            self.assertIn("failure_audit_error=OSError: audit persistence failure", notes)

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
                self.assertEqual(len(tuple((root / "candidates_antigravity").glob("*.json"))), expected_prefix)
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
            original = prediction._CommitSessionAntigravity.record_progress
            def fail_at_48(session):
                if len(session._commits) == 48:
                    raise OSError("persistent progress channel failure")
                return original(session)
            with patch.object(prediction._CommitSessionAntigravity, "record_progress", autospec=True, side_effect=fail_at_48), self.assertRaisesRegex(OSError, "persistent progress"):
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

    def test_descriptor_capture_returns_actual_task5_authority_and_canonical_rows_antigravity(self) -> None:
        """Break caught: Task6 accepts a caller-projected row tuple instead of a validated Task5 capture."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import capture_frozen_prediction_authority_antigravity, run_synthetic_lifecycle_antigravity

        with tempfile.TemporaryDirectory(prefix="exp449_task5_capture_antigravity_") as temporary:
            root = Path(temporary) / "run"
            authority = run_synthetic_lifecycle_antigravity(root, "a" * 40)
            capture = capture_frozen_prediction_authority_antigravity(root)
            self.assertEqual(capture.authority, authority)
            self.assertEqual((len(capture.canonical_rows.splitlines()), len(capture.canonical_row_sha256), capture.authority.label_access_count), (96, 96, 0))

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
