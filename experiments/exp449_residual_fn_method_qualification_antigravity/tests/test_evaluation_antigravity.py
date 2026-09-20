"""Synthetic-only Task 6 evaluation boundaries."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.exp449_residual_fn_method_qualification_antigravity.config_antigravity import CANDIDATE_MATRIX_ANTIGRAVITY
from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity, DatasetBaselineAntigravity


_FAMILIES_ANTIGRAVITY = tuple(name for name, _ in CANDIDATE_MATRIX_ANTIGRAVITY)
_COUNTS_ANTIGRAVITY = (24, 7, 2, 2, 2, 5, 3, 3)


@dataclass(frozen=True, slots=True)
class _Exp448Antigravity:
    target_dataset_names: tuple[str, ...]
    target_baseline_rows: tuple[DatasetBaselineAntigravity, ...]
    baseline_confusion: ConfusionAntigravity
    artifact_map_sha256: str = "1" * 64
    evaluation_sha256: str = "2" * 64
    decision_sha256: str = "3" * 64


class _SessionAntigravity:
    def __init__(self, labels: dict[str, tuple[tuple[int, int], ...]], *, query_error: BaseException | None = None, close_error: BaseException | None = None) -> None:
        self.labels, self.query_error, self.close_error = labels, query_error, close_error
        self.queried: list[str] = []
        self.closed = False

    def labels_for_dataset_antigravity(self, dataset: str) -> tuple[tuple[int, int], ...]:
        self.queried.append(dataset)
        if self.query_error is not None:
            raise self.query_error
        return self.labels[dataset]

    def close_antigravity(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _DependenciesAntigravity:
    def __init__(self, root: Path, *, open_error: BaseException | None = None, query_error: BaseException | None = None, close_error: BaseException | None = None, drift_at: int | None = None) -> None:
        datasets = ( *(f"ChlorineConcentration_normal_{i}" for i in range(1, 4)), *(f"Crop_normal_{i}" for i in (1, *range(10, 20), 2, *range(20, 25), *range(3, 10))), *(f"ECG5000_normal_{i}" for i in range(1, 6)), *(f"ElectricDevices_normal_{i}" for i in range(1, 8)), "FordA_normal_-1", "FordA_normal_1", "FordB_normal_-1", "FordB_normal_1", *(f"StarLightCurves_normal_{i}" for i in range(1, 4)), "Yoga_normal_1", "Yoga_normal_2")
        lengths = (1500,) * 47 + (1800,)
        labels = {dataset: tuple((index, 1 if index < 40 else 0) for index in range(length)) for dataset, length in zip(datasets, lengths, strict=True)}
        rows = tuple(DatasetBaselineAntigravity(order, next(family for family in _FAMILIES_ANTIGRAVITY if dataset.startswith(family + "_")), frozenset(), ConfusionAntigravity(0, 0, 40, length - 40)) for order, (dataset, length) in enumerate(zip(datasets, lengths, strict=True)))
        self.authority = _Exp448Antigravity(datasets, rows, ConfusionAntigravity(1930, 314, 1988, 191668))
        self.root, self.audit_path, self.labels = root, root.parent / "evaluation_failure_audit_antigravity.json", labels
        self.open_error, self.query_error, self.close_error, self.drift_at = open_error, query_error, close_error, drift_at
        self.open_labels_calls = 0
        self.capture_calls = 0
        self.session: _SessionAntigravity | None = None

    def capture_exp448_antigravity(self) -> _Exp448Antigravity:
        self.capture_calls += 1
        if self.drift_at is not None and self.capture_calls == self.drift_at:
            return _Exp448Antigravity(self.authority.target_dataset_names, self.authority.target_baseline_rows, ConfusionAntigravity(1930, 314, 1987, 191669))
        return self.authority

    def frozen_predictions_antigravity(self) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for sequence, dataset in enumerate(self.authority.target_dataset_names):
            family = self.authority.target_baseline_rows[sequence].family
            for method in dict(CANDIDATE_MATRIX_ANTIGRAVITY)[family]:
                rows.append({"sequence": len(rows), "candidate_name": f"{dataset}__{method}", "dataset": dataset, "family": family, "method": method, "indices": [0, 1], "target_length": len(self.labels[dataset])})
        return tuple(rows)

    def capture_task5_prediction_antigravity(self):
        """Synthetic fixture stands in for the source-free Task-5 descriptor capture."""
        import hashlib
        from experiments.exp449_residual_fn_method_qualification_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import FrozenPredictionAuthorityAntigravity, FrozenPredictionCaptureAntigravity
        rows = self.frozen_predictions_antigravity()
        payload = b"".join(canonical_json_bytes_antigravity(row) for row in rows)
        row_sha256 = tuple(hashlib.sha256(line).hexdigest() for line in payload.splitlines())
        inventory = tuple(f"candidates_antigravity/{index:03d}_antigravity.json" for index in range(96))
        authority = FrozenPredictionAuthorityAntigravity("a" * 40, 16, 96, tuple(row["candidate_name"] for row in rows), inventory, tuple((name, "0" * 64) for name in inventory), "0" * 64, 0)
        return FrozenPredictionCaptureAntigravity(authority, payload, row_sha256, hashlib.sha256("".join(row_sha256).encode("ascii")).hexdigest(), "1" * 64, "2" * 64, "3" * 64)

    def open_labels_antigravity(self, prediction_authority: object) -> _SessionAntigravity:
        self.open_labels_calls += 1
        if self.open_error is not None:
            raise self.open_error
        self.session = _SessionAntigravity(self.labels, query_error=self.query_error, close_error=self.close_error)
        return self.session


class EvaluationAntigravityTests(unittest.TestCase):
    def test_postrename_failure_audit_records_phase_and_forensic_outcome_antigravity(self) -> None:
        """Break caught: audit calls every retained-identity recovery failure precommit."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module
        import json
        with tempfile.TemporaryDirectory(prefix="exp449_audit_recovery_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            original, calls = module.inspect_replay_at_fd_antigravity, 0
            def fail_postrename(fd: int, **kwargs: object):
                nonlocal calls; calls += 1
                if calls == 2: raise OSError("postrename replay")
                return original(fd, **kwargs)
            with patch.object(module, "inspect_replay_at_fd_antigravity", side_effect=fail_postrename), self.assertRaisesRegex(OSError, "postrename replay"):
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            audit = json.loads(dependencies.audit_path.read_text())
            self.assertEqual(audit["phase"], "post_rename_replay")
            self.assertEqual(audit["recovery"]["outcome"], "moved")
            self.assertFalse(dependencies.root.exists())
    def test_task6_requires_a_descriptor_captured_task5_authority_antigravity(self) -> None:
        """Break caught: Task6 mints a reduced authority from caller-projected rows."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity

        with tempfile.TemporaryDirectory(prefix="exp449_task5_capture_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            dependencies.capture_task5_prediction_antigravity = dependencies.frozen_predictions_antigravity
            with self.assertRaisesRegex(ValueError, "Task5 prediction capture"):
                _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertEqual(dependencies.open_labels_calls, 0)

    def test_postrename_replay_failure_recovers_only_the_retained_publication_antigravity(self) -> None:
        """Break caught: a canonical post-rename semantic failure can be reported as success."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        with tempfile.TemporaryDirectory(prefix="exp449_postrename_recovery_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            original = module.inspect_replay_at_fd_antigravity
            calls = 0
            def fail_only_postrename(descriptor: int, **kwargs: object):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("postrename semantic failure")
                return original(descriptor, **kwargs)
            with patch.object(module, "inspect_replay_at_fd_antigravity", side_effect=fail_only_postrename), self.assertRaisesRegex(OSError, "postrename semantic failure"):
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertFalse(dependencies.root.exists())
            self.assertEqual(len(list(Path(temporary).glob(".publication_antigravity.forensic_*_antigravity"))), 1)

    def test_postmove_recovery_keeps_forensic_inode_and_moved_audit_when_validation_or_fsync_fails_antigravity(self) -> None:
        """Break caught: a post-move recovery diagnostic overwrites the successful quarantine outcome."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module
        from experiments.exp449_residual_fn_method_qualification_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        import json

        for recovery_fault in ("validation", "fsync"):
            with self.subTest(recovery_fault=recovery_fault), tempfile.TemporaryDirectory(prefix="exp449_postmove_truth_antigravity_") as temporary:
                dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
                original_replay = module.inspect_replay_at_fd_antigravity
                original_rename = module._rename_no_replace_at_antigravity
                original_assert = module._assert_directory_entry_identity_at_antigravity
                original_fsync = module._fsync_directory_antigravity
                replay_calls = 0
                moved: dict[str, object] = {}

                def fail_postrename_replay(descriptor: int, **kwargs: object):
                    nonlocal replay_calls
                    replay_calls += 1
                    if replay_calls == 2:
                        raise OSError("primary postrename truth")
                    return original_replay(descriptor, **kwargs)

                def record_forensic_move(parent_fd: int, source: str, destination: str) -> None:
                    original_rename(parent_fd, source, destination)
                    if source == dependencies.root.name and ".forensic_" in destination:
                        moved["basename"] = destination
                        moved["identity"] = (os.stat(destination, dir_fd=parent_fd, follow_symlinks=False).st_dev, os.stat(destination, dir_fd=parent_fd, follow_symlinks=False).st_ino)

                def fail_recovery_validation(parent_fd: int, name: str, expected: object) -> None:
                    if recovery_fault == "validation" and name == moved.get("basename"):
                        raise OSError("recovery validation truth")
                    return original_assert(parent_fd, name, expected)

                def fail_recovery_fsync(descriptor: int) -> None:
                    if recovery_fault == "fsync" and moved:
                        raise OSError("recovery fsync truth")
                    return original_fsync(descriptor)

                with patch.object(module, "inspect_replay_at_fd_antigravity", side_effect=fail_postrename_replay), patch.object(module, "_rename_no_replace_at_antigravity", side_effect=record_forensic_move), patch.object(module, "_assert_directory_entry_identity_at_antigravity", side_effect=fail_recovery_validation), patch.object(module, "_fsync_directory_antigravity", side_effect=fail_recovery_fsync):
                    with self.assertRaisesRegex(OSError, "primary postrename truth") as raised:
                        module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)

                self.assertFalse(dependencies.root.exists())
                forensic = Path(temporary) / str(moved["basename"])
                self.assertTrue(forensic.is_dir())
                self.assertEqual((forensic.stat().st_dev, forensic.stat().st_ino), moved["identity"])
                raw_audit = dependencies.audit_path.read_bytes()
                audit = json.loads(raw_audit)
                self.assertEqual(raw_audit, canonical_json_bytes_antigravity(audit))
                self.assertEqual(audit["recovery"]["outcome"], "moved")
                self.assertEqual(audit["recovery"]["forensic_basename"], moved["basename"])
                self.assertEqual(audit["error"], "OSError: primary postrename truth")
                self.assertEqual(str(raised.exception), "primary postrename truth")
                key = f"recovery_{recovery_fault}_error"
                self.assertEqual(audit["recovery"][key], f"OSError: recovery {recovery_fault} truth")
                self.assertNotEqual(audit["recovery"]["outcome"], "recovery_failed")
                self.assertIsNotNone(audit["recovery"]["forensic_basename"])

    def test_postrename_close_failure_is_a_note_not_a_primary_replacement_antigravity(self) -> None:
        """Break caught: retained-descriptor close masks an active recovery primary error."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        with tempfile.TemporaryDirectory(prefix="exp449_close_primary_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            original_replay, original_close = module.inspect_replay_at_fd_antigravity, module._close_descriptor_antigravity
            calls = 0
            def fail_only_postrename(descriptor: int, **kwargs: object):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("primary postrename replay")
                return original_replay(descriptor, **kwargs)
            close_calls = 0
            def close_then_fail(descriptor: int) -> None:
                nonlocal close_calls
                close_calls += 1
                original_close(descriptor)
                if close_calls == 13:
                    raise OSError("descriptor close")
            with patch.object(module, "inspect_replay_at_fd_antigravity", side_effect=fail_only_postrename), patch.object(module, "_close_descriptor_antigravity", side_effect=close_then_fail):
                with self.assertRaisesRegex(OSError, "primary postrename replay") as raised:
                    module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertIn("descriptor_close_error=OSError: descriptor close", "\n".join(raised.exception.__notes__))

    def test_qualification_edges_are_exact_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.evaluation_antigravity import qualifies_antigravity

        self.assertTrue(qualifies_antigravity(ConfusionAntigravity(1931, 314, 1987, 191668), target_fn_delta=-1))
        self.assertFalse(qualifies_antigravity(ConfusionAntigravity(1931, 315, 1987, 191667), target_fn_delta=-1))
        self.assertFalse(qualifies_antigravity(ConfusionAntigravity(1930, 314, 1988, 191668), target_fn_delta=0))
        for bad in (True, 0.0, "-1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                qualifies_antigravity(ConfusionAntigravity(1931, 314, 1987, 191668), target_fn_delta=bad)

    def test_one_session_evaluates_fixed_inventory_and_publishes_exact_twelve_leaves_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity, _LEAVES_ANTIGRAVITY

        with tempfile.TemporaryDirectory(prefix="exp449_evaluation_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            result = _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertEqual((result.label_session_count, result.candidate_count, result.label_query_count, result.label_row_count), (1, 16, 48, 72300))
            self.assertEqual(dependencies.open_labels_calls, 1)
            self.assertEqual(tuple(path.name for path in sorted(dependencies.root.iterdir())), tuple(sorted(_LEAVES_ANTIGRAVITY)))
            crop = result.candidate_metrics[0]
            self.assertEqual((crop.family, crop.method, crop.target_confusion, crop.global_confusion, crop.target_fn_delta, crop.qualified), ("Crop", "minirocket_local_gap", ConfusionAntigravity(48, 0, 912, 35040), ConfusionAntigravity(1978, 314, 1940, 191668), -48, True))

    def test_qualification_truth_drives_research_routing_only_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        import json

        with tempfile.TemporaryDirectory(prefix="exp449_routing_truth_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            result = _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertGreater(result.qualified_count, 0)
            self.assertTrue(result.routing_research_eligible)
            decision = json.loads((dependencies.root / "decision_antigravity.json").read_text())
            dashboard = json.loads((dependencies.root / "dashboard_snapshot_antigravity.json").read_text())
            self.assertTrue(decision["routing_research_eligible"])
            self.assertTrue(dashboard["routing_research_eligible"])
            self.assertFalse(decision["automatic_routing"])
            self.assertFalse(decision["automatic_promotion"])

    def test_bootstrap_draw_is_frozen_from_independently_derived_literal_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.bootstrap_antigravity import BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY, bootstrap_draws_antigravity

        draws = bootstrap_draws_antigravity()
        self.assertEqual((draws.dtype.name, draws.shape, draws.flags.writeable, BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY), ("int64", (20000, 48), False, "264c780d7e5949994f1de3053158ea20a7cc3d0ae858f29770d9849b18ad4d6a"))

    def test_open_query_close_and_source_drift_fail_closed_with_audit_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity

        for failure in ("open", "query", "close", "after_close", "after_compute", "precommit"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="exp449_failure_antigravity_") as temporary:
                kwargs: dict[str, object] = {"open_error": OSError("open") if failure == "open" else None, "query_error": OSError("query") if failure == "query" else None, "close_error": OSError("close") if failure == "close" else None, "drift_at": {"after_close": 2, "after_compute": 3, "precommit": 4}.get(failure)}
                dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity", **kwargs)
                with self.assertRaises((OSError, ValueError)):
                    _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                self.assertFalse(dependencies.root.exists())
                self.assertTrue(dependencies.audit_path.is_file())

    def test_existing_or_symlink_destination_never_opens_label_session_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity

        with tempfile.TemporaryDirectory(prefix="exp449_existing_antigravity_") as temporary:
            root = Path(temporary) / "publication_antigravity"; root.mkdir()
            dependencies = _DependenciesAntigravity(root)
            with self.assertRaises(ValueError): _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertEqual(dependencies.open_labels_calls, 0)
        with tempfile.TemporaryDirectory(prefix="exp449_symlink_destination_antigravity_") as temporary:
            root = Path(temporary) / "publication_antigravity"; root.symlink_to("missing_antigravity")
            dependencies = _DependenciesAntigravity(root)
            with self.assertRaises(ValueError): _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertEqual(dependencies.open_labels_calls, 0)

    def test_no_replace_commit_refuses_directory_that_appears_after_private_replay_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        with tempfile.TemporaryDirectory(prefix="exp449_directory_replace_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            original = module._rename_no_replace_at_antigravity
            def appeared(parent_fd: int, source: str, destination: str) -> None:
                if destination != dependencies.root.name:
                    return original(parent_fd, source, destination)
                dependencies.root.mkdir(); (dependencies.root / "attacker_antigravity").write_text("not publication")
                raise FileExistsError("destination appeared")
            with patch.object(module, "_rename_no_replace_at_antigravity", side_effect=appeared), self.assertRaisesRegex(FileExistsError, "destination appeared"):
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertEqual({path.name for path in dependencies.root.iterdir()}, {"attacker_antigravity"})
            self.assertTrue(dependencies.audit_path.is_file())

    def test_private_leaf_write_rejects_a_zero_byte_os_write_antigravity(self) -> None:
        """A zero-byte write must fail rather than spin while publication is private."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        with tempfile.TemporaryDirectory(prefix="exp449_zero_write_antigravity_") as temporary:
            directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with patch.object(module.os, "write", return_value=0), self.assertRaisesRegex(OSError, "made no progress"):
                    module._write_leaf_at_antigravity(directory_fd, "context_antigravity.json", b"payload")
            finally:
                os.close(directory_fd)

    def test_parent_descriptor_closes_on_staging_and_final_rename_failures_antigravity(self) -> None:
        """Every pre-commit failure releases the retained parent directory descriptor."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        for failure in ("staging_mkdir", "final_rename"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="exp449_parent_close_antigravity_") as temporary:
                dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
                opened: list[int] = []
                original_open, original_mkdir, original_rename = module.os.open, module.os.mkdir, module._rename_no_replace_at_antigravity
                def record_open(*args: object, **kwargs: object) -> int:
                    descriptor = original_open(*args, **kwargs)
                    if len(args) > 1 and type(args[1]) is int and args[1] & os.O_DIRECTORY:
                        opened.append(descriptor)
                    return descriptor
                def fail_staging_mkdir(name: str, *args: object, **kwargs: object) -> None:
                    if failure == "staging_mkdir" and name.startswith(f".{dependencies.root.name}.staging_"):
                        raise OSError("staging mkdir failure")
                    original_mkdir(name, *args, **kwargs)
                def fail_final_rename(parent_fd: int, source: str, destination: str) -> None:
                    if failure == "final_rename" and destination == dependencies.root.name:
                        raise OSError("final rename failure")
                    original_rename(parent_fd, source, destination)
                with patch.object(module.os, "open", side_effect=record_open), patch.object(module.os, "mkdir", side_effect=fail_staging_mkdir), patch.object(module, "_rename_no_replace_at_antigravity", side_effect=fail_final_rename), self.assertRaisesRegex(OSError, "failure"):
                    module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                self.assertGreaterEqual(len(opened), 1)
                with self.assertRaises(OSError):
                    os.fstat(opened[0])

    def test_publish_releases_retained_directory_descriptors_on_success_and_precommit_failure_antigravity(self) -> None:
        """Both retained directory FDs must be closed; a successful commit cannot report close failure."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="exp449_fd_release_antigravity_") as temporary:
                fd_count_before = len(list(Path('/dev/fd').iterdir()))
                dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
                closed: list[int] = []
                directory_fds: list[int] = []
                original_close, original_open = module._close_descriptor_antigravity, module.os.open
                def record_open(*args: object, **kwargs: object) -> int:
                    descriptor = original_open(*args, **kwargs)
                    if type(args[1]) is int and args[1] & os.O_DIRECTORY:
                        directory_fds.append(descriptor)
                    return descriptor
                def record_close(descriptor: int) -> None:
                    closed.append(descriptor)
                    original_close(descriptor)
                replay_patch = patch.object(module, "inspect_replay_at_fd_antigravity", side_effect=OSError("sealed replay failure")) if failure else None
                with patch.object(module.os, "open", side_effect=record_open), patch.object(module, "_close_descriptor_antigravity", side_effect=record_close):
                    if replay_patch is None:
                        module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                    else:
                        with replay_patch, self.assertRaisesRegex(OSError, "sealed replay failure"):
                            module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                retained_directory_fds = directory_fds[:2]
                self.assertEqual(len(retained_directory_fds), 2)
                self.assertTrue(set(retained_directory_fds).issubset(closed))
                for descriptor in retained_directory_fds:
                    with self.assertRaises(OSError):
                        os.fstat(descriptor)
                self.assertEqual(len(list(Path('/dev/fd').iterdir())), fd_count_before)

    def test_successful_canonical_publish_swallows_staging_and_parent_close_errors_antigravity(self) -> None:
        """Break caught: a close error after all canonical verification converts a successful publication into a failure audit."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        with tempfile.TemporaryDirectory(prefix="exp449_success_close_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            original_open, original_close = module.os.open, module._close_descriptor_antigravity
            directory_fds: list[int] = []; closed: list[int] = []
            def record_open(*args: object, **kwargs: object) -> int:
                descriptor = original_open(*args, **kwargs)
                if type(args[1]) is int and args[1] & os.O_DIRECTORY: directory_fds.append(descriptor)
                return descriptor
            def close_then_fail(descriptor: int) -> None:
                closed.append(descriptor); original_close(descriptor)
                if descriptor in directory_fds[:2]: raise OSError(f"successful close fd={descriptor}")
            with patch.object(module.os, "open", side_effect=record_open), patch.object(module, "_close_descriptor_antigravity", side_effect=close_then_fail):
                result = module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertEqual(result.label_session_count, 1)
            self.assertTrue(dependencies.root.is_dir())
            self.assertFalse(dependencies.audit_path.exists())
            self.assertTrue(set(directory_fds[:2]).issubset(closed))

    def test_precommit_primary_retains_both_retained_directory_close_notes_antigravity(self) -> None:
        """Break caught: best-effort successful-close handling hides close diagnostics from an active precommit primary."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        with tempfile.TemporaryDirectory(prefix="exp449_primary_close_notes_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            original_open, original_close = module.os.open, module._close_descriptor_antigravity
            directory_fds: list[int] = []
            def record_open(*args: object, **kwargs: object) -> int:
                descriptor = original_open(*args, **kwargs)
                if type(args[1]) is int and args[1] & os.O_DIRECTORY: directory_fds.append(descriptor)
                return descriptor
            def close_then_fail(descriptor: int) -> None:
                original_close(descriptor)
                if descriptor in directory_fds[:2]: raise OSError(f"primary close fd={descriptor}")
            with patch.object(module.os, "open", side_effect=record_open), patch.object(module, "_write_leaf_at_antigravity", side_effect=OSError("precommit primary truth")), patch.object(module, "_close_descriptor_antigravity", side_effect=close_then_fail):
                with self.assertRaisesRegex(OSError, "precommit primary truth") as raised:
                    module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            notes = "\n".join(raised.exception.__notes__)
            self.assertIn(f"descriptor_close_error=OSError: primary close fd={directory_fds[1]}", notes)
            self.assertIn(f"parent_descriptor_close_error=OSError: primary close fd={directory_fds[0]}", notes)

    def test_partial_staging_and_audit_write_failure_preserve_primary_truth_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        with tempfile.TemporaryDirectory(prefix="exp449_partial_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            original, calls = module._write_leaf_at_antigravity, 0
            def fail_third(directory_fd: int, name: str, payload: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 3: raise OSError("partial staging write")
                original(directory_fd, name, payload)
            with patch.object(module, "_write_leaf_at_antigravity", side_effect=fail_third), self.assertRaisesRegex(OSError, "partial staging write"):
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertFalse(dependencies.root.exists())
            self.assertTrue(dependencies.audit_path.is_file())
        with tempfile.TemporaryDirectory(prefix="exp449_audit_failure_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity", query_error=OSError("query primary"))
            with patch.object(module, "write_create_only_antigravity", side_effect=OSError("audit write")), self.assertRaisesRegex(OSError, "query primary") as raised:
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertIn("failure_audit_error=OSError: audit write", "\n".join(raised.exception.__notes__))

    def test_duplicate_or_missing_frozen_candidate_rows_fail_before_labels_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity

        class BadDependenciesAntigravity(_DependenciesAntigravity):
            def frozen_predictions_antigravity(self) -> tuple[dict[str, object], ...]:
                rows = list(super().frozen_predictions_antigravity()); rows.pop()
                return tuple(rows)
        with tempfile.TemporaryDirectory(prefix="exp449_inventory_antigravity_") as temporary:
            dependencies = BadDependenciesAntigravity(Path(temporary) / "publication_antigravity")
            with self.assertRaises(ValueError): _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertEqual(dependencies.open_labels_calls, 0)


if __name__ == "__main__":
    unittest.main()
