"""Synthetic-only Task 6 evaluation boundaries."""
from __future__ import annotations

from dataclasses import dataclass
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

    def open_labels_antigravity(self, prediction_authority: object) -> _SessionAntigravity:
        self.open_labels_calls += 1
        if self.open_error is not None:
            raise self.open_error
        self.session = _SessionAntigravity(self.labels, query_error=self.query_error, close_error=self.close_error)
        return self.session


class EvaluationAntigravityTests(unittest.TestCase):
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
            def appeared(staging: Path, destination: Path) -> None:
                destination.mkdir(); (destination / "attacker_antigravity").write_text("not publication")
                raise FileExistsError("destination appeared")
            with patch.object(module, "_rename_no_replace_antigravity", side_effect=appeared), self.assertRaisesRegex(FileExistsError, "destination appeared"):
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertEqual({path.name for path in dependencies.root.iterdir()}, {"attacker_antigravity"})
            self.assertTrue(dependencies.audit_path.is_file())

    def test_post_rename_fsync_failure_never_leaves_valid_publication_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        with tempfile.TemporaryDirectory(prefix="exp449_postrename_fsync_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            original, calls = module._fsync_directory_antigravity, 0
            def fail_after_rename(path: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 3: raise OSError("post rename fsync")
                original(path)
            with patch.object(module, "_fsync_directory_antigravity", side_effect=fail_after_rename), self.assertRaisesRegex(OSError, "post rename fsync"):
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertFalse(dependencies.root.exists())
            self.assertTrue(dependencies.audit_path.is_file())

    def test_postpublication_replay_failure_quarantines_canonical_root_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        with tempfile.TemporaryDirectory(prefix="exp449_postpublication_replay_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            original, calls = module.inspect_replay_at_antigravity, 0
            def fail_after_staging(root: Path):
                nonlocal calls
                calls += 1
                if calls > 1: raise OSError("post publication replay")
                return original(root)
            with patch.object(module, "inspect_replay_at_antigravity", side_effect=fail_after_staging), self.assertRaisesRegex(OSError, "post publication replay"):
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertEqual(calls, 2); self.assertFalse(dependencies.root.exists())
            self.assertTrue((Path(temporary) / ".publication_antigravity.forensic_antigravity").is_dir())
            self.assertTrue(dependencies.audit_path.is_file())

    def test_quarantine_failure_retains_primary_error_and_never_keeps_expected_root_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module
        with tempfile.TemporaryDirectory(prefix="exp449_quarantine_failure_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            original_replay, original_rename = module.inspect_replay_at_antigravity, module._rename_no_replace_antigravity
            calls = 0
            def replay_then_fail(root: Path):
                nonlocal calls
                calls += 1
                if calls == 2: raise OSError("post publication primary")
                return original_replay(root)
            def fail_quarantine(source: Path, destination: Path) -> None:
                if source == dependencies.root: raise OSError("quarantine primitive")
                original_rename(source, destination)
            with patch.object(module, "inspect_replay_at_antigravity", side_effect=replay_then_fail), patch.object(module, "_rename_no_replace_antigravity", side_effect=fail_quarantine), self.assertRaisesRegex(OSError, "post publication primary") as raised:
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertFalse(dependencies.root.exists())
            self.assertTrue((Path(temporary) / ".publication_antigravity.forensic_antigravity").is_dir())
            self.assertTrue(dependencies.audit_path.is_file())

    def test_partial_staging_and_audit_write_failure_preserve_primary_truth_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as module

        with tempfile.TemporaryDirectory(prefix="exp449_partial_antigravity_") as temporary:
            dependencies = _DependenciesAntigravity(Path(temporary) / "publication_antigravity")
            original, calls = module.write_create_only_antigravity, 0
            def fail_third(path: Path, payload: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 3: raise OSError("partial staging write")
                original(path, payload)
            with patch.object(module, "write_create_only_antigravity", side_effect=fail_third), self.assertRaisesRegex(OSError, "partial staging write"):
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
