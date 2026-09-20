"""Red-first synthetic evaluator fixtures for the Exp454 authority-boundary fork."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.config_antigravity import (
    CANDIDATE_MATRIX_ANTIGRAVITY,
    EVALUATION_LEAVES_ANTIGRAVITY,
    FIXED_EXECUTABLE_ANTIGRAVITY,
    FIXED_PREFIX_ANTIGRAVITY,
    RUNTIME_VERSIONS_ANTIGRAVITY,
)


_LEAVES_ANTIGRAVITY = EVALUATION_LEAVES_ANTIGRAVITY
_ORDERED_INDICES_ANTIGRAVITY = [981, 983, 984, 986, 988, 989, 992, 993, 994, 995, 996, 998, 999]
_BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY = "264c780d7e5949994f1de3053158ea20a7cc3d0ae858f29770d9849b18ad4d6a"
_IDENTITY_KEYS_ANTIGRAVITY = frozenset({
    "schema", "experiment", "reviewed_commit", "runtime_binding", "predecessor", "prediction_authority",
})
_DERIVED_HASH_KEYS_ANTIGRAVITY = frozenset({
    "exp449_reviewed_commit",
})


class ExpectedLabelBoundary(Exception):
    """Injected label-opener sentinel; it must fire before any TEST-label query."""


def _block_test_labels_antigravity(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("TEST labels were queried")


def raise_expected_label_boundary_sentinel(_authority: object) -> object:
    raise ExpectedLabelBoundary("reached")


class EvaluationAntigravityTests(unittest.TestCase):
    def _dependencies_antigravity(self, root: Path, ordered_indices: list[int] | None = None, **kwargs: object):
        from experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_evaluation_antigravity import _DependenciesAntigravity
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.replay_antigravity import FrozenPredictionCaptureAntigravity
        legacy = _DependenciesAntigravity(root, **kwargs)
        captured = legacy.capture_task5_prediction_antigravity()
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        rows = [json.loads(line) for line in captured.canonical_rows.splitlines()]
        if ordered_indices is not None:
            for row in rows:
                row["indices"] = list(ordered_indices)
        canonical_rows = b"".join(canonical_json_bytes_antigravity(row) for row in rows)
        canonical_row_sha256 = tuple(hashlib.sha256(line).hexdigest() for line in canonical_rows.splitlines())
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.contracts_antigravity import ConfusionAntigravity, DatasetBaselineAntigravity

        def converted_target_antigravity():
            legacy_target = legacy.capture_exp448_antigravity()
            baselines = tuple(
                DatasetBaselineAntigravity(
                    row.dataset_order,
                    row.family,
                    row.stable_candidates,
                    ConfusionAntigravity(row.confusion.tp, row.confusion.fp, row.confusion.fn, row.confusion.tn),
                )
                for row in legacy_target.target_baseline_rows
            )
            return SimpleNamespace(
                target_dataset_names=legacy_target.target_dataset_names,
                target_baseline_rows=baselines,
                baseline_confusion=ConfusionAntigravity(
                    legacy_target.baseline_confusion.tp,
                    legacy_target.baseline_confusion.fp,
                    legacy_target.baseline_confusion.fn,
                    legacy_target.baseline_confusion.tn,
                ),
            )

        capture = FrozenPredictionCaptureAntigravity(
            captured.authority,
            canonical_rows,
            canonical_row_sha256,
            hashlib.sha256("".join(canonical_row_sha256).encode("ascii")).hexdigest(),
        )
        return SimpleNamespace(
            root=root,
            audit_path=root.parent / "failure_audit_antigravity.json",
            capture_exp448_antigravity=converted_target_antigravity,
            capture_task5_prediction_antigravity=lambda: capture,
            open_labels_antigravity=legacy.open_labels_antigravity,
            legacy_antigravity=legacy,
        )

    def _lifecycle_paths_antigravity(self, lifecycle, root: Path):
        python_version, numpy_version, aeon_version, sklearn_version = RUNTIME_VERSIONS_ANTIGRAVITY
        return lifecycle._LifecyclePathsAntigravity(
            receipt_directory=root.parent / "receipts_antigravity",
            audit_path=root.parent / "audit_antigravity.json",
            evaluation_root=root,
            executable=FIXED_EXECUTABLE_ANTIGRAVITY,
            prefix=FIXED_PREFIX_ANTIGRAVITY,
            python_version=python_version,
            numpy_version=numpy_version,
            aeon_version=aeon_version,
            sklearn_version=sklearn_version,
        )

    def _normalize_scientific_antigravity(self, value: object) -> object:
        if type(value) is dict:
            normalized: dict[str, object] = {}
            for key, item in value.items():
                if key in _IDENTITY_KEYS_ANTIGRAVITY or key in _DERIVED_HASH_KEYS_ANTIGRAVITY:
                    continue
                if key.endswith("_sha256") and key not in {
                    "draw_sha256",
                    "frozen_prediction_sha256",
                    "exp448_target_identity_sha256",
                    "target_authority_sha256",
                    "exp449_prediction_aggregate_sha256",
                    "predecessor_attestation_sha256",
                }:
                    continue
                normalized[key] = self._normalize_scientific_antigravity(item)
            return normalized
        if type(value) is list:
            return [self._normalize_scientific_antigravity(item) for item in value]
        return value

    def _leaf_scientific_bytes_antigravity(self, name: str, payload: bytes) -> bytes:
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.artifacts_antigravity import (
            canonical_json_bytes_antigravity,
            canonical_jsonl_bytes_antigravity,
        )

        if name == "artifact_sha256_antigravity.json":
            return b"DERIVED_HASH_MAP"
        if name == "frozen_predictions_antigravity.jsonl":
            return payload
        if name.endswith(".jsonl"):
            rows = tuple(json.loads(line) for line in payload.splitlines())
            return canonical_jsonl_bytes_antigravity(tuple(self._normalize_scientific_antigravity(row) for row in rows))
        parsed = json.loads(payload)
        return canonical_json_bytes_antigravity(self._normalize_scientific_antigravity(parsed))

    def _real_prediction_capture_antigravity(self, authority: object, capture_cls: type):
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.artifacts_antigravity import (
            canonical_json_bytes_antigravity,
        )

        rows: list[dict[str, object]] = []
        methods = dict(CANDIDATE_MATRIX_ANTIGRAVITY)
        for dataset, baseline in zip(authority.target_dataset_names, authority.target_baseline_rows, strict=True):
            for method in methods[baseline.family]:
                rows.append(
                    {
                        "sequence": len(rows),
                        "candidate_name": f"{dataset}__{method}",
                        "dataset": dataset,
                        "family": baseline.family,
                        "method": method,
                        "indices": [0, 1],
                        "target_length": 8,
                    }
                )
        canonical_rows = b"".join(canonical_json_bytes_antigravity(row) for row in rows)
        row_sha256 = tuple(hashlib.sha256(line).hexdigest() for line in canonical_rows.splitlines())
        return capture_cls(
            SimpleNamespace(reviewed_commit="a" * 40),
            canonical_rows,
            row_sha256,
            hashlib.sha256("".join(row_sha256).encode("ascii")).hexdigest(),
        )

    def test_exp453_evaluator_rejects_real_normalized_authority_before_label_opener_antigravity(self) -> None:
        """Break caught: Exp453 type identity is treated as already repaired."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.evaluation_antigravity import (
            _evaluate_with_dependencies_antigravity as evaluate_exp453,
        )
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.replay_antigravity import (
            FrozenPredictionCaptureAntigravity as Exp453CaptureAntigravity,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.predecessor_antigravity import (
            capture_normalized_target_authority_antigravity,
        )

        opened = {"called": False}

        def unexpected_opener(_authority: object) -> object:
            opened["called"] = True
            raise AssertionError("Exp453 opened labels after the type-boundary failure")

        with tempfile.TemporaryDirectory(prefix="exp454_exp453_reject_antigravity_") as temporary, patch.object(
            sqlite3, "connect", _block_test_labels_antigravity
        ):
            authority = capture_normalized_target_authority_antigravity()
            capture = self._real_prediction_capture_antigravity(authority, Exp453CaptureAntigravity)
            dependencies = SimpleNamespace(
                root=Path(temporary) / "publication_antigravity",
                audit_path=Path(temporary) / "failure_audit_antigravity.json",
                capture_exp448_antigravity=lambda: authority,
                capture_task5_prediction_antigravity=lambda: capture,
                open_labels_antigravity=unexpected_opener,
            )
            with self.assertRaisesRegex(ValueError, "Exp448 target identity differs"):
                evaluate_exp453("a" * 40, dependencies)
        self.assertFalse(opened["called"])

    def test_real_exp449_authority_reaches_label_opener_antigravity(self) -> None:
        """Break caught: production evaluator never consumes the Task 2 normalized capture."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.evaluation_antigravity import (
            _evaluate_with_dependencies_antigravity,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.predecessor_antigravity import (
            capture_normalized_target_authority_antigravity,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.replay_antigravity import (
            FrozenPredictionCaptureAntigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp454_label_sentinel_antigravity_") as temporary, patch.object(
            sqlite3, "connect", _block_test_labels_antigravity
        ):
            authority = capture_normalized_target_authority_antigravity()
            self.assertEqual(len(authority.target_dataset_names), 48)
            self.assertEqual(len(authority.target_baseline_rows), 48)
            capture = self._real_prediction_capture_antigravity(authority, FrozenPredictionCaptureAntigravity)
            dependencies = SimpleNamespace(
                root=Path(temporary) / "publication_antigravity",
                audit_path=Path(temporary) / "failure_audit_antigravity.json",
                capture_exp448_antigravity=capture_normalized_target_authority_antigravity,
                capture_task5_prediction_antigravity=lambda: capture,
                open_labels_antigravity=raise_expected_label_boundary_sentinel,
            )
            with self.assertRaisesRegex(ExpectedLabelBoundary, "reached"):
                _evaluate_with_dependencies_antigravity("a" * 40, dependencies)

    def test_production_evaluator_uses_task2_normalized_capture_only_antigravity(self) -> None:
        """Break caught: evaluator reintroduces a second converter or raw Exp449 capture."""
        package = Path(__file__).resolve().parents[1]
        evaluation = (package / "evaluation_antigravity.py").read_text(encoding="utf-8")
        replay = (package / "replay_antigravity.py").read_text(encoding="utf-8")
        self.assertIn("capture_normalized_target_authority_antigravity", evaluation)
        self.assertNotIn("def normalize_exp449_target_authority_antigravity", evaluation)
        self.assertNotIn("def normalize_exp449_target_authority_antigravity", replay)
        self.assertNotIn(
            "from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import capture_exp448_authority_antigravity",
            evaluation,
        )
        self.assertIn("capture_predecessors_antigravity", evaluation)
        self.assertIn("open_test_labels_once_antigravity", evaluation)

    def test_minimal_order_fixture_emits_ascending_evidence_indices_antigravity(self) -> None:
        """Break caught: replacing sorted with list preserves unordered frozenset traversal."""
        self.assertEqual(sorted(frozenset(_ORDERED_INDICES_ANTIGRAVITY)), _ORDERED_INDICES_ANTIGRAVITY)

    def test_full_48_dataset_96_row_fixture_matches_science_and_leaf_inventory_antigravity(self) -> None:
        """Break caught: any unordered evidence row or retuned metric makes strict replay reject its frozen peer."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.replay_antigravity import inspect_replay_at_antigravity

        with tempfile.TemporaryDirectory(prefix="exp454_evaluation_antigravity_") as temporary:
            root = Path(temporary) / "publication_antigravity"
            dependencies = self._dependencies_antigravity(root, _ORDERED_INDICES_ANTIGRAVITY)
            result = _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            published = tuple(path.name for path in sorted(root.iterdir(), key=lambda item: _LEAVES_ANTIGRAVITY.index(item.name)))
            self.assertEqual(published, _LEAVES_ANTIGRAVITY)
            self.assertEqual(len(published), 12)
            frozen = [json.loads(line)["indices"] for line in (root / "frozen_predictions_antigravity.jsonl").read_text().splitlines()]
            evidence = json.loads((root / "evaluation_antigravity.json").read_text())["dataset_evidence"]
            actual = [entry["candidates"][method]["indices"] for entry in evidence for method in entry["candidates"]]
            self.assertEqual(len(evidence), 48)
            self.assertEqual(len(actual), 96)
            self.assertEqual(actual, frozen)
            self.assertTrue(all(indices == sorted(indices) for indices in actual))
            evaluation = json.loads((root / "evaluation_antigravity.json").read_text())
            self.assertEqual(evaluation["schema"], "exp454")
            self.assertEqual(evaluation["candidate_count"], 16)
            metrics = [json.loads(line) for line in (root / "candidate_metrics_antigravity.jsonl").read_text().splitlines()]
            self.assertEqual(len(metrics), 16)
            for row in metrics:
                confusion = row["global_confusion"]
                self.assertEqual(set(confusion), {"tp", "fp", "fn", "tn"})
                self.assertEqual(set(row["target_confusion"]), {"tp", "fp", "fn", "tn"})
                self.assertIsInstance(row["target_fn_delta"], int)
                self.assertIsInstance(row["global_f1"], float)
                self.assertIn("fn", row["target_confusion"])
                self.assertIn("fp", row["target_confusion"])
            bootstrap = json.loads((root / "bootstrap_antigravity.json").read_text())
            self.assertEqual(bootstrap["draw_sha256"], _BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY)
            decision = json.loads((root / "decision_antigravity.json").read_text())
            self.assertEqual(decision["schema"], "exp454.decision")
            self.assertEqual(decision["gates"], {"global_fn_lt": 1988, "global_fp_lte": 314, "global_f1_nonregression": 0.6264199935, "target_fn_delta_lt": 0})
            compatibility = json.loads((root / "compatibility_antigravity.json").read_text())
            self.assertEqual(compatibility["schema"], "exp454.task4")
            self.assertIn("runtime_binding", compatibility)
            self.assertEqual(inspect_replay_at_antigravity(root, expected_prediction_authority=dependencies.capture_task5_prediction_antigravity(), expected_reviewed_commit="a" * 40), result)

    def test_runtime_differential_normalized_scientific_bytes_match_exp453_antigravity(self) -> None:
        """Break caught: Exp454 retunes scientific leaf bytes beyond identity, schema, path, commit, runtime binding, or derived hashes."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.evaluation_antigravity import (
            _evaluate_with_dependencies_antigravity as evaluate_exp453,
        )
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.tests.test_evaluation_antigravity import (
            EvaluationAntigravityTests as Exp453EvaluationAntigravityTests,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.evaluation_antigravity import (
            _evaluate_with_dependencies_antigravity as evaluate_exp454,
        )

        with tempfile.TemporaryDirectory(prefix="exp454_runtime_differential_antigravity_") as temporary:
            predecessor = Path(temporary) / "exp453_publication_antigravity"
            successor = Path(temporary) / "exp454_publication_antigravity"
            exp453_dependencies = Exp453EvaluationAntigravityTests()._dependencies_antigravity(predecessor, _ORDERED_INDICES_ANTIGRAVITY)
            exp454_dependencies = self._dependencies_antigravity(successor, _ORDERED_INDICES_ANTIGRAVITY)
            self.assertEqual(
                exp453_dependencies.capture_task5_prediction_antigravity().canonical_rows,
                exp454_dependencies.capture_task5_prediction_antigravity().canonical_rows,
            )
            evaluate_exp453("a" * 40, exp453_dependencies)
            evaluate_exp454("a" * 40, exp454_dependencies)
            self.assertEqual(tuple(path.name for path in predecessor.iterdir()), tuple(path.name for path in successor.iterdir()))
            self.assertEqual(tuple(sorted(successor.iterdir(), key=lambda item: _LEAVES_ANTIGRAVITY.index(item.name))[i].name for i in range(12)), _LEAVES_ANTIGRAVITY)
            for name in _LEAVES_ANTIGRAVITY:
                left = self._leaf_scientific_bytes_antigravity(name, (predecessor / name).read_bytes())
                right = self._leaf_scientific_bytes_antigravity(name, (successor / name).read_bytes())
                self.assertEqual(left, right, msg=name)

    def test_distinct_exp449_and_exp454_commits_publish_and_replay_antigravity(self) -> None:
        """Break caught: a predecessor commit is treated as the Exp454 lifecycle commit."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import evaluation_antigravity as evaluation
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import lifecycle_antigravity as lifecycle
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import replay_antigravity as replay
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity

        exp449_commit, exp454_commit = "a" * 40, "b" * 40
        with tempfile.TemporaryDirectory(prefix="exp454_commit_domains_antigravity_") as temporary:
            root = Path(temporary) / "publication_antigravity"
            dependencies = self._dependencies_antigravity(root)
            capture = dependencies.capture_task5_prediction_antigravity()
            self.assertEqual(capture.authority.reviewed_commit, exp449_commit)
            paths = self._lifecycle_paths_antigravity(lifecycle, root)
            lifecycle._run_phase_with_paths_antigravity("readiness", exp454_commit, lambda: None, paths)
            result = evaluation._evaluate_with_lifecycle_dependencies_antigravity(
                exp454_commit,
                lambda: capture,
                lambda pinned: evaluation._evaluate_with_dependencies_antigravity(exp454_commit, dependencies),
                evaluation_root=root,
                receipt_dir=paths.receipt_directory,
                audit_path=paths.audit_path,
            )
            self.assertEqual(len(tuple(root.iterdir())), 12)
            context = json.loads((root / "context_antigravity.json").read_text())
            manifest = json.loads((root / "prediction_manifest_antigravity.json").read_text())
            evaluation_payload = json.loads((root / "evaluation_antigravity.json").read_text())
            self.assertEqual((context["reviewed_commit"], evaluation_payload["reviewed_commit"], manifest["exp449_reviewed_commit"]), (exp454_commit, exp454_commit, exp449_commit))
            replayed = replay._inspect_with_dependencies_antigravity(
                lambda: capture,
                lambda publication, prediction, lifecycle_commit: replay.inspect_replay_at_antigravity(publication, expected_prediction_authority=prediction, expected_reviewed_commit=lifecycle_commit),
                evaluation_root=root,
                receipt_dir=paths.receipt_directory,
                audit_path=paths.audit_path,
            )
            self.assertEqual(replayed, result)

            def rehash(name: str) -> None:
                hashes = json.loads((root / "artifact_sha256_antigravity.json").read_text())
                hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
                (root / "artifact_sha256_antigravity.json").write_bytes(canonical_json_bytes_antigravity(hashes))

            for name, key, wrong in (("context_antigravity.json", "reviewed_commit", exp449_commit), ("evaluation_antigravity.json", "reviewed_commit", exp449_commit), ("prediction_manifest_antigravity.json", "exp449_reviewed_commit", exp454_commit)):
                with self.subTest(name=name):
                    original = (root / name).read_bytes()
                    payload = json.loads(original)
                    payload[key] = wrong
                    (root / name).write_bytes(canonical_json_bytes_antigravity(payload))
                    rehash(name)
                    with self.assertRaises(ValueError):
                        replay.inspect_replay_at_antigravity(root, expected_prediction_authority=capture, expected_reviewed_commit=exp454_commit)
                    (root / name).write_bytes(original)
                    rehash(name)

    def test_publication_destination_and_label_failure_matrix_fails_closed_antigravity(self) -> None:
        """Break caught: unsafe publication or one label boundary failure publishes a valid root."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity

        cases = {
            "existing": {}, "partial": {}, "symlink": {},
            "setup": {"open_error": OSError("setup")}, "query": {"query_error": OSError("query")},
            "close": {"close_error": OSError("close")}, "post_close": {"drift_at": 2},
            "post_compute": {"drift_at": 3}, "precommit": {"drift_at": 4},
        }
        for case, kwargs in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="exp454_failure_antigravity_") as temporary:
                root = Path(temporary) / "publication_antigravity"
                if case == "existing":
                    root.mkdir()
                elif case == "partial":
                    root.mkdir()
                    (root / "partial_antigravity.json").write_text("{}")
                elif case == "symlink":
                    root.symlink_to("missing_antigravity")
                dependencies = self._dependencies_antigravity(root, **kwargs)
                with self.assertRaises((OSError, ValueError)):
                    _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                self.assertFalse(root.is_dir() and not root.is_symlink() and len(tuple(root.iterdir())) == 12)
                self.assertTrue(dependencies.audit_path.is_file())
                self.assertEqual(json.loads(dependencies.audit_path.read_text())["schema"], "exp454.task4.failure")
                if case in {"existing", "partial", "symlink"}:
                    self.assertEqual(dependencies.legacy_antigravity.open_labels_calls, 0)

    def test_leaf_write_fsync_and_namespace_replacement_preserve_failure_truth_antigravity(self) -> None:
        """Break caught: a publish I/O or destination-replacement fault becomes a valid evaluation."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import evaluation_antigravity as module

        for case in ("write", "fsync", "replacement"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="exp454_publish_failure_antigravity_") as temporary:
                root = Path(temporary) / "publication_antigravity"
                dependencies = self._dependencies_antigravity(root)
                if case == "write":
                    with patch.object(module, "_write_leaf_at_antigravity", side_effect=OSError("write")), self.assertRaisesRegex(OSError, "write"):
                        module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                elif case == "fsync":
                    with patch.object(module, "_fsync_directory_antigravity", side_effect=OSError("fsync")), self.assertRaisesRegex(OSError, "fsync"):
                        module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                else:
                    original = module._rename_no_replace_at_antigravity

                    def appeared(parent_fd: int, source: str, destination: str) -> None:
                        if destination == root.name:
                            root.mkdir()
                            (root / "attacker_antigravity").write_text("foreign")
                            raise FileExistsError("replacement")
                        original(parent_fd, source, destination)

                    with patch.object(module, "_rename_no_replace_at_antigravity", side_effect=appeared), self.assertRaisesRegex(FileExistsError, "replacement"):
                        module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                self.assertTrue(dependencies.audit_path.is_file())

    def test_audit_write_failure_preserves_primary_error_antigravity(self) -> None:
        """Break caught: failure-audit I/O replaces the active synthetic label failure."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import evaluation_antigravity as module
        with tempfile.TemporaryDirectory(prefix="exp454_audit_failure_antigravity_") as temporary:
            dependencies = self._dependencies_antigravity(Path(temporary) / "publication_antigravity", query_error=OSError("query primary"))
            with patch.object(module, "write_create_only_antigravity", side_effect=OSError("audit write")), self.assertRaisesRegex(OSError, "query primary") as raised:
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertIn("failure_audit_error=OSError: audit write", "\n".join(raised.exception.__notes__))

    def test_postrename_replay_failure_moves_retained_inode_and_preserves_recovery_truth_antigravity(self) -> None:
        """Break caught: post-rename replay drift leaves a canonical root or loses forensic recovery truth."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import evaluation_antigravity as module
        for recovery_fault in (None, "validation", "fsync"):
            with self.subTest(recovery_fault=recovery_fault), tempfile.TemporaryDirectory(prefix="exp454_postrename_antigravity_") as temporary:
                root = Path(temporary) / "publication_antigravity"
                dependencies = self._dependencies_antigravity(root)
                original_replay, original_assert, original_fsync = module.inspect_replay_at_fd_antigravity, module._assert_directory_entry_identity_at_antigravity, module._fsync_directory_antigravity
                calls = 0
                moved: dict[str, str] = {}

                def replay(descriptor: int, **kwargs: object):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise OSError("postrename replay drift")
                    return original_replay(descriptor, **kwargs)

                def assert_identity(parent_fd: int, name: str, retained: object) -> None:
                    if ".forensic_" in name:
                        moved["name"] = name
                        if recovery_fault == "validation":
                            raise OSError("recovery validation")
                    return original_assert(parent_fd, name, retained)

                def fsync(descriptor: int) -> None:
                    if recovery_fault == "fsync" and moved:
                        raise OSError("recovery fsync")
                    return original_fsync(descriptor)

                with patch.object(module, "inspect_replay_at_fd_antigravity", side_effect=replay), patch.object(module, "_assert_directory_entry_identity_at_antigravity", side_effect=assert_identity), patch.object(module, "_fsync_directory_antigravity", side_effect=fsync), self.assertRaisesRegex(OSError, "postrename replay drift") as raised:
                    module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                self.assertFalse(root.exists())
                forensic = Path(temporary) / moved["name"]
                self.assertTrue(forensic.is_dir())
                audit = json.loads(dependencies.audit_path.read_text())
                self.assertEqual(audit["schema"], "exp454.task4.failure")
                self.assertEqual((audit["phase"], audit["recovery"]["outcome"], audit["recovery"]["forensic_basename"]), ("post_rename_replay", "moved", moved["name"]))
                if recovery_fault is not None:
                    self.assertIn(f"recovery_{recovery_fault}_error=OSError: recovery {recovery_fault}", "\n".join(raised.exception.__notes__))

    def test_dual_predecessor_authority_without_reviewed_commit_publishes_manifest_antigravity(self) -> None:
        """Break caught: production dual-predecessor authority raises AttributeError on manifest reviewed_commit."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.predecessor_antigravity import (
            NormalizedTargetAuthorityAntigravity,
            _Exp454PredecessorAuthorityAntigravity,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.replay_antigravity import FrozenPredictionCaptureAntigravity, inspect_replay_at_antigravity

        with tempfile.TemporaryDirectory(prefix="exp454_dual_authority_manifest_antigravity_") as temporary:
            root = Path(temporary) / "publication_antigravity"
            dependencies = self._dependencies_antigravity(root)
            captured = dependencies.capture_task5_prediction_antigravity()
            target = dependencies.capture_exp448_antigravity()
            normalized = NormalizedTargetAuthorityAntigravity(
                target.target_dataset_names,
                target.target_baseline_rows,
                target.baseline_confusion,
            )
            authority = _Exp454PredecessorAuthorityAntigravity(
                "1" * 64,
                "2" * 64,
                "3" * 64,
                "4" * 64,
                "5" * 64,
                "6" * 64,
                "7" * 64,
                True,
                target.target_dataset_names,
                captured.canonical_rows,
                normalized,
                48,
                96,
                0,
                0,
                0,
            )
            self.assertFalse(hasattr(authority, "reviewed_commit"))
            capture = FrozenPredictionCaptureAntigravity(
                authority,
                captured.canonical_rows,
                captured.canonical_row_sha256,
                captured.canonical_aggregate_sha256,
            )
            dependencies.capture_task5_prediction_antigravity = lambda: capture
            result = _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            manifest = json.loads((root / "prediction_manifest_antigravity.json").read_text())
            self.assertIsNone(manifest["exp449_reviewed_commit"])
            self.assertEqual(
                inspect_replay_at_antigravity(root, expected_prediction_authority=capture, expected_reviewed_commit="a" * 40),
                result,
            )

    def test_official_surfaces_consume_task3_and_task2_normalized_capture_antigravity(self) -> None:
        """Break caught: official evaluate/inspect keep raw Exp449 capture or skip Task 3 surfaces."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.evaluation_antigravity import evaluate_antigravity, inspect_antigravity

        package = Path(__file__).resolve().parents[1]
        evaluation = (package / "evaluation_antigravity.py").read_text(encoding="utf-8")
        replay = (package / "replay_antigravity.py").read_text(encoding="utf-8")
        self.assertTrue(callable(evaluate_antigravity))
        self.assertTrue(callable(inspect_antigravity))
        self.assertIn("capture_predecessors_antigravity", evaluation)
        self.assertIn("open_test_labels_once_antigravity", evaluation)
        self.assertIn("capture_normalized_target_authority_antigravity", evaluation)
        self.assertIn("capture_predecessors_antigravity", replay)
        self.assertNotIn("capture_exp449_stopped_authority_antigravity", evaluation)
        self.assertNotIn("capture_exp449_stopped_authority_antigravity", replay)
        self.assertIn("EXP454_EVALUATION_ROOT_ANTIGRAVITY", evaluation + replay)


if __name__ == "__main__":
    unittest.main()
