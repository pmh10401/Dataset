"""Red-first order fixtures for the Exp452 controlled evaluator fork."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class EvaluationAntigravityTests(unittest.TestCase):
    def _dependencies_antigravity(self, root: Path, ordered_indices: list[int] | None = None, **kwargs: object):
        from experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_evaluation_antigravity import _DependenciesAntigravity
        from experiments.exp452_exp449_index_order_repair_antigravity.replay_antigravity import FrozenPredictionCaptureAntigravity
        legacy = _DependenciesAntigravity(root, **kwargs)
        captured = legacy.capture_task5_prediction_antigravity()
        from experiments.exp452_exp449_index_order_repair_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        rows = [json.loads(line) for line in captured.canonical_rows.splitlines()]
        if ordered_indices is not None:
            for row in rows:
                row["indices"] = ordered_indices
        canonical_rows = b"".join(canonical_json_bytes_antigravity(row) for row in rows)
        canonical_row_sha256 = tuple(__import__("hashlib").sha256(line).hexdigest() for line in canonical_rows.splitlines())
        from experiments.exp452_exp449_index_order_repair_antigravity.contracts_antigravity import ConfusionAntigravity, DatasetBaselineAntigravity
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
            __import__("hashlib").sha256("".join(canonical_row_sha256).encode("ascii")).hexdigest(),
        )
        return SimpleNamespace(
            root=root,
            audit_path=root.parent / "failure_audit_antigravity.json",
            capture_exp448_antigravity=converted_target_antigravity,
            capture_task5_prediction_antigravity=lambda: capture,
            open_labels_antigravity=legacy.open_labels_antigravity,
            legacy_antigravity=legacy,
        )

    def test_minimal_order_fixture_emits_ascending_evidence_indices_antigravity(self) -> None:
        """Break caught: replacing sorted with list preserves unordered frozenset traversal."""
        prediction = [981, 983, 984, 986, 988, 989, 992, 993, 994, 995, 996, 998, 999]
        self.assertEqual(sorted(frozenset(prediction)), prediction)

    def test_full_48_dataset_96_row_fixture_matches_frozen_lists_in_order_antigravity(self) -> None:
        """Break caught: any unordered evidence row makes strict replay reject its frozen peer."""
        from experiments.exp452_exp449_index_order_repair_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp452_exp449_index_order_repair_antigravity.replay_antigravity import inspect_replay_at_antigravity
        import json

        with tempfile.TemporaryDirectory(prefix="exp452_evaluation_antigravity_") as temporary:
            root = Path(temporary) / "publication_antigravity"
            dependencies = self._dependencies_antigravity(root, [981, 983, 984, 986, 988, 989, 992, 993, 994, 995, 996, 998, 999])
            result = _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            frozen = [json.loads(line)["indices"] for line in (root / "frozen_predictions_antigravity.jsonl").read_text().splitlines()]
            evidence = json.loads((root / "evaluation_antigravity.json").read_text())["dataset_evidence"]
            actual = [entry["candidates"][method]["indices"] for entry in evidence for method in entry["candidates"]]
            self.assertEqual(actual, frozen)
            self.assertEqual(inspect_replay_at_antigravity(root, expected_prediction_authority=dependencies.capture_task5_prediction_antigravity()), result)

    def test_publication_destination_and_label_failure_matrix_fails_closed_antigravity(self) -> None:
        """Break caught: unsafe publication or one label boundary failure publishes a valid root."""
        from experiments.exp452_exp449_index_order_repair_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity

        cases = {
            "existing": {}, "partial": {}, "symlink": {},
            "setup": {"open_error": OSError("setup")}, "query": {"query_error": OSError("query")},
            "close": {"close_error": OSError("close")}, "post_close": {"drift_at": 2},
            "post_compute": {"drift_at": 3}, "precommit": {"drift_at": 4},
        }
        for case, kwargs in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="exp452_failure_antigravity_") as temporary:
                root = Path(temporary) / "publication_antigravity"
                if case == "existing": root.mkdir()
                elif case == "partial": root.mkdir(); (root / "partial_antigravity.json").write_text("{}")
                elif case == "symlink": root.symlink_to("missing_antigravity")
                dependencies = self._dependencies_antigravity(root, **kwargs)
                with self.assertRaises((OSError, ValueError)):
                    _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                self.assertFalse(root.is_dir() and not root.is_symlink() and len(tuple(root.iterdir())) == 12)
                self.assertTrue(dependencies.audit_path.is_file())
                if case in {"existing", "partial", "symlink"}:
                    self.assertEqual(dependencies.legacy_antigravity.open_labels_calls, 0)

    def test_leaf_write_fsync_and_namespace_replacement_preserve_failure_truth_antigravity(self) -> None:
        """Break caught: a publish I/O or destination-replacement fault becomes a valid evaluation."""
        from experiments.exp452_exp449_index_order_repair_antigravity import evaluation_antigravity as module

        for case in ("write", "fsync", "replacement"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="exp452_publish_failure_antigravity_") as temporary:
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
                            root.mkdir(); (root / "attacker_antigravity").write_text("foreign")
                            raise FileExistsError("replacement")
                        original(parent_fd, source, destination)
                    with patch.object(module, "_rename_no_replace_at_antigravity", side_effect=appeared), self.assertRaisesRegex(FileExistsError, "replacement"):
                        module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                self.assertTrue(dependencies.audit_path.is_file())

    def test_audit_write_failure_preserves_primary_error_antigravity(self) -> None:
        """Break caught: failure-audit I/O replaces the active synthetic label failure."""
        from experiments.exp452_exp449_index_order_repair_antigravity import evaluation_antigravity as module
        with tempfile.TemporaryDirectory(prefix="exp452_audit_failure_antigravity_") as temporary:
            dependencies = self._dependencies_antigravity(Path(temporary) / "publication_antigravity", query_error=OSError("query primary"))
            with patch.object(module, "write_create_only_antigravity", side_effect=OSError("audit write")), self.assertRaisesRegex(OSError, "query primary") as raised:
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertIn("failure_audit_error=OSError: audit write", "\n".join(raised.exception.__notes__))


if __name__ == "__main__":
    unittest.main()
