"""Red-first order fixtures for the Exp452 controlled evaluator fork."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class EvaluationAntigravityTests(unittest.TestCase):
    def _dependencies_antigravity(self, root: Path):
        from experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_evaluation_antigravity import _DependenciesAntigravity
        from experiments.exp452_exp449_index_order_repair_antigravity.replay_antigravity import FrozenPredictionCaptureAntigravity
        legacy = _DependenciesAntigravity(root)
        captured = legacy.capture_task5_prediction_antigravity()
        legacy_target = legacy.capture_exp448_antigravity()
        from experiments.exp452_exp449_index_order_repair_antigravity.contracts_antigravity import ConfusionAntigravity, DatasetBaselineAntigravity
        baselines = tuple(
            DatasetBaselineAntigravity(
                row.dataset_order,
                row.family,
                row.stable_candidates,
                ConfusionAntigravity(row.confusion.tp, row.confusion.fp, row.confusion.fn, row.confusion.tn),
            )
            for row in legacy_target.target_baseline_rows
        )
        target = SimpleNamespace(
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
            captured.canonical_rows,
            captured.canonical_row_sha256,
            captured.canonical_aggregate_sha256,
        )
        return SimpleNamespace(
            root=root,
            audit_path=root.parent / "failure_audit_antigravity.json",
            capture_exp448_antigravity=lambda: target,
            capture_task5_prediction_antigravity=lambda: capture,
            open_labels_antigravity=legacy.open_labels_antigravity,
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
            dependencies = self._dependencies_antigravity(root)
            result = _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            frozen = [json.loads(line)["indices"] for line in (root / "frozen_predictions_antigravity.jsonl").read_text().splitlines()]
            evidence = json.loads((root / "evaluation_antigravity.json").read_text())["dataset_evidence"]
            actual = [entry["candidates"][method]["indices"] for entry in evidence for method in entry["candidates"]]
            self.assertEqual(actual, frozen)
            self.assertEqual(inspect_replay_at_antigravity(root, expected_prediction_authority=dependencies.capture_task5_prediction_antigravity()), result)


if __name__ == "__main__":
    unittest.main()
