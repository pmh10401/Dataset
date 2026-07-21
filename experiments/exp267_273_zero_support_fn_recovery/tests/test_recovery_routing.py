from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.contracts import RecoveryLanes
from experiments.exp267_273_zero_support_fn_recovery.recovery_routing import (
    freeze_recovery_predictions,
    route_recovery_review,
    verify_recovery_prediction_hash,
)


def _existing() -> RecoveryLanes:
    return RecoveryLanes(
        hard_indices=(0,), shadow_indices=(1,), priority_indices=(2,),
        standard_indices=(3,), recovery_shadow_indices=(), recovery_priority_indices=(),
    )


class RecoveryRoutingTests(unittest.TestCase):
    def test_existing_lanes_are_immutable(self) -> None:
        existing = _existing()
        result = route_recovery_review(
            existing=existing,
            source_candidates={"a": {0, 4}, "b": {4}, "c": {4}},
            quality_flags=set(), n_test=8,
        )
        self.assertEqual(result.hard_indices, existing.hard_indices)
        self.assertEqual(result.standard_indices, existing.standard_indices)
        self.assertEqual(result.recovery_shadow_indices, (4,))

    def test_three_two_one_and_quality_route_separately(self) -> None:
        result = route_recovery_review(
            existing=_existing(),
            source_candidates={
                "a": {4, 5, 6, 7}, "b": {4, 5, 7}, "c": {4, 7}, "d": {7},
            },
            quality_flags={7}, n_test=8,
        )
        self.assertEqual(result.recovery_shadow_indices, (4,))
        self.assertEqual(result.recovery_priority_indices, (5,))
        self.assertEqual(result.data_quality_review_indices, (7,))
        self.assertNotIn(6, set().union(*(
            set(result.recovery_shadow_indices), set(result.recovery_priority_indices),
            set(result.data_quality_review_indices),
        )))

    def test_duplicate_variants_must_be_collapsed_before_routing(self) -> None:
        result = route_recovery_review(
            existing=_existing(),
            source_candidates={"normal_forecast": {4}, "structural": {4}},
            quality_flags=set(), n_test=8,
        )
        self.assertEqual(result.recovery_priority_indices, (4,))

    def test_out_of_range_or_existing_overlap_cannot_create_recovery(self) -> None:
        with self.assertRaises(ValueError):
            route_recovery_review(
                existing=_existing(), source_candidates={"a": {8}},
                quality_flags=set(), n_test=8,
            )
        result = route_recovery_review(
            existing=_existing(), source_candidates={"a": {0}, "b": {0}, "c": {0}},
            quality_flags=set(), n_test=8,
        )
        self.assertFalse(result.recovery_shadow_indices)

    def test_freeze_is_label_free_complete_and_hash_verified(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = [{
                "dataset": "d", "n_test": 8,
                "hard_indices": [0], "shadow_indices": [1], "priority_indices": [2],
                "standard_indices": [3], "recovery_shadow_indices": [4],
                "recovery_priority_indices": [5], "data_quality_review_indices": [7],
                "quality_flag_indices": [7],
                "one_source_sensitivity_indices": [6],
                "support_by_index": {"4": ["a", "b", "c"], "5": ["a", "b"], "6": ["a"], "7": ["a", "b", "c", "d"]},
                "source_eligibility": {"a": True}, "input_sha256": {"fixture": "hash"},
                "config_sha256": "config", "contains_test_labels": False,
                "contains_evaluation_metrics": False, "review_only": True, "retrospective": True,
            }]
            prediction, manifest = freeze_recovery_predictions(
                rows, output_dir=root, expected_datasets=("d",),
                membership_sha256="membership", input_sha256={"fixture": "hash"},
            )
            self.assertTrue(verify_recovery_prediction_hash(
                prediction, manifest, expected_datasets=("d",)
            ))
            prediction.write_text(prediction.read_text() + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_recovery_prediction_hash(prediction, manifest, expected_datasets=("d",))


if __name__ == "__main__":
    unittest.main()
