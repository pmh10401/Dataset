from __future__ import annotations

import unittest
from dataclasses import replace

from experiments.exp266_fn_recovery_shadow.config import EVIDENCE_GROUPS
from experiments.exp266_fn_recovery_shadow.contracts import FrozenDatasetInputs
from experiments.exp266_fn_recovery_shadow.routing import (
    RecoveryPrediction,
    route_recovery,
    validate_recovery_prediction,
)


def example_inputs(
    evidence: dict[str, set[int]], quality_flags: frozenset[int] = frozenset()
) -> FrozenDatasetInputs:
    return FrozenDatasetInputs(
        dataset="demo",
        n_train=100,
        n_test=8,
        hard=frozenset({0}),
        shadow=frozenset({1}),
        priority=frozenset({2}),
        standard=frozenset({3}),
        evidence={
            name: frozenset(evidence.get(name, set())) for name in EVIDENCE_GROUPS
        },
        available_groups=frozenset(EVIDENCE_GROUPS),
        quality_flags=quality_flags,
    )


class RecoveryRoutingTests(unittest.TestCase):
    def test_routes_only_outside_existing_union(self) -> None:
        inputs = example_inputs(
            {"kernel": {0, 4}, "statistical": {4}, "structural": {4}}
        )

        row = route_recovery(inputs)

        self.assertEqual(row.recovery_shadow, frozenset({4}))
        self.assertEqual(row.hard, frozenset({0}))
        self.assertEqual(row.support_by_index, {4: 3})

    def test_quality_review_precedes_agreement_lane(self) -> None:
        row = route_recovery(
            example_inputs(
                {"kernel": {6}, "statistical": {6}, "structural": {6}},
                quality_flags=frozenset({6}),
            )
        )

        self.assertEqual(row.data_quality_review, frozenset({6}))
        self.assertFalse(row.recovery_shadow)
        self.assertFalse(row.recovery_priority)

    def test_support_three_or_more_is_shadow_and_two_is_priority(self) -> None:
        row = route_recovery(
            example_inputs(
                {
                    "kernel": {4, 5, 6, 7},
                    "statistical": {5, 6, 7},
                    "structural": {6, 7},
                    "change_point": {7},
                }
            )
        )

        self.assertEqual(row.recovery_shadow, frozenset({6, 7}))
        self.assertEqual(row.recovery_priority, frozenset({5}))
        self.assertNotIn(4, row.recovery_shadow | row.recovery_priority)
        self.assertEqual(row.support_by_index, {4: 1, 5: 2, 6: 3, 7: 4})

    def test_unavailable_evidence_group_does_not_add_support(self) -> None:
        inputs = example_inputs(
            {"kernel": {4}, "statistical": {4}, "structural": {4}}
        )
        inputs = replace(
            inputs,
            available_groups=frozenset({"kernel", "statistical"}),
        )

        row = route_recovery(inputs)

        self.assertEqual(row.recovery_priority, frozenset({4}))
        self.assertEqual(row.support_by_index, {4: 2})

    def test_baseline_lanes_are_preserved_exactly(self) -> None:
        inputs = example_inputs({"kernel": {0, 1, 2, 3, 4}})

        row = route_recovery(inputs)

        self.assertEqual(row.hard, inputs.hard)
        self.assertEqual(row.shadow, inputs.shadow)
        self.assertEqual(row.priority, inputs.priority)
        self.assertEqual(row.standard, inputs.standard)

    def test_quality_flag_without_evidence_is_not_routed(self) -> None:
        row = route_recovery(example_inputs({}, quality_flags=frozenset({7})))

        self.assertFalse(row.data_quality_review)
        self.assertFalse(row.recovery_shadow)
        self.assertFalse(row.recovery_priority)

    def test_prediction_validation_rejects_lane_overlap(self) -> None:
        inputs = example_inputs({})
        prediction = RecoveryPrediction(
            dataset="demo",
            hard=frozenset({0}),
            shadow=frozenset({0}),
            priority=frozenset({2}),
            standard=frozenset({3}),
            recovery_shadow=frozenset(),
            recovery_priority=frozenset(),
            data_quality_review=frozenset(),
            support_by_index={},
        )

        with self.assertRaisesRegex(ValueError, "lane overlap"):
            validate_recovery_prediction(inputs, prediction)

    def test_prediction_validation_rejects_non_integral_and_out_of_range_indices(
        self,
    ) -> None:
        inputs = example_inputs({})
        for invalid in ("4", 8):
            with self.subTest(invalid=invalid):
                prediction = RecoveryPrediction(
                    dataset="demo",
                    hard=frozenset({invalid}),  # type: ignore[arg-type]
                    shadow=frozenset({1}),
                    priority=frozenset({2}),
                    standard=frozenset({3}),
                    recovery_shadow=frozenset(),
                    recovery_priority=frozenset(),
                    data_quality_review=frozenset(),
                    support_by_index={},
                )
                with self.assertRaisesRegex(ValueError, "integer|TEST range"):
                    validate_recovery_prediction(inputs, prediction)


if __name__ == "__main__":
    unittest.main()
