from __future__ import annotations

from dataclasses import fields, replace
import unittest


class LabelFreeFeaturesAntigravityTests(unittest.TestCase):
    def test_builds_exact_eight_rows_in_frozen_family_order_antigravity(self) -> None:
        """Break caught: target order or the frozen family-to-method bindings drift."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.contracts_antigravity import (
            ConfusionAntigravity,
            DatasetRoutingAuthorityAntigravity,
            FrozenRoutingAuthorityAntigravity,
        )
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.features_antigravity import build_label_free_features_antigravity

        rows = tuple(
            DatasetRoutingAuthorityAntigravity(
                dataset=name, family=family, target_length=10, method=method,
                baseline_indices=frozenset({1, 2}), candidate_indices=frozenset({2, 3}),
                baseline_confusion=ConfusionAntigravity(0, 0, 0, 0), candidate_confusion=ConfusionAntigravity(0, 0, 0, 0),
            )
            for name, family, method in (
                *((f"ECG5000__{index}", "ECG5000", "hydra_local_gap") for index in range(5)),
                *((f"ChlorineConcentration__{index}", "ChlorineConcentration", "multirocket_local_gap") for index in range(3)),
            )
        )
        source_order = rows[5:] + rows[:5]
        result = build_label_free_features_antigravity(FrozenRoutingAuthorityAntigravity(source_order, (), 0))
        self.assertEqual(tuple(feature.dataset for feature in result), tuple(row.dataset for row in rows))
        self.assertEqual(tuple(feature.family for feature in result), ("ECG5000",) * 5 + ("ChlorineConcentration",) * 3)
        self.assertEqual(len(result), 8)
        self.assertEqual(result[0].baseline_count, 2)
        self.assertEqual(result[0].candidate_count, 2)
        self.assertEqual(result[0].overlap_count, 1)
        self.assertEqual(result[0].union_count, 3)
        self.assertEqual(result[0].added_count, 1)
        self.assertEqual(result[0].removed_count, 1)
        self.assertEqual(result[0].overlap_ppm, 333333)

    def test_feature_object_never_exposes_outcomes_or_mutates_authority_antigravity(self) -> None:
        """Break caught: confusion, TEST labels, or mutable authority data becomes a router input."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.contracts_antigravity import (
            ConfusionAntigravity,
            DatasetRoutingAuthorityAntigravity,
            DatasetRoutingFeaturesAntigravity,
            FrozenRoutingAuthorityAntigravity,
        )
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.features_antigravity import build_label_free_features_antigravity

        rows = tuple(
            DatasetRoutingAuthorityAntigravity(
                f"ECG5000__{index}", "ECG5000", 4, "hydra_local_gap", frozenset({0}), frozenset({1}),
                ConfusionAntigravity(1, 2, 3, 4), ConfusionAntigravity(4, 3, 2, 1),
            )
            for index in range(5)
        ) + tuple(
            DatasetRoutingAuthorityAntigravity(
                f"ChlorineConcentration__{index}", "ChlorineConcentration", 4, "multirocket_local_gap", frozenset({0}), frozenset({1}),
                ConfusionAntigravity(1, 2, 3, 4), ConfusionAntigravity(4, 3, 2, 1),
            )
            for index in range(3)
        )
        authority = FrozenRoutingAuthorityAntigravity(rows, (), 0)
        features = build_label_free_features_antigravity(authority)
        names = {field.name for field in fields(DatasetRoutingFeaturesAntigravity)}
        self.assertFalse({"confusion", "tp", "fp", "fn", "tn", "test_label", "variant_number"} & names)
        self.assertEqual(replace(features[0], added_count=99).added_count, 99)
        self.assertEqual(authority.rows[0].baseline_indices, frozenset({0}))
        self.assertEqual(authority.label_query_count, 0)


if __name__ == "__main__":
    unittest.main()
