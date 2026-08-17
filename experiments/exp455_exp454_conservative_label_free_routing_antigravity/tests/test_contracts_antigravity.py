from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import unittest

from experiments.exp455_exp454_conservative_label_free_routing_antigravity.contracts_antigravity import (
    ComparisonMetricAntigravity,
    ConfusionAntigravity,
    DatasetRoutingAuthorityAntigravity,
    DatasetRoutingFeaturesAntigravity,
    FoldDecisionAntigravity,
    FrozenResultAntigravity,
    FrozenRoutingAuthorityAntigravity,
    RouteRuleAntigravity,
    RoutingEvaluationAntigravity,
)


class ContractsAntigravityTest(unittest.TestCase):
    def test_contract_field_lists_are_frozen_antigravity(self) -> None:
        """Break caught: a planned phase changes the persisted routing data shape."""
        expected = {
            ConfusionAntigravity: ("tp", "fp", "fn", "tn"),
            DatasetRoutingFeaturesAntigravity: (
                "dataset", "family", "target_length", "baseline_count", "candidate_count",
                "overlap_count", "union_count", "added_count", "removed_count", "overlap_ppm",
            ),
            DatasetRoutingAuthorityAntigravity: (
                "dataset", "family", "target_length", "method", "baseline_indices", "candidate_indices",
                "baseline_confusion", "candidate_confusion",
            ),
            FrozenRoutingAuthorityAntigravity: ("rows", "exp454_leaf_sha256", "label_query_count"),
            RouteRuleAntigravity: ("feature_name", "threshold_twice", "candidate_when"),
            FoldDecisionAntigravity: ("dataset", "family", "method", "selected_route", "rule"),
            ComparisonMetricAntigravity: ("name", "confusion", "f1"),
            RoutingEvaluationAntigravity: ("comparisons", "family_fn_delta", "decisions", "passed", "status"),
            FrozenResultAntigravity: ("root", "leaf_sha256", "artifact_map_sha256"),
        }
        for contract, field_names in expected.items():
            with self.subTest(contract=contract.__name__):
                self.assertEqual(tuple(field.name for field in fields(contract)), field_names)
                self.assertTrue(hasattr(contract, "__slots__"))

    def test_contracts_construct_the_label_free_routing_boundary_antigravity(self) -> None:
        """Break caught: later phases cannot construct the planned immutable authority and decision records."""
        baseline = ConfusionAntigravity(tp=10, fp=2, fn=3, tn=100)
        candidate = ConfusionAntigravity(tp=11, fp=2, fn=2, tn=100)
        authority = DatasetRoutingAuthorityAntigravity(
            dataset="ECG5000_1", family="ECG5000", target_length=113,
            method="hydra_local_gap", baseline_indices=frozenset({1, 3}), candidate_indices=frozenset({1, 4}),
            baseline_confusion=baseline, candidate_confusion=candidate,
        )
        features = DatasetRoutingFeaturesAntigravity(
            dataset="ECG5000_1", family="ECG5000", target_length=113, baseline_count=2,
            candidate_count=2, overlap_count=1, union_count=3, added_count=1, removed_count=1, overlap_ppm=500000,
        )
        rule = RouteRuleAntigravity(feature_name="added_count", threshold_twice=3, candidate_when="lte")
        decision = FoldDecisionAntigravity(
            dataset=features.dataset, family=features.family, method="hydra_local_gap", selected_route="candidate", rule=rule,
        )
        frozen_authority = FrozenRoutingAuthorityAntigravity(rows=(authority,), exp454_leaf_sha256=(("decision_antigravity.json", "a" * 64),))
        evaluation = RoutingEvaluationAntigravity(
            comparisons=(ComparisonMetricAntigravity(name="router", confusion=candidate, f1=0.8),),
            family_fn_delta=(("ECG5000", -1),), decisions=(decision,), passed=True,
            status="routing_candidate_qualified_for_external_validation",
        )
        result = FrozenResultAntigravity(root="/tmp/result", leaf_sha256=(("result.json", "b" * 64),), artifact_map_sha256="c" * 64)
        self.assertEqual(frozen_authority.label_query_count, 0)
        self.assertEqual(evaluation.decisions[0].selected_route, "candidate")
        self.assertEqual(result.artifact_map_sha256, "c" * 64)
        with self.assertRaises(FrozenInstanceError):
            features.dataset = "changed"  # type: ignore[misc]
