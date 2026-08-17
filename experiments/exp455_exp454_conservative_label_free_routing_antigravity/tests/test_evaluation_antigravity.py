from __future__ import annotations

import unittest


class RoutingEvaluationAntigravityTests(unittest.TestCase):
    @staticmethod
    def _inputs_antigravity():
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.contracts_antigravity import (
            ConfusionAntigravity,
            DatasetRoutingAuthorityAntigravity,
            DatasetRoutingFeaturesAntigravity,
            FoldDecisionAntigravity,
            FrozenRoutingAuthorityAntigravity,
            RouteRuleAntigravity,
        )

        rows = []
        features = []
        decisions = []
        targets = (("ECG5000", 5, "hydra_local_gap"), ("ChlorineConcentration", 3, "multirocket_local_gap"))
        for family, count, method in targets:
            for index in range(count):
                dataset = f"{family}__{index}"
                baseline = ConfusionAntigravity(10, 1, 5, 90)
                candidate = ConfusionAntigravity(12, 2, 1 if family == "ECG5000" else 7, 86)
                rows.append(DatasetRoutingAuthorityAntigravity(dataset, family, 100 + index, method, frozenset({1, 2}), frozenset({1, 3}), baseline, candidate))
                features.append(DatasetRoutingFeaturesAntigravity(dataset, family, 100 + index, 2, 2, 1, 3, 1, 1, 333_333))
                decisions.append(FoldDecisionAntigravity(dataset, family, method, "candidate" if family == "ECG5000" else "baseline", RouteRuleAntigravity(None, None, "always" if family == "ECG5000" else "never")))
        for index in range(40):
            rows.append(DatasetRoutingAuthorityAntigravity(f"Other__{index}", "Other", 50, None, frozenset(), None, ConfusionAntigravity(10, 1, 5, 90), None))
        return FrozenRoutingAuthorityAntigravity(tuple(rows), (), 0), tuple(features), tuple(decisions)

    def test_evaluates_all_four_preregistered_comparisons_and_family_deltas_antigravity(self) -> None:
        """Break caught: a comparison skips baseline rows or uses an approximate F1."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.evaluation_antigravity import evaluate_routing_antigravity

        authority, features, decisions = self._inputs_antigravity()
        evaluation = evaluate_routing_antigravity(authority, features, decisions)

        self.assertEqual(tuple(row.name for row in evaluation.comparisons), ("always_baseline", "family_wide_local_gap", "oracle_per_dataset", "lodo_router"))
        self.assertEqual(tuple((row.confusion.tp, row.confusion.fp, row.confusion.fn, row.confusion.tn) for row in evaluation.comparisons), ((480, 48, 240, 4320), (496, 56, 226, 4288), (490, 53, 220, 4300), (490, 53, 220, 4300)))
        self.assertEqual(evaluation.family_fn_delta, (("ECG5000", -20), ("ChlorineConcentration", 0)))
        self.assertTrue(evaluation.passed)
        self.assertEqual(evaluation.status, "routing_candidate_qualified_for_external_validation")

    def test_rejects_missing_duplicate_or_non_integer_confusion_authority_antigravity(self) -> None:
        """Break caught: an incomplete authority or boolean confusion silently changes a 48-row sum."""
        from dataclasses import replace
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.contracts_antigravity import ConfusionAntigravity

        authority, features, decisions = self._inputs_antigravity()
        for broken in (
            replace(authority, rows=authority.rows[:-1]),
            replace(authority, rows=authority.rows + (authority.rows[-1],)),
            replace(authority, rows=(replace(authority.rows[0], baseline_confusion=ConfusionAntigravity(True, 1, 1, 1)),) + authority.rows[1:]),
        ):
            with self.assertRaises(ValueError):
                evaluate_routing_antigravity(broken, features, decisions)


if __name__ == "__main__":
    unittest.main()
