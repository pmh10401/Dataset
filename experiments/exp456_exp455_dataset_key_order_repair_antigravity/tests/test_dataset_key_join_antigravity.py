from __future__ import annotations

import unittest
from dataclasses import replace


class DatasetKeyJoinAntigravityTests(unittest.TestCase):
    @staticmethod
    def _inputs_antigravity(reverse_authority: bool):
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
            ConfusionAntigravity,
            DatasetRoutingAuthorityAntigravity,
            DatasetRoutingFeaturesAntigravity,
            FrozenRoutingAuthorityAntigravity,
        )
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.router_antigravity import leave_one_dataset_out_antigravity
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.config_antigravity import EVALUATION_LEAVES_ANTIGRAVITY

        targets = (("ECG5000", 5, "hydra_local_gap"), ("ChlorineConcentration", 3, "multirocket_local_gap"))
        target_rows = []
        features = []
        for family, count, method in targets:
            for index in range(count):
                dataset = f"{family}__{index}"
                baseline = ConfusionAntigravity(10, 1, 5, 90)
                candidate = ConfusionAntigravity(12, 2, 1 if family == "ECG5000" else 7, 86)
                target_rows.append(DatasetRoutingAuthorityAntigravity(dataset, family, 100 + index, method, frozenset({1, 2}), frozenset({1, 3}), baseline, candidate))
                features.append(DatasetRoutingFeaturesAntigravity(dataset, family, 100 + index, 2, 2, 1, 3, 1, 1, 333_333))
        rows = list(reversed(target_rows) if reverse_authority else target_rows)
        rows.extend(DatasetRoutingAuthorityAntigravity(f"Other__{index}", "Other", 50, None, frozenset(), None, ConfusionAntigravity(10, 1, 5, 90), None) for index in range(40))
        configured = dict(FROZEN_CONFIG_ANTIGRAVITY.exp454_sha256)
        aliases = {"artifact_sha256_antigravity.json": configured["artifact_map"], "evaluation_antigravity.json": configured["evaluation"], "decision_antigravity.json": configured["decision"]}
        inventory = tuple(sorted((name, aliases.get(name, "a" * 64)) for name in EVALUATION_LEAVES_ANTIGRAVITY))
        authority = FrozenRoutingAuthorityAntigravity(tuple(rows), inventory, 0)
        outcomes = {row.dataset: (row.baseline_confusion, row.candidate_confusion) for row in target_rows}
        return authority, tuple(features), leave_one_dataset_out_antigravity(tuple(features), outcomes)

    def test_reversed_authority_order_matches_public_feature_order_antigravity(self) -> None:
        """Break caught: positional evaluation rejects valid frozen authority/public ordering."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity

        same_authority, features, decisions = self._inputs_antigravity(False)
        reversed_authority, _same_features, _same_decisions = self._inputs_antigravity(True)
        same_order = evaluate_routing_antigravity(same_authority, features, decisions)
        reversed_order = evaluate_routing_antigravity(reversed_authority, features, decisions)

        self.assertEqual(reversed_order, same_order)
        self.assertEqual(tuple(row.dataset for row in reversed_order.decisions), tuple(row.dataset for row in features))
        self.assertEqual(tuple((row.confusion.tp, row.confusion.fp, row.confusion.fn, row.confusion.tn, row.f1) for row in reversed_order.comparisons), ((480, 48, 240, 4320, 0.7692307692307693), (496, 56, 226, 4288, 0.7786499215070644), (490, 53, 220, 4300, 0.7821229050279329), (490, 53, 220, 4300, 0.7821229050279329)))
        self.assertEqual(reversed_order.family_fn_delta, (("ECG5000", -20), ("ChlorineConcentration", 0)))
        self.assertTrue(reversed_order.passed)
        self.assertEqual(reversed_order.status, "routing_candidate_qualified_for_external_validation")

    def test_reordered_valid_decisions_join_by_dataset_antigravity(self) -> None:
        """Break caught: valid decision ordering is treated as a positional routing input."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity

        authority, features, decisions = self._inputs_antigravity(True)
        self.assertEqual(
            evaluate_routing_antigravity(authority, features, tuple(reversed(decisions))),
            evaluate_routing_antigravity(authority, features, decisions),
        )

    def test_rejects_invalid_dataset_key_triples_before_publication_antigravity(self) -> None:
        """Break caught: malformed keys or a mismatched family, method, or length reaches publication."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity

        authority, features, decisions = self._inputs_antigravity(True)
        bad_targets = tuple(replace(authority.rows[index], dataset=authority.rows[1].dataset) if index == 0 else authority.rows[index] for index in range(len(authority.rows)))
        cases = (
            replace(authority, rows=bad_targets),
            authority,
            authority,
            authority,
            authority,
            authority,
            authority,
        )
        bad_features = (
            features,
            features[:-1],
            features + (features[0],),
            (replace(features[0], dataset=True),) + features[1:],
            (replace(features[0], family="WrongFamily"),) + features[1:],
            (replace(features[0], target_length=True),) + features[1:],
            (replace(features[0], target_length=features[0].target_length + 1),) + features[1:],
        )
        bad_decisions = (
            decisions,
            decisions,
            decisions,
            decisions,
            decisions,
            (replace(decisions[0], method="wrong_method"),) + decisions[1:],
            decisions,
        )
        for broken_authority, broken_features, broken_decisions in zip(cases, bad_features, bad_decisions, strict=True):
            with self.subTest(authority=broken_authority is not authority, feature_count=len(broken_features)):
                with self.assertRaises(ValueError):
                    evaluate_routing_antigravity(broken_authority, broken_features, broken_decisions)

    def test_rejects_string_subclasses_in_join_identity_fields_antigravity(self) -> None:
        """Break caught: equality-compatible string subclasses enter frozen dataset-key evidence."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity

        class StringSubclassAntigravity(str):
            pass

        authority, features, decisions = self._inputs_antigravity(True)
        target_index = next(index for index, row in enumerate(authority.rows) if row.method is not None)
        cases = (
            (replace(authority, rows=authority.rows[:target_index] + (replace(authority.rows[target_index], family=StringSubclassAntigravity(authority.rows[target_index].family)),) + authority.rows[target_index + 1:]), features, decisions),
            (replace(authority, rows=authority.rows[:target_index] + (replace(authority.rows[target_index], method=StringSubclassAntigravity(authority.rows[target_index].method)),) + authority.rows[target_index + 1:]), features, decisions),
            (authority, (replace(features[0], family=StringSubclassAntigravity(features[0].family)),) + features[1:], decisions),
            (authority, features, (replace(decisions[0], family=StringSubclassAntigravity(decisions[0].family)),) + decisions[1:]),
            (authority, features, (replace(decisions[0], method=StringSubclassAntigravity(decisions[0].method)),) + decisions[1:]),
            (authority, features, (replace(decisions[0], selected_route=StringSubclassAntigravity(decisions[0].selected_route)),) + decisions[1:]),
        )
        for broken_authority, broken_features, broken_decisions in cases:
            with self.subTest():
                with self.assertRaises(ValueError):
                    evaluate_routing_antigravity(broken_authority, broken_features, broken_decisions)


if __name__ == "__main__":
    unittest.main()
