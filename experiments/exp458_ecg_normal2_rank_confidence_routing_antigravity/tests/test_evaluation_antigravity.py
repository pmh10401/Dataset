from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import unittest

from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.tests.test_router_antigravity import (
    DATASET_ANTIGRAVITY,
    feature_antigravity,
    qualification_antigravity,
)


ARTIFACT_MAP_SHA256_ANTIGRAVITY = "3c1321f74016d363899b41357609edca62ecf410d71708a9c90a6b803a4445e0"
COMPARISON_SHA256_ANTIGRAVITY = "6117a5f2ef28a3d543efcf4b499dd715508a4345a5e92679e1c14f0b5e1087cc"
DECISION_SHA256_ANTIGRAVITY = "ea49b140475d4c9d8245a85ec0479a60f1ba6c46e8763b0c35877ec16ca6dc01"
EXP456_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp456_exp455_dataset_key_order_repair_antigravity_20260817"
)


def comparison_antigravity(name: str, values: tuple[int, int, int, int]):
    from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
        ComparisonMetricAntigravity,
        ConfusionAntigravity,
    )

    tp, fp, fn, tn = values
    return ComparisonMetricAntigravity(
        name,
        ConfusionAntigravity(tp, fp, fn, tn),
        float(Fraction(2 * tp, 2 * tp + fp + fn)),
    )


def exp456_authority_antigravity():
    from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
        ConfusionAntigravity,
        DatasetRoutingAuthorityAntigravity,
        DatasetRoutingFeaturesAntigravity,
        FrozenRoutingAuthorityAntigravity,
    )
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
        Exp456ResultAuthorityAntigravity,
    )

    def confusion_antigravity(value: dict[str, int]) -> ConfusionAntigravity:
        return ConfusionAntigravity(value["tp"], value["fp"], value["fn"], value["tn"])

    context = json.loads((EXP456_ROOT_ANTIGRAVITY / "context_antigravity.json").read_text(encoding="utf-8"))
    feature_rows = [
        json.loads(line)
        for line in (EXP456_ROOT_ANTIGRAVITY / "features_antigravity.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    comparison_rows = json.loads(
        (EXP456_ROOT_ANTIGRAVITY / "comparison_metrics_antigravity.json").read_text(encoding="utf-8")
    )["comparisons"]
    rows = tuple(
        DatasetRoutingAuthorityAntigravity(
            row["dataset"],
            row["family"],
            row["target_length"],
            row["method"],
            frozenset(),
            None if row["candidate_confusion"] is None else frozenset(),
            confusion_antigravity(row["baseline_confusion"]),
            None if row["candidate_confusion"] is None else confusion_antigravity(row["candidate_confusion"]),
        )
        for row in context["rows"]
    )
    features = tuple(DatasetRoutingFeaturesAntigravity(**row) for row in feature_rows)
    comparisons = (
        *(
            comparison_antigravity(
                row["name"],
                (
                    row["confusion"]["tp"],
                    row["confusion"]["fp"],
                    row["confusion"]["fn"],
                    row["confusion"]["tn"],
                ),
            )
            for row in comparison_rows[:3]
        ),
    )
    frozen = FrozenRoutingAuthorityAntigravity(rows, tuple(tuple(item) for item in context["exp454_leaf_sha256"]), 0)
    return Exp456ResultAuthorityAntigravity(
        frozen,
        features,
        comparisons,
        ARTIFACT_MAP_SHA256_ANTIGRAVITY,
        COMPARISON_SHA256_ANTIGRAVITY,
        DECISION_SHA256_ANTIGRAVITY,
        0,
    )


def normal2_decision_antigravity(authority=None, passed: bool = True):
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.router_antigravity import (
        route_normal2_antigravity,
    )

    authority = authority or exp456_authority_antigravity()
    by_dataset = {feature.dataset: feature for feature in authority.features}
    return route_normal2_antigravity(by_dataset[DATASET_ANTIGRAVITY], qualification_antigravity(passed))


class EvaluationAntigravityTest(unittest.TestCase):
    def test_reproduces_exp456_comparisons_and_computes_normal2_router_from_48_rows_antigravity(self) -> None:
        """Break caught: evaluator literals the proposed result or changes predecessor comparison math."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )

        authority = exp456_authority_antigravity()
        result = evaluate_routes_antigravity(authority, normal2_decision_antigravity(authority))
        self.assertEqual(len(authority.authority.rows), 48)
        self.assertEqual(len({row.dataset for row in authority.authority.rows}), 48)
        self.assertEqual(
            tuple(row.name for row in result.comparisons),
            (
                "always_baseline",
                "family_wide_local_gap",
                "oracle_per_dataset",
                "normal2_rank_router",
            ),
        )
        self.assertEqual(
            tuple((row.confusion.tp, row.confusion.fp, row.confusion.fn, row.confusion.tn, row.f1) for row in result.comparisons),
            (
                (609, 66, 837, 70788, 0.5742574257425742),
                (611, 63, 835, 70791, 0.5764150943396227),
                (618, 64, 828, 70790, 0.5808270676691729),
                (617, 66, 829, 70788, 0.5796148426491311),
            ),
        )
        final = result.comparisons[-1]
        self.assertEqual(final.confusion.tp + final.confusion.fn, 1446)
        self.assertEqual(final.confusion.fp + final.confusion.tn, 70854)
        self.assertEqual(final.f1, float(Fraction(2 * 617, 2 * 617 + 66 + 829)))
        self.assertEqual(
            tuple((row.family, row.fn_delta, row.fp_delta) for row in result.family_metrics),
            (("ECG5000", -7, 2), ("ChlorineConcentration", -1, -2)),
        )

    def test_criteria_boundaries_are_fn_fp_inclusive_and_f1_strict_antigravity(self) -> None:
        """Break caught: pass criteria use the wrong inclusivity at the published boundary."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
            ComparisonMetricAntigravity,
            ConfusionAntigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            EvaluationResultAntigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity import (
            _criteria_for_final_antigravity,
        )

        final = ComparisonMetricAntigravity(
            "normal2_rank_router",
            ConfusionAntigravity(611, 66, 834, 70788),
            0.5764150943396227,
        )
        criteria, passed, status = _criteria_for_final_antigravity(final)
        self.assertEqual(tuple(row.passed for row in criteria), (True, True, False))
        self.assertFalse(passed)
        self.assertEqual(status, "routing_candidate_not_qualified_for_external_validation")

    def test_rejects_extra_missing_reordered_or_duplicate_authority_rows_antigravity(self) -> None:
        """Break caught: the frozen 48-row dataset authority is not exact by key and order."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )

        authority = exp456_authority_antigravity()
        cases = (
            replace(authority, authority=replace(authority.authority, rows=authority.authority.rows[:-1])),
            replace(authority, authority=replace(authority.authority, rows=authority.authority.rows + (authority.authority.rows[-1],))),
            replace(authority, authority=replace(authority.authority, rows=(authority.authority.rows[1], authority.authority.rows[0]) + authority.authority.rows[2:])),
            replace(authority, authority=replace(authority.authority, rows=(replace(authority.authority.rows[0], dataset=authority.authority.rows[1].dataset),) + authority.authority.rows[1:])),
            replace(authority, authority=replace(authority.authority, rows=authority.authority.rows[:3] + (authority.authority.rows[4], authority.authority.rows[3]) + authority.authority.rows[5:])),
            replace(authority, authority=replace(authority.authority, rows=authority.authority.rows[:4] + (replace(authority.authority.rows[4], dataset="Crop_normal_10_renamed"),) + authority.authority.rows[5:])),
        )
        for case in cases:
            with self.subTest(rows=len(case.authority.rows)), self.assertRaises(ValueError):
                evaluate_routes_antigravity(case, normal2_decision_antigravity(authority))

    def test_rejects_coherent_rehashed_normal2_feature_drift_antigravity(self) -> None:
        """Break caught: added=99/removed=0 normal_2 structure evidence can replace the Exp456 feature row."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )

        authority = exp456_authority_antigravity()
        index = next(index for index, row in enumerate(authority.features) if row.dataset == DATASET_ANTIGRAVITY)
        drifted_features = (
            authority.features[:index]
            + (replace(authority.features[index], added_count=99, removed_count=0),)
            + authority.features[index + 1 :]
        )
        drifted_authority = replace(authority, features=drifted_features)
        with self.assertRaises(ValueError):
            evaluate_routes_antigravity(drifted_authority, normal2_decision_antigravity(drifted_authority))

    def test_rejects_forged_predecessor_or_caller_minted_route_antigravity(self) -> None:
        """Break caught: caller-supplied comparison or route objects bypass semantic provenance."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )

        authority = exp456_authority_antigravity()
        decision = normal2_decision_antigravity(authority)
        wrong_authority = replace(
            authority,
            predecessor_comparisons=(replace(authority.predecessor_comparisons[0], f1=0.0),)
            + authority.predecessor_comparisons[1:],
        )
        with self.assertRaises(ValueError):
            evaluate_routes_antigravity(wrong_authority, decision)
        with self.assertRaises(ValueError):
            evaluate_routes_antigravity(authority, replace(decision, route="baseline"))


if __name__ == "__main__":
    unittest.main()
