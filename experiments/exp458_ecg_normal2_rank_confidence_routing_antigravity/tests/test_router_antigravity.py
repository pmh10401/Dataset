from __future__ import annotations

from dataclasses import replace
import unittest


DATASET_ANTIGRAVITY = "ECG5000_normal_2"
PRIMARY_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
SEEDS_ANTIGRAVITY = (20260721, 20261721)


def feature_antigravity(dataset: str = DATASET_ANTIGRAVITY, added: int = 14, removed: int = 5):
    from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
        DatasetRoutingFeaturesAntigravity,
    )

    family = "ECG5000" if dataset.startswith("ECG5000") else "Other"
    return DatasetRoutingFeaturesAntigravity(
        dataset, family, 100, 10, 10, 5, 15, added, removed, 333_333
    )


def seed_evidence_antigravity(seed: int, passed: bool = True):
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
        RankSeedEvidenceAntigravity,
    )

    per_type = (
        {
            "level_shift": 0.85,
            "gradual_drift": 0.90,
            "amplitude_scaling": 0.90,
            "period_change": 0.95,
        }
        if passed
        else {
            "level_shift": 0.84,
            "gradual_drift": 0.90,
            "amplitude_scaling": 0.90,
            "period_change": 0.95,
        }
    )
    return RankSeedEvidenceAntigravity(
        DATASET_ANTIGRAVITY,
        seed,
        100,
        0,
        0.05,
        per_type,
        sum(per_type.values()) / 4.0,
        passed,
    )


def qualification_antigravity(passed: bool = True):
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
        RankQualificationAntigravity,
    )

    seeds = tuple(seed_evidence_antigravity(seed, passed) for seed in SEEDS_ANTIGRAVITY)
    return RankQualificationAntigravity(DATASET_ANTIGRAVITY, seeds, passed)


class RouterAntigravityTest(unittest.TestCase):
    def test_route_truth_table_uses_rank_and_strict_structure_gates_antigravity(self) -> None:
        """Break caught: normal_2 uses a candidate unless both gates pass, or treats equality as pass."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.router_antigravity import (
            route_normal2_antigravity,
        )

        cases = (
            (qualification_antigravity(True), feature_antigravity(added=14, removed=5), "candidate"),
            (qualification_antigravity(False), feature_antigravity(added=14, removed=5), "baseline"),
            (qualification_antigravity(True), feature_antigravity(added=5, removed=5), "baseline"),
            (qualification_antigravity(True), feature_antigravity(added=4, removed=5), "baseline"),
        )
        for qualification, feature, expected in cases:
            with self.subTest(expected=expected, added=feature.added_count, passed=qualification.passed):
                decision = route_normal2_antigravity(feature, qualification)
                self.assertEqual(decision.dataset, DATASET_ANTIGRAVITY)
                self.assertEqual(decision.structure_gate, feature.added_count > feature.removed_count)
                self.assertEqual(decision.route, expected)

    def test_rejects_normal_two_feature_qualification_key_mismatch_antigravity(self) -> None:
        """Break caught: a caller can bind normal_2 routing to evidence for a different key."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            validate_rank_qualification_semantics_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.router_antigravity import (
            route_normal2_antigravity,
        )

        qualification = qualification_antigravity(True)
        object.__setattr__(qualification, "dataset", "ECG5000_normal_1")
        with self.assertRaises(ValueError):
            validate_rank_qualification_semantics_antigravity(qualification)
        with self.assertRaises(ValueError):
            route_normal2_antigravity(feature_antigravity(), qualification)

    def test_non_normal_two_feature_stays_baseline_even_with_passing_rank_antigravity(self) -> None:
        """Break caught: normal_1 or another dataset inherits the normal_2 rank pass."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.router_antigravity import (
            route_normal2_antigravity,
        )

        decision = route_normal2_antigravity(
            feature_antigravity("ECG5000_normal_1", added=99, removed=0),
            qualification_antigravity(True),
        )
        self.assertEqual(
            (decision.dataset, decision.rank_gate, decision.structure_gate, decision.route),
            ("ECG5000_normal_1", False, True, "baseline"),
        )

    def test_rejects_forged_rank_pass_inside_qualification_antigravity(self) -> None:
        """Break caught: router trusts the stored qualification boolean instead of recomputing seed semantics."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.router_antigravity import (
            route_normal2_antigravity,
        )

        qualification = qualification_antigravity(False)
        object.__setattr__(qualification, "passed", True)
        with self.assertRaises(ValueError):
            route_normal2_antigravity(feature_antigravity(), qualification)

    def test_route_decision_contract_rejects_bool_as_number_and_bad_route_antigravity(self) -> None:
        """Break caught: serialized boolean or route type drift can enter result evidence."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            RouteDecisionAntigravity,
        )

        with self.assertRaises(ValueError):
            RouteDecisionAntigravity(DATASET_ANTIGRAVITY, 1, True, "candidate")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            RouteDecisionAntigravity(DATASET_ANTIGRAVITY, True, True, "other")


if __name__ == "__main__":
    unittest.main()
