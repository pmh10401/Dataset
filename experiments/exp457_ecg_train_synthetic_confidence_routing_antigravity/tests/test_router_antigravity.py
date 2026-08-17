from __future__ import annotations

from dataclasses import replace
import unittest


def qualification_antigravity(dataset: str, passed: bool):
    from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.contracts_antigravity import (
        TrainQualificationAntigravity,
        TrainSeedEvidenceAntigravity,
    )

    primary = {
        "level_shift": 0.60,
        "gradual_drift": 0.60,
        "amplitude_scaling": 0.60,
        "period_change": 0.60,
    }
    evidence = tuple(
        TrainSeedEvidenceAntigravity(
            dataset, seed, 100, 2, 0.049, 0.75, primary, passed
        )
        for seed in (20260721, 20261721)
    )
    return TrainQualificationAntigravity(dataset, evidence, passed)


def feature_antigravity(dataset: str, added: int, removed: int):
    from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
        DatasetRoutingFeaturesAntigravity,
    )

    return DatasetRoutingFeaturesAntigravity(
        dataset, "ECG5000", 100, 10, 10, 5, 15, added, removed, 333_333
    )


class RouterAntigravityTest(unittest.TestCase):
    def test_candidate_requires_both_recomputed_gates_antigravity(self) -> None:
        """Break caught: equality, TRAIN failure, or removed-heavy structure reaches candidate."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.router_antigravity import (
            route_ecg_antigravity,
        )

        cases = (
            (True, 14, 5, True, True, "candidate"),
            (False, 14, 5, False, True, "baseline"),
            (True, 5, 5, True, False, "baseline"),
            (True, 4, 5, True, False, "baseline"),
        )
        for passed, added, removed, train_gate, structure_gate, route in cases:
            with self.subTest(passed=passed, added=added, removed=removed):
                decision = route_ecg_antigravity(
                    feature_antigravity("ECG5000_normal_1", added, removed),
                    qualification_antigravity("ECG5000_normal_1", passed),
                )
                self.assertEqual(
                    (decision.train_gate, decision.structure_gate, decision.route),
                    (train_gate, structure_gate, route),
                )

    def test_rejects_key_mismatch_wrong_family_and_numeric_type_drift_antigravity(self) -> None:
        """Break caught: another dataset/family or bool-as-count supplies routing evidence."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.router_antigravity import (
            route_ecg_antigravity,
        )

        feature = feature_antigravity("ECG5000_normal_1", 14, 5)
        qualification = qualification_antigravity("ECG5000_normal_1", True)
        broken = (
            (feature, qualification_antigravity("ECG5000_normal_2", True)),
            (replace(feature, family="ChlorineConcentration"), qualification),
            (replace(feature, added_count=True), qualification),
            (replace(feature, removed_count=5.0), qualification),
        )
        for bad_feature, bad_qualification in broken:
            with self.subTest(feature=bad_feature, qualification=bad_qualification):
                with self.assertRaises(ValueError):
                    route_ecg_antigravity(bad_feature, bad_qualification)


if __name__ == "__main__":
    unittest.main()
