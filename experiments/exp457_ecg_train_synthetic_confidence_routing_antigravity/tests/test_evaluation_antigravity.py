from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import tempfile
from pathlib import Path
import unittest

from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.tests.test_router_antigravity import (
    qualification_antigravity,
)


ARTIFACT_MAP_SHA256_ANTIGRAVITY = "3c1321f74016d363899b41357609edca62ecf410d71708a9c90a6b803a4445e0"
COMPARISON_SHA256_ANTIGRAVITY = "6117a5f2ef28a3d543efcf4b499dd715508a4345a5e92679e1c14f0b5e1087cc"
DECISION_SHA256_ANTIGRAVITY = "ea49b140475d4c9d8245a85ec0479a60f1ba6c46e8763b0c35877ec16ca6dc01"


def _comparison_antigravity(name: str, values: tuple[int, int, int, int]):
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
    from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.contracts_antigravity import (
        Exp456ResultAuthorityAntigravity,
    )

    ecg = tuple(f"ECG5000_normal_{index}" for index in range(1, 6))
    chlorine = tuple(f"ChlorineConcentration_normal_{index}" for index in range(1, 4))
    baselines = {
        ecg[0]: ConfusionAntigravity(7, 1, 0, 0),
        ecg[1]: ConfusionAntigravity(0, 0, 8, 0),
        ecg[2]: ConfusionAntigravity(0, 2, 1, 0),
    }
    candidates = {
        ecg[0]: ConfusionAntigravity(0, 0, 7, 1),
        ecg[1]: ConfusionAntigravity(8, 0, 0, 0),
        ecg[2]: ConfusionAntigravity(1, 0, 0, 2),
    }
    feature_gates = ((12, 20), (14, 5), (0, 0), (0, 0), (0, 1))
    rows = []
    features = []
    for dataset, (added, removed) in zip(ecg, feature_gates, strict=True):
        baseline = baselines.get(dataset, ConfusionAntigravity(0, 0, 0, 0))
        candidate = candidates.get(dataset, baseline)
        rows.append(
            DatasetRoutingAuthorityAntigravity(
                dataset,
                "ECG5000",
                100,
                "hydra_local_gap",
                frozenset(),
                frozenset(),
                baseline,
                candidate,
            )
        )
        features.append(
            DatasetRoutingFeaturesAntigravity(
                dataset, "ECG5000", 100, 10, 10, 5, 15, added, removed, 333_333
            )
        )
    for dataset in chlorine:
        zero = ConfusionAntigravity(0, 0, 0, 0)
        rows.append(
            DatasetRoutingAuthorityAntigravity(
                dataset,
                "ChlorineConcentration",
                100,
                "multirocket_local_gap",
                frozenset(),
                frozenset(),
                zero,
                zero,
            )
        )
        features.append(
            DatasetRoutingFeaturesAntigravity(
                dataset,
                "ChlorineConcentration",
                100,
                10,
                10,
                5,
                15,
                1,
                1,
                333_333,
            )
        )
    rows.append(
        DatasetRoutingAuthorityAntigravity(
            "Other_normal_1",
            "Other",
            100,
            None,
            frozenset(),
            None,
            ConfusionAntigravity(602, 63, 828, 70788),
            None,
        )
    )
    for index in range(2, 41):
        rows.append(
            DatasetRoutingAuthorityAntigravity(
                f"Other_normal_{index}",
                "Other",
                100,
                None,
                frozenset(),
                None,
                ConfusionAntigravity(0, 0, 0, 0),
                None,
            )
        )
    comparisons = (
        _comparison_antigravity("always_baseline", (609, 66, 837, 70788)),
        _comparison_antigravity("family_wide_local_gap", (611, 63, 835, 70791)),
        _comparison_antigravity("oracle_per_dataset", (618, 64, 828, 70790)),
    )
    frozen = FrozenRoutingAuthorityAntigravity(tuple(rows), tuple(), 0)
    return Exp456ResultAuthorityAntigravity(
        frozen,
        tuple(features),
        comparisons,
        ARTIFACT_MAP_SHA256_ANTIGRAVITY,
        COMPARISON_SHA256_ANTIGRAVITY,
        DECISION_SHA256_ANTIGRAVITY,
        0,
    )


def decisions_antigravity(authority=None, passed_datasets=("ECG5000_normal_2",)):
    from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.router_antigravity import (
        route_ecg_antigravity,
    )

    authority = authority or exp456_authority_antigravity()
    by_dataset = {feature.dataset: feature for feature in authority.features}
    return tuple(
        route_ecg_antigravity(
            by_dataset[dataset], qualification_antigravity(dataset, dataset in passed_datasets)
        )
        for dataset in tuple(f"ECG5000_normal_{index}" for index in range(1, 6))
    )


class EvaluationAntigravityTest(unittest.TestCase):
    def test_reproduces_exp456_comparisons_before_hand_derived_exp457_result_antigravity(self) -> None:
        """Break caught: predecessor math drifts or TEST outcomes influence the two routing gates."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )

        result = evaluate_routes_antigravity(
            exp456_authority_antigravity(), decisions_antigravity()
        )
        self.assertEqual(
            tuple(row.name for row in result.comparisons),
            (
                "always_baseline",
                "family_wide_local_gap",
                "oracle_per_dataset",
                "ecg_train_confidence_router",
            ),
        )
        self.assertEqual(
            tuple(
                (row.confusion.tp, row.confusion.fp, row.confusion.fn, row.confusion.tn, row.f1)
                for row in result.comparisons
            ),
            (
                (609, 66, 837, 70788, 0.5742574257425742),
                (611, 63, 835, 70791, 0.5764150943396227),
                (618, 64, 828, 70790, 0.5808270676691729),
                (617, 66, 829, 70788, 0.5796148426491311),
            ),
        )
        self.assertEqual(
            tuple((row.family, row.fn_delta, row.fp_delta) for row in result.family_metrics),
            (("ECG5000", -8, 0), ("ChlorineConcentration", 0, 0)),
        )
        self.assertEqual(
            tuple((row.metric, row.operator, row.passed) for row in result.criteria),
            (("final_fn", "<=", True), ("final_fp", "<=", True), ("final_f1", ">", True)),
        )
        self.assertTrue(result.passed)
        self.assertEqual(
            result.status, "routing_candidate_qualified_for_external_validation"
        )

    def test_only_normal_two_candidate_plus_fixed_chlorine_is_fn829_fp66_antigravity(self) -> None:
        """Break caught: an ECG equality or a Chlorine baseline route changes the conditional result."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )

        final = evaluate_routes_antigravity(
            exp456_authority_antigravity(), decisions_antigravity()
        ).comparisons[-1]
        self.assertEqual((final.confusion.fn, final.confusion.fp), (829, 66))

    def test_rejects_self_consistent_caller_flip_of_normal_two_train_gate_and_route_antigravity(self) -> None:
        """Break caught: evaluator accepts caller booleans instead of exact Gate 1/router evidence."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )

        authority = exp456_authority_antigravity()
        decisions = decisions_antigravity(authority)
        altered = decisions[:1] + (
            replace(decisions[1], train_gate=False, route="baseline"),
        ) + decisions[2:]
        with self.assertRaises(ValueError):
            evaluate_routes_antigravity(authority, altered)

    def test_rejects_forged_route_structure_gate_or_predecessor_comparison_antigravity(self) -> None:
        """Break caught: caller route text or a rewritten Exp456 aggregate controls evaluation."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )

        authority = exp456_authority_antigravity()
        decisions = decisions_antigravity(authority)
        with self.assertRaises(ValueError):
            evaluate_routes_antigravity(
                authority, (replace(decisions[0], route="candidate"),) + decisions[1:]
            )
        with self.assertRaises(ValueError):
            evaluate_routes_antigravity(
                authority, (replace(decisions[0], structure_gate=True),) + decisions[1:]
            )
        wrong = replace(
            authority,
            predecessor_comparisons=(
                replace(authority.predecessor_comparisons[0], f1=0.0),
            )
            + authority.predecessor_comparisons[1:],
        )
        with self.assertRaises(ValueError):
            evaluate_routes_antigravity(wrong, decisions)
        for changed_feature in (
            replace(authority.features[0], family="ChlorineConcentration"),
            replace(authority.features[0], target_length=101),
        ):
            misaligned = replace(
                authority, features=(changed_feature,) + authority.features[1:]
            )
            with self.subTest(feature=changed_feature), self.assertRaises(ValueError):
                evaluate_routes_antigravity(misaligned, decisions)

    def test_capture_dependency_seam_checks_all_three_identities_and_zero_labels_antigravity(self) -> None:
        """Break caught: a lookalike Exp456 root or a label-reading authority enters Exp457."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
            RoutingEvaluationAntigravity,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity import (
            evaluation_antigravity,
        )

        fixture = exp456_authority_antigravity()
        predecessor = RoutingEvaluationAntigravity(
            fixture.predecessor_comparisons
            + (_comparison_antigravity("lodo_router", (605, 65, 841, 70789)),),
            tuple(),
            tuple(),
            False,
            "routing_candidate_not_qualified_for_external_validation",
        )
        payloads = {
            "artifact_sha256_antigravity.json": b"artifact-map\n",
            "comparison_metrics_antigravity.json": b"comparisons\n",
            "decision_antigravity.json": b"decision\n",
        }
        identities = {
            "artifact_sha256_antigravity.json": sha256(payloads["artifact_sha256_antigravity.json"]).hexdigest(),
            "comparison_metrics_antigravity.json": sha256(payloads["comparison_metrics_antigravity.json"]).hexdigest(),
            "decision_antigravity.json": sha256(payloads["decision_antigravity.json"]).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exp456"
            root.mkdir()
            for name in evaluation_antigravity._EXP456_LEAVES_ANTIGRAVITY:
                (root / name).write_bytes(payloads.get(name, f"{name}\n".encode()))
            captured = evaluation_antigravity._capture_exp456_result_authority_with_dependencies_antigravity(
                root,
                identities,
                lambda _root: predecessor,
                lambda: fixture.authority,
                lambda _authority: fixture.features,
            )
            self.assertEqual(captured.label_query_count, 0)
            self.assertEqual(captured.predecessor_comparisons, fixture.predecessor_comparisons)

            for name in identities:
                wrong = dict(identities)
                wrong[name] = "0" * 64
                with self.subTest(name=name), self.assertRaises(ValueError):
                    evaluation_antigravity._capture_exp456_result_authority_with_dependencies_antigravity(
                        root,
                        wrong,
                        lambda _root: predecessor,
                        lambda: fixture.authority,
                        lambda _authority: fixture.features,
                    )
            labels_opened = replace(fixture.authority, label_query_count=1)
            with self.assertRaises(ValueError):
                evaluation_antigravity._capture_exp456_result_authority_with_dependencies_antigravity(
                    root,
                    identities,
                    lambda _root: predecessor,
                    lambda: labels_opened,
                    lambda _authority: fixture.features,
                )


if __name__ == "__main__":
    unittest.main()
