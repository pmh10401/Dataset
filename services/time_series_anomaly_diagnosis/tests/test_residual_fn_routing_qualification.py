from __future__ import annotations

import unittest

import numpy as np

from services.time_series_anomaly_diagnosis.residual_fn_routing_qualification import (
    EXPANDED_FEATURE_KEYS,
    SCORE_STABILITY_FEATURE_KEYS,
    MethodOutcome,
    ResidualRoutingRecord,
    common_detection_flags,
    evaluate_cached_router_replay,
    extract_score_stability_features,
    leave_family_out_routes,
    select_residual_fn_method,
    summarize_method_outcome,
)
from services.time_series_anomaly_diagnosis.routing_policy import FEATURE_KEYS
from services.time_series_anomaly_diagnosis.anomaly_detection import FIXED_SEEDS
from services.time_series_anomaly_diagnosis.score_sources import METHOD_NAMES


def features(value: float) -> dict[str, float]:
    return dict(zip(FEATURE_KEYS, (10 + value, 40, value, 0, 0, 1, 0.5), strict=True))


def outcomes(*, baseline_fn: int, multirocket_fn: int, hydra_fn: int) -> tuple[MethodOutcome, ...]:
    return (
        MethodOutcome(METHOD_NAMES[0], baseline_fn, 2, 1),
        MethodOutcome(METHOD_NAMES[1], multirocket_fn, 3, 1),
        MethodOutcome(METHOD_NAMES[2], hydra_fn, 2, 1),
    )


def array_record(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype="<f8")
    return {"shape": list(array.shape), "dtype": array.dtype.str, "bytes": array.tobytes().hex()}


def cached_score_row(name: str, value: float) -> dict[str, object]:
    method_tests = {
        METHOD_NAMES[0]: [-1.0, -1.0],
        METHOD_NAMES[1]: [-1.0, 2.0],
        METHOD_NAMES[2]: [2.0, 2.0],
    }
    return {
        "dataset": name,
        "features": features(value),
        "metadata": {"train_count": 2, "test_count": 2, "test_indices": [0, 1]},
        "scores": {
            method: {
                str(seed): {
                    "feature_dimension": 1,
                    "seed": seed,
                    "train_oof": array_record([0.0, 1.0]),
                    "test_scores": array_record(method_tests[method]),
                }
                for seed in FIXED_SEEDS
            }
            for method in METHOD_NAMES
        },
    }


class CommonDecisionRuleTest(unittest.TestCase):
    def test_uses_final_p_floor_for_review_without_overlapping_automatic_detection(self) -> None:
        automatic, review = common_detection_flags(
            np.array([0.001, 0.005, 0.01, 0.02]),
            train_count=99,
        )

        np.testing.assert_array_equal(automatic, [True, True, False, False])
        np.testing.assert_array_equal(review, [False, False, True, False])

    def test_counts_residual_fn_and_both_fp_guardrails(self) -> None:
        outcome = summarize_method_outcome(
            METHOD_NAMES[0],
            labels=np.array([0, 0, 1, 1]),
            automatic=np.array([False, True, False, False]),
            review=np.array([True, False, True, False]),
        )

        self.assertEqual(outcome, MethodOutcome(METHOD_NAMES[0], 1, 2, 1))

    def test_extracts_signed_seed_rank_agreement_and_normalized_upper_tail_width(self) -> None:
        row = cached_score_row("dataset", 0)
        row["metadata"]["train_count"] = 10
        for method in METHOD_NAMES:
            for seed in FIXED_SEEDS:
                row["scores"][method][str(seed)]["train_oof"] = array_record(list(range(10)))
        row["scores"][METHOD_NAMES[2]][str(FIXED_SEEDS[1])]["train_oof"] = array_record(
            list(reversed(range(10)))
        )

        extracted = extract_score_stability_features(row)

        self.assertAlmostEqual(extracted[f"{METHOD_NAMES[0]}_seed_rank_agreement"], 1.0)
        self.assertAlmostEqual(extracted[f"{METHOD_NAMES[2]}_seed_rank_agreement"], -1.0)
        self.assertAlmostEqual(
            extracted[f"{METHOD_NAMES[0]}_upper_tail_width"],
            0.21853500607041687,
        )


class ResidualFnRouterTest(unittest.TestCase):
    def test_includes_every_group_tied_at_the_neighbor_distance_boundary(self) -> None:
        records = []
        for group in range(40):
            pattern = group // 5
            signs = tuple(1.0 if pattern & (1 << bit) else -1.0 for bit in range(3))
            records.append(
                ResidualRoutingRecord(
                    group,
                    dict(zip(FEATURE_KEYS, (0.0, 0.0, *signs, 0.0, 0.0), strict=True)),
                    (
                        MethodOutcome(METHOD_NAMES[0], 1, 0, 0),
                        MethodOutcome(METHOD_NAMES[1], 0, 0, 0),
                        MethodOutcome(METHOD_NAMES[2], 2, 0, 0),
                    ),
                    METHOD_NAMES[0],
                )
            )

        decision = select_residual_fn_method(
            records,
            dict.fromkeys(FEATURE_KEYS, 0.0),
            fallback_method=METHOD_NAMES[0],
        )

        self.assertEqual(decision.support, 40)

    def test_selects_lower_fn_method_only_when_both_fp_totals_do_not_increase(self) -> None:
        records = tuple(
            ResidualRoutingRecord(
                group_id=group,
                features=features(group / 100),
                outcomes=outcomes(baseline_fn=3, multirocket_fn=2, hydra_fn=1),
                baseline_method=METHOD_NAMES[0],
            )
            for group in range(20)
        )

        decision = select_residual_fn_method(
            records, features(0), fallback_method=METHOD_NAMES[0]
        )

        self.assertEqual(decision.selected_method, METHOD_NAMES[2])
        self.assertEqual(decision.support, 20)
        evidence = {row.method_name: row for row in decision.evidence}
        self.assertFalse(evidence[METHOD_NAMES[1]].passed)
        self.assertTrue(evidence[METHOD_NAMES[2]].passed)

    def test_expanded_train_features_change_neighbors_without_changing_the_gate(self) -> None:
        basic_rows = []
        expanded_rows = []
        for group in range(40):
            method_outcomes = (
                MethodOutcome(METHOD_NAMES[0], 1, 0, 0),
                MethodOutcome(METHOD_NAMES[1], 2 if group < 20 else 0, 0, 0),
                MethodOutcome(METHOD_NAMES[2], 2, 0, 0),
            )
            basic = features(group / 100)
            expanded = {
                **basic,
                **{key: 10.0 if group < 20 else 0.0 for key in SCORE_STABILITY_FEATURE_KEYS},
            }
            basic_rows.append(
                ResidualRoutingRecord(group, basic, method_outcomes, METHOD_NAMES[0])
            )
            expanded_rows.append(
                ResidualRoutingRecord(group, expanded, method_outcomes, METHOD_NAMES[0])
            )

        original = select_residual_fn_method(
            basic_rows, features(0), fallback_method=METHOD_NAMES[0]
        )
        expanded = select_residual_fn_method(
            expanded_rows,
            {**features(0), **{key: 0.0 for key in SCORE_STABILITY_FEATURE_KEYS}},
            fallback_method=METHOD_NAMES[0],
        )

        self.assertEqual(original.selected_method, METHOD_NAMES[0])
        self.assertEqual(expanded.selected_method, METHOD_NAMES[1])

    def test_leave_family_out_route_ignores_its_own_outcomes_and_input_order(self) -> None:
        names = tuple(f"dataset-{index:02d}" for index in range(21))
        feature_rows = {name: features(index / 100) for index, name in enumerate(names)}
        family_rows = {name: f"family-{index:02d}" for index, name in enumerate(names)}
        safe_candidate = (
            MethodOutcome(METHOD_NAMES[0], 1, 0, 0),
            MethodOutcome(METHOD_NAMES[1], 0, 0, 0),
            MethodOutcome(METHOD_NAMES[2], 0, 1, 0),
        )
        outcome_rows = {name: safe_candidate for name in names}
        baseline_rows = {name: METHOD_NAMES[0] for name in names}

        expected = leave_family_out_routes(
            feature_rows, outcome_rows, family_rows, baseline_rows
        )
        changed = dict(outcome_rows)
        changed[names[0]] = (
            MethodOutcome(METHOD_NAMES[0], 0, 0, 0),
            MethodOutcome(METHOD_NAMES[1], 99, 99, 99),
            MethodOutcome(METHOD_NAMES[2], 0, 0, 0),
        )
        reversed_routes = leave_family_out_routes(
            dict(reversed(feature_rows.items())),
            dict(reversed(outcome_rows.items())),
            dict(reversed(family_rows.items())),
            dict(reversed(baseline_rows.items())),
        )

        self.assertEqual(expected[names[0]], METHOD_NAMES[1])
        self.assertEqual(
            leave_family_out_routes(
                feature_rows, changed, family_rows, baseline_rows
            )[names[0]],
            expected[names[0]],
        )
        self.assertEqual(reversed_routes, expected)

    def test_compares_neighbor_methods_with_each_neighbors_current_baseline(self) -> None:
        records = tuple(
            ResidualRoutingRecord(
                group,
                features(group / 100),
                (
                    MethodOutcome(METHOD_NAMES[0], 1, 0, 0),
                    MethodOutcome(METHOD_NAMES[1], 0, 0, 0),
                    MethodOutcome(METHOD_NAMES[2], 0, 0, 0),
                ),
                METHOD_NAMES[2],
            )
            for group in range(20)
        )

        decision = select_residual_fn_method(
            records, features(0), fallback_method=METHOD_NAMES[2]
        )

        self.assertEqual(decision.selected_method, METHOD_NAMES[2])
        self.assertFalse(
            next(row for row in decision.evidence if row.method_name == METHOD_NAMES[1]).passed
        )

    def test_dataset_renaming_cannot_change_equal_distance_route(self) -> None:
        target = "target"
        original_names = [
            *[f"a-good-{index:02d}" for index in range(16)],
            *[f"z-bad-{index:02d}" for index in range(16)],
            target,
        ]
        renamed_names = [
            *[f"z-good-{index:02d}" for index in range(16)],
            *[f"a-bad-{index:02d}" for index in range(16)],
            target,
        ]

        def replay(names: list[str], good_names: set[str]) -> str:
            feature_rows = {name: features(0) for name in names}
            family_rows = {
                name: "target-family" if name == target else f"family-{index:02d}"
                for index, name in enumerate(names)
            }
            outcome_rows = {
                name: (
                    MethodOutcome(METHOD_NAMES[0], 1, 0, 0),
                    MethodOutcome(
                        METHOD_NAMES[1],
                        0 if name in good_names else (1 if name == target else 2),
                        0,
                        0,
                    ),
                    MethodOutcome(METHOD_NAMES[2], 2, 0, 0),
                )
                for name in names
            }
            baseline_rows = {name: METHOD_NAMES[0] for name in names}
            return leave_family_out_routes(
                feature_rows, outcome_rows, family_rows, baseline_rows
            )[target]

        original_good = set(original_names[:16])
        renamed_good = set(renamed_names[:16])
        self.assertEqual(replay(original_names, original_good), METHOD_NAMES[0])
        self.assertEqual(replay(renamed_names, renamed_good), METHOD_NAMES[0])

    def test_cached_replay_uses_common_rule_and_qualifies_strict_fn_improvement(self) -> None:
        names = tuple(f"dataset-{index:02d}" for index in range(21))
        score_rows = {name: cached_score_row(name, index / 100) for index, name in enumerate(names)}
        labels = {name: {0: 0, 1: 1} for name in names}
        families = {name: f"family-{index:02d}" for index, name in enumerate(names)}
        baseline = {name: METHOD_NAMES[0] for name in names}

        result = evaluate_cached_router_replay(score_rows, labels, families, baseline)

        self.assertTrue(result["qualified"])
        self.assertEqual(result["baseline_metrics"].residual_fn_total, 21)
        self.assertEqual(result["candidate_metrics"].residual_fn_total, 0)
        self.assertEqual(result["candidate_metrics"].review_inclusive_fp_total, 0)
        self.assertEqual(set(result["routes"].values()), {METHOD_NAMES[1]})

        expanded = evaluate_cached_router_replay(
            score_rows,
            labels,
            families,
            baseline,
            include_score_stability=True,
        )
        self.assertEqual(expanded["feature_keys"], EXPANDED_FEATURE_KEYS)


if __name__ == "__main__":
    unittest.main()
