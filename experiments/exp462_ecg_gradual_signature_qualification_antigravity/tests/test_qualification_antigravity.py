from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np


METHOD_ORDER_ANTIGRAVITY = (
    "hydra_period_rank_max",
    "gradual_signature_local_gap",
    "hydra_period_gradual_rank_max",
)
CHANGE_ORDER_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


def train_antigravity() -> np.ndarray:
    train = np.linspace(0.0, 1.0, 1413 * 140, dtype=np.float32).reshape(1413, 140)
    train.setflags(write=False)
    return train


def fold_scorer_antigravity(
    _train: np.ndarray,
    _fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
):
    from experiments.exp462_ecg_gradual_signature_qualification_antigravity.scoring_antigravity import (
        FoldMethodScoresAntigravity,
    )

    del seed, fold_index
    query_size = len(query_indices)
    original = np.arange(query_size, dtype=np.float64)
    originals = {method: original for method in METHOD_ORDER_ANTIGRAVITY}
    primary = {
        method: {
            change_type: original + 1000.0 + float((method_index + 1) * (change_index + 2))
            for change_index, change_type in enumerate(CHANGE_ORDER_ANTIGRAVITY)
        }
        for method_index, method in enumerate(METHOD_ORDER_ANTIGRAVITY)
    }
    return FoldMethodScoresAntigravity(
        methods=METHOD_ORDER_ANTIGRAVITY,
        original=originals,
        primary=primary,
        feature_dimensions={method: 2 for method in METHOD_ORDER_ANTIGRAVITY},
        provenance={method: "synthetic_dependency_injected" for method in METHOD_ORDER_ANTIGRAVITY},
    )


def qualification_antigravity():
    from experiments.exp462_ecg_gradual_signature_qualification_antigravity.predecessor_antigravity import (
        capture_exp461_gradual_authority_antigravity,
    )
    from experiments.exp462_ecg_gradual_signature_qualification_antigravity.qualification_antigravity import (
        _qualify_methods_with_dependencies_antigravity,
    )

    return _qualify_methods_with_dependencies_antigravity(
        authority=capture_exp461_gradual_authority_antigravity(),
        loader=train_antigravity,
        fold_scorer=fold_scorer_antigravity,
    )


class QualificationAntigravityTest(unittest.TestCase):
    def test_qualification_has_exact_method_seed_fold_and_type_order_antigravity(self) -> None:
        """Break caught: a qualification drops a frozen method/fold/type record."""
        result = qualification_antigravity()
        self.assertEqual(result.dataset, "ECG5000_normal_2")
        self.assertEqual(result.method_order, METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(result.primary_candidate, "hydra_period_gradual_rank_max")
        self.assertEqual(len(result.fold_metrics), 30)
        self.assertEqual(
            tuple((row.method, row.seed, row.fold_index) for row in result.fold_metrics),
            tuple(
                (method, seed, fold)
                for method in METHOD_ORDER_ANTIGRAVITY
                for seed in (20260721, 20261721)
                for fold in range(5)
            ),
        )
        for row in result.fold_metrics:
            with self.subTest(method=row.method, seed=row.seed, fold=row.fold_index):
                self.assertEqual(tuple(row.per_type_auc), CHANGE_ORDER_ANTIGRAVITY)
                self.assertTrue(row.scores_finite)
                self.assertLessEqual(row.normal_wilson_upper, 0.05)
                self.assertTrue(all(value >= 0.85 for value in row.per_type_auc.values()))
                self.assertGreaterEqual(row.pooled_auc, 0.90)

        primary = [
            row
            for row in result.fold_metrics
            if row.method == "hydra_period_gradual_rank_max"
        ]
        self.assertEqual(sum(row.passed for row in primary), 10)

    def test_normalization_recomputes_metrics_and_pass_reasons_from_scores_antigravity(self) -> None:
        """Break caught: caller booleans/metrics can forge a qualification decision."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.predecessor_antigravity import (
            capture_exp461_gradual_authority_antigravity,
        )

        qualification = qualification_antigravity()
        rows = list(qualification.fold_metrics)
        index = next(
            i
            for i, row in enumerate(rows)
            if row.method == "hydra_period_gradual_rank_max"
            and row.seed == 20260721
            and row.fold_index == 3
        )
        forged = replace(
            rows[index],
            per_type_auc={**rows[index].per_type_auc, "period_change": 0.5},
            pooled_auc=sum(
                value
                for name, value in rows[index].per_type_auc.items()
                if name != "period_change"
            ) / 4.0 + 0.5 / 4.0,
            passed=True,
            failure_reasons=(),
            primary_scores={
                **rows[index].primary_scores,
                "period_change": rows[index].normal_scores,
            },
        )
        rows[index] = forged
        altered = replace(qualification, fold_metrics=tuple(rows), summaries=qualification.summaries)
        decision = evaluate_qualification_antigravity(
            capture_exp461_gradual_authority_antigravity(), altered
        )
        self.assertFalse(decision.eligible_for_routing_research)
        self.assertEqual(decision.primary_passed_count, 9)
        self.assertIn("period_change_auc_below_min", decision.failure_reasons)
        self.assertIn(
            "historical_period_change_below_min:20260721:3",
            decision.failure_reasons,
        )


if __name__ == "__main__":
    unittest.main()
