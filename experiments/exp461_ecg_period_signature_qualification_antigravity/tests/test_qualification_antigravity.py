from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np


PRIMARY_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
METHOD_ORDER_ANTIGRAVITY = (
    "hydra_local_gap",
    "period_signature_local_gap",
    "hydra_period_rank_max",
)


def train_antigravity() -> np.ndarray:
    train = np.linspace(0.0, 1.0, 1413 * 140, dtype=np.float32).reshape(1413, 140)
    train.setflags(write=False)
    return train


def branch_scores_antigravity(
    query_size: int,
    *,
    fail_period: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    original = np.arange(query_size, dtype=np.float64)
    primary = {
        change_type: original + float((index + 1) * query_size)
        for index, change_type in enumerate(PRIMARY_TYPES_ANTIGRAVITY)
    }
    if fail_period:
        primary["period_change"] = original.copy()
    return original, primary


def fold_scorer_antigravity(
    _train: np.ndarray,
    _fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
):
    from experiments.exp461_ecg_period_signature_qualification_antigravity.scoring_antigravity import (
        FoldMethodScoresAntigravity,
    )

    query_size = len(query_indices)
    originals: dict[str, np.ndarray] = {}
    primary: dict[str, dict[str, np.ndarray]] = {}
    for method in METHOD_ORDER_ANTIGRAVITY:
        fail_period = (
            method == "hydra_local_gap"
            and (seed, fold_index) in {(20260721, 3), (20261721, 4)}
        ) or (method == "period_signature_local_gap" and seed == 20260721 and fold_index == 0)
        originals[method], primary[method] = branch_scores_antigravity(
            query_size,
            fail_period=fail_period,
        )
    return FoldMethodScoresAntigravity(
        methods=METHOD_ORDER_ANTIGRAVITY,
        original=originals,
        primary=primary,
        feature_dimensions={
            "hydra_local_gap": 512,
            "period_signature_local_gap": 120,
            "hydra_period_rank_max": 2,
        },
        provenance={
            "hydra_local_gap": "exp459_hydra_local_gap",
            "period_signature_local_gap": "exp461_period_signature_local_gap",
            "hydra_period_rank_max": "exp461_branch_rank_max",
        },
    )


def qualification_antigravity():
    from experiments.exp461_ecg_period_signature_qualification_antigravity.predecessor_antigravity import (
        capture_exp460_period_authority_antigravity,
    )
    from experiments.exp461_ecg_period_signature_qualification_antigravity.qualification_antigravity import (
        _qualify_methods_with_dependencies_antigravity,
    )

    authority = capture_exp460_period_authority_antigravity()
    return _qualify_methods_with_dependencies_antigravity(
        authority=authority,
        loader=train_antigravity,
        fold_scorer=fold_scorer_antigravity,
    )


def qualification_with_primary_auc_antigravity(seed: int, fold_index: int, auc: float):
    qualification = qualification_antigravity()
    rows = list(qualification.fold_metrics)
    for index, row in enumerate(rows):
        if row.method == "hydra_period_rank_max" and row.seed == seed and row.fold_index == fold_index:
            normal_scores = tuple(row.normal_scores)
            primary_scores = dict(row.primary_scores)
            if auc <= 0.5:
                primary_scores["period_change"] = normal_scores
            else:
                primary_scores["period_change"] = tuple(
                    score + len(normal_scores) for score in normal_scores
                )
            per_type_auc = dict(row.per_type_auc)
            per_type_auc["period_change"] = auc
            rows[index] = replace(
                row,
                per_type_auc=per_type_auc,
                pooled_auc=sum(per_type_auc.values()) / len(per_type_auc),
                primary_scores=primary_scores,
            )
            break
    return replace(qualification, fold_metrics=tuple(rows))


class QualificationAntigravityTest(unittest.TestCase):
    def test_qualification_preserves_exact_method_seed_fold_and_type_order_antigravity(self) -> None:
        """Break caught: qualification drops a method/fold or changes frozen row/type order."""
        result = qualification_antigravity()

        self.assertEqual(result.dataset, "ECG5000_normal_2")
        self.assertEqual(result.method_order, METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(result.primary_candidate, "hydra_period_rank_max")
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
                self.assertEqual(tuple(row.per_type_auc), PRIMARY_TYPES_ANTIGRAVITY)
                self.assertTrue(np.isfinite(tuple(row.per_type_auc.values())).all())
                self.assertGreaterEqual(row.normal_trials, 282)
                self.assertLessEqual(row.normal_wilson_upper, 0.05)
                self.assertTrue(row.scores_finite)

    def test_primary_candidate_requires_ten_passes_and_both_historical_improvements_antigravity(
        self,
    ) -> None:
        """Break caught: primary candidate qualifies with a missed fold or non-strict historical gain."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.predecessor_antigravity import (
            capture_exp460_period_authority_antigravity,
        )

        authority = capture_exp460_period_authority_antigravity()
        result = qualification_antigravity()
        decision = evaluate_qualification_antigravity(authority, result)

        primary_rows = [row for row in result.fold_metrics if row.method == "hydra_period_rank_max"]
        self.assertEqual(sum(row.passed for row in primary_rows), 10)
        self.assertTrue(decision.eligible_for_routing_research)
        self.assertFalse(decision.routing_applied)
        self.assertEqual(decision.status, "qualification_passed_ready_for_task5_reproducibility")

        not_improved = qualification_with_primary_auc_antigravity(
            20260721,
            3,
            0.5,
        )
        failed = evaluate_qualification_antigravity(authority, not_improved)
        self.assertFalse(failed.eligible_for_routing_research)
        self.assertIn(
            "historical_period_change_not_strictly_improved:20260721:3",
            failed.failure_reasons,
        )


if __name__ == "__main__":
    unittest.main()
