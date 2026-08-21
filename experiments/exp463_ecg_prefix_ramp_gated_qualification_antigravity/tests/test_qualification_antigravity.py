from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np


METHODS_ANTIGRAVITY = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


def train_antigravity() -> np.ndarray:
    values = np.linspace(0.0, 1.0, 1413 * 140, dtype=np.float32).reshape(1413, 140)
    values.setflags(write=False)
    return values


def fold_scorer_antigravity(
    _train: np.ndarray,
    _fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
):
    from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.scoring_antigravity import (
        FoldMethodScoresAntigravity,
    )

    del seed, fold_index
    size = len(query_indices)
    normal = np.linspace(0.0, 1.0, size, dtype=np.float64)
    primary = {
        change_type: normal + 10.0 + index
        for index, change_type in enumerate(TYPES_ANTIGRAVITY)
    }
    return FoldMethodScoresAntigravity(
        methods=METHODS_ANTIGRAVITY,
        original={method: normal for method in METHODS_ANTIGRAVITY},
        primary={method: primary for method in METHODS_ANTIGRAVITY},
        feature_dimensions={method: 1 for method in METHODS_ANTIGRAVITY},
        provenance={method: "synthetic_task4_score_evidence" for method in METHODS_ANTIGRAVITY},
    )


def qualification_antigravity():
    from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
        capture_exp462_authority_antigravity,
    )
    from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.qualification_antigravity import (
        _qualify_methods_with_dependencies_antigravity,
    )

    return _qualify_methods_with_dependencies_antigravity(
        authority=capture_exp462_authority_antigravity(),
        loader=train_antigravity,
        fold_scorer=fold_scorer_antigravity,
    )


class QualificationAntigravityTest(unittest.TestCase):
    def test_qualification_emits_exact_ordered_70_rows_and_seven_summaries_antigravity(self) -> None:
        """Break caught: a method, seed, fold, or summary is silently dropped."""
        result = qualification_antigravity()
        self.assertEqual(result.method_order, METHODS_ANTIGRAVITY)
        self.assertEqual(len(result.fold_metrics), 70)
        self.assertEqual(
            tuple((row.method, row.seed, row.fold_index) for row in result.fold_metrics),
            tuple(
                (method, seed, fold)
                for method in METHODS_ANTIGRAVITY
                for seed in (20260721, 20261721)
                for fold in range(5)
            ),
        )
        self.assertEqual(tuple(summary.method for summary in result.summaries), METHODS_ANTIGRAVITY)
        for row in result.fold_metrics:
            self.assertEqual(tuple(row.per_type_auc), TYPES_ANTIGRAVITY)
            self.assertEqual(tuple(row.primary_scores), TYPES_ANTIGRAVITY)
            self.assertTrue(row.scores_finite)

    def test_normalization_recomputes_metrics_and_ignores_caller_pass_boolean_antigravity(self) -> None:
        """Break caught: stored AUCs or pass flags can forge the M2 decision."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )

        qualification = qualification_antigravity()
        rows = list(qualification.fold_metrics)
        index = next(
            i
            for i, row in enumerate(rows)
            if row.method == "M2" and row.seed == 20260721 and row.fold_index == 3
        )
        row = rows[index]
        forged = replace(
            row,
            passed=True,
            failure_reasons=(),
            primary_scores={
                **row.primary_scores,
                "period_change": row.normal_scores,
            },
        )
        rows[index] = forged
        altered = replace(qualification, fold_metrics=tuple(rows))
        decision = evaluate_qualification_antigravity(
            capture_exp462_authority_antigravity(), altered
        )
        self.assertFalse(decision.eligible_for_routing_research)
        self.assertIn("primary_candidate_not_10_of_10", decision.failure_reasons)
        self.assertIn("period_change_auc_below_global_min", decision.failure_reasons)
        self.assertIn("historical_period_change_below_min:20260721:3", decision.failure_reasons)

    def test_m0_m6_control_reproduction_is_required_antigravity(self) -> None:
        """Break caught: Exp463 silently accepts drifted predecessor controls."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.qualification_antigravity import (
            validate_control_reproduction_antigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )

        self.assertEqual(
            validate_control_reproduction_antigravity(
                qualification_antigravity(), capture_exp462_authority_antigravity()
            ),
            ("m0_control_not_reproduced", "m6_control_not_reproduced"),
        )


if __name__ == "__main__":
    unittest.main()
