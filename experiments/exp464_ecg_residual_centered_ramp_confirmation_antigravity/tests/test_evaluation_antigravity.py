from __future__ import annotations

from dataclasses import replace
import unittest

from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_qualification_antigravity import (
    TYPES_ANTIGRAVITY,
    _replace_c6_row_with_numeric_evidence_antigravity,
    numeric_boundary_qualification_antigravity,
    qualification_antigravity,
)


class EvaluationAntigravityTest(unittest.TestCase):
    def test_decision_is_fixed_c6_only_and_blocks_before_routing_research_antigravity(self) -> None:
        """Break caught: Task 4 promotes another method or opens routing before review."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        qualification = qualification_antigravity()
        decision = evaluate_qualification_antigravity(qualification, authority)
        self.assertEqual(decision.primary_candidate, "template_residual_centered_ramp_confirmed_max")
        self.assertEqual(decision.primary_required_count, 10)
        self.assertEqual(decision.primary_passed_count, 10)
        self.assertEqual(decision.failure_reasons, ())
        self.assertTrue(decision.eligible_for_routing_research)
        self.assertFalse(decision.routing_applied)
        self.assertFalse(decision.test_labels_opened)
        self.assertEqual(decision.label_query_count, 0)
        self.assertEqual(
            decision.status,
            "qualification_passed_ready_for_task5_reproducibility",
        )

    def test_decision_recomputes_c6_gates_from_score_evidence_antigravity(self) -> None:
        """Break caught: caller summaries or booleans bypass the frozen C6 gates."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        qualification = qualification_antigravity()
        expected_decision = evaluate_qualification_antigravity(qualification, authority)
        rows = list(qualification.fold_metrics)
        index = next(
            i
            for i, row in enumerate(rows)
            if row.method == "C6" and row.seed == 20261721 and row.fold_index == 4
        )
        row = rows[index]
        rows[index] = replace(
            row,
            passed=True,
            failure_reasons=(),
        )
        decision = evaluate_qualification_antigravity(
            replace(qualification, fold_metrics=tuple(rows)),
            authority,
        )
        self.assertEqual(decision, expected_decision)

    def test_source_free_c6_gate_matrix_covers_every_frozen_decision_boundary_antigravity(self) -> None:
        """Break caught: real numeric C6 gates drift, disappear, or change reason."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity import (
            normalize_qualification_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        boundary = numeric_boundary_qualification_antigravity()

        def promoted_antigravity(
            *,
            seed: int,
            fold_index: int,
            target: str,
            normal_count: int,
            target_count: int = 0,
        ) -> dict[str, tuple[int, ...]]:
            c0 = next(
                row
                for row in boundary.fold_metrics
                if row.method == "C0" and row.seed == seed and row.fold_index == fold_index
            )
            return {
                "normal": tuple(range(normal_count)),
                **{
                    change_type: tuple(
                        range(
                            target_count
                            if change_type == target
                            else len(c0.primary_scores[change_type])
                        )
                    )
                    for change_type in TYPES_ANTIGRAVITY
                },
            }

        def changed_antigravity(
            *,
            seed: int,
            fold_index: int,
            promoted_indices: dict[str, tuple[int, ...]],
        ):
            return _replace_c6_row_with_numeric_evidence_antigravity(
                boundary,
                seed=seed,
                fold_index=fold_index,
                promoted_indices=promoted_indices,
            )

        passing = normalize_qualification_antigravity(boundary, authority)
        passing_summary = next(summary for summary in passing.summaries if summary.method == "C6")
        self.assertEqual(passing_summary.min_gradual_drift_auc, 0.8500955686333685)
        self.assertEqual(passing_summary.min_period_change_auc, 0.907392736783864)
        for seed, fold_index, expected_period in (
            (20260721, 3, 0.907392736783864),
            (20261721, 4, 0.9216148584075248),
        ):
            row = next(
                row
                for row in passing.fold_metrics
                if row.method == "C6" and row.seed == seed and row.fold_index == fold_index
            )
            self.assertEqual(row.per_type_auc["period_change"], expected_period)
            self.assertEqual(row.failure_reasons, ())
        self.assertEqual(
            evaluate_qualification_antigravity(boundary, authority).failure_reasons,
            (),
        )

        cases = (
            (
                "gradual_per_fold",
                20260721,
                0,
                promoted_antigravity(seed=20260721, fold_index=0, target="gradual_drift", normal_count=1),
                "gradual_drift",
                0.8634394236411992,
                ("gradual_drift_auc_below_c0",),
                ("primary_candidate_not_10_of_10", "primary_gradual_drift_below_c0_on_some_fold"),
            ),
            (
                "period_per_fold",
                20260721,
                0,
                promoted_antigravity(seed=20260721, fold_index=0, target="period_change", normal_count=1),
                "period_change",
                0.924402851827342,
                ("period_change_auc_below_c0",),
                ("primary_candidate_not_10_of_10", "primary_period_change_below_c0_on_some_fold"),
            ),
            (
                "global_period_floor",
                20260721,
                0,
                promoted_antigravity(seed=20260721, fold_index=0, target="period_change", normal_count=7, target_count=2),
                "period_change",
                0.907371798873753,
                ("period_change_auc_below_c0",),
                (
                    "primary_candidate_not_10_of_10",
                    "primary_period_change_below_c0_on_some_fold",
                    "primary_period_change_min_below_exp463_global_floor",
                ),
            ),
            (
                "historical_20260721_fold3",
                20260721,
                3,
                promoted_antigravity(seed=20260721, fold_index=3, target="period_change", normal_count=1),
                "period_change",
                0.9039032241838941,
                ("historical_period_change_below_min:20260721:3", "period_change_auc_below_c0"),
                (
                    "historical_period_change_below_min:20260721:3",
                    "primary_candidate_not_10_of_10",
                    "primary_period_change_below_c0_on_some_fold",
                    "primary_period_change_min_below_exp463_global_floor",
                ),
            ),
            (
                "historical_20261721_fold4",
                20261721,
                4,
                promoted_antigravity(seed=20261721, fold_index=4, target="period_change", normal_count=1),
                "period_change",
                0.9181001961671948,
                ("historical_period_change_below_min:20261721:4", "period_change_auc_below_c0"),
                (
                    "historical_period_change_below_min:20261721:4",
                    "primary_candidate_not_10_of_10",
                    "primary_period_change_below_c0_on_some_fold",
                ),
            ),
            (
                "strict_gradual_floor",
                20261721,
                3,
                {},
                "gradual_drift",
                0.8402306222021025,
                ("gradual_drift_auc_below_min",),
                (
                    "primary_candidate_not_10_of_10",
                    "primary_gradual_drift_auc_below_min",
                    "primary_gradual_drift_min_below_0.85",
                    "primary_gradual_drift_not_strictly_above_exp463_c0",
                ),
            ),
            (
                "level_absolute_floor",
                20260721,
                0,
                promoted_antigravity(seed=20260721, fold_index=0, target="level_shift", normal_count=40),
                "level_shift",
                0.849754647954151,
                ("level_shift_auc_below_min", "period_change_auc_below_c0"),
                (
                    "primary_candidate_not_10_of_10",
                    "primary_level_shift_auc_below_min",
                    "primary_period_change_below_c0_on_some_fold",
                ),
            ),
            (
                "gradual_absolute_floor",
                20260721,
                3,
                promoted_antigravity(seed=20260721, fold_index=3, target="gradual_drift", normal_count=1),
                "gradual_drift",
                0.8492593430913938,
                ("gradual_drift_auc_below_c0", "gradual_drift_auc_below_min"),
                (
                    "primary_candidate_not_10_of_10",
                    "primary_gradual_drift_auc_below_min",
                    "primary_gradual_drift_below_c0_on_some_fold",
                    "primary_gradual_drift_min_below_0.85",
                ),
            ),
            (
                "amplitude_absolute_floor",
                20260721,
                0,
                promoted_antigravity(seed=20260721, fold_index=0, target="amplitude_scaling", normal_count=49),
                "amplitude_scaling",
                0.8492989049682228,
                ("amplitude_scaling_auc_below_min", "mean_auc_below_min", "period_change_auc_below_c0"),
                (
                    "primary_amplitude_scaling_auc_below_min",
                    "primary_candidate_not_10_of_10",
                    "primary_mean_auc_below_0.90",
                    "primary_period_change_below_c0_on_some_fold",
                ),
            ),
            (
                "period_absolute_floor",
                20260721,
                0,
                promoted_antigravity(seed=20260721, fold_index=0, target="period_change", normal_count=27),
                "period_change",
                0.8481314537576946,
                ("period_change_auc_below_c0", "period_change_auc_below_min"),
                (
                    "primary_candidate_not_10_of_10",
                    "primary_period_change_auc_below_min",
                    "primary_period_change_below_c0_on_some_fold",
                    "primary_period_change_min_below_exp463_global_floor",
                ),
            ),
            (
                "mean_floor",
                20260721,
                0,
                {
                    "normal": tuple(range(27)),
                    "gradual_drift": tuple(range(136)),
                    "period_change": tuple(range(222)),
                },
                "mean_auc",
                0.8989780119616926,
                ("mean_auc_below_min",),
                ("primary_candidate_not_10_of_10", "primary_mean_auc_below_0.90"),
            ),
        )

        for name, seed, fold_index, promoted_indices, metric, expected_value, expected_row_reasons, expected_reasons in cases:
            with self.subTest(case=name):
                qualification = changed_antigravity(
                    seed=seed,
                    fold_index=fold_index,
                    promoted_indices=promoted_indices,
                )
                normalized = normalize_qualification_antigravity(qualification, authority)
                row = next(
                    row
                    for row in normalized.fold_metrics
                    if row.method == "C6" and row.seed == seed and row.fold_index == fold_index
                )
                actual_value = row.mean_auc if metric == "mean_auc" else row.per_type_auc[metric]
                self.assertEqual(actual_value, expected_value)
                self.assertEqual(row.failure_reasons, expected_row_reasons)
                decision = evaluate_qualification_antigravity(qualification, authority)
                self.assertEqual(decision.failure_reasons, expected_reasons)
                self.assertEqual(decision.status, "qualification_failed_blocked_before_routing_research")
                self.assertFalse(decision.eligible_for_routing_research)


if __name__ == "__main__":
    unittest.main()
