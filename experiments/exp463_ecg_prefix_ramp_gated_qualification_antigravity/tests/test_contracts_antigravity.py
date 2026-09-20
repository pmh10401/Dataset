from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from types import MappingProxyType, SimpleNamespace
import unittest


class ContractsAntigravityTest(unittest.TestCase):
    def test_future_contract_inventory_is_immutable_and_exactly_70_by_7_antigravity(self) -> None:
        """Break caught: future metric containers lose order, coverage, or immutability."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.contracts_antigravity import (
            Exp463MethodFoldMetricAntigravity,
            Exp463MethodSummaryAntigravity,
            Exp463QualificationContractAntigravity,
        )

        self.assertEqual(
            tuple(field.name for field in fields(Exp463MethodFoldMetricAntigravity)),
            (
                "method", "seed", "fold_index", "normal_trials", "normal_alerts",
                "normal_wilson_upper", "per_type_auc", "pooled_auc", "scores_finite",
                "normal_scores", "primary_scores", "passed", "failure_reasons",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp463MethodSummaryAntigravity)),
            (
                "method", "fold_count", "passed_count", "failed_count",
                "min_per_type_auc", "min_gradual_drift_auc", "min_period_change_auc",
                "min_pooled_auc", "max_normal_wilson_upper", "all_scores_finite", "passed",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp463QualificationContractAntigravity)),
            ("dataset", "method_order", "primary_candidate", "fold_metrics", "summaries", "label_query_count"),
        )
        self.assertEqual(Exp463QualificationContractAntigravity.expected_metric_row_count_antigravity(), 70)
        self.assertEqual(Exp463QualificationContractAntigravity.expected_summary_count_antigravity(), 7)

        # Construction is intentionally not a scoring/evaluation implementation;
        # the contracts only require exact container shape for future tasks.
        self.assertTrue(issubclass(Exp463MethodFoldMetricAntigravity, object))
        self.assertTrue(issubclass(Exp463MethodSummaryAntigravity, object))
        self.assertTrue(issubclass(Exp463QualificationContractAntigravity, object))

    def test_exp462_snapshot_contract_exposes_30_rows_three_summaries_seven_leaves_antigravity(self) -> None:
        """Break caught: Exp463 loses exact Exp462 result inventory or primary binding."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.contracts_antigravity import (
            Exp462ResultSnapshotAntigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )

        authority = capture_exp462_authority_antigravity()
        snapshot = authority.result_snapshot
        self.assertIsInstance(snapshot, Exp462ResultSnapshotAntigravity)
        self.assertEqual(snapshot.row_count, 30)
        self.assertEqual(snapshot.failed_row_count, 18)
        self.assertEqual(snapshot.summary_count, 3)
        self.assertEqual(snapshot.leaf_count, 7)
        self.assertEqual(snapshot.primary_candidate, "hydra_period_gradual_rank_max")
        self.assertEqual(snapshot.primary_passed_count, 3)
        self.assertEqual(snapshot.primary_required_count, 10)
        self.assertEqual(snapshot.primary_min_per_type_auc["gradual_drift"], 0.7953196519289775)
        self.assertEqual(snapshot.primary_min_per_type_auc["period_change"], 0.8960376741612595)
        self.assertEqual(snapshot.primary_min_pooled_auc, 0.9010220185101353)
        self.assertEqual(
            tuple(summary.method for summary in snapshot.summaries),
            ("hydra_period_rank_max", "gradual_signature_local_gap", "hydra_period_gradual_rank_max"),
        )
        self.assertEqual(tuple(snapshot.leaf_sha256.items()), snapshot.leaf_inventory)
        self.assertIsInstance(snapshot.leaf_sha256, MappingProxyType)
        with self.assertRaises(TypeError):
            snapshot.leaf_sha256["context_antigravity.json"] = "0" * 64  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            snapshot.primary_candidate = "hydra_period_rank_max"  # type: ignore[misc]

    def test_exp462_rejects_coherent_forged_primary_summary_antigravity(self) -> None:
        """Break caught: a forged 9/10 primary summary cannot replace frozen 3/10 authority."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.contracts_antigravity import (
            Exp462MethodSummaryAntigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )

        snapshot = capture_exp462_authority_antigravity().result_snapshot
        forged_primary = Exp462MethodSummaryAntigravity(
            method="hydra_period_gradual_rank_max",
            fold_count=10,
            passed_count=9,
            failed_count=1,
            min_gradual_drift_auc=0.99,
            min_period_change_auc=0.99,
            min_pooled_auc=0.99,
            max_normal_wilson_upper=0.01,
            all_scores_finite=True,
            passed=True,
        )
        with self.assertRaises(ValueError):
            replace(snapshot, summaries=snapshot.summaries[:-1] + (forged_primary,))

    def test_exp462_rejects_mutable_simple_namespace_summary_before_mutation_antigravity(self) -> None:
        """Break caught: mutable lookalikes must not survive snapshot validation or later mutation."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )

        snapshot = capture_exp462_authority_antigravity().result_snapshot
        summary_fields = (
            "method", "fold_count", "passed_count", "failed_count",
            "min_gradual_drift_auc", "min_period_change_auc", "min_pooled_auc",
            "max_normal_wilson_upper", "all_scores_finite", "passed",
        )
        mutable_summaries = tuple(
            SimpleNamespace(**{field: getattr(summary, field) for field in summary_fields})
            for summary in snapshot.summaries
        )
        try:
            forged_snapshot = replace(snapshot, summaries=mutable_summaries)
        except ValueError:
            return

        mutable_summaries[-1].passed_count = 9
        self.assertEqual(
            forged_snapshot.summaries[-1].passed_count,
            3,
            "a mutable summary changed after validation",
        )


if __name__ == "__main__":
    unittest.main()
