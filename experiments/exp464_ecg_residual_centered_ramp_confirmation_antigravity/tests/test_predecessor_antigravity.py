from __future__ import annotations

import inspect
import unittest


class PredecessorAntigravityTest(unittest.TestCase):
    def test_capture_returns_exact_frozen_exp463_authority_antigravity(self) -> None:
        """Break caught: Exp464 consumes non-terminal or altered Exp463 private TRAIN authority."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        self.assertEqual(authority.predecessor_experiment, "Exp463")
        self.assertEqual(
            authority.terminal_closeout_commit,
            "147225831e9c13e9df5043fa88757bb03e01b79a",
        )
        self.assertEqual(
            authority.task5_report_sha256,
            "e43abf66e43921941dece398514858be91e39d121c1173b26b352b67ba78e8e5",
        )
        self.assertEqual(
            authority.status,
            "stopped_before_official_evaluation_qualification_failed",
        )
        self.assertEqual(
            authority.reviewed_cli_commit,
            "923cc4ad6dd6737f84f4ea2c4f621eb652ac54b4",
        )
        self.assertEqual(authority.private_run_count, 2)
        self.assertEqual(authority.result_row_count, 70)
        self.assertEqual(authority.result_failed_row_count, 30)
        self.assertEqual(authority.result_summary_count, 7)
        self.assertEqual(authority.result_leaf_count, 7)
        self.assertEqual(
            authority.artifact_map_sha256,
            "25fb4ab237ae8f2601c2bc03b05f4e6a2d0cb9889876a83fa0f2c4ee7c4f0ea5",
        )
        self.assertEqual(authority.official_evaluate_count, 0)
        self.assertEqual(authority.official_inspect_count, 0)
        self.assertEqual(authority.label_query_count, 0)
        self.assertEqual(
            tuple(
                (method, summary.passed_count, summary.min_gradual_drift_auc, summary.min_period_change_auc)
                for method, summary in authority.control_summaries.items()
            ),
            (
                ("C0", 9, 0.8402306222021025, 0.907392736783864),
                ("C1", 7, 0.83450279161008, 0.8984897640963734),
                ("C2", 9, 0.8399225391076908, 0.9044753785020874),
                ("C7", 3, 0.7953196519289775, 0.8960376741612595),
            ),
        )

    def test_capture_stays_source_free_and_uses_only_versioned_exp463_evidence_antigravity(self) -> None:
        """Break caught: predecessor capture opens result roots or bypasses the tracked report and ledgers."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import (
            predecessor_antigravity,
        )

        source = inspect.getsource(predecessor_antigravity)
        for required in (
            "RESEARCH_STATE.json",
            "EXPERIMENT_CATALOG.json",
            "CHAMPION_LEDGER.json",
            "task-5-report-antigravity.html",
            "docs/EXP463_IMPLEMENTATION_REPORT_antigravity.html",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        for forbidden in (
            "sqlite3",
            "load_normal2_train_antigravity",
            "/private/tmp/exp463_task5_private_run1_antigravity",
            "/Users/mac/Documents/Dataset/exp463_ecg_prefix_ramp_gated_qualification_antigravity_20260821",
            "TEST",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
