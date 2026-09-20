from __future__ import annotations

import inspect
import unittest


class PredecessorAntigravityTest(unittest.TestCase):
    def test_capture_returns_exact_frozen_exp462_authority_antigravity(self) -> None:
        """Break caught: Exp463 consumes a non-terminal or altered Exp462 result."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )

        authority = capture_exp462_authority_antigravity()
        self.assertEqual(authority.predecessor_experiment, "Exp462")
        self.assertEqual(authority.terminal_closeout_commit, "040422f62406e3ca19038b9609903c4ba76e1694")
        self.assertEqual(authority.status, "stopped_before_official_evaluation_qualification_failed")
        self.assertEqual(authority.primary_candidate, "hydra_period_gradual_rank_max")
        self.assertEqual(authority.passed_record_count, 3)
        self.assertEqual(authority.required_record_count, 10)
        self.assertEqual(authority.failed_gradual_drift_auc, 0.7953196519289775)
        self.assertEqual(authority.period_min_auc, 0.8960376741612595)
        self.assertEqual(authority.result_snapshot.row_count, 30)
        self.assertEqual(authority.result_snapshot.summary_count, 3)
        self.assertEqual(authority.result_snapshot.leaf_count, 7)
        self.assertEqual(
            tuple((row.seed, row.fold_index, row.period_change_auc) for row in authority.historical_period_folds),
            ((20260721, 3, 0.8960376741612595), (20261721, 4, 0.9285624465570143)),
        )
        self.assertEqual(authority.label_query_count, 0)
        self.assertEqual(
            tuple(authority.private_sha256.items()),
            (
                ("context_antigravity.json", "fd5a5271ae68183d9a121ba46f8dfb18aab8a8399140ee9d224e87af7bfd1291"),
                ("method_fold_metrics_antigravity.jsonl", "f3d0eab11fa3902c9e4ea30a4d83b19cb5cff52cc3f51a20034c9b3b553ccf0b"),
                ("failed_folds_antigravity.jsonl", "e7848e96f5380448eb38a21ecfe6e9a19fcca830fead0058b85a285f13da90ea"),
                ("method_summary_antigravity.json", "0f63f0ee9d1788810fddb27fcb576225c73301098e3ddbf24704710a627eb0a9"),
                ("decision_antigravity.json", "6eec416030c0b01f945feee824468dc0766dfd036945b1e9e7307e9498d0d58b"),
                ("dashboard_snapshot_antigravity.json", "d2f1ed5c289aa4eb061c04065cc7b49420f3ca4ebe894b269853703bb36a558b"),
                ("artifact_sha256_antigravity.json", "5bfdeaeb04013f26987131980f8681afb4eaec2d1aab086cc5ee1220aee69c50"),
            ),
        )

    def test_capture_stays_source_free_and_label_free_antigravity(self) -> None:
        """Break caught: predecessor capture opens source data, labels, or a runtime root."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity import predecessor_antigravity

        source = inspect.getsource(predecessor_antigravity)
        for forbidden in ("sqlite3", "load_normal2_train_antigravity", "TEST", "label_query_count = 1", "official_result_root"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
