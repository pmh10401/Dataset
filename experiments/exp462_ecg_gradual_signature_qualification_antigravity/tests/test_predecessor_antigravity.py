from __future__ import annotations

import inspect
import unittest


class PredecessorAntigravityTest(unittest.TestCase):
    def test_capture_returns_exact_frozen_exp461_authority_antigravity(self) -> None:
        """Break caught: Exp462 consumes a non-terminal or altered Exp461 authority."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.predecessor_antigravity import (
            capture_exp461_gradual_authority_antigravity,
        )

        authority = capture_exp461_gradual_authority_antigravity()
        self.assertEqual(authority.predecessor_experiment, "Exp461")
        self.assertEqual(authority.reviewed_code_commit, "3a306ff3e276dae13b28b9733db30e9934da7eb2")
        self.assertEqual(authority.terminal_closeout_commit, "0340b6ee737da84777283668449738d0c1585566")
        self.assertEqual(authority.artifact_map_sha256, "8d0992d362cefab305ef8801afd9d1f5d51522e623fa372ba18f6d5d0f3478be")
        self.assertEqual(authority.decision_sha256, "27f6d8db82b71051a16b68d3ad2f69f6608f6a922d765b35029e594afc581315")
        self.assertEqual(authority.dashboard_snapshot_sha256, "e0e70dd42536f71aa736a8c178d14ed84b268f703623937dfe1f8663967bbabc")
        self.assertEqual(authority.summary_sha256, "7500d2599060b257a171856fb68bed39a0ec0f4ddb26ca6084e606c911251e25")
        self.assertEqual(authority.status, "qualification_failed_before_official_evaluation")
        self.assertEqual(authority.primary_candidate, "hydra_period_rank_max")
        self.assertEqual(authority.passed_record_count, 9)
        self.assertEqual(authority.required_record_count, 10)
        self.assertEqual(authority.failed_gradual_drift_auc, 0.8402306222021025)
        self.assertEqual(authority.period_min_auc, 0.907392736783864)
        self.assertEqual(
            tuple((row.seed, row.fold_index, row.period_change_auc) for row in authority.historical_period_folds),
            ((20260721, 3, 0.907392736783864), (20261721, 4, 0.9216148584075248)),
        )
        self.assertEqual(authority.label_query_count, 0)
        self.assertEqual(
            tuple(authority.private_sha256.items()),
            (
                ("artifact_map_antigravity.json", "8d0992d362cefab305ef8801afd9d1f5d51522e623fa372ba18f6d5d0f3478be"),
                ("decision_antigravity.json", "27f6d8db82b71051a16b68d3ad2f69f6608f6a922d765b35029e594afc581315"),
                ("dashboard_snapshot_antigravity.json", "e0e70dd42536f71aa736a8c178d14ed84b268f703623937dfe1f8663967bbabc"),
                ("summary_antigravity.json", "7500d2599060b257a171856fb68bed39a0ec0f4ddb26ca6084e606c911251e25"),
            ),
        )

    def test_capture_stays_source_free_and_label_free_antigravity(self) -> None:
        """Break caught: predecessor capture opens source data, labels, or CLI output."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity import predecessor_antigravity

        source = inspect.getsource(predecessor_antigravity)
        for forbidden in ("sqlite3", "load_normal2_train_antigravity", "TEST", "label_query_count = 1", "official_result_root"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
