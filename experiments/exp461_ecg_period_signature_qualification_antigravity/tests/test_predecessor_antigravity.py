from __future__ import annotations

import inspect
import unittest


EXPECTED_FAILED_FOLDS_ANTIGRAVITY = (
    (20260721, 3, 0.8208339620743423),
    (20261721, 4, 0.8004124541019063),
)


class PredecessorAntigravityTest(unittest.TestCase):
    def test_capture_returns_the_exact_frozen_exp460_authority_antigravity(self) -> None:
        from experiments.exp461_ecg_period_signature_qualification_antigravity.predecessor_antigravity import (
            capture_exp460_period_authority_antigravity,
        )

        authority = capture_exp460_period_authority_antigravity()
        self.assertEqual(authority.predecessor_experiment, "Exp460")
        self.assertEqual(
            authority.truth_commit,
            "67d39f5c0a75e061e2d107fc9a9f58f52e76c2f1",
        )
        self.assertEqual(
            authority.run_root,
            "/Users/mac/Documents/Dataset/exp460_ecg_strict_fallback_antigravity_20260820",
        )
        self.assertEqual(
            authority.reviewed_commit,
            "ca51813450f26b8a3fada6f39cf810452467297e",
        )
        self.assertEqual(
            authority.artifact_map_sha256,
            "8286c4da86d3f180e8b5e20cf54fc6f0011a26cf54d92b1a4867f68e5f44d916",
        )
        self.assertEqual(
            authority.dashboard_snapshot_sha256,
            "fea8b4e43d256feddef48545e201f674bfe39fe85559050ed405f4e996709ee5",
        )
        self.assertEqual(
            authority.decision_sha256,
            "0859be73fb8b7c77cc803d67745ecd1bd42766ea5e3c005ce1bb9d2d3e6342c1",
        )
        self.assertEqual(
            authority.status,
            "fallback_triggered_performance_not_improved",
        )
        self.assertEqual(authority.strict_route, "baseline")
        self.assertIs(authority.joint_success, False)
        self.assertEqual(authority.passed_record_count, 8)
        self.assertEqual(authority.failed_fold_count, 2)
        self.assertEqual(authority.worst_period_change_auc, 0.8004124541019063)
        self.assertEqual(
            tuple(
                (row.seed, row.fold_index, row.period_change_auc)
                for row in authority.failed_folds
            ),
            EXPECTED_FAILED_FOLDS_ANTIGRAVITY,
        )
        self.assertEqual(authority.label_query_count, 0)
        self.assertEqual(
            authority.leaf_sha256["artifact_sha256_antigravity.json"],
            "8286c4da86d3f180e8b5e20cf54fc6f0011a26cf54d92b1a4867f68e5f44d916",
        )

    def test_capture_reuses_frozen_exp459_failures_and_stays_source_free_antigravity(
        self,
    ) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.frozen_validation_authority_antigravity import (
            FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity import (
            predecessor_antigravity,
        )

        authority = predecessor_antigravity.capture_exp460_period_authority_antigravity()
        self.assertEqual(
            tuple(
                (record.seed, record.fold_index, record.per_type_auc["period_change"])
                for record in FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.fold_evidence
                if not record.passed
            ),
            tuple(
                (row.seed, row.fold_index, row.period_change_auc)
                for row in authority.failed_folds
            ),
        )

        source = inspect.getsource(predecessor_antigravity)
        self.assertIn("FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY", source)
        for forbidden in (
            "sqlite3",
            "load_normal2_train_antigravity",
            "label_query_count = 1",
            "TEST",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
