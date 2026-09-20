from __future__ import annotations

import inspect
import unittest


EXPECTED_FOLD_ROWS_ANTIGRAVITY = (
    (20260721, 0, 0.878922199053553, True),
    (20260721, 1, 0.9676734632721098, True),
    (20260721, 2, 0.9438249946934036, True),
    (20260721, 3, 0.8208339620743423, False),
    (20260721, 4, 0.8865625471555757, True),
    (20261721, 0, 0.8684588395410107, True),
    (20261721, 1, 0.9500056187491416, True),
    (20261721, 2, 0.8542871055950255, True),
    (20261721, 3, 0.9407725969518635, True),
    (20261721, 4, 0.8004124541019063, False),
)


class PredecessorAntigravityTest(unittest.TestCase):
    def test_capture_returns_exact_frozen_exp459_authority_antigravity(self) -> None:
        from experiments.exp460_ecg_strict_fallback_antigravity.predecessor_antigravity import (
            capture_exp459_fallback_authority_antigravity,
        )

        authority = capture_exp459_fallback_authority_antigravity()
        self.assertEqual(
            tuple(
                (row.seed, row.fold_index, row.period_change_auc, row.passed)
                for row in authority.fold_records
            ),
            EXPECTED_FOLD_ROWS_ANTIGRAVITY,
        )
        self.assertEqual(
            tuple(
                (summary.seed, len(summary.fold_records), summary.passed)
                for summary in authority.seed_summaries
            ),
            (
                (20260721, 5, False),
                (20261721, 5, False),
            ),
        )
        self.assertEqual(authority.decision.status, "unstable")
        self.assertFalse(authority.decision.passed)
        self.assertEqual(
            (
                authority.baseline_confusion.tp,
                authority.baseline_confusion.fp,
                authority.baseline_confusion.fn,
                authority.baseline_confusion.tn,
            ),
            (609, 66, 837, 70788),
        )
        self.assertEqual(
            (
                authority.candidate_confusion.tp,
                authority.candidate_confusion.fp,
                authority.candidate_confusion.fn,
                authority.candidate_confusion.tn,
            ),
            (617, 66, 829, 70788),
        )
        self.assertEqual(
            authority.frozen_authority_digest,
            "f661c4ce6c7fd6625ac5cf767db688ce7ede7a0d0557721c9e9fa7e7ea6508ee",
        )
        self.assertEqual(
            authority.official_artifact_map_sha256,
            "a30fbe7ed34f88c1ab9b7fd40963edebbf1a4127d8d64c99141224cb2dbb106d",
        )
        self.assertEqual(
            authority.decision_sha256,
            "527318cacdbc63905ab58237d793803eb286192456eb007362e6bed11c8f8716",
        )
        self.assertEqual(authority.label_query_count, 0)
        self.assertEqual(
            tuple(
                (row.seed, row.fold_index, row.period_change_auc)
                for row in authority.fold_records
                if not row.passed
            ),
            (
                (20260721, 3, 0.8208339620743423),
                (20261721, 4, 0.8004124541019063),
            ),
        )

    def test_capture_reuses_frozen_exp459_science_and_stays_source_free_antigravity(
        self,
    ) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import (
            EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.frozen_validation_authority_antigravity import (
            FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity import predecessor_antigravity

        authority = predecessor_antigravity.capture_exp459_fallback_authority_antigravity()
        self.assertEqual(
            authority.frozen_authority_digest,
            FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.canonical_digest,
        )
        self.assertEqual(
            (
                authority.candidate_confusion.tp,
                authority.candidate_confusion.fp,
                authority.candidate_confusion.fn,
                authority.candidate_confusion.tn,
            ),
            (
                EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.tp,
                EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.fp,
                EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.fn,
                EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.tn,
            ),
        )

        source = inspect.getsource(predecessor_antigravity)
        self.assertIn("FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY", source)
        self.assertIn("EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY", source)
        for forbidden in (
            "load_normal2_train_antigravity",
            "sqlite3",
            "Hydra",
            "label_query_count = 1",
            "official_result_root",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
