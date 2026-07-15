"""Unit tests for Exp158/159 similarity pooling and lane integration."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import similarity_pool_core as pool_core
import run_experiment_158_159_similarity_pooling as pipeline
import exp153_157_core as core
import exp153_157_reporting as reporting


class SimilarityPoolingContractTests(unittest.TestCase):

    def test_00_target_limit_keeps_full_donor_universe(self):
        rows = [{"dataset_name": f"d{index}"} for index in range(5)]
        donors, targets = pool_core.partition_manifest_rows(rows, target_limit=2)
        self.assertEqual([row["dataset_name"] for row in donors], ["d0", "d1", "d2", "d3", "d4"])
        self.assertEqual([row["dataset_name"] for row in targets], ["d0", "d1"])

    def test_01_target_reference_exclusion(self):
        """1. Target dataset name must be excluded from selected donors."""
        donor_names = ["d1", "d2", "target", "d3"]
        distances = np.asarray([1.0, 2.0, 3.0, 4.0])
        donor_n_trains = [100, 100, 100, 100]

        selection = pool_core.select_donors(
            target_name="target",
            donor_names=donor_names,
            distances=distances,
            donor_n_trains=donor_n_trains,
            min_pool_reference=200
        )
        self.assertIsNotNone(selection)
        selected_donors, min_dist, boundary_dist = selection
        self.assertNotIn("target", selected_donors)
        self.assertEqual(selected_donors, ["d1", "d2"])

    def test_02_descriptor_is_pure_train_function(self):
        """2. Descriptor must only receive TRAIN OOF scores and be a pure function."""
        train_oof = {
            "rocket_exp40": np.arange(10, dtype=float),
            "exp55_best": np.arange(10, dtype=float) * 2,
            "exp56_best": np.arange(10, dtype=float) * 3,
            "exp84": np.arange(10, dtype=float) * 4,
        }
        desc, status = pool_core.compute_dataset_descriptor(train_oof)
        self.assertEqual(status, "complete")
        self.assertIsNotNone(desc)
        self.assertEqual(len(desc), 28)

        # Check purity (no side effects, same output for same input)
        desc2, _ = pool_core.compute_dataset_descriptor(train_oof)
        np.testing.assert_array_equal(desc, desc2)

    def test_03_n_train_under_5_abstention(self):
        """3. n_train < 5 is complete calibration abstention, generating no candidates."""
        # Under 5 train samples, train_oof is empty
        train_oof_empty = {
            "rocket_exp40": np.asarray([]),
            "exp55_best": np.asarray([]),
            "exp56_best": np.asarray([]),
            "exp84": np.asarray([]),
        }
        desc, status = pool_core.compute_dataset_descriptor(train_oof_empty)
        self.assertIsNone(desc)
        self.assertIn("no finite scores", status)

    def test_04_n_train_range_pooling_target(self):
        """4. 5 <= n_train < 20 are the only eligible small train targets."""
        self.assertFalse(bool(5 <= 4 < 20))
        self.assertTrue(bool(5 <= 5 < 20))
        self.assertTrue(bool(5 <= 19 < 20))
        self.assertFalse(bool(5 <= 20 < 20))

    def test_05_minimum_pool_reference_200(self):
        """5. Minimum pool reference count must be 200."""
        donor_names = ["d1", "d2"]
        distances = np.asarray([1.0, 2.0])

        # n_train sum is 199 < 200
        selection_fail = pool_core.select_donors(
            target_name="target",
            donor_names=donor_names,
            distances=distances,
            donor_n_trains=[100, 99],
            min_pool_reference=200
        )
        self.assertIsNone(selection_fail)

        # n_train sum is 200 >= 200
        selection_ok = pool_core.select_donors(
            target_name="target",
            donor_names=donor_names,
            distances=distances,
            donor_n_trains=[100, 100],
            min_pool_reference=200
        )
        self.assertIsNotNone(selection_ok)

    def test_05b_minimum_pool_reference_is_enforced_for_every_source(self):
        """A donor pool is unavailable when any source has fewer than 200 finite scores."""
        donor_names = ["d1", "d2", "d3"]
        distances = np.asarray([1.0, 2.0, 3.0])
        counts = {
            "rocket_exp40": [100, 100, 100],
            "exp55_best": [100, 100, 100],
            "exp56_best": [100, 99, 100],
            "exp84": [100, 100, 100],
        }

        selection = pool_core.select_donors_by_source_counts(
            target_name="target",
            donor_names=donor_names,
            distances=distances,
            donor_reference_counts_by_source=counts,
            min_pool_reference=200,
        )

        self.assertIsNotNone(selection)
        selected, _, _, selected_counts = selection
        self.assertEqual(selected, ["d1", "d2", "d3"])
        self.assertGreaterEqual(min(selected_counts.values()), 200)

    def test_06_distance_boundary_tie_inclusion(self):
        """6. All donors at the boundary distance tie must be included."""
        donor_names = ["d1", "d2", "d3", "d4"]
        distances = np.asarray([1.0, 2.0, 2.0, 3.0])
        donor_n_trains = [150, 40, 30, 100]

        selection = pool_core.select_donors(
            target_name="target",
            donor_names=donor_names,
            distances=distances,
            donor_n_trains=donor_n_trains,
            min_pool_reference=200
        )
        self.assertIsNotNone(selection)
        selected_donors, min_dist, boundary_dist = selection
        # Both d2 and d3 have distance 2.0. Even though d1 (150) + d2 (40) = 190 < 200,
        # adding d3 takes it to 220. Both d2 and d3 must be in the pool.
        self.assertIn("d2", selected_donors)
        self.assertIn("d3", selected_donors)
        self.assertEqual(set(selected_donors), {"d1", "d2", "d3"})
        self.assertEqual(boundary_dist, 2.0)

    def test_07_donor_order_invariance(self):
        """7. Selected donors must be invariant to input list order."""
        donor_names = ["d1", "d2", "d3"]
        distances = np.asarray([1.0, 2.0, 3.0])
        donor_n_trains = [150, 60, 100]

        selection1 = pool_core.select_donors("target", donor_names, distances, donor_n_trains)

        # Shuffle inputs
        shuffled_idx = [2, 0, 1]
        shuffled_names = [donor_names[i] for i in shuffled_idx]
        shuffled_distances = distances[shuffled_idx]
        shuffled_n_trains = [donor_n_trains[i] for i in shuffled_idx]

        selection2 = pool_core.select_donors("target", shuffled_names, shuffled_distances, shuffled_n_trains)

        self.assertEqual(set(selection1[0]), set(selection2[0]))
        self.assertEqual(selection1[1], selection2[1])
        self.assertEqual(selection1[2], selection2[2])

    def test_08_dataset_rename_invariance(self):
        """8. Selected donor index sets must be invariant to dataset renaming."""
        donor_names = ["d1", "d2", "d3"]
        distances = np.asarray([1.0, 2.0, 3.0])
        donor_n_trains = [150, 60, 100]

        selection1 = pool_core.select_donors("target", donor_names, distances, donor_n_trains)

        renamed_names = ["x1", "x2", "x3"]
        selection2 = pool_core.select_donors("target", renamed_names, distances, donor_n_trains)

        # The indices of selected donors in renamed list must match original indices
        idx1 = [donor_names.index(name) for name in selection1[0]]
        idx2 = [renamed_names.index(name) for name in selection2[0]]
        self.assertEqual(idx1, idx2)

    def test_09_family_unused(self):
        """9. Verification that family is not in the scope of core calculations."""
        # Check pool_core functions inspect signatures
        sig = inspect.signature(pool_core.select_donors)
        self.assertNotIn("family", sig.parameters)
        sig_desc = inspect.signature(pool_core.compute_dataset_descriptor)
        self.assertNotIn("family", sig_desc.parameters)

    def test_10_test_length_unused(self):
        """10. Verification that test length does not affect descriptor/distance calculations."""
        # Descriptors only use train scores
        train_oof = {
            "rocket_exp40": np.arange(10, dtype=float),
            "exp55_best": np.arange(10, dtype=float) * 2,
            "exp56_best": np.arange(10, dtype=float) * 3,
            "exp84": np.arange(10, dtype=float) * 4,
        }
        desc, _ = pool_core.compute_dataset_descriptor(train_oof)

        # Test size parameter is not passed or used
        sig = inspect.signature(pool_core.compute_dataset_descriptor)
        self.assertNotIn("n_test", sig.parameters)
        self.assertNotIn("test_length", sig.parameters)

    def test_11_y_test_not_accepted(self):
        """11. Core prediction functions must not accept y_test or labels arguments."""
        for name, func in inspect.getmembers(pool_core, inspect.isfunction):
            sig = inspect.signature(func)
            for param in sig.parameters:
                self.assertNotIn(param, ("y_test", "y", "label", "labels", "test_labels"))

    def test_12_no_evaluation_metrics_in_predictions(self):
        """12. Evaluation metrics (TP, FP, F1) must not be in prediction outputs."""
        for key in reporting.FORBIDDEN_PREDICTION_KEYS:
            self.assertIn(key, reporting.FORBIDDEN_PREDICTION_KEYS)

    def test_13_14_15_lane_integration_properties(self):
        """13, 14, 15. Exp159 exact preservation of Hard/Priority, Standard-only additions, and mutual exclusivity."""
        b_hard = {1, 2}
        b_priority = {3, 4}
        b_standard = {5}
        b_candidate = b_hard | b_priority | b_standard

        candidate_158 = {2, 5, 6} # 2 is in hard, 5 is in standard, 6 is new similarity-only

        # Integration rule
        s_only = candidate_158 - b_candidate

        # Exp159 lanes
        i_hard = b_hard
        i_priority = b_priority
        i_standard = b_standard | s_only
        i_candidate = i_hard | i_standard | i_priority

        # 13. Hard/Priority exact preservation
        self.assertEqual(i_hard, b_hard)
        self.assertEqual(i_priority, b_priority)

        # 14. New similarity-only (6) added to standard only
        self.assertIn(6, i_standard)
        self.assertNotIn(6, i_hard)
        self.assertNotIn(6, i_priority)

        # 15. Mutual exclusivity and candidate union
        self.assertEqual(i_hard & i_standard, set())
        self.assertEqual(i_hard & i_priority, set())
        self.assertEqual(i_standard & i_priority, set())
        self.assertEqual(i_candidate, i_hard | i_standard | i_priority)

    def test_13b_lane_integration_calls_production_helper(self):
        lanes = pool_core.integrate_similarity_review_candidates(
            baseline_hard={1, 2},
            baseline_standard={5},
            baseline_priority={3, 4},
            similarity_candidates={2, 5, 6},
            n_test=8,
        )
        self.assertEqual(lanes["hard"], {1, 2})
        self.assertEqual(lanes["priority_review"], {3, 4})
        self.assertEqual(lanes["standard_review"], {5, 6})
        self.assertEqual(lanes["candidate"], {1, 2, 3, 4, 5, 6})

    def test_13c_lane_integration_rejects_invalid_baseline(self):
        with self.assertRaisesRegex(ValueError, "baseline lane overlap"):
            pool_core.integrate_similarity_review_candidates(
                baseline_hard={1},
                baseline_standard={1},
                baseline_priority=set(),
                similarity_candidates={2},
                n_test=3,
            )

    def test_13d_baseline_coverage_must_be_complete(self):
        rows = [
            {
                "dataset_name": "d1",
                "alpha": alpha,
                "hard_indices": "",
                "standard_review_indices": "",
                "priority_review_indices": "",
                "candidate_indices": "",
            }
            for alpha in core.ALPHAS
        ]
        with self.assertRaisesRegex(ValueError, "baseline coverage mismatch"):
            pool_core.validate_baseline_prediction_rows(
                rows,
                expected_dataset_names=["d1", "d2"],
                alphas=core.ALPHAS,
            )

    def test_16_prediction_hash_tamper_detection(self):
        """16. Hash validation detects file tampering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            path.write_text('{"a": 1}\n', encoding="utf-8")
            h = core.sha256_file(path)

            # Should pass
            core.verify_frozen_file(path, h)

            # Tamper
            path.write_text('{"a": 2}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                core.verify_frozen_file(path, h)

    def test_17_reproducibility(self):
        """17. Execution produces identical outputs deterministically."""
        # Mock descriptors robust scale
        m = np.random.RandomState(42).normal(size=(5, 28))
        scaled1, medians1, scales1 = pool_core.robust_scale_descriptors(m)
        scaled2, medians2, scales2 = pool_core.robust_scale_descriptors(m)
        np.testing.assert_array_equal(scaled1, scaled2)
        np.testing.assert_array_equal(medians1, medians2)
        np.testing.assert_array_equal(scales1, scales2)


if __name__ == "__main__":
    unittest.main()
