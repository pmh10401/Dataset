from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exp153_157_core as core


class RobustCalibrationTests(unittest.TestCase):
    def test_robust_standardize_uses_train_reference_only(self):
        reference = np.asarray([0.0, 1.0, 2.0, 3.0, 100.0])
        values = np.asarray([1.0, 4.0])
        z, location, scale = core.robust_standardize(reference, values)
        self.assertAlmostEqual(location, 2.0)
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(z, (values - location) / scale)

    def test_leave_one_out_conformal_has_no_self_comparison(self):
        scores = np.asarray([0.1, 0.2, 0.3, 0.4])
        p = core.leave_one_out_conformal_p_values(scores)
        np.testing.assert_allclose(p, np.asarray([1.0, 0.75, 0.5, 0.25]))

    def test_pooled_reference_excludes_target_dataset(self):
        standardized = {
            "a": np.asarray([0.0, 1.0]),
            "b": np.asarray([10.0, 11.0]),
            "c": np.asarray([20.0]),
        }
        pooled = core.leave_one_dataset_out_pool(standardized, "b")
        np.testing.assert_array_equal(pooled, np.asarray([0.0, 1.0, 20.0]))

    def test_hybrid_p_values_use_local_when_resolution_is_sufficient(self):
        local_reference = np.arange(0.0, 250.0)
        test_scores = np.asarray([249.0])
        pooled_reference = np.arange(0.0, 1000.0)
        p, source = core.hybrid_conformal_p_values(
            local_reference,
            test_scores,
            pooled_reference,
            alpha=0.005,
            minimum_pool_size=200,
        )
        self.assertEqual(source, "local")
        self.assertAlmostEqual(float(p[0]), 2 / 251)

    def test_hybrid_p_values_use_pool_when_local_resolution_is_insufficient(self):
        local_reference = np.asarray([0.0, 1.0, 2.0])
        test_scores = np.asarray([3.0])
        pooled_reference = np.arange(0.0, 1000.0)
        p, source = core.hybrid_conformal_p_values(
            local_reference,
            test_scores,
            pooled_reference,
            alpha=0.01,
            minimum_pool_size=200,
        )
        self.assertEqual(source, "pooled")
        self.assertEqual(p.shape, (1,))

    def test_excluded_pool_conformal_removes_target_reference(self):
        global_reference = np.asarray([0.0, 1.0, 2.0, 10.0, 11.0])
        target_reference = np.asarray([10.0, 11.0])
        p = core.excluded_pool_conformal_p_values(
            global_reference,
            target_reference,
            np.asarray([2.0]),
        )
        # Remaining pool is [0, 1, 2], so (1 + 1) / (3 + 1) = 0.5.
        self.assertAlmostEqual(float(p[0]), 0.5)


class GroupingAndFusionTests(unittest.TestCase):
    def test_train_only_group_key_does_not_use_family_or_test_size(self):
        boundaries = (64.0, 128.0, 256.0)
        first = core.train_only_group_key(16, 100, boundaries)
        second = core.train_only_group_key(16, 100, boundaries)
        self.assertEqual(first, second)
        self.assertEqual(first, "n005_019|len_q2")

    def test_top2_mean_score_requires_two_large_sources(self):
        matrix = np.asarray(
            [
                [5.0, 4.0, 0.0, -1.0],
                [5.0, 0.0, -1.0, -2.0],
            ]
        )
        score = core.top2_mean_score(matrix)
        np.testing.assert_allclose(score, np.asarray([4.5, 2.5]))

    def test_correlation_weights_downweight_redundant_sources(self):
        matrix = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.2],
                [2.0, 2.0, -0.1],
                [3.0, 3.0, 0.1],
            ]
        )
        weights, corr = core.inverse_correlation_weights(matrix)
        self.assertEqual(weights.shape, (3,))
        self.assertEqual(corr.shape, (3, 3))
        self.assertLess(weights[0], weights[2])
        self.assertLess(weights[1], weights[2])
        self.assertAlmostEqual(float(np.mean(weights)), 1.0)

    def test_meta_conformal_prediction_uses_train_meta_reference(self):
        train_matrix = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [2.0, 1.5, 0.0],
                [3.0, 2.5, 0.1],
            ]
        )
        test_matrix = np.asarray([[4.0, 3.0, 0.0]])
        p, meta_train, meta_test = core.meta_conformal_p_values(
            train_matrix,
            test_matrix,
            method="top2_mean",
        )
        self.assertEqual(p.shape, (1,))
        self.assertEqual(meta_train.shape, (4,))
        self.assertEqual(meta_test.shape, (1,))
        self.assertLessEqual(float(p[0]), 0.4)


class RiskControlTests(unittest.TestCase):
    def test_wilson_upper_bound_is_between_observed_rate_and_one(self):
        upper = core.wilson_upper_bound(successes=1, trials=100, confidence=0.95)
        self.assertGreaterEqual(upper, 0.01)
        self.assertLessEqual(upper, 1.0)

    def test_select_largest_alpha_uses_only_train_audit_upper_bounds(self):
        rows = [
            {"alpha": 0.005, "hard_false_alerts": 0, "pseudo_runs": 100},
            {"alpha": 0.01, "hard_false_alerts": 0, "pseudo_runs": 100},
            {"alpha": 0.02, "hard_false_alerts": 1, "pseudo_runs": 100},
            {"alpha": 0.05, "hard_false_alerts": 10, "pseudo_runs": 100},
        ]
        selected = core.select_largest_alpha_by_risk(
            rows,
            target_rate=0.05,
            confidence=0.95,
        )
        self.assertIn(selected["selected_alpha"], {0.005, 0.01, 0.02})
        self.assertNotEqual(selected["selected_alpha"], 0.05)

    def test_select_largest_alpha_abstains_when_no_alpha_satisfies_target(self):
        rows = [
            {"alpha": 0.005, "hard_false_alerts": 5, "pseudo_runs": 10},
            {"alpha": 0.01, "hard_false_alerts": 6, "pseudo_runs": 10},
        ]
        selected = core.select_largest_alpha_by_risk(
            rows,
            target_rate=0.01,
            confidence=0.95,
        )
        self.assertIsNone(selected["selected_alpha"])
        self.assertTrue(selected["abstain"])


class ConfirmationAndRoutingTests(unittest.TestCase):
    def test_agreement_indices_requires_two_sources_at_same_index(self):
        p_values = {
            "source_a": np.asarray([0.001, 0.50, 0.001]),
            "source_b": np.asarray([0.002, 0.001, 0.50]),
            "source_c": np.asarray([0.50, 0.002, 0.002]),
        }
        selected, counts = core.agreement_indices(
            p_values,
            alpha=0.01,
            required_sources=2,
        )
        self.assertEqual(selected, {0, 1, 2})
        np.testing.assert_array_equal(counts, np.asarray([2, 2, 2]))

    def test_leave_one_out_countcap_flags_only_extreme_train_scores(self):
        scores = np.asarray([0.0, 0.1, 0.2, 10.0])
        flags = core.leave_one_out_countcap_flags(scores, rate=0.25)
        self.assertEqual(flags.tolist(), [False, False, False, True])

    def test_confirmation_variants_are_pre_registered_and_not_selected(self):
        names = [item["name"] for item in core.CONFIRMATION_VARIANTS]
        self.assertEqual(
            names,
            ["legacy_countcap", "conformal_strict", "conformal_balanced"],
        )

    def test_route_lanes_is_disjoint(self):
        lanes = core.route_lanes(
            candidates={1, 2, 3, 4},
            block_b={1, 4},
            block_c={2, 4},
            test_size=6,
        )
        self.assertEqual(lanes["hard"], {1, 4})
        self.assertEqual(lanes["priority_review"], {2})
        self.assertEqual(lanes["standard_review"], {3})
        self.assertEqual(lanes["no_alert"], {0, 5})


class ArtifactTests(unittest.TestCase):
    def test_frozen_prediction_hash_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.jsonl"
            path.write_text(json.dumps({"dataset": "d1"}) + "\n", encoding="utf-8")
            digest = core.sha256_file(path)
            core.verify_frozen_file(path, digest)
            path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                core.verify_frozen_file(path, digest)

    def test_experiment_constants_are_fixed(self):
        self.assertEqual(core.ALPHAS, (0.005, 0.01, 0.02, 0.05))
        self.assertEqual(core.RISK_TARGETS_PER_RUN, (0.005, 0.01, 0.02))
        self.assertEqual(core.REQUIRED_SOURCE_AGREEMENT, 2)


if __name__ == "__main__":
    unittest.main()
