from __future__ import annotations

import unittest

from experiments.exp319_321_candidate_reliability_architecture.scoring import (
    candidate_confidence,
    calibration_stability,
    robust_margin_strength,
)


class CandidateConfidenceTests(unittest.TestCase):
    def test_larger_test_score_has_no_lower_stability_or_margin(self):
        train = list(range(20))
        self.assertLessEqual(
            calibration_stability(train, 18.0, 0.20),
            calibration_stability(train, 30.0, 0.20),
        )
        self.assertLessEqual(
            robust_margin_strength(train, 18.0, 0.20),
            robust_margin_strength(train, 30.0, 0.20),
        )

    def test_representation_sources_count_once(self):
        result = candidate_confidence(
            weights={"contrastive": 0.4, "svdd": 0.8, "discord": 0.6, "wavelet": 0.5},
            source_strengths={"contrastive": 1.0, "svdd": 0.5, "discord": 1.0},
            groups={
                "contrastive": "representation", "svdd": "representation",
                "discord": "shape", "wavelet": "frequency",
            },
        )
        self.assertEqual(result.available_groups, 3)
        self.assertEqual(result.supporting_groups, 2)
        self.assertAlmostEqual(result.available_weight, 1.9)
        self.assertAlmostEqual(result.supporting_evidence, 1.0)
        self.assertAlmostEqual(result.confidence, 1.0 / 1.9)


if __name__ == "__main__":
    unittest.main()
