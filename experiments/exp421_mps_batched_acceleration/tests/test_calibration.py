"""TRAIN-only calibration-cache parity tests for Exp421."""

from __future__ import annotations

from dataclasses import fields
import unittest

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.scoring import score_horizon
from experiments.exp421_mps_batched_acceleration.calibration import (
    CalibrationCache,
    build_calibration_cache,
    calibration_digest,
    score_query_with_cache,
)


HORIZON = 2


def _oof_residuals() -> tuple[np.ndarray, ...]:
    """Finite absolute residuals with one exact aligned length."""
    return (
        np.array([0.10, 0.35, 0.80, 1.20, 0.50, 1.80, 0.25, 1.05], dtype=np.float64),
        np.array([0.15, 0.55, 0.75, 1.40, 0.30, 1.55, 0.45, 0.95], dtype=np.float64),
        np.array([0.20, 0.60, 1.05, 1.65, 0.40, 1.30, 0.70, 1.10], dtype=np.float64),
    )


class CalibrationCacheTests(unittest.TestCase):
    def test_cached_scores_match_the_frozen_reference_for_normal_and_tail_queries(self) -> None:
        """Catches a changed robust normalization, percentile, p-value, or persistence rule."""
        oof = _oof_residuals()
        cache = build_calibration_cache(oof, horizon=HORIZON)
        queries = (
            np.array([0.12, 0.45, 0.70, 1.10, 0.35, 1.30, 0.40, 0.90]),
            np.array([0.20, 2.20, 2.30, 2.40, 0.30, 0.40, 0.50, 0.60]),
            np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]),
        )
        for query in queries:
            with self.subTest(query=query.tolist()):
                expected = score_horizon(oof, query, horizon=HORIZON)
                actual = score_query_with_cache(cache, query)
                self.assertEqual(actual.horizon, expected.horizon)
                self.assertEqual(actual.score, expected.score)
                self.assertEqual(actual.p_value, expected.p_value)
                self.assertEqual(actual.persistence, expected.persistence)
                self.assertEqual(
                    actual.p_value <= FROZEN_CONFIG.candidate_alpha
                    and actual.persistence >= FROZEN_CONFIG.persistence_fraction,
                    expected.p_value <= FROZEN_CONFIG.candidate_alpha
                    and expected.persistence >= FROZEN_CONFIG.persistence_fraction,
                )

    def test_cache_contains_only_train_calibration_evidence(self) -> None:
        """Catches accidental retention of labels or TEST identity data in a calibration entry."""
        cache = build_calibration_cache(_oof_residuals(), horizon=HORIZON)
        field_names = {item.name.lower() for item in fields(cache)}
        self.assertFalse(any("label" in name or "test" in name for name in field_names))
        self.assertEqual(len(cache.normalized_oof_residuals), 3)
        self.assertEqual(len(cache.oof_scores), 3)

    def test_cache_copies_inputs_and_retains_only_read_only_arrays(self) -> None:
        """Catches source aliasing or a mutable cache array that could change future scores."""
        mutable = [item.copy() for item in _oof_residuals()]
        cache = build_calibration_cache(mutable, horizon=HORIZON)
        baseline = score_query_with_cache(cache, np.array([0.2, 2.2, 2.3, 2.4, 0.3, 0.4, 0.5, 0.6]))
        mutable[0][:] = 99.0
        actual = score_query_with_cache(cache, np.array([0.2, 2.2, 2.3, 2.4, 0.3, 0.4, 0.5, 0.6]))
        self.assertEqual(actual, baseline)
        self.assertTrue(all(not values.flags.writeable for values in cache.normalized_oof_residuals))
        with self.assertRaises(ValueError):
            CalibrationCache(
                horizon=cache.horizon,
                center=cache.center,
                scale=cache.scale,
                exceedance_threshold=cache.exceedance_threshold,
                normalized_oof_residuals=tuple(np.array(values, copy=True) for values in cache.normalized_oof_residuals),
                oof_scores=cache.oof_scores,
                digest=cache.digest,
            )

    def test_digest_is_canonical_and_changes_when_residual_bytes_change(self) -> None:
        """Catches a digest that omits horizon, shape, or canonical residual bytes."""
        original = _oof_residuals()
        changed = list(_oof_residuals())
        changed[1] = changed[1].copy()
        changed[1][3] += 0.125
        self.assertEqual(
            calibration_digest(original, horizon=HORIZON),
            build_calibration_cache(original, horizon=HORIZON).digest,
        )
        self.assertNotEqual(
            calibration_digest(original, horizon=HORIZON),
            calibration_digest(tuple(changed), horizon=HORIZON),
        )
        self.assertNotEqual(
            calibration_digest(original, horizon=HORIZON),
            calibration_digest(original, horizon=HORIZON + 1),
        )

    def test_build_and_query_fail_closed_for_invalid_calibration_inputs(self) -> None:
        """Catches acceptance of misaligned, zero-scale, or nonfinite calibration evidence."""
        with self.assertRaisesRegex(ValueError, "aligned"):
            build_calibration_cache((np.array([0.0, 1.0]), np.array([0.0])), horizon=HORIZON)
        with self.assertRaisesRegex(ValueError, "scale"):
            build_calibration_cache((np.array([1.0, 1.0]), np.array([1.0, 1.0])), horizon=HORIZON)
        cache = build_calibration_cache(_oof_residuals(), horizon=HORIZON)
        with self.assertRaisesRegex(ValueError, "finite"):
            score_query_with_cache(cache, np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6, 0.7, 0.8]))
        with self.assertRaisesRegex(ValueError, "aligned"):
            score_query_with_cache(cache, np.array([0.1, 0.2]))


if __name__ == "__main__":
    unittest.main()
