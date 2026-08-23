"""Executable specification for the frozen Exp468 local-STFT score."""

from __future__ import annotations

import math
import unittest

import numpy as np

from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.local_stft_antigravity import (
    jensen_shannon_antigravity,
    score_local_stft_antigravity,
    strict_confirmation_antigravity,
    window_power_distributions_antigravity,
)


STARTS_ANTIGRAVITY = (0, 14, 28, 42, 56)


def _manual_distribution_antigravity(frame: np.ndarray) -> np.ndarray:
    """Independent length-28 DFT fixture; does not call the production FFT path."""

    centered = frame - float(np.mean(frame))
    weighted = centered * np.array(
        [0.5 - 0.5 * math.cos(2.0 * math.pi * n / 27.0) for n in range(28)],
        dtype=np.float64,
    )
    power = []
    for frequency in range(1, 15):
        real = sum(
            float(weighted[n]) * math.cos(2.0 * math.pi * frequency * n / 28.0)
            for n in range(28)
        )
        imag = -sum(
            float(weighted[n]) * math.sin(2.0 * math.pi * frequency * n / 28.0)
            for n in range(28)
        )
        power.append(real * real + imag * imag + 1e-12)
    result = np.array(power, dtype=np.float64)
    return result / float(np.sum(result))


def _manual_js_antigravity(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = (p + q) / 2.0
    left = np.where(p > 0.0, p * np.log(p / midpoint), 0.0)
    right = np.where(q > 0.0, q * np.log(q / midpoint), 0.0)
    return float(0.5 * np.sum(left) + 0.5 * np.sum(right))


def _rows_antigravity(frequencies: tuple[float, ...]) -> np.ndarray:
    sample = np.arange(140, dtype=np.float64)
    return np.stack(
        tuple(
            0.03 * sample
            + np.sin(2.0 * math.pi * frequency * sample / 28.0)
            + 0.17 * np.cos(2.0 * math.pi * (frequency + 1.0) * sample / 28.0)
            for frequency in frequencies
        ),
    )


class LocalStftTransformAntigravityTest(unittest.TestCase):
    def test_exact_complete_frames_dc_removal_and_nyquist_antigravity(self) -> None:
        suffix = np.arange(84, dtype=np.float64) ** 2 + 0.1 * np.arange(84)
        row = np.concatenate((np.full(56, -12345.0), suffix))

        actual = window_power_distributions_antigravity(row)
        expected = np.stack(
            tuple(
                _manual_distribution_antigravity(suffix[start : start + 28])
                for start in STARTS_ANTIGRAVITY
            ),
        )

        self.assertEqual(actual.shape, (5, 14))
        self.assertEqual(actual.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(
            actual.sum(axis=1), np.ones(5), rtol=0.0, atol=1e-15,
        )
        self.assertTrue(np.isfinite(actual).all())

    def test_transform_is_level_and_positive_amplitude_invariant_antigravity(self) -> None:
        row = _rows_antigravity((2.25,))[0]
        expected = window_power_distributions_antigravity(row)

        np.testing.assert_allclose(
            window_power_distributions_antigravity(row + 7.0),
            expected,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            window_power_distributions_antigravity(row * 3.5),
            expected,
            rtol=0.0,
            atol=1e-12,
        )

    def test_constant_frames_are_finite_normalized_antigravity(self) -> None:
        actual = window_power_distributions_antigravity(
            np.full(140, 19.25, dtype=np.float64),
        )
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(
            actual,
            np.full((5, 14), 1.0 / 14.0),
            rtol=0.0,
            atol=1e-15,
        )

    def test_transform_rejects_nonexact_input_contract_antigravity(self) -> None:
        with self.assertRaises(ValueError):
            window_power_distributions_antigravity(np.zeros(139, dtype=np.float64))
        with self.assertRaises(TypeError):
            window_power_distributions_antigravity(np.zeros(140, dtype=np.float32))
        with self.assertRaises(ValueError):
            window_power_distributions_antigravity(
                np.full(140, np.nan, dtype=np.float64),
            )


class JensenShannonAntigravityTest(unittest.TestCase):
    def test_identity_is_exact_zero_and_hand_fixture_uses_natural_log_antigravity(self) -> None:
        p = np.array([0.5, 0.5], dtype=np.float64)
        q = np.array([1.0, 0.0], dtype=np.float64)

        self.assertEqual(jensen_shannon_antigravity(p, p), 0.0)
        self.assertAlmostEqual(
            jensen_shannon_antigravity(p, q),
            0.75 * math.log(4.0 / 3.0),
            places=15,
        )

    def test_js_rejects_invalid_distributions_antigravity(self) -> None:
        with self.assertRaises(ValueError):
            jensen_shannon_antigravity(
                np.array([0.4, 0.4], dtype=np.float64),
                np.array([0.5, 0.5], dtype=np.float64),
            )
        with self.assertRaises(ValueError):
            jensen_shannon_antigravity(
                np.array([0.5, -0.5], dtype=np.float64),
                np.array([0.5, 0.5], dtype=np.float64),
            )


class LocalStftScoringAntigravityTest(unittest.TestCase):
    def test_fit_uses_loo_query_uses_full_fit_with_tied_medians_antigravity(self) -> None:
        fit_rows = _rows_antigravity((1.0, 1.0, 4.0, 4.0))
        query_rows = _rows_antigravity((2.5, 6.0))
        fit_power = np.stack(
            tuple(window_power_distributions_antigravity(row) for row in fit_rows),
        )
        query_power = np.stack(
            tuple(window_power_distributions_antigravity(row) for row in query_rows),
        )

        expected_fit = []
        for index in range(len(fit_rows)):
            reference = np.median(np.delete(fit_power, index, axis=0), axis=0)
            reference /= reference.sum(axis=1, keepdims=True)
            expected_fit.append(
                np.median(
                    tuple(
                        _manual_js_antigravity(fit_power[index, frame], reference[frame])
                        for frame in range(5)
                    ),
                ),
            )
        full_reference = np.median(fit_power, axis=0)
        full_reference /= full_reference.sum(axis=1, keepdims=True)
        expected_query = np.array(
            [
                np.median(
                    tuple(
                        _manual_js_antigravity(query_power[row, frame], full_reference[frame])
                        for frame in range(5)
                    ),
                )
                for row in range(len(query_rows))
            ],
            dtype=np.float64,
        )

        scores = score_local_stft_antigravity(fit_rows, query_rows)

        np.testing.assert_allclose(
            scores.fit_raw_js, np.array(expected_fit), rtol=0.0, atol=1e-15,
        )
        np.testing.assert_allclose(
            scores.query_raw_js, expected_query, rtol=0.0, atol=1e-15,
        )
        for index, raw in enumerate(expected_query):
            expected_rank = (
                1 + np.count_nonzero(np.array(expected_fit) <= raw)
            ) / 5.0
            self.assertEqual(scores.query_ranks[index], expected_rank)

    def test_query_is_not_leaked_into_fit_reference_antigravity(self) -> None:
        fit_rows = _rows_antigravity((1.0, 2.0, 3.0, 4.0, 5.0))
        query_rows = _rows_antigravity((9.5,))

        fit_only = score_local_stft_antigravity(fit_rows, query_rows)
        leaked = score_local_stft_antigravity(
            np.concatenate((fit_rows, query_rows), axis=0),
            query_rows,
        )

        self.assertNotEqual(
            float(fit_only.query_raw_js[0]), float(leaked.query_raw_js[0]),
        )

    def test_rank_counts_tied_fit_raw_with_less_than_or_equal_antigravity(self) -> None:
        identical = _rows_antigravity((2.0, 2.0, 2.0, 2.0))
        scores = score_local_stft_antigravity(identical, identical[:1])
        np.testing.assert_array_equal(scores.fit_raw_js, np.zeros(4, dtype=np.float64))
        np.testing.assert_array_equal(scores.query_raw_js, np.zeros(1, dtype=np.float64))
        np.testing.assert_array_equal(scores.query_ranks, np.ones(1, dtype=np.float64))

    def test_strict_confirmation_boundaries_antigravity(self) -> None:
        maximum = 0.25
        query = np.array(
            [0.0, maximum, np.nextafter(maximum, math.inf)],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(
            strict_confirmation_antigravity(query, maximum),
            np.array([False, False, True]),
        )

    def test_score_result_is_validated_and_immutable_antigravity(self) -> None:
        scores = score_local_stft_antigravity(
            _rows_antigravity((1.0, 2.0, 3.0)),
            _rows_antigravity((4.0,)),
        )
        self.assertEqual(scores.backend, "numpy")
        self.assertFalse(scores.fit_raw_js.flags.writeable)
        self.assertFalse(scores.query_raw_js.flags.writeable)
        self.assertFalse(scores.query_ranks.flags.writeable)
        self.assertFalse(scores.confirmation_mask.flags.writeable)
        self.assertEqual(
            scores.fit_normal_max_raw_js,
            float(np.max(scores.fit_raw_js)),
        )
        with self.assertRaises(ValueError):
            scores.query_raw_js[0] = 99.0


if __name__ == "__main__":
    unittest.main()
