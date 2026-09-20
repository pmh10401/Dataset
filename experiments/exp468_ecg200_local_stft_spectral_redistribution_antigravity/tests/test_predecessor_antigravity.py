from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from experiments.exp466_ecg200_phase_registration_margin_antigravity.interpolation_antigravity import (
    interpolate_to_140_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.predecessor_antigravity import (
    capture_exp465_authority_antigravity as exp466_capture_authority_antigravity,
    score_c0_fold_antigravity as exp466_score_c0_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.sources_antigravity import (
    load_ecg200_normal_train_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.predecessor_antigravity import (
    capture_exp465_authority_antigravity as exp468_capture_authority_antigravity,
    score_c0_fold_antigravity as exp468_score_c0_antigravity,
)


_TRAIN_PATH_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp465_ecg200_train_antigravity/"
    "ECG200_TRAIN_antigravity.ts",
)
_CHANGE_ORDER_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


class PredecessorAntigravityTest(unittest.TestCase):
    def test_exports_are_exact_exp466_c0_authority_objects_antigravity(self) -> None:
        """Break caught: Exp468 wraps either reviewed Exp466 C0 entrypoint."""
        self.assertIs(exp468_capture_authority_antigravity, exp466_capture_authority_antigravity)
        self.assertIs(exp468_score_c0_antigravity, exp466_score_c0_antigravity)

    def test_c0_normal_and_four_change_scores_are_byte_equal_antigravity(self) -> None:
        """Break caught: Exp468 changes C0 dtype, shape, order, or score bytes."""
        train96 = load_ecg200_normal_train_antigravity(_TRAIN_PATH_ANTIGRAVITY)
        train140 = interpolate_to_140_antigravity(train96)
        fit_indices = np.arange(55, dtype=np.int64)
        query_indices = np.arange(55, 69, dtype=np.int64)
        authority = exp468_capture_authority_antigravity()

        actual = exp468_score_c0_antigravity(
            train140,
            fit_indices,
            query_indices,
            seed=20260721,
            fold_index=0,
            authority=authority,
        )
        expected = exp466_score_c0_antigravity(
            train140,
            fit_indices,
            query_indices,
            seed=20260721,
            fold_index=0,
            authority=authority,
        )

        np.testing.assert_array_equal(actual.normal_scores, expected.normal_scores)
        self.assertEqual(actual.normal_scores.dtype, expected.normal_scores.dtype)
        self.assertEqual(actual.normal_scores.shape, expected.normal_scores.shape)
        self.assertEqual(actual.normal_scores.tobytes(), expected.normal_scores.tobytes())
        self.assertEqual(tuple(actual.primary_scores), _CHANGE_ORDER_ANTIGRAVITY)
        self.assertEqual(tuple(expected.primary_scores), _CHANGE_ORDER_ANTIGRAVITY)
        for name in _CHANGE_ORDER_ANTIGRAVITY:
            actual_values = actual.primary_scores[name]
            expected_values = expected.primary_scores[name]
            np.testing.assert_array_equal(actual_values, expected_values)
            self.assertEqual(actual_values.dtype, expected_values.dtype)
            self.assertEqual(actual_values.shape, expected_values.shape)
            self.assertEqual(actual_values.tobytes(), expected_values.tobytes())


if __name__ == "__main__":
    unittest.main()
