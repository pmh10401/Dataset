from __future__ import annotations

import unittest


class Exp471PredecessorTestAntigravity(unittest.TestCase):
    def test_exports_are_exact_exp470_authority_objects_antigravity(self) -> None:
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.interpolation_antigravity import (
            interpolate_to_140_antigravity as exp470_interpolate,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.predecessor_antigravity import (
            capture_exp465_authority_antigravity as exp470_capture,
            score_c0_fold_antigravity as exp470_score_c0,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.sources_antigravity import (
            load_ecg200_normal_train_antigravity as exp470_load,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.synthetic_antigravity import (
            primary_rows_antigravity as exp470_primary_rows,
        )
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.interpolation_antigravity import (
            interpolate_to_140_antigravity,
        )
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.predecessor_antigravity import (
            capture_exp465_authority_antigravity,
            score_c0_fold_antigravity,
        )
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.sources_antigravity import (
            load_ecg200_normal_train_antigravity,
        )
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.synthetic_antigravity import (
            primary_rows_antigravity,
        )

        self.assertIs(load_ecg200_normal_train_antigravity, exp470_load)
        self.assertIs(interpolate_to_140_antigravity, exp470_interpolate)
        self.assertIs(primary_rows_antigravity, exp470_primary_rows)
        self.assertIs(capture_exp465_authority_antigravity, exp470_capture)
        self.assertIs(score_c0_fold_antigravity, exp470_score_c0)
