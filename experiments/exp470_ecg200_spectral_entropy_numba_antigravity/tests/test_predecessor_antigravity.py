from __future__ import annotations

import unittest


class Exp470PredecessorTestAntigravity(unittest.TestCase):
    def test_exports_are_exact_exp468_authority_objects_antigravity(self) -> None:
        """Break caught: Exp470 replaces a reviewed Exp468 authority object with a wrapper."""
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.interpolation_antigravity import (
            interpolate_to_140_antigravity as exp468_interpolate,
        )
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.predecessor_antigravity import (
            capture_exp465_authority_antigravity as exp468_capture,
            score_c0_fold_antigravity as exp468_score_c0,
        )
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.sources_antigravity import (
            load_ecg200_normal_train_antigravity as exp468_load,
        )
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.synthetic_antigravity import (
            primary_rows_antigravity as exp468_primary_rows,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.interpolation_antigravity import (
            interpolate_to_140_antigravity,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.predecessor_antigravity import (
            capture_exp465_authority_antigravity,
            score_c0_fold_antigravity,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.sources_antigravity import (
            load_ecg200_normal_train_antigravity,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.synthetic_antigravity import (
            primary_rows_antigravity,
        )

        self.assertIs(load_ecg200_normal_train_antigravity, exp468_load)
        self.assertIs(interpolate_to_140_antigravity, exp468_interpolate)
        self.assertIs(primary_rows_antigravity, exp468_primary_rows)
        self.assertIs(capture_exp465_authority_antigravity, exp468_capture)
        self.assertIs(score_c0_fold_antigravity, exp468_score_c0)


if __name__ == "__main__":
    unittest.main()
