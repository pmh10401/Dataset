from __future__ import annotations

import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp466_ecg200_phase_registration_margin_antigravity.sources_antigravity import (
    load_ecg200_normal_train_antigravity as exp466_load_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.sources_antigravity import (
    load_ecg200_normal_train_antigravity as exp468_load_antigravity,
)


_TRAIN_PATH_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp465_ecg200_train_antigravity/"
    "ECG200_TRAIN_antigravity.ts",
)
_TRAIN_MD5_ANTIGRAVITY = "808875aeac1c358a90c630fe369327c4"


class SourcesAntigravityTest(unittest.TestCase):
    def test_export_is_exact_exp466_loader_antigravity(self) -> None:
        """Break caught: Exp468 wraps or copies the frozen Exp466 TRAIN loader."""
        self.assertIs(exp468_load_antigravity, exp466_load_antigravity)

    def test_loader_accepts_only_the_approved_normal_train_antigravity(self) -> None:
        """Break caught: approved TRAIN identity or the 69x96 normal boundary drifts."""
        self.assertEqual(
            hashlib.md5(_TRAIN_PATH_ANTIGRAVITY.read_bytes()).hexdigest(),
            _TRAIN_MD5_ANTIGRAVITY,
        )
        normal = exp468_load_antigravity(_TRAIN_PATH_ANTIGRAVITY)

        self.assertEqual(normal.shape, (69, 96))
        self.assertEqual(normal.dtype, np.dtype(np.float64))
        self.assertTrue(normal.flags.owndata)
        self.assertTrue(normal.flags.c_contiguous)
        self.assertFalse(normal.flags.writeable)
        self.assertTrue(np.isfinite(normal).all())

    def test_test_name_is_rejected_before_any_file_open_antigravity(self) -> None:
        """Break caught: a forbidden ECG200 TEST name reaches file-open activity."""
        forbidden = Path("local_fixture_ECG200_TEST_antigravity.ts")
        with patch.object(
            Path,
            "open",
            side_effect=AssertionError("forbidden TEST file was opened"),
        ) as open_spy:
            with self.assertRaisesRegex(ValueError, "TEST paths are forbidden"):
                exp468_load_antigravity(forbidden)
        open_spy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
