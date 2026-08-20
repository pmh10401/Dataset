"""RED/GREEN tests for Exp450's area-pooled image variants."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from paper.equipment_anomaly_detection.sources_paper.imaging_paper import (
    EXP55_CONFIG_PAPER,
    EXP56_CONFIG_PAPER,
)

from experiments.exp450_resampling_image_ablation_antigravity import imaging_antigravity
from experiments.exp450_resampling_image_ablation_antigravity.imaging_antigravity import (
    FROZEN_IMAGING_SOURCE_ANTIGRAVITY,
    imaging_score_pair_exp450_antigravity,
    recurrence_images_exp450_antigravity,
    spectrogram_images_exp450_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.transforms_antigravity import (
    adaptive_average_1d_antigravity,
)


TRAIN = np.asarray(
    [
        [0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 3.0, -2.0, 1.5],
        [2.0, -1.0, 0.0, 1.0, -2.0, 0.5, 1.5, -0.5, 0.0],
        [-1.5, 0.0, 2.0, -0.5, 1.0, 0.5, -1.0, 2.5, 0.0],
        [0.5, 1.5, -0.5, -1.0, 2.0, 0.0, 1.0, -2.0, 0.5],
    ],
    dtype=np.float32,
)
TEST = np.asarray(
    [[1.0, 0.0, -1.0, 2.0, 0.0, -0.5, 1.0, 0.5, -1.5]], dtype=np.float32
)


class Exp450ImagingTests(unittest.TestCase):
    def test_linear_imaging_delegates_to_frozen_exp449(self) -> None:
        """Changing the linear control to a copied pipeline must fail."""
        expected = (np.asarray([1.0]), np.asarray([2.0]))
        with patch.object(
            imaging_antigravity, "imaging_score_pair_paper", return_value=expected
        ) as frozen:
            actual = imaging_score_pair_exp450_antigravity(
                TRAIN, TEST, EXP55_CONFIG_PAPER, image_mode="linear", device="cpu"
            )
        frozen.assert_called_once_with(TRAIN, TEST, EXP55_CONFIG_PAPER)
        self.assertEqual(actual[1].tolist(), [2.0])

    def test_area_recurrence_recomputes_symmetric_image_after_pooling(self) -> None:
        """Resizing a completed recurrence matrix must fail this pooled-distance check."""
        size = 4
        images = recurrence_images_exp450_antigravity(TRAIN, size, device="cpu")
        fixed = adaptive_average_1d_antigravity(TRAIN, size, device="cpu")
        distance = np.abs(fixed[:, :, None] - fixed[:, None, :])
        scale = np.quantile(distance.reshape(len(TRAIN), -1), 0.75, axis=1)
        expected = np.exp(-distance / np.maximum(scale, 1e-6)[:, None, None]).astype(
            np.float32
        )
        self.assertEqual(images.shape, (len(TRAIN), size, size))
        self.assertEqual(images.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(images, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(images, images.transpose(0, 2, 1))

    def test_area_spectrogram_has_exact_shape_finite_values_and_no_linear_resize(self) -> None:
        """Replacing area pooling with ``np.interp`` must fail before any interpolation."""
        with patch.object(np, "interp", side_effect=AssertionError("linear resize called")):
            images = spectrogram_images_exp450_antigravity(TRAIN, 32, device="cpu")
        self.assertEqual(images.shape, (len(TRAIN), 32, 32))
        self.assertEqual(images.dtype, np.dtype(np.float32))
        self.assertTrue(np.isfinite(images).all())

    def test_area_score_pair_preserves_frozen_glcm_pca_scaling_and_knn_closure(self) -> None:
        """Area mode must retain the frozen Exp56 estimator stages after imaging changes."""
        train_scores, test_scores = imaging_score_pair_exp450_antigravity(
            TRAIN, TEST, EXP56_CONFIG_PAPER, image_mode="area", device="cpu"
        )
        self.assertEqual(train_scores.shape, (len(TRAIN),))
        self.assertEqual(test_scores.shape, (len(TEST),))
        self.assertTrue(np.isfinite(train_scores).all())
        self.assertTrue(np.isfinite(test_scores).all())

    def test_frozen_imaging_source_provenance_is_recorded(self) -> None:
        """Changing the copied frozen closure without its source identity must fail."""
        self.assertEqual(
            FROZEN_IMAGING_SOURCE_ANTIGRAVITY,
            {
                "path": "paper/equipment_anomaly_detection/sources_paper/imaging_paper.py",
                "sha256": "1c18b6c6e3f70c922f9bba9689d26b99d2eb6320864e9c6425727af6e95d61e2",
            },
        )

    def test_mps_area_images_complete_without_cpu_parity_when_available(self) -> None:
        """MPS completion needs finite area images, not equality with CPU output."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        recurrence = recurrence_images_exp450_antigravity(TRAIN, 32, device="mps")
        spectrogram = spectrogram_images_exp450_antigravity(TRAIN, 32, device="mps")
        self.assertEqual(recurrence.shape, (len(TRAIN), 32, 32))
        self.assertEqual(spectrogram.shape, (len(TRAIN), 32, 32))
        self.assertTrue(np.isfinite(recurrence).all())
        self.assertTrue(np.isfinite(spectrogram).all())
        np.testing.assert_array_equal(recurrence, recurrence.transpose(0, 2, 1))

    def test_mps_area_public_c_d_image_paths_handle_mixed_axis_short_inputs(self) -> None:
        """C/D's frozen Exp55/56 area paths must finish on MPS for 64-sample rows."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        grid = np.linspace(0.0, 2.0 * np.pi, 64, dtype=np.float32)
        train = np.stack([np.sin(grid + offset) for offset in range(4)]).astype(np.float32)
        test = np.cos(grid, dtype=np.float32).reshape(1, -1)
        for config in (EXP55_CONFIG_PAPER, EXP56_CONFIG_PAPER):
            with self.subTest(config=config["name"]):
                train_score, test_score = imaging_score_pair_exp450_antigravity(
                    train, test, config, image_mode="area", device="mps"
                )
                self.assertEqual(train_score.shape, (len(train),))
                self.assertEqual(test_score.shape, (len(test),))
                self.assertTrue(np.isfinite(train_score).all())
                self.assertTrue(np.isfinite(test_score).all())


if __name__ == "__main__":
    unittest.main()
