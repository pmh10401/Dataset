"""RED/GREEN tests for Exp450 pure transforms and backend selection."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
import scipy.signal

from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import (
    resample_series_paper,
)

from experiments.exp450_resampling_image_ablation_antigravity import (
    backend_antigravity,
    transforms_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.backend_antigravity import (
    assess_exp450_backend_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.transforms_antigravity import (
    adaptive_average_1d_antigravity,
    adaptive_average_2d_antigravity,
    align_rows_exp450_antigravity,
    resample_series_exp450_antigravity,
)


class Exp450TransformsBackendTests(unittest.TestCase):
    def test_control_resampler_is_byte_exact_to_exp449(self) -> None:
        """Changing control delegation to any new interpolation must fail."""
        row = np.asarray([0.0, 3.0, -1.0, 2.0], dtype=np.float32)
        actual = resample_series_exp450_antigravity(row, 11, mode="linear")
        expected = resample_series_paper(row, 11)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_hybrid_upsampling_preserves_monotone_shape(self) -> None:
        """Changing the PCHIP branch to an overshooting interpolator must fail."""
        row = np.asarray([0.0, 0.2, 0.7, 1.0], dtype=np.float32)
        result = resample_series_exp450_antigravity(row, 31, mode="polyphase_pchip")
        self.assertEqual(result.shape, (31,))
        self.assertTrue(np.all(np.diff(result) >= 0.0))
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_hybrid_downsampling_uses_reduced_polyphase_factors(self) -> None:
        """Changing the downsampling path away from SciPy polyphase must fail."""
        row = np.asarray([0.0, 1.0, -1.0, 3.0, 2.0, -2.0], dtype=np.float32)
        expected = scipy.signal.resample_poly(row, 2, 3).astype(np.float32)
        actual = resample_series_exp450_antigravity(row, 4, mode="polyphase_pchip")
        np.testing.assert_array_equal(actual, expected)

    def test_hybrid_singleton_and_alignment_have_exact_float32_shapes(self) -> None:
        """Dropping singleton expansion or row stacking must fail."""
        singleton = resample_series_exp450_antigravity(
            np.asarray([2.5], dtype=np.float32), 4, mode="polyphase_pchip"
        )
        aligned = align_rows_exp450_antigravity(
            (np.asarray([0.0, 2.0], dtype=np.float32), np.asarray([3.0], dtype=np.float32)),
            4,
            mode="polyphase_pchip",
        )
        np.testing.assert_array_equal(singleton, np.asarray([2.5, 2.5, 2.5, 2.5], dtype=np.float32))
        self.assertEqual(aligned.shape, (2, 4))
        self.assertEqual(aligned.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(aligned[1], np.asarray([3.0, 3.0, 3.0, 3.0], dtype=np.float32))

    def test_adaptive_average_pooling_returns_local_mean_float32_arrays(self) -> None:
        """Replacing adaptive average pooling with linear resizing must fail."""
        one_dimensional = adaptive_average_1d_antigravity(
            np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32), 2, device="cpu"
        )
        two_dimensional = adaptive_average_2d_antigravity(
            np.arange(16, dtype=np.float32).reshape(1, 4, 4), 2, device="cpu"
        )
        np.testing.assert_array_equal(one_dimensional, np.asarray([[0.5, 2.5]], dtype=np.float32))
        np.testing.assert_array_equal(
            two_dimensional,
            np.asarray([[[2.5, 4.5], [10.5, 12.5]]], dtype=np.float32),
        )
        self.assertEqual(one_dimensional.dtype, np.dtype(np.float32))
        self.assertEqual(two_dimensional.dtype, np.dtype(np.float32))

    def test_mps_is_selected_without_cpu_parity_when_available(self) -> None:
        """Adding a parity gate or selecting CPU despite MPS must fail."""
        with patch.object(
            backend_antigravity,
            "_probe_antigravity",
            return_value={
                "torch_version": "2.13.0",
                "mps_available": True,
                "mlx_available": True,
                "logical_cpu_count": 12,
            },
        ):
            decision = assess_exp450_backend_antigravity()
        self.assertEqual(decision.tensor_backend, "mps")
        self.assertEqual(decision.cpu_workers, 11)
        self.assertEqual(decision.inner_numerical_threads, 1)

    def test_mps_pooling_completes_with_finite_exact_shape_when_available(self) -> None:
        """An MPS pooling failure is observable without CPU--MPS comparison."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        result = adaptive_average_1d_antigravity(
            np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32), 3, device="mps"
        )
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(math.isfinite(float(result.sum())))

    def test_mps_pooling_falls_back_for_both_native_nondivisible_error_wordings(self) -> None:
        """Rejecting either observed MPS wording must not bypass on-device windows."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        messages = (
            "Adaptive pool MPS: input sizes must be divisible by output sizes.",
            "Adaptive pool MPS: output sizes must be divisible by input sizes.",
            "Adaptive pool MPS: Input height and width must both be greater than, or equal to, or lesser than output height and width.",
        )
        values = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        expected = np.asarray(
            [[[2.5, 3.5, 4.5], [6.5, 7.5, 8.5], [10.5, 11.5, 12.5]]],
            dtype=np.float32,
        )
        for message in messages:
            with self.subTest(message=message), patch.object(
                transforms_antigravity.torch_functional,
                "adaptive_avg_pool2d",
                side_effect=RuntimeError(message),
            ), patch.object(
                transforms_antigravity,
                "_adaptive_pool_windows_antigravity",
                wraps=transforms_antigravity._adaptive_pool_windows_antigravity,
            ) as fallback:
                result = adaptive_average_2d_antigravity(values, 3, device="mps")
                np.testing.assert_array_equal(result, expected)
                self.assertEqual(fallback.call_count, 1)
                self.assertEqual(fallback.call_args.args[0].device.type, "mps")

    def test_mps_pooling_handles_mixed_axis_pool_limit_on_device(self) -> None:
        """A mixed down/up MPS pool must use local windows instead of CPU."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        values = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
        with patch.object(
            transforms_antigravity,
            "_adaptive_pool_windows_antigravity",
            wraps=transforms_antigravity._adaptive_pool_windows_antigravity,
        ) as fallback:
            result = adaptive_average_2d_antigravity(values, 3, device="mps")
        self.assertEqual(result.shape, (1, 3, 3))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(fallback.call_args.args[0].device.type, "mps")

    def test_mps_pooling_reraises_unrelated_native_runtime_error(self) -> None:
        """Broadly swallowing an unrelated MPS failure must fail this guard."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        with patch.object(
            transforms_antigravity.torch_functional,
            "adaptive_avg_pool2d",
            side_effect=RuntimeError("Adaptive pool MPS: unrelated native failure"),
        ), patch.object(
            transforms_antigravity,
            "_adaptive_pool_windows_antigravity",
            wraps=transforms_antigravity._adaptive_pool_windows_antigravity,
        ) as fallback:
            with self.assertRaisesRegex(RuntimeError, "unrelated native failure"):
                adaptive_average_2d_antigravity(
                    np.arange(16, dtype=np.float32).reshape(1, 4, 4), 3, device="mps"
                )
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
