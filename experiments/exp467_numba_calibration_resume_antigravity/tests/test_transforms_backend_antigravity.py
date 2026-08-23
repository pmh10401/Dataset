"""RED/GREEN tests for Exp467 pure transforms and backend selection."""

from __future__ import annotations

import math
import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as torch_functional

from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import (
    resample_series_paper,
)

from experiments.exp467_numba_calibration_resume_antigravity import (
    backend_antigravity,
    transforms_antigravity,
)
from experiments.exp467_numba_calibration_resume_antigravity.backend_antigravity import (
    assess_exp467_backend_antigravity,
)
from experiments.exp467_numba_calibration_resume_antigravity.transforms_antigravity import (
    adaptive_average_1d_antigravity,
    adaptive_average_2d_antigravity,
    align_rows_exp450_antigravity,
    align_rows_exp467_antigravity,
    resample_series_exp450_antigravity,
    resample_series_exp467_antigravity,
)


class Exp467TransformsBackendTests(unittest.TestCase):
    def test_control_resampler_is_byte_exact_to_exp449(self) -> None:
        """Changing control delegation to any new interpolation must fail."""
        row = np.asarray([0.0, 3.0, -1.0, 2.0], dtype=np.float32)
        actual = resample_series_exp467_antigravity(row, 11, mode="linear")
        expected = resample_series_paper(row, 11)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        actual_alias = resample_series_exp450_antigravity(row, 11, mode="linear")
        self.assertEqual(actual_alias.tobytes(), expected.tobytes())

    def test_hybrid_upsampling_preserves_monotone_shape(self) -> None:
        """Changing the PCHIP branch to an overshooting interpolator must fail."""
        row = np.asarray([0.0, 0.2, 0.7, 1.0], dtype=np.float32)
        result = resample_series_exp467_antigravity(row, 31, mode="polyphase_pchip")
        self.assertEqual(result.shape, (31,))
        self.assertTrue(np.all(np.diff(result) >= 0.0))
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_hybrid_downsampling_uses_reduced_polyphase_factors(self) -> None:
        """Changing the downsampling path away from SciPy polyphase must fail."""
        row = np.asarray([0.0, 1.0, -1.0, 3.0, 2.0, -2.0], dtype=np.float32)
        expected = scipy.signal.resample_poly(row, 2, 3).astype(np.float32)
        actual = resample_series_exp467_antigravity(row, 4, mode="polyphase_pchip")
        np.testing.assert_array_equal(actual, expected)

    def test_hybrid_singleton_and_alignment_have_exact_float32_shapes(self) -> None:
        """Dropping singleton expansion or row stacking must fail."""
        singleton = resample_series_exp467_antigravity(
            np.asarray([2.5], dtype=np.float32), 4, mode="polyphase_pchip"
        )
        aligned = align_rows_exp467_antigravity(
            (np.asarray([0.0, 2.0], dtype=np.float32), np.asarray([3.0], dtype=np.float32)),
            4,
            mode="polyphase_pchip",
        )
        np.testing.assert_array_equal(singleton, np.asarray([2.5, 2.5, 2.5, 2.5], dtype=np.float32))
        self.assertEqual(aligned.shape, (2, 4))
        self.assertEqual(aligned.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(aligned[1], np.asarray([3.0, 3.0, 3.0, 3.0], dtype=np.float32))

        aligned_alias = align_rows_exp450_antigravity(
            (np.asarray([0.0, 2.0], dtype=np.float32), np.asarray([3.0], dtype=np.float32)),
            4,
            mode="polyphase_pchip",
        )
        np.testing.assert_array_equal(aligned_alias, aligned)

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
                "mlx_version": "0.32.0",
                "mlx_metal_available": True,
                "mlx_metal_version": "0.32.0",
                "mlx_device": "Device(gpu, 0)",
                "logical_cpu_count": 12,
            },
        ):
            decision = assess_exp467_backend_antigravity()
        self.assertEqual(decision.tensor_backend, "mps")
        self.assertEqual(decision.cpu_workers, 11)
        self.assertEqual(decision.inner_numerical_threads, 1)

    def test_cpu_is_selected_when_mps_is_unavailable(self) -> None:
        """Selecting MPS when unavailable must fail."""
        with patch.object(
            backend_antigravity,
            "_probe_antigravity",
            return_value={
                "torch_version": "2.13.0",
                "mps_available": False,
                "mlx_available": False,
                "logical_cpu_count": 8,
            },
        ):
            decision = assess_exp467_backend_antigravity()
        self.assertEqual(decision.tensor_backend, "cpu")
        self.assertEqual(decision.cpu_workers, 7)
        self.assertEqual(decision.inner_numerical_threads, 1)

    def test_mps_pooling_completes_with_finite_exact_shape_when_available(self) -> None:
        """An MPS pooling failure is observable without CPU--MPS comparison."""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        result = adaptive_average_1d_antigravity(
            np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32), 3, device="mps"
        )
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(math.isfinite(float(result.sum())))

    def test_mps_1d_pooling_falls_back_to_cpu_on_nondivisible_error(self) -> None:
        """1D pooling on MPS falls back to native CPU adaptive pooling without scalar windows."""
        original_pool1d = torch_functional.adaptive_avg_pool1d

        def fail_only_on_mps_1d(tensor: torch.Tensor, output_size: int) -> torch.Tensor:
            if tensor.device.type == "mps":
                raise RuntimeError(
                    "Adaptive pool MPS: input sizes must be divisible by output sizes."
                )
            return original_pool1d(tensor, output_size)

        with patch.object(
            transforms_antigravity.torch_functional,
            "adaptive_avg_pool1d",
            side_effect=fail_only_on_mps_1d,
        ):
            actual = adaptive_average_1d_antigravity(
                np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32), 3, device="mps"
            )
        expected = np.asarray([[0.5, 1.5, 2.5]], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(actual.shape, (1, 3))
        self.assertEqual(actual.dtype, np.dtype(np.float32))

    def test_mps_2d_pooling_falls_back_for_all_observed_native_nondivisible_error_wordings(
        self,
    ) -> None:
        """MPS 2D pooling falls back to native CPU adaptive pooling for all 3 observed shape errors."""
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
        original_pool2d = torch_functional.adaptive_avg_pool2d

        for message in messages:
            def make_failer(msg: str):
                def fail_on_mps(tensor: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
                    if tensor.device.type == "mps":
                        raise RuntimeError(msg)
                    return original_pool2d(tensor, output_size)
                return fail_on_mps

            with self.subTest(message=message), patch.object(
                transforms_antigravity.torch_functional,
                "adaptive_avg_pool2d",
                side_effect=make_failer(message),
            ):
                result = adaptive_average_2d_antigravity(values, 3, device="mps")
                np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
                self.assertEqual(result.shape, (1, 3, 3))
                self.assertEqual(result.dtype, np.dtype(np.float32))

    def test_mps_2d_pooling_handles_mixed_axis_pool_limit(self) -> None:
        """A mixed down/up MPS pool falls back to native CPU adaptive pooling."""
        values = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
        expected_cpu = adaptive_average_2d_antigravity(values, 3, device="cpu")

        original_pool2d = torch_functional.adaptive_avg_pool2d

        def fail_mixed_on_mps(tensor: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
            if tensor.device.type == "mps":
                raise RuntimeError(
                    "Adaptive pool MPS: Input height and width must both be greater than, or equal to, or lesser than output height and width."
                )
            return original_pool2d(tensor, output_size)

        with patch.object(
            transforms_antigravity.torch_functional,
            "adaptive_avg_pool2d",
            side_effect=fail_mixed_on_mps,
        ):
            result = adaptive_average_2d_antigravity(values, 3, device="mps")

        self.assertEqual(result.shape, (1, 3, 3))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result, expected_cpu, rtol=1e-5, atol=1e-5)

    def test_mps_pooling_reraises_unrelated_native_runtime_error(self) -> None:
        """Broadly swallowing an unrelated MPS failure must fail this guard."""
        values_2d = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        with patch.object(
            transforms_antigravity.torch_functional,
            "adaptive_avg_pool2d",
            side_effect=RuntimeError("Adaptive pool MPS: unrelated native failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unrelated native failure"):
                adaptive_average_2d_antigravity(values_2d, 3, device="mps")

        values_1d = np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
        with patch.object(
            transforms_antigravity.torch_functional,
            "adaptive_avg_pool1d",
            side_effect=RuntimeError("Adaptive pool MPS: out of memory"),
        ):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                adaptive_average_1d_antigravity(values_1d, 3, device="mps")

    def test_backend_assessment_records_typed_provenance_record(self) -> None:
        """Backend assessment must record tensor eligibility, overhead, parity, fallback, and benchmark requirement."""
        with patch.object(
            backend_antigravity,
            "_probe_antigravity",
            return_value={
                "torch_version": "2.13.0",
                "mps_available": True,
                "mlx_available": False,
                "logical_cpu_count": 12,
            },
        ):
            decision = assess_exp467_backend_antigravity()
        self.assertEqual(decision.tensor_backend, "mps")
        self.assertEqual(decision.cpu_workers, 11)
        self.assertEqual(decision.inner_numerical_threads, 1)
        self.assertTrue(decision.tensor_eligible)
        self.assertTrue(decision.tensor_eligibility)
        self.assertEqual(
            decision.transfer_compilation_overhead,
            "bounded_device_transfer_and_cpu_pool_fallback",
        )
        self.assertEqual(
            decision.deterministic_numerical_parity,
            "finite_scores_identical_candidate_indices_within_tolerance",
        )
        self.assertEqual(
            decision.parity_numerical_requirements,
            "finite_scores_identical_candidate_indices_within_tolerance",
        )
        self.assertEqual(
            decision.canonical_cpu_fallback,
            "pytorch_cpu_native_adaptive_avg_pool",
        )
        self.assertEqual(
            decision.benchmark_requirement,
            "mlx_or_mps_speedup_ge_1_1x_with_device_transfer",
        )
        self.assertEqual(
            decision.benchmark_required,
            "mlx_or_mps_speedup_ge_1_1x_with_device_transfer",
        )
        record = decision.artifact_record()
        mlx_assessment = record.pop("mlx_assessment")
        numba_calibration = record.pop("numba_calibration")
        self.assertEqual(
            record,
            {
                "tensor_backend": "mps",
                "torch_version": "2.13.0",
                "mps_available": True,
                "mlx_available": False,
                "logical_cpu_count": 12,
                "cpu_workers": 11,
                "inner_numerical_threads": 1,
                "tensor_eligible": True,
                "transfer_compilation_overhead": "bounded_device_transfer_and_cpu_pool_fallback",
                "deterministic_numerical_parity": "finite_scores_identical_candidate_indices_within_tolerance",
                "canonical_cpu_fallback": "pytorch_cpu_native_adaptive_avg_pool",
                "benchmark_requirement": "mlx_or_mps_speedup_ge_1_1x_with_device_transfer",
            },
        )
        self.assertFalse(mlx_assessment["mlx_available"])
        self.assertEqual(mlx_assessment["mlx_version"], "unavailable")
        self.assertEqual(mlx_assessment["parity_status"], "not_run_gate_1_failed")
        self.assertEqual(numba_calibration["selected_backend"], "numba")
        self.assertEqual(numba_calibration["fast_path_dtype"], "float64")
        self.assertEqual(
            mlx_assessment["speed_benchmark_status"], "not_run_gate_1_failed"
        )

    def test_mlx_assessment_records_factual_gate_one_failure(self) -> None:
        """Unavailable exact MLX science records no invented parity or speed result."""
        with patch.object(
            backend_antigravity,
            "_probe_antigravity",
            return_value={
                "torch_version": "2.13.0",
                "mps_available": True,
                "mlx_available": True,
                "mlx_version": "0.32.0",
                "mlx_metal_available": True,
                "mlx_metal_version": "0.32.0",
                "mlx_device": "Device(gpu, 0)",
                "logical_cpu_count": 12,
            },
        ):
            decision = assess_exp467_backend_antigravity()

        self.assertEqual(
            decision.artifact_record().get("mlx_assessment"),
            {
                "schema": "exp467-mlx-assessment-antigravity.v1",
                "tensor_eligibility": "partial_tensor_substages_only",
                "exact_frozen_algorithm_expressible": False,
                "exact_expression_status": "gate_failed",
                "exact_expression_reason": "frozen_scipy_sklearn_aeon_and_pytorch_score_path_has_no_validated_exact_mlx_expression",
                "mlx_available": True,
                "mlx_version": "0.32.0",
                "mlx_metal_available": True,
                "mlx_metal_version": "0.32.0",
                "mlx_device": "Device(gpu, 0)",
                "transfer_compilation_overhead_status": "not_run_gate_1_failed",
                "transfer_compilation_overhead_result": "not_run_without_exact_mlx_expression",
                "parity_requirement": "same_input_and_seed_finite_outputs_identical_candidates_at_every_preregistered_alpha_scores_within_tolerance",
                "parity_status": "not_run_gate_1_failed",
                "parity_result": "not_run_without_exact_mlx_expression",
                "speed_benchmark_requirement": "mlx_at_least_1.1x_faster_including_transfer_compilation_and_synchronization",
                "speed_benchmark_status": "not_run_gate_1_failed",
                "speed_benchmark_result": "not_run_without_exact_mlx_expression",
                "canonical_cpu_mps_fallback": "frozen_scipy_sklearn_aeon_cpu_plus_pytorch_mps_with_native_cpu_pool_fallback",
                "fallback_reason": "mlx_gate_1_failed_exact_frozen_algorithm_not_expressible_and_validated",
            },
        )

    def test_mps_pooling_reraises_adversarial_error_with_recognized_substring_and_extra_text(
        self,
    ) -> None:
        """Adversarial errors containing recognized substrings plus extra text must propagate."""
        adversarial_messages = (
            "Adaptive pool MPS: input sizes must be divisible by output sizes extra adversarial text",
            "Adaptive pool MPS: input sizes must be divisible by output sizes. Memory corruption occurred",
            "Adaptive pool MPS: output sizes must be divisible by input sizes. Unrelated failure",
            "Adaptive pool MPS: Input height and width must both be greater than, or equal to, or lesser than output height and width. System crash",
            "Corrupted prefix: Adaptive pool MPS: input sizes must be divisible by output sizes",
            "Adaptive pool MPS: input sizes must be divisible by output sizes..",
        )
        values_2d = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        for message in adversarial_messages:
            with self.subTest(message=message), patch.object(
                transforms_antigravity.torch_functional,
                "adaptive_avg_pool2d",
                side_effect=RuntimeError(message),
            ):
                with self.assertRaises(RuntimeError) as context:
                    adaptive_average_2d_antigravity(values_2d, 3, device="mps")
                self.assertEqual(str(context.exception), message)

    def test_regression_guard_fallback_never_invokes_scalar_window_stack(self) -> None:
        """1D, 2D, and mixed-axis fallbacks must use native CPU pooling and never invoke torch.stack."""
        original_pool1d = torch_functional.adaptive_avg_pool1d
        original_pool2d = torch_functional.adaptive_avg_pool2d

        def fail_1d_on_mps(tensor: torch.Tensor, output_size: int) -> torch.Tensor:
            if tensor.device.type == "mps":
                raise RuntimeError(
                    "Adaptive pool MPS: input sizes must be divisible by output sizes."
                )
            return original_pool1d(tensor, output_size)

        def fail_2d_on_mps(tensor: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
            if tensor.device.type == "mps":
                raise RuntimeError(
                    "Adaptive pool MPS: input sizes must be divisible by output sizes."
                )
            return original_pool2d(tensor, output_size)

        def fail_mixed_on_mps(tensor: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
            if tensor.device.type == "mps":
                raise RuntimeError(
                    "Adaptive pool MPS: Input height and width must both be greater than, or equal to, or lesser than output height and width."
                )
            return original_pool2d(tensor, output_size)

        # Patch torch.stack to fail loudly if any scalar-window stack is invoked
        with patch(
            "torch.stack",
            side_effect=AssertionError("scalar window torch.stack must not be invoked"),
        ):
            # 1. 1D fallback
            with patch.object(
                transforms_antigravity.torch_functional,
                "adaptive_avg_pool1d",
                side_effect=fail_1d_on_mps,
            ):
                res_1d = adaptive_average_1d_antigravity(
                    np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32), 3, device="mps"
                )
                self.assertEqual(res_1d.shape, (1, 3))

            # 2. 2D fallback
            with patch.object(
                transforms_antigravity.torch_functional,
                "adaptive_avg_pool2d",
                side_effect=fail_2d_on_mps,
            ):
                res_2d = adaptive_average_2d_antigravity(
                    np.arange(16, dtype=np.float32).reshape(1, 4, 4), 3, device="mps"
                )
                self.assertEqual(res_2d.shape, (1, 3, 3))

            # 3. Mixed-axis 2D fallback
            with patch.object(
                transforms_antigravity.torch_functional,
                "adaptive_avg_pool2d",
                side_effect=fail_mixed_on_mps,
            ):
                res_mixed = adaptive_average_2d_antigravity(
                    np.arange(8, dtype=np.float32).reshape(1, 2, 4), 3, device="mps"
                )
                self.assertEqual(res_mixed.shape, (1, 3, 3))

    def test_cpu_device_does_not_catch_runtime_error(self) -> None:
        """When device is CPU, runtime errors are not caught by MPS fallback."""
        values = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        with patch.object(
            transforms_antigravity.torch_functional,
            "adaptive_avg_pool2d",
            side_effect=RuntimeError(
                "Adaptive pool MPS: input sizes must be divisible by output sizes."
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "divisible"):
                adaptive_average_2d_antigravity(values, 3, device="cpu")

    def test_real_mps_bounded_timing_when_available(self) -> None:
        """Real MPS execution with non-divisible shape completes with bounded timing."""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        values = np.ones((1, 512, 513), dtype=np.float32)
        start_time = time.perf_counter()
        result = adaptive_average_2d_antigravity(values, (224, 224), device="mps")
        elapsed = time.perf_counter() - start_time
        self.assertEqual(result.shape, (1, 224, 224))
        self.assertTrue(np.isfinite(result).all())
        # CPU native adaptive pool fallback must be very fast (< 0.5s), unlike scalar loops (> 10s)
        self.assertLess(elapsed, 0.5)

    def test_invalid_inputs_raise_value_errors(self) -> None:
        """Invalid inputs must raise ValueError."""
        # Invalid device
        with self.assertRaises(ValueError):
            adaptive_average_1d_antigravity(np.ones((2, 4), dtype=np.float32), 2, device="cuda")
        with self.assertRaises(ValueError):
            adaptive_average_2d_antigravity(np.ones((2, 4, 4), dtype=np.float32), 2, device="cuda")

        # Nonfinite values
        with self.assertRaises(ValueError):
            adaptive_average_1d_antigravity(np.asarray([[np.nan, 1.0]], dtype=np.float32), 1, device="cpu")
        with self.assertRaises(ValueError):
            adaptive_average_2d_antigravity(np.asarray([[[np.inf, 1.0]]], dtype=np.float32), 1, device="cpu")

        # Invalid dimensions
        with self.assertRaises(ValueError):
            adaptive_average_1d_antigravity(np.ones((2, 3, 4), dtype=np.float32), 2, device="cpu")
        with self.assertRaises(ValueError):
            adaptive_average_2d_antigravity(np.ones((2, 4), dtype=np.float32), 2, device="cpu")

        # Invalid sizes
        with self.assertRaises(ValueError):
            adaptive_average_1d_antigravity(np.ones((2, 4), dtype=np.float32), 0, device="cpu")
        with self.assertRaises(ValueError):
            adaptive_average_2d_antigravity(np.ones((2, 4, 4), dtype=np.float32), -1, device="cpu")


if __name__ == "__main__":
    unittest.main()
