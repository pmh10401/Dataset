"""MPS-first Exp467 tensor-backend readiness assessment.

The returned decision is the provenance record for the frozen selection: when
MPS is available, adaptive pooling and recurrence distances use MPS; SciPy
polyphase/PCHIP and sklearn/aeon stages remain CPU-only. MLX package, metal and
device availability are recorded with an explicit gate-one failure: the full
frozen score path has no validated exact MLX expression, so parity and speed
claims remain truthfully unrun.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os

from paper.equipment_anomaly_detection.numba_calibration_antigravity import (
    numba_calibration_backend_antigravity,
)

from .contracts_antigravity import Exp467BackendDecision


def _probe_antigravity() -> dict[str, object]:
    """Observe readiness without benchmarking or comparing tensor outputs."""
    try:
        torch = importlib.import_module("torch")
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_available = bool(mps_backend is not None and mps_backend.is_available())
        torch_version = str(torch.__version__)
    except (ImportError, AttributeError):
        torch_version = "unavailable"
        mps_available = False
    try:
        mlx = importlib.import_module("mlx.core")
        mlx_available = True
        try:
            mlx_version = importlib.metadata.version("mlx")
        except importlib.metadata.PackageNotFoundError:
            mlx_version = "unavailable"
        try:
            mlx_metal_version = importlib.metadata.version("mlx-metal")
        except importlib.metadata.PackageNotFoundError:
            mlx_metal_version = "unavailable"
        metal = getattr(mlx, "metal", None)
        mlx_metal_available = bool(
            metal is not None
            and callable(getattr(metal, "is_available", None))
            and metal.is_available()
        )
        default_device = getattr(mlx, "default_device", None)
        mlx_device = str(default_device()) if callable(default_device) else "unavailable"
    except ImportError:
        mlx_available = False
        mlx_version = "unavailable"
        mlx_metal_available = False
        mlx_metal_version = "unavailable"
        mlx_device = "unavailable"
    return {
        "torch_version": torch_version,
        "mps_available": mps_available,
        "mlx_available": mlx_available,
        "mlx_version": mlx_version,
        "mlx_metal_available": mlx_metal_available,
        "mlx_metal_version": mlx_metal_version,
        "mlx_device": mlx_device,
        "logical_cpu_count": os.cpu_count() or 1,
    }


def assess_exp467_backend_antigravity() -> Exp467BackendDecision:
    """Select MPS whenever available, otherwise retain the canonical CPU path."""
    observed = _probe_antigravity()
    selected = "mps" if observed["mps_available"] else "cpu"
    count = int(observed["logical_cpu_count"])
    return Exp467BackendDecision(
        tensor_backend=selected,
        torch_version=str(observed["torch_version"]),
        mps_available=bool(observed["mps_available"]),
        mlx_available=bool(observed["mlx_available"]),
        logical_cpu_count=count,
        cpu_workers=max(1, count - 1),
        inner_numerical_threads=1,
        tensor_eligible=True,
        transfer_compilation_overhead="bounded_device_transfer_and_cpu_pool_fallback",
        deterministic_numerical_parity="finite_scores_identical_candidate_indices_within_tolerance",
        canonical_cpu_fallback="pytorch_cpu_native_adaptive_avg_pool",
        benchmark_requirement="mlx_or_mps_speedup_ge_1_1x_with_device_transfer",
        mlx_assessment={
            "schema": "exp467-mlx-assessment-antigravity.v1",
            "tensor_eligibility": "partial_tensor_substages_only",
            "exact_frozen_algorithm_expressible": False,
            "exact_expression_status": "gate_failed",
            "exact_expression_reason": "frozen_scipy_sklearn_aeon_and_pytorch_score_path_has_no_validated_exact_mlx_expression",
            "mlx_available": bool(observed["mlx_available"]),
            "mlx_version": str(observed.get("mlx_version", "unavailable")),
            "mlx_metal_available": bool(observed.get("mlx_metal_available", False)),
            "mlx_metal_version": str(observed.get("mlx_metal_version", "unavailable")),
            "mlx_device": str(observed.get("mlx_device", "unavailable")),
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
        numba_calibration=numba_calibration_backend_antigravity(),
    )


__all__ = ["assess_exp467_backend_antigravity"]
