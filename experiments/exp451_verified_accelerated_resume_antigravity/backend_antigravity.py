"""MPS-first Exp451 tensor-backend readiness assessment.

The returned decision is the provenance record for the frozen selection: when
MPS is available, adaptive pooling and recurrence distances use MPS; SciPy
polyphase/PCHIP and sklearn/aeon stages remain CPU-only. MLX availability is
observed for provenance only and never becomes an execution option.
"""

from __future__ import annotations

import importlib
import os

from .contracts_antigravity import Exp451BackendDecision


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
        importlib.import_module("mlx.core")
        mlx_available = True
    except ImportError:
        mlx_available = False
    return {
        "torch_version": torch_version,
        "mps_available": mps_available,
        "mlx_available": mlx_available,
        "logical_cpu_count": os.cpu_count() or 1,
    }


def assess_exp451_backend_antigravity() -> Exp451BackendDecision:
    """Select MPS whenever available, otherwise retain the canonical CPU path."""
    observed = _probe_antigravity()
    selected = "mps" if observed["mps_available"] else "cpu"
    count = int(observed["logical_cpu_count"])
    return Exp451BackendDecision(
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
    )


__all__ = ["assess_exp451_backend_antigravity"]
