"""MLX eligibility evidence required before public-baseline full runs."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import platform
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .config import ALPHAS, CLASSICAL_METHOD_IDS, DEEP_METHOD_IDS, METHOD_IDS


@dataclass(frozen=True, slots=True)
class AccelerationRecord:
    method_id: str
    exact_stage: str
    condition_1_eligible: bool
    cpu_finite: bool
    mlx_finite: bool
    parity_passed: bool
    candidate_indices_equal_by_alpha: Mapping[str, bool]
    numerical_tolerance: float
    cpu_seconds: float | None
    mlx_seconds: float | None
    speed_ratio: float | None
    selected_backend: str
    fallback_reason: str
    versions: Mapping[str, str]
    device: str


def assess_mlx(method_id: str, smoke_input: np.ndarray) -> AccelerationRecord:
    """Record a fail-closed CPU decision when no exact MLX path is admissible."""
    if method_id not in METHOD_IDS:
        raise ValueError(f"unknown method for MLX assessment: {method_id}")
    values = np.asarray(smoke_input)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("MLX smoke input must be a finite two-dimensional array")
    if method_id in CLASSICAL_METHOD_IDS:
        reason = (
            "The frozen method is an official scikit-learn CPU estimator; an MLX "
            "rewrite would change the implementation and fails policy condition 1."
        )
        stage = "official scikit-learn estimator fit and score"
    elif method_id in DEEP_METHOD_IDS:
        reason = (
            "No exact MLX implementation preserves the frozen PyTorch initialization, "
            "optimizer, autograd, and training path; policy condition 1 is not met."
        )
        stage = "frozen PyTorch one-class training and score"
    else:  # pragma: no cover - guarded by METHOD_IDS
        raise AssertionError(method_id)
    versions = MappingProxyType(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": _version("scikit-learn"),
            "torch": _version("torch"),
            "mlx": _version("mlx"),
            "mlx_metal": _version("mlx-metal"),
        }
    )
    candidates = MappingProxyType(
        {format(alpha, ".12g"): False for alpha in ALPHAS}
    )
    return AccelerationRecord(
        method_id=method_id,
        exact_stage=stage,
        condition_1_eligible=False,
        cpu_finite=bool(np.isfinite(values).all()),
        mlx_finite=False,
        parity_passed=False,
        candidate_indices_equal_by_alpha=candidates,
        numerical_tolerance=1e-8,
        cpu_seconds=None,
        mlx_seconds=None,
        speed_ratio=None,
        selected_backend="cpu",
        fallback_reason=reason,
        versions=versions,
        device=f"CPU ({platform.machine()})",
    )


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


__all__ = ["AccelerationRecord", "assess_mlx"]
