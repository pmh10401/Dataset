"""CPU-versus-MLX parity and speed gate for Exp480 distances."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .contracts_antigravity import EvidenceRowAntigravity
from .policy_antigravity import (
    FamilyBalancedCandidatePolicyAntigravity,
    nearest_family_indices_antigravity,
)


def choose_backend_antigravity(
    *,
    finite: bool,
    indices_equal: bool,
    max_abs_error: float,
    tolerance: float,
    cpu_seconds: float,
    mlx_seconds: float,
) -> dict[str, object]:
    """Choose MLX only when every frozen parity and speed condition passes."""
    if not finite:
        reason = "nonfinite_output"
    elif not indices_equal:
        reason = "candidate_indices_changed"
    elif not math.isfinite(max_abs_error) or max_abs_error > tolerance:
        reason = "distance_tolerance_exceeded"
    elif cpu_seconds <= 0.0 or mlx_seconds <= 0.0:
        reason = "invalid_timing"
    elif cpu_seconds / mlx_seconds < 1.1:
        reason = "speedup_below_1_1x"
    else:
        return {
            "selected_backend": "mlx",
            "fallback_reason": None,
            "speedup": cpu_seconds / mlx_seconds,
        }
    speedup = cpu_seconds / mlx_seconds if cpu_seconds > 0.0 and mlx_seconds > 0.0 else 0.0
    return {
        "selected_backend": "cpu_numpy",
        "fallback_reason": reason,
        "speedup": speedup,
    }


def _package_version_antigravity(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _cpu_distances_antigravity(matrix: np.ndarray, queries: np.ndarray) -> np.ndarray:
    return np.linalg.norm(matrix[None, :, :] - queries[:, None, :], axis=2)


def _indices_antigravity(
    distances: np.ndarray, families: Sequence[str], k: int
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in nearest_family_indices_antigravity(row, families, k))
        for row in distances
    )


def assess_mlx_antigravity(
    rows: Sequence[EvidenceRowAntigravity],
    *,
    assessment_path: Path | None = None,
    tolerance: float = 1e-10,
) -> dict[str, object]:
    """Benchmark the exact dense-distance stage including transfers and sync."""
    policy = FamilyBalancedCandidatePolicyAntigravity().fit(rows)
    matrix = np.asarray(policy._matrix, dtype=np.float64)  # exact fitted CPU authority
    queries = matrix[: min(32, len(matrix))].copy()
    families = tuple(policy._families)
    k = min(policy.k, len(set(families)))
    started = perf_counter()
    cpu_distances = _cpu_distances_antigravity(matrix, queries)
    cpu_seconds = perf_counter() - started
    cpu_indices = _indices_antigravity(cpu_distances, families, k)
    base: dict[str, object] = {
        "schema": "exp480_mlx_assessment_v1",
        "operation": "standardized_dense_l2_distances_and_family_indices",
        "rows": len(matrix),
        "queries": len(queries),
        "tolerance": tolerance,
        "cpu_seconds": cpu_seconds,
        "mlx_version": _package_version_antigravity("mlx"),
        "mlx_metal_version": _package_version_antigravity("mlx-metal"),
        "device": None,
    }
    try:
        import mlx.core as mx

        base["device"] = str(mx.default_device())
        started = perf_counter()
        mlx_matrix = mx.array(matrix, dtype=mx.float64)
        mlx_queries = mx.array(queries, dtype=mx.float64)
        mlx_value = mx.linalg.norm(
            mlx_matrix[None, :, :] - mlx_queries[:, None, :], axis=2
        )
        mx.eval(mlx_value)
        mlx_distances = np.asarray(mlx_value, dtype=np.float64)
        mlx_seconds = perf_counter() - started
        mlx_indices = _indices_antigravity(mlx_distances, families, k)
        finite = bool(np.isfinite(cpu_distances).all() and np.isfinite(mlx_distances).all())
        indices_equal = mlx_indices == cpu_indices
        max_abs_error = float(np.max(np.abs(cpu_distances - mlx_distances)))
        choice = choose_backend_antigravity(
            finite=finite,
            indices_equal=indices_equal,
            max_abs_error=max_abs_error,
            tolerance=tolerance,
            cpu_seconds=cpu_seconds,
            mlx_seconds=mlx_seconds,
        )
        base.update(
            {
                "mlx_seconds": mlx_seconds,
                "finite": finite,
                "candidate_indices_equal": indices_equal,
                "max_abs_error": max_abs_error,
                **choice,
            }
        )
    except Exception as error:
        base.update(
            {
                "mlx_seconds": None,
                "finite": False,
                "candidate_indices_equal": False,
                "max_abs_error": None,
                "selected_backend": "cpu_numpy",
                "fallback_reason": f"mlx_unavailable:{type(error).__name__}",
                "fallback_detail": str(error),
                "speedup": None,
            }
        )
    if assessment_path is not None:
        target = Path(assessment_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(base, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return base


__all__ = ["assess_mlx_antigravity", "choose_backend_antigravity"]
