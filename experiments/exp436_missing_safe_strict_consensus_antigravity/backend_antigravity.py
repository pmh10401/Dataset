"""Neutral Exp436 accelerator selection and scalar dispatch.

This boundary deliberately does not construct or import either optional engine.
The runtime creates a validated selected engine once and supplies it here.
"""

from __future__ import annotations

from numbers import Real

import numpy as np

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import BackendBenchmarkEvidence, BackendDecision
from .features_antigravity import score_channel_cpu


def _is_selected_accelerator_device(backend: str, value: object) -> bool:
    device = str(value).strip().lower()
    if not device or "cpu" in device:
        return False
    return ("gpu" in device or "metal" in device) if backend == "mlx" else ("mps" in device or "metal" in device)


def select_backend(mlx: BackendBenchmarkEvidence, mps: BackendBenchmarkEvidence) -> BackendDecision:
    """Choose the fastest passing evidence, with the frozen lexical tie break."""

    if not isinstance(mlx, BackendBenchmarkEvidence) or mlx.backend != "mlx":
        raise ValueError("MLX benchmark evidence is required")
    if not isinstance(mps, BackendBenchmarkEvidence) or mps.backend != "mps":
        raise ValueError("MPS benchmark evidence is required")
    passing = tuple(item for item in (mlx, mps) if item.passed)
    if not passing:
        return BackendDecision(mlx, mps, "cpu", False, "no_accelerator_passed")
    chosen = min(passing, key=lambda item: (item.selected_device_median_seconds, item.backend))
    return BackendDecision(mlx, mps, chosen.backend, True, "selected_fastest_passing_backend")


def _validate_accelerator_engine(decision: BackendDecision, engine: object | None) -> object:
    if engine is None:
        raise ValueError("adopted backend requires a caller-supplied validated engine")
    selected = decision.mlx if decision.backend == "mlx" else decision.mps
    if getattr(engine, "backend", None) != decision.backend:
        raise ValueError("engine backend does not match decision")
    if getattr(engine, "metal_available", None) is not True:
        raise ValueError("engine Metal provenance does not match decision")
    for field in ("device", "device_info"):
        if getattr(engine, field, None) != getattr(selected, field):
            raise ValueError(f"engine {field} provenance does not match decision")
    if not _is_selected_accelerator_device(decision.backend, getattr(engine, "device", None)):
        raise ValueError("engine device is not backend-specific accelerator provenance")
    if decision.backend == "mlx":
        for field in ("mlx_version", "mlx_metal_version"):
            if getattr(engine, field, None) != getattr(selected, field):
                raise ValueError(f"engine {field} provenance does not match decision")
    elif getattr(engine, "pytorch_version", None) != selected.pytorch_version:
        raise ValueError("engine pytorch_version provenance does not match decision")
    return engine


def validate_backend_engine(decision: BackendDecision, engine: object | None) -> object:
    """Public runtime gate for the one adopted engine instance.

    CPU has deliberately no engine.  Accelerator callers must validate the
    exact selected object once and retain that same identity for the serial
    run; this boundary never manufactures a fallback.
    """

    if not isinstance(decision, BackendDecision):
        raise ValueError("decision must be a BackendDecision")
    if decision.backend == "cpu":
        if engine is not None:
            raise ValueError("CPU backend does not accept an accelerator engine")
        return None
    return _validate_accelerator_engine(decision, engine)


def score_with_backend(decision: BackendDecision, query: object, references: object, k: int, *, engine: object | None = None) -> float:
    """Score exactly one float32 query; accelerator errors are never hidden by CPU."""

    if not isinstance(decision, BackendDecision):
        raise ValueError("decision must be a BackendDecision")
    if type(k) is not int or k != FROZEN_CONFIG.neighbor_count:
        raise ValueError("k must equal the frozen neighbor count")
    if decision.backend == "cpu":
        score = score_channel_cpu(query, references, k)
    else:
        active = _validate_accelerator_engine(decision, engine)
        method = getattr(active, "score", None)
        if not callable(method):
            raise ValueError("engine must expose score")
        score = method(query, references, k, batch_size=decision.selected_batch_size)
    if isinstance(score, bool) or not isinstance(score, Real) or not np.isfinite(float(score)):
        raise ValueError("backend scalar score must be finite")
    return float(score)


__all__ = ["score_with_backend", "select_backend", "validate_backend_engine"]
