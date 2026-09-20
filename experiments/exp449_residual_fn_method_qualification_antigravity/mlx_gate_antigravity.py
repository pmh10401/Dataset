"""Truthful stage-specific CPU/MLX/PyTorch-MPS acceleration decision record."""
from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True, slots=True)
class MlxDecisionAntigravity:
    backend: Literal["cpu", "mlx", "mps"]
    reason: str
    tensor_eligible: bool
    mlx_version: str | None
    mlx_metal_version: str | None
    torch_version: str | None
    mps_built: bool | None
    mps_available: bool | None
    device: str | None
    parity_run: bool
    candidate_indices_identical: bool | None
    numerical_tolerance_met: bool | None
    benchmark_includes_transfer_sync_warmup_compile: bool
    speedup: float | None

    def __post_init__(self) -> None:
        if self.backend not in ("cpu", "mlx", "mps") or type(self.reason) is not str or not self.reason:
            raise ValueError("backend and reason are required")
        for field in ("tensor_eligible", "parity_run", "benchmark_includes_transfer_sync_warmup_compile"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be an exact bool")
        for field in ("mps_built", "mps_available", "candidate_indices_identical", "numerical_tolerance_met"):
            if getattr(self, field) is not None and type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be an exact bool or None")
        if self.speedup is not None and (type(self.speedup) is not float or not math.isfinite(self.speedup) or self.speedup < 0.0):
            raise ValueError("speedup must be a finite exact float or None")


def _package_probe_antigravity(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_probe_antigravity() -> tuple[bool | None, bool | None]:
    try:
        import torch
    except ImportError:
        return None, None
    return bool(torch.backends.mps.is_built()), bool(torch.backends.mps.is_available())


def assess_mlx_antigravity(*, package_probe: Callable[[str], str | None] | None = None, torch_probe: Callable[[], tuple[bool, bool]] | None = None, stage_probe: Callable[[], dict[str, object]] | None = None) -> MlxDecisionAntigravity:
    """Record the complete gate; aeon CPU transforms are tensor-ineligible today."""
    packages = package_probe or _package_probe_antigravity
    mlx_version = packages("mlx")
    mlx_metal_version = packages("mlx-metal")
    torch_version = packages("torch")
    mps_built, mps_available = (torch_probe or _torch_probe_antigravity)()
    stage = stage_probe() if stage_probe is not None else {"tensor_eligible": False}
    if type(stage) is not dict:
        stage = {}
    requested_backend = stage.get("backend", "mlx")
    if requested_backend not in ("mlx", "mps"):
        return MlxDecisionAntigravity("cpu", "invalid_acceleration_backend", False, mlx_version, mlx_metal_version, torch_version, mps_built, mps_available, None, False, None, None, False, None)
    tensor_eligible = stage.get("tensor_eligible") is True
    if not tensor_eligible:
        if requested_backend == "mlx" and mlx_version is None and mlx_metal_version is None:
            return MlxDecisionAntigravity("cpu", "mlx_and_mlx_metal_absent_in_fixed_environment", False, None, None, torch_version, mps_built, mps_available, None, False, None, None, False, None)
        return MlxDecisionAntigravity("cpu", "tensor_stage_ineligible_cpu_canonical", False, mlx_version, mlx_metal_version, torch_version, mps_built, mps_available, None, False, None, None, False, None)
    if requested_backend == "mlx" and (mlx_version is None or mlx_metal_version is None):
        return MlxDecisionAntigravity("cpu", "mlx_runtime_incomplete", True, mlx_version, mlx_metal_version, torch_version, mps_built, mps_available, None, False, None, None, False, None)
    if requested_backend == "mps" and (mps_built is not True or mps_available is not True):
        return MlxDecisionAntigravity("cpu", "mps_unavailable", True, mlx_version, mlx_metal_version, torch_version, mps_built, mps_available, None, False, None, None, False, None)
    finite = stage.get("finite") is True
    identical = stage.get("candidate_indices_identical")
    tolerance = stage.get("numerical_tolerance_met")
    inclusive = stage.get("benchmark_includes_transfer_sync_warmup_compile") is True
    raw_speedup = stage.get("speedup")
    speedup = float(raw_speedup) if type(raw_speedup) in (int, float) and type(raw_speedup) is not bool else None
    valid_speedup = speedup is not None and math.isfinite(speedup) and speedup >= 0.0
    common = ("cpu", True, mlx_version, mlx_metal_version, torch_version, mps_built, mps_available, None, True, identical if type(identical) is bool else None, tolerance if type(tolerance) is bool else None, inclusive, speedup if valid_speedup else None)
    if not inclusive:
        return MlxDecisionAntigravity(*common[:1], "benchmark_not_inclusive", *common[1:])
    if not finite:
        return MlxDecisionAntigravity(*common[:1], "nonfinite_outputs", *common[1:])
    if identical is not True:
        return MlxDecisionAntigravity(*common[:1], "candidate_indices_not_identical", *common[1:])
    if tolerance is not True:
        return MlxDecisionAntigravity(*common[:1], "numerical_tolerance_not_met", *common[1:])
    if not valid_speedup:
        return MlxDecisionAntigravity(*common[:1], "invalid_speedup", *common[1:])
    if speedup < 1.1:
        return MlxDecisionAntigravity(*common[:1], "speedup_below_1_1", *common[1:])
    return MlxDecisionAntigravity(requested_backend, f"complete_{requested_backend}_gate_passed", True, mlx_version, mlx_metal_version, torch_version, mps_built, mps_available, requested_backend, True, True, True, True, speedup)
