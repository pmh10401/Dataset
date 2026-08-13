"""Fail-closed CPU/accelerator selection for the dense LocalGap stage only."""

from __future__ import annotations

import importlib.util
import math
import platform
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity


BackendNameAntigravity = Literal["cpu", "mlx", "mps"]


def _exact_bool_antigravity(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be an exact bool")
    return value


def _finite_float_antigravity(value: object, field: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"{field} must be an exact finite {'positive ' if positive else ''}float")
    return value


@dataclass(frozen=True, slots=True)
class BackendProbeAntigravity:
    name: BackendNameAntigravity
    available: bool
    finite: bool
    candidate_indices_match: bool
    maximum_normalized_error: float
    inclusive_cpu_seconds: float
    inclusive_accelerator_seconds: float
    versions: tuple[tuple[str, str], ...]
    device: str
    reason: str

    def __post_init__(self) -> None:
        if self.name not in {"mlx", "mps"}:
            raise ValueError("probe name must be mlx or mps")
        _exact_bool_antigravity(self.available, "available")
        _exact_bool_antigravity(self.finite, "finite")
        _exact_bool_antigravity(self.candidate_indices_match, "candidate_indices_match")
        _finite_float_antigravity(self.maximum_normalized_error, "maximum_normalized_error")
        _finite_float_antigravity(self.inclusive_cpu_seconds, "inclusive_cpu_seconds", positive=True)
        _finite_float_antigravity(self.inclusive_accelerator_seconds, "inclusive_accelerator_seconds", positive=True)
        if type(self.versions) is not tuple or any(type(item) is not tuple or len(item) != 2 or any(type(part) is not str or not part for part in item) for item in self.versions):
            raise ValueError("versions must be exact non-empty string pairs")
        if type(self.device) is not str or type(self.reason) is not str:
            raise ValueError("device and reason must be exact strings")

    @property
    def speedup(self) -> float:
        return self.inclusive_cpu_seconds / self.inclusive_accelerator_seconds


@dataclass(frozen=True, slots=True)
class BackendDecisionAntigravity:
    selected_name: BackendNameAntigravity
    probes: tuple[BackendProbeAntigravity, ...]
    fallback_reason: str

    def __post_init__(self) -> None:
        if self.selected_name not in {"cpu", "mlx", "mps"} or type(self.probes) is not tuple or any(type(item) is not BackendProbeAntigravity for item in self.probes) or type(self.fallback_reason) is not str:
            raise ValueError("backend decision is malformed")
        if self.selected_name == "cpu" and not self.fallback_reason:
            raise ValueError("CPU decision requires a fallback reason")
        if self.selected_name != "cpu" and (self.fallback_reason or len(self.probes) != 1 or self.probes[0].name != self.selected_name):
            raise ValueError("accelerator decision must have one selected passing probe")


class LocalGapBackendAntigravity(Protocol):
    def fit(self, reference: np.ndarray) -> "LocalGapBackendAntigravity": ...
    def score(self, query: np.ndarray) -> np.ndarray: ...


class _CpuLocalGapBackendAntigravity:
    def __init__(self) -> None:
        self._model: LocalGapModelAntigravity | None = None

    def fit(self, reference: np.ndarray) -> "_CpuLocalGapBackendAntigravity":
        self._model = LocalGapModelAntigravity(reference, neighbors=3)
        return self

    def score(self, query: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("LocalGap backend must be fit before score")
        scores = np.asarray(self._model.score(query), dtype=float)
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError("LocalGap returned invalid scores")
        return scores


def build_local_gap_backend_antigravity(name: BackendNameAntigravity) -> LocalGapBackendAntigravity:
    if name != "cpu":
        # No approximate accelerator implementation is permitted until an exact gate exists.
        raise ValueError("only CPU LocalGap is qualified")
    return _CpuLocalGapBackendAntigravity()


def decide_backend_antigravity(probe: BackendProbeAntigravity) -> BackendDecisionAntigravity:
    if type(probe) is not BackendProbeAntigravity:
        raise ValueError("probe must be an exact BackendProbeAntigravity")
    passes = probe.available and probe.finite and probe.candidate_indices_match and probe.maximum_normalized_error <= 1e-5 and probe.speedup >= 1.1
    if passes:
        return BackendDecisionAntigravity(probe.name, (probe,), "")
    reasons = []
    if not probe.available: reasons.append("unavailable")
    if not probe.finite: reasons.append("nonfinite")
    if not probe.candidate_indices_match: reasons.append("candidate_indices_mismatch")
    if probe.maximum_normalized_error > 1e-5: reasons.append("normalized_error")
    if probe.speedup < 1.1: reasons.append("speedup")
    if probe.reason: reasons.append(probe.reason)
    return BackendDecisionAntigravity("cpu", (probe,), ",".join(reasons) or "gate_failed")


def backend_name_from_decision_antigravity(decision: BackendDecisionAntigravity) -> BackendNameAntigravity:
    if type(decision) is not BackendDecisionAntigravity:
        raise ValueError("decision must be exact BackendDecisionAntigravity")
    return decision.selected_name


def installed_backend_probe_antigravity(name: BackendNameAntigravity) -> BackendProbeAntigravity:
    """Record optional-runtime availability without importing optional modules at import time."""
    if name == "cpu":
        raise ValueError("CPU is not an accelerator probe")
    package = "mlx" if name == "mlx" else "torch"
    available = importlib.util.find_spec(package) is not None
    version = "not_installed"
    device = platform.platform()
    reason = "exact_accelerator_localgap_not_implemented"
    if available:
        module = __import__(package)
        version = str(getattr(module, "__version__", "unknown"))
        if name == "mps":
            device = "mps" if bool(module.backends.mps.is_available()) else "mps_unavailable"
            available = bool(module.backends.mps.is_available())
    return BackendProbeAntigravity(name, available, False, False, 0.0, 1.0, 1.0, ((package, version),), device, reason)


__all__ = ["BackendNameAntigravity", "BackendProbeAntigravity", "BackendDecisionAntigravity", "backend_name_from_decision_antigravity", "build_local_gap_backend_antigravity", "decide_backend_antigravity", "installed_backend_probe_antigravity"]

