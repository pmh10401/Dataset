from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

from .config import FROZEN_CONFIG
from .distances import cpu_pairwise_rms


@dataclass(frozen=True)
class BackendProbe:
    name: str
    available: bool
    version: str | None
    device: str | None
    reason: str | None


@dataclass(frozen=True)
class BackendAssessment:
    name: str
    max_abs_error: float
    tag_identity: bool
    selected_quantile_identity: bool
    cpu_seconds: float
    backend_seconds: float
    speedup: float
    canonical_eligible: bool


_NUMBA_KERNEL: object | None = None


def _numba_kernel() -> object:
    global _NUMBA_KERNEL
    if _NUMBA_KERNEL is None:
        import numba

        @numba.njit(cache=False)
        def kernel(fit: np.ndarray, query: np.ndarray) -> np.ndarray:
            output = np.empty(query.shape[0], dtype=np.float64)
            for query_index in range(query.shape[0]):
                best = np.inf
                for fit_index in range(fit.shape[0]):
                    total = 0.0
                    for column in range(fit.shape[1]):
                        delta = query[query_index, column] - fit[fit_index, column]
                        total += delta * delta
                    distance = math.sqrt(total / fit.shape[1])
                    if distance < best:
                        best = distance
                output[query_index] = best
            return output

        _NUMBA_KERNEL = kernel
    return _NUMBA_KERNEL


def reset_numba_benchmark_state() -> None:
    global _NUMBA_KERNEL
    _NUMBA_KERNEL = None


def pairwise_rms(
    fit: object,
    query: object,
    *,
    backend: str,
) -> np.ndarray:
    fit_values = np.ascontiguousarray(fit, dtype=np.float64)
    query_values = np.ascontiguousarray(query, dtype=np.float64)
    if (
        fit_values.ndim != 2
        or query_values.ndim != 2
        or fit_values.shape[0] == 0
        or query_values.shape[0] == 0
        or fit_values.shape[1] == 0
        or fit_values.shape[1] != query_values.shape[1]
        or not np.isfinite(fit_values).all()
        or not np.isfinite(query_values).all()
    ):
        raise ValueError("Exp411 backend RMS inputs are invalid")
    if backend == "numpy_cpu":
        return cpu_pairwise_rms(fit_values, query_values)
    if backend == "numba_cpu":
        kernel = _numba_kernel()
        return np.asarray(
            kernel(fit_values, query_values),  # type: ignore[operator]
            dtype=np.float64,
        )
    if backend == "torch_cpu":
        return _torch_pairwise_rms(fit_values, query_values, device="cpu")
    if backend == "torch_mps":
        return _torch_pairwise_rms(fit_values, query_values, device="mps")
    raise ValueError(f"unknown Exp411 backend: {backend}")


def _torch_pairwise_rms(
    fit: np.ndarray,
    query: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    import torch

    dtype = torch.float32 if device == "mps" else torch.float64
    fit_tensor = torch.as_tensor(fit, dtype=dtype, device=device)
    query_tensor = torch.as_tensor(query, dtype=dtype, device=device)
    with torch.no_grad():
        delta = query_tensor[:, None, :] - fit_tensor[None, :, :]
        values = torch.sqrt(torch.mean(delta * delta, dim=2)).min(dim=1).values
    return np.asarray(values.cpu(), dtype=np.float64)


def probe_backends() -> dict[str, BackendProbe | str]:
    return {
        "numpy_cpu": BackendProbe(
            name="numpy_cpu",
            available=True,
            version=np.__version__,
            device="cpu",
            reason=None,
        ),
        "numba_cpu": _safe_probe("numba_cpu", _probe_numba_cpu),
        "torch_cpu": _safe_probe("torch_cpu", _probe_torch_cpu),
        "torch_mps": _safe_probe("torch_mps", _probe_torch_mps),
        "mlx_status": "excluded_by_user_decision",
    }


def _safe_probe(
    name: str,
    probe: Callable[[], BackendProbe],
) -> BackendProbe:
    try:
        return probe()
    except Exception as error:
        return BackendProbe(
            name=name,
            available=False,
            version=None,
            device=None,
            reason=f"{type(error).__name__}: {error}",
        )


def _probe_numba_cpu() -> BackendProbe:
    import numba

    values = pairwise_rms(
        np.asarray([[0.0, 0.0], [3.0, 4.0]]),
        np.asarray([[4.0, 4.0]]),
        backend="numba_cpu",
    )
    if not np.allclose(values, np.asarray([np.sqrt(0.5)]), atol=1e-12):
        raise RuntimeError("Numba JIT result mismatch")
    return BackendProbe(
        name="numba_cpu",
        available=True,
        version=numba.__version__,
        device="cpu",
        reason=None,
    )


def _probe_torch_cpu() -> BackendProbe:
    import torch

    values = pairwise_rms(
        np.asarray([[0.0, 0.0]]),
        np.asarray([[0.0, 0.0]]),
        backend="torch_cpu",
    )
    if values.tolist() != [0.0]:
        raise RuntimeError("PyTorch CPU result mismatch")
    return BackendProbe(
        name="torch_cpu",
        available=True,
        version=torch.__version__,
        device="cpu",
        reason=None,
    )


def _probe_torch_mps() -> BackendProbe:
    import torch

    if not torch.backends.mps.is_built():
        raise RuntimeError("PyTorch was not built with MPS")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS device unavailable")
    values = pairwise_rms(
        np.asarray([[0.0, 0.0]]),
        np.asarray([[0.0, 0.0]]),
        backend="torch_mps",
    )
    if not np.allclose(values, np.asarray([0.0]), atol=FROZEN_CONFIG.distance_tolerance):
        raise RuntimeError("PyTorch MPS result mismatch")
    return BackendProbe(
        name="torch_mps",
        available=True,
        version=torch.__version__,
        device="mps",
        reason=None,
    )


def assess_backend(
    *,
    name: str,
    max_abs_error: float,
    tag_identity: bool,
    selected_quantile_identity: bool,
    cpu_seconds: float,
    backend_seconds: float,
) -> BackendAssessment:
    values = (
        float(max_abs_error),
        float(cpu_seconds),
        float(backend_seconds),
    )
    if (
        not all(math.isfinite(value) for value in values)
        or max_abs_error < 0.0
        or cpu_seconds <= 0.0
        or backend_seconds <= 0.0
    ):
        raise ValueError("Exp411 backend assessment values are invalid")
    speedup = float(cpu_seconds) / float(backend_seconds)
    eligible = (
        float(max_abs_error) <= FROZEN_CONFIG.distance_tolerance
        and bool(tag_identity)
        and bool(selected_quantile_identity)
        and speedup >= FROZEN_CONFIG.speedup_gate
    )
    return BackendAssessment(
        name=str(name),
        max_abs_error=float(max_abs_error),
        tag_identity=bool(tag_identity),
        selected_quantile_identity=bool(selected_quantile_identity),
        cpu_seconds=float(cpu_seconds),
        backend_seconds=float(backend_seconds),
        speedup=speedup,
        canonical_eligible=eligible,
    )


__all__ = [
    "BackendAssessment",
    "BackendProbe",
    "assess_backend",
    "pairwise_rms",
    "probe_backends",
    "reset_numba_benchmark_state",
]
