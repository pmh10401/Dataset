"""Execution-only acceleration with fail-closed numerical parity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import os
import platform

import numpy as np

from .contracts import DatasetBackendAudit
from .distances import cpu_pairwise_rms


@dataclass(frozen=True)
class AccelerationPolicy:
    numba_threads: int = 12
    numpy_cost_max: int = 250_000
    audit_pairs: int = 64
    synthetic_batch_rows: int = 32
    distance_tolerance: float = 1e-6
    adoption_speedup: float = 2.0

    def validate(self) -> None:
        if (
            self.numba_threads != 12
            or self.numpy_cost_max < 1
            or self.audit_pairs < 1
            or self.synthetic_batch_rows < 1
            or self.distance_tolerance != 1e-6
            or self.adoption_speedup != 2.0
        ):
            raise ValueError("Exp411 acceleration policy is invalid")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


ACCELERATION_POLICY = AccelerationPolicy()
_NESTED_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_runtime(
    policy: AccelerationPolicy = ACCELERATION_POLICY,
) -> Mapping[str, object]:
    policy.validate()
    for name in _NESTED_THREAD_VARIABLES:
        os.environ[name] = "1"
    os.environ["NUMBA_NUM_THREADS"] = str(policy.numba_threads)

    import llvmlite
    import numba

    numba.set_num_threads(policy.numba_threads)
    observed_threads = int(numba.get_num_threads())
    if observed_threads != policy.numba_threads:
        raise RuntimeError(
            f"Exp411 requires {policy.numba_threads} Numba threads, "
            f"observed {observed_threads}"
        )
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        "llvmlite_version": llvmlite.__version__,
        "numba_threads": observed_threads,
        "nested_threads": 1,
        "nested_thread_environment": {
            name: os.environ[name] for name in _NESTED_THREAD_VARIABLES
        },
    }


def choose_backend(
    pair_count: int,
    width: int,
    policy: AccelerationPolicy = ACCELERATION_POLICY,
) -> str:
    policy.validate()
    if int(pair_count) < 1 or int(width) < 1:
        raise ValueError("Exp411 acceleration cost inputs are invalid")
    cost = int(pair_count) * int(width)
    return "numpy_cpu" if cost <= policy.numpy_cost_max else "numba_cpu"


def full_pairwise_rms(
    rows: object,
    *,
    backend: str,
) -> np.ndarray:
    values = _finite_matrix(rows, minimum_rows=1)
    if backend == "numpy_cpu":
        output = np.empty((len(values), len(values)), dtype=np.float64)
        for index, query in enumerate(values):
            delta = values - query
            output[index] = np.sqrt(np.mean(delta * delta, axis=1))
        return output
    if backend == "numba_cpu":
        return np.asarray(_numba_full_pairwise(values), dtype=np.float64)
    raise ValueError(f"unknown Exp411 acceleration backend: {backend}")


def minimum_rms(
    fit: object,
    query: object,
    *,
    backend: str,
) -> np.ndarray:
    fit_values = _finite_matrix(fit, minimum_rows=1)
    query_values = _finite_matrix(query, minimum_rows=1)
    if fit_values.shape[1] != query_values.shape[1]:
        raise ValueError("Exp411 acceleration RMS widths do not match")
    if backend == "numpy_cpu":
        return cpu_pairwise_rms(fit_values, query_values)
    if backend == "numba_cpu":
        return np.asarray(
            _numba_minimum_rms(fit_values, query_values),
            dtype=np.float64,
        )
    raise ValueError(f"unknown Exp411 acceleration backend: {backend}")


def audit_dataset_pairs(
    rows: np.ndarray,
    identity_sha256: Sequence[str],
    policy: AccelerationPolicy = ACCELERATION_POLICY,
) -> DatasetBackendAudit:
    policy.validate()
    values = _finite_matrix(rows, minimum_rows=2)
    identities = tuple(str(value) for value in identity_sha256)
    if (
        len(identities) != len(values)
        or len(set(identities)) != len(identities)
        or any(len(value) != 64 for value in identities)
    ):
        raise ValueError("Exp411 acceleration audit identities are invalid")
    pairs = _sample_pairs(identities, policy.audit_pairs)
    pair_array = np.asarray(pairs, dtype=np.int64)
    expected = _numpy_pair_distances(values, pair_array)
    observed = np.asarray(
        _numba_pair_distances(values, pair_array),
        dtype=np.float64,
    )
    if observed.shape != expected.shape or not np.isfinite(observed).all():
        raise ValueError("Exp411 acceleration pair audit output is invalid")
    max_abs_error = float(np.max(np.abs(expected - observed)))
    selected = choose_backend(
        len(values) * len(values),
        values.shape[1],
        policy,
    )
    fallback_reason = None
    if max_abs_error > policy.distance_tolerance:
        selected = "numpy_cpu"
        fallback_reason = "pair_audit_tolerance_exceeded"
    return DatasetBackendAudit(
        backend=selected,
        sampled_pairs=pairs,
        max_abs_error=max_abs_error,
        fallback_reason=fallback_reason,
    )


def boundary_safe_threshold(
    fit: np.ndarray,
    reference: np.ndarray,
    quantile: float,
    *,
    tolerance: float,
) -> float:
    del tolerance
    values = _finite_matrix(fit, minimum_rows=2)
    distances = np.asarray(reference, dtype=np.float64).reshape(-1)
    if (
        len(distances) != len(values)
        or not np.isfinite(distances).all()
        or not 0.0 < float(quantile) < 1.0
    ):
        raise ValueError("Exp411 acceleration threshold inputs are invalid")
    rank = int(math.floor(float(quantile) * (len(distances) - 1)))
    query_index = int(np.argsort(distances, kind="stable")[rank])
    fit_without_query = np.concatenate(
        (values[:query_index], values[query_index + 1 :]),
        axis=0,
    )
    return float(
        cpu_pairwise_rms(
            fit_without_query,
            values[query_index : query_index + 1],
        )[0]
    )


def boundary_safe_minimum(
    fit: np.ndarray,
    query: np.ndarray,
    observed: np.ndarray,
    thresholds: Sequence[float],
    *,
    tolerance: float,
) -> np.ndarray:
    fit_values = _finite_matrix(fit, minimum_rows=1)
    query_values = _finite_matrix(query, minimum_rows=1)
    result = np.asarray(observed, dtype=np.float64).reshape(-1).copy()
    limits = tuple(float(value) for value in thresholds)
    if (
        fit_values.shape[1] != query_values.shape[1]
        or len(result) != len(query_values)
        or not np.isfinite(result).all()
        or not limits
        or not all(math.isfinite(value) for value in limits)
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0.0
    ):
        raise ValueError("Exp411 acceleration boundary inputs are invalid")
    boundary = np.asarray(
        [
            any(
                abs(float(value) - threshold) <= float(tolerance)
                for threshold in limits
            )
            for value in result
        ],
        dtype=bool,
    )
    if boundary.any():
        result[boundary] = cpu_pairwise_rms(
            fit_values,
            query_values[boundary],
        )
    return result


def _finite_matrix(values: object, *, minimum_rows: int) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[0] < int(minimum_rows)
        or result.shape[1] < 1
        or not np.isfinite(result).all()
    ):
        raise ValueError("Exp411 acceleration matrix is invalid")
    return result


def _sample_pairs(
    identity_sha256: Sequence[str],
    requested: int,
) -> tuple[tuple[int, int], ...]:
    count = len(identity_sha256)
    target = min(int(requested), count * (count - 1))
    material = "\n".join(sorted(identity_sha256)).encode("ascii")
    seed = int.from_bytes(sha256(material).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < target:
        left = int(rng.integers(0, count))
        right = int(rng.integers(0, count - 1))
        if right >= left:
            right += 1
        pairs.add((left, right))
    return tuple(sorted(pairs))


def _numpy_pair_distances(
    rows: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(pairs), dtype=np.float64)
    for index, (left, right) in enumerate(pairs):
        delta = rows[int(left)] - rows[int(right)]
        output[index] = math.sqrt(float(np.mean(delta * delta)))
    return output


def _numba_full_pairwise(rows: np.ndarray) -> np.ndarray:
    import numba

    return _compiled_full_pairwise(numba)(rows)


def _numba_minimum_rms(
    fit: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    import numba

    return _compiled_minimum_rms(numba)(fit, query)


def _numba_pair_distances(
    rows: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    import numba

    return _compiled_pair_distances(numba)(rows, pairs)


_FULL_PAIRWISE_KERNEL: object | None = None
_MINIMUM_RMS_KERNEL: object | None = None
_PAIR_DISTANCES_KERNEL: object | None = None


def _compiled_full_pairwise(numba: object) -> object:
    global _FULL_PAIRWISE_KERNEL
    if _FULL_PAIRWISE_KERNEL is None:
        njit = numba.njit  # type: ignore[attr-defined]
        prange = numba.prange  # type: ignore[attr-defined]

        @njit(parallel=True, cache=False)
        def kernel(rows: np.ndarray) -> np.ndarray:
            count, width = rows.shape
            output = np.empty((count, count), dtype=np.float64)
            for left in prange(count):
                for right in range(count):
                    total = 0.0
                    for column in range(width):
                        delta = rows[left, column] - rows[right, column]
                        total += delta * delta
                    output[left, right] = math.sqrt(total / width)
            return output

        _FULL_PAIRWISE_KERNEL = kernel
    return _FULL_PAIRWISE_KERNEL


def _compiled_minimum_rms(numba: object) -> object:
    global _MINIMUM_RMS_KERNEL
    if _MINIMUM_RMS_KERNEL is None:
        njit = numba.njit  # type: ignore[attr-defined]
        prange = numba.prange  # type: ignore[attr-defined]

        @njit(parallel=True, cache=False)
        def kernel(fit: np.ndarray, query: np.ndarray) -> np.ndarray:
            output = np.empty(query.shape[0], dtype=np.float64)
            for query_index in prange(query.shape[0]):
                best = np.inf
                for fit_index in range(fit.shape[0]):
                    total = 0.0
                    for column in range(fit.shape[1]):
                        delta = (
                            query[query_index, column] - fit[fit_index, column]
                        )
                        total += delta * delta
                    distance = math.sqrt(total / fit.shape[1])
                    if distance < best:
                        best = distance
                output[query_index] = best
            return output

        _MINIMUM_RMS_KERNEL = kernel
    return _MINIMUM_RMS_KERNEL


def _compiled_pair_distances(numba: object) -> object:
    global _PAIR_DISTANCES_KERNEL
    if _PAIR_DISTANCES_KERNEL is None:
        njit = numba.njit  # type: ignore[attr-defined]
        prange = numba.prange  # type: ignore[attr-defined]

        @njit(parallel=True, cache=False)
        def kernel(rows: np.ndarray, pairs: np.ndarray) -> np.ndarray:
            output = np.empty(len(pairs), dtype=np.float64)
            for index in prange(len(pairs)):
                left = pairs[index, 0]
                right = pairs[index, 1]
                total = 0.0
                for column in range(rows.shape[1]):
                    delta = rows[left, column] - rows[right, column]
                    total += delta * delta
                output[index] = math.sqrt(total / rows.shape[1])
            return output

        _PAIR_DISTANCES_KERNEL = kernel
    return _PAIR_DISTANCES_KERNEL


__all__ = [
    "ACCELERATION_POLICY",
    "AccelerationPolicy",
    "audit_dataset_pairs",
    "boundary_safe_minimum",
    "boundary_safe_threshold",
    "choose_backend",
    "configure_runtime",
    "full_pairwise_rms",
    "minimum_rms",
]
