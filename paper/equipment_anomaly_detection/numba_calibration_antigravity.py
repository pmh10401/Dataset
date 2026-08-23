"""Exp467 Numba candidate for exact finite-sample upper-tail calibration."""
from __future__ import annotations

import platform
import sys

import llvmlite
import numba
from numba import njit
import numpy as np

from .calibration_paper import _numeric_vector, upper_tail_pvalues


@njit(cache=True, fastmath=False, nogil=True, parallel=False)
def _float64_upper_tail_counts_numba_antigravity(
    reference: np.ndarray, query: np.ndarray
) -> np.ndarray:
    counts = np.empty(query.size, dtype=np.int64)
    for query_index in range(query.size):
        count = 0
        value = query[query_index]
        for reference_index in range(reference.size):
            if reference[reference_index] >= value:
                count += 1
        counts[query_index] = count
    return counts


def upper_tail_pvalues_numba_antigravity(
    reference: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Use Numba only for equal float64 inputs; preserve legacy semantics otherwise."""
    reference_values = _numeric_vector(reference, "reference", nonempty=True)
    query_values = _numeric_vector(query, "query")
    if (
        reference_values.dtype != np.dtype(np.float64)
        or query_values.dtype != np.dtype(np.float64)
    ):
        return upper_tail_pvalues(reference_values, query_values)
    if query_values.size == 0:
        return np.empty(0, dtype=np.float64)
    counts = _float64_upper_tail_counts_numba_antigravity(
        np.ascontiguousarray(reference_values),
        np.ascontiguousarray(query_values),
    )
    denominator = reference_values.size + 1
    return np.asarray(
        [(1 + int(count)) / denominator for count in counts], dtype=np.float64
    )


def numba_calibration_backend_antigravity() -> dict[str, object]:
    """Return the immutable facts required to reproduce the selected JIT path."""
    return {
        "schema": "exp467-numba-calibration-backend-antigravity.v1",
        "selected_backend": "numba",
        "fast_path_dtype": "float64",
        "fastmath": False,
        "parallel": False,
        "nogil": True,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


__all__ = [
    "numba_calibration_backend_antigravity",
    "upper_tail_pvalues_numba_antigravity",
]
