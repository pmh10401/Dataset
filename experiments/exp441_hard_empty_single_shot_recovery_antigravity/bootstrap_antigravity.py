"""Pure Exp441 project-F1 and paired-dataset bootstrap calculations."""
from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from .contracts_antigravity import BootstrapResultAntigravity


def project_f1_antigravity(confusion: Sequence[int | float]) -> float:
    """Return the frozen project F1 for ``(tp, fp, fn, tn)`` counts."""
    try:
        values = tuple(confusion)
    except TypeError as error:
        raise ValueError("confusion must contain four values") from error
    if len(values) != 4:
        raise ValueError("confusion must contain four values")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in values
    ):
        raise ValueError("confusion values must be numeric")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
        raise ValueError("confusion values must be finite and nonnegative")

    tp, fp, fn, _tn = (float(value) for value in values)
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else 0.0
    harmonic_denominator = precision + recall
    if harmonic_denominator == 0.0:
        return 0.0
    return float((2.0 * precision * recall) / harmonic_denominator)


def require_numpy_2_4_6_antigravity() -> None:
    """Require the exact preregistered NumPy release for bootstrap replay."""
    if np.__version__ != "2.4.6":
        raise RuntimeError("NumPy 2.4.6 is required for the frozen bootstrap")


def paired_dataset_bootstrap_antigravity(rows: np.ndarray) -> BootstrapResultAntigravity:
    """Return paired-dataset micro and macro confidence intervals."""
    require_numpy_2_4_6_antigravity()
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (1117, 9) or not np.isfinite(values).all():
        raise ValueError("bootstrap rows differ")
    rng = np.random.Generator(np.random.PCG64(44120260808))
    micro = np.empty(20000, dtype=np.float64)
    macro = np.empty(20000, dtype=np.float64)
    for repetition in range(20000):
        indices = rng.integers(
            low=0, high=1117, size=1117,
            dtype=np.int64, endpoint=False,
        )
        sample = values[indices]
        baseline = sample[:, :4].sum(axis=0, dtype=np.float64)
        union = sample[:, 4:8].sum(axis=0, dtype=np.float64)
        micro[repetition] = project_f1_antigravity(union) - project_f1_antigravity(baseline)
        macro[repetition] = sample[:, 8].mean(dtype=np.float64)
    q = np.asarray([0.025, 0.975], dtype=np.float64)
    micro_ci = np.quantile(micro, q, method="linear")
    macro_ci = np.quantile(macro, q, method="linear")
    return BootstrapResultAntigravity(
        micro_ci=tuple(float(value) for value in micro_ci),
        macro_ci=tuple(float(value) for value in macro_ci),
    )


__all__ = [
    "project_f1_antigravity",
    "require_numpy_2_4_6_antigravity",
    "paired_dataset_bootstrap_antigravity",
]
