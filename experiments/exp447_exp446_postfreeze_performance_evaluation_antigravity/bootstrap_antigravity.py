"""Deterministic shared-draw paired bootstrap for descriptive Exp447 evidence."""
from __future__ import annotations

from hashlib import sha256
from collections.abc import Sequence

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    BootstrapComparisonAntigravity,
    DatasetConfusionRowAntigravity,
    normalize_dataset_confusion_rows_antigravity,
)


def project_f1_antigravity(confusion: object) -> float:
    """Return the frozen project F1 for a confusion tuple or row object."""
    if isinstance(confusion, DatasetConfusionRowAntigravity):
        counts = confusion.confusion
        tp, fp, fn = counts.tp, counts.fp, counts.fn
    elif isinstance(confusion, Sequence) and not isinstance(confusion, (str, bytes)) and len(confusion) == 4:
        if any(type(value) is not int or value < 0 for value in confusion):
            raise ValueError("confusion must contain exact non-negative integers")
        tp, fp, fn = confusion[0], confusion[1], confusion[2]
    else:
        raise ValueError("confusion is invalid")
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def bootstrap_draws_antigravity(population: int) -> np.ndarray:
    """Construct the one frozen PCG64 draw matrix reused across comparisons."""
    if type(population) is not int or population != FROZEN_CONFIG_ANTIGRAVITY.dataset_count:
        raise ValueError("bootstrap population differs from frozen dataset count")
    rng = np.random.Generator(np.random.PCG64(FROZEN_CONFIG_ANTIGRAVITY.bootstrap_seed))
    draws = rng.integers(
        0, population,
        size=(FROZEN_CONFIG_ANTIGRAVITY.bootstrap_samples, population),
        dtype=np.int64, endpoint=False,
    )
    draws.setflags(write=False)
    return draws


def _validated_draws_antigravity(draws: object, population: int) -> np.ndarray:
    value = bootstrap_draws_antigravity(population) if draws is None else draws
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("int64") or value.shape != (FROZEN_CONFIG_ANTIGRAVITY.bootstrap_samples, population):
        raise ValueError("bootstrap draws have the wrong shape or dtype")
    if not np.all((value >= 0) & (value < population)):
        raise ValueError("bootstrap draws contain an invalid index")
    return value


def _counts_antigravity(rows: tuple[DatasetConfusionRowAntigravity, ...]) -> np.ndarray:
    return np.asarray(
        [[row.confusion.tp, row.confusion.fp, row.confusion.fn, row.confusion.tn] for row in rows],
        dtype=np.int64,
    )


def _f1_vector_antigravity(counts: np.ndarray) -> np.ndarray:
    denominator = 2 * counts[:, 0] + counts[:, 1] + counts[:, 2]
    return np.divide(2.0 * counts[:, 0], denominator, out=np.zeros(len(counts), dtype=np.float64), where=denominator != 0)


def paired_bootstrap_antigravity(
    baseline_rows: object,
    comparator_rows: object,
    comparator_name: str,
    *,
    draws: object = None,
) -> BootstrapComparisonAntigravity:
    """Compute comparator-minus-baseline F1 uncertainty with shared provided draws."""
    baseline = normalize_dataset_confusion_rows_antigravity(baseline_rows)
    comparator = normalize_dataset_confusion_rows_antigravity(comparator_rows)
    if len(baseline) != FROZEN_CONFIG_ANTIGRAVITY.dataset_count:
        raise ValueError("bootstrap population differs from frozen dataset count")
    if tuple(row.dataset for row in baseline) != tuple(row.dataset for row in comparator):
        raise ValueError("paired dataset order differs")
    draw_matrix = _validated_draws_antigravity(draws, len(baseline))
    base_counts, compared_counts = _counts_antigravity(baseline), _counts_antigravity(comparator)
    base_macro = np.asarray([project_f1_antigravity(row) for row in baseline], dtype=np.float64)
    compared_macro = np.asarray([project_f1_antigravity(row) for row in comparator], dtype=np.float64)
    micro_delta = np.empty(FROZEN_CONFIG_ANTIGRAVITY.bootstrap_samples, dtype=np.float64)
    macro_delta = np.empty(FROZEN_CONFIG_ANTIGRAVITY.bootstrap_samples, dtype=np.float64)
    for start in range(0, len(draw_matrix), 256):
        stop = min(start + 256, len(draw_matrix))
        indices = draw_matrix[start:stop]
        sampled_base = base_counts[indices].sum(axis=1, dtype=np.int64)
        sampled_compared = compared_counts[indices].sum(axis=1, dtype=np.int64)
        micro_delta[start:stop] = _f1_vector_antigravity(sampled_compared) - _f1_vector_antigravity(sampled_base)
        macro_delta[start:stop] = compared_macro[indices].mean(axis=1) - base_macro[indices].mean(axis=1)
    base_total = tuple(int(value) for value in base_counts.sum(axis=0, dtype=np.int64))
    compared_total = tuple(int(value) for value in compared_counts.sum(axis=0, dtype=np.int64))
    return BootstrapComparisonAntigravity(
        comparator_name=comparator_name,
        population=len(baseline), seed=FROZEN_CONFIG_ANTIGRAVITY.bootstrap_seed,
        samples=FROZEN_CONFIG_ANTIGRAVITY.bootstrap_samples, sample_dtype="int64", endpoint=False,
        percentile_method="linear", baseline_micro_f1=project_f1_antigravity(base_total),
        comparator_micro_f1=project_f1_antigravity(compared_total), baseline_macro_f1=float(base_macro.mean()),
        comparator_macro_f1=float(compared_macro.mean()),
        micro_delta=project_f1_antigravity(compared_total) - project_f1_antigravity(base_total),
        macro_delta=float(compared_macro.mean() - base_macro.mean()),
        micro_interval=tuple(float(value) for value in np.percentile(micro_delta, (2.5, 97.5), method="linear")),
        macro_interval=tuple(float(value) for value in np.percentile(macro_delta, (2.5, 97.5), method="linear")),
        sample_indices_sha256=sha256(draw_matrix.tobytes(order="C")).hexdigest(),
    )


__all__ = ["bootstrap_draws_antigravity", "paired_bootstrap_antigravity", "project_f1_antigravity"]
