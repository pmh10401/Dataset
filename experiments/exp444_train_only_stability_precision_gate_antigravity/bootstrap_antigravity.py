"""Deterministic, descriptive-only paired bootstrap for Exp444 evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .config_antigravity import FROZEN_CONFIG


_CONFUSION_NAMES_ANTIGRAVITY = ("tp", "fp", "fn", "tn")


def _count_antigravity(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative exact integer")
    return int(value)


def _confusion_antigravity(row: Mapping[str, object] | Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(row, Mapping):
        if not set(_CONFUSION_NAMES_ANTIGRAVITY).issubset(row):
            raise ValueError("paired row has no complete confusion")
        return tuple(_count_antigravity(row[name], name) for name in _CONFUSION_NAMES_ANTIGRAVITY)  # type: ignore[return-value]
    values = tuple(row)
    if len(values) != 4:
        raise ValueError("paired row confusion order is invalid")
    return tuple(_count_antigravity(value, name) for value, name in zip(values, _CONFUSION_NAMES_ANTIGRAVITY, strict=True))  # type: ignore[return-value]


def project_f1_antigravity(confusion: Mapping[str, object] | Sequence[int]) -> float:
    """Return project F1 in the frozen [TP, FP, FN, TN] order."""
    tp, fp, fn, _tn = _confusion_antigravity(confusion)
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


@dataclass(frozen=True)
class PairedIntervalsAntigravity:
    """Complete replayable bootstrap evidence, with no decision interpretation."""

    population: int
    seed: int
    samples: int
    sample_dtype: str
    endpoint: bool
    percentile_method: str
    baseline_micro_f1: float
    comparator_micro_f1: float
    baseline_macro_f1: float
    comparator_macro_f1: float
    micro_delta: float
    macro_delta: float
    micro_interval: tuple[float, float]
    macro_interval: tuple[float, float]
    sample_indices_sha256: str

    def to_payload_antigravity(self) -> dict[str, object]:
        return {
            "population": self.population, "seed": self.seed, "samples": self.samples,
            "sample_dtype": self.sample_dtype, "endpoint": self.endpoint,
            "percentile_method": self.percentile_method,
            "baseline_micro_f1": self.baseline_micro_f1, "comparator_micro_f1": self.comparator_micro_f1,
            "baseline_macro_f1": self.baseline_macro_f1, "comparator_macro_f1": self.comparator_macro_f1,
            "micro_delta": self.micro_delta, "macro_delta": self.macro_delta,
            "micro_interval": list(self.micro_interval), "macro_interval": list(self.macro_interval),
            "sample_indices_sha256": self.sample_indices_sha256,
        }


def _rows_antigravity(rows: Sequence[Mapping[str, object] | Sequence[int]]) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    values: list[tuple[str, tuple[int, int, int, int]]] = []
    for index, row in enumerate(rows):
        dataset = row.get("dataset") if isinstance(row, Mapping) else None
        if type(dataset) is not str or not dataset:
            raise ValueError("paired row dataset identity is invalid")
        values.append((dataset, _confusion_antigravity(row)))
    if not values or len({dataset for dataset, _ in values}) != len(values):
        raise ValueError("paired dataset population is invalid")
    return tuple(values)


def paired_bootstrap_antigravity(
    baseline_rows: Sequence[Mapping[str, object] | Sequence[int]],
    comparator_rows: Sequence[Mapping[str, object] | Sequence[int]],
) -> PairedIntervalsAntigravity:
    """Bootstrap comparator-minus-baseline dataset pairs with frozen PCG64 draws."""
    baseline, comparator = _rows_antigravity(baseline_rows), _rows_antigravity(comparator_rows)
    if tuple(dataset for dataset, _ in baseline) != tuple(dataset for dataset, _ in comparator):
        raise ValueError("paired dataset order differs")
    population = len(baseline)
    base = np.asarray([item for _, item in baseline], dtype=np.int64)
    compared = np.asarray([item for _, item in comparator], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(FROZEN_CONFIG.bootstrap_seed))
    sample_indices = rng.integers(0, population, size=(FROZEN_CONFIG.bootstrap_samples, population), dtype=np.int64, endpoint=False)
    sampled_base = base[sample_indices].sum(axis=1, dtype=np.int64)
    sampled_compared = compared[sample_indices].sum(axis=1, dtype=np.int64)
    f1 = lambda counts: np.divide(2.0 * counts[:, 0], 2.0 * counts[:, 0] + counts[:, 1] + counts[:, 2], out=np.zeros(counts.shape[0], dtype=np.float64), where=(2.0 * counts[:, 0] + counts[:, 1] + counts[:, 2]) != 0)
    micro = f1(sampled_compared) - f1(sampled_base)
    baseline_macro_rows = np.asarray([project_f1_antigravity(item) for _, item in baseline], dtype=np.float64)
    comparator_macro_rows = np.asarray([project_f1_antigravity(item) for _, item in comparator], dtype=np.float64)
    macro = comparator_macro_rows[sample_indices].mean(axis=1) - baseline_macro_rows[sample_indices].mean(axis=1)
    base_total, compared_total = tuple(base.sum(axis=0, dtype=np.int64)), tuple(compared.sum(axis=0, dtype=np.int64))
    from hashlib import sha256
    return PairedIntervalsAntigravity(
        population, FROZEN_CONFIG.bootstrap_seed, FROZEN_CONFIG.bootstrap_samples, "int64", False, "linear",
        project_f1_antigravity(base_total), project_f1_antigravity(compared_total),
        float(baseline_macro_rows.mean()), float(comparator_macro_rows.mean()),
        project_f1_antigravity(compared_total) - project_f1_antigravity(base_total),
        float(comparator_macro_rows.mean() - baseline_macro_rows.mean()),
        tuple(float(value) for value in np.percentile(micro, (2.5, 97.5), method="linear")),
        tuple(float(value) for value in np.percentile(macro, (2.5, 97.5), method="linear")),
        sha256(sample_indices.tobytes(order="C")).hexdigest(),
    )


__all__ = ["PairedIntervalsAntigravity", "paired_bootstrap_antigravity", "project_f1_antigravity"]
