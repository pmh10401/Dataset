"""Pure Exp443 project metrics and frozen paired-dataset bootstrap."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np

from .config_antigravity import FROZEN_CONFIG


@dataclass(frozen=True)
class MetricSetAntigravity:
    """Project confusion-derived metrics with explicit zero denominators."""

    precision: float
    recall: float
    f1: float
    false_positive_rate: float

    def __post_init__(self) -> None:
        for name in ("precision", "recall", "f1", "false_positive_rate"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} is invalid")

    def to_payload_antigravity(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
        }


@dataclass(frozen=True)
class BootstrapResultAntigravity:
    """One deterministic paired-dataset confidence-interval replay."""

    micro_delta_ci: tuple[float, float]
    macro_f1_delta_ci: tuple[float, float]
    seed: int
    samples: int
    population: int
    numpy_version: str
    sample_indices_sha256: str
    bit_generator: str = "PCG64"
    sample_dtype: str = "int64"
    endpoint: bool = False
    quantile_method: str = "linear"

    def __post_init__(self) -> None:
        for name in ("micro_delta_ci", "macro_f1_delta_ci"):
            values = tuple(getattr(self, name))
            if (
                len(values) != 2
                or any(type(value) is not float or not math.isfinite(value) for value in values)
                or values[0] > values[1]
            ):
                raise ValueError(f"{name} is invalid")
            object.__setattr__(self, name, values)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("bootstrap seed is invalid")
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError("bootstrap samples are invalid")
        if type(self.population) is not int or self.population < 1:
            raise ValueError("bootstrap population is invalid")
        if type(self.numpy_version) is not str or not self.numpy_version:
            raise ValueError("bootstrap NumPy version is invalid")
        if (
            type(self.sample_indices_sha256) is not str
            or len(self.sample_indices_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sample_indices_sha256)
        ):
            raise ValueError("bootstrap sample-index SHA256 is invalid")
        if (
            self.bit_generator != "PCG64"
            or self.sample_dtype != "int64"
            or type(self.endpoint) is not bool
            or self.endpoint is not False
            or self.quantile_method != "linear"
        ):
            raise ValueError("bootstrap implementation identity is invalid")

    def to_payload_antigravity(self) -> dict[str, object]:
        return {
            "micro_delta_ci": list(self.micro_delta_ci),
            "macro_f1_delta_ci": list(self.macro_f1_delta_ci),
            "seed": self.seed,
            "samples": self.samples,
            "population": self.population,
            "numpy_version": self.numpy_version,
            "sample_indices_sha256": self.sample_indices_sha256,
            "bit_generator": self.bit_generator,
            "sample_dtype": self.sample_dtype,
            "endpoint": self.endpoint,
            "quantile_method": self.quantile_method,
        }


def _confusion_antigravity(confusion: Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(confusion, (str, bytes)):
        raise ValueError("confusion must contain four integer counts")
    try:
        values = tuple(confusion)
    except TypeError as error:
        raise ValueError("confusion must contain four integer counts") from error
    if len(values) != 4 or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("confusion must contain four nonnegative integer counts")
    return values


def project_metrics_antigravity(confusion: Sequence[int]) -> MetricSetAntigravity:
    """Return precision, recall, F1, and FPR for ``(TP, FP, FN, TN)``."""

    tp, fp, fn, tn = _confusion_antigravity(confusion)
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float((2 * tp) / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    false_positive_rate = float(fp / (fp + tn)) if fp + tn else 0.0
    return MetricSetAntigravity(precision, recall, f1, false_positive_rate)


def _f1_from_float_counts_antigravity(values: np.ndarray) -> float:
    tp, fp, fn = (float(values[index]) for index in range(3))
    denominator = 2.0 * tp + fp + fn
    return float((2.0 * tp) / denominator) if denominator else 0.0


def _bootstrap_rows_antigravity(rows: object, population: int) -> np.ndarray:
    if type(population) is not int or population < 1:
        raise ValueError("bootstrap population is invalid")
    try:
        values = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("bootstrap rows are invalid") from error
    if values.shape != (population, 9) or not np.isfinite(values).all():
        raise ValueError("bootstrap rows or population differ")
    counts = values[:, :8]
    if np.any(counts < 0.0) or not np.array_equal(counts, np.floor(counts)):
        raise ValueError("bootstrap confusion counts are invalid")
    return values


def _paired_dataset_bootstrap_for_test_antigravity(
    rows: object,
    *,
    population: int,
    samples: int,
    seed: int,
) -> BootstrapResultAntigravity:
    """Injectable-size implementation used to verify the frozen algorithm."""

    values = _bootstrap_rows_antigravity(rows, population)
    if type(samples) is not int or samples < 1:
        raise ValueError("bootstrap samples are invalid")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed is invalid")
    generator = np.random.Generator(np.random.PCG64(seed))
    micro_delta = np.empty(samples, dtype=np.float64)
    macro_delta = np.empty(samples, dtype=np.float64)
    sample_digest = sha256()
    for repetition in range(samples):
        indices = generator.integers(
            low=0,
            high=population,
            size=population,
            dtype=np.int64,
            endpoint=False,
        )
        sample_digest.update(indices.tobytes(order="C"))
        sample = values[indices]
        baseline = sample[:, :4].sum(axis=0, dtype=np.float64)
        policy = sample[:, 4:8].sum(axis=0, dtype=np.float64)
        micro_delta[repetition] = (
            _f1_from_float_counts_antigravity(policy)
            - _f1_from_float_counts_antigravity(baseline)
        )
        macro_delta[repetition] = sample[:, 8].mean(dtype=np.float64)
    quantiles = np.asarray(FROZEN_CONFIG.bootstrap_percentiles, dtype=np.float64)
    micro_ci = np.quantile(micro_delta, quantiles / 100.0, method="linear")
    macro_ci = np.quantile(macro_delta, quantiles / 100.0, method="linear")
    return BootstrapResultAntigravity(
        micro_delta_ci=tuple(float(value) for value in micro_ci),
        macro_f1_delta_ci=tuple(float(value) for value in macro_ci),
        seed=seed,
        samples=samples,
        population=population,
        numpy_version=np.__version__,
        sample_indices_sha256=sample_digest.hexdigest(),
    )


def paired_dataset_bootstrap_antigravity(
    rows: object,
    *,
    population: int = 1117,
    samples: int = 20000,
) -> BootstrapResultAntigravity:
    """Run the exact preregistered Exp443 paired-dataset bootstrap."""

    if np.__version__ != FROZEN_CONFIG.numpy_version or np.__version__ != "2.3.5":
        raise RuntimeError("NumPy 2.3.5 is required for the frozen Exp443 bootstrap")
    if population != FROZEN_CONFIG.dataset_count or population != 1117:
        raise ValueError("bootstrap population must be exactly 1,117")
    if samples != FROZEN_CONFIG.bootstrap_samples or samples != 20000:
        raise ValueError("bootstrap samples must be exactly 20,000")
    return _paired_dataset_bootstrap_for_test_antigravity(
        rows,
        population=population,
        samples=samples,
        seed=FROZEN_CONFIG.bootstrap_seed,
    )


__all__ = [
    "MetricSetAntigravity",
    "BootstrapResultAntigravity",
    "project_metrics_antigravity",
    "paired_dataset_bootstrap_antigravity",
]
