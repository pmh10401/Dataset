from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ruptures as rpt

from .config import MOSUM_FRACTIONS, PELT_PENALTY_MULTIPLIER


FEATURE_DIMENSIONS = {
    "cusum_mosum_18": 18,
    "pelt_l1_l2_10": 10,
    "pelt_rbf_5": 5,
    "change_point_combined_33": 33,
}
BASE_BLOCKS = ("cusum_mosum_18", "pelt_l1_l2_10", "pelt_rbf_5")


def _aligned(rows: np.ndarray, working_length: int) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError("aligned time-series rows must be two-dimensional")
    if values.shape[1] != working_length:
        old = np.linspace(0.0, 1.0, values.shape[1])
        new = np.linspace(0.0, 1.0, working_length)
        values = np.vstack([np.interp(new, old, row) for row in values])
    if not np.isfinite(values).all():
        raise ValueError("change-point input contains non-finite values")
    return values


@dataclass(frozen=True)
class RobustReference:
    location: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train_rows: np.ndarray) -> "RobustReference":
        values = np.asarray(train_rows, dtype=float)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("finite two-dimensional TRAIN rows are required")
        location = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - location), axis=0)
        positive = scale[np.isfinite(scale) & (scale > 1e-12)]
        fallback = float(np.median(positive)) if len(positive) else float(np.std(values))
        if not np.isfinite(fallback) or fallback <= 1e-12:
            fallback = 1.0
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, fallback)
        return cls(location=location.astype(float), scale=scale.astype(float))

    def transform(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.location):
            raise ValueError("rows do not match fitted TRAIN reference length")
        transformed = (values - self.location) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("robust transformation produced non-finite values")
        return transformed


def detector_parameters(length: int) -> dict[str, float | int]:
    length = int(length)
    if length < 8:
        raise ValueError("PELT requires at least eight time points")
    return {
        "min_size": max(4, length // 32),
        "jump": max(1, length // 64),
        "penalty": PELT_PENALTY_MULTIPLIER * np.log(length),
    }


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    x -= np.mean(x)
    denominator = float(np.dot(x, x))
    return float(np.dot(x, values - np.mean(values)) / denominator) if denominator else 0.0


def _cusum_mosum_row(row: np.ndarray) -> np.ndarray:
    length = len(row)
    centered = row - np.median(row)
    cusum_level = float(np.max(np.abs(np.cumsum(centered))) / max(1, length))
    differences = np.abs(np.diff(row))
    centered_diff = differences - np.median(differences) if len(differences) else differences
    cusum_variation = float(np.max(np.abs(np.cumsum(centered_diff))) / max(1, len(centered_diff))) if len(centered_diff) else 0.0
    output = [cusum_level, cusum_variation]
    for fraction in MOSUM_FRACTIONS:
        window = max(3, min(length // 2 - 1, int(round(length * fraction))))
        summaries = []
        for split in range(window, length - window + 1):
            left = row[split - window:split]
            right = row[split:split + window]
            summaries.append(
                (
                    abs(float(np.mean(right) - np.mean(left))),
                    abs(float(np.median(right) - np.median(left))),
                    abs(_slope(right) - _slope(left)),
                    abs(float(np.log((np.var(right) + 1e-8) / (np.var(left) + 1e-8)))),
                )
            )
        output.extend(np.max(np.asarray(summaries, dtype=float), axis=0).tolist())
    return np.asarray(output, dtype=float)


def _pelt_summary(row: np.ndarray, model: str) -> np.ndarray:
    if float(np.ptp(row)) <= 1e-10 * max(1.0, float(np.max(np.abs(row)))):
        return np.zeros(5, dtype=float)
    parameters = detector_parameters(len(row))
    breakpoints = rpt.Pelt(
        model=model,
        min_size=int(parameters["min_size"]),
        jump=int(parameters["jump"]),
    ).fit(row).predict(pen=float(parameters["penalty"]))
    internal = [int(value) for value in breakpoints if 0 < int(value) < len(row)]
    if not internal:
        return np.zeros(5, dtype=float)
    boundaries = [0, *internal, len(row)]
    segments = [row[start:end] for start, end in zip(boundaries[:-1], boundaries[1:])]
    adjacent = list(zip(segments[:-1], segments[1:]))
    mean_shift = max(abs(float(np.mean(right) - np.mean(left))) for left, right in adjacent)
    median_shift = max(abs(float(np.median(right) - np.median(left))) for left, right in adjacent)
    variance_shift = max(
        abs(float(np.log((np.var(right) + 1e-8) / (np.var(left) + 1e-8))))
        for left, right in adjacent
    )
    shortest = min(map(len, segments)) / len(row)
    return np.asarray([len(internal) / len(row), mean_shift, median_shift, variance_shift, shortest], dtype=float)


class ChangePointFeatureExtractor:
    def __init__(self, *, working_length: int):
        self.working_length = int(working_length)
        self.reference: RobustReference | None = None

    def _blocks(self, values: np.ndarray) -> dict[str, np.ndarray]:
        cusum = np.vstack([_cusum_mosum_row(row) for row in values])
        l1 = np.vstack([_pelt_summary(row, "l1") for row in values])
        l2 = np.vstack([_pelt_summary(row, "l2") for row in values])
        rbf = np.vstack([_pelt_summary(row, "rbf") for row in values])
        blocks = {
            "cusum_mosum_18": cusum,
            "pelt_l1_l2_10": np.concatenate([l1, l2], axis=1),
            "pelt_rbf_5": rbf,
        }
        blocks["change_point_combined_33"] = np.concatenate(
            [blocks[name] for name in BASE_BLOCKS], axis=1
        )
        for name, width in FEATURE_DIMENSIONS.items():
            if blocks[name].shape != (len(values), width):
                raise ValueError(f"feature shape mismatch: {name} {blocks[name].shape}")
            if not np.isfinite(blocks[name]).all():
                raise ValueError(f"non-finite change-point features: {name}")
        return blocks

    def fit_transform_train(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        values = _aligned(rows, self.working_length)
        self.reference = RobustReference.fit(values)
        return self._blocks(self.reference.transform(values))

    def transform(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        if self.reference is None:
            raise RuntimeError("ChangePointFeatureExtractor must be fitted on TRAIN first")
        values = _aligned(rows, self.working_length)
        return self._blocks(self.reference.transform(values))
