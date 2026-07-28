from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


VIEW_NAMES = ("raw_rms", "diff_rms", "spectrum_rms")


def view_matrices(rows: object) -> dict[str, np.ndarray]:
    values = np.asarray(rows, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] < 2
        or not np.isfinite(values).all()
    ):
        raise ValueError("Exp411 rows must be a finite non-empty 2D matrix")
    return {
        "raw_rms": values,
        "diff_rms": np.diff(values, axis=1),
        "spectrum_rms": np.abs(np.fft.rfft(values, axis=1))[:, 1:],
    }


def cpu_pairwise_rms(fit: object, query: object) -> np.ndarray:
    fit_values = np.asarray(fit, dtype=np.float64)
    query_values = np.asarray(query, dtype=np.float64)
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
        raise ValueError("Exp411 RMS inputs must be finite compatible matrices")
    delta = query_values[:, None, :] - fit_values[None, :, :]
    return np.sqrt(np.mean(delta * delta, axis=2)).min(axis=1)


def calibrate_thresholds(
    values_by_view: Mapping[str, object],
    quantile: float,
) -> dict[str, float]:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("Exp411 quantile must be in (0, 1)")
    result: dict[str, float] = {}
    for name in VIEW_NAMES:
        if name not in values_by_view:
            continue
        values = np.asarray(values_by_view[name], dtype=np.float64).reshape(-1)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"Exp411 {name} calibration values are invalid")
        result[name] = float(np.quantile(values, quantile, method="lower"))
    return result


def tag_queries(
    distances_by_view: Mapping[str, object],
    thresholds: Mapping[str, float],
    *,
    exact: Sequence[bool],
) -> tuple[str, ...]:
    arrays = {
        name: np.asarray(distances_by_view[name], dtype=np.float64).reshape(-1)
        for name in VIEW_NAMES
        if name in distances_by_view and name in thresholds
    }
    count = len(tuple(exact))
    if any(len(values) != count or not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Exp411 query distance coverage is invalid")
    if any(not np.isfinite(float(thresholds[name])) for name in arrays):
        raise ValueError("Exp411 query thresholds are invalid")

    tags: list[str] = []
    for index, is_exact in enumerate(exact):
        if bool(is_exact):
            tags.append("exact")
            continue
        if len(arrays) < 2 or "raw_rms" not in arrays:
            tags.append("abstain")
            continue
        passing = {
            name
            for name, values in arrays.items()
            if float(values[index]) <= float(thresholds[name])
        }
        tags.append(
            "near"
            if "raw_rms" in passing and len(passing) >= 2
            else "unfamiliar"
        )
    return tuple(tags)


__all__ = [
    "VIEW_NAMES",
    "calibrate_thresholds",
    "cpu_pairwise_rms",
    "tag_queries",
    "view_matrices",
]
