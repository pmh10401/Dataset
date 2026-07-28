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


def cpu_pairwise_rms(
    fit: object,
    query: object,
    *,
    chunk_rows: int = 256,
) -> np.ndarray:
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
        or int(chunk_rows) < 1
    ):
        raise ValueError("Exp411 RMS inputs must be finite compatible matrices")
    output = np.full(len(query_values), np.inf, dtype=np.float64)
    width = fit_values.shape[1]
    for start in range(0, len(query_values), int(chunk_rows)):
        query_chunk = query_values[start : start + int(chunk_rows)]
        fit_block_rows = max(
            1,
            min(
                len(fit_values),
                1_000_000 // max(1, len(query_chunk) * width),
            ),
        )
        best = np.full(len(query_chunk), np.inf, dtype=np.float64)
        for fit_start in range(0, len(fit_values), fit_block_rows):
            fit_chunk = fit_values[fit_start : fit_start + fit_block_rows]
            delta = query_chunk[:, None, :] - fit_chunk[None, :, :]
            squared = np.mean(delta * delta, axis=2)
            best = np.minimum(best, squared.min(axis=1))
        output[start : start + len(query_chunk)] = np.sqrt(best)
    return output


def cpu_self_excluded_rms(
    rows: object,
    *,
    chunk_rows: int = 256,
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] == 0
        or not np.isfinite(values).all()
        or int(chunk_rows) < 1
    ):
        raise ValueError("Exp411 self-excluded RMS inputs are invalid")
    output = np.full(len(values), np.inf, dtype=np.float64)
    width = values.shape[1]
    for start in range(0, len(values), int(chunk_rows)):
        query = values[start : start + int(chunk_rows)]
        fit_block_rows = max(
            1,
            min(
                len(values),
                1_000_000 // max(1, len(query) * width),
            ),
        )
        best = np.full(len(query), np.inf, dtype=np.float64)
        query_indices = np.arange(start, start + len(query))
        for fit_start in range(0, len(values), fit_block_rows):
            fit = values[fit_start : fit_start + fit_block_rows]
            delta = query[:, None, :] - fit[None, :, :]
            squared = np.mean(delta * delta, axis=2)
            fit_indices = np.arange(fit_start, fit_start + len(fit))
            squared[query_indices[:, None] == fit_indices[None, :]] = np.inf
            best = np.minimum(best, squared.min(axis=1))
        output[start : start + len(query)] = np.sqrt(best)
    return output


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
    "cpu_self_excluded_rms",
    "tag_queries",
    "view_matrices",
]
