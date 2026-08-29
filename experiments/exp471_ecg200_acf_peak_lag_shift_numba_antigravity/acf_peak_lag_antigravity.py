from dataclasses import dataclass

import numba
import numpy as np


@dataclass(frozen=True, slots=True)
class ACFPeakLagScoresAntigravity:
    fit_early_peak_lags: np.ndarray
    fit_late_peak_lags: np.ndarray
    fit_peak_presence: np.ndarray
    query_early_peak_lags: np.ndarray
    query_late_peak_lags: np.ndarray
    query_peak_presence: np.ndarray
    fit_shifts: np.ndarray
    query_shifts: np.ndarray
    fit_raw_shift: np.ndarray
    query_raw_shift: np.ndarray
    query_ranks: np.ndarray
    confirmation_mask: np.ndarray
    fit_normal_max_raw_shift: float
    backend: str


@numba.njit(cache=False, fastmath=False)
def _median_copy_numba_antigravity(values):
    ordered = np.sort(values)
    count = ordered.size
    middle = count // 2
    if count % 2 == 1:
        return float(ordered[middle])
    return (float(ordered[middle - 1]) + float(ordered[middle])) / 2.0


@numba.njit(cache=False, fastmath=False)
def dominant_peak_from_acf_numba_antigravity(acf):
    if acf.ndim != 1 or acf.size != 25:
        raise ValueError("acf must have exactly 25 finite float64 values")
    best_lag = 0
    best_value = -np.inf
    for lag in range(2, 25):
        value = acf[lag - 1]
        eligible = value > 0.0 and value > acf[lag - 2] and value >= acf[lag]
        if eligible and (
            value > best_value or (value == best_value and lag < best_lag)
        ):
            best_value = value
            best_lag = lag
    return best_lag, best_lag != 0


@numba.njit(cache=False, fastmath=False)
def _window_peak_numba_antigravity(row, start):
    values = row[start : start + 56].copy()
    center = _median_copy_numba_antigravity(values)
    deviations = np.empty(56, dtype=np.float64)
    for index in range(56):
        deviations[index] = abs(values[index] - center)
    scale = 1.4826 * _median_copy_numba_antigravity(deviations)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("window MAD must be positive and finite")

    z = (values - center) / scale
    denominator = 0.0
    for index in range(56):
        denominator += z[index] * z[index]

    acf = np.empty(25, dtype=np.float64)
    for lag in range(1, 26):
        numerator = 0.0
        for index in range(56 - lag):
            numerator += z[index] * z[index + lag]
        acf[lag - 1] = numerator / denominator
    return dominant_peak_from_acf_numba_antigravity(acf)


@numba.njit(cache=False, fastmath=False)
def peak_shift_row_numba_antigravity(row):
    if row.ndim != 1 or row.size != 140:
        raise ValueError("row must be a one-dimensional float64 array of width 140")
    for index in range(row.size):
        if not np.isfinite(row[index]):
            raise ValueError("row values must be finite")
    early, early_present = _window_peak_numba_antigravity(row, 0)
    late, late_present = _window_peak_numba_antigravity(row, 84)
    return early, late, early_present, late_present, late - early


@numba.njit(cache=False, fastmath=False)
def assemble_raw_rank_numba_antigravity(fit_shifts, query_shifts):
    if fit_shifts.ndim != 1 or query_shifts.ndim != 1:
        raise ValueError("shift arrays must be one-dimensional")
    if fit_shifts.size < 2:
        raise ValueError("fit rows require at least two leave-one-out candidates")

    fit_raw = np.empty(fit_shifts.size, dtype=np.float64)
    for excluded in range(fit_shifts.size):
        others = np.empty(fit_shifts.size - 1, dtype=np.float64)
        position = 0
        for index in range(fit_shifts.size):
            if index != excluded:
                others[position] = fit_shifts[index]
                position += 1
        fit_raw[excluded] = abs(
            float(fit_shifts[excluded]) - _median_copy_numba_antigravity(others)
        )

    fit_center = _median_copy_numba_antigravity(fit_shifts.copy())
    maximum = np.max(fit_raw)
    query_raw = np.empty(query_shifts.size, dtype=np.float64)
    ranks = np.empty(query_shifts.size, dtype=np.float64)
    confirmed = np.empty(query_shifts.size, dtype=np.bool_)
    for query_index in range(query_shifts.size):
        raw = abs(float(query_shifts[query_index]) - fit_center)
        query_raw[query_index] = raw
        count = 0
        for fit_index in range(fit_raw.size):
            if fit_raw[fit_index] <= raw:
                count += 1
        ranks[query_index] = (1.0 + float(count)) / (float(fit_raw.size) + 1.0)
        confirmed[query_index] = raw > 0.0 and raw > maximum
    return fit_raw, query_raw, ranks, confirmed, maximum


def _validated_float64_rows_antigravity(rows, minimum_rows):
    if not isinstance(rows, np.ndarray):
        raise ValueError("input rows must be a NumPy array")
    if rows.ndim != 2 or rows.shape[1] != 140:
        raise ValueError("input rows must be two-dimensional with width 140")
    if rows.shape[0] < minimum_rows:
        raise ValueError("input rows do not satisfy the required row count")
    if rows.dtype != np.dtype(np.float64):
        raise ValueError("input rows must use the float64 dtype")
    copied = np.array(rows, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(copied)):
        raise ValueError("input rows must contain only finite values")
    return copied


def _frozen_array_antigravity(values, dtype):
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.flags.writeable = False
    return array


def _frozen_presence_array_antigravity(values, row_count):
    array = np.ascontiguousarray(
        np.asarray(values, dtype=np.bool_).reshape((row_count, 2))
    )
    array.flags.writeable = False
    return array


def score_acf_peak_lag_numba_antigravity(fit_rows, query_rows):
    validated_fit = _validated_float64_rows_antigravity(fit_rows, 2)
    validated_query = _validated_float64_rows_antigravity(query_rows, 0)

    fit_peaks = []
    query_peaks = []
    for row in validated_fit:
        fit_peaks.append(peak_shift_row_numba_antigravity(row))
    for row in validated_query:
        query_peaks.append(peak_shift_row_numba_antigravity(row))

    fit_early = [int(values[0]) for values in fit_peaks]
    fit_late = [int(values[1]) for values in fit_peaks]
    fit_presence = [
        (bool(values[2]), bool(values[3])) for values in fit_peaks
    ]
    query_early = [int(values[0]) for values in query_peaks]
    query_late = [int(values[1]) for values in query_peaks]
    query_presence = [
        (bool(values[2]), bool(values[3])) for values in query_peaks
    ]
    fit_shift_list = [late - early for early, late in zip(fit_early, fit_late)]
    fit_shift_values = np.asarray(fit_shift_list, dtype=np.float64)
    query_shift_values = np.asarray(
        [late - early for early, late in zip(query_early, query_late)],
        dtype=np.float64,
    )
    fit_raw, query_raw, ranks, confirmed, maximum = (
        assemble_raw_rank_numba_antigravity(fit_shift_values, query_shift_values)
    )

    return ACFPeakLagScoresAntigravity(
        fit_early_peak_lags=_frozen_array_antigravity(fit_early, np.int64),
        fit_late_peak_lags=_frozen_array_antigravity(fit_late, np.int64),
        fit_peak_presence=_frozen_presence_array_antigravity(
            fit_presence,
            len(validated_fit),
        ),
        query_early_peak_lags=_frozen_array_antigravity(query_early, np.int64),
        query_late_peak_lags=_frozen_array_antigravity(query_late, np.int64),
        query_peak_presence=_frozen_presence_array_antigravity(
            query_presence,
            len(validated_query),
        ),
        fit_shifts=_frozen_array_antigravity(fit_shift_list, np.int64),
        query_shifts=_frozen_array_antigravity(
            [late - early for early, late in zip(query_early, query_late)],
            np.int64,
        ),
        fit_raw_shift=_frozen_array_antigravity(fit_raw, np.float64),
        query_raw_shift=_frozen_array_antigravity(query_raw, np.float64),
        query_ranks=_frozen_array_antigravity(ranks, np.float64),
        confirmation_mask=_frozen_array_antigravity(confirmed, np.bool_),
        fit_normal_max_raw_shift=float(maximum),
        backend="numba",
    )
