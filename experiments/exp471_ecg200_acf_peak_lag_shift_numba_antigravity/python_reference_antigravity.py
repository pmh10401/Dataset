import math

import numpy as np

from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.acf_peak_lag_antigravity import (
    ACFPeakLagScoresAntigravity,
)


def median_python_antigravity(values):
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return math.fsum((ordered[middle - 1], ordered[middle])) / 2.0


def peak_shift_row_python_antigravity(row):
    if not isinstance(row, np.ndarray):
        raise ValueError("row must be a NumPy array")
    if row.ndim != 1 or row.shape[0] != 140:
        raise ValueError("row must be a one-dimensional float64 array of width 140")
    if row.dtype != np.dtype(np.float64):
        raise ValueError("row must use the float64 dtype")

    peaks = []
    presence = []
    for start in (0, 84):
        window = []
        for value in row[start : start + 56]:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError("row values must be finite")
            window.append(numeric_value)

        center = median_python_antigravity(window)
        mad = median_python_antigravity(abs(value - center) for value in window)
        scale = 1.4826 * mad
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("window MAD must be positive and finite")

        z = [(value - center) / scale for value in window]
        denominator = math.fsum(value * value for value in z)
        acf = []
        for lag in range(1, 26):
            numerator = math.fsum(z[t] * z[t + lag] for t in range(56 - lag))
            acf.append(numerator / denominator)

        candidates = [
            (acf[lag - 1], lag)
            for lag in range(2, 25)
            if acf[lag - 1] > 0.0
            and acf[lag - 1] > acf[lag - 2]
            and acf[lag - 1] >= acf[lag]
        ]
        peak = min((-value, lag) for value, lag in candidates)[1] if candidates else 0
        peaks.append(peak)
        presence.append(bool(candidates))

    return peaks[0], peaks[1], presence[0], presence[1], peaks[1] - peaks[0]


def _validated_float64_rows_antigravity(rows, minimum_rows):
    if not isinstance(rows, np.ndarray):
        raise ValueError("input rows must be a NumPy array")
    if rows.ndim != 2 or rows.shape[1] != 140:
        raise ValueError("input rows must be two-dimensional with width 140")
    if rows.shape[0] < minimum_rows:
        raise ValueError("input rows do not satisfy the required row count")
    if rows.dtype != np.dtype(np.float64):
        raise ValueError("input rows must use the float64 dtype")
    copied_rows = []
    for original_row in rows:
        copied_row = []
        for value in original_row:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError("input rows must contain only finite values")
            copied_row.append(numeric_value)
        if len(copied_row) != 140:
            raise ValueError("input rows must have width 140")
        copied_rows.append(copied_row)
    return copied_rows


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


def score_acf_peak_lag_python_reference_antigravity(fit_rows, query_rows):
    validated_fit = _validated_float64_rows_antigravity(fit_rows, 2)
    validated_query = _validated_float64_rows_antigravity(query_rows, 0)

    fit_peaks = [
        peak_shift_row_python_antigravity(np.asarray(row, dtype=np.float64))
        for row in validated_fit
    ]
    query_peaks = [
        peak_shift_row_python_antigravity(np.asarray(row, dtype=np.float64))
        for row in validated_query
    ]

    fit_early = [int(values[0]) for values in fit_peaks]
    fit_late = [int(values[1]) for values in fit_peaks]
    fit_presence = [(bool(values[2]), bool(values[3])) for values in fit_peaks]
    query_early = [int(values[0]) for values in query_peaks]
    query_late = [int(values[1]) for values in query_peaks]
    query_presence = [(bool(values[2]), bool(values[3])) for values in query_peaks]
    fit_shifts = [late - early for early, late in zip(fit_early, fit_late)]
    query_shifts = [late - early for early, late in zip(query_early, query_late)]

    fit_raw = []
    for excluded, shift in enumerate(fit_shifts):
        others = [
            other_shift
            for index, other_shift in enumerate(fit_shifts)
            if index != excluded
        ]
        fit_raw.append(abs(float(shift) - median_python_antigravity(others)))

    fit_center = median_python_antigravity(fit_shifts)
    maximum = max(fit_raw)
    query_raw = []
    ranks = []
    confirmations = []
    for shift in query_shifts:
        raw = abs(float(shift) - fit_center)
        query_raw.append(raw)
        count = sum(1 for fit_value in fit_raw if fit_value <= raw)
        ranks.append((1.0 + float(count)) / (float(len(fit_raw)) + 1.0))
        confirmations.append(bool(raw > 0.0 and raw > maximum))

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
        fit_shifts=_frozen_array_antigravity(fit_shifts, np.int64),
        query_shifts=_frozen_array_antigravity(query_shifts, np.int64),
        fit_raw_shift=_frozen_array_antigravity(fit_raw, np.float64),
        query_raw_shift=_frozen_array_antigravity(query_raw, np.float64),
        query_ranks=_frozen_array_antigravity(ranks, np.float64),
        confirmation_mask=_frozen_array_antigravity(confirmations, np.bool_),
        fit_normal_max_raw_shift=float(maximum),
        backend="python_reference",
    )
