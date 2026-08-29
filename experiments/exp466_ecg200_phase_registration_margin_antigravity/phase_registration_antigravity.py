"""Fixed phase-registration margin evidence for Exp466."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity import (
    anomaly_rank_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_MAD_TO_NORMAL_SCALE_ANTIGRAVITY = 1.4826


def _readonly_float_vector_antigravity(values: object, name: str) -> np.ndarray:
    if type(values) is not np.ndarray:
        raise ValueError(f"{name} must be an exact ndarray")
    if values.ndim != 1 or values.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if values.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must use float64")
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _readonly_bool_vector_antigravity(values: object, name: str) -> np.ndarray:
    if type(values) is not np.ndarray:
        raise ValueError(f"{name} must be an exact ndarray")
    if values.ndim != 1 or values.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if values.dtype != np.dtype(np.bool_):
        raise ValueError(f"{name} must use bool dtype")
    result = np.array(values, dtype=np.bool_, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PhaseScoresAntigravity:
    fit_margins: np.ndarray
    query_margins: np.ndarray
    ranks: np.ndarray
    fit_normal_max_margin: float
    confirmed: np.ndarray

    def __post_init__(self) -> None:
        fit = _readonly_float_vector_antigravity(self.fit_margins, "fit_margins")
        if len(fit) < 2:
            raise ValueError("fit_margins must contain at least two values")
        query = _readonly_float_vector_antigravity(
            self.query_margins,
            "query_margins",
        )
        ranks = _readonly_float_vector_antigravity(self.ranks, "ranks")
        confirmed = _readonly_bool_vector_antigravity(self.confirmed, "confirmed")
        if len(ranks) != len(query) or len(confirmed) != len(query):
            raise ValueError("query_margins, ranks, and confirmed lengths must match")
        threshold = self.fit_normal_max_margin
        if type(threshold) is not float or not isfinite(threshold):
            raise ValueError("fit_normal_max_margin must be a finite float")
        if threshold != float(np.max(fit)):
            raise ValueError("fit_normal_max_margin must equal max(fit_margins)")
        expected_ranks = anomaly_rank_antigravity(fit, query)
        if not np.array_equal(ranks, expected_ranks):
            raise ValueError("ranks must use the frozen anomaly rank definition")
        expected_confirmed = (query > 0.0) & (query > threshold)
        if not np.array_equal(confirmed, expected_confirmed):
            raise ValueError("confirmed must use the strict positive fit-max rule")
        object.__setattr__(self, "fit_margins", fit)
        object.__setattr__(self, "query_margins", query)
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "confirmed", confirmed)


def _row_antigravity(values: object, name: str) -> np.ndarray:
    if type(values) is not np.ndarray:
        raise TypeError(f"{name} must be an exact ndarray")
    if values.shape != (FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,):
        raise ValueError(f"{name} must have shape (140,)")
    if values.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must use float64")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _rows_antigravity(
    values: object,
    name: str,
    *,
    minimum_rows: int,
) -> np.ndarray:
    if type(values) is not np.ndarray:
        raise TypeError(f"{name} must be an exact ndarray")
    if (
        values.ndim != 2
        or values.shape[0] < minimum_rows
        or values.shape[1] != FROZEN_CONFIG_ANTIGRAVITY.interpolated_width
    ):
        raise ValueError(
            f"{name} must have at least {minimum_rows} rows and width 140",
        )
    if values.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must use float64")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _standardized_suffix_antigravity(row: object) -> np.ndarray:
    values = _row_antigravity(row, "row")
    suffix = values[
        FROZEN_CONFIG_ANTIGRAVITY.onset :
        FROZEN_CONFIG_ANTIGRAVITY.onset
        + FROZEN_CONFIG_ANTIGRAVITY.suffix_length
    ]
    median = float(np.median(suffix))
    mad = float(np.median(np.abs(suffix - median)))
    scale = _MAD_TO_NORMAL_SCALE_ANTIGRAVITY * mad
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("row suffix must have positive finite MAD")
    return np.asarray((suffix - median) / scale, dtype=np.float64)


def phase_registration_margin_antigravity(
    row: np.ndarray,
    template: np.ndarray,
) -> float:
    """Return identity error minus the frozen inverse-warp error."""
    query = _standardized_suffix_antigravity(row)
    target = _standardized_suffix_antigravity(template)
    suffix_axis = np.arange(
        FROZEN_CONFIG_ANTIGRAVITY.suffix_length,
        dtype=np.float64,
    )
    shared_axis = np.arange(
        FROZEN_CONFIG_ANTIGRAVITY.shared_length,
        dtype=np.float64,
    )
    identity_error = float(
        np.mean(
            (
                query[: FROZEN_CONFIG_ANTIGRAVITY.shared_length]
                - target[: FROZEN_CONFIG_ANTIGRAVITY.shared_length]
            )
            ** 2,
        ),
    )
    undone = np.interp(
        FROZEN_CONFIG_ANTIGRAVITY.warp_factor * shared_axis,
        suffix_axis,
        query,
    )
    undo_error = float(
        np.mean(
            (
                undone
                - target[: FROZEN_CONFIG_ANTIGRAVITY.shared_length]
            )
            ** 2,
        ),
    )
    return identity_error - undo_error


def score_phase_registration_antigravity(
    fit_rows: np.ndarray,
    query_rows: np.ndarray,
) -> PhaseScoresAntigravity:
    """Calibrate LOO fit margins and score queries against the full-fit median."""
    fit = _rows_antigravity(fit_rows, "fit_rows", minimum_rows=2)
    query = _rows_antigravity(query_rows, "query_rows", minimum_rows=1)

    fit_margins = np.empty(len(fit), dtype=np.float64)
    for index, row in enumerate(fit):
        template = np.median(np.delete(fit, index, axis=0), axis=0)
        fit_margins[index] = phase_registration_margin_antigravity(row, template)

    query_template = np.median(fit, axis=0)
    query_margins = np.array(
        [
            phase_registration_margin_antigravity(row, query_template)
            for row in query
        ],
        dtype=np.float64,
    )
    fit_normal_max_margin = float(np.max(fit_margins))
    confirmed = (query_margins > 0.0) & (
        query_margins > fit_normal_max_margin
    )
    ranks = anomaly_rank_antigravity(fit_margins, query_margins)
    return PhaseScoresAntigravity(
        fit_margins=fit_margins,
        query_margins=query_margins,
        ranks=ranks,
        fit_normal_max_margin=fit_normal_max_margin,
        confirmed=confirmed,
    )


__all__ = (
    "PhaseScoresAntigravity",
    "phase_registration_margin_antigravity",
    "score_phase_registration_antigravity",
)
