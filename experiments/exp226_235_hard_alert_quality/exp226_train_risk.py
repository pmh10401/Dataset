from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from experiments.exp206_210.core import empirical_upper_pvalues


FoldPlan = Sequence[tuple[np.ndarray, np.ndarray]]


def _validate_plan(plan: FoldPlan, n_rows: int) -> None:
    seen: list[int] = []
    universe = set(range(int(n_rows)))
    for fit_raw, query_raw in plan:
        fit = {int(value) for value in np.asarray(fit_raw).reshape(-1)}
        query = [int(value) for value in np.asarray(query_raw).reshape(-1)]
        if fit & set(query):
            raise ValueError("fit and query indices must be disjoint")
        if not fit <= universe or not set(query) <= universe:
            raise ValueError("fold index outside TRAIN range")
        seen.extend(query)
    if sorted(seen) != list(range(int(n_rows))):
        raise ValueError("fold query indices must cover every TRAIN row exactly once")


def restore_fold_concatenated(
    values: Iterable[float], plan: FoldPlan, n_rows: int
) -> np.ndarray:
    concatenated = np.asarray(list(values), dtype=float)
    _validate_plan(plan, n_rows)
    if len(concatenated) != int(n_rows):
        raise ValueError("fold-concatenated value count must equal TRAIN row count")
    restored = np.empty(int(n_rows), dtype=float)
    offset = 0
    for _, query_raw in plan:
        query = np.asarray(query_raw, dtype=int).reshape(-1)
        restored[query] = concatenated[offset : offset + len(query)]
        offset += len(query)
    if not np.isfinite(restored).all():
        raise ValueError("restored p-values must be finite")
    return restored


def fold_local_pvalues(scores: Iterable[float], plan: FoldPlan) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("OOF scores must be a finite vector")
    _validate_plan(plan, len(values))
    output = np.empty(len(values), dtype=float)
    for fit_raw, query_raw in plan:
        fit = np.asarray(fit_raw, dtype=int).reshape(-1)
        query = np.asarray(query_raw, dtype=int).reshape(-1)
        output[query] = empirical_upper_pvalues(values[fit], values[query])
    return output


def conditional_risk_lanes(
    *,
    shape_p: Iterable[float],
    temporal_p: Iterable[float],
    canonical_p: Iterable[float],
    shape_alpha: float,
    confirmation_alpha: float,
) -> dict[str, set[int]]:
    shape = np.asarray(list(shape_p), dtype=float)
    temporal = np.asarray(list(temporal_p), dtype=float)
    canonical = np.asarray(list(canonical_p), dtype=float)
    if shape.ndim != 1 or temporal.shape != shape.shape or canonical.shape != shape.shape:
        raise ValueError("aligned p-value vectors must have the same one-dimensional shape")
    if not all(np.isfinite(array).all() for array in (shape, temporal, canonical)):
        raise ValueError("p-value vectors must be finite")
    if not 0.0 < float(shape_alpha) < 1.0 or not 0.0 < float(confirmation_alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    candidates = set(np.flatnonzero(shape <= float(shape_alpha)).astype(int))
    temporal_hits = set(np.flatnonzero(temporal <= float(confirmation_alpha)).astype(int))
    canonical_hits = set(np.flatnonzero(canonical <= float(confirmation_alpha)).astype(int))
    dual = candidates & temporal_hits & canonical_hits
    single = candidates & (temporal_hits ^ canonical_hits)
    unconfirmed = candidates - dual - single
    if (dual & single) or (dual & unconfirmed) or (single & unconfirmed):
        raise AssertionError("conditional risk lanes overlap")
    if dual | single | unconfirmed != candidates:
        raise AssertionError("conditional risk lanes do not partition candidates")
    return {"dual": dual, "single": single, "unconfirmed": unconfirmed}

