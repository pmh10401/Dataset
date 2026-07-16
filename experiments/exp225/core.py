from __future__ import annotations

from collections.abc import Iterable

import numpy as np


EXPERIMENT_ID = "experiment_225_fixed_dictionary_independent_confirmation_review"


def confirmation_lanes(
    *,
    candidate_indices: Iterable[int],
    temporal_p: np.ndarray,
    canonical_p: np.ndarray,
    alpha: float,
) -> dict[str, set[int]]:
    temporal = np.asarray(temporal_p, dtype=float)
    canonical = np.asarray(canonical_p, dtype=float)
    if temporal.ndim != 1 or canonical.shape != temporal.shape:
        raise ValueError("confirmation score arrays must be aligned vectors")
    if not np.all(np.isfinite(temporal)) or not np.all(np.isfinite(canonical)):
        raise ValueError("confirmation p-values must be finite")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")

    candidates = {int(index) for index in candidate_indices}
    if candidates and (min(candidates) < 0 or max(candidates) >= len(temporal)):
        raise ValueError("candidate index outside score range")
    temporal_hits = set(np.flatnonzero(temporal <= float(alpha)).astype(int))
    canonical_hits = set(np.flatnonzero(canonical <= float(alpha)).astype(int))
    dual = candidates & temporal_hits & canonical_hits
    single = candidates & (temporal_hits ^ canonical_hits)
    unconfirmed = candidates - dual - single
    return {
        "dual_confirmed_review": dual,
        "single_confirmed_review": single,
        "unconfirmed": unconfirmed,
    }


def build_prediction_payload(
    *,
    dataset: str,
    n_test: int,
    candidate_indices: Iterable[int],
    temporal_p: np.ndarray,
    canonical_p: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    lanes = confirmation_lanes(
        candidate_indices=candidate_indices,
        temporal_p=temporal_p,
        canonical_p=canonical_p,
        alpha=alpha,
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset": str(dataset),
        "n_test": int(n_test),
        "confirmation_alpha": float(alpha),
        "candidate_indices": sorted(int(value) for value in candidate_indices),
        "dual_confirmed_review_indices": sorted(lanes["dual_confirmed_review"]),
        "single_confirmed_review_indices": sorted(lanes["single_confirmed_review"]),
        "unconfirmed_indices": sorted(lanes["unconfirmed"]),
        "autonomous_alert": False,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "metric_scope": "retrospective_review_only",
    }
