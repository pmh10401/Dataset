"""Published anomaly detector score adapters."""

from __future__ import annotations

import numpy as np

from ..config import CLASSICAL_METHOD_IDS
from ..contracts import ScoreResult
from .classical import crossfit_scores, score_classical
from .deep_svdd import score_deep_svdd
from .drocc import score_drocc


def score_published_method(
    method_id: str,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
    *,
    epochs: int = 50,
) -> ScoreResult:
    """Dispatch one preregistered published detector without substitution."""
    if method_id in CLASSICAL_METHOD_IDS:
        return crossfit_scores(method_id, train, test, seed)
    if method_id == "deep_svdd_1d":
        return score_deep_svdd(train, test, seed, epochs=epochs)
    if method_id == "drocc_1d":
        return score_drocc(train, test, seed, epochs=epochs)
    raise ValueError(f"unknown published method: {method_id}")

__all__ = [
    "crossfit_scores",
    "score_classical",
    "score_deep_svdd",
    "score_drocc",
    "score_published_method",
]
