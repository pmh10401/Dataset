"""Conservative TRAIN-only fallback policy for Exp474."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


BASELINE_METHOD_ANTIGRAVITY = "minirocket_local_gap"
FEATURE_KEYS_ANTIGRAVITY = (
    "width",
    "n_train",
    "spectral_entropy",
    "skewness",
    "kurtosis",
    "diff_std_ratio",
    "zero_crossing_rate",
)


@dataclass(frozen=True)
class PolicyDecisionAntigravity:
    method: str
    proposal: str
    accepted: bool
    confidence: float
    support: int
    lower_auc_gain: float
    median_zf_gain: float
    reason: str


def _raw_vector_antigravity(features: Mapping[str, float]) -> np.ndarray:
    values = np.asarray([float(features[key]) for key in FEATURE_KEYS_ANTIGRAVITY], dtype=np.float64)
    values[0] = np.log1p(max(0.0, values[0]))
    values[1] = np.log1p(max(0.0, values[1]))
    return values


class ConservativeKNNPolicyAntigravity:
    """Accept the current non-baseline proposal only with strong local evidence.

    The model uses only seven TRAIN-derived numeric features. Dataset names and
    held-out scores are never input features. Uncertain cases return MiniRocket.
    """

    def __init__(
        self,
        *,
        k: int = 31,
        min_support: int = 20,
        positive_rate_threshold: float = 0.90,
        lower_quantile: float = 0.20,
        min_auc_gain: float = 0.0,
        min_zf_gain: float = 0.0,
    ) -> None:
        if k < 1 or min_support < 1 or min_support > k:
            raise ValueError("require 1 <= min_support <= k")
        if not 0.0 <= positive_rate_threshold <= 1.0:
            raise ValueError("positive_rate_threshold must be in [0, 1]")
        if not 0.0 <= lower_quantile <= 1.0:
            raise ValueError("lower_quantile must be in [0, 1]")
        self.k = int(k)
        self.min_support = int(min_support)
        self.positive_rate_threshold = float(positive_rate_threshold)
        self.lower_quantile = float(lower_quantile)
        self.min_auc_gain = float(min_auc_gain)
        self.min_zf_gain = float(min_zf_gain)
        self._fitted = False

    def fit(self, rows: Iterable[Mapping[str, object]]) -> "ConservativeKNNPolicyAntigravity":
        fitted_rows = list(rows)
        if not fitted_rows:
            raise ValueError("at least one training row is required")

        matrix = np.stack(
            [_raw_vector_antigravity(row["features"]) for row in fitted_rows]  # type: ignore[arg-type]
        )
        finite = np.isfinite(matrix)
        medians = np.asarray(
            [np.median(matrix[finite[:, index], index]) if finite[:, index].any() else 0.0 for index in range(matrix.shape[1])]
        )
        matrix = np.where(finite, matrix, medians[None, :])
        scales = matrix.std(axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)

        self._medians = medians
        self._mean = matrix.mean(axis=0)
        self._scale = scales
        self._matrix = (matrix - self._mean[None, :]) / self._scale[None, :]
        self._proposals = np.asarray([str(row["proposal"]) for row in fitted_rows], dtype=object)
        self._auc_gains = np.asarray([float(row["auc_gain"]) for row in fitted_rows], dtype=np.float64)
        self._zf_gains = np.asarray([float(row["zf_gain"]) for row in fitted_rows], dtype=np.float64)
        if not np.isfinite(self._auc_gains).all() or not np.isfinite(self._zf_gains).all():
            raise ValueError("training gains must be finite")
        self._fitted = True
        return self

    def route(self, features: Mapping[str, float], proposal: str) -> PolicyDecisionAntigravity:
        if not self._fitted:
            raise RuntimeError("fit must be called before route")
        if proposal == BASELINE_METHOD_ANTIGRAVITY:
            return PolicyDecisionAntigravity(
                method=BASELINE_METHOD_ANTIGRAVITY,
                proposal=proposal,
                accepted=False,
                confidence=1.0,
                support=0,
                lower_auc_gain=0.0,
                median_zf_gain=0.0,
                reason="baseline_proposal",
            )

        candidate_index = np.flatnonzero(self._proposals == proposal)
        if candidate_index.size < self.min_support:
            return self._fallback_antigravity(proposal, "insufficient_support", int(candidate_index.size))

        raw = _raw_vector_antigravity(features)
        raw = np.where(np.isfinite(raw), raw, self._medians)
        query = (raw - self._mean) / self._scale
        distances = np.linalg.norm(self._matrix[candidate_index] - query[None, :], axis=1)
        order = np.argsort(distances, kind="stable")[: self.k]
        neighbours = candidate_index[order]
        auc_gains = self._auc_gains[neighbours]
        zf_gains = self._zf_gains[neighbours]
        support = int(neighbours.size)
        confidence = float(np.mean((auc_gains >= 0.0) & (zf_gains >= 0.0)))
        lower_auc_gain = float(np.quantile(auc_gains, self.lower_quantile))
        median_zf_gain = float(np.median(zf_gains))

        if confidence < self.positive_rate_threshold:
            reason = "positive_rate_too_low"
        elif lower_auc_gain < self.min_auc_gain:
            reason = "auc_tail_not_positive"
        elif median_zf_gain < self.min_zf_gain:
            reason = "zf_median_not_positive"
        else:
            return PolicyDecisionAntigravity(
                method=proposal,
                proposal=proposal,
                accepted=True,
                confidence=confidence,
                support=support,
                lower_auc_gain=lower_auc_gain,
                median_zf_gain=median_zf_gain,
                reason="high_confidence_accept",
            )
        return PolicyDecisionAntigravity(
            method=BASELINE_METHOD_ANTIGRAVITY,
            proposal=proposal,
            accepted=False,
            confidence=confidence,
            support=support,
            lower_auc_gain=lower_auc_gain,
            median_zf_gain=median_zf_gain,
            reason=reason,
        )

    @staticmethod
    def _fallback_antigravity(proposal: str, reason: str, support: int) -> PolicyDecisionAntigravity:
        return PolicyDecisionAntigravity(
            method=BASELINE_METHOD_ANTIGRAVITY,
            proposal=proposal,
            accepted=False,
            confidence=0.0,
            support=support,
            lower_auc_gain=float("nan"),
            median_zf_gain=float("nan"),
            reason=reason,
        )


__all__ = [
    "BASELINE_METHOD_ANTIGRAVITY",
    "ConservativeKNNPolicyAntigravity",
    "FEATURE_KEYS_ANTIGRAVITY",
    "PolicyDecisionAntigravity",
]
