"""Exp476 policy: reject any neighbourhood containing negative ZF gain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from experiments.exp474_conservative_fallback_routing_antigravity.policy_antigravity import (
    BASELINE_METHOD_ANTIGRAVITY,
    _raw_vector_antigravity,
)
from experiments.exp475_joint_tail_fallback_routing_antigravity.policy_antigravity import (
    JointTailGuardKNNPolicyAntigravity,
)


@dataclass(frozen=True)
class MinimumZFDecisionAntigravity:
    method: str
    proposal: str
    accepted: bool
    confidence: float
    support: int
    lower_auc_gain: float
    median_zf_gain: float
    minimum_zf_gain: float
    reason: str


class MinimumZFGuardKNNPolicyAntigravity(JointTailGuardKNNPolicyAntigravity):
    """Accept only if every selected neighbour has nonnegative ZF gain."""

    def _lower_zf_gain_antigravity(self, features: Mapping[str, float], proposal: str) -> float:
        if proposal == BASELINE_METHOD_ANTIGRAVITY:
            return 0.0
        candidate_index = np.flatnonzero(self._proposals == proposal)
        if candidate_index.size < self.min_support:
            return float("nan")
        raw = _raw_vector_antigravity(features)
        raw = np.where(np.isfinite(raw), raw, self._medians)
        query = (raw - self._mean) / self._scale
        distances = np.linalg.norm(self._matrix[candidate_index] - query[None, :], axis=1)
        neighbours = candidate_index[np.argsort(distances, kind="stable")[: self.k]]
        return float(np.min(self._zf_gains[neighbours]))

    def route(self, features: Mapping[str, float], proposal: str) -> MinimumZFDecisionAntigravity:
        decision = super().route(features, proposal)
        reason = decision.reason
        if reason == "zf_tail_not_positive":
            reason = "minimum_zf_not_positive"
        elif reason == "high_confidence_joint_tail_accept":
            reason = "high_confidence_minimum_zf_accept"
        return MinimumZFDecisionAntigravity(
            method=decision.method,
            proposal=decision.proposal,
            accepted=decision.accepted,
            confidence=decision.confidence,
            support=decision.support,
            lower_auc_gain=decision.lower_auc_gain,
            median_zf_gain=decision.median_zf_gain,
            minimum_zf_gain=decision.lower_zf_gain,
            reason=reason,
        )


__all__ = ["MinimumZFDecisionAntigravity", "MinimumZFGuardKNNPolicyAntigravity"]
