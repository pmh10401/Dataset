"""Exp475 policy: Exp474 acceptance plus a ZF lower-tail guard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from experiments.exp474_conservative_fallback_routing_antigravity.policy_antigravity import (
    BASELINE_METHOD_ANTIGRAVITY,
    ConservativeKNNPolicyAntigravity,
    PolicyDecisionAntigravity,
    _raw_vector_antigravity,
)


@dataclass(frozen=True)
class JointTailDecisionAntigravity:
    method: str
    proposal: str
    accepted: bool
    confidence: float
    support: int
    lower_auc_gain: float
    median_zf_gain: float
    lower_zf_gain: float
    reason: str


class JointTailGuardKNNPolicyAntigravity(ConservativeKNNPolicyAntigravity):
    """Require both AUC and ZF neighbour lower tails to be nonnegative."""

    def __init__(self, *, min_lower_zf_gain: float = 0.0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.min_lower_zf_gain = float(min_lower_zf_gain)

    def route(self, features: Mapping[str, float], proposal: str) -> JointTailDecisionAntigravity:
        base = super().route(features, proposal)
        lower_zf_gain = self._lower_zf_gain_antigravity(features, proposal)
        if not base.accepted:
            return self._from_base_antigravity(base, lower_zf_gain)
        if lower_zf_gain < self.min_lower_zf_gain:
            return JointTailDecisionAntigravity(
                method=BASELINE_METHOD_ANTIGRAVITY,
                proposal=proposal,
                accepted=False,
                confidence=base.confidence,
                support=base.support,
                lower_auc_gain=base.lower_auc_gain,
                median_zf_gain=base.median_zf_gain,
                lower_zf_gain=lower_zf_gain,
                reason="zf_tail_not_positive",
            )
        return JointTailDecisionAntigravity(
            method=base.method,
            proposal=proposal,
            accepted=True,
            confidence=base.confidence,
            support=base.support,
            lower_auc_gain=base.lower_auc_gain,
            median_zf_gain=base.median_zf_gain,
            lower_zf_gain=lower_zf_gain,
            reason="high_confidence_joint_tail_accept",
        )

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
        order = np.argsort(distances, kind="stable")[: self.k]
        neighbours = candidate_index[order]
        return float(np.quantile(self._zf_gains[neighbours], self.lower_quantile))

    @staticmethod
    def _from_base_antigravity(
        base: PolicyDecisionAntigravity, lower_zf_gain: float
    ) -> JointTailDecisionAntigravity:
        return JointTailDecisionAntigravity(
            method=base.method,
            proposal=base.proposal,
            accepted=base.accepted,
            confidence=base.confidence,
            support=base.support,
            lower_auc_gain=base.lower_auc_gain,
            median_zf_gain=base.median_zf_gain,
            lower_zf_gain=lower_zf_gain,
            reason=base.reason,
        )


__all__ = ["JointTailDecisionAntigravity", "JointTailGuardKNNPolicyAntigravity"]
