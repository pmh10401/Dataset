"""Immutable operational backend decision for Exp468.

The acceleration assessment remains the scientific speed/parity record.  This
module records the separate operational authorization needed when that gate
fails and NumPy is retained for the next implementation task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

from .acceleration_antigravity import AccelerationAssessmentAntigravity


_LOWERCASE_SHA256_ANTIGRAVITY = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class BackendDecisionAntigravity:
    assessment: AccelerationAssessmentAntigravity = field(repr=False)
    selected_backend: str
    acceleration_gate_passed: bool
    fallback_authorized: bool
    approval_evidence_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.assessment) is not AccelerationAssessmentAntigravity:
            raise TypeError("assessment must be AccelerationAssessmentAntigravity")
        if type(self.acceleration_gate_passed) is not bool:
            raise TypeError("acceleration_gate_passed must be bool")
        if type(self.fallback_authorized) is not bool:
            raise TypeError("fallback_authorized must be bool")
        if self.selected_backend != self.assessment.selected_backend:
            raise ValueError("selected_backend differs from the assessment")
        if self.acceleration_gate_passed is not self.assessment.gate_passed:
            raise ValueError("acceleration_gate_passed differs from the assessment")

        if self.acceleration_gate_passed:
            self._validate_accelerated_selection_antigravity()
            return
        self._validate_fallback_selection_antigravity()

    def _validate_accelerated_selection_antigravity(self) -> None:
        if self.selected_backend == "numpy":
            raise ValueError("a passed acceleration gate requires a non-NumPy backend")
        candidate = self.assessment.backends[self.selected_backend]
        if candidate.eligible is not True:
            raise ValueError("selected backend eligible must be true")
        if candidate.parity_passed is not True:
            raise ValueError("selected backend parity_passed must be true")
        if candidate.deterministic is not True:
            raise ValueError("selected backend deterministic must be true")
        if (
            type(candidate.speedup) is not float
            or not math.isfinite(candidate.speedup)
            or candidate.speedup < self.assessment.required_speedup
        ):
            raise ValueError("selected backend speedup must meet the frozen 1.1x gate")
        if self.fallback_authorized:
            raise ValueError("a passed acceleration gate cannot authorize fallback")
        if self.approval_evidence_sha256 is not None:
            raise ValueError("a passed acceleration gate must not carry fallback approval")

    def _validate_fallback_selection_antigravity(self) -> None:
        if self.selected_backend != "numpy":
            raise ValueError("a failed acceleration gate must retain NumPy")
        if not self.fallback_authorized:
            raise PermissionError("failed acceleration assessment requires fallback approval")
        if (
            type(self.approval_evidence_sha256) is not str
            or _LOWERCASE_SHA256_ANTIGRAVITY.fullmatch(
                self.approval_evidence_sha256,
            )
            is None
        ):
            raise ValueError("fallback approval must be a 64-character lowercase SHA-256")
        for name, candidate in self.assessment.backends.items():
            if name == "numpy":
                continue
            if candidate.eligible:
                raise ValueError(f"failed fallback candidate {name} cannot be eligible")
            if (
                type(candidate.fallback_reason) is not str
                or not candidate.fallback_reason.strip()
            ):
                raise ValueError(
                    f"failed fallback candidate {name} requires a nonempty failure reason",
                )


def finalize_backend_decision_antigravity(
    assessment: AccelerationAssessmentAntigravity,
    *,
    fallback_approval_sha256: str | None = None,
) -> BackendDecisionAntigravity:
    """Finalize the assessed backend without changing the acceleration gate."""

    if type(assessment) is not AccelerationAssessmentAntigravity:
        raise TypeError("assessment must be AccelerationAssessmentAntigravity")
    if assessment.gate_passed:
        return BackendDecisionAntigravity(
            assessment=assessment,
            selected_backend=assessment.selected_backend,
            acceleration_gate_passed=True,
            fallback_authorized=False,
            approval_evidence_sha256=None,
        )
    if fallback_approval_sha256 is None:
        raise PermissionError("failed acceleration assessment requires fallback approval")
    return BackendDecisionAntigravity(
        assessment=assessment,
        selected_backend=assessment.selected_backend,
        acceleration_gate_passed=False,
        fallback_authorized=True,
        approval_evidence_sha256=fallback_approval_sha256,
    )


__all__ = (
    "BackendDecisionAntigravity",
    "finalize_backend_decision_antigravity",
)
