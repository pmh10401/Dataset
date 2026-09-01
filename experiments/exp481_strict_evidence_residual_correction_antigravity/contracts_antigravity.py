"""Strict immutable residual-correction contracts for Exp481."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config_antigravity import METHODS_ANTIGRAVITY


def _require_method_antigravity(method: object) -> None:
    if type(method) is not str or method not in METHODS_ANTIGRAVITY:
        raise ValueError("method must be one of the four frozen methods")


def _require_exact_bool_antigravity(value: object, label: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{label} must be an exact bool")


def _require_nonnegative_int_antigravity(value: object, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be an exact nonnegative integer")


def _require_finite_float_antigravity(value: object, label: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be an exact finite float")


@dataclass(frozen=True)
class ReferenceStatisticsAntigravity:
    passed: bool
    zf_wins: int
    zf_losses: int
    zf_ties: int
    min_auc_gain: float
    q20_auc_gain: float

    def __post_init__(self) -> None:
        _require_exact_bool_antigravity(self.passed, "passed")
        _require_nonnegative_int_antigravity(self.zf_wins, "zf_wins")
        _require_nonnegative_int_antigravity(self.zf_losses, "zf_losses")
        _require_nonnegative_int_antigravity(self.zf_ties, "zf_ties")
        _require_finite_float_antigravity(self.min_auc_gain, "min_auc_gain")
        _require_finite_float_antigravity(self.q20_auc_gain, "q20_auc_gain")


@dataclass(frozen=True)
class StrictCandidateAssessmentAntigravity:
    method: str
    passed: bool
    support: int
    zf_wins: int
    zf_losses: int
    zf_ties: int
    min_auc_gain: float
    q20_auc_gain: float
    neighbour_families: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_method_antigravity(self.method)
        _require_exact_bool_antigravity(self.passed, "passed")
        _require_nonnegative_int_antigravity(self.support, "support")
        _require_nonnegative_int_antigravity(self.zf_wins, "zf_wins")
        _require_nonnegative_int_antigravity(self.zf_losses, "zf_losses")
        _require_nonnegative_int_antigravity(self.zf_ties, "zf_ties")
        _require_finite_float_antigravity(self.min_auc_gain, "min_auc_gain")
        _require_finite_float_antigravity(self.q20_auc_gain, "q20_auc_gain")
        if type(self.neighbour_families) is not tuple:
            raise ValueError("neighbour_families must be a tuple")
        if any(type(family) is not str or not family for family in self.neighbour_families):
            raise ValueError("neighbour_families must contain non-empty strings")
        if len(set(self.neighbour_families)) != len(self.neighbour_families):
            raise ValueError("neighbour_families must not contain duplicates")
        if self.support != self.zf_wins + self.zf_losses + self.zf_ties:
            raise ValueError("support must equal zf_wins + zf_losses + zf_ties")
        if self.support != len(self.neighbour_families):
            raise ValueError("support must equal the number of unique neighbour families")


@dataclass(frozen=True)
class InnerCandidateGateAntigravity:
    method: str
    enabled: bool
    changed_families: int
    winning_families: int
    losing_families: int
    tied_families: int
    min_auc_gain: float
    q20_auc_gain: float

    def __post_init__(self) -> None:
        _require_method_antigravity(self.method)
        _require_exact_bool_antigravity(self.enabled, "enabled")
        _require_nonnegative_int_antigravity(self.changed_families, "changed_families")
        _require_nonnegative_int_antigravity(self.winning_families, "winning_families")
        _require_nonnegative_int_antigravity(self.losing_families, "losing_families")
        _require_nonnegative_int_antigravity(self.tied_families, "tied_families")
        _require_finite_float_antigravity(self.min_auc_gain, "min_auc_gain")
        _require_finite_float_antigravity(self.q20_auc_gain, "q20_auc_gain")


@dataclass(frozen=True)
class ResidualDecisionAntigravity:
    method: str
    changed: bool
    reason: str
    selected_candidate: str | None
    assessments: tuple[StrictCandidateAssessmentAntigravity, ...]

    def __post_init__(self) -> None:
        _require_method_antigravity(self.method)
        _require_exact_bool_antigravity(self.changed, "changed")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if self.selected_candidate is not None:
            _require_method_antigravity(self.selected_candidate)
        if type(self.assessments) is not tuple:
            raise ValueError("assessments must be a tuple")
        if any(
            type(item) is not StrictCandidateAssessmentAntigravity
            for item in self.assessments
        ):
            raise ValueError("assessments must contain StrictCandidateAssessmentAntigravity")


__all__ = [
    "InnerCandidateGateAntigravity",
    "ReferenceStatisticsAntigravity",
    "ResidualDecisionAntigravity",
    "StrictCandidateAssessmentAntigravity",
]
