from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class SourceSpec:
    """Frozen identity and eligibility policy for one score source."""

    name: str
    family: str
    minimum_train: int
    variants: tuple[str, ...]

    @property
    def min_train(self) -> int:
        return self.minimum_train


@dataclass(frozen=True)
class SourceScore:
    """Label-free scores and provenance emitted by a source implementation."""

    source: str = ""
    variant: str = ""
    train_oof: object | None = None
    test_scores: object | None = None
    eligible: bool = True
    abstained: bool = False
    abstention_reason: str | None = None
    provenance: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class FrozenSourcePrediction:
    """A prediction freeze record that cannot carry TEST evaluation data."""

    dataset: str
    source: str
    variant: str
    alpha: float
    candidate_indices: tuple[int, ...]
    score_hash: str
    eligibility: bool
    abstention_reason: str | None
    config_hash: str
    contains_test_labels: bool = False

    def __post_init__(self) -> None:
        if self.contains_test_labels:
            raise ValueError("FrozenSourcePrediction cannot contain TEST labels")


@dataclass(frozen=True)
class RecoveryLanes:
    """Existing Exp266 lanes plus review-only recovery lanes."""

    hard_indices: tuple[int, ...] = ()
    shadow_indices: tuple[int, ...] = ()
    priority_indices: tuple[int, ...] = ()
    standard_indices: tuple[int, ...] = ()
    recovery_shadow_indices: tuple[int, ...] = ()
    recovery_priority_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class PromotionAudit:
    """Hypothetical promotion result; it never activates an autonomous lane."""

    independent_groups: int
    train_u95: float
    precision: float
    overall_precision: float
    invariant: bool
    eligible_for_future_experiment: bool
