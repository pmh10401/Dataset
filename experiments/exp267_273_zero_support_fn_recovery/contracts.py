from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _freeze_value(value: object) -> object:
    if isinstance(value, MappingABC):
        return MappingProxyType(
            {key: _freeze_value(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(child) for child in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(child) for child in value)
    return value


def _index_tuple(values: object, field_name: str) -> tuple[int, ...]:
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an index collection") from exc
    if any(type(index) is not int for index in items):
        raise TypeError(f"{field_name} must contain only integers")
    if any(index < 0 for index in items):
        raise ValueError(f"{field_name} must contain non-negative indices")
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} contains duplicate indices")
    return tuple(sorted(items))


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

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, MappingABC):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "provenance", _freeze_value(self.provenance))


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
        object.__setattr__(
            self,
            "candidate_indices",
            _index_tuple(self.candidate_indices, "candidate_indices"),
        )


@dataclass(frozen=True)
class RecoveryLanes:
    """Existing Exp266 lanes plus review-only recovery lanes."""

    hard_indices: tuple[int, ...]
    shadow_indices: tuple[int, ...]
    priority_indices: tuple[int, ...]
    standard_indices: tuple[int, ...]
    recovery_shadow_indices: tuple[int, ...]
    recovery_priority_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        names = (
            "hard_indices",
            "shadow_indices",
            "priority_indices",
            "standard_indices",
            "recovery_shadow_indices",
            "recovery_priority_indices",
        )
        normalized = {
            name: _index_tuple(getattr(self, name), name) for name in names
        }
        for position, name in enumerate(names):
            current = set(normalized[name])
            for other in names[position + 1 :]:
                if current.intersection(normalized[other]):
                    raise ValueError(f"lane overlap: {name} and {other}")
        for name, values in normalized.items():
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class PromotionAudit:
    """Hypothetical promotion result; it never activates an autonomous lane."""

    independent_groups: int
    train_u95: float
    precision: float
    overall_precision: float
    invariant: bool
    eligible_for_future_experiment: bool
