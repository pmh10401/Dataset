from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .contracts import FrozenDatasetInputs


@dataclass(frozen=True)
class RecoveryPrediction:
    dataset: str
    hard: frozenset[int]
    shadow: frozenset[int]
    priority: frozenset[int]
    standard: frozenset[int]
    recovery_shadow: frozenset[int]
    recovery_priority: frozenset[int]
    data_quality_review: frozenset[int]
    support_by_index: Mapping[int, int]


_BASELINE_LANES = ("hard", "shadow", "priority", "standard")
_RECOVERY_LANES = (
    "recovery_shadow",
    "recovery_priority",
    "data_quality_review",
)


def _validate_indices(
    values: Mapping[int, int] | frozenset[int] | set[int],
    description: str,
    n_test: int,
) -> None:
    for index in values:
        if type(index) is not int:
            raise ValueError(f"{description} contains non-integer index")
        if index < 0 or index >= n_test:
            raise ValueError(f"{description} contains index outside TEST range")


def _support_by_index(inputs: FrozenDatasetInputs) -> dict[int, int]:
    existing = inputs.hard | inputs.shadow | inputs.priority | inputs.standard
    for lane in _BASELINE_LANES:
        _validate_indices(getattr(inputs, lane), f"{lane} baseline", inputs.n_test)
    _validate_indices(inputs.quality_flags, "quality flags", inputs.n_test)
    for group, indices in inputs.evidence.items():
        _validate_indices(indices, f"{group} evidence", inputs.n_test)

    outside = {
        group: indices - existing
        for group, indices in inputs.evidence.items()
        if group in inputs.available_groups
    }
    support = Counter(index for indices in outside.values() for index in indices)
    return dict(sorted(support.items()))


def validate_recovery_prediction(
    inputs: FrozenDatasetInputs, prediction: RecoveryPrediction
) -> None:
    if prediction.dataset != inputs.dataset:
        raise ValueError("prediction dataset mismatch")

    for lane in (*_BASELINE_LANES, *_RECOVERY_LANES):
        _validate_indices(getattr(prediction, lane), lane, inputs.n_test)
    _validate_indices(prediction.support_by_index, "support", inputs.n_test)
    if any(type(count) is not int for count in prediction.support_by_index.values()):
        raise ValueError("support counts must be integers")

    for left, lane in enumerate((*_BASELINE_LANES, *_RECOVERY_LANES)):
        for other in (*_BASELINE_LANES, *_RECOVERY_LANES)[left + 1 :]:
            if getattr(prediction, lane) & getattr(prediction, other):
                raise ValueError(f"lane overlap: {lane} and {other}")

    for lane in _BASELINE_LANES:
        if getattr(prediction, lane) != getattr(inputs, lane):
            raise ValueError(f"baseline {lane} changed")

    support = _support_by_index(inputs)
    if dict(prediction.support_by_index) != support:
        raise ValueError("support mapping mismatch")
    quality = frozenset(index for index in support if index in inputs.quality_flags)
    expected = {
        "recovery_shadow": frozenset(
            index for index, count in support.items() if count >= 3 and index not in quality
        ),
        "recovery_priority": frozenset(
            index for index, count in support.items() if count == 2 and index not in quality
        ),
        "data_quality_review": quality,
    }
    for lane, indices in expected.items():
        if getattr(prediction, lane) != indices:
            raise ValueError(f"{lane} routing mismatch")


def route_recovery(inputs: FrozenDatasetInputs) -> RecoveryPrediction:
    support = _support_by_index(inputs)
    quality = frozenset(index for index in support if index in inputs.quality_flags)
    prediction = RecoveryPrediction(
        dataset=inputs.dataset,
        hard=inputs.hard,
        shadow=inputs.shadow,
        priority=inputs.priority,
        standard=inputs.standard,
        recovery_shadow=frozenset(
            index for index, count in support.items() if count >= 3 and index not in quality
        ),
        recovery_priority=frozenset(
            index for index, count in support.items() if count == 2 and index not in quality
        ),
        data_quality_review=quality,
        support_by_index=MappingProxyType(support),
    )
    validate_recovery_prediction(inputs, prediction)
    return prediction
