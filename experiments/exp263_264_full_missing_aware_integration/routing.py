from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import MIN_TRAIN_FOR_PROMOTION, REQUIRED_PROMOTION_AGREEMENT


SOURCE_NAMES = ("statistical", "structural", "change_point", "reconstruction")


@dataclass(frozen=True)
class RoutingInputs:
    """Label-free inputs for one dataset's final lane routing."""

    dataset: str
    n_train: int
    n_test: int
    baseline_hard: set[int]
    baseline_standard: set[int]
    baseline_priority: set[int]
    sources: Mapping[str, set[int]]
    quality_flags: set[int]


def _sorted(values: set[int]) -> list[int]:
    return sorted(values)


def _validate_indices(dataset: str, n_test: int, values: set[int]) -> None:
    if any(type(index) is not int for index in values):
        raise ValueError(f"non-integral index: dataset={dataset}")
    if any(index < 0 or index >= n_test for index in values):
        raise ValueError(f"candidate outside TEST range: dataset={dataset}")


def _validate_inputs(inputs: RoutingInputs) -> None:
    if type(inputs.n_train) is not int or inputs.n_train < 0:
        raise ValueError(f"invalid n_train: dataset={inputs.dataset}")
    if type(inputs.n_test) is not int or inputs.n_test < 0:
        raise ValueError(f"invalid n_test: dataset={inputs.dataset}")
    if set(inputs.sources) != set(SOURCE_NAMES):
        raise ValueError(f"invalid routing sources: dataset={inputs.dataset}")

    lanes = (
        inputs.baseline_hard,
        inputs.baseline_standard,
        inputs.baseline_priority,
    )
    if any(lanes[left] & lanes[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError(f"baseline lane overlap: dataset={inputs.dataset}")
    for values in (*lanes, inputs.quality_flags, *inputs.sources.values()):
        _validate_indices(inputs.dataset, inputs.n_test, values)


def route_dataset(inputs: RoutingInputs) -> dict[str, object]:
    """Route a dataset without consuming TEST labels or evaluation metrics."""

    _validate_inputs(inputs)
    standard_sources = {
        name: indices & inputs.baseline_standard
        for name, indices in inputs.sources.items()
    }
    support = {
        index: sum(index in indices for indices in standard_sources.values())
        for index in inputs.baseline_standard
    }
    promoted = {
        index
        for index, count in support.items()
        if inputs.n_train >= MIN_TRAIN_FOR_PROMOTION
        and count >= REQUIRED_PROMOTION_AGREEMENT
        and index not in inputs.quality_flags
    }
    priority_added = {
        index for index, count in support.items() if count >= 2 and index not in promoted
    }
    final_hard = inputs.baseline_hard | promoted
    final_priority = inputs.baseline_priority | priority_added
    final_standard = inputs.baseline_standard - promoted - priority_added

    final_lanes = (final_hard, final_standard, final_priority)
    if not inputs.baseline_hard <= final_hard:
        raise ValueError(f"baseline Hard was removed: dataset={inputs.dataset}")
    if any(
        final_lanes[left] & final_lanes[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise ValueError(f"final lane overlap: dataset={inputs.dataset}")

    return {
        "dataset": inputs.dataset,
        "n_train": inputs.n_train,
        "n_test": inputs.n_test,
        "baseline_hard_indices": _sorted(inputs.baseline_hard),
        "baseline_standard_indices": _sorted(inputs.baseline_standard),
        "baseline_priority_indices": _sorted(inputs.baseline_priority),
        "promoted_indices": _sorted(promoted),
        "final_hard_indices": _sorted(final_hard),
        "final_standard_indices": _sorted(final_standard),
        "final_priority_indices": _sorted(final_priority),
    }
