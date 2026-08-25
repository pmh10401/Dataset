"""Strict typed-abstain classification for frozen public-baseline inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Literal


BASELINE_METHODS_ANTIGRAVITY = (
    "knn_distance",
    "lof_novelty",
    "ocsvm_rbf",
    "isolation_forest",
    "deep_svdd_1d",
    "drocc_1d",
)
_ABSTENTION_REASON_ANTIGRAVITY = "insufficient_train_rows"


@dataclass(frozen=True, slots=True)
class Exp471BaselinePreflightEntryAntigravity:
    method: str
    dataset: str
    status: Literal["scored", "abstained"]
    test_indices: tuple[int, ...]
    train_oof_scores: tuple[float, ...]
    test_scores: tuple[float, ...]
    abstention_reason: str | None


@dataclass(frozen=True, slots=True)
class Exp471BaselinePreflightAntigravity:
    entries: Mapping[tuple[str, str], Exp471BaselinePreflightEntryAntigravity]
    scored_count: int
    abstained_count: int
    failed_count: int


def _invalid_antigravity(detail: str) -> ValueError:
    return ValueError(f"Exp471 baseline preflight {detail} differs")


def _float_vector_antigravity(value: object) -> tuple[float, ...]:
    if not isinstance(value, (Sequence, list, tuple)) or isinstance(value, (str, bytes)):
        raise _invalid_antigravity("baseline score vector")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise _invalid_antigravity("baseline score vector") from error
    if not all(math.isfinite(item) for item in result):
        raise _invalid_antigravity("baseline score vector")
    return result


def _entry_from_shard_antigravity(
    method: str,
    dataset: str,
    shard: Mapping[str, object],
) -> Exp471BaselinePreflightEntryAntigravity:
    count = shard.get("count")
    indices = shard.get("indices")
    if (
        type(count) is not int
        or count < 1
        or not isinstance(indices, (Sequence, list, tuple))
        or isinstance(indices, (str, bytes))
        or len(indices) != count
        or len(set(indices)) != count
        or any(type(index) is not int or index < 0 for index in indices)
        or tuple(indices) != tuple(range(count))
    ):
        raise _invalid_antigravity(f"TEST coordinates for {method}/{dataset}")
    expected_indices = tuple(indices)
    status = shard.get("status")
    if type(status) is not str:
        raise _invalid_antigravity(f"source status for {method}/{dataset}")
    reason = shard.get("reason")
    if status == "complete":
        if reason is not None:
            raise _invalid_antigravity(f"scored reason for {method}/{dataset}")
        return Exp471BaselinePreflightEntryAntigravity(
            method=method,
            dataset=dataset,
            status="scored",
            test_indices=expected_indices,
            train_oof_scores=_float_vector_antigravity(shard.get("train_oof_scores")),
            test_scores=_float_vector_antigravity(shard.get("test_scores")),
            abstention_reason=None,
        )
    if status == "abstain":
        if reason != _ABSTENTION_REASON_ANTIGRAVITY:
            raise _invalid_antigravity(f"abstention reason for {method}/{dataset}")
        forbidden = ("train_oof_scores", "test_scores")
        if any(key in shard and shard[key] not in (None,) for key in forbidden):
            raise _invalid_antigravity(
                f"nonempty abstention scores for {method}/{dataset}"
            )
        return Exp471BaselinePreflightEntryAntigravity(
            method=method,
            dataset=dataset,
            status="abstained",
            test_indices=expected_indices,
            train_oof_scores=(),
            test_scores=(),
            abstention_reason=_ABSTENTION_REASON_ANTIGRAVITY,
        )
    raise _invalid_antigravity(f"source status for {method}/{dataset}")


def preflight_frozen_baselines_antigravity(
    *,
    datasets: tuple[str, ...],
    coordinates: Mapping[str, object],
) -> Exp471BaselinePreflightAntigravity:
    """Classify every baseline identity without exposing source failure states."""

    if (
        not isinstance(datasets, tuple)
        or not datasets
        or any(type(dataset) is not str or not dataset for dataset in datasets)
        or len(datasets) != len(set(datasets))
    ):
        raise _invalid_antigravity("dataset order")
    entries: dict[tuple[str, str], Exp471BaselinePreflightEntryAntigravity] = {}
    scored_count = 0
    abstained_count = 0
    methods = tuple(coordinates)
    if (
        methods != BASELINE_METHODS_ANTIGRAVITY
    ):
        raise _invalid_antigravity("method order")
    for method in methods:
        rows = coordinates.get(method)
        if not isinstance(rows, Mapping) or tuple(rows) != datasets:
            raise _invalid_antigravity(f"coverage for {method}")
        for dataset in datasets:
            shard = rows[dataset]
            if not isinstance(shard, Mapping):
                raise _invalid_antigravity(f"shard for {method}/{dataset}")
            entry = _entry_from_shard_antigravity(method, dataset, shard)
            key = (method, dataset)
            if key in entries:
                raise _invalid_antigravity("duplicate identity")
            entries[key] = entry
            if (
                entry.status == "scored"
                and (
                    len(entry.train_oof_scores) != shard["count"]
                    or len(entry.test_scores) != shard["count"]
                )
            ):
                raise _invalid_antigravity(
                    f"score vector lengths for {method}/{dataset}"
                )
            if entry.status == "scored":
                scored_count += 1
            else:
                abstained_count += 1
    return Exp471BaselinePreflightAntigravity(
        entries=MappingProxyType(entries),
        scored_count=scored_count,
        abstained_count=abstained_count,
        failed_count=0,
    )


__all__ = [
    "BASELINE_METHODS_ANTIGRAVITY",
    "Exp471BaselinePreflightAntigravity",
    "Exp471BaselinePreflightEntryAntigravity",
    "preflight_frozen_baselines_antigravity",
]
