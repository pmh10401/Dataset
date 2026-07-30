"""Deterministic TRAIN-only workload cost estimation for Exp419."""

from collections.abc import Sequence

from .contracts import DatasetCost, ShardAssignment, ShardPlan, TrainLengthGroupMeta
from .config import FROZEN_CONFIG


_SIGNED_64_BIT_MAX = 9_223_372_036_854_775_807


def estimate_dataset_cost(
    dataset: str, groups: Sequence[TrainLengthGroupMeta]
) -> DatasetCost:
    """Return Exp419's frozen training-work cost proxy for one dataset."""
    if not groups:
        raise ValueError("groups must not be empty")

    cost = 0
    largest_count = 0
    for group in groups:
        if not isinstance(group, TrainLengthGroupMeta):
            raise ValueError("groups must contain TrainLengthGroupMeta records")
        if group.dataset != dataset:
            raise ValueError("groups must match dataset")
        weight = 1 if group.series_count < 50 else 30 if group.series_count < 100 else 81
        increment = group.series_count * group.length * weight
        if increment > _SIGNED_64_BIT_MAX - cost:
            raise OverflowError("cost exceeds signed 64-bit maximum")
        cost += increment
        largest_count = max(largest_count, group.series_count)

    if largest_count < 50:
        stratum = "preprocessing_only"
    elif largest_count < 100:
        stratum = "diagnostic_oof"
    else:
        stratum = "candidate_eligible"
    return DatasetCost(dataset, cost, stratum)


def select_benchmark_datasets(costs: Sequence[DatasetCost]) -> tuple[str, ...]:
    """Select frozen workload quantiles from each TRAIN-derived stratum."""
    selected: list[str] = []
    for stratum in ("preprocessing_only", "diagnostic_oof", "candidate_eligible"):
        members = sorted(
            (cost for cost in costs if cost.stratum == stratum),
            key=lambda cost: (cost.cost, cost.dataset),
        )
        for quantile in FROZEN_CONFIG.benchmark_quantiles:
            if members:
                dataset = members[int(quantile * (len(members) - 1))].dataset
                if dataset not in selected:
                    selected.append(dataset)
    return tuple(sorted(selected))


def build_shard_plan(costs: Sequence[DatasetCost], worker_count: int) -> ShardPlan:
    """Assign descending-cost datasets to the current lightest frozen shard."""
    if type(worker_count) is not int or worker_count not in FROZEN_CONFIG.worker_candidates:
        raise ValueError("worker_count is not frozen")
    if len(costs) < worker_count:
        raise ValueError("costs must cover each worker")
    if any(not isinstance(cost, DatasetCost) for cost in costs):
        raise ValueError("costs must contain DatasetCost records")

    shard_datasets: list[list[str]] = [[] for _ in range(worker_count)]
    shard_totals = [0] * worker_count
    for cost in sorted(costs, key=lambda item: (-item.cost, item.dataset)):
        shard_index = min(range(worker_count), key=lambda index: (shard_totals[index], index))
        shard_datasets[shard_index].append(cost.dataset)
        shard_totals[shard_index] += cost.cost

    assignments = tuple(
        ShardAssignment(index, tuple(shard_datasets[index]), shard_totals[index])
        for index in range(worker_count)
    )
    return ShardPlan(
        worker_count,
        assignments,
        tuple(sorted(cost.dataset for cost in costs)),
    )
