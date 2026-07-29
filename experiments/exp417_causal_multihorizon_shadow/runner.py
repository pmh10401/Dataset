"""Sequential, label-free orchestration for the frozen Exp417 workflow.

This module deliberately has no production database configuration.  The only
data-bearing stages are private injectable boundaries, so a caller must supply
an audited values-only implementation before it can run against a database.
That keeps this orchestration layer incapable of discovering TEST labels or
silently substituting an unverified artifact path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .checkpoints import Exp417CheckpointStore
from .contracts import (
    BackendAudit,
    DatasetDecisionEvidence,
    DatasetResult,
    FrozenRunContext,
    FrozenTrainContract,
    GroupQualification,
    RunSummary,
    TestInstanceDecision,
)
from .inputs import assert_label_free_sql_contract
from .qualification import (
    apply_dataset_workload,
    apply_global_workload,
    exclude_frozen_lanes,
    freeze_train_contract,
)


@dataclass(frozen=True)
class TrainQualification:
    """TRAIN-only qualification output handed to the contract-freezing stage."""

    groups: tuple[GroupQualification, ...]
    backend: BackendAudit

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if not groups or any(not isinstance(group, GroupQualification) for group in groups):
            raise ValueError("TRAIN qualification requires GroupQualification records")
        if tuple(group.length for group in groups) != tuple(sorted(group.length for group in groups)):
            raise ValueError("TRAIN qualification groups must be sorted")
        if not isinstance(self.backend, BackendAudit):
            raise ValueError("TRAIN qualification requires a backend audit")
        object.__setattr__(self, "groups", groups)


def run_dataset(dataset: str, context: FrozenRunContext) -> DatasetResult:
    """Run one frozen dataset through the sequential label-free state machine."""
    return _run_dataset(dataset, context, None)


def _run_dataset(
    dataset: str,
    context: FrozenRunContext,
    checkpoints: Exp417CheckpointStore | None,
) -> DatasetResult:
    _validate_dataset(dataset, context)
    verify_inputs(context)
    train = load_train_values(dataset, context)
    processed = preprocess(train)
    oof = train_oof(processed)
    qualification = qualify_train(dataset, processed, oof)
    if not isinstance(qualification, TrainQualification):
        raise ValueError("TRAIN qualification boundary returned an invalid record")
    contract = freeze_contract(dataset, qualification)
    if not isinstance(contract, FrozenTrainContract) or contract.dataset != dataset:
        raise ValueError("frozen TRAIN contract boundary returned an invalid contract")
    test = load_test_values(dataset, contract, context)
    decisions = tuple(score_test(test, contract, oof))
    evidence = DatasetDecisionEvidence(
        dataset=dataset,
        context=context,
        frozen_train_contract=contract,
        database_sha256=context.database_sha256,
        train_identity_sha256=contract.train_identity_sha256,
        decisions=decisions,
    )
    excluded = tuple(exclude_lanes(evidence))
    if excluded != evidence.derived_candidates:
        raise ValueError("frozen-lane exclusion must retain every and only eligible coordinate")
    result = dataset_workload(evidence)
    _validate_result(result, dataset, context, evidence)
    checkpoint(result, checkpoints)
    return result


def resume_or_run_population(
    context: FrozenRunContext, checkpoints: Exp417CheckpointStore
) -> RunSummary:
    """Resume exactly the incomplete frozen datasets in sorted, sequential order."""
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be a FrozenRunContext")
    # Tests and audited integration wrappers may expose the transactional
    # store through a narrow protocol; require the two operations used here
    # instead of reaching into a concrete filesystem implementation.
    if not callable(getattr(checkpoints, "load_completed", None)) or not hasattr(checkpoints, "expected_datasets"):
        raise ValueError("checkpoints must provide the Exp417 checkpoint protocol")
    if checkpoints.expected_datasets != context.datasets:
        raise ValueError("checkpoint dataset universe does not match the frozen context")
    completed = checkpoints.load_completed()
    if set(completed) - set(context.datasets):
        raise ValueError("checkpoint contains a dataset outside the frozen context")

    results: list[DatasetResult] = []
    for dataset in context.datasets:
        result = completed.get(dataset)
        if result is None:
            result = _run_dataset(dataset, context, checkpoints)
        _validate_result(result, dataset, context)
        results.append(result)

    frozen_results = tuple(results)
    total_test = sum(context.test_sizes.values())
    if any(result.status == "failed" for result in frozen_results):
        return RunSummary(frozen_results, total_test, False, "failed", context)
    return apply_global_workload(frozen_results, total_test, context=context)


def build_prediction_rows(summary: RunSummary) -> tuple[Mapping[str, object], ...]:
    """Return sorted candidate coordinates with explicit false governance flags.

    Rows intentionally contain no score, evaluation, or label material.  A
    blocked global publication has no official payload, even if local evidence
    was checkpointed before the global workload decision.
    """
    if not isinstance(summary, RunSummary):
        raise ValueError("summary must be a RunSummary")
    if not summary.publication_allowed:
        return ()
    rows: list[Mapping[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for result in summary.results:
        for candidate in result.candidates:
            coordinate = (candidate.dataset, candidate.instance_index)
            if coordinate in seen:
                raise ValueError("prediction rows contain duplicate coordinates")
            if summary.context.lanes.contains(*coordinate):
                raise ValueError("prediction row overlaps a frozen lane")
            seen.add(coordinate)
            rows.append(MappingProxyType({
                "dataset": candidate.dataset,
                "instance_index": candidate.instance_index,
                "is_hard": False,
                "is_standard": False,
                "is_priority": False,
                "is_automatic_normal": False,
            }))
    return tuple(rows)


# The following are intentionally narrow value-only seams.  They are patched
# by synthetic tests or replaced by an audited integration entrypoint; this
# module never supplies a database path or an artifact path on its own.
def verify_inputs(context: FrozenRunContext) -> None:
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be a FrozenRunContext")
    assert_label_free_sql_contract()


def load_train_values(dataset: str, context: FrozenRunContext) -> Any:
    raise RuntimeError("Exp417 TRAIN values boundary is not configured")


def preprocess(train: Any) -> Sequence[Any]:
    raise RuntimeError("Exp417 preprocessing boundary is not configured")


def train_oof(processed: Sequence[Any]) -> Any:
    raise RuntimeError("Exp417 OOF boundary is not configured")


def qualify_train(dataset: str, processed: Sequence[Any], oof: Any) -> TrainQualification:
    raise RuntimeError("Exp417 TRAIN qualification boundary is not configured")


def freeze_contract(dataset: str, qualification: TrainQualification) -> FrozenTrainContract:
    if not isinstance(qualification, TrainQualification):
        raise ValueError("qualification must be a TrainQualification")
    return freeze_train_contract(dataset, qualification.groups, qualification.backend)


def load_test_values(
    dataset: str, contract: FrozenTrainContract, context: FrozenRunContext
) -> Any:
    raise RuntimeError("Exp417 TEST values boundary is not configured")


def score_test(
    test: Any, contract: FrozenTrainContract, oof: Any
) -> Sequence[TestInstanceDecision]:
    raise RuntimeError("Exp417 TEST scoring boundary is not configured")


def exclude_lanes(evidence: DatasetDecisionEvidence):
    return exclude_frozen_lanes(evidence)


def dataset_workload(evidence: DatasetDecisionEvidence) -> DatasetResult:
    return apply_dataset_workload(evidence)


def checkpoint(result: DatasetResult, checkpoints: Exp417CheckpointStore | None) -> None:
    """Reserved transactional boundary; no store is touched without full evidence."""
    if checkpoints is not None:
        raise RuntimeError("Exp417 checkpoint material boundary is not configured")


def _validate_dataset(dataset: str, context: FrozenRunContext) -> None:
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be a FrozenRunContext")
    if not isinstance(dataset, str) or dataset not in context.datasets:
        raise ValueError("dataset must be an exact frozen context dataset")


def _validate_result(
    result: object,
    dataset: str,
    context: FrozenRunContext,
    evidence: DatasetDecisionEvidence | None = None,
) -> None:
    if not isinstance(result, DatasetResult):
        raise ValueError("dataset boundary returned an invalid result")
    if result.dataset != dataset or result.context != context or result.lanes != context.lanes:
        raise ValueError("dataset result does not bind the frozen context")
    if evidence is not None and result.decision_evidence != evidence:
        raise ValueError("dataset workload result does not bind complete decision evidence")


__all__ = [
    "TrainQualification",
    "build_prediction_rows",
    "resume_or_run_population",
    "run_dataset",
]
