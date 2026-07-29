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
from math import isfinite
import re
from types import MappingProxyType
from typing import Any

from .checkpoints import Exp417CheckpointStore
from .config import FROZEN_CONFIG
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


@dataclass(frozen=True)
class CheckpointEvidence:
    """The complete non-prediction evidence required by a checkpoint record."""

    raw_train_identity_sha256: str
    repaired_train_identity_sha256: str | None
    length_group_identity_sha256: Mapping[int, str]
    canonical_model_hashes: Mapping[int, Mapping[int, str]]
    score_calibration_hashes: Mapping[int, str]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        _sha256(self.raw_train_identity_sha256, "raw TRAIN identity")
        if self.repaired_train_identity_sha256 is not None:
            _sha256(self.repaired_train_identity_sha256, "repaired TRAIN identity")
        lengths = _digest_mapping(
            self.length_group_identity_sha256, "length-group identities"
        )
        calibrations = _digest_mapping(
            self.score_calibration_hashes, "score calibration hashes"
        )
        if tuple(lengths) != tuple(calibrations):
            raise ValueError("checkpoint length evidence must cover identical groups")
        if (
            not isinstance(self.canonical_model_hashes, Mapping)
            or tuple(self.canonical_model_hashes) != tuple(lengths)
        ):
            raise ValueError("checkpoint model hashes must cover identical groups")
        models: dict[int, Mapping[int, str]] = {}
        for length, hashes in self.canonical_model_hashes.items():
            if type(length) is not int or not isinstance(hashes, Mapping):
                raise ValueError("checkpoint model hashes are invalid")
            if set(hashes) != set(FROZEN_CONFIG.model_seeds):
                raise ValueError("checkpoint model hashes must cover frozen seeds")
            models[length] = MappingProxyType({
                seed: _sha256(digest, "canonical model hash")
                for seed, digest in hashes.items()
            })
        if (
            not isinstance(self.elapsed_seconds, (int, float))
            or isinstance(self.elapsed_seconds, bool)
            or not isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0.0
        ):
            raise ValueError("checkpoint elapsed seconds must be finite and non-negative")
        object.__setattr__(self, "length_group_identity_sha256", MappingProxyType(lengths))
        object.__setattr__(self, "canonical_model_hashes", MappingProxyType(models))
        object.__setattr__(self, "score_calibration_hashes", MappingProxyType(calibrations))
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))


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
    material = (
        None if checkpoints is None
        else build_checkpoint_evidence(train, processed, oof, contract, decisions)
    )
    checkpoint(result, checkpoints, material)
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
    if (
        not callable(getattr(checkpoints, "load_completed", None))
        or not hasattr(checkpoints, "expected_datasets")
    ):
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
    global_summary = apply_global_workload(frozen_results, total_test, context=context)
    if global_summary.status == "global_workload_gate_failed":
        return global_summary
    if any(result.status == "failed" for result in frozen_results):
        return RunSummary(frozen_results, total_test, False, "failed", context)
    return global_summary


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


def build_checkpoint_evidence(
    train: Any,
    processed: Sequence[Any],
    oof: Any,
    contract: FrozenTrainContract,
    decisions: Sequence[TestInstanceDecision],
) -> CheckpointEvidence:
    """Require an audited adapter to provide every transactional record field."""
    raise RuntimeError("Exp417 checkpoint evidence boundary is not configured")


def checkpoint(
    result: DatasetResult,
    checkpoints: Exp417CheckpointStore | None,
    material: CheckpointEvidence | None = None,
) -> None:
    """Publish a result only through Task 8's transactional store API."""
    if checkpoints is None:
        return
    if not isinstance(material, CheckpointEvidence):
        raise ValueError("checkpoint material must be complete typed evidence")
    if not callable(getattr(checkpoints, "write_dataset", None)):
        raise ValueError("checkpoints must provide transactional dataset writes")
    checkpoints.write_dataset(
        result,
        raw_train_identity_sha256=material.raw_train_identity_sha256,
        repaired_train_identity_sha256=material.repaired_train_identity_sha256,
        length_group_identity_sha256=dict(material.length_group_identity_sha256),
        canonical_model_hashes={
            length: dict(hashes)
            for length, hashes in material.canonical_model_hashes.items()
        },
        score_calibration_hashes=dict(material.score_calibration_hashes),
        backend_audit=result.decision_evidence.frozen_train_contract.backend,
        elapsed_seconds=material.elapsed_seconds,
    )


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


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _digest_mapping(values: object, name: str) -> dict[int, str]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[int, str] = {}
    for length, digest in values.items():
        if type(length) is not int or length < 1:
            raise ValueError(f"{name} must use positive integer lengths")
        result[length] = _sha256(digest, name)
    if not result or tuple(result) != tuple(sorted(result)):
        raise ValueError(f"{name} must be non-empty and sorted")
    return result


__all__ = [
    "TrainQualification",
    "CheckpointEvidence",
    "build_prediction_rows",
    "resume_or_run_population",
    "run_dataset",
]
