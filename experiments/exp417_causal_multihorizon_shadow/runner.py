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
from hashlib import sha256
from math import isfinite
import re
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np

from .checkpoints import Exp417CheckpointStore
from .config import FROZEN_CONFIG
from .contracts import (
    BackendAudit,
    DatasetDecisionEvidence,
    DatasetResult,
    FrozenRunContext,
    FrozenTrainContract,
    GroupQualification,
    ProcessedGroup,
    RawSeries,
    RawTrainDataset,
    RunSummary,
    TestInstanceDecision,
)
from .model import OOFModelBundle, TrainedSeedModel
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


@dataclass(frozen=True)
class TrainModelEvidence:
    """Per-length OOF and final-model evidence actually available to scoring."""

    oof_bundles: Mapping[int, OOFModelBundle]
    final_models: Mapping[int, Sequence[TrainedSeedModel]]
    backend: BackendAudit
    train_identity_sha256: str

    def __post_init__(self) -> None:
        bundles = _oof_bundles(self.oof_bundles)
        models = _final_models(self.final_models, tuple(bundles))
        if not isinstance(self.backend, BackendAudit):
            raise ValueError("model evidence requires a backend audit")
        _sha256(self.train_identity_sha256, "model evidence TRAIN contract identity")
        object.__setattr__(self, "oof_bundles", MappingProxyType(bundles))
        object.__setattr__(self, "final_models", MappingProxyType(models))


@dataclass(frozen=True)
class TestScoringEvidence:
    """Exhaustive TEST decisions and exact final-model identities used to score them."""

    decisions: tuple[TestInstanceDecision, ...]
    final_model_hashes: Mapping[int, Mapping[int, str]]

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        if any(not isinstance(decision, TestInstanceDecision) for decision in decisions):
            raise ValueError("TEST scoring evidence requires instance decisions")
        hashes = _model_hashes(self.final_model_hashes)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "final_model_hashes", MappingProxyType(hashes))


def run_dataset(dataset: str, context: FrozenRunContext) -> DatasetResult:
    """Run one frozen dataset through the sequential label-free state machine."""
    return _run_dataset(dataset, context, None)


def _run_dataset(
    dataset: str,
    context: FrozenRunContext,
    checkpoints: Exp417CheckpointStore | None,
) -> DatasetResult:
    _validate_dataset(dataset, context)
    started = perf_counter()
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
    scoring = score_test(test, contract, oof)
    if not isinstance(scoring, TestScoringEvidence):
        raise ValueError("TEST scoring boundary returned invalid evidence")
    decisions = scoring.decisions
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
        else build_checkpoint_evidence(
            train, processed, oof, contract, scoring, perf_counter() - started
        )
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


def train_oof(processed: Sequence[Any]) -> TrainModelEvidence:
    raise RuntimeError("Exp417 OOF boundary is not configured")


def qualify_train(
    dataset: str, processed: Sequence[Any], oof: TrainModelEvidence
) -> TrainQualification:
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
    test: Any, contract: FrozenTrainContract, oof: TrainModelEvidence
) -> TestScoringEvidence:
    raise RuntimeError("Exp417 TEST scoring boundary is not configured")


def exclude_lanes(evidence: DatasetDecisionEvidence):
    return exclude_frozen_lanes(evidence)


def dataset_workload(evidence: DatasetDecisionEvidence) -> DatasetResult:
    return apply_dataset_workload(evidence)


def build_checkpoint_evidence(
    train: RawTrainDataset,
    processed: Sequence[ProcessedGroup],
    oof: TrainModelEvidence,
    contract: FrozenTrainContract,
    scoring: TestScoringEvidence,
    elapsed_seconds: float,
) -> CheckpointEvidence:
    """Derive every Task 8 record field from typed immutable stage evidence."""
    if not isinstance(train, RawTrainDataset):
        raise ValueError("checkpoint evidence requires a RawTrainDataset")
    groups = _processed_groups(processed, train)
    if not isinstance(oof, TrainModelEvidence):
        raise ValueError("checkpoint evidence requires TrainModelEvidence")
    if not isinstance(contract, FrozenTrainContract) or contract.dataset != train.dataset:
        raise ValueError("checkpoint evidence contract does not match TRAIN dataset")
    if not isinstance(scoring, TestScoringEvidence):
        raise ValueError("checkpoint evidence requires TestScoringEvidence")
    if oof.train_identity_sha256 != contract.train_identity_sha256:
        raise ValueError("checkpoint model evidence TRAIN contract identity mismatch")
    if oof.backend != contract.backend:
        raise ValueError("checkpoint model evidence backend mismatch")
    lengths = tuple(group.group.length for group in groups)
    if tuple(group.length for group in contract.groups) != lengths:
        raise ValueError("checkpoint processed lengths do not match frozen contract")
    if any(
        qualification.series_count != len(group.group.series)
        for qualification, group in zip(contract.groups, groups, strict=True)
    ):
        raise ValueError("checkpoint processed group counts do not match frozen contract")
    if tuple(oof.oof_bundles) != lengths or tuple(oof.final_models) != lengths:
        raise ValueError("checkpoint model evidence lengths do not match frozen contract")
    expected_hashes = {
        length: {model.seed: model.model_sha256 for model in oof.final_models[length]}
        for length in lengths
    }
    if dict(scoring.final_model_hashes) != expected_hashes:
        raise ValueError("TEST scoring did not use the exact final model evidence")
    if any(decision.length not in set(lengths) for decision in scoring.decisions):
        raise ValueError("TEST scoring evidence includes an unknown TRAIN length")
    return CheckpointEvidence(
        raw_train_identity_sha256=_series_digest("exp417.raw-train.v1", train.dataset, train.series),
        repaired_train_identity_sha256=_groups_digest("exp417.repaired-train.v1", train.dataset, groups),
        length_group_identity_sha256={
            group.group.length: _series_digest(
                "exp417.repaired-length-group.v1", train.dataset, group.group.series
            )
            for group in groups
        },
        canonical_model_hashes=expected_hashes,
        score_calibration_hashes={
            length: _oof_calibration_digest(oof.oof_bundles[length])
            for length in lengths
        },
        elapsed_seconds=elapsed_seconds,
    )


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


def _oof_bundles(values: object) -> dict[int, OOFModelBundle]:
    if not isinstance(values, Mapping):
        raise ValueError("OOF evidence must map exact lengths to bundles")
    bundles: dict[int, OOFModelBundle] = {}
    for length, bundle in values.items():
        if type(length) is not int or not isinstance(bundle, OOFModelBundle) or bundle.length != length:
            raise ValueError("OOF evidence length binding is invalid")
        bundles[length] = bundle
    if not bundles or tuple(bundles) != tuple(sorted(bundles)):
        raise ValueError("OOF evidence lengths must be non-empty and sorted")
    return bundles


def _final_models(
    values: object, lengths: tuple[int, ...]
) -> dict[int, tuple[TrainedSeedModel, ...]]:
    if not isinstance(values, Mapping) or tuple(values) != lengths:
        raise ValueError("final model evidence must cover exact sorted lengths")
    result: dict[int, tuple[TrainedSeedModel, ...]] = {}
    for length in lengths:
        models = tuple(values[length]) if not isinstance(values[length], (str, bytes)) else ()
        if (
            len(models) != len(FROZEN_CONFIG.model_seeds)
            or any(not isinstance(model, TrainedSeedModel) for model in models)
            or tuple(model.seed for model in models) != FROZEN_CONFIG.model_seeds
        ):
            raise ValueError("final model evidence must contain each frozen seed once")
        if any(model.horizons != _horizons_for_length(length) for model in models):
            raise ValueError("final model evidence horizons do not match exact length")
        result[length] = models
    return result


def _model_hashes(values: object) -> dict[int, Mapping[int, str]]:
    if not isinstance(values, Mapping):
        raise ValueError("TEST scoring model hashes must be a mapping")
    result: dict[int, Mapping[int, str]] = {}
    for length, hashes in values.items():
        if type(length) is not int or length < 1 or not isinstance(hashes, Mapping):
            raise ValueError("TEST scoring model hash lengths are invalid")
        if tuple(hashes) != FROZEN_CONFIG.model_seeds:
            raise ValueError("TEST scoring model hashes must cover frozen seeds in order")
        result[length] = MappingProxyType({
            seed: _sha256(digest, "TEST scoring model hash")
            for seed, digest in hashes.items()
        })
    if not result or tuple(result) != tuple(sorted(result)):
        raise ValueError("TEST scoring model hash lengths must be non-empty and sorted")
    return result


def _processed_groups(
    values: object, train: RawTrainDataset
) -> tuple[ProcessedGroup, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("checkpoint processed groups must be a sequence")
    groups = tuple(values)
    if not groups or any(not isinstance(group, ProcessedGroup) for group in groups):
        raise ValueError("checkpoint processed groups are invalid")
    lengths = tuple(group.group.length for group in groups)
    if lengths != tuple(sorted(lengths)) or len(lengths) != len(set(lengths)):
        raise ValueError("checkpoint processed groups must have sorted unique lengths")
    raw_by_length: dict[int, tuple[RawSeries, ...]] = {}
    for series in train.series:
        raw_by_length.setdefault(len(series.values), tuple())
        raw_by_length[len(series.values)] += (series,)
    if tuple(sorted(raw_by_length)) != lengths:
        raise ValueError("checkpoint processed groups do not cover exact raw TRAIN lengths")
    for group in groups:
        raw_indices = tuple(item.instance_index for item in raw_by_length[group.group.length])
        repaired_indices = tuple(item.instance_index for item in group.group.series)
        if repaired_indices != raw_indices:
            raise ValueError("checkpoint repaired groups do not preserve raw TRAIN identities")
    return groups


def _series_digest(schema: str, dataset: str, series: Sequence[RawSeries]) -> str:
    digest = sha256()
    _digest_text(digest, schema)
    _digest_text(digest, dataset)
    digest.update(len(series).to_bytes(8, "little", signed=False))
    for item in series:
        if not isinstance(item, RawSeries):
            raise ValueError("checkpoint identity series are invalid")
        digest.update(item.instance_index.to_bytes(8, "little", signed=False))
        values = np.asarray(item.values, dtype="<f4")
        if values.ndim != 1 or len(values) != len(item.values):
            raise ValueError("checkpoint identity values are invalid")
        digest.update(len(values).to_bytes(8, "little", signed=False))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _groups_digest(schema: str, dataset: str, groups: Sequence[ProcessedGroup]) -> str:
    digest = sha256()
    _digest_text(digest, schema)
    _digest_text(digest, dataset)
    digest.update(len(groups).to_bytes(8, "little", signed=False))
    for group in groups:
        digest.update(group.group.length.to_bytes(8, "little", signed=False))
        _digest_text(digest, group.status)
        _digest_text(digest, _series_digest("exp417.repaired-group-members.v1", dataset, group.group.series))
    return digest.hexdigest()


def _oof_calibration_digest(bundle: OOFModelBundle) -> str:
    digest = sha256()
    _digest_text(digest, "exp417.oof-calibration.v1")
    digest.update(bundle.length.to_bytes(8, "little", signed=False))
    for horizon in bundle.horizons:
        digest.update(horizon.to_bytes(8, "little", signed=False))
    for seed in FROZEN_CONFIG.model_seeds:
        digest.update(seed.to_bytes(8, "little", signed=False))
        by_index = bundle.oof_residuals[seed]
        for index in bundle.universe_indices:
            digest.update(index.to_bytes(8, "little", signed=False))
            for horizon in bundle.horizons:
                residual = by_index[index][horizon]
                if residual.flags.writeable or not np.isfinite(residual).all():
                    raise ValueError("OOF calibration residual evidence must be immutable and finite")
                payload = np.asarray(residual, dtype="<f8").tobytes(order="C")
                digest.update(len(payload).to_bytes(8, "little", signed=False))
                digest.update(payload)
    return digest.hexdigest()


def _digest_text(digest: Any, value: str) -> None:
    payload = value.encode("utf-8")
    digest.update(len(payload).to_bytes(8, "little", signed=False))
    digest.update(payload)


def _horizons_for_length(length: int) -> tuple[int, int, int]:
    from .preprocessing import build_horizons
    return build_horizons(length)


__all__ = [
    "TrainQualification",
    "CheckpointEvidence",
    "TestScoringEvidence",
    "TrainModelEvidence",
    "build_prediction_rows",
    "resume_or_run_population",
    "run_dataset",
]
