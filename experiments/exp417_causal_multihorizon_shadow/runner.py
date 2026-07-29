"""Sequential, label-free orchestration for the frozen Exp417 workflow.

The production runtime is explicit and immutable.  Synthetic tests inject a
``RunnerRuntime``; production uses only the frozen paths and public Task 2--8
functions defined by this experiment.  TEST values are consequently not
reachable until a concrete TRAIN contract has been frozen.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from pathlib import Path
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
    DatasetTerminalEvidence,
    FrozenRunContext,
    FrozenTrainContract,
    GroupQualification,
    ProcessedGroup,
    RawSeries,
    RawTrainDataset,
    RunSummary,
    TestInstanceDecision,
)
from .model import OOFModelBundle, TrainedSeedModel, fit_final_seed_models, predict_residuals, train_oof_models
from .inputs import (
    assert_label_free_sql_contract, load_raw_test_values, load_raw_train_values,
    verify_input_snapshots,
)
from .preprocessing import preprocess_length_groups
from experiments.exp403_408.paths import (
    DB_PATH, EXP291_MANIFEST_PATH, EXP291_PREDICTION_PATH,
    EXP292_MANIFEST_PATH, EXP292_PREDICTION_PATH, EXP417_BACKEND_AUDIT_PATH,
)
from .backend import load_frozen_backend_audit
from .qualification import (
    apply_dataset_workload,
    apply_global_workload,
    build_fixed_perturbations,
    exclude_frozen_lanes,
    freeze_train_contract,
    qualify_group,
)
from .scoring import consensus_candidate, score_horizon, seed_decision


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
class RunnerResources:
    """Pinned production inputs; paths are never inferred from a TEST query."""

    database: Path
    exp291_prediction: Path
    exp291_manifest: Path
    exp292_prediction: Path
    exp292_manifest: Path
    backend_audit_path: Path = EXP417_BACKEND_AUDIT_PATH

    @classmethod
    def frozen_defaults(cls) -> "RunnerResources":
        return cls(DB_PATH, EXP291_PREDICTION_PATH, EXP291_MANIFEST_PATH,
                   EXP292_PREDICTION_PATH, EXP292_MANIFEST_PATH,
                   EXP417_BACKEND_AUDIT_PATH)


@dataclass(frozen=True)
class RunnerRuntime:
    """Typed value-only stage plan used by production and synthetic tests.

    Every stage is an explicit keyword-only dependency.  This eliminates the
    former module-global monkeypatch path and makes the test-load barrier part
    of the callable signature itself.
    """

    verify: Callable[[FrozenRunContext], None]
    load_train: Callable[[str, FrozenRunContext], RawTrainDataset]
    preprocess: Callable[[RawTrainDataset], Sequence[ProcessedGroup]]
    train_models: Callable[[Sequence[ProcessedGroup]], "TrainModelEvidence"]
    qualify: Callable[[str, Sequence[ProcessedGroup], "TrainModelEvidence"], TrainQualification]
    freeze: Callable[[str, TrainQualification, "TrainModelEvidence"], FrozenTrainContract]
    load_test: Callable[[str, FrozenTrainContract, FrozenRunContext], Any]
    score: Callable[[Any, FrozenTrainContract, "TrainModelEvidence"], "TestScoringEvidence"]
    exclude: Callable[[DatasetDecisionEvidence], Sequence[Any]] = exclude_frozen_lanes
    workload: Callable[[DatasetDecisionEvidence], DatasetResult] = apply_dataset_workload

    @classmethod
    def production(cls, resources: RunnerResources | None = None) -> "RunnerRuntime":
        pinned = RunnerResources.frozen_defaults() if resources is None else resources
        if not isinstance(pinned, RunnerResources):
            raise ValueError("resources must be RunnerResources")
        audit = load_frozen_backend_audit(pinned.backend_audit_path)
        if not isinstance(audit, BackendAudit) or audit.fallback:
            raise ValueError("production resources require a passing Task 7 backend audit")
        if audit.backend not in {"cpu", "mps"}:
            raise ValueError("selected Task 7 backend must be cpu or mps for model execution")
        return cls(
            verify=lambda context: _production_verify(context, pinned),
            load_train=lambda dataset, context: load_raw_train_values(pinned.database, dataset, context),
            preprocess=preprocess_length_groups,
            train_models=lambda processed: _production_train_models(processed, audit),
            qualify=_production_qualify_train,
            freeze=lambda dataset, qualification, models: freeze_train_contract(
                dataset, qualification.groups, qualification.backend,
                train_identity_sha256=models.train_identity_sha256,
            ),
            load_test=lambda dataset, contract, context: load_raw_test_values(
                pinned.database, dataset, contract, context
            ),
            score=_production_score_test,
        )


@dataclass(frozen=True)
class CheckpointEvidence:
    """The complete non-prediction evidence required by a checkpoint record."""

    raw_train_identity_sha256: str
    repaired_train_identity_sha256: str | None
    length_group_identity_sha256: Mapping[int, str]
    canonical_model_hashes: Mapping[int, Mapping[int, str]]
    score_calibration_hashes: Mapping[int, str]
    elapsed_seconds: float
    dataset: str | None = None
    database_sha256: str | None = None
    train_contract_identity_sha256: str | None = None
    result_evidence_sha256: str | None = None

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
        for name in ("database_sha256", "train_contract_identity_sha256", "result_evidence_sha256"):
            value = getattr(self, name)
            if value is not None:
                _sha256(value, name)
        if self.dataset is not None and (not isinstance(self.dataset, str) or not self.dataset):
            raise ValueError("checkpoint dataset must be a non-empty string")


@dataclass(frozen=True)
class TrainModelEvidence:
    """Per-length OOF and final-model evidence actually available to scoring."""

    processed: tuple[ProcessedGroup, ...]
    oof_bundles: Mapping[int, OOFModelBundle]
    final_models: Mapping[int, Sequence[TrainedSeedModel]]
    backend: BackendAudit
    train_identity_sha256: str = field(init=False)
    processed_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        processed = _processed_model_groups(self.processed)
        expected_indices = {
            group.group.length: tuple(series.instance_index for series in group.group.series)
            for group in processed
        }
        bundles = _oof_bundles(self.oof_bundles)
        if tuple(bundles) != tuple(expected_indices):
            raise ValueError("OOF evidence must cover exact processed TRAIN lengths")
        models = _final_models(self.final_models, tuple(bundles))
        for length, bundle in bundles.items():
            if bundle.universe_indices != expected_indices[length]:
                raise ValueError("OOF evidence identities do not match processed TRAIN")
        for length, by_seed in models.items():
            if any(
                model.fit_indices != expected_indices[length]
                or model.refit_indices != expected_indices[length]
                or model.validation_indices
                for model in by_seed
            ):
                raise ValueError("final refit identities do not match processed TRAIN")
        if not isinstance(self.backend, BackendAudit):
            raise ValueError("model evidence requires a backend audit")
        identity = _model_evidence_digest(processed, bundles, models, self.backend)
        processed_identity = _processed_identity_digest(processed)
        object.__setattr__(self, "processed", processed)
        object.__setattr__(self, "oof_bundles", MappingProxyType(bundles))
        object.__setattr__(self, "final_models", MappingProxyType(models))
        object.__setattr__(self, "train_identity_sha256", identity)
        object.__setattr__(self, "processed_identity_sha256", processed_identity)


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


def run_dataset(
    dataset: str, context: FrozenRunContext, *, runtime: RunnerRuntime | None = None
) -> DatasetResult:
    """Run one frozen dataset through the sequential label-free state machine."""
    return _run_dataset(dataset, context, None, runtime or RunnerRuntime.production())


def _run_dataset(
    dataset: str,
    context: FrozenRunContext,
    checkpoints: Exp417CheckpointStore | None,
    runtime: RunnerRuntime,
) -> DatasetResult:
    _validate_dataset(dataset, context)
    if not isinstance(runtime, RunnerRuntime):
        raise ValueError("runtime must be RunnerRuntime")
    started = perf_counter()
    try:
        runtime.verify(context)
        train = runtime.load_train(dataset, context)
    except Exception as error:
        return _terminal(dataset, context, "insufficient_train", error, checkpoints, started)
    if not isinstance(train, RawTrainDataset) or train.dataset != dataset:
        return _terminal(dataset, context, "insufficient_train", ValueError("invalid TRAIN values"), checkpoints, started)
    try:
        processed = tuple(runtime.preprocess(train))
    except Exception as error:
        return _terminal(dataset, context, "preprocessing", error, checkpoints, started)
    reason = _processed_terminal_reason(processed)
    if reason is not None:
        return _terminal(dataset, context, "preprocessing", ValueError(reason), checkpoints, started)
    try:
        oof = runtime.train_models(processed)
    except Exception as error:
        return _terminal(dataset, context, "oof", error, checkpoints, started)
    if not isinstance(oof, TrainModelEvidence):
        return _terminal(dataset, context, "oof", ValueError("invalid OOF model evidence"), checkpoints, started)
    if oof.backend.fallback:
        return _terminal(dataset, context, "backend", ValueError("backend audit requires fallback"), checkpoints, started)
    try:
        qualification = runtime.qualify(dataset, processed, oof)
    except Exception as error:
        return _terminal(dataset, context, "qualification", error, checkpoints, started)
    if not isinstance(qualification, TrainQualification):
        return _terminal(dataset, context, "qualification", ValueError("invalid TRAIN qualification"), checkpoints, started)
    if qualification.backend != oof.backend or qualification.backend.fallback:
        return _terminal(dataset, context, "backend", ValueError("qualification backend audit failed"), checkpoints, started)
    if any(group.status != "qualified" for group in qualification.groups):
        return _terminal(dataset, context, "qualification", ValueError("TRAIN qualification did not qualify every length"), checkpoints, started)
    try:
        contract = runtime.freeze(dataset, qualification, oof)
    except Exception as error:
        return _terminal(dataset, context, "qualification", error, checkpoints, started)
    if not isinstance(contract, FrozenTrainContract) or contract.dataset != dataset:
        return _terminal(dataset, context, "qualification", ValueError("invalid frozen TRAIN contract"), checkpoints, started)
    try:
        test = runtime.load_test(dataset, contract, context)
        scoring = runtime.score(test, contract, oof)
    except Exception as error:
        return _terminal(dataset, context, "backend", error, checkpoints, started, contract)
    if not isinstance(scoring, TestScoringEvidence):
        return _terminal(dataset, context, "backend", ValueError("invalid TEST scoring evidence"), checkpoints, started, contract)
    try:
        decisions = scoring.decisions
        evidence = DatasetDecisionEvidence(
            dataset=dataset,
            context=context,
            frozen_train_contract=contract,
            database_sha256=context.database_sha256,
            train_identity_sha256=contract.train_identity_sha256,
            decisions=decisions,
        )
        excluded = tuple(runtime.exclude(evidence))
        if excluded != evidence.derived_candidates:
            raise ValueError("frozen-lane exclusion must retain every and only eligible coordinate")
        result = runtime.workload(evidence)
        _validate_result(result, dataset, context, evidence)
    except Exception as error:
        return _terminal(dataset, context, "backend", error, checkpoints, started, contract)
    material = None if checkpoints is None else build_checkpoint_evidence(
        train, processed, oof, contract, scoring, result, perf_counter() - started
    )
    checkpoint(result, checkpoints, material)
    return result


def resume_or_run_population(
    context: FrozenRunContext, checkpoints: Exp417CheckpointStore, *, runtime: RunnerRuntime | None = None
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
    active_runtime = runtime or RunnerRuntime.production()
    if not isinstance(active_runtime, RunnerRuntime):
        raise ValueError("runtime must be RunnerRuntime")
    completed = checkpoints.load_completed()
    if set(completed) - set(context.datasets):
        raise ValueError("checkpoint contains a dataset outside the frozen context")

    results: list[DatasetResult] = []
    for dataset in context.datasets:
        result = completed.get(dataset)
        if result is None:
            result = _run_dataset(dataset, context, checkpoints, active_runtime)
        _validate_result(result, dataset, context)
        results.append(result)

    frozen_results = tuple(results)
    total_test = sum(context.test_sizes.values())
    if any(
        isinstance(result, DatasetResult)
        and isinstance(result.decision_evidence, DatasetTerminalEvidence)
        for result in frozen_results
    ):
        return RunSummary(frozen_results, total_test, False, "failed", context)
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


def _production_verify(context: FrozenRunContext, resources: RunnerResources) -> None:
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be a FrozenRunContext")
    assert_label_free_sql_contract()
    verify_input_snapshots(resources.database, resources.exp291_prediction,
                           resources.exp291_manifest, resources.exp292_prediction,
                           resources.exp292_manifest)


def _production_train_models(
    processed: Sequence[ProcessedGroup], backend_audit: BackendAudit
) -> TrainModelEvidence:
    groups = _processed_model_groups(processed)
    if not isinstance(backend_audit, BackendAudit) or backend_audit.fallback:
        raise ValueError("model training requires the selected passing Task 7 backend audit")
    import torch
    device = torch.device(backend_audit.backend)
    bundles = {group.group.length: train_oof_models(group, device=device) for group in groups}
    finals = {group.group.length: fit_final_seed_models(group, device=device) for group in groups}
    return TrainModelEvidence(
        groups, bundles, finals,
        backend_audit,
    )


def _residual_decisions(
    bundle: OOFModelBundle, residuals_by_seed: Mapping[int, Mapping[int, np.ndarray]]
) -> tuple[Any, ...]:
    decisions = []
    for seed in FROZEN_CONFIG.model_seeds:
        scores = {
            horizon: score_horizon(
                [bundle.oof_residuals[seed][index][horizon] for index in bundle.universe_indices],
                residuals_by_seed[seed][horizon], horizon=horizon,
            )
            for horizon in bundle.horizons
        }
        decisions.append(seed_decision(scores, seed=seed))
    return tuple(decisions)


def _production_qualify_train(
    dataset: str, processed: Sequence[ProcessedGroup], models: TrainModelEvidence
) -> TrainQualification:
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("processed TRAIN groups are invalid")
    groups = _processed_model_groups(processed)
    qualifications: list[GroupQualification] = []
    for group in groups:
        length = group.group.length
        bundle = models.oof_bundles[length]
        normal = tuple(
            _residual_decisions(
                bundle,
                {seed: bundle.oof_residuals[seed][index] for seed in FROZEN_CONFIG.model_seeds},
            )
            for index in bundle.universe_indices
        )
        synthetic: dict[str, list[tuple[Any, ...]]] = {
            name: [] for name in ("level_shift", "gradual_drift", "amplitude_scaling", "period_change", "secondary_spike")
        }
        for series in group.group.series:
            base = np.asarray(series.values, dtype=np.float64)
            for name, perturbation in build_fixed_perturbations(base, group.raw_scale, seed=FROZEN_CONFIG.perturbation_seeds[0]).items():
                channels = _channels_from_values(perturbation, group)
                # Qualification is deliberately based on a frozen final fit; all
                # scoring calibration remains normal OOF TRAIN residual evidence.
                by_seed = {
                    seed: _prediction_residuals(models.final_models[length], seed, channels)
                    for seed in FROZEN_CONFIG.model_seeds
                }
                decisions = tuple(
                    seed_decision({horizon: score_horizon(
                        [bundle.oof_residuals[seed][index][horizon] for index in bundle.universe_indices],
                        by_seed[seed][horizon], horizon=horizon,
                    ) for horizon in bundle.horizons}, seed=seed)
                    for seed in FROZEN_CONFIG.model_seeds
                )
                synthetic[name].append(decisions)
        qualifications.append(qualify_group(
            length=length, normal_oof_decisions=normal, synthetic_decisions=synthetic,
            finite=True, backend_fallback=models.backend.fallback,
        ))
    return TrainQualification(tuple(qualifications), models.backend)


def _production_score_test(
    test: Any, contract: FrozenTrainContract, models: TrainModelEvidence
) -> TestScoringEvidence:
    from .contracts import RawTestDataset
    if not isinstance(test, RawTestDataset) or test.dataset != contract.dataset:
        raise ValueError("TEST values do not match frozen TRAIN contract")
    by_length = {group.group.length: group for group in models.processed}
    decisions: list[TestInstanceDecision] = []
    for series in test.series:
        group = by_length.get(len(series.values))
        if group is None:
            raise ValueError("TEST length is absent from processed TRAIN")
        length = group.group.length
        bundle = models.oof_bundles[length]
        channels = _channels_from_values(np.asarray(series.values, dtype=np.float64), group)
        seeds = tuple(
            seed_decision({horizon: score_horizon(
                [bundle.oof_residuals[seed][index][horizon] for index in bundle.universe_indices],
                _prediction_residuals(models.final_models[length], seed, channels)[horizon], horizon=horizon,
            ) for horizon in bundle.horizons}, seed=seed)
            for seed in FROZEN_CONFIG.model_seeds
        )
        decisions.append(TestInstanceDecision(test.dataset, series.instance_index, length, seeds))
    hashes = {length: {model.seed: model.model_sha256 for model in fitted}
              for length, fitted in models.final_models.items()}
    return TestScoringEvidence(tuple(decisions), hashes)


def build_checkpoint_evidence(
    train: RawTrainDataset,
    processed: Sequence[ProcessedGroup],
    oof: TrainModelEvidence,
    contract: FrozenTrainContract,
    scoring: TestScoringEvidence,
    result: DatasetResult,
    elapsed_seconds: float,
) -> CheckpointEvidence:
    """Derive every Task 8 record field from typed immutable stage evidence."""
    if not isinstance(train, RawTrainDataset):
        raise ValueError("checkpoint evidence requires a RawTrainDataset")
    groups = _processed_groups(processed, train)
    if not isinstance(oof, TrainModelEvidence):
        raise ValueError("checkpoint evidence requires TrainModelEvidence")
    if oof.processed_identity_sha256 != _processed_identity_digest(groups):
        raise ValueError("checkpoint processed groups do not match model evidence values")
    if not isinstance(contract, FrozenTrainContract) or contract.dataset != train.dataset:
        raise ValueError("checkpoint evidence contract does not match TRAIN dataset")
    if not isinstance(scoring, TestScoringEvidence):
        raise ValueError("checkpoint evidence requires TestScoringEvidence")
    if not isinstance(result, DatasetResult) or result.dataset != train.dataset:
        raise ValueError("checkpoint evidence result does not match TRAIN dataset")
    if result.context.database_sha256 != result.decision_evidence.context.database_sha256:
        raise ValueError("checkpoint result context evidence is inconsistent")
    if result.decision_evidence.frozen_train_contract != contract:
        raise ValueError("checkpoint result does not bind the supplied TRAIN contract")
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
        dataset=result.dataset,
        database_sha256=result.context.database_sha256,
        train_contract_identity_sha256=contract.train_identity_sha256,
        result_evidence_sha256=_result_evidence_digest(result),
    )


def checkpoint(
    result: DatasetResult,
    checkpoints: Exp417CheckpointStore | None,
    material: CheckpointEvidence | None = None,
    *,
    elapsed_seconds: float | None = None,
) -> None:
    """Publish a result only through Task 8's transactional store API."""
    if checkpoints is None:
        return
    terminal = isinstance(result.decision_evidence, DatasetTerminalEvidence)
    if terminal and material is None:
        if elapsed_seconds is None:
            raise ValueError("terminal checkpoint requires elapsed time")
        checkpoints.write_dataset(
            result, raw_train_identity_sha256=None, repaired_train_identity_sha256=None,
            length_group_identity_sha256={}, canonical_model_hashes={},
            score_calibration_hashes={}, backend_audit=None, elapsed_seconds=elapsed_seconds,
        )
        return
    if not isinstance(material, CheckpointEvidence):
        raise ValueError("checkpoint material must be complete typed evidence")
    contract = result.decision_evidence.frozen_train_contract
    if (
        material.dataset != result.dataset
        or material.database_sha256 != result.context.database_sha256
        or contract is None
        or material.train_contract_identity_sha256 != contract.train_identity_sha256
        or material.result_evidence_sha256 != _result_evidence_digest(result)
    ):
        raise ValueError("checkpoint material does not bind the exact dataset result evidence")
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
        backend_audit=contract.backend,
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


def _processed_model_groups(values: object) -> tuple[ProcessedGroup, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("model evidence processed groups must be a sequence")
    groups = tuple(values)
    if not groups or any(not isinstance(group, ProcessedGroup) for group in groups):
        raise ValueError("model evidence processed groups are invalid")
    if any(group.status != "candidate_eligible" for group in groups):
        raise ValueError("model fitting requires candidate-eligible processed groups")
    lengths = tuple(group.group.length for group in groups)
    if lengths != tuple(sorted(lengths)) or len(set(lengths)) != len(lengths):
        raise ValueError("model evidence processed groups must have sorted unique lengths")
    return groups


def _processed_terminal_reason(values: object) -> str | None:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return "preprocessing did not return a sequence"
    groups = tuple(values)
    if not groups or any(not isinstance(group, ProcessedGroup) for group in groups):
        return "preprocessing did not return processed groups"
    ineligible = tuple(group for group in groups if group.status != "candidate_eligible")
    if not ineligible:
        return None
    statuses = ",".join(sorted({group.status for group in ineligible}))
    return f"preprocessing produced no candidate-eligible complete population: {statuses}"


def _model_evidence_digest(
    groups: Sequence[ProcessedGroup], bundles: Mapping[int, OOFModelBundle],
    finals: Mapping[int, Sequence[TrainedSeedModel]], backend: BackendAudit,
) -> str:
    digest = sha256()
    _digest_text(digest, "exp417.train-evidence.v2")
    _digest_text(digest, backend.backend)
    _digest_text(digest, _processed_identity_digest(groups))
    for group in groups:
        length = group.group.length
        digest.update(length.to_bytes(8, "little", signed=False))
        for series in group.group.series:
            digest.update(series.instance_index.to_bytes(8, "little", signed=False))
            digest.update(np.asarray(series.values, dtype="<f4").tobytes(order="C"))
        bundle = bundles[length]
        for seed in FROZEN_CONFIG.model_seeds:
            for model in (model for model in bundle.models if model.seed == seed):
                _digest_text(digest, model.model_sha256)
            final = next((model for model in finals[length] if model.seed == seed), None)
            if final is None:
                raise ValueError("final model evidence lacks frozen seed")
            _digest_text(digest, final.model_sha256)
    return digest.hexdigest()


def _processed_identity_digest(groups: Sequence[ProcessedGroup]) -> str:
    digest = sha256()
    _digest_text(digest, "exp417.processed-groups.v2")
    for group in groups:
        digest.update(group.group.length.to_bytes(8, "little", signed=False))
        _digest_text(digest, group.status)
        digest.update(np.asarray(
            [group.raw_center, group.raw_scale, group.difference_center, group.difference_scale],
            dtype="<f8",
        ).tobytes(order="C"))
        for series in group.group.series:
            digest.update(series.instance_index.to_bytes(8, "little", signed=False))
            digest.update(np.asarray(series.values, dtype="<f4").tobytes(order="C"))
            channels = group.channels.get(series.instance_index)
            if not isinstance(channels, np.ndarray):
                raise ValueError("processed identity requires exact model-input channels")
            digest.update(np.asarray(channels, dtype="<f4", order="C").tobytes(order="C"))
    return digest.hexdigest()


def _result_evidence_digest(result: DatasetResult) -> str:
    if not isinstance(result, DatasetResult):
        raise ValueError("result evidence digest requires DatasetResult")
    evidence = result.decision_evidence
    digest = sha256()
    _digest_text(digest, "exp417.result-evidence.v1")
    _digest_text(digest, result.dataset)
    _digest_text(digest, result.context.database_sha256)
    _digest_text(digest, result.status)
    _digest_text(digest, type(evidence).__name__)
    if isinstance(evidence, DatasetDecisionEvidence):
        _digest_text(digest, evidence.train_identity_sha256)
        for decision in evidence.decisions:
            digest.update(decision.instance_index.to_bytes(8, "little", signed=False))
            digest.update(decision.length.to_bytes(8, "little", signed=False))
            for seed in decision.seed_decisions:
                digest.update(seed.seed.to_bytes(8, "little", signed=False))
                for score in seed.scores.values():
                    _digest_text(digest, repr((score.horizon, score.score, score.p_value, score.persistence)))
    else:
        _digest_text(digest, evidence.stage)
        _digest_text(digest, evidence.reason)
    return digest.hexdigest()


def _channels_from_values(values: np.ndarray, group: ProcessedGroup) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or raw.size != group.group.length or not np.isfinite(raw).all():
        raise ValueError("values do not match a finite processed length group")
    differences = np.diff(raw, prepend=raw[:1])
    channels = np.stack(((raw - group.raw_center) / group.raw_scale,
                         (differences - group.difference_center) / group.difference_scale)).astype(np.float32)
    if not np.isfinite(channels).all():
        raise ValueError("processed TEST channels are non-finite")
    channels.setflags(write=False)
    return channels


def _prediction_residuals(
    models: Sequence[TrainedSeedModel], seed: int, channels: np.ndarray
) -> Mapping[int, np.ndarray]:
    model = next((item for item in models if item.seed == seed), None)
    if model is None:
        raise ValueError("final model evidence lacks frozen seed")
    return predict_residuals(model, channels)


def _terminal(
    dataset: str, context: FrozenRunContext, stage: str, error: Exception,
    checkpoints: Exp417CheckpointStore | None, started: float,
    contract: FrozenTrainContract | None = None,
) -> DatasetResult:
    evidence = DatasetTerminalEvidence(dataset, context, stage, str(error) or type(error).__name__, contract)
    result = DatasetResult(dataset, context.test_sizes[dataset], 0 if contract is None else len(contract.groups),
                           () if contract is None else contract.groups, (), evidence,
                           context.lanes, context, "failed")
    checkpoint(result, checkpoints, None, elapsed_seconds=perf_counter() - started)
    return result


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
