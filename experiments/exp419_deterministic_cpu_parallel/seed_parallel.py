"""Deterministic CPU seed concurrency for Exp419 TRAIN model evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType

import torch

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    ProcessedGroup,
)
from experiments.exp417_causal_multihorizon_shadow.model import (
    OOFModelBundle,
    SeedOOFTraining,
    TrainedSeedModel,
    fit_seed_final_from_state,
    fit_seed_oof_from_state,
    prepare_seed_initial_states,
)
from experiments.exp417_causal_multihorizon_shadow.preprocessing import build_horizons
from experiments.exp417_causal_multihorizon_shadow.runner import (
    TrainModelEvidence,
    _processed_evidence_groups,
)


StageCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class _SeedGroupTraining:
    """All evidence produced by one independent seed for one length group."""

    oof: SeedOOFTraining
    final: TrainedSeedModel | None


def train_models_seed_parallel(
    processed: Sequence[ProcessedGroup],
    backend: BackendAudit,
    seed_workers: int,
    on_stage: StageCallback | None = None,
) -> TrainModelEvidence:
    """Build canonical Exp417 evidence with bounded independent seed tasks.

    This function deliberately supports CPU only.  Its caller has already
    pinned PyTorch and BLAS to one thread per model fit; ``seed_workers`` only
    controls the number of independent frozen seeds that run at once.
    """
    groups = _processed_evidence_groups(processed)
    _validate_runtime(backend, seed_workers, on_stage)
    device = torch.device("cpu")
    bundles: dict[int, OOFModelBundle] = {}
    finals: dict[int, tuple[TrainedSeedModel, ...]] = {}

    for group in groups:
        if group.status not in {"candidate_eligible", "diagnostic_only"}:
            continue
        by_seed = _fit_group_seeds(group, device, seed_workers, on_stage)
        bundles[group.group.length] = _reassemble_oof(group, by_seed)
        if group.status == "candidate_eligible":
            finals[group.group.length] = tuple(
                _required_final(by_seed[seed])
                for seed in FROZEN_CONFIG.model_seeds
            )

    return TrainModelEvidence(groups, bundles, finals, backend)


def _validate_runtime(
    backend: object, seed_workers: object, on_stage: object,
) -> None:
    if not isinstance(backend, BackendAudit) or backend.fallback:
        raise ValueError("seed-parallel training requires a passing backend audit")
    if backend.backend != "cpu":
        raise ValueError("seed-parallel training requires the frozen CPU backend")
    if (
        type(seed_workers) is not int
        or not 1 <= seed_workers <= len(FROZEN_CONFIG.model_seeds)
    ):
        raise ValueError("seed_workers must be between one and the frozen seed count")
    if on_stage is not None and not callable(on_stage):
        raise ValueError("on_stage must be callable")


def _fit_group_seeds(
    group: ProcessedGroup,
    device: torch.device,
    seed_workers: int,
    on_stage: StageCallback | None,
) -> Mapping[int, _SeedGroupTraining]:
    initial_states = prepare_seed_initial_states(
        build_horizons(group.group.length),
        device,
    )
    with ThreadPoolExecutor(max_workers=seed_workers, thread_name_prefix="exp419-seed") as pool:
        futures: dict[int, Future[_SeedGroupTraining]] = {
            seed: pool.submit(
                _fit_one_seed,
                group,
                seed,
                initial_states[seed],
                device,
                on_stage,
            )
            for seed in FROZEN_CONFIG.model_seeds
        }
        try:
            return MappingProxyType({
                seed: futures[seed].result()
                for seed in FROZEN_CONFIG.model_seeds
            })
        except BaseException:
            for future in futures.values():
                future.cancel()
            raise


def _fit_one_seed(
    group: ProcessedGroup,
    seed: int,
    initial_state: Mapping[str, tuple[str, tuple[int, ...], bytes]],
    device: torch.device,
    on_stage: StageCallback | None,
) -> _SeedGroupTraining:
    _notify(on_stage, "oof", group.group.length, seed)
    oof = fit_seed_oof_from_state(
        group,
        seed=seed,
        initial_state=initial_state,
        device=device,
    )
    final: TrainedSeedModel | None = None
    if group.status == "candidate_eligible":
        _notify(on_stage, "final_fit", group.group.length, seed)
        final = fit_seed_final_from_state(
            group,
            seed=seed,
            initial_state=initial_state,
            device=device,
        )
    return _SeedGroupTraining(oof, final)


def _notify(
    callback: StageCallback | None, phase: str, length: int, seed: int
) -> None:
    if callback is not None:
        callback(phase, length, seed)


def _reassemble_oof(
    group: ProcessedGroup,
    by_seed: Mapping[int, _SeedGroupTraining],
) -> OOFModelBundle:
    first = by_seed[FROZEN_CONFIG.model_seeds[0]].oof
    models = tuple(
        by_seed[seed].oof.models[fold_index]
        for fold_index in range(len(first.folds))
        for seed in FROZEN_CONFIG.model_seeds
    )
    residuals = {
        seed: by_seed[seed].oof.oof_residuals
        for seed in FROZEN_CONFIG.model_seeds
    }
    return OOFModelBundle(
        length=group.group.length,
        universe_indices=tuple(item.instance_index for item in group.group.series),
        horizons=first.horizons,
        folds=first.folds,
        models=models,
        oof_residuals=residuals,
    )


def _required_final(training: _SeedGroupTraining) -> TrainedSeedModel:
    if training.final is None:
        raise ValueError("candidate-eligible seed training is missing its final model")
    return training.final


__all__ = ["StageCallback", "train_models_seed_parallel"]
