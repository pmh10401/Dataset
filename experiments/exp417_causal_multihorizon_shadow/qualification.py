"""Frozen TRAIN-only synthetic qualification and fail-closed workload gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from math import isfinite, sqrt
from numbers import Real
from statistics import NormalDist
from types import MappingProxyType

import numpy as np

from .config import Exp417FrozenConfig, FROZEN_CONFIG
from .contracts import (
    BackendAudit,
    DatasetDecisionEvidence,
    DatasetResult,
    FrozenLaneSnapshot,
    FrozenRunContext,
    FrozenTrainContract,
    GroupQualification,
    RunSummary,
    SeedDecision,
    ShadowCandidate,
)


_PRIMARY_TYPES = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_PERTURBATION_TYPES = _PRIMARY_TYPES + ("secondary_spike",)


def build_fixed_perturbations(
    series: np.ndarray, robust_scale: float, *, seed: int
) -> Mapping[str, np.ndarray]:
    """Build deterministic TRAIN-only perturbations without changing its length."""
    values = _finite_vector(series, "series")
    scale = _positive_finite(robust_scale, "robust_scale")
    if type(seed) is not int or seed not in FROZEN_CONFIG.perturbation_seeds:
        raise ValueError("seed must be one of the exact frozen perturbation seeds")
    if values.size < 2:
        raise ValueError("series must have at least two points for fixed perturbations")

    onset = int(np.floor(values.size * 0.40))
    if onset >= values.size:
        raise ValueError("series leaves no valid perturbation region")
    duration = values.size - onset
    level = np.array(values, copy=True)
    level[onset:] += 3.0 * scale

    drift = np.array(values, copy=True)
    drift[onset:] += np.linspace(0.0, 3.0 * scale, duration, dtype=np.float64)

    amplitude = values * 1.5

    # The 1.25 period change is confined to the post-onset valid region.  It
    # reads a deterministic, linearly interpolated 1.25x source trajectory and
    # writes exactly the original number of samples.
    warped = np.array(values, copy=True)
    region = values[onset:]
    source = np.arange(duration, dtype=np.float64)
    warped[onset:] = np.interp(source / 1.25, source, region)

    spike = np.array(values, copy=True)
    spike_index = int(np.random.default_rng(seed).integers(onset, values.size))
    spike[spike_index] += 5.0 * scale

    return MappingProxyType({
        "level_shift": _readonly(level),
        "gradual_drift": _readonly(drift),
        "amplitude_scaling": _readonly(amplitude),
        "period_change": _readonly(warped),
        "secondary_spike": _readonly(spike),
    })


def wilson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    """Return the normal-approximation Wilson upper confidence bound."""
    if type(successes) is not int or type(trials) is not int or trials < 1:
        raise ValueError("successes and trials must be integers with trials >= 1")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    confidence_value = _finite_unit_interval(confidence, "confidence", exclusive=True)
    z = NormalDist().inv_cdf(0.5 + (confidence_value / 2.0))
    rate = successes / trials
    z_squared = z * z
    center = (rate + (z_squared / (2.0 * trials))) / (1.0 + (z_squared / trials))
    radius = z * sqrt(
        (rate * (1.0 - rate) / trials) + (z_squared / (4.0 * trials * trials))
    ) / (1.0 + (z_squared / trials))
    return float(center + radius)


def qualify_group(
    *,
    length: int,
    normal_oof_decisions: Sequence[Sequence[SeedDecision]],
    synthetic_decisions: Mapping[str, Sequence[Sequence[SeedDecision]]],
    finite: bool = True,
    backend_fallback: bool = False,
    config: Exp417FrozenConfig = FROZEN_CONFIG,
) -> GroupQualification:
    """Qualify one length group only from normal OOF and TRAIN synthetic evidence."""
    _validate_config(config)
    if type(length) is not int or length < 1:
        raise ValueError("length must be a positive integer")
    if type(finite) is not bool or type(backend_fallback) is not bool:
        raise ValueError("finite and backend_fallback must be bool")
    normal = _decision_bundles(normal_oof_decisions, "normal_oof_decisions")
    synthetic = _synthetic_bundles(synthetic_decisions)
    agreement = all(_bundle_agrees(bundle) for bundle in normal) and all(
        _bundle_agrees(bundle) for bundles in synthetic.values() for bundle in bundles
    )
    normal_candidates = sum(_bundle_candidate(bundle) for bundle in normal)
    oof_rate = normal_candidates / len(normal)
    upper = wilson_upper(normal_candidates, len(normal))
    sensitivities = {
        name: sum(_bundle_candidate(bundle) for bundle in bundles) / len(bundles)
        for name, bundles in synthetic.items()
    }
    pooled = sum(
        _bundle_candidate(bundle) for bundles in synthetic.values() for bundle in bundles
    ) / sum(len(bundles) for bundles in synthetic.values())
    qualified = (
        len(normal) >= config.min_train_candidate
        and oof_rate <= config.oof_candidate_rate
        and upper <= config.oof_wilson_upper
        and pooled >= config.pooled_synthetic_sensitivity
        and all(sensitivities[name] >= config.per_type_synthetic_sensitivity for name in _PRIMARY_TYPES)
        and agreement
        and finite
        and not backend_fallback
    )
    return GroupQualification(
        length=length,
        series_count=len(normal),
        status="qualified" if qualified else "failed",
        oof_candidate_rate=oof_rate,
        wilson_upper=upper,
        pooled_synthetic_sensitivity=pooled,
        per_type_synthetic_sensitivity={name: sensitivities[name] for name in _PRIMARY_TYPES},
        seed_agreement=agreement,
        finite=finite,
        backend_fallback=backend_fallback,
    )


def freeze_train_contract(
    dataset: str,
    groups: Sequence[GroupQualification],
    backend: BackendAudit,
    *,
    train_identity_sha256: str | None = None,
    config: Exp417FrozenConfig = FROZEN_CONFIG,
) -> FrozenTrainContract:
    """Freeze deterministic TRAIN qualification evidence with no near-repeat claim."""
    _validate_config(config)
    frozen_groups = _groups(groups)
    if not isinstance(backend, BackendAudit):
        raise ValueError("backend must be a BackendAudit")
    derived_identity = sha256(
        json.dumps(
            {
                "dataset": dataset,
                "groups": [
                    {
                        "length": group.length,
                        "series_count": group.series_count,
                        "status": group.status,
                        "oof_candidate_rate": group.oof_candidate_rate,
                        "wilson_upper": group.wilson_upper,
                        "pooled_synthetic_sensitivity": group.pooled_synthetic_sensitivity,
                        "per_type_synthetic_sensitivity": dict(group.per_type_synthetic_sensitivity),
                        "seed_agreement": group.seed_agreement,
                        "finite": group.finite,
                        "backend_fallback": group.backend_fallback,
                    }
                    for group in frozen_groups
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if train_identity_sha256 is None:
        identity = derived_identity
    elif isinstance(train_identity_sha256, str) and len(train_identity_sha256) == 64 and all(
        character in "0123456789abcdef" for character in train_identity_sha256
    ):
        identity = train_identity_sha256
    else:
        raise ValueError("train_identity_sha256 must be a lowercase SHA-256 digest")
    return FrozenTrainContract(
        dataset=dataset,
        groups=frozen_groups,
        backend=backend,
        train_identity_sha256=identity,
        near_repeat_status="not_available",
    )


def exclude_frozen_lanes(evidence: DatasetDecisionEvidence) -> tuple[ShadowCandidate, ...]:
    """Derive every eligible candidate from complete evidence, excluding frozen lanes."""
    if not isinstance(evidence, DatasetDecisionEvidence):
        raise ValueError("evidence must be DatasetDecisionEvidence")
    return evidence.derived_candidates


def apply_dataset_workload(evidence: DatasetDecisionEvidence) -> DatasetResult:
    """Apply the frozen dataset cap as an all-or-none workload gate."""
    if not isinstance(evidence, DatasetDecisionEvidence):
        raise ValueError("evidence must be DatasetDecisionEvidence")
    retained = exclude_frozen_lanes(evidence)
    prospective_rate = len(retained) / evidence.test_size
    if prospective_rate > FROZEN_CONFIG.dataset_workload_rate:
        return DatasetResult(
            dataset=evidence.dataset,
            test_size=evidence.test_size,
            group_count=len(evidence.frozen_train_contract.groups),
            groups=evidence.frozen_train_contract.groups,
            candidates=(),
            decision_evidence=evidence,
            lanes=evidence.context.lanes,
            context=evidence.context,
            status="abstained",
        )
    return DatasetResult(
        dataset=evidence.dataset,
        test_size=evidence.test_size,
        group_count=len(evidence.frozen_train_contract.groups),
        groups=evidence.frozen_train_contract.groups,
        candidates=retained,
        decision_evidence=evidence,
        lanes=evidence.context.lanes,
        context=evidence.context,
        status="qualified",
    )


def apply_global_workload(
    results: Sequence[DatasetResult], total_test: int, *, context: FrozenRunContext
) -> RunSummary:
    """Block publication and clear every official payload above the global cap."""
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be a FrozenRunContext")
    frozen_results = _results(results)
    if type(total_test) is not int or total_test < 0:
        raise ValueError("total_test must be a non-negative integer")
    if total_test != sum(result.test_size for result in frozen_results):
        raise ValueError("total_test does not conserve dataset totals")
    if total_test != sum(context.test_sizes.values()):
        raise ValueError("total_test must match the frozen TEST population")
    prospective_rate = sum(len(result.observed_candidates) for result in frozen_results) / total_test if total_test else 0.0
    if prospective_rate > FROZEN_CONFIG.global_workload_rate:
        cleared = tuple(
            DatasetResult(
                dataset=result.dataset,
                test_size=result.test_size,
                group_count=result.group_count,
                groups=result.groups,
                candidates=(),
                decision_evidence=result.decision_evidence,
                lanes=result.lanes,
                context=result.context,
                status="abstained",
            )
            for result in frozen_results
        )
        return RunSummary(
            results=cleared,
            total_test=total_test,
            publication_allowed=False,
            status="global_workload_gate_failed",
            context=context,
        )
    return RunSummary(
        results=frozen_results,
        total_test=total_test,
        publication_allowed=True,
        status="complete",
        context=context,
    )


def _finite_vector(values: object, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional ndarray")
    if values.dtype == np.dtype(bool) or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    copied = np.asarray(values, dtype=np.float64)
    if not np.isfinite(copied).all():
        raise ValueError(f"{name} must be finite")
    return copied


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return float(value)


def _finite_unit_interval(value: object, name: str, *, exclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if (exclusive and not 0.0 < number < 1.0) or (not exclusive and not 0.0 <= number <= 1.0):
        raise ValueError(f"{name} must be in the unit interval")
    return number


def _readonly(values: np.ndarray) -> np.ndarray:
    copy = np.ascontiguousarray(values, dtype=np.float64)
    copy.setflags(write=False)
    return copy


def _decision_bundles(values: Sequence[Sequence[SeedDecision]], name: str) -> tuple[tuple[SeedDecision, ...], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence")
    bundles = tuple(_one_bundle(value, name) for value in values)
    if not bundles:
        raise ValueError(f"{name} must be non-empty")
    return bundles


def _one_bundle(value: Sequence[SeedDecision], name: str) -> tuple[SeedDecision, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must contain seed-decision sequences")
    bundle = tuple(value)
    if len(bundle) != len(FROZEN_CONFIG.model_seeds) or any(not isinstance(item, SeedDecision) for item in bundle):
        raise ValueError(f"{name} must contain exactly the frozen SeedDecision records")
    if {item.seed for item in bundle} != set(FROZEN_CONFIG.model_seeds) or len({item.seed for item in bundle}) != len(bundle):
        raise ValueError(f"{name} must contain each frozen model seed once")
    return bundle


def _synthetic_bundles(values: Mapping[str, Sequence[Sequence[SeedDecision]]]) -> Mapping[str, tuple[tuple[SeedDecision, ...], ...]]:
    if not isinstance(values, Mapping) or set(values) != set(_PERTURBATION_TYPES):
        raise ValueError("synthetic_decisions must cover every fixed perturbation")
    return {name: _decision_bundles(values[name], f"synthetic_decisions[{name}]") for name in _PERTURBATION_TYPES}


def _bundle_agrees(bundle: tuple[SeedDecision, ...]) -> bool:
    return len({decision.candidate for decision in bundle}) == 1


def _bundle_candidate(bundle: tuple[SeedDecision, ...]) -> bool:
    return _bundle_agrees(bundle) and all(decision.candidate for decision in bundle)


def _groups(values: Sequence[GroupQualification]) -> tuple[GroupQualification, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("groups must be a sequence")
    groups = tuple(values)
    if not groups or any(not isinstance(group, GroupQualification) for group in groups):
        raise ValueError("groups must contain GroupQualification records")
    return groups


def _candidates(values: Sequence[ShadowCandidate]) -> tuple[ShadowCandidate, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("candidates must be a sequence")
    candidates = tuple(values)
    if any(not isinstance(candidate, ShadowCandidate) for candidate in candidates):
        raise ValueError("candidates must contain ShadowCandidate records")
    return candidates


def _results(values: Sequence[DatasetResult]) -> tuple[DatasetResult, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("results must be a sequence")
    results = tuple(values)
    if any(not isinstance(result, DatasetResult) for result in results):
        raise ValueError("results must contain DatasetResult records")
    return results


def _validate_config(config: Exp417FrozenConfig) -> None:
    if not isinstance(config, Exp417FrozenConfig):
        raise ValueError("config must be an Exp417 frozen config")
    config.validate()


__all__ = [
    "apply_dataset_workload",
    "apply_global_workload",
    "build_fixed_perturbations",
    "exclude_frozen_lanes",
    "freeze_train_contract",
    "qualify_group",
    "wilson_upper",
]
