"""TRAIN-only normal_2 rank qualification for Exp458 Task 3."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
    crossfit_plan_antigravity,
)
from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.config_antigravity import (
    FROZEN_CONFIG_ANTIGRAVITY as EXP457_CONFIG_ANTIGRAVITY,
)
from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
    _score_fold_antigravity as _exp457_score_fold_antigravity,
    empirical_pvalues_antigravity,
    wilson_upper_antigravity,
)
from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.config_antigravity import (
    FROZEN_CONFIG_ANTIGRAVITY,
)
from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
    RankQualificationAntigravity,
    RankSeedEvidenceAntigravity,
)
from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
    mann_whitney_auc_antigravity,
)
from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.sources_antigravity import (
    load_normal2_train_antigravity,
)


_EXPECTED_SEEDS_ANTIGRAVITY = (20260721, 20261721)
_EXPECTED_PRIMARY_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


def _vector_antigravity(values: object, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional ndarray")
    if values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must have a real numeric dtype")
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class FoldScoresAntigravity:
    original: np.ndarray
    primary: Mapping[str, np.ndarray]
    feature_dimension: int

    def __post_init__(self) -> None:
        original = _vector_antigravity(self.original, "original fold scores")
        if tuple(self.primary) != _EXPECTED_PRIMARY_TYPES_ANTIGRAVITY:
            raise ValueError("primary fold scores must use the exact ordered primary types")
        frozen: dict[str, np.ndarray] = {}
        for name in _EXPECTED_PRIMARY_TYPES_ANTIGRAVITY:
            values = _vector_antigravity(self.primary[name], f"{name} fold scores")
            if len(values) != len(original):
                raise ValueError("fold score lengths must match")
            frozen[name] = np.array(values, dtype=float, copy=True)
        if type(self.feature_dimension) is not int or self.feature_dimension < 1:
            raise ValueError("feature_dimension must be an exact positive integer")
        object.__setattr__(self, "original", np.array(original, dtype=float, copy=True))
        object.__setattr__(self, "primary", MappingProxyType(frozen))


def _score_fold_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
) -> FoldScoresAntigravity:
    fold = _exp457_score_fold_antigravity(
        train,
        fit_indices,
        query_indices,
        seed=seed,
        fold_index=fold_index,
    )
    return FoldScoresAntigravity(
        np.asarray(fold.original, dtype=float),
        {name: np.asarray(fold.primary[name], dtype=float) for name in _EXPECTED_PRIMARY_TYPES_ANTIGRAVITY},
        fold.feature_dimension,
    )


def _train_antigravity(train: object) -> np.ndarray:
    if not isinstance(train, np.ndarray) or train.ndim != 2:
        raise ValueError("train must be a two-dimensional ndarray")
    if train.dtype != np.dtype(np.float32):
        raise ValueError("train must be float32")
    if len(train) < FROZEN_CONFIG_ANTIGRAVITY.folds:
        raise ValueError("train must have at least one row per fold")
    if not np.isfinite(train).all():
        raise ValueError("train must be finite")
    return train


def _seed_evidence_antigravity(
    seed: int,
    original: np.ndarray,
    primary: Mapping[str, np.ndarray],
) -> RankSeedEvidenceAntigravity:
    normal_alerts = int(
        np.count_nonzero(
            empirical_pvalues_antigravity(original, original) <= EXP457_CONFIG_ANTIGRAVITY.alpha
        )
    )
    per_type_auc = {
        name: float(mann_whitney_auc_antigravity(original, primary[name]))
        for name in _EXPECTED_PRIMARY_TYPES_ANTIGRAVITY
    }
    pooled_auc = float(sum(per_type_auc.values()) / len(_EXPECTED_PRIMARY_TYPES_ANTIGRAVITY))
    normal_wilson_upper = wilson_upper_antigravity(normal_alerts, len(original))
    passed = bool(
        normal_wilson_upper <= FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max
        and pooled_auc >= FROZEN_CONFIG_ANTIGRAVITY.pooled_auc_min
        and all(
            per_type_auc[name] >= FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min
            for name in _EXPECTED_PRIMARY_TYPES_ANTIGRAVITY
        )
    )
    return RankSeedEvidenceAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        seed=seed,
        normal_trials=len(original),
        normal_alerts=normal_alerts,
        normal_wilson_upper=normal_wilson_upper,
        per_type_auc=per_type_auc,
        pooled_auc=pooled_auc,
        passed=passed,
    )


def _qualify_normal2_with_dependencies_antigravity(
    loader: Callable[[], np.ndarray],
    fold_scorer: Callable[..., FoldScoresAntigravity],
) -> RankQualificationAntigravity:
    values = _train_antigravity(loader())
    seed_evidence: list[RankSeedEvidenceAntigravity] = []
    for seed in _EXPECTED_SEEDS_ANTIGRAVITY:
        original = np.empty(len(values), dtype=float)
        primary = {
            name: np.empty(len(values), dtype=float)
            for name in _EXPECTED_PRIMARY_TYPES_ANTIGRAVITY
        }
        coverage = np.zeros(len(values), dtype=np.uint8)
        dimensions: set[int] = set()
        for fold_index, (fit_indices, query_indices) in enumerate(
            crossfit_plan_antigravity(len(values), seed=seed)
        ):
            if np.intersect1d(fit_indices, query_indices).size:
                raise ValueError("fit and query indices must remain disjoint")
            fold = fold_scorer(
                values,
                fit_indices,
                query_indices,
                seed=seed,
                fold_index=fold_index,
            )
            if type(fold) is not FoldScoresAntigravity or len(fold.original) != len(query_indices):
                raise ValueError("fold scorer returned an invalid result")
            original[query_indices] = fold.original
            for name in _EXPECTED_PRIMARY_TYPES_ANTIGRAVITY:
                primary[name][query_indices] = fold.primary[name]
            coverage[query_indices] += 1
            dimensions.add(fold.feature_dimension)
        if not np.all(coverage == 1):
            raise ValueError("cross-fit must cover every TRAIN row exactly once")
        if len(dimensions) != 1:
            raise ValueError("Hydra feature dimensions differ across folds")
        seed_evidence.append(_seed_evidence_antigravity(seed, original, primary))
    frozen = tuple(seed_evidence)
    return RankQualificationAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        seed_evidence=frozen,
        passed=all(seed.passed for seed in frozen),
    )


def qualify_normal2_antigravity() -> RankQualificationAntigravity:
    return _qualify_normal2_with_dependencies_antigravity(
        loader=load_normal2_train_antigravity,
        fold_scorer=_score_fold_antigravity,
    )


__all__ = (
    "FoldScoresAntigravity",
    "_qualify_normal2_with_dependencies_antigravity",
    "qualify_normal2_antigravity",
)
