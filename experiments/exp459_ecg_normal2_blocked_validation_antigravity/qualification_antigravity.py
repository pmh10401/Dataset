"""Blocked fold-seed qualification for Exp459 Task 3."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
    _score_fold_antigravity as _exp457_score_fold_antigravity,
    empirical_pvalues_antigravity,
)

from .blocked_folds_antigravity import blocked_plan_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    BlockedFoldAntigravity,
    BlockedQualificationAntigravity,
    FoldEvidenceAntigravity,
    SeedSummaryAntigravity,
    mann_whitney_auc_antigravity,
    wilson_upper_antigravity,
)
from .sources_antigravity import load_normal2_train_antigravity


def _readonly_float_vector_antigravity(values: object, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional ndarray")
    if values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must have a real numeric dtype")
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return np.frombuffer(np.array(result, dtype=np.float64, copy=True).tobytes(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FoldScoresAntigravity:
    original: np.ndarray
    primary: Mapping[str, np.ndarray]
    feature_dimension: int

    def __post_init__(self) -> None:
        original = _readonly_float_vector_antigravity(self.original, "original fold scores")
        if tuple(self.primary) != FROZEN_CONFIG_ANTIGRAVITY.primary_types:
            raise ValueError("primary fold scores must use the exact ordered primary types")
        frozen: dict[str, np.ndarray] = {}
        for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types:
            values = _readonly_float_vector_antigravity(self.primary[name], f"{name} fold scores")
            if len(values) != len(original):
                raise ValueError("fold score lengths must match")
            frozen[name] = values
        if type(self.feature_dimension) is not int or self.feature_dimension < 1:
            raise ValueError("feature_dimension must be an exact positive integer")
        object.__setattr__(self, "original", original)
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
        original=np.asarray(fold.original, dtype=float),
        primary={
            name: np.asarray(fold.primary[name], dtype=float)
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
        },
        feature_dimension=fold.feature_dimension,
    )


def _train_antigravity(train: object) -> np.ndarray:
    if not isinstance(train, np.ndarray) or train.ndim != 2:
        raise ValueError("train must be a two-dimensional ndarray")
    if train.shape != (
        FROZEN_CONFIG_ANTIGRAVITY.normal2_train_rows,
        FROZEN_CONFIG_ANTIGRAVITY.normal2_width,
    ):
        raise ValueError("train must be the exact 1413x140 normal_2 TRAIN matrix")
    if train.dtype != np.dtype(np.float32):
        raise ValueError("train must be float32")
    if not np.isfinite(train).all():
        raise ValueError("train must be finite")
    result = np.array(train, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


def _blocked_plan_for_train_antigravity(train_rows: int) -> tuple[BlockedFoldAntigravity, ...]:
    folds = tuple(blocked_plan_antigravity(train_rows))
    if len(folds) != FROZEN_CONFIG_ANTIGRAVITY.folds:
        raise ValueError("blocked plan must contain the exact five folds")
    if any(type(fold) is not BlockedFoldAntigravity for fold in folds):
        raise ValueError("blocked plan returned an invalid fold")
    if tuple(fold.fold_index for fold in folds) != tuple(range(FROZEN_CONFIG_ANTIGRAVITY.folds)):
        raise ValueError("blocked plan must stay in exact fold order")
    return folds


def _fold_evidence_antigravity(
    seed: int,
    fold: BlockedFoldAntigravity,
    scores: FoldScoresAntigravity,
) -> FoldEvidenceAntigravity:
    normal_alerts = int(
        np.count_nonzero(
            empirical_pvalues_antigravity(scores.original, scores.original)
            <= FROZEN_CONFIG_ANTIGRAVITY.alpha
        )
    )
    per_type_auc = {
        name: float(mann_whitney_auc_antigravity(scores.original, scores.primary[name]))
        for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
    }
    pooled_auc = float(sum(per_type_auc.values()) / len(FROZEN_CONFIG_ANTIGRAVITY.primary_types))
    normal_wilson_upper = wilson_upper_antigravity(normal_alerts, len(scores.original))
    passed = bool(
        normal_wilson_upper <= FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max
        and pooled_auc >= FROZEN_CONFIG_ANTIGRAVITY.pooled_auc_min
        and all(
            per_type_auc[name] >= FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
        )
    )
    return FoldEvidenceAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        seed=seed,
        fold_index=fold.fold_index,
        fit_indices=fold.fit_indices,
        query_indices=fold.query_indices,
        normal_trials=len(scores.original),
        normal_alerts=normal_alerts,
        normal_wilson_upper=normal_wilson_upper,
        per_type_auc=per_type_auc,
        pooled_auc=pooled_auc,
        passed=passed,
    )


def _qualify_blocked_normal2_with_dependencies_antigravity(
    loader: Callable[[], np.ndarray],
    fold_scorer: Callable[..., FoldScoresAntigravity],
) -> BlockedQualificationAntigravity:
    train = _train_antigravity(loader())
    folds = _blocked_plan_for_train_antigravity(len(train))
    fold_evidence: list[FoldEvidenceAntigravity] = []
    seed_summaries: list[SeedSummaryAntigravity] = []
    all_feature_dimensions: set[int] = set()

    for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        seed_records: list[FoldEvidenceAntigravity] = []
        coverage = np.zeros(len(train), dtype=np.uint8)
        for fold in folds:
            if np.intersect1d(fold.fit_indices, fold.query_indices).size:
                raise ValueError("fit and query indices must remain disjoint")
            scores = fold_scorer(
                train,
                fold.fit_indices,
                fold.query_indices,
                seed=seed,
                fold_index=fold.fold_index,
            )
            if type(scores) is not FoldScoresAntigravity:
                raise ValueError("fold scorer returned an invalid result")
            if len(scores.original) != len(fold.query_indices):
                raise ValueError("fold score length must match query size")
            all_feature_dimensions.add(scores.feature_dimension)
            record = _fold_evidence_antigravity(seed, fold, scores)
            seed_records.append(record)
            fold_evidence.append(record)
            np.add.at(coverage, fold.query_indices, 1)
        if not np.all(coverage == 1):
            raise ValueError("blocked folds must cover every TRAIN row exactly once per seed")
        seed_summaries.append(
            SeedSummaryAntigravity(
                dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
                seed=seed,
                fold_records=tuple(seed_records),
                passed=all(record.passed for record in seed_records),
            )
        )

    if len(all_feature_dimensions) != 1:
        raise ValueError("Hydra feature dimensions differ across folds")
    summaries = tuple(seed_summaries)
    return BlockedQualificationAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        fold_evidence=tuple(fold_evidence),
        seed_summaries=summaries,  # type: ignore[arg-type]
        passed=all(summary.passed for summary in summaries),
    )


def qualify_blocked_normal2_antigravity() -> BlockedQualificationAntigravity:
    return _qualify_blocked_normal2_with_dependencies_antigravity(
        loader=load_normal2_train_antigravity,
        fold_scorer=_score_fold_antigravity,
    )


__all__ = (
    "FoldScoresAntigravity",
    "_qualify_blocked_normal2_with_dependencies_antigravity",
    "_score_fold_antigravity",
    "qualify_blocked_normal2_antigravity",
)
