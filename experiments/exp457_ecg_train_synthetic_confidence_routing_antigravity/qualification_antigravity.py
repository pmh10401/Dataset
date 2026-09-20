"""TRAIN-only five-fold Hydra LocalGap qualification for Exp457."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import sqrt
from types import MappingProxyType

import numpy as np

from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import (
    LocalGapModelAntigravity,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import (
    make_representation_antigravity,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
    _feature_matrix_antigravity,
    _reference_indices_antigravity,
    _score_chunks_antigravity,
    crossfit_plan_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    TrainQualificationAntigravity,
    TrainSeedEvidenceAntigravity,
)
from .perturbations_antigravity import build_primary_perturbations_antigravity
from .sources_antigravity import load_ecg_train_antigravity


_WILSON_95_Z_ANTIGRAVITY = 1.959963984540054


def _vector_antigravity(values: np.ndarray, name: str, *, finite: bool = True) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional ndarray")
    if not (
        np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    ):
        raise ValueError(f"{name} must have a real numeric dtype")
    result = np.asarray(values, dtype=float)
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def empirical_pvalues_antigravity(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return frozen upper-tail plus-one empirical p-values."""
    reference_values = _vector_antigravity(reference, "reference")
    query_values = _vector_antigravity(query, "query")
    ordered = np.sort(reference_values)
    counts = len(ordered) - np.searchsorted(ordered, query_values, side="left")
    return np.asarray((1.0 + counts) / (1.0 + len(ordered)), dtype=float)


def wilson_upper_antigravity(successes: int, trials: int) -> float:
    """Return the two-sided 95% Wilson interval's upper endpoint."""
    if (
        type(successes) is not int
        or type(trials) is not int
        or trials < 1
        or successes < 0
        or successes > trials
    ):
        raise ValueError("successes and trials must be exact valid integers")
    proportion = successes / trials
    z_squared = _WILSON_95_Z_ANTIGRAVITY**2
    denominator = 1.0 + z_squared / trials
    center = proportion + z_squared / (2.0 * trials)
    adjustment = _WILSON_95_Z_ANTIGRAVITY * sqrt(
        (proportion * (1.0 - proportion) + z_squared / (4.0 * trials)) / trials
    )
    return float(min(1.0, (center + adjustment) / denominator))


@dataclass(frozen=True, slots=True)
class _FoldScoresAntigravity:
    original: np.ndarray
    primary: Mapping[str, np.ndarray]
    feature_dimension: int

    def __post_init__(self) -> None:
        original = _vector_antigravity(self.original, "original fold scores", finite=False)
        if tuple(self.primary) != FROZEN_CONFIG_ANTIGRAVITY.primary_types:
            raise ValueError("primary fold scores must use the exact ordered types")
        frozen: dict[str, np.ndarray] = {}
        for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types:
            values = _vector_antigravity(self.primary[name], f"{name} fold scores", finite=False)
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
    representation_factory: Callable[[str, int], object] = make_representation_antigravity,
    model_factory: Callable[..., object] = LocalGapModelAntigravity,
) -> _FoldScoresAntigravity:
    reference_indices = _reference_indices_antigravity(
        fit_indices,
        seed=seed + 1009 * (fold_index + 1) + FROZEN_CONFIG_ANTIGRAVITY.kernels,
    )
    representation = representation_factory("hydra_local_gap", seed)
    reference_features = _feature_matrix_antigravity(
        representation.fit_transform(train[reference_indices])
    )
    model = model_factory(reference_features, neighbors=FROZEN_CONFIG_ANTIGRAVITY.knn_k)
    original = _score_chunks_antigravity(
        representation,
        model,
        train[query_indices],
        feature_dimension=reference_features.shape[1],
    )
    row_perturbations = tuple(
        build_primary_perturbations_antigravity(row) for row in train[query_indices]
    )
    perturbations = {
        name: np.stack([row[name] for row in row_perturbations])
        for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
    }
    primary = {
        name: _score_chunks_antigravity(
            representation,
            model,
            perturbations[name],
            feature_dimension=reference_features.shape[1],
        )
        for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
    }
    return _FoldScoresAntigravity(original, primary, int(reference_features.shape[1]))


def _train_antigravity(dataset: str, train: np.ndarray) -> np.ndarray:
    if type(dataset) is not str or dataset not in FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets:
        raise ValueError("dataset must be one of the five exact ECG5000 datasets")
    if (
        type(train) is not np.ndarray
        or train.ndim != 2
        or len(train) < FROZEN_CONFIG_ANTIGRAVITY.folds
        or train.dtype != np.dtype(np.float32)
        or not np.isfinite(train).all()
    ):
        raise ValueError("train must be a finite float32 two-dimensional TRAIN matrix")
    return train


def _seed_evidence_antigravity(
    dataset: str,
    seed: int,
    original: np.ndarray,
    primary: Mapping[str, np.ndarray],
) -> TrainSeedEvidenceAntigravity:
    finite = np.isfinite(original).all() and all(
        np.isfinite(primary[name]).all() for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
    )
    if finite:
        normal_alerts = int(
            np.count_nonzero(
                empirical_pvalues_antigravity(original, original)
                <= FROZEN_CONFIG_ANTIGRAVITY.alpha
            )
        )
        sensitivities = {
            name: float(
                np.mean(
                    empirical_pvalues_antigravity(original, primary[name])
                    <= FROZEN_CONFIG_ANTIGRAVITY.alpha
                )
            )
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
        }
    else:
        normal_alerts = 0
        sensitivities = {name: 0.0 for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types}
    pooled = float(sum(sensitivities.values()) / len(FROZEN_CONFIG_ANTIGRAVITY.primary_types))
    wilson = wilson_upper_antigravity(normal_alerts, len(original))
    passed = bool(
        finite
        and wilson <= FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max
        and pooled >= FROZEN_CONFIG_ANTIGRAVITY.pooled_sensitivity_min
        and all(
            sensitivities[name] >= FROZEN_CONFIG_ANTIGRAVITY.per_type_sensitivity_min
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
        )
    )
    return TrainSeedEvidenceAntigravity(
        dataset=dataset,
        seed=seed,
        normal_trials=len(original),
        normal_alerts=normal_alerts,
        normal_wilson_upper=wilson,
        pooled_sensitivity=pooled,
        per_type_sensitivity=sensitivities,
        passed=passed,
    )


def _qualify_ecg_dataset_antigravity(
    dataset: str,
    train: np.ndarray,
    *,
    fold_scorer: Callable[..., _FoldScoresAntigravity] = _score_fold_antigravity,
) -> TrainQualificationAntigravity:
    values = _train_antigravity(dataset, train)
    evidence: list[TrainSeedEvidenceAntigravity] = []
    for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        original = np.empty(len(values), dtype=float)
        primary = {
            name: np.empty(len(values), dtype=float)
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
        }
        coverage = np.zeros(len(values), dtype=np.uint8)
        dimensions: set[int] = set()
        for fold_index, (fit_indices, query_indices) in enumerate(
            crossfit_plan_antigravity(len(values), seed=seed)
        ):
            fold = fold_scorer(
                values,
                fit_indices,
                query_indices,
                seed=seed,
                fold_index=fold_index,
            )
            if type(fold) is not _FoldScoresAntigravity or len(fold.original) != len(query_indices):
                raise ValueError("fold scorer returned an invalid result")
            original[query_indices] = fold.original
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types:
                primary[name][query_indices] = fold.primary[name]
            coverage[query_indices] += 1
            dimensions.add(fold.feature_dimension)
        if not np.all(coverage == 1):
            raise ValueError("cross-fit must cover every TRAIN row exactly once")
        if len(dimensions) != 1:
            raise ValueError("Hydra feature dimensions differ across folds")
        evidence.append(_seed_evidence_antigravity(dataset, seed, original, primary))
    frozen_evidence = tuple(evidence)
    return TrainQualificationAntigravity(
        dataset=dataset,
        seed_evidence=frozen_evidence,
        passed=all(item.passed for item in frozen_evidence),
    )


def qualify_ecg_dataset_antigravity(
    dataset: str,
    train: np.ndarray,
) -> TrainQualificationAntigravity:
    return _qualify_ecg_dataset_antigravity(dataset, train)


def _qualify_all_ecg_antigravity(
    *,
    loader: Callable[[str], np.ndarray] = load_ecg_train_antigravity,
    qualifier: Callable[[str, np.ndarray], TrainQualificationAntigravity] = qualify_ecg_dataset_antigravity,
) -> tuple[TrainQualificationAntigravity, ...]:
    return tuple(
        qualifier(dataset, loader(dataset))
        for dataset in FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets
    )


def qualify_all_ecg_antigravity() -> tuple[TrainQualificationAntigravity, ...]:
    return _qualify_all_ecg_antigravity()


__all__ = (
    "empirical_pvalues_antigravity",
    "qualify_all_ecg_antigravity",
    "qualify_ecg_dataset_antigravity",
    "wilson_upper_antigravity",
)
