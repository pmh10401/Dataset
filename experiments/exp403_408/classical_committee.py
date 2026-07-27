"""Exp404 TRAIN-cross-fit normality committee over frozen Exp403 features."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .contracts import ALPHAS, SEED, CommitteeScores, TrainEligibility, train_eligibility
from .foundation import crossfit_plan


ModelFactory = Callable[..., object]
MODEL_NAMES = (
    "shrinkage_mahalanobis",
    "isolation_forest",
    "one_class_svm",
    "local_outlier_factor",
)


def score_classical_committee(
    train_features: object,
    test_features: object,
    eligibility: TrainEligibility,
    seed: int = SEED,
    *,
    lof_factory: ModelFactory | None = None,
) -> CommitteeScores:
    """Score only unseen rows and calibrate each model against TRAIN OOF scores.

    The four methods use the same Exp403 feature matrix, so their results are
    collapsed into one normality-evidence group for each frozen alpha.
    """
    train = _feature_matrix(train_features, name="train_features")
    test = _feature_matrix(test_features, name="test_features")
    if train.shape[1] != test.shape[1]:
        raise ValueError("TRAIN and TEST feature dimensions must match")
    if not len(train) or not len(test):
        raise ValueError("non-empty TRAIN and TEST feature matrices required")
    if eligibility != train_eligibility(len(train)):
        raise ValueError("eligibility must match the TRAIN row count")
    if not eligibility.robust_distance or len(train) < 5:
        return CommitteeScores()

    available_models = (
        MODEL_NAMES if eligibility.classical_committee else (MODEL_NAMES[0],)
    )
    model_factories = _model_factories(
        available_models, seed=int(seed), lof_factory=lof_factory
    )
    oof_scores = {
        name: _crossfit_scores(train, factory, name, seed=int(seed))
        for name, factory in model_factories.items()
    }
    test_scores = {
        name: _fit_and_score(train, test, factory, name)
        for name, factory in model_factories.items()
    }
    test_p_values = {
        name: _empirical_p_values(oof_scores[name], test_scores[name])
        for name in available_models
    }

    scores_by_alpha: dict[str, tuple[float, ...]] = {}
    confidence_by_alpha: dict[str, tuple[str, ...]] = {}
    p_value_matrix = np.vstack([test_p_values[name] for name in available_models])
    group_scores = np.median(p_value_matrix, axis=0)
    for alpha in ALPHAS:
        key = str(alpha)
        anomaly_flags = p_value_matrix <= float(alpha)
        scores_by_alpha[key] = tuple(float(value) for value in group_scores)
        confidence_by_alpha[key] = tuple(
            _confidence_grade(anomaly_flags[:, index])
            for index in range(anomaly_flags.shape[1])
        )
    return CommitteeScores(
        scores_by_alpha=scores_by_alpha,
        available_models=available_models,
        confidence_by_alpha=confidence_by_alpha,
    )


def grade_frozen_hard(
    *, frozen_hard: Iterable[int], committee: CommitteeScores
) -> CommitteeScores:
    """Attach the original Hard indices without altering the frozen lane."""
    hard_indices = frozenset(int(index) for index in frozen_hard)
    confidence_by_alpha = committee.confidence_by_alpha
    if not committee.available_models and not confidence_by_alpha:
        confidence_by_alpha = {
            str(alpha): tuple("abstain" for _ in sorted(hard_indices))
            for alpha in ALPHAS
        }
    return CommitteeScores(
        scores_by_alpha=committee.scores_by_alpha,
        available_models=committee.available_models,
        confidence_by_alpha=confidence_by_alpha,
        hard_indices=hard_indices,
    )


def _feature_matrix(values: object, *, name: str) -> np.ndarray:
    if isinstance(values, Mapping):
        columns = [np.asarray(values[key], dtype=float).reshape(-1) for key in sorted(values)]
        if not columns:
            raise ValueError(f"{name} mapping must not be empty")
        matrix = np.column_stack(columns)
    else:
        matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix")
    if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite feature values")
    return matrix


def _model_factories(
    names: tuple[str, ...], *, seed: int, lof_factory: ModelFactory | None
) -> dict[str, ModelFactory]:
    factories: dict[str, ModelFactory] = {
        "shrinkage_mahalanobis": LedoitWolf,
    }
    if "isolation_forest" in names:
        factories["isolation_forest"] = lambda: IsolationForest(
            n_estimators=256,
            max_samples="auto",
            contamination="auto",
            random_state=seed,
            n_jobs=1,
        )
        factories["one_class_svm"] = lambda: OneClassSVM(
            kernel="rbf", gamma="scale", nu=0.01
        )
        factories["local_outlier_factor"] = (
            lof_factory if lof_factory is not None else _default_lof_factory
        )
    return {name: factories[name] for name in names}


def _default_lof_factory(*, n_neighbors: int) -> LocalOutlierFactor:
    return LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
        contamination="auto",
        n_jobs=1,
    )


def _crossfit_scores(
    train: np.ndarray, factory: ModelFactory, name: str, *, seed: int
) -> np.ndarray:
    scores = np.empty(len(train), dtype=float)
    for fold in crossfit_plan(len(train), seed=seed):
        fit_indices = np.asarray(fold.train_indices, dtype=int)
        query_indices = np.asarray(fold.validation_indices, dtype=int)
        scores[query_indices] = _fit_and_score(
            train[fit_indices], train[query_indices], factory, name
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite TRAIN OOF scores from {name}")
    return scores


def _fit_and_score(
    fit_features: np.ndarray, query_features: np.ndarray, factory: ModelFactory, name: str
) -> np.ndarray:
    if name == "local_outlier_factor":
        model = _make_lof(factory, n_neighbors=min(20, len(fit_features) - 1))
    else:
        model = factory()
    fitted = model.fit(fit_features)
    if name == "shrinkage_mahalanobis":
        scores = fitted.mahalanobis(query_features)
    else:
        scores = -fitted.score_samples(query_features)
    result = np.asarray(scores, dtype=float).reshape(-1)
    if len(result) != len(query_features) or not np.isfinite(result).all():
        raise ValueError(f"non-finite query scores from {name}")
    return result


def _make_lof(factory: ModelFactory, *, n_neighbors: int) -> object:
    """Allow the brief's no-argument test factory without relaxing defaults."""
    try:
        return factory(n_neighbors=n_neighbors)
    except TypeError as error:
        try:
            return factory()
        except TypeError:
            raise error


def _empirical_p_values(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (1.0 + np.count_nonzero(reference >= value)) / (len(reference) + 1.0)
            for value in query
        ],
        dtype=float,
    )


def _confidence_grade(anomaly_flags: np.ndarray) -> str:
    if bool(np.all(anomaly_flags)):
        return "high"
    if bool(np.any(anomaly_flags)):
        return "medium"
    return "recheck"
