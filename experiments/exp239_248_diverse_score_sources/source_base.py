from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import (
    DatasetBundle,
    ScoreResult,
)
from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
from experiments.exp239_248_diverse_score_sources.robust_distance import RobustDistanceModel


FeatureExtractor = Callable[[np.ndarray], np.ndarray]


def abstained_result(source: str, bundle: DatasetBundle, minimum: int) -> ScoreResult:
    return ScoreResult(
        source=source,
        train_oof=np.asarray([], dtype=float),
        test_scores=np.asarray([], dtype=float),
        abstained=True,
        abstention_reason=f"n_train_below_{minimum}",
        provenance={
            "n_train": len(bundle.train_normal),
            "n_test": len(bundle.test_values),
            "minimum_train": minimum,
            "cross_fitted": False,
            "self_reference_violations": 0,
            "contains_test_labels": False,
        },
    )


def feature_distance_result(
    *,
    source: str,
    bundle: DatasetBundle,
    seed: int,
    extractor: FeatureExtractor,
    minimum_train: int = 5,
    extra_provenance: dict[str, object] | None = None,
) -> ScoreResult:
    if len(bundle.train_normal) < minimum_train:
        return abstained_result(source, bundle, minimum_train)
    train_features = np.asarray(extractor(bundle.train_normal), dtype=float)
    test_features = np.asarray(extractor(bundle.test_values), dtype=float)
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError(f"{source} features must be two-dimensional")
    train_oof = np.empty(len(train_features), dtype=float)
    for fit, query in crossfit_plan(len(train_features), seed=seed):
        model = RobustDistanceModel(train_features[fit])
        train_oof[query] = model.score(train_features[query])
    test_scores = RobustDistanceModel(train_features).score(test_features)
    if not np.isfinite(train_oof).all() or not np.isfinite(test_scores).all():
        raise ValueError(f"non-finite source scores: {source}")
    provenance = {
        "seed": int(seed),
        "n_train": len(train_features),
        "n_test": len(test_features),
        "feature_dimension": train_features.shape[1],
        "cross_fitted": True,
        "self_reference_violations": 0,
        "contains_test_labels": False,
    }
    provenance.update(extra_provenance or {})
    return ScoreResult(source, train_oof, test_scores, False, None, provenance)


def resample_rows(rows: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    if values.shape[1] <= maximum:
        return values
    old = np.linspace(0.0, 1.0, values.shape[1])
    new = np.linspace(0.0, 1.0, int(maximum))
    return np.vstack([np.interp(new, old, row) for row in values])

