from __future__ import annotations

from functools import lru_cache
from math import floor, log2
from pathlib import Path
import warnings

import numpy as np
import scipy.special


_MISSING = object()


def _legacy_sph_harm(m: object, n: object, theta: object, phi: object) -> object:
    """Adapt SciPy's removed ``sph_harm(m, n, theta, phi)`` signature."""
    return scipy.special.sph_harm_y(n, m, theta, phi)


def _load_scattering1d() -> object:
    """Import Kymatio without leaking its temporary SciPy compatibility hook."""
    previous = getattr(scipy.special, "sph_harm", _MISSING)
    scipy.special.sph_harm = _legacy_sph_harm
    try:
        from kymatio.numpy import Scattering1D

        return Scattering1D
    finally:
        if previous is _MISSING:
            delattr(scipy.special, "sph_harm")
        else:
            scipy.special.sph_harm = previous


Scattering1D = _load_scattering1d()

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp239_248_diverse_score_sources.source_base import (
    feature_distance_result,
)
from experiments.exp267_273_zero_support_fn_recovery.contracts import SourceScore


def scattering_config(target_length: int) -> dict[str, object]:
    if target_length < 4:
        raise ValueError("scattering target length must be at least 4")
    return {
        "target_length": int(target_length),
        "J": max(2, min(6, floor(log2(target_length)) - 2)),
        "Q": (8, 1),
        "max_order": 2,
        "backend": "kymatio.numpy",
        "aggregation": ("log_abs_mean", "log_abs_std", "log_abs_p90", "log_abs_max"),
    }


@lru_cache(maxsize=None)
def _scattering_transform(target_length: int) -> Scattering1D:
    config = scattering_config(target_length)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Signal support is too small to avoid border effects",
            category=UserWarning,
        )
        return Scattering1D(
            J=config["J"],
            shape=config["target_length"],
            Q=config["Q"],
            max_order=config["max_order"],
        )


def scattering_features(
    rows: np.ndarray, *, target_length: int | None = None
) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2:
        raise ValueError("scattering rows must have shape (batch, time)")
    expected_length = values.shape[1] if target_length is None else int(target_length)
    if values.shape[1] != expected_length:
        raise ValueError("scattering rows must match the train target length")
    if not np.isfinite(values).all():
        raise ValueError("scattering features require finite input rows")

    coefficients = np.asarray(_scattering_transform(expected_length)(values), dtype=float)
    if coefficients.ndim != 3 or coefficients.shape[0] != values.shape[0]:
        raise ValueError("unexpected Kymatio batch output shape")
    logged = np.log(np.abs(coefficients) + 1e-12)
    features = np.concatenate(
        (
            logged.mean(axis=-1),
            logged.std(axis=-1),
            np.quantile(logged, 0.90, axis=-1),
            logged.max(axis=-1),
        ),
        axis=1,
    )
    if not np.isfinite(features).all():
        raise ValueError("non-finite scattering features")
    return features


@lru_cache(maxsize=None)
def scattering_feature_dimension(target_length: int) -> int:
    return int(
        scattering_features(
            np.zeros((1, target_length), dtype=float), target_length=target_length
        ).shape[1]
    )


class MultiscaleScatteringDeviation:
    name = "multiscale_scattering_deviation"
    minimum_train = 5

    def score(
        self, bundle: DatasetBundle, seed: int, artifact_dir: Path
    ) -> SourceScore:
        del artifact_dir
        train_rows = np.asarray(bundle.train_normal)
        if train_rows.ndim != 2:
            raise ValueError("scattering train rows must have shape (batch, time)")
        target_length = int(train_rows.shape[1])
        config = scattering_config(target_length)
        dimension = scattering_feature_dimension(target_length)
        result = feature_distance_result(
            source=self.name,
            bundle=bundle,
            seed=seed,
            extractor=lambda rows: scattering_features(
                rows, target_length=target_length
            ),
            minimum_train=self.minimum_train,
            extra_provenance={
                "scattering": config,
                "scattering_feature_dimension": dimension,
            },
        )
        provenance = dict(result.provenance)
        provenance.update(
            {
                "scattering": config,
                "scattering_feature_dimension": dimension,
                "contains_test_labels": False,
            }
        )
        return SourceScore(
            source=result.source,
            variant="primary",
            train_oof=result.train_oof,
            test_scores=result.test_scores,
            eligible=not result.abstained,
            abstained=result.abstained,
            abstention_reason=result.abstention_reason,
            provenance=provenance,
        )
