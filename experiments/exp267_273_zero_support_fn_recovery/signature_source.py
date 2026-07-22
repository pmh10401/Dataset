from __future__ import annotations

from pathlib import Path

import iisignature
import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp239_248_diverse_score_sources.source_base import (
    feature_distance_result,
)
from experiments.exp267_273_zero_support_fn_recovery.contracts import SourceScore


_CHANNEL_ORDER = ("value", "first_difference", "normalized_time")
_DEPTH = 3
_WINDOW_NAMES = ("full", "first_half", "second_half")
_WINDOW_DEFINITIONS = (
    ("full", 0.0, 1.0),
    ("first_half", 0.0, 0.5),
    ("second_half", 0.5, 1.0),
)
_MINIMUM_TRAIN = 5
_MINIMUM_TARGET_LENGTH = 4


def signature_config() -> dict[str, object]:
    """Return the frozen Exp269 ordered-path signature configuration."""
    expected_length = int(iisignature.siglength(len(_CHANNEL_ORDER), _DEPTH))
    return {
        "depth": _DEPTH,
        "channels": len(_CHANNEL_ORDER),
        "channel_count": len(_CHANNEL_ORDER),
        "channel_order": _CHANNEL_ORDER,
        "windows": _WINDOW_NAMES,
        "window_definitions": _WINDOW_DEFINITIONS,
        "minimum_target_length": _MINIMUM_TARGET_LENGTH,
        "expected_signature_length": expected_length,
        "iisignature_version": str(iisignature.__version__),
    }


def _ordered_paths(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2:
        raise ValueError("signature rows must have shape (batch, time)")
    if values.shape[1] < _MINIMUM_TARGET_LENGTH:
        raise ValueError("signature rows must contain at least four time points")
    if not np.isfinite(values).all():
        raise ValueError("signature features require finite input rows")

    first_difference = np.diff(values, axis=1, prepend=values[:, :1])
    normalized_time = np.broadcast_to(
        np.linspace(0.0, 1.0, values.shape[1], dtype=float), values.shape
    )
    return np.stack((values, first_difference, normalized_time), axis=-1)


def signature_features(
    rows: np.ndarray, *, target_length: int | None = None
) -> np.ndarray:
    """Compute fixed depth-3 signatures for the frozen three trace windows."""
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2:
        raise ValueError("signature rows must have shape (batch, time)")
    expected_length = values.shape[1] if target_length is None else int(target_length)
    if values.shape[1] != expected_length:
        raise ValueError("signature rows must match the train target length")
    paths = _ordered_paths(values)
    midpoint = paths.shape[1] // 2
    windows = (paths, paths[:, :midpoint], paths[:, midpoint:])
    signatures = tuple(
        np.asarray(iisignature.sig(window, _DEPTH), dtype=float) for window in windows
    )
    features = np.concatenate(signatures, axis=1)
    expected_width = signature_config()["expected_signature_length"] * len(_WINDOW_NAMES)
    if features.shape != (paths.shape[0], expected_width):
        raise ValueError("unexpected ordered path signature feature shape")
    if not np.isfinite(features).all():
        raise ValueError("non-finite ordered path signature features")
    return features


class OrderedPathSignatureDeviation:
    name = "ordered_path_signature_deviation"
    minimum_train = _MINIMUM_TRAIN

    def score(
        self, bundle: DatasetBundle, seed: int, artifact_dir: Path
    ) -> SourceScore:
        del artifact_dir
        train_rows = np.asarray(bundle.train_normal)
        if train_rows.ndim != 2:
            raise ValueError("signature train rows must have shape (batch, time)")
        target_length = int(train_rows.shape[1])
        config = signature_config()
        feature_dimension = int(
            config["expected_signature_length"] * len(config["windows"])
        )
        if target_length < _MINIMUM_TARGET_LENGTH:
            return SourceScore(
                source=self.name,
                variant="primary",
                train_oof=np.asarray([], dtype=float),
                test_scores=np.asarray([], dtype=float),
                eligible=False,
                abstained=True,
                abstention_reason="target_length_below_4",
                provenance={
                    "seed": int(seed),
                    "n_train": len(bundle.train_normal),
                    "n_test": len(bundle.test_values),
                    "minimum_train": self.minimum_train,
                    "minimum_target_length": _MINIMUM_TARGET_LENGTH,
                    "cross_fitted": False,
                    "self_reference_violations": 0,
                    "signature": config,
                    "signature_feature_dimension": feature_dimension,
                    "iisignature_version": config["iisignature_version"],
                    "contains_test_labels": False,
                },
            )
        result = feature_distance_result(
            source=self.name,
            bundle=bundle,
            seed=seed,
            extractor=lambda rows: signature_features(
                rows, target_length=target_length
            ),
            minimum_train=self.minimum_train,
            extra_provenance={
                "signature": config,
                "signature_feature_dimension": feature_dimension,
                "minimum_train": self.minimum_train,
            },
        )
        provenance = dict(result.provenance)
        provenance.update(
            {
                "signature": config,
                "signature_feature_dimension": feature_dimension,
                "minimum_train": self.minimum_train,
                "minimum_target_length": _MINIMUM_TARGET_LENGTH,
                "iisignature_version": config["iisignature_version"],
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
