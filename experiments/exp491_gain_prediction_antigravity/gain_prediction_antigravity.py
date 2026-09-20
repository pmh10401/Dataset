"""Family-held-out Ridge probe of cached descriptors onto historical gain labels."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import (
    METHODS_ANTIGRAVITY,
)
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import jsonable_antigravity


ALTERNATIVE_METHODS_ANTIGRAVITY = METHODS_ANTIGRAVITY[1:]
TARGET_KEYS_ANTIGRAVITY = tuple(
    f"delta_auc_{method}" for method in ALTERNATIVE_METHODS_ANTIGRAVITY
) + tuple(f"sign_delta_zf_{method}" for method in ALTERNATIVE_METHODS_ANTIGRAVITY)
RIDGE_ALPHA_ANTIGRAVITY = 10.0


def target_vector_antigravity(row: object) -> np.ndarray:
    scores = tuple(row.scores)
    if tuple(metric.method for metric in scores) != METHODS_ANTIGRAVITY:
        raise ValueError("scores must contain the four frozen methods in order")
    baseline = scores[0]
    deltas = [float(metric.auc) - float(baseline.auc) for metric in scores[1:]]
    signs = [float(np.sign(int(metric.zf) - int(baseline.zf))) for metric in scores[1:]]
    values = np.asarray(deltas + signs, dtype=np.float64)
    if values.shape != (len(TARGET_KEYS_ANTIGRAVITY),) or not np.isfinite(values).all():
        raise ValueError("targets must be a finite six-vector")
    return values


def _matrix_antigravity(
    rows: Sequence[object],
    descriptor_values: Mapping[str, Sequence[object]],
    feature_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if type(feature_keys) not in (tuple, list) or not feature_keys:
        raise ValueError("feature_keys must be a non-empty sequence")
    keys = tuple(str(key) for key in feature_keys)
    if len(set(keys)) != len(keys):
        raise ValueError("feature keys must be unique")
    width = len(keys)
    features = np.empty((len(rows), width), dtype=np.float64)
    targets = np.empty((len(rows), len(TARGET_KEYS_ANTIGRAVITY)), dtype=np.float64)
    datasets: list[str] = []
    families: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        dataset = str(row.dataset)
        if dataset in seen:
            raise ValueError("dataset IDs must be unique")
        seen.add(dataset)
        if dataset not in descriptor_values:
            raise ValueError(f"missing descriptors for {dataset}")
        values = descriptor_values[dataset]
        if len(values) != width:
            raise ValueError(f"descriptor width mismatch for {dataset}")
        numeric = []
        for value in values:
            if value is None:
                numeric.append(np.nan)
                continue
            number = float(value)
            if math.isinf(number):
                raise ValueError("descriptor values must not be infinite")
            numeric.append(number)
        features[index] = np.asarray(numeric, dtype=np.float64)
        if np.isinf(features[index]).any():
            raise ValueError("descriptor values must not be infinite")
        targets[index] = target_vector_antigravity(row)
        datasets.append(dataset)
        families.append(str(row.source_family))
    return features, targets, tuple(datasets), tuple(families)


def _family_weights_antigravity(families: Sequence[str]) -> np.ndarray:
    counts: dict[str, int] = {}
    for family in families:
        counts[family] = counts.get(family, 0) + 1
    return np.asarray([1.0 / counts[family] for family in families], dtype=np.float64)


def _donor_medians_antigravity(features: np.ndarray) -> np.ndarray:
    medians = np.empty(features.shape[1], dtype=np.float64)
    for column in range(features.shape[1]):
        observed = features[:, column]
        if np.isinf(observed).any():
            raise ValueError("descriptor values must not be infinite")
        finite = observed[~np.isnan(observed)]
        medians[column] = float(np.median(finite)) if finite.size else 0.0
    return medians


def _impute_antigravity(features: np.ndarray, medians: np.ndarray) -> np.ndarray:
    filled = np.array(features, dtype=np.float64, copy=True)
    if np.isinf(filled).any():
        raise ValueError("descriptor values must not be infinite")
    for column, median in enumerate(medians):
        missing = np.isnan(filled[:, column])
        if missing.any():
            filled[missing, column] = median
    return filled


def _standardize_antigravity(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    scaled = (features - mean) / std
    if not np.isfinite(scaled).all():
        raise ValueError("standardized donor features must be finite")
    return scaled, mean, std


def _apply_standardize_antigravity(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    scaled = (features - mean) / std
    if not np.isfinite(scaled).all():
        raise ValueError("standardized query features must be finite")
    return scaled


def _weighted_mean_antigravity(targets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("donor weights must be positive and finite")
    return (weights[:, None] * targets).sum(axis=0) / total


def _family_mean_antigravity(values: np.ndarray, families: Sequence[str]) -> np.ndarray:
    grouped: dict[str, list[np.ndarray]] = {}
    for family, row in zip(families, values):
        grouped.setdefault(family, []).append(row)
    means = [np.mean(np.stack(items, axis=0), axis=0) for _, items in sorted(grouped.items())]
    return np.mean(np.stack(means, axis=0), axis=0)


def _directional_accuracy_antigravity(
    predictions: np.ndarray, targets: np.ndarray, families: Sequence[str]
) -> dict[str, object]:
    match = (np.sign(predictions) == np.sign(targets)).astype(np.float64)
    nonzero = targets != 0.0
    summary = {}
    for index, key in enumerate(TARGET_KEYS_ANTIGRAVITY):
        all_rows = match[:, index]
        family_all = _family_mean_antigravity(all_rows[:, None], families)[0]
        mask = nonzero[:, index]
        if not mask.any():
            nonzero_value: float | None = None
        else:
            kept_pred = match[mask, index][:, None]
            kept_families = tuple(family for family, keep in zip(families, mask) if keep)
            nonzero_value = float(_family_mean_antigravity(kept_pred, kept_families)[0])
        summary[key] = {
            "family_balanced": float(family_all),
            "nonzero_target_family_balanced": nonzero_value,
        }
    return summary


def _positive_gain_auroc_antigravity(
    predictions: np.ndarray,
    targets: np.ndarray,
    families: Sequence[str],
) -> dict[str, object]:
    counts: dict[str, int] = {}
    for family in families:
        counts[family] = counts.get(family, 0) + 1
    weights = np.asarray([1.0 / counts[family] for family in families], dtype=np.float64)
    summary = {}
    for index, key in enumerate(TARGET_KEYS_ANTIGRAVITY):
        positive = (targets[:, index] > 0.0).astype(np.int64)
        classes = set(positive.tolist())
        if classes != {0, 1}:
            summary[key] = {
                "auroc": None,
                "reason": "all_same_class",
                "n_positive": int(positive.sum()),
                "n_negative": int(len(positive) - positive.sum()),
            }
            continue
        score = roc_auc_score(positive, predictions[:, index], sample_weight=weights)
        if not math.isfinite(float(score)):
            raise ValueError("positive-gain AUROC must be finite when both classes exist")
        summary[key] = {
            "auroc": float(score),
            "reason": None,
            "n_positive": int(positive.sum()),
            "n_negative": int(len(positive) - positive.sum()),
        }
    return summary


def evaluate_meta_antigravity(
    rows: Sequence[object],
    descriptor_values: Mapping[str, Sequence[object]],
    feature_keys: Sequence[str],
) -> dict[str, object]:
    """Leave whole source families out; fit imputation, scale, and Ridge on donors only."""
    if not rows:
        raise ValueError("rows must be non-empty")
    features, targets, datasets, families = _matrix_antigravity(rows, descriptor_values, feature_keys)
    unique_families = tuple(sorted(set(families)))
    if len(unique_families) < 2:
        raise ValueError("at least two source families are required")
    fold_predictions = np.empty_like(targets)
    fold_baselines = np.empty_like(targets)
    folds = []
    for held in unique_families:
        donor_mask = np.asarray([family != held for family in families], dtype=bool)
        query_mask = ~donor_mask
        if not donor_mask.any() or not query_mask.any():
            raise ValueError(f"held family {held} has no donors or no query rows")
        donor_families = tuple(sorted({families[index] for index, keep in enumerate(donor_mask) if keep}))
        if held in donor_families:
            raise ValueError("held family leaked into donors")
        donor_x = features[donor_mask]
        donor_y = targets[donor_mask]
        query_x = features[query_mask]
        donor_family_names = tuple(family for family, keep in zip(families, donor_mask) if keep)
        donor_weights = _family_weights_antigravity(donor_family_names)
        family_weight_totals = {
            family: float(sum(weight for name, weight in zip(donor_family_names, donor_weights) if name == family))
            for family in donor_families
        }
        if any(abs(total - 1.0) > 1e-12 for total in family_weight_totals.values()):
            raise ValueError("each donor family must have total Ridge weight 1")
        medians = _donor_medians_antigravity(donor_x)
        donor_filled = _impute_antigravity(donor_x, medians)
        donor_scaled, mean, std = _standardize_antigravity(donor_filled)
        query_filled = _impute_antigravity(query_x, medians)
        query_scaled = _apply_standardize_antigravity(query_filled, mean, std)
        model = Ridge(alpha=RIDGE_ALPHA_ANTIGRAVITY, fit_intercept=True, solver="cholesky")
        model.fit(donor_scaled, donor_y, sample_weight=donor_weights)
        predicted = np.asarray(model.predict(query_scaled), dtype=np.float64)
        baseline = np.repeat(_weighted_mean_antigravity(donor_y, donor_weights)[None, :], predicted.shape[0], axis=0)
        if not np.isfinite(predicted).all() or not np.isfinite(baseline).all():
            raise ValueError("meta predictions must be finite")
        fold_predictions[query_mask] = predicted
        fold_baselines[query_mask] = baseline
        query_datasets = tuple(dataset for dataset, keep in zip(datasets, query_mask) if keep)
        query_targets = targets[query_mask]
        folds.append(
            {
                "held_family": held,
                "donor_families": list(donor_families),
                "n_donors": int(donor_mask.sum()),
                "n_query": int(query_mask.sum()),
                "datasets": list(query_datasets),
                "predictions": {
                    key: predicted[:, index].tolist() for index, key in enumerate(TARGET_KEYS_ANTIGRAVITY)
                },
                "targets": {
                    key: query_targets[:, index].tolist() for index, key in enumerate(TARGET_KEYS_ANTIGRAVITY)
                },
                "constant_baseline": {
                    key: float(baseline[0, index]) for index, key in enumerate(TARGET_KEYS_ANTIGRAVITY)
                },
                "preprocessing": {
                    "medians": medians.tolist(),
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "fit_on": "donors_only_unweighted_rows",
                },
                "ridge_alpha": RIDGE_ALPHA_ANTIGRAVITY,
                "n_donor_families": int(len(donor_families)),
                "family_weight_sum": float(donor_weights.sum()),
                "family_weight_totals": family_weight_totals,
            }
        )
    ridge_se = (fold_predictions - targets) ** 2
    constant_se = (fold_baselines - targets) ** 2
    ridge_family = _family_mean_antigravity(ridge_se, families)
    constant_family = _family_mean_antigravity(constant_se, families)
    auc_slice = slice(0, 3)
    zf_slice = slice(3, 6)
    summary = {
        "family_mean_squared_error": {
            key: {
                "ridge": float(ridge_family[index]),
                "constant": float(constant_family[index]),
                "delta_ridge_minus_constant": float(ridge_family[index] - constant_family[index]),
            }
            for index, key in enumerate(TARGET_KEYS_ANTIGRAVITY)
        },
        "block_mean_squared_error": {
            "delta_auc": {
                "ridge": float(ridge_family[auc_slice].mean()),
                "constant": float(constant_family[auc_slice].mean()),
            },
            "sign_delta_zf": {
                "ridge": float(ridge_family[zf_slice].mean()),
                "constant": float(constant_family[zf_slice].mean()),
            },
        },
        "directional_accuracy": _directional_accuracy_antigravity(fold_predictions, targets, families),
        "positive_gain_auroc": _positive_gain_auroc_antigravity(fold_predictions, targets, families),
    }
    return jsonable_antigravity(
        {
            "schema": "exp491_meta_evaluation_v1",
            "feature_keys": list(feature_keys),
            "target_keys": list(TARGET_KEYS_ANTIGRAVITY),
            "n_rows": int(len(rows)),
            "n_families": int(len(unique_families)),
            "held_families": list(unique_families),
            "folds": folds,
            "summary": summary,
            "label_policy": (
                "historical_supervised_meta_development; "
                "TEST-derived evidence metrics are development labels, not pristine validation"
            ),
        }
    )


__all__ = [
    "ALTERNATIVE_METHODS_ANTIGRAVITY",
    "RIDGE_ALPHA_ANTIGRAVITY",
    "TARGET_KEYS_ANTIGRAVITY",
    "evaluate_meta_antigravity",
    "target_vector_antigravity",
]
