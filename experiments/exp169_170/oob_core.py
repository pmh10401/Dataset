from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OOBSplit:
    model_index: int
    fit_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]


def balanced_oob_plan(n_train: int, models: int = 32, holdout_fraction: float = 0.35, seed: int = 20260715) -> list[OOBSplit]:
    holdout_size = max(2, int(math.ceil(holdout_fraction * n_train)))
    if n_train < 5 or holdout_size >= n_train:
        raise ValueError("OOB plan requires at least five TRAIN rows")
    order = np.random.default_rng(seed + n_train).permutation(n_train).tolist()
    splits = []
    for model_index in range(models):
        start = model_index % n_train
        held = tuple(sorted(order[(start + offset) % n_train] for offset in range(holdout_size)))
        fit = tuple(index for index in range(n_train) if index not in set(held))
        splits.append(OOBSplit(model_index, fit, held))
    return splits


def robust_model_scale(fit_scores: np.ndarray) -> tuple[float, float]:
    fit_scores = np.asarray(fit_scores, dtype=np.float64)
    if fit_scores.ndim != 1 or fit_scores.size == 0:
        raise ValueError("fit_scores must not be empty")
    if not np.isfinite(fit_scores).all():
        raise ValueError("fit_scores contain non-finite values")
    center = float(np.median(fit_scores))
    mad = float(np.median(np.abs(fit_scores - center)))
    q75, q25 = np.percentile(fit_scores, [75, 25])
    iqr = float(q75 - q25)
    std = float(np.std(fit_scores))
    epsilon = max(1e-12, 1e-9 * max(1.0, abs(center)))
    scale = max(1.4826 * mad, iqr / 1.349, std, epsilon)
    return center, scale


def aggregate_oob_scores(
    plan: list[OOBSplit],
    train_components: np.ndarray,
    test_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_components = np.asarray(train_components, dtype=np.float64)
    test_components = np.asarray(test_components, dtype=np.float64)
    if train_components.ndim != 2 or test_components.ndim != 2:
        raise ValueError("score components must be two-dimensional matrices")
    if not np.isfinite(train_components).all() or not np.isfinite(test_components).all():
        raise ValueError("non-finite components in input scores")

    n_models = len(plan)
    if n_models == 0:
        raise ValueError("OOB plan must not be empty")
    if train_components.shape[0] != n_models or test_components.shape[0] != n_models:
        raise ValueError("number of models in plan and component shapes mismatch")
    if [split.model_index for split in plan] != list(range(n_models)):
        raise ValueError("OOB plan model indices must be consecutive and ordered")

    n_train = train_components.shape[1]
    n_test = test_components.shape[1]
    expected_indices = set(range(n_train))
    for split in plan:
        fit = set(split.fit_indices)
        holdout = set(split.holdout_indices)
        if fit & holdout or fit | holdout != expected_indices or not holdout:
            raise ValueError(f"invalid OOB split partition for model {split.model_index}")

    oob_models_per_instance: list[list[int]] = [[] for _ in range(n_train)]
    for split in plan:
        for idx in split.holdout_indices:
            oob_models_per_instance[idx].append(split.model_index)

    q_d = min(len(models) for models in oob_models_per_instance)
    if q_d < 1:
        raise ValueError("Each TRAIN instance must be in holdout at least once.")

    centers = []
    scales = []
    for split in plan:
        fit_scores = train_components[split.model_index, list(split.fit_indices)]
        center, scale = robust_model_scale(fit_scores)
        centers.append(center)
        scales.append(scale)

    aggregated_train = np.empty(n_train, dtype=np.float64)
    for i in range(n_train):
        selected_models = oob_models_per_instance[i][:q_d]
        z_scores = []
        for m in selected_models:
            val = train_components[m, i]
            z = (val - centers[m]) / scales[m]
            z_scores.append(z)
        z_arr = np.array(z_scores, dtype=np.float64)
        if not np.isfinite(z_arr).all():
            raise ValueError(f"non-finite z-score calculated for train instance {i}")
        aggregated_train[i] = np.median(z_arr)

    aggregated_test = np.empty(n_test, dtype=np.float64)
    selected_test_models = list(range(q_d))
    for j in range(n_test):
        z_scores = []
        for m in selected_test_models:
            val = test_components[m, j]
            z = (val - centers[m]) / scales[m]
            z_scores.append(z)
        z_arr = np.array(z_scores, dtype=np.float64)
        if not np.isfinite(z_arr).all():
            raise ValueError(f"non-finite z-score calculated for test instance {j}")
        aggregated_test[j] = np.median(z_arr)

    return aggregated_train, aggregated_test, q_d
