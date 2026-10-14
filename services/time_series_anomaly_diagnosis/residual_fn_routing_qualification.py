"""Replay-only qualification for routing against residual FN and FP."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.stats import rankdata

from .anomaly_detection import ALPHA, FIXED_SEEDS, empirical_upper_pvalues
from .policy_qualification import _array_from_record, evaluate_frozen_predictions
from .routing_policy import BASELINE_METHOD, FEATURE_KEYS, _feature_vector
from .score_sources import METHOD_NAMES

SCORE_STABILITY_FEATURE_KEYS = tuple(
    f"{method}_{suffix}"
    for method in METHOD_NAMES
    for suffix in ("seed_rank_agreement", "upper_tail_width")
)
EXPANDED_FEATURE_KEYS = (*FEATURE_KEYS, *SCORE_STABILITY_FEATURE_KEYS)


@dataclass(frozen=True, slots=True)
class MethodOutcome:
    method_name: str
    residual_fn: int
    review_inclusive_fp: int
    automatic_fp: int

    def __post_init__(self) -> None:
        if self.method_name not in METHOD_NAMES:
            raise ValueError("method_name must be an approved score source")
        if any(type(value) is not int or value < 0 for value in (
            self.residual_fn, self.review_inclusive_fp, self.automatic_fp
        )):
            raise ValueError("outcome counts must be non-negative integers")


@dataclass(frozen=True, slots=True)
class ResidualRoutingRecord:
    group_id: int
    features: Mapping[str, float]
    outcomes: tuple[MethodOutcome, ...]
    baseline_method: str

    def __post_init__(self) -> None:
        if type(self.group_id) is not int or self.group_id < 0:
            raise ValueError("group_id must be a non-negative integer")
        feature_set = set(self.features)
        if feature_set == set(FEATURE_KEYS):
            keys = FEATURE_KEYS
        elif feature_set == set(EXPANDED_FEATURE_KEYS):
            keys = EXPANDED_FEATURE_KEYS
        else:
            raise ValueError("features must match the original or expanded TRAIN inputs")
        if tuple(outcome.method_name for outcome in self.outcomes) != METHOD_NAMES:
            raise ValueError("outcomes must use the three approved methods in order")
        if self.baseline_method not in METHOD_NAMES:
            raise ValueError("baseline_method must be an approved score source")
        object.__setattr__(self, "features", MappingProxyType({
            key: float(self.features[key]) for key in keys
        }))


@dataclass(frozen=True, slots=True)
class ResidualGateEvidence:
    method_name: str
    passed: bool
    residual_fn: float
    review_inclusive_fp: float
    automatic_fp: float


@dataclass(frozen=True, slots=True)
class ResidualRouteDecision:
    selected_method: str
    support: int
    evidence: tuple[ResidualGateEvidence, ...]


def _routing_vector(features: Mapping[str, float], keys: tuple[str, ...]) -> np.ndarray:
    if keys == FEATURE_KEYS:
        return _feature_vector(features)
    if keys != EXPANDED_FEATURE_KEYS or set(features) != set(keys):
        raise ValueError("query features must match router inputs")
    values = np.asarray([float(features[key]) for key in keys], dtype=float)
    for index in (0, 1):
        if values[index] > 0.0:
            values[index] = np.log1p(values[index])
    return values


def common_detection_flags(final_pvalues: np.ndarray, *, train_count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(final_pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("final p-values must be finite, one-dimensional, and in [0, 1]")
    if type(train_count) is not int or train_count < 1:
        raise ValueError("train_count must be a positive integer")
    floor = 1.0 / (train_count + 1.0)
    automatic = values <= ALPHA
    review = (~automatic) & (floor > ALPHA) & np.isclose(values, floor, atol=1e-15, rtol=0)
    return automatic, review


def summarize_method_outcome(
    method_name: str,
    *,
    labels: np.ndarray,
    automatic: np.ndarray,
    review: np.ndarray,
) -> MethodOutcome:
    actual = np.asarray(labels)
    automatic_flags = np.asarray(automatic)
    review_flags = np.asarray(review)
    if (
        actual.ndim != 1
        or set(actual.tolist()) != {0, 1}
        or automatic_flags.dtype != np.bool_
        or review_flags.dtype != np.bool_
        or len(actual) != len(automatic_flags)
        or len(actual) != len(review_flags)
        or np.any(automatic_flags & review_flags)
    ):
        raise ValueError("aligned binary labels and disjoint boolean decisions are required")
    positive = actual == 1
    negative = ~positive
    return MethodOutcome(
        method_name,
        int(np.count_nonzero(positive & ~automatic_flags & ~review_flags)),
        int(np.count_nonzero(negative & (automatic_flags | review_flags))),
        int(np.count_nonzero(negative & automatic_flags)),
    )


def extract_score_stability_features(row: Mapping[str, object]) -> dict[str, float]:
    scores = row.get("scores")
    if not isinstance(scores, Mapping) or set(scores) != set(METHOD_NAMES):
        raise ValueError("score row requires all approved methods")
    result = {}
    for method in METHOD_NAMES:
        method_scores = scores[method]
        if not isinstance(method_scores, Mapping) or set(method_scores) != {str(seed) for seed in FIXED_SEEDS}:
            raise ValueError("each method requires both frozen seeds")
        seed_values = [
            _array_from_record(method_scores[str(seed)]["train_oof"], ndim=1)
            for seed in FIXED_SEEDS
        ]
        if not seed_values[0].size or seed_values[0].shape != seed_values[1].shape:
            raise ValueError("seed TRAIN OOF arrays must be non-empty and aligned")
        ranks = [rankdata(values, method="average") for values in seed_values]
        centered = [values - values.mean() for values in ranks]
        denominator = float(np.linalg.norm(centered[0]) * np.linalg.norm(centered[1]))
        agreement = float(np.dot(centered[0], centered[1]) / denominator) if denominator else 0.0

        median_scores = np.median(np.stack(seed_values), axis=0)
        center = float(np.median(median_scores))
        mad_scale = 1.4826 * float(np.median(np.abs(median_scores - center)))
        q25, q75, q90, q99 = np.quantile(
            median_scores, (0.25, 0.75, 0.90, 0.99), method="linear"
        )
        epsilon = max(1e-12, 1e-9 * max(1.0, abs(center)))
        scale = max(mad_scale, float((q75 - q25) / 1.349), float(median_scores.std()), epsilon)
        tail_width = max(0.0, float(q99 - q90)) / scale
        result[f"{method}_seed_rank_agreement"] = agreement
        result[f"{method}_upper_tail_width"] = tail_width
    return {key: result[key] for key in SCORE_STABILITY_FEATURE_KEYS}


def select_residual_fn_method(
    records: Sequence[ResidualRoutingRecord],
    query_features: Mapping[str, float],
    *,
    fallback_method: str,
) -> ResidualRouteDecision:
    rows = tuple(records)
    if fallback_method not in METHOD_NAMES:
        raise ValueError("fallback_method must be an approved score source")
    if len({row.group_id for row in rows}) < 20:
        raise ValueError("router requires at least 20 anonymous groups")
    input_keys = tuple(rows[0].features)
    if any(tuple(row.features) != input_keys for row in rows):
        raise ValueError("all router records must use one feature schema")
    matrix = np.stack([_routing_vector(row.features, input_keys) for row in rows])
    finite = np.isfinite(matrix)
    if np.any(~finite.any(axis=0)):
        raise ValueError("each feature column requires a finite training value")
    medians = np.asarray([
        np.median(matrix[finite[:, index], index]) for index in range(matrix.shape[1])
    ])
    matrix = np.where(finite, matrix, medians)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    query = _routing_vector(query_features, input_keys)
    query = np.where(np.isfinite(query), query, medians)
    distances = np.linalg.norm((matrix - means) / scales - (query - means) / scales, axis=1)
    outcome_maps = [
        {outcome.method_name: outcome for outcome in row.outcomes} for row in rows
    ]
    row_signatures = [tuple(matrix[index].tolist()) for index in range(len(rows))]
    members_by_group: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        members_by_group.setdefault(row.group_id, []).append(index)
    closest_by_group = {}
    for group, indices in members_by_group.items():
        best = min((distances[index], row_signatures[index]) for index in indices)
        closest_by_group[group] = tuple(
            index
            for index in indices
            if (distances[index], row_signatures[index]) == best
        )
    group_signatures = {
        group: tuple(sorted(row_signatures[index] for index in indices))
        for group, indices in members_by_group.items()
    }
    group_keys = {
        group: (
            distances[closest_by_group[group][0]],
            row_signatures[closest_by_group[group][0]],
            group_signatures[group],
        )
        for group in members_by_group
    }
    ordered_groups = sorted(
        members_by_group,
        key=group_keys.__getitem__,
    )
    cutoff_distance = group_keys[ordered_groups[min(31, len(ordered_groups)) - 1]][0]
    selected_groups = [
        group for group in ordered_groups if group_keys[group][0] <= cutoff_distance
    ]

    def totals_for(method_name: str | None) -> tuple[float, float, float]:
        totals = np.zeros(3, dtype=float)
        for group in selected_groups:
            indices = closest_by_group[group]
            group_totals = np.zeros(3, dtype=float)
            for index in indices:
                outcome = outcome_maps[index][
                    rows[index].baseline_method if method_name is None else method_name
                ]
                group_totals += (
                    outcome.residual_fn,
                    outcome.review_inclusive_fp,
                    outcome.automatic_fp,
                )
            totals += group_totals / len(indices)
        return tuple(float(value) for value in totals)

    totals = {method: totals_for(method) for method in METHOD_NAMES}
    baseline = totals_for(None)
    evidence = tuple(
        ResidualGateEvidence(
            method,
            totals[method][0] < baseline[0]
            and totals[method][1] <= baseline[1]
            and totals[method][2] <= baseline[2],
            *totals[method],
        )
        for method in METHOD_NAMES
    )
    passed = [row for row in evidence if row.passed]
    selected_method = min(
        passed,
        key=lambda row: (
            row.residual_fn,
            row.review_inclusive_fp,
            row.automatic_fp,
            METHOD_NAMES.index(row.method_name),
        ),
    ).method_name if passed else fallback_method
    return ResidualRouteDecision(selected_method, len(selected_groups), evidence)


def leave_family_out_routes(
    features_by_dataset: Mapping[str, Mapping[str, float]],
    outcomes_by_dataset: Mapping[str, tuple[MethodOutcome, ...]],
    family_by_dataset: Mapping[str, str],
    baseline_method_by_dataset: Mapping[str, str],
) -> dict[str, str]:
    names = set(features_by_dataset)
    if (
        not names
        or names != set(outcomes_by_dataset)
        or names != set(family_by_dataset)
        or names != set(baseline_method_by_dataset)
        or any(method not in METHOD_NAMES for method in baseline_method_by_dataset.values())
    ):
        raise ValueError("router inputs require identical valid dataset coverage")
    routes = {}
    for outer_family in sorted(set(family_by_dataset.values())):
        inner_names = sorted(name for name in names if family_by_dataset[name] != outer_family)
        outer_names = sorted(name for name in names if family_by_dataset[name] == outer_family)
        groups = {
            family: index
            for index, family in enumerate(sorted({family_by_dataset[name] for name in inner_names}))
        }
        records = tuple(
            ResidualRoutingRecord(
                groups[family_by_dataset[name]],
                features_by_dataset[name],
                outcomes_by_dataset[name],
                baseline_method_by_dataset[name],
            )
            for name in inner_names
        )
        for name in outer_names:
            routes[name] = select_residual_fn_method(
                records,
                features_by_dataset[name],
                fallback_method=baseline_method_by_dataset[name],
            ).selected_method
    return dict(sorted(routes.items()))


def _cached_method_prediction(
    dataset: str,
    row: Mapping[str, object],
    method_name: str,
    *,
    policy_id: str,
) -> dict[str, object]:
    metadata = row["metadata"]
    scores = row["scores"]
    if not isinstance(metadata, Mapping) or not isinstance(scores, Mapping):
        raise ValueError("score row metadata and scores must be mappings")
    method_scores = scores[method_name]
    if not isinstance(method_scores, Mapping) or set(method_scores) != {str(seed) for seed in FIXED_SEEDS}:
        raise ValueError("each method requires both frozen seeds")
    train = np.stack([
        _array_from_record(method_scores[str(seed)]["train_oof"], ndim=1)
        for seed in FIXED_SEEDS
    ])
    test = np.stack([
        _array_from_record(method_scores[str(seed)]["test_scores"], ndim=1)
        for seed in FIXED_SEEDS
    ])
    train_count = int(metadata["train_count"])
    test_indices = list(metadata["test_indices"])
    if train.shape[1] != train_count or test.shape[1] != len(test_indices):
        raise ValueError("frozen score arrays do not match metadata")
    pvalues = empirical_upper_pvalues(np.median(train, axis=0), np.median(test, axis=0))
    automatic, review = common_detection_flags(pvalues, train_count=train_count)
    return {
        "policy_id": policy_id,
        "dataset": dataset,
        "test_indices": test_indices,
        "evaluation_scores": (1.0 - pvalues).tolist(),
        "p_values": pvalues.tolist(),
        "automatic": automatic.tolist(),
        "review": review.tolist(),
        "selected_method": method_name,
    }


def evaluate_cached_router_replay(
    score_rows: Mapping[str, Mapping[str, object]],
    labels_by_dataset: Mapping[str, Mapping[int, int]],
    family_by_dataset: Mapping[str, str],
    baseline_method_by_dataset: Mapping[str, str],
    *,
    include_score_stability: bool = False,
) -> dict[str, object]:
    names = set(score_rows)
    if (
        not names
        or names != set(labels_by_dataset)
        or names != set(family_by_dataset)
        or names != set(baseline_method_by_dataset)
        or any(method not in METHOD_NAMES for method in baseline_method_by_dataset.values())
    ):
        raise ValueError("replay inputs require identical valid dataset coverage")

    predictions = {}
    outcomes_by_dataset = {}
    features_by_dataset = {}
    for name in sorted(names):
        row = score_rows[name]
        if row.get("dataset") != name:
            raise ValueError("score row dataset mismatch")
        features_by_dataset[name] = (
            {**row["features"], **extract_score_stability_features(row)}
            if include_score_stability
            else row["features"]
        )
        by_method = {
            method: _cached_method_prediction(name, row, method, policy_id="cached_method")
            for method in METHOD_NAMES
        }
        predictions[name] = by_method
        indices = by_method[BASELINE_METHOD]["test_indices"]
        labels = np.asarray([labels_by_dataset[name][index] for index in indices], dtype=int)
        outcomes_by_dataset[name] = tuple(
            summarize_method_outcome(
                method,
                labels=labels,
                automatic=np.asarray(by_method[method]["automatic"], dtype=bool),
                review=np.asarray(by_method[method]["review"], dtype=bool),
            )
            for method in METHOD_NAMES
        )

    routes = leave_family_out_routes(
        features_by_dataset,
        outcomes_by_dataset,
        family_by_dataset,
        baseline_method_by_dataset,
    )
    baseline_rows = []
    candidate_rows = []
    for name in sorted(names):
        baseline_rows.append({
            **predictions[name][baseline_method_by_dataset[name]],
            "policy_id": "current_baseline",
        })
        candidate_rows.append({
            **predictions[name][routes[name]],
            "policy_id": "residual_fn_router",
        })
    baseline_metrics = evaluate_frozen_predictions(
        baseline_rows, labels_by_dataset, family_by_dataset
    )
    candidate_metrics = evaluate_frozen_predictions(
        candidate_rows, labels_by_dataset, family_by_dataset
    )
    qualified = bool(
        baseline_metrics.complete
        and candidate_metrics.complete
        and candidate_metrics.residual_fn_total < baseline_metrics.residual_fn_total
        and candidate_metrics.family_balanced_residual_fn_rate
        < baseline_metrics.family_balanced_residual_fn_rate
        and candidate_metrics.review_inclusive_fp_total
        <= baseline_metrics.review_inclusive_fp_total
        and candidate_metrics.automatic_fp_total <= baseline_metrics.automatic_fp_total
    )
    return {
        "qualified": qualified,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "feature_keys": EXPANDED_FEATURE_KEYS if include_score_stability else FEATURE_KEYS,
        "routes": routes,
        "route_counts": dict(sorted(Counter(routes.values()).items())),
        "candidate_predictions": candidate_rows,
    }
