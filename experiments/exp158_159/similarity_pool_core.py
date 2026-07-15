"""Pure TRAIN-only descriptor, distance, and pooling helpers for Exp158 and Exp159.

This module is completely decoupled from database access and TEST label access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXP153_DIR = REPO_ROOT / "experiments" / "exp153_157"
if str(EXP153_DIR) not in sys.path:
    sys.path.insert(0, str(EXP153_DIR))

import exp153_157_core as core


def partition_manifest_rows(
    rows: Sequence[Mapping[str, object]], target_limit: int | None
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    """Keep the full donor universe while optionally limiting smoke targets."""

    donor_rows = list(rows)
    if target_limit is None:
        return donor_rows, list(donor_rows)
    if int(target_limit) < 1:
        raise ValueError("target_limit must be positive")
    return donor_rows, donor_rows[: int(target_limit)]


def compute_source_robust_z_scores(scores: Iterable[float]) -> np.ndarray:
    """Convert finite scores to robust z-scores using median/MAD center and scale."""
    finite_scores = core.finite(scores)
    if len(finite_scores) == 0:
        raise ValueError("No finite scores available for robust standardization")
    # robust_standardize returns (z_scores, location, scale)
    z_scores, _, _ = core.robust_standardize(finite_scores, finite_scores)
    return z_scores


def compute_source_descriptor(z_scores: np.ndarray) -> np.ndarray:
    """Calculate q10, q25, q50, q75, q90, mean, std on robust z-scores."""
    if len(z_scores) == 0:
        raise ValueError("Cannot compute descriptor on empty z-scores")
    q10, q25, q50, q75, q90 = np.percentile(z_scores, [10, 25, 50, 75, 90])
    mean = float(np.mean(z_scores))
    std = float(np.std(z_scores))
    return np.asarray([q10, q25, q50, q75, q90, mean, std], dtype=float)


def compute_dataset_descriptor(
    train_oof_by_source: Mapping[str, np.ndarray],
    source_names: Sequence[str] = ("rocket_exp40", "exp55_best", "exp56_best", "exp84"),
) -> tuple[np.ndarray | None, str]:
    """Compute 28-dimensional descriptor for a dataset by concatenating source statistics.

    Returns:
        (descriptor, status_msg): (np.ndarray, "complete") or (None, reason)
    """
    for name in source_names:
        if name not in train_oof_by_source:
            return None, f"missing source: {name}"
        scores = train_oof_by_source[name]
        if scores is None or len(core.finite(scores)) == 0:
            return None, f"no finite scores for source: {name}"

    pieces = []
    for name in source_names:
        scores = train_oof_by_source[name]
        finite_scores = core.finite(scores)
        try:
            z_scores = compute_source_robust_z_scores(finite_scores)
            stats = compute_source_descriptor(z_scores)
            pieces.append(stats)
        except Exception as e:
            return None, f"failed to compute descriptor for source {name}: {e}"

    descriptor = np.concatenate(pieces)
    return descriptor, "complete"


def robust_scale_descriptors(
    descriptors_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust scale each of the 28 columns of descriptors_matrix using median/IQR.

    Returns:
        scaled_matrix: shape (N, 28)
        medians: shape (28,)
        scales: shape (28,)
    """
    medians = np.median(descriptors_matrix, axis=0)
    q75, q25 = np.percentile(descriptors_matrix, [75, 25], axis=0)
    iqr = q75 - q25
    scales = np.zeros(28)
    for col in range(28):
        scale = iqr[col]
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.std(descriptors_matrix[:, col])
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        scales[col] = scale

    scaled_matrix = (descriptors_matrix - medians) / scales
    return scaled_matrix, medians, scales


def compute_distances(target_scaled: np.ndarray, donors_scaled: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance between target and all eligible donors."""
    return np.linalg.norm(donors_scaled - target_scaled, axis=1)


def select_donors(
    target_name: str,
    donor_names: Sequence[str],
    distances: np.ndarray,
    donor_n_trains: Sequence[int],
    min_pool_reference: int = 200,
) -> tuple[list[str], float, float] | None:
    """Select closest donors until we accumulate at least min_pool_reference training samples.

    Includes all ties at the boundary distance exactly.

    Returns:
        (selected_donor_names, min_distance, boundary_distance)
        or None if total n_train from all eligible donors is less than min_pool_reference.
    """
    # Exclude target from donor list
    valid_indices = [i for i, name in enumerate(donor_names) if name != target_name]
    if not valid_indices:
        return None

    sub_names = [donor_names[i] for i in valid_indices]
    sub_distances = distances[valid_indices]
    sub_n_trains = [donor_n_trains[i] for i in valid_indices]

    total_n_train = sum(sub_n_trains)
    if total_n_train < min_pool_reference:
        return None

    # Sort indices by distance ascending
    sorted_idx = np.argsort(sub_distances)
    running_sum = 0
    boundary_distance = 0.0
    found = False

    for idx in sorted_idx:
        running_sum += sub_n_trains[idx]
        if running_sum >= min_pool_reference:
            boundary_distance = sub_distances[idx]
            found = True
            break

    if not found:
        return None

    # Include all donors with distance <= boundary_distance + 1e-9 to avoid float ties differences
    selected_donor_names = []
    min_dist = float(np.min(sub_distances))

    for name, dist in zip(sub_names, sub_distances):
        if dist <= boundary_distance + 1e-9:
            selected_donor_names.append(name)

    return selected_donor_names, min_dist, float(boundary_distance)


def select_donors_by_source_counts(
    target_name: str,
    donor_names: Sequence[str],
    distances: np.ndarray,
    donor_reference_counts_by_source: Mapping[str, Sequence[int]],
    min_pool_reference: int = 200,
) -> tuple[list[str], float, float, dict[str, int]] | None:
    """Select complete distance shells until every source has enough references."""

    names = list(donor_names)
    distance_values = np.asarray(distances, dtype=float)
    if len(names) != len(distance_values):
        raise ValueError("donor name/distance length mismatch")
    if not donor_reference_counts_by_source:
        raise ValueError("source reference counts are required")

    source_counts: dict[str, np.ndarray] = {}
    for source, counts in donor_reference_counts_by_source.items():
        values = np.asarray(counts, dtype=int)
        if len(values) != len(names):
            raise ValueError(f"donor count length mismatch source={source}")
        if np.any(values < 0):
            raise ValueError(f"negative donor reference count source={source}")
        source_counts[str(source)] = values

    valid = np.asarray([name != target_name for name in names], dtype=bool)
    if not np.any(valid):
        return None
    if not np.all(np.isfinite(distance_values[valid])):
        raise ValueError("non-finite donor distance")
    if any(int(np.sum(values[valid])) < int(min_pool_reference) for values in source_counts.values()):
        return None

    selected = np.zeros(len(names), dtype=bool)
    boundary = None
    for distance in np.unique(np.sort(distance_values[valid])):
        shell = valid & np.isclose(distance_values, distance, rtol=0.0, atol=1e-9)
        selected |= shell
        totals = {
            source: int(np.sum(values[selected]))
            for source, values in source_counts.items()
        }
        if all(total >= int(min_pool_reference) for total in totals.values()):
            boundary = float(distance)
            break

    if boundary is None:
        return None
    selected_names = [name for name, keep in zip(names, selected) if keep]
    selected_totals = {
        source: int(np.sum(values[selected]))
        for source, values in source_counts.items()
    }
    return (
        selected_names,
        float(np.min(distance_values[valid])),
        boundary,
        selected_totals,
    )


def integrate_similarity_review_candidates(
    *,
    baseline_hard: Iterable[int],
    baseline_standard: Iterable[int],
    baseline_priority: Iterable[int],
    similarity_candidates: Iterable[int],
    n_test: int,
) -> dict[str, set[int]]:
    """Preserve autonomous lanes and add new similarity evidence to review only."""

    hard = {int(index) for index in baseline_hard}
    standard = {int(index) for index in baseline_standard}
    priority = {int(index) for index in baseline_priority}
    similarity = {int(index) for index in similarity_candidates}
    if hard & standard or hard & priority or standard & priority:
        raise ValueError("baseline lane overlap")
    all_indices = hard | standard | priority | similarity
    invalid = sorted(index for index in all_indices if index < 0 or index >= int(n_test))
    if invalid:
        raise ValueError(f"candidate index out of range: {invalid}")

    baseline_candidate = hard | standard | priority
    standard |= similarity - baseline_candidate
    return {
        "hard": hard,
        "standard_review": standard,
        "priority_review": priority,
        "candidate": hard | standard | priority,
    }


def validate_baseline_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_dataset_names: Sequence[str],
    alphas: Sequence[float],
) -> dict[tuple[str, float], Mapping[str, object]]:
    """Validate frozen baseline coverage and lane integrity before integration."""

    expected = {
        (str(dataset), float(alpha))
        for dataset in expected_dataset_names
        for alpha in alphas
    }
    indexed: dict[tuple[str, float], Mapping[str, object]] = {}
    for row in rows:
        key = (str(row["dataset_name"]), float(row["alpha"]))
        if key in indexed:
            raise ValueError(f"duplicate baseline prediction row: {key}")
        hard = core.parse_indices(row.get("hard_indices"))
        standard = core.parse_indices(row.get("standard_review_indices"))
        priority = core.parse_indices(row.get("priority_review_indices"))
        candidate = core.parse_indices(row.get("candidate_indices"))
        if hard & standard or hard & priority or standard & priority:
            raise ValueError(f"baseline lane overlap: {key}")
        if candidate != hard | standard | priority:
            raise ValueError(f"baseline candidate/lane union mismatch: {key}")
        if "n_test" in row:
            n_test = int(row["n_test"])
            invalid = sorted(index for index in candidate if index < 0 or index >= n_test)
            if invalid:
                raise ValueError(f"baseline candidate index out of range: {key} {invalid}")
        indexed[key] = row

    actual = set(indexed)
    if actual != expected:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise ValueError(
            f"baseline coverage mismatch expected={len(expected)} actual={len(actual)} "
            f"missing={missing} extra={extra}"
        )
    return indexed
