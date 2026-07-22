from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple

SEED = 20260718

FORBIDDEN_PREDICTION_FIELDS = {
    "y_test", "labels", "test_labels", "tp", "fp", "fn", "f1",
    "hard_tp", "hard_fp", "hard_fn", "hard_f1", "combined_f1",
}

import numpy as np

def empirical_upper_tail_pvalues(reference: Iterable[float], query: Iterable[float]) -> np.ndarray:
    ref = np.asarray(list(reference), dtype=float)
    values = np.asarray(list(query), dtype=float)
    ref = ref[np.isfinite(ref)]
    if not len(ref):
        return np.ones(len(values), dtype=float)
    ordered = np.sort(ref)
    counts = len(ordered) - np.searchsorted(ordered, values, side="left")
    result = (1.0 + counts.astype(float)) / (len(ordered) + 1.0)
    result[~np.isfinite(values)] = 1.0
    return result


def leave_one_out_pvalues(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    result = np.ones(len(values), dtype=float)
    for index, value in enumerate(values):
        result[index] = empirical_upper_tail_pvalues(
            np.delete(values, index), np.asarray([value])
        )[0]
    return result



class ReviewTiers(NamedTuple):
    high: set[int]
    standard: set[int]
    exploratory: set[int]
    all_indices: set[int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def agreement_tiers(conservative_indices: set[int], canonical_knn_indices: set[int]) -> ReviewTiers:
    high = conservative_indices & canonical_knn_indices
    standard = conservative_indices - canonical_knn_indices
    exploratory = canonical_knn_indices - conservative_indices
    all_indices = conservative_indices | canonical_knn_indices
    return ReviewTiers(high=high, standard=standard, exploratory=exploratory, all_indices=all_indices)


def stable_dataset_guard(train_folds_agreement_met: Iterable[bool]) -> bool:
    return sum(bool(val) for val in train_folds_agreement_met) >= 4


def conformal_count_cap(train_normal_counts: Iterable[int], alpha: float = 0.01) -> int:
    counts = sorted(list(train_normal_counts))
    n = len(counts)
    if n == 0:
        return 0
    idx = math.ceil((n + 1) * (1.0 - alpha)) - 1
    idx = max(0, min(n - 1, idx))
    return counts[idx]


def apply_count_cap(
    candidate_indices: Iterable[int],
    cap_value: int,
    evidence_by_index: Mapping[int, tuple[float, ...]],
) -> set[int]:
    candidates = sorted(candidate_indices, key=lambda idx: (evidence_by_index.get(idx, ()), idx))
    return set(candidates[:max(0, cap_value)])


def validate_disjoint_indices(index_sets: Mapping[str, Iterable[int]]) -> None:
    seen = {}
    for name, indices in index_sets.items():
        for idx in indices:
            if idx in seen:
                raise ValueError(f"Index {idx} is duplicated between '{seen[idx]}' and '{name}'")
            seen[idx] = name


def validate_prediction_payload(payload: Mapping[str, object]) -> None:
    forbidden = FORBIDDEN_PREDICTION_FIELDS & {str(key).lower() for key in payload}
    if forbidden:
        raise ValueError(f"evaluation fields in frozen prediction: {sorted(forbidden)}")
    if payload.get("contains_test_labels") is not False:
        raise ValueError("prediction must declare contains_test_labels=false")
    if payload.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction must declare contains_evaluation_metrics=false")
    n_test = int(payload["n_test"])
    indices = [int(value) for value in payload.get("candidate_indices", [])]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate candidate indices")
    if indices and (min(indices) < 0 or max(indices) >= n_test):
        raise ValueError("candidate index outside TEST range")
