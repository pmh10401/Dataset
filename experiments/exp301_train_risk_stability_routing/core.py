from __future__ import annotations

import math
import hashlib
from collections.abc import Iterable, Mapping
from statistics import NormalDist

import numpy as np


EXPERIMENT_ID = "experiment_301_train_risk_stability_routing"
AUTOMATIC_ALPHA = 0.01
PRIORITY_ALPHA = 0.02
GENERAL_ALPHA = 0.005
BOOTSTRAP_REPLICATES = 32
BOOTSTRAP_SEED = 20260722
STABILITY_MINIMUM = 0.80
TRAIN_RISK_RATE_MAXIMUM = 0.005
TRAIN_RISK_UPPER_MAXIMUM = 0.01


def loo_upper_pvalues(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=float).reshape(-1)
    if len(scores) == 0:
        return np.asarray([], dtype=float)
    ordered = np.sort(scores)
    first_equal = np.searchsorted(ordered, scores, side="left")
    return (len(scores) - first_equal) / len(scores)


def empirical_upper_pvalues(reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    query = np.asarray(queries, dtype=float).reshape(-1)
    if len(ref) == 0:
        return np.ones(len(query), dtype=float)
    ordered = np.sort(ref)
    count_greater_equal = len(ref) - np.searchsorted(ordered, query, side="left")
    return (1.0 + count_greater_equal) / (len(ref) + 1.0)


def wilson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return 1.0
    proportion = successes / trials
    z = NormalDist().inv_cdf(confidence)
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return float((center + radius) / denominator)


def consensus_train_flags(
    multirocket_train_by_seed: np.ndarray,
    structural_train: np.ndarray,
    change_point_train: np.ndarray,
    *,
    alpha: float,
) -> dict[str, np.ndarray]:
    multi = np.asarray(multirocket_train_by_seed, dtype=float)
    structure = np.asarray(structural_train, dtype=float).reshape(-1)
    change = np.asarray(change_point_train, dtype=float).reshape(-1)
    if multi.ndim != 2 or multi.shape[0] != 2:
        raise ValueError("MultiROCKET TRAIN scores must have two seed rows")
    lengths = {multi.shape[1], len(structure), len(change)}
    if len(lengths) != 1:
        raise ValueError(f"TRAIN score length mismatch: {sorted(lengths)}")
    multi_flags = np.logical_and.reduce(
        [loo_upper_pvalues(multi[seed]) <= float(alpha) for seed in range(2)]
    )
    structural_flags = loo_upper_pvalues(structure) <= float(alpha)
    change_flags = loo_upper_pvalues(change) <= float(alpha)
    return {
        "multirocket": multi_flags,
        "structural": structural_flags,
        "change_point": change_flags,
        "three_of_three": multi_flags & structural_flags & change_flags,
    }


def _bootstrap_indices(dataset: str, replicate: int, n_train: int, seed: int) -> np.ndarray:
    material = f"{int(seed)}:{dataset}:{int(replicate)}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return np.random.default_rng(local_seed).integers(0, n_train, size=n_train, dtype=np.int64)


def bootstrap_consensus_stability(
    *,
    dataset: str,
    multirocket_train_by_seed: np.ndarray,
    multirocket_test_by_seed: np.ndarray,
    structural_train: np.ndarray,
    structural_test: np.ndarray,
    change_point_train: np.ndarray,
    change_point_test: np.ndarray,
    candidate_indices: Iterable[int],
    alpha: float,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[int, float]:
    multi_train = np.asarray(multirocket_train_by_seed, dtype=float)
    multi_test = np.asarray(multirocket_test_by_seed, dtype=float)
    structure_train = np.asarray(structural_train, dtype=float).reshape(-1)
    structure_test = np.asarray(structural_test, dtype=float).reshape(-1)
    change_train = np.asarray(change_point_train, dtype=float).reshape(-1)
    change_test = np.asarray(change_point_test, dtype=float).reshape(-1)
    if multi_train.ndim != 2 or multi_test.ndim != 2 or multi_train.shape[0] != 2 or multi_test.shape[0] != 2:
        raise ValueError("MultiROCKET score arrays must contain two seed rows")
    if len({multi_train.shape[1], len(structure_train), len(change_train)}) != 1:
        raise ValueError("TRAIN score length mismatch")
    if len({multi_test.shape[1], len(structure_test), len(change_test)}) != 1:
        raise ValueError("TEST score length mismatch")
    candidates = sorted(_clean_indices(candidate_indices, multi_test.shape[1]))
    if not candidates:
        return {}
    counts = {index: 0 for index in candidates}
    for replicate in range(int(replicates)):
        sampled = _bootstrap_indices(dataset, replicate, multi_train.shape[1], seed)
        multi_flags = np.logical_and.reduce([
            empirical_upper_pvalues(multi_train[seed_index, sampled], multi_test[seed_index]) <= float(alpha)
            for seed_index in range(2)
        ])
        structure_flags = empirical_upper_pvalues(structure_train[sampled], structure_test) <= float(alpha)
        change_flags = empirical_upper_pvalues(change_train[sampled], change_test) <= float(alpha)
        consensus = multi_flags & structure_flags & change_flags
        for index in candidates:
            counts[index] += int(consensus[index])
    return {index: counts[index] / int(replicates) for index in candidates}


def _clean_indices(values: Iterable[int], n_test: int) -> set[int]:
    selected = {int(value) for value in values}
    if selected and (min(selected) < 0 or max(selected) >= int(n_test)):
        raise ValueError("candidate outside TEST range")
    return selected


def build_routing_row(
    *,
    dataset: str,
    n_train: int,
    n_test: int,
    current_hard_indices: Iterable[int],
    general_candidates: Iterable[int],
    automatic_candidates: Iterable[int],
    priority_candidates: Iterable[int],
    automatic_stability: Mapping[int, float],
    priority_stability: Mapping[int, float],
    automatic_risk_contract_pass: bool,
) -> dict[str, object]:
    hard = _clean_indices(current_hard_indices, n_test)
    general_source = _clean_indices(general_candidates, n_test) - hard
    automatic_source = _clean_indices(automatic_candidates, n_test) - hard
    priority_source = _clean_indices(priority_candidates, n_test) - hard
    automatic = {
        index
        for index in automatic_source
        if automatic_risk_contract_pass
        and float(automatic_stability.get(index, 0.0)) >= STABILITY_MINIMUM
    }
    priority = priority_source - automatic
    general = general_source - automatic - priority
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset": str(dataset),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "current_hard_indices": sorted(hard),
        "automatic_source_indices": sorted(automatic_source),
        "priority_source_indices": sorted(priority_source),
        "general_source_indices": sorted(general_source),
        "automatic_qualification_indices": sorted(automatic),
        "priority_review_indices": sorted(priority),
        "general_review_indices": sorted(general),
        "automatic_stability": {str(key): float(value) for key, value in sorted(automatic_stability.items())},
        "priority_stability": {str(key): float(value) for key, value in sorted(priority_stability.items())},
        "automatic_risk_contract_pass": bool(automatic_risk_contract_pass),
        "automatic_alpha": AUTOMATIC_ALPHA,
        "priority_alpha": PRIORITY_ALPHA,
        "general_alpha": GENERAL_ALPHA,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "stability_minimum": STABILITY_MINIMUM,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "uses_family_name": False,
        "uses_dataset_name_for_policy": False,
        "uses_test_length_for_policy": False,
        "autonomous_policy_changed": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }


def _metrics(selected: set[int], truth: set[int]) -> dict[str, float | int]:
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision = tp / len(selected) if selected else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"count": len(selected), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_predictions(
    predictions: list[dict[str, object]], labels_by_dataset: Mapping[str, np.ndarray]
) -> list[dict[str, object]]:
    evaluated = []
    for prediction in predictions:
        dataset = str(prediction["dataset"])
        labels = np.asarray(labels_by_dataset[dataset], dtype=np.int8).reshape(-1)
        if len(labels) != int(prediction["n_test"]):
            raise ValueError(f"TEST label length mismatch: {dataset}")
        truth = set(np.flatnonzero(labels == 1).astype(int))
        views = {
            "baseline_hard": set(prediction["current_hard_indices"]),
            "automatic_qualification": set(prediction["automatic_qualification_indices"]),
            "priority_review": set(prediction["priority_review_indices"]),
            "general_review": set(prediction["general_review_indices"]),
        }
        views["counterfactual_hard_plus_qualification"] = views["baseline_hard"] | views["automatic_qualification"]
        row = dict(prediction)
        row["true_anomaly_count"] = len(truth)
        for prefix, selected in views.items():
            row.update({f"{prefix}_{key}": value for key, value in _metrics(selected, truth).items()})
        evaluated.append(row)
    return evaluated


def summarize_evaluated(evaluated: list[dict[str, object]]) -> list[dict[str, object]]:
    if not evaluated:
        return []
    prefixes = (
        "baseline_hard", "automatic_qualification", "priority_review", "general_review",
        "counterfactual_hard_plus_qualification",
    )
    summary: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "datasets": len(evaluated),
        "automatic_risk_contract_pass": all(bool(row["automatic_risk_contract_pass"]) for row in evaluated),
        "autonomous_policy_changed": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }
    for prefix in prefixes:
        count = sum(int(row[f"{prefix}_count"]) for row in evaluated)
        tp = sum(int(row[f"{prefix}_tp"]) for row in evaluated)
        fp = sum(int(row[f"{prefix}_fp"]) for row in evaluated)
        summary.update({
            f"{prefix}_count": count,
            f"{prefix}_tp": tp,
            f"{prefix}_fp": fp,
            f"{prefix}_precision": tp / count if count else 0.0,
            f"{prefix}_mean_f1": float(np.mean([float(row[f"{prefix}_f1"]) for row in evaluated])),
        })
    return [summary]
