from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from statistics import NormalDist

import numpy as np

from experiments.exp293_294_train_only_moe_lite.routing import (
    GATE_VARIANTS,
    PRIMARY_EXPERTS,
    gate_experts,
)


def fast_loo_upper_pvalues(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=float).reshape(-1)
    if len(scores) == 0:
        return np.asarray([], dtype=float)
    ordered = np.sort(scores)
    first_equal = np.searchsorted(ordered, scores, side="left")
    return (len(scores) - first_equal) / len(scores)


def deterministic_bootstrap_indices(
    dataset: str, *, replicate: int, n_train: int, seed: int
) -> np.ndarray:
    material = f"{int(seed)}:{dataset}:{int(replicate)}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = np.random.default_rng(local_seed)
    return rng.integers(0, int(n_train), size=int(n_train), dtype=np.int64)


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if len(left_rank) < 2 or float(np.std(left_rank)) <= 1e-12 or float(np.std(right_rank)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return value if np.isfinite(value) else 0.0


def _wilson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return 1.0
    p = successes / trials
    z = NormalDist().inv_cdf(confidence)
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return float((center + radius) / denominator)


def _audit_resampled_expert(
    expert: str,
    train_variants: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    alpha: float,
) -> dict[str, object]:
    arrays = {
        name: np.asarray(values, dtype=float).reshape(-1)[indices]
        for name, values in train_variants.items()
    }
    names = sorted(arrays)
    n_train = len(indices)
    votes = np.sum(
        np.vstack([fast_loo_upper_pvalues(arrays[name]) <= float(alpha) for name in names]),
        axis=0,
    )
    false_count = int(np.count_nonzero(votes >= math.ceil(len(names) / 2)))
    correlations = [
        abs(_spearman(arrays[left], arrays[right]))
        for position, left in enumerate(names)
        for right in names[position + 1 :]
    ]
    return {
        "expert": expert,
        "available": True,
        "provenance_verified": True,
        "n_train": n_train,
        "risk_upper_95": _wilson_upper(false_count, n_train),
        "stability": float(np.median(correlations)) if correlations else 1.0,
        "attainable_p": 1.0 / n_train,
    }


def _consensus_tier(selected_support_counts: list[int]) -> str:
    supporting = sum(count >= 1 for count in selected_support_counts)
    high = sum(count == 2 for count in selected_support_counts)
    if high == len(GATE_VARIANTS):
        return "unanimous_high"
    if supporting == len(GATE_VARIANTS):
        return "unanimous_supported"
    if supporting > 0:
        return "partial_supported"
    return "no_selected_support"


def bootstrap_alert_stability(
    *,
    dataset: str,
    baseline_tiers: Mapping[int, str],
    supporting_experts_by_index: Mapping[int, set[str]],
    train_scores_by_expert: Mapping[str, Mapping[str, np.ndarray]],
    replicates: int,
    seed: int,
    alpha: float,
) -> dict[int, dict[str, object]]:
    lengths = set()
    valid = set(train_scores_by_expert) == set(PRIMARY_EXPERTS)
    for expert in PRIMARY_EXPERTS:
        variants = train_scores_by_expert.get(expert, {})
        variant_lengths = {len(np.asarray(values).reshape(-1)) for values in variants.values()}
        valid = valid and bool(variants) and len(variant_lengths) == 1
        if len(variant_lengths) == 1:
            lengths.update(variant_lengths)
    valid = valid and len(lengths) == 1
    n_train = next(iter(lengths), 0) if len(lengths) == 1 else 0
    valid = valid and n_train >= 5 and int(replicates) > 0
    if not valid:
        return {
            int(index): {
                "bootstrap_eligible": False,
                "bootstrap_replicates": 0,
                "tier_match_count": 0,
                "stability_rate": None,
                "bootstrap_tier_counts": {},
            }
            for index in baseline_tiers
        }

    tier_counts = {int(index): Counter() for index in baseline_tiers}
    for replicate in range(int(replicates)):
        indices = deterministic_bootstrap_indices(
            dataset, replicate=replicate, n_train=n_train, seed=seed
        )
        audits = {
            expert: _audit_resampled_expert(
                expert,
                train_scores_by_expert[expert],
                indices,
                alpha=alpha,
            )
            for expert in PRIMARY_EXPERTS
        }
        selected_by_gate = [
            set(gate_experts(audits, variant=variant)[:2])
            for variant in GATE_VARIANTS
        ]
        for index in baseline_tiers:
            support = set(supporting_experts_by_index[int(index)])
            counts = [len(support & selected) for selected in selected_by_gate]
            tier_counts[int(index)][_consensus_tier(counts)] += 1

    return {
        int(index): {
            "bootstrap_eligible": True,
            "bootstrap_replicates": int(replicates),
            "tier_match_count": int(tier_counts[int(index)][str(baseline_tier)]),
            "stability_rate": tier_counts[int(index)][str(baseline_tier)] / int(replicates),
            "bootstrap_tier_counts": dict(sorted(tier_counts[int(index)].items())),
        }
        for index, baseline_tier in baseline_tiers.items()
    }
