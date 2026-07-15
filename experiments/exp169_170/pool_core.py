from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from experiments.exp169_170.oob_core import robust_model_scale

ALPHAS = (0.005, 0.01, 0.02, 0.05)
SOURCES = ("spectral", "shapelet", "interval")
SOURCE_PAIRS = {
    "spectral+shapelet": ("spectral", "shapelet"),
    "spectral+interval": ("spectral", "interval"),
    "shapelet+interval": ("shapelet", "interval"),
}
SOURCE_CONFIGS = (*SOURCE_PAIRS, "2of3", "3of3")
POLICIES = ("similarity_pool", "coarse_strata_pool", "global_balanced_pool")


@dataclass(frozen=True)
class DonorPool:
    donors: list[str]
    references: dict[str, np.ndarray]  # source -> concatenated OOB scores
    per_donor_counts: dict[str, dict[str, int]]  # source -> donor -> count
    expansion_dc: int = 0
    expansion_dq: int = 0


def build_descriptor(
    n_train: int,
    target_length: float | int,
    oob_scores: dict[str, np.ndarray],
) -> np.ndarray:
    if int(n_train) < 1 or not np.isfinite(target_length) or float(target_length) <= 0:
        raise ValueError("descriptor count and target length must be positive")
    features = []
    features.append(math.log(n_train))
    features.append(math.log(target_length))
    for source in SOURCES:
        scores = np.asarray(oob_scores[source], dtype=np.float64)
        if scores.size == 0:
            raise ValueError(f"OOB scores for {source} cannot be empty")
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError(f"OOB scores for {source} contain non-finite values")
        q10, q25, q50, q75, q90 = np.percentile(scores, [10, 25, 50, 75, 90])
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        features.extend([q10, q25, q50, q75, q90, mean, std])
    return np.array(features, dtype=np.float64)


def build_descriptors(packs: dict[str, dict]) -> dict[str, np.ndarray]:
    descriptors = {}
    for d, p in packs.items():
        desc = build_descriptor(
            n_train=p["n_train"],
            target_length=p["target_length"],
            oob_scores=p["oob_scores"]
        )
        descriptors[d] = desc

    datasets = sorted(descriptors.keys())
    matrix = np.vstack([descriptors[d] for d in datasets])

    medians = np.median(matrix, axis=0)
    q75, q25 = np.percentile(matrix, [75, 25], axis=0)
    iqr = q75 - q25
    scales = np.zeros(23)
    for col in range(23):
        scale = iqr[col]
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.std(matrix[:, col])
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        scales[col] = scale

    scaled_matrix = (matrix - medians) / scales

    scaled_descriptors = {}
    for idx, d in enumerate(datasets):
        scaled_descriptors[d] = scaled_matrix[idx]

    return scaled_descriptors


def c_idx(n_train: int) -> int:
    if n_train < 8:
        return 0
    elif n_train < 12:
        return 1
    elif n_train < 16:
        return 2
    else:
        return 3


def package_pool(
    donors: list[str],
    packs: dict[str, dict],
    donor_cap: int,
    expansion_dc: int = 0,
    expansion_dq: int = 0,
) -> DonorPool:
    donors = sorted(donors)
    if len(donors) != len(set(donors)):
        raise ValueError("donor list contains duplicates")
    if donor_cap < 1:
        raise ValueError("donor_cap must be positive")
    references = {}
    per_donor_counts = {}

    for source in SOURCES:
        ref_list = []
        counts = {}
        for d in donors:
            scores = np.asarray(packs[d]["oob_scores"][source], dtype=np.float64)
            if scores.ndim != 1 or scores.size == 0:
                raise ValueError(f"invalid OOB scores for donor {d} source {source}")
            if not np.isfinite(scores).all():
                raise ValueError(f"non-finite OOB scores for donor {d} source {source}")
            contrib = scores[:donor_cap]
            ref_list.append(contrib)
            counts[d] = len(contrib)
        references[source] = np.concatenate(ref_list) if ref_list else np.empty(0, dtype=np.float64)
        per_donor_counts[source] = counts

    return DonorPool(
        donors=donors,
        references=references,
        per_donor_counts=per_donor_counts,
        expansion_dc=expansion_dc,
        expansion_dq=expansion_dq,
    )


def _reference_size(
    donors: list[str],
    packs: dict[str, dict],
    source: str,
    donor_cap: int,
) -> int:
    return sum(
        min(np.asarray(packs[d]["oob_scores"][source]).size, donor_cap)
        for d in donors
    )


def _meets_reference_contract(
    donors: list[str],
    packs: dict[str, dict],
    minimum_reference: int,
    donor_cap: int,
) -> bool:
    return all(
        _reference_size(donors, packs, source, donor_cap) >= minimum_reference
        for source in SOURCES
    )


def build_pool_manifest(
    dataset: str,
    packs: dict[str, dict],
    minimum_reference: int = 200,
    donor_cap: int = 20,
    scaled_descriptors: dict[str, np.ndarray] | None = None,
) -> dict[str, DonorPool]:
    if dataset not in packs:
        raise ValueError(f"Target dataset {dataset} not found in packs")

    other_datasets = [d for d in packs if d != dataset]
    if not other_datasets:
        raise ValueError("No eligible donors available")

    # 1. Similarity pool
    scaled_descs = scaled_descriptors if scaled_descriptors is not None else build_descriptors(packs)
    t_desc = scaled_descs[dataset]

    distances = {}
    for d in other_datasets:
        distances[d] = np.linalg.norm(scaled_descs[d] - t_desc)

    selected_similarity_donors = []
    found_similarity = False
    ordered_distances = sorted((float(distance), donor) for donor, distance in distances.items())
    position = 0
    while position < len(ordered_distances):
        shell_distance = ordered_distances[position][0]
        shell = []
        while position < len(ordered_distances) and math.isclose(
            ordered_distances[position][0],
            shell_distance,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            shell.append(ordered_distances[position][1])
            position += 1
        selected_similarity_donors.extend(shell)

        if _meets_reference_contract(
            selected_similarity_donors,
            packs,
            minimum_reference,
            donor_cap,
        ):
            found_similarity = True
            break

    if not found_similarity:
        raise ValueError(f"Similarity pool for {dataset} could not meet minimum reference of {minimum_reference}")

    similarity_pool = package_pool(selected_similarity_donors, packs, donor_cap)

    # 2. Coarse-strata pool
    lengths = [packs[d]["target_length"] for d in packs]
    p25, p50, p75 = np.percentile(lengths, [25, 50, 75])

    def q_idx(length: float) -> int:
        if length <= p25:
            return 0
        elif length <= p50:
            return 1
        elif length <= p75:
            return 2
        else:
            return 3

    target_c = c_idx(packs[dataset]["n_train"])
    target_q = q_idx(packs[dataset]["target_length"])

    found_coarse = False
    selected_coarse_donors = []
    final_dc = 0
    final_dq = 0
    for dc in range(4):
        for dq in range(4):
            c_set = {c for c in range(4) if abs(c - target_c) <= dc}
            q_set = {q for q in range(4) if abs(q - target_q) <= dq}
            donors = [
                d for d in other_datasets
                if c_idx(packs[d]["n_train"]) in c_set and q_idx(packs[d]["target_length"]) in q_set
            ]

            if _meets_reference_contract(donors, packs, minimum_reference, donor_cap):
                selected_coarse_donors = donors
                final_dc = dc
                final_dq = dq
                found_coarse = True
                break
        if found_coarse:
            break

    if not found_coarse:
        raise ValueError(f"Coarse-strata pool for {dataset} could not meet minimum reference of {minimum_reference}")

    coarse_pool = package_pool(
        selected_coarse_donors,
        packs,
        donor_cap,
        expansion_dc=final_dc,
        expansion_dq=final_dq,
    )

    # 3. Global-balanced pool
    selected_global_donors = sorted(other_datasets)
    if not _meets_reference_contract(
        selected_global_donors,
        packs,
        minimum_reference,
        donor_cap,
    ):
        raise ValueError(
            f"Global-balanced pool for {dataset} could not meet minimum reference of {minimum_reference}"
        )

    global_pool = package_pool(selected_global_donors, packs, donor_cap)

    return {
        "similarity_pool": similarity_pool,
        "coarse_strata_pool": coarse_pool,
        "global_balanced_pool": global_pool,
    }


def pooled_p_values(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.size == 0:
        raise ValueError("reference pool cannot be empty")
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError("non-finite values in query or reference")

    exceedance = (reference[:, None] >= query).sum(axis=0)
    return (1.0 + exceedance) / (reference.size + 1.0)


def assert_label_free_prediction(row: dict) -> None:
    forbidden = {
        "y_test",
        "test_labels",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "hard_tp",
        "hard_fp",
        "hard_f1",
        "combined_f1",
    }
    overlap = forbidden & set(row)
    if overlap:
        raise ValueError(f"prediction contains evaluation fields: {sorted(overlap)}")
    if row.get("contains_test_labels") is not False:
        raise ValueError("prediction must declare contains_test_labels=false")
    if row.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction must declare contains_evaluation_metrics=false")
    indices = [int(value) for value in row.get("candidate_indices", [])]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate candidate index")
    if any(value < 0 or value >= int(row["n_test"]) for value in indices):
        raise ValueError("candidate index out of TEST range")
    if len(indices) != int(row["candidate_count"]):
        raise ValueError("candidate count mismatch")


def fixed_candidate_rows(
    dataset: str,
    n_test: int,
    p_values: dict[str, dict[str, np.ndarray]],
    config_hash: str,
    n_train: int = 10,
    reference_counts: dict[str, dict[str, int]] | None = None,
) -> list[dict]:
    if reference_counts is None:
        reference_counts = {
            policy: {source: 200 for source in SOURCES}
            for policy in POLICIES
        }

    rows = []
    for policy in POLICIES:
        policy_p = p_values[policy]
        policy_ref = reference_counts[policy]

        source_p = {
            source: np.asarray(policy_p[source], dtype=np.float64)
            for source in SOURCES
        }
        if any(values.shape != (n_test,) for values in source_p.values()):
            raise ValueError("source p-value shape mismatch")
        if any(not np.isfinite(values).all() for values in source_p.values()):
            raise ValueError("source p-values must be finite")

        for alpha in ALPHAS:
            flags = {source: source_p[source] <= alpha for source in SOURCES}
            source_count = np.vstack([flags[source] for source in SOURCES]).sum(axis=0)
            config_flags = {
                name: flags[sources[0]] & flags[sources[1]]
                for name, sources in SOURCE_PAIRS.items()
            }
            config_flags["2of3"] = source_count >= 2
            config_flags["3of3"] = source_count >= 3

            for source_config in SOURCE_CONFIGS:
                indices = np.flatnonzero(config_flags[source_config]).astype(int).tolist()

                config_sources = SOURCE_PAIRS.get(source_config, SOURCES)
                min_ref = min(policy_ref[src] for src in config_sources)
                res_blocked = float(alpha) < 1.0 / (min_ref + 1.0)

                row = {
                    "dataset": dataset,
                    "pool_policy": policy,
                    "n_train": int(n_train),
                    "n_test": int(n_test),
                    "alpha": float(alpha),
                    "source_config": source_config,
                    "candidate_indices": indices,
                    "candidate_count": len(indices),
                    "source_candidate_counts": {
                        source: int(flags[source].sum()) for source in SOURCES
                    },
                    "source_reference_counts": {
                        source: int(policy_ref[source]) for source in SOURCES
                    },
                    "source_minimum_p_values": {
                        source: 1.0 / (int(policy_ref[source]) + 1.0) for source in SOURCES
                    },
                    "minimum_p_value": 1.0 / (min_ref + 1.0),
                    "resolution_blocked": res_blocked,
                    "config_hash": config_hash,
                    "lane": "research_review_candidate",
                    "contains_test_labels": False,
                    "contains_evaluation_metrics": False,
                    "routing_uses_family_name": 0,
                    "routing_uses_test_length": 0,
                    "routing_uses_test_labels": 0,
                    "routing_uses_test_positions": 0,
                }
                assert_label_free_prediction(row)
                rows.append(row)
    return rows
