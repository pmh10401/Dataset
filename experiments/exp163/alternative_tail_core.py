from __future__ import annotations

import numpy as np


ALPHAS = (0.005, 0.01, 0.02, 0.05)
VARIANTS = ("tail_rank_2of3", "tail_rank_3of3")
SOURCE_NAMES = ("spectral", "shapelet", "interval")
SOURCE_DISPLAY_NAMES = {
    "spectral": "미분·스펙트럼 통계 점수",
    "shapelet": "부분 형태 일치 점수",
    "interval": "구간 분포 점수",
}


def calculate_quantile_threshold(train_scores: np.ndarray, alpha: float) -> float:
    """
    Calculate threshold using 1-alpha empirical quantile over finite train scores.
    Uses method="higher".
    """
    finite_train_scores = train_scores[np.isfinite(train_scores)]
    if len(finite_train_scores) == 0:
        raise ValueError("No finite train scores available for threshold calculation")
    q = 1.0 - alpha
    return float(np.quantile(finite_train_scores, q, method="higher"))


def find_tail_candidates(test_scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Find indices where test_scores strictly exceed threshold.
    """
    # Strict greater-than is required, so no >=
    values = np.asarray(test_scores, dtype=float)
    indices = np.where(np.isfinite(values) & (values > threshold))[0]
    return indices


def is_candidate_zero_target(n_train: int, baseline_candidate_count: int) -> bool:
    """Return the preregistered label-free target gate."""

    return int(n_train) >= 5 and int(baseline_candidate_count) == 0


def compute_agreement(
    spectral_indices: set[int],
    shapelet_indices: set[int],
    interval_indices: set[int],
    variant: str,
) -> set[int]:
    """
    Combine tail candidates from 3 alternative sources using 2-of-3 or 3-of-3 agreement.
    """
    if variant == "tail_rank_2of3":
        # At least 2 of 3 sources must agree on the index
        agreement = set()
        # Find union
        all_indices = spectral_indices | shapelet_indices | interval_indices
        for idx in all_indices:
            count = 0
            if idx in spectral_indices:
                count += 1
            if idx in shapelet_indices:
                count += 1
            if idx in interval_indices:
                count += 1
            if count >= 2:
                agreement.add(idx)
        return agreement
    elif variant == "tail_rank_3of3":
        # All 3 sources must agree on the index
        return spectral_indices & shapelet_indices & interval_indices
    else:
        raise ValueError(f"Unknown variant type: {variant}")


def integrate_lanes(
    hard_indices: set[int],
    priority_indices: set[int],
    base_standard_indices: set[int],
    new_agreement_indices: set[int],
) -> tuple[set[int], set[int], set[int]]:
    """
    Integrate new tail agreement candidates into the Standard review lane,
    while fully preserving the baseline Hard and Priority lanes.
    """
    # Standard review: existing standard review + new agreement candidates
    # minus any indices that are already in Hard or Priority.
    hard = {int(index) for index in hard_indices}
    priority = {int(index) for index in priority_indices}
    standard = {int(index) for index in base_standard_indices}
    agreement = {int(index) for index in new_agreement_indices}
    new_standard = (standard | agreement) - hard - priority
    return hard, priority, new_standard


def validate_lane_contract(
    *,
    hard: set[int],
    standard: set[int],
    priority: set[int],
    candidate: set[int],
    n_test: int,
) -> None:
    """Validate disjoint lanes, their union, and TEST index bounds."""

    if hard & standard or hard & priority or standard & priority:
        raise ValueError("lane overlap")
    if candidate != hard | standard | priority:
        raise ValueError("candidate/lane union mismatch")
    invalid = sorted(index for index in candidate if index < 0 or index >= int(n_test))
    if invalid:
        raise ValueError(f"candidate index range error: {invalid[:10]}")


def parse_indices(value: object) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {int(item) for item in value}
    text = str(value).replace(",", " ").strip()
    return {int(token) for token in text.split() if token}


def format_indices(indices: set[int]) -> str:
    return " ".join(str(index) for index in sorted({int(i) for i in indices}))


def build_prediction_row(
    *,
    baseline_row: dict,
    scores: dict[str, np.ndarray],
    alpha: float,
    variant: str,
) -> dict:
    """Build one label-free Exp163 prediction row."""

    if float(alpha) not in ALPHAS:
        raise ValueError(f"alpha is not preregistered: {alpha}")
    if variant not in VARIANTS:
        raise ValueError(f"variant is not preregistered: {variant}")

    train_lengths = {len(np.asarray(scores[f"{source}_train"])) for source in SOURCE_NAMES}
    test_lengths = {len(np.asarray(scores[f"{source}_test"])) for source in SOURCE_NAMES}
    if len(train_lengths) != 1 or len(test_lengths) != 1:
        raise ValueError("source score row count mismatch")
    n_train = train_lengths.pop()
    n_test = test_lengths.pop()
    if int(baseline_row["n_train"]) != n_train or int(baseline_row["n_test"]) != n_test:
        raise ValueError("Exp160/cache TRAIN or TEST row count mismatch")

    hard = parse_indices(baseline_row.get("hard_indices"))
    standard = parse_indices(baseline_row.get("standard_review_indices"))
    priority = parse_indices(baseline_row.get("priority_review_indices"))
    baseline_candidate = parse_indices(baseline_row.get("candidate_indices"))
    validate_lane_contract(
        hard=hard,
        standard=standard,
        priority=priority,
        candidate=baseline_candidate,
        n_test=n_test,
    )
    if len(baseline_candidate) != int(baseline_row["candidate_count"]):
        raise ValueError("Exp160 candidate count mismatch")

    target = is_candidate_zero_target(n_train, len(baseline_candidate))
    thresholds: dict[str, float] = {}
    source_candidates: dict[str, list[int]] = {source: [] for source in SOURCE_NAMES}
    agreement: set[int] = set()
    if target:
        source_sets: dict[str, set[int]] = {}
        for source in SOURCE_NAMES:
            threshold = calculate_quantile_threshold(scores[f"{source}_train"], alpha)
            candidates = {
                int(index)
                for index in find_tail_candidates(scores[f"{source}_test"], threshold)
            }
            thresholds[source] = threshold
            source_sets[source] = candidates
            source_candidates[source] = sorted(candidates)
        agreement = compute_agreement(
            source_sets["spectral"],
            source_sets["shapelet"],
            source_sets["interval"],
            variant,
        )

    new_hard, new_priority, new_standard = integrate_lanes(
        hard, priority, standard, agreement
    )
    candidate = new_hard | new_standard | new_priority
    validate_lane_contract(
        hard=new_hard,
        standard=new_standard,
        priority=new_priority,
        candidate=candidate,
        n_test=n_test,
    )
    if new_hard != hard or new_priority != priority:
        raise ValueError("Exp160 Hard/Priority baseline invariance failed")
    added_standard = new_standard - standard

    return {
        "experiment_id": "experiment_163_alternative_source_tail_agreement_review",
        "dataset_name": str(baseline_row["dataset_name"]),
        "alpha": float(alpha),
        "variant": variant,
        "n_train": n_train,
        "n_test": n_test,
        "baseline_candidate_count": len(baseline_candidate),
        "candidate_zero_target": int(target),
        "candidate_empty_gate_applied": int(target),
        "abstained": int(n_train < 5),
        "calibration_status": (
            "complete_calibration_abstention"
            if n_train < 5
            else "alternative_tail_agreement_evaluated"
            if target
            else "not_target_baseline_has_candidates"
        ),
        "source_display_names": dict(SOURCE_DISPLAY_NAMES),
        "threshold_method": "train_empirical_quantile_higher_strict_gt",
        "source_thresholds": thresholds,
        "source_candidate_indices": source_candidates,
        "source_candidate_counts": {
            source: len(source_candidates[source]) for source in SOURCE_NAMES
        },
        "agreement_indices": format_indices(agreement),
        "agreement_count": len(agreement),
        "added_standard_indices": format_indices(added_standard),
        "added_standard_count": len(added_standard),
        "hard_indices": format_indices(new_hard),
        "hard_count": len(new_hard),
        "standard_review_indices": format_indices(new_standard),
        "standard_review_count": len(new_standard),
        "priority_review_indices": format_indices(new_priority),
        "priority_review_count": len(new_priority),
        "candidate_indices": format_indices(candidate),
        "candidate_count": len(candidate),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "uses_test_candidate_empty_gate": 1,
        "uses_test_length_for_policy": 0,
        "uses_family_name": 0,
        "uses_dataset_name_for_policy": 0,
        "retrospective_counterfactual": 1,
        "prospective_validated": 0,
    }


def _tier_metrics(labels: np.ndarray, indices: set[int]) -> dict[str, float | int]:
    selected = sorted(indices)
    if any(index < 0 or index >= len(labels) for index in selected):
        raise ValueError("metric index out of range")
    tp = int(np.sum(labels[selected])) if selected else 0
    fp = len(selected) - tp
    total_positive = int(np.sum(labels))
    fn = total_positive - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "count": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_prediction_rows(predictions: list[dict], label_loader) -> list[dict]:
    """Evaluate frozen rows after prediction; labels never enter prediction code."""

    labels_cache: dict[str, np.ndarray] = {}
    results = []
    for prediction in predictions:
        dataset = str(prediction["dataset_name"])
        if dataset not in labels_cache:
            labels_cache[dataset] = np.asarray(label_loader(dataset), dtype=int)
        labels = labels_cache[dataset]
        if len(labels) != int(prediction["n_test"]):
            raise ValueError(f"label length mismatch dataset={dataset}")
        hard = parse_indices(prediction.get("hard_indices"))
        standard = parse_indices(prediction.get("standard_review_indices"))
        priority = parse_indices(prediction.get("priority_review_indices"))
        candidate = parse_indices(prediction.get("candidate_indices"))
        added = parse_indices(prediction.get("added_standard_indices"))
        validate_lane_contract(
            hard=hard,
            standard=standard,
            priority=priority,
            candidate=candidate,
            n_test=len(labels),
        )
        if not added <= standard:
            raise ValueError(f"added Standard indices not in Standard lane dataset={dataset}")
        evaluated = dict(prediction)
        for prefix, indices in (
            ("hard", hard),
            ("standard_review", standard),
            ("priority_review", priority),
            ("added_standard", added),
            ("combined", candidate),
        ):
            metrics = _tier_metrics(labels, indices)
            evaluated.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        evaluated["mean_combined_f1_scope"] = "human_assisted_diagnostic_only"
        results.append(evaluated)
    return results
