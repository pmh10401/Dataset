from dataclasses import dataclass
from numba import njit
import numpy as np
from sklearn.metrics import roc_auc_score

from .dataset_repository import PreparedDatasetValues
from .score_sources import METHOD_NAMES, ScoreResult, score_method

POLICY_IDS = ("single_minirocket", "three_method_consensus", "train_feature_router")

ALPHA = 0.005
FIXED_SEEDS = (20260721, 20261721)


def _protected_float_array(values: np.ndarray, field: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{field} must be finite and one-dimensional")
    return np.frombuffer(result.tobytes(), dtype=float)


@njit(cache=True)
def _empirical_upper_pvalues_numba(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    result = np.empty(len(query), dtype=np.float64)
    for index in range(len(query)):
        count = 0
        for value in reference:
            if value >= query[index]:
                count += 1
        result[index] = (1.0 + count) / (len(reference) + 1.0)
    return result


def empirical_upper_pvalues(train_oof: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    reference = _protected_float_array(train_oof, "TRAIN OOF reference")
    query = _protected_float_array(test_scores, "TEST scores")
    if len(reference) == 0:
        raise ValueError("TRAIN OOF reference must be non-empty")
    return _empirical_upper_pvalues_numba(reference, query)


def stable_descending_indices(scores: np.ndarray) -> tuple[int, ...]:
    values = _protected_float_array(scores, "scores")
    return tuple(np.argsort(-values, kind="mergesort").astype(int))


@dataclass(frozen=True, slots=True)
class MethodDetection:
    method_name: str
    train_scores: np.ndarray
    test_scores: np.ndarray
    final_pvalues: np.ndarray
    seed_scores: tuple[ScoreResult, ...]
    seed_pvalues: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        for field in ("train_scores", "test_scores", "final_pvalues"):
            object.__setattr__(self, field, _protected_float_array(getattr(self, field), field))
        object.__setattr__(
            self,
            "seed_pvalues",
            tuple(_protected_float_array(values, "seed_pvalues") for values in self.seed_pvalues),
        )


def run_method_detection(
    prepared: PreparedDatasetValues,
    method_name: str,
    *,
    scorer=score_method,
) -> MethodDetection:
    scored = tuple(
        scorer(prepared.train_values, prepared.test_values, method_name, seed)
        for seed in FIXED_SEEDS
    )
    train_scores = np.median(np.stack([result.train_oof for result in scored]), axis=0)
    test_scores = np.median(np.stack([result.test_scores for result in scored]), axis=0)
    return MethodDetection(
        method_name=method_name,
        train_scores=train_scores,
        test_scores=test_scores,
        final_pvalues=empirical_upper_pvalues(train_scores, test_scores),
        seed_scores=scored,
        seed_pvalues=tuple(
            empirical_upper_pvalues(result.train_oof, result.test_scores)
            for result in scored
        ),
    )


@dataclass(frozen=True)
class DetectionResult:
    method_name: str
    rows: tuple[dict[str, object], ...]
    automatic_anomaly_indices: tuple[int, ...]
    train_reference: dict[str, list[float]]
    target_length: int
    resolution_limited_warning_indices: tuple[int, ...] = ()
    policy_id: str | None = None
    selected_method: str | None = None
    seed_consensus_review_indices: tuple[int, ...] = ()
    method_consensus_review_indices: tuple[int, ...] = ()
    review_indices: tuple[int, ...] = ()


def _train_reference(prepared: PreparedDatasetValues) -> dict[str, list[float]]:
    return {
        "median": np.median(prepared.train_values, axis=0).astype(float).tolist(),
        "q10": np.quantile(prepared.train_values, 0.10, axis=0).astype(float).tolist(),
        "q90": np.quantile(prepared.train_values, 0.90, axis=0).astype(float).tolist(),
    }


def run_analysis_policy(
    prepared: PreparedDatasetValues,
    policy_id: str,
    router=None,
    scorer=score_method,
) -> DetectionResult:
    if policy_id not in POLICY_IDS:
        raise ValueError("unknown analysis policy")
    if policy_id == "single_minirocket":
        selected_method = METHOD_NAMES[0]
        methods = (selected_method,)
    elif policy_id == "three_method_consensus":
        selected_method = None
        methods = METHOD_NAMES
    else:
        from .routing_policy import TrainFeatureRouter, extract_train_features

        if not isinstance(router, TrainFeatureRouter):
            raise ValueError("train_feature_router requires a fitted router")
        if prepared.raw_train_rows is None:
            raise ValueError("train_feature_router requires authoritative raw TRAIN rows")
        selected_method = router.route(extract_train_features(prepared.raw_train_rows)).selected_method
        methods = (selected_method,)

    detections = {
        method: run_method_detection(prepared, method, scorer=scorer) for method in methods
    }
    expected = len(prepared.test_indices)
    for detection in detections.values():
        arrays = (detection.test_scores, detection.final_pvalues, *detection.seed_pvalues)
        if (
            len(detection.train_scores) != len(prepared.train_values)
            or any(len(values) != expected or not np.isfinite(values).all() for values in arrays)
        ):
            raise ValueError("method results must be finite and aligned to prepared TEST indices")

    attainable = 1.0 / (len(prepared.train_values) + 1.0)
    review_enabled = ALPHA + 1e-15 < attainable
    automatic_positions = set()
    seed_positions = set()
    method_positions = set()
    pvalues = np.empty(expected, dtype=float)
    supporting = []
    for position in range(expected):
        final = {method: float(detections[method].final_pvalues[position]) for method in methods}
        seed_support = {
            method
            for method in methods
            if review_enabled
            and all(abs(float(values[position]) - attainable) <= 1e-15
                    for values in detections[method].seed_pvalues)
        }
        method_support = {
            method for method, value in final.items()
            if review_enabled and abs(value - attainable) <= 1e-15
        }
        automatic_support = {method for method, value in final.items() if value <= ALPHA}
        is_automatic = (
            len(automatic_support) >= 2
            if policy_id == "three_method_consensus"
            else bool(automatic_support)
        )
        if is_automatic:
            automatic_positions.add(position)
        else:
            if seed_support:
                seed_positions.add(position)
            if policy_id == "three_method_consensus" and len(method_support) >= 2:
                method_positions.add(position)
        pvalues[position] = (
            sorted(final.values())[1]
            if policy_id == "three_method_consensus"
            else final[selected_method]
        )
        evidence_support = automatic_support | seed_support
        if policy_id == "three_method_consensus" and len(method_support) >= 2:
            evidence_support |= method_support
        supporting.append(tuple(method for method in methods if method in evidence_support))

    scores = 1.0 - pvalues
    order = stable_descending_indices(scores)
    rows = tuple(
        {
            "series_index": prepared.test_indices[position],
            "score": float(scores[position]),
            "p_value": float(pvalues[position]),
            "automatic_anomaly": position in automatic_positions,
            "method_p_values": {
                method: float(detections[method].final_pvalues[position]) for method in methods
            },
            "method_seed_p_values": {
                method: tuple(float(values[position]) for values in detections[method].seed_pvalues)
                for method in methods
            },
            "supporting_methods": supporting[position],
        }
        for position in order
    )
    automatic = tuple(prepared.test_indices[position] for position in order if position in automatic_positions)
    seed_review = tuple(prepared.test_indices[position] for position in order if position in seed_positions)
    method_review = tuple(prepared.test_indices[position] for position in order if position in method_positions)
    review = tuple(
        prepared.test_indices[position]
        for position in order
        if position in seed_positions or position in method_positions
    )
    method_name = policy_id if selected_method is None else selected_method
    return DetectionResult(
        method_name=method_name,
        rows=rows,
        automatic_anomaly_indices=automatic,
        train_reference=_train_reference(prepared),
        target_length=prepared.test_values.shape[1],
        policy_id=policy_id,
        selected_method=selected_method,
        seed_consensus_review_indices=seed_review,
        method_consensus_review_indices=method_review,
        review_indices=review,
    )


def run_anomaly_detection(
    prepared_values: PreparedDatasetValues,
    method_name: str,
    *,
    scorer=score_method,
) -> DetectionResult:
    if prepared_values.preparation_mode not in {"standard", "recovery_numba", "value_only_numba"}:
        raise ValueError(f"unsupported preparation mode: {prepared_values.preparation_mode}")
    detection = run_method_detection(prepared_values, method_name, scorer=scorer)
    train_scores = detection.train_scores
    test_scores = detection.test_scores
    pvalues = detection.final_pvalues
    attainable_p = 1.0 / (len(train_scores) + 1.0)
    resolution_blocked = ALPHA + 1e-15 < attainable_p
    selected_positions = set(
        np.flatnonzero(pvalues <= (attainable_p if resolution_blocked else ALPHA) + 1e-15)
    )
    sorted_positions = stable_descending_indices(test_scores)
    rows = tuple(
        {
            "series_index": prepared_values.test_indices[position],
            "score": float(test_scores[position]),
            "p_value": float(pvalues[position]),
            "automatic_anomaly": position in selected_positions,
        }
        for position in sorted_positions
    )
    reference = _train_reference(prepared_values)
    automatic = tuple(row["series_index"] for row in rows if row["automatic_anomaly"])
    warnings = tuple(
        prepared_values.test_indices[position]
        for position in sorted_positions
        if resolution_blocked
        and pvalues[position] <= attainable_p + 1e-15
    )
    return DetectionResult(
        method_name=method_name,
        rows=rows,
        automatic_anomaly_indices=automatic,
        train_reference=reference,
        target_length=prepared_values.test_values.shape[1],
        resolution_limited_warning_indices=warnings,
    )


def calculate_evaluation_metrics(
    detection_rows,
    labels_by_index: dict[int, int],
) -> dict[str, object]:
    rows = tuple(detection_rows)
    if not rows:
        raise ValueError("evaluation requires non-empty detection rows")
    for row in rows:
        if row["series_index"] not in labels_by_index:
            raise ValueError(f"label missing for series index {row['series_index']}")
    labels = np.asarray([labels_by_index[row["series_index"]] for row in rows], dtype=np.int64)
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    predictions = np.asarray([row["automatic_anomaly"] for row in rows], dtype=bool)
    if set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("evaluation requires finite scores and both derived TEST classes")
    positives = labels == 1
    negatives = labels == 0
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "true_positives": int(np.count_nonzero(predictions & positives)),
        "false_positives": int(np.count_nonzero(predictions & negatives)),
        "true_negatives": int(np.count_nonzero(~predictions & negatives)),
        "false_negatives": int(np.count_nonzero(~predictions & positives)),
    }
