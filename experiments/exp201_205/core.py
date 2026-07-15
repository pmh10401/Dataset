from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

ALPHA = 0.01
TRAIN_RISK_CEILING = 0.001
SEED = 20260720
SOURCE_NAMES = ("spectral", "shapelet", "interval")

FORBIDDEN_PREDICTION_FIELDS = {
    "y_test", "labels", "test_labels", "anomaly_positions", "zero_f1",
    "tp", "fp", "fn", "precision", "recall", "f1", "hard_tp",
    "hard_fp", "hard_fn", "hard_f1", "combined_f1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_size_stratum(n_train: int) -> str | None:
    if int(n_train) < 100:
        return None
    return "100-199" if int(n_train) < 200 else "200+"


def parse_indices(value: object) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {int(item) for item in value}
    text = str(value).replace(",", " ").strip()
    return {int(token) for token in text.split() if token}


def empirical_tail_candidates(
    train_scores: Iterable[float], test_scores: Iterable[float], alpha: float = ALPHA
) -> set[int]:
    if float(alpha) != ALPHA:
        raise ValueError(f"alpha must remain frozen at {ALPHA}")
    train = np.asarray(list(train_scores), dtype=float)
    test = np.asarray(list(test_scores), dtype=float)
    train = train[np.isfinite(train)]
    if not len(train):
        raise ValueError("finite TRAIN-normal scores are required")
    threshold = float(np.quantile(train, 1.0 - ALPHA, method="higher"))
    return {int(i) for i in np.where(np.isfinite(test) & (test > threshold))[0]}


def three_source_agreement(
    spectral_indices: Iterable[int],
    shapelet_indices: Iterable[int],
    interval_indices: Iterable[int],
) -> set[int]:
    return (
        {int(i) for i in spectral_indices}
        & {int(i) for i in shapelet_indices}
        & {int(i) for i in interval_indices}
    )


def cross_confirmed_candidates(
    *,
    alternative_indices: Iterable[int],
    exp198_indices: Iterable[int],
    excluded_exp137_indices: Iterable[int],
) -> set[int]:
    return (
        {int(i) for i in alternative_indices}
        & {int(i) for i in exp198_indices}
    ) - {int(i) for i in excluded_exp137_indices}


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


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def binary_metrics(labels: np.ndarray, indices: Iterable[int]) -> dict[str, float | int]:
    selected = sorted({int(i) for i in indices})
    if selected and (selected[0] < 0 or selected[-1] >= len(labels)):
        raise ValueError("metric index outside TEST range")
    tp = int(np.sum(labels[selected])) if selected else 0
    fp = len(selected) - tp
    positives = int(np.sum(labels))
    fn = positives - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, positives)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "count": len(selected), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }
