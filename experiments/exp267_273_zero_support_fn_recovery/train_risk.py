"""TRAIN-only pseudo-run risk audit for Exp270-C.

This module never accepts TEST scores or labels.  An OOF query is calibrated
against the remaining TRAIN OOF scores, so its own score cannot lower its
empirical upper-tail p-value.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from .config import ALPHAS


def array_sha256(values: np.ndarray) -> str:
    """Hash numeric evidence with dtype and shape, not only raw bytes."""
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def risk_row_sha256(row: Mapping[str, object]) -> str:
    payload = {key: value for key, value in row.items() if key != "risk_row_sha256"}
    normalized = {
        key: (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in payload.items()
    }
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


class TrainRiskError(ValueError):
    """Raised when a TRAIN-only risk input violates its contract."""


_FORBIDDEN_TRAIN_RISK_FIELDS = frozenset({
    "y_test", "test_scores", "label", "labels", "test_labels", "tp", "fp", "fn",
    "precision", "recall", "f1", "combined_f1", "true_indices",
})


def train_tier(n_train: int) -> str:
    value = int(n_train)
    if value < 5:
        return "lt_5"
    if value < 20:
        return "5_19"
    if value < 100:
        return "20_99"
    if value < 200:
        return "100_199"
    if value < 500:
        return "200_499"
    return "ge_500"


def wilson_upper(successes: int, trials: int, *, confidence: float = 0.95) -> float:
    """One-sided Wilson upper confidence bound for a Bernoulli rate."""
    if type(successes) is not int or type(trials) is not int:
        raise TypeError("successes and trials must be integers")
    if trials < 0 or successes < 0 or successes > trials:
        raise TrainRiskError("invalid Wilson successes/trials")
    if not 0.5 < float(confidence) < 1.0:
        raise TrainRiskError("confidence must be strictly between 0.5 and 1")
    if trials == 0:
        return math.nan
    z = NormalDist().inv_cdf(float(confidence))
    n = float(trials)
    proportion = float(successes) / n
    denominator = 1.0 + (z * z) / n
    center = (proportion + (z * z) / (2.0 * n)) / denominator
    radius = (z / denominator) * math.sqrt(
        (proportion * (1.0 - proportion) / n) + (z * z) / (4.0 * n * n)
    )
    return min(1.0, center + radius)


def leave_one_out_upper_pvalues(train_oof: np.ndarray) -> np.ndarray:
    """Finite-sample upper-tail p-values with each query removed from reference."""
    scores = np.asarray(train_oof, dtype=float).reshape(-1)
    if len(scores) < 2:
        raise TrainRiskError("at least two finite TRAIN OOF scores are required")
    if not np.isfinite(scores).all():
        raise TrainRiskError("TRAIN OOF scores must be finite")
    output = np.empty(len(scores), dtype=float)
    for index, score in enumerate(scores):
        reference = np.concatenate((scores[:index], scores[index + 1 :]))
        output[index] = (1.0 + float(np.count_nonzero(reference >= score))) / (1.0 + len(reference))
    return output


def _risk_row(record: Mapping[str, object], alpha: float) -> dict[str, object]:
    if set(record).intersection(_FORBIDDEN_TRAIN_RISK_FIELDS):
        raise TrainRiskError("TRAIN-risk input must not contain TEST or evaluation fields")
    dataset = record.get("dataset")
    source = record.get("source")
    variant = record.get("variant")
    if not all(isinstance(value, str) and value for value in (dataset, source, variant)):
        raise TrainRiskError("dataset, source, and variant are required non-empty strings")
    n_train = int(record.get("n_train", 0))
    eligible = bool(record.get("eligible", True))
    abstained = bool(record.get("abstained", False))
    base: dict[str, object] = {
        "dataset": dataset,
        "source": source,
        "variant": variant,
        "alpha": float(alpha),
        "n_train": n_train,
        "train_tier": train_tier(n_train),
        "eligible": eligible,
        "abstained": abstained,
        "abstention_reason": record.get("abstention_reason"),
        "calibration_reference": "query_excluded_train_oof",
        "coverage_complete": bool(record.get("coverage_complete", True)),
        "cache_sha256": record.get("cache_sha256"),
        "run_contract_sha256": record.get("run_contract_sha256"),
        "train_oof_sha256": record.get("train_oof_sha256"),
    }
    if eligible and not abstained:
        for field in ("cache_sha256", "run_contract_sha256", "train_oof_sha256"):
            if not isinstance(base[field], str) or not base[field]:
                raise TrainRiskError(f"eligible TRAIN-risk record is missing {field}")
    if abstained or not eligible:
        result = {
            **base,
            "status": "abstained",
            "query_count": 0,
            "reference_count": 0,
            "normal_candidate_count": 0,
            "normal_candidate_rate": math.nan,
            "wilson_upper_95": math.nan,
        }
        result["risk_row_sha256"] = risk_row_sha256(result)
        return result
    scores = np.asarray(record.get("train_oof"), dtype=float).reshape(-1)
    if len(scores) < 2:
        result = {
            **base,
            "status": "insufficient_reference",
            "query_count": len(scores),
            "reference_count": max(0, len(scores) - 1),
            "normal_candidate_count": 0,
            "normal_candidate_rate": math.nan,
            "wilson_upper_95": math.nan,
        }
        result["risk_row_sha256"] = risk_row_sha256(result)
        return result
    if not np.isfinite(scores).all():
        result = {
            **base,
            "status": "nonfinite_train_oof",
            "query_count": len(scores),
            "reference_count": len(scores) - 1,
            "normal_candidate_count": 0,
            "normal_candidate_rate": math.nan,
            "wilson_upper_95": math.nan,
        }
        result["risk_row_sha256"] = risk_row_sha256(result)
        return result
    if len(scores) != n_train:
        result = {
            **base,
            "status": "train_oof_length_mismatch",
            "query_count": len(scores),
            "reference_count": max(0, len(scores) - 1),
            "normal_candidate_count": 0,
            "normal_candidate_rate": math.nan,
            "wilson_upper_95": math.nan,
        }
        result["risk_row_sha256"] = risk_row_sha256(result)
        return result
    if array_sha256(scores) != base["train_oof_sha256"]:
        raise TrainRiskError("TRAIN OOF hash does not match the verified record")
    supplied_pvalues = record.get("query_excluded_pvalues")
    if supplied_pvalues is None:
        pvalues = leave_one_out_upper_pvalues(scores)
    else:
        pvalues = np.asarray(supplied_pvalues, dtype=float).reshape(-1)
        if len(pvalues) != n_train or not np.isfinite(pvalues).all():
            raise TrainRiskError("query-excluded p-values must be finite and match n_train")
        if np.any((pvalues < 0.0) | (pvalues > 1.0)):
            raise TrainRiskError("query-excluded p-values must lie in [0, 1]")
        base["calibration_reference"] = "query_excluded_per_seed_intersection"
        base["query_excluded_pvalues_sha256"] = array_sha256(pvalues)
    candidates = int(np.count_nonzero(pvalues <= float(alpha)))
    result = {
        **base,
        "status": "ok",
        "query_count": len(scores),
        "reference_count": len(scores) - 1,
        "normal_candidate_count": candidates,
        "normal_candidate_rate": candidates / len(scores),
        "wilson_upper_95": wilson_upper(candidates, len(scores)),
    }
    result["risk_row_sha256"] = risk_row_sha256(result)
    return result


def audit_train_risk(
    source_records: Iterable[Mapping[str, object]], *, alphas: Iterable[float] = ALPHAS
) -> list[dict[str, object]]:
    """Return one query-excluded TRAIN-risk row per source/dataset/alpha."""
    selected_alphas = tuple(float(alpha) for alpha in alphas)
    if not selected_alphas or any(not 0.0 < alpha < 1.0 for alpha in selected_alphas):
        raise TrainRiskError("risk audit requires valid alpha values")
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, float]] = set()
    for record in source_records:
        for alpha in selected_alphas:
            row = _risk_row(record, alpha)
            key = (str(row["dataset"]), str(row["source"]), str(row["variant"]), alpha)
            if key in seen:
                raise TrainRiskError("duplicate source/dataset/variant/alpha risk row")
            seen.add(key)
            rows.append(row)
    return sorted(rows, key=lambda item: (str(item["source"]), str(item["variant"]), str(item["dataset"]), float(item["alpha"])))


def summarize_train_risk(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate risk by source, variant, alpha, and TRAIN tier without TEST data."""
    grouped: dict[tuple[str, str, float, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["variant"]), float(row["alpha"]), str(row["train_tier"]))].append(row)
    summary: list[dict[str, object]] = []
    for key, members in sorted(grouped.items()):
        valid = [row for row in members if row.get("status") == "ok"]
        invalid_eligible = [
            row for row in members
            if bool(row.get("eligible", False)) and row.get("status") != "ok"
        ]
        hashes_valid = all(row.get("risk_row_sha256") == risk_row_sha256(row) for row in members)
        queries = sum(int(row["query_count"]) for row in valid)
        candidates = sum(int(row["normal_candidate_count"]) for row in valid)
        summary.append({
            "source": key[0], "variant": key[1], "alpha": key[2], "train_tier": key[3],
            "datasets": len(members), "valid_datasets": len(valid),
            "abstained_datasets": sum(row.get("status") == "abstained" for row in members),
            "invalid_datasets": len(members) - len(valid) - sum(row.get("status") == "abstained" for row in members),
            "invalid_eligible_datasets": len(invalid_eligible),
            "all_rows_hash_valid": hashes_valid,
            "all_required_valid": not invalid_eligible and hashes_valid,
            "normal_candidate_count": candidates, "query_count": queries,
            "normal_candidate_rate": (candidates / queries) if queries else math.nan,
            "wilson_upper_95": wilson_upper(candidates, queries) if queries else math.nan,
            "calibration_reference": "query_excluded_train_oof",
        })
    return summary


def write_train_risk_audit(output_dir: Path, rows: Iterable[Mapping[str, object]]) -> tuple[Path, Path]:
    """Persist deterministic dataset and aggregate TRAIN-only risk reports."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    summaries = summarize_train_risk(materialized)
    fields = tuple(sorted({key for row in materialized for key in row}))
    summary_fields = tuple(sorted({key for row in summaries for key in row}))
    detail_path = root / "exp270_train_risk_dataset.csv"
    summary_path = root / "exp270_train_risk_summary.csv"
    for path, values, headers in ((detail_path, materialized, fields), (summary_path, summaries, summary_fields)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(values)
    return detail_path, summary_path


__all__ = [
    "TrainRiskError", "array_sha256", "audit_train_risk", "leave_one_out_upper_pvalues",
    "risk_row_sha256",
    "summarize_train_risk", "train_tier", "wilson_upper", "write_train_risk_audit",
]
