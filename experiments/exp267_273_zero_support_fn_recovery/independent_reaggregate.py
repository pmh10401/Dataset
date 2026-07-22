"""CSV-only independent reaggregation for Exp272/273 outputs."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("evaluation results CSV is empty")
    names = [row.get("dataset", "") for row in rows]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("evaluation results dataset population is invalid")
    return rows


def reaggregate(
    results_path: Path,
    summary_path: Path,
    *,
    expected_datasets: int = 1117,
    expected_total_anomalies: int = 3918,
    expected_hard_alerts: int = 2253,
    tolerance: float = 1e-12,
) -> dict[str, int | float]:
    """Rebuild core totals without importing the evaluation implementation."""
    rows = _read_rows(results_path)
    if len(rows) != expected_datasets:
        raise ValueError("independent dataset-count mismatch")

    def total(field: str) -> int:
        try:
            return sum(int(row[field]) for row in rows)
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid integer field in results: {field}") from error

    anomalies = total("true_anomaly_count")
    hard_count = total("hard_count")
    hard_tp = total("hard_tp")
    hard_fp = total("hard_fp")
    review_count = total("review_count")
    review_tp = total("review_tp")
    review_fp = total("review_fp")
    overlap = total("lane_overlap_count")
    if anomalies != expected_total_anomalies:
        raise ValueError("independent total-anomaly mismatch")
    if hard_count != expected_hard_alerts:
        raise ValueError("Exp253 Hard parity mismatch")
    if overlap:
        raise ValueError("independent lane overlap is non-zero")
    rebuilt: dict[str, int | float] = {
        "datasets": len(rows), "total_anomalies": anomalies,
        "hard_alerts": hard_count, "hard_tp": hard_tp, "hard_fp": hard_fp,
        "hard_precision": hard_tp / hard_count if hard_count else 0.0,
        "hard_recall": hard_tp / anomalies if anomalies else 0.0,
        "mean_hard_f1": sum(float(row["hard_f1"]) for row in rows) / len(rows),
        "review_candidates": review_count, "review_tp": review_tp, "review_fp": review_fp,
        "review_precision": review_tp / review_count if review_count else 0.0,
        "mean_combined_f1": sum(float(row["combined_f1"]) for row in rows) / len(rows),
        "lane_overlap_count": overlap,
    }
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError("saved evaluation summary is invalid")
    comparisons = {
        "datasets": summary.get("datasets"),
        "total_anomalies": summary.get("total_anomalies"),
        "hard_alerts": summary.get("autonomous", {}).get("hard_alerts"),
        "hard_tp": summary.get("autonomous", {}).get("hard_tp"),
        "hard_fp": summary.get("autonomous", {}).get("hard_fp"),
        "hard_precision": summary.get("autonomous", {}).get("hard_precision"),
        "hard_recall": summary.get("autonomous", {}).get("hard_recall"),
        "mean_hard_f1": summary.get("autonomous", {}).get("mean_hard_f1"),
        "review_candidates": summary.get("human_assisted_review", {}).get("review_candidates"),
        "review_tp": summary.get("human_assisted_review", {}).get("review_tp"),
        "review_fp": summary.get("human_assisted_review", {}).get("review_fp"),
        "review_precision": summary.get("human_assisted_review", {}).get("review_precision"),
        "mean_combined_f1": summary.get("human_assisted_diagnostic", {}).get("mean_combined_f1"),
        "lane_overlap_count": summary.get("lane_overlap_count"),
    }
    for field, saved in comparisons.items():
        current = rebuilt[field]
        if isinstance(current, int):
            if type(saved) is not int or saved != current:
                raise ValueError(f"saved summary integer mismatch: {field}")
        else:
            try:
                saved_float = float(saved)
            except (TypeError, ValueError) as error:
                raise ValueError(f"saved summary float is invalid: {field}") from error
            if not math.isclose(float(current), saved_float, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(f"saved summary float mismatch: {field}")
    return rebuilt


__all__ = ["reaggregate"]
