from __future__ import annotations

from collections.abc import Sequence
import csv
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from experiments.exp393_396.contracts import GrammarConfig


SCORE_NAMES = ("transition", "dwell", "recovery", "composite")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def one_sided_wilson_upper(successes: int, total: int, confidence: float = 0.95) -> float:
    if total <= 0:
        return 0.0
    z = NormalDist().inv_cdf(confidence)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2)
    )
    return min(1.0, (center + radius) / denominator)


def _oof_conformal_p_values(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    result = np.empty(values.size, dtype=float)
    for index, value in enumerate(values):
        reference = np.delete(values, index)
        result[index] = (
            (1 + np.count_nonzero(reference >= value)) / (reference.size + 1)
            if reference.size
            else 1.0
        )
    return result


def audit_prediction_row(
    row: dict[str, Any],
    config: GrammarConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": str(row.get("dataset", "")),
        "n_train_normal": int(row.get("n_train_normal", 0)),
        "n_test": int(row.get("n_test", 0)),
        "effective_state_count": int(row.get("effective_state_count", 0)),
        "abstained": bool(row.get("abstained", False)),
        "abstention_reason": str(row.get("abstention_reason", "")),
    }
    if result["abstained"]:
        result["conformal_resolution"] = (
            1.0 / (result["n_train_normal"] + 1)
            if result["n_train_normal"] >= 0
            else 1.0
        )
        return result
    references = row.get("train_oof_scores", {})
    result["conformal_resolution"] = 1.0 / (result["n_train_normal"] + 1)
    for name in SCORE_NAMES:
        p_values = _oof_conformal_p_values(references.get(name, []))
        result[f"{name}_oof_runs"] = int(p_values.size)
        for alpha in config.alphas:
            slug = f"{alpha:.3f}"
            alerts = int(np.count_nonzero(p_values <= alpha))
            result[f"{name}_alerts_alpha_{slug}"] = alerts
            result[f"{name}_rate_alpha_{slug}"] = (
                alerts / p_values.size if p_values.size else 0.0
            )
            result[f"{name}_wilson_upper_alpha_{slug}"] = one_sided_wilson_upper(
                alerts,
                int(p_values.size),
            )
    return result


def aggregate_calibration_rows(
    rows: Sequence[dict[str, Any]],
    config: GrammarConfig,
) -> list[dict[str, Any]]:
    rows = list(rows)
    summary: list[dict[str, Any]] = []
    for name in SCORE_NAMES:
        for alpha in config.alphas:
            slug = f"{alpha:.3f}"
            run_key = f"{name}_oof_runs"
            alert_key = f"{name}_alerts_alpha_{slug}"
            runs = sum(int(row.get(run_key, 0)) for row in rows)
            alerts = sum(int(row.get(alert_key, 0)) for row in rows)
            summary.append(
                {
                    "score": name,
                    "alpha": alpha,
                    "datasets": len(rows),
                    "available_datasets": sum(
                        int(not bool(row.get("abstained", False))) for row in rows
                    ),
                    "abstained_datasets": sum(
                        int(bool(row.get("abstained", False))) for row in rows
                    ),
                    "pseudo_normal_runs": runs,
                    "false_alerts": alerts,
                    "false_alert_rate": alerts / runs if runs else 0.0,
                    "false_alerts_per_100_runs": 100.0 * alerts / runs if runs else 0.0,
                    "wilson_upper_95": one_sided_wilson_upper(alerts, runs),
                    "retrospective_counterfactual": True,
                    "contains_test_labels": False,
                    "test_metrics_select_policy": False,
                }
            )
    return summary


def render_calibration_report(
    rows: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
) -> str:
    rows = list(rows)
    lines = [
        "# Exp394 TRAIN-normal calibration audit",
        "",
        "이 감사는 TEST 라벨과 이상 위치를 사용하지 않는다.",
        "",
        f"- datasets: {len(rows):,}",
        f"- abstained: {sum(bool(row.get('abstained')) for row in rows):,}",
        "- interpretation: retrospective calibration audit",
        "- policy selection from TEST metrics: false",
        "",
        "## Alpha summary",
        "",
        "| score | alpha | pseudo-normal runs | alerts | alerts / 100 | Wilson upper |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['score']} | {row['alpha']:.3f} | "
            f"{row['pseudo_normal_runs']:,} | {row['false_alerts']:,} | "
            f"{row['false_alerts_per_100_runs']:.4f} | "
            f"{row['wilson_upper_95']:.6f} |"
        )
    lines.extend(
        [
            "",
            "기존 점수원의 raw TRAIN OOF vector가 동결 artifact에 없으면 "
            "source correlation은 확인 불가로 기록한다.",
            "",
        ]
    )
    return "\n".join(lines)
