from __future__ import annotations

import csv
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping

from experiments.exp220_223.prediction import verify_prediction_manifest


OUTCOME_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_outcomes (
    batch_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    adjudicated_at_utc TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    prediction_sha256 TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL,
    PRIMARY KEY (batch_id, instance_id)
);
"""

LANE_FIELDS = (
    "exp137_hard",
    "exp137_standard",
    "exp137_priority",
    "exp190_high",
    "exp190_standard",
    "exp195_shadow",
    "exp224_shadow",
)

GATES = {
    "exp190_high": {
        "minimum_candidates": 100,
        "minimum_profiles": 30,
        "precision_lower": 0.80,
        "false_alert_upper": 0.005,
    },
    "exp190_standard": {
        "minimum_candidates": 200,
        "minimum_profiles": 50,
        "precision_lower": 0.70,
        "false_alert_upper": 0.010,
    },
    "exp195_shadow": {
        "minimum_candidates": 100,
        "minimum_profiles": 30,
        "precision_lower": 0.90,
        "false_alert_upper": 0.005,
    },
    "exp224_shadow": {
        "minimum_candidates": 100,
        "minimum_profiles": 30,
        "precision_lower": 0.90,
        "false_alert_upper": 0.005,
    },
}


def _utc(value: str, field: str) -> datetime:
    text = str(value).strip()
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {text}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include timezone")
    return parsed.astimezone(timezone.utc)


def wilson_bound(
    successes: int,
    trials: int,
    *,
    side: str,
    confidence: float = 0.95,
) -> float:
    successes = int(successes)
    trials = int(trials)
    if side not in {"lower", "upper"}:
        raise ValueError("side must be lower or upper")
    if trials <= 0:
        return 0.0 if side == "lower" else 1.0
    if successes < 0 or successes > trials:
        raise ValueError("successes must be within trials")
    z = NormalDist().inv_cdf(float(confidence))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    value = (center - radius if side == "lower" else center + radius) / denominator
    return min(1.0, max(0.0, value))


def import_outcomes(
    *,
    outcomes_db_path: Path,
    outcomes_csv_path: Path,
    prediction_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    manifest, predictions = verify_prediction_manifest(prediction_path, manifest_path)
    prediction_ids = {str(row["instance_id"]) for row in predictions}
    required = {"batch_id", "instance_id", "label", "adjudicated_at", "reviewer_id"}
    with Path(outcomes_csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"outcome columns missing: {sorted(required)}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("outcomes CSV is empty")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in source_rows:
        batch_id = str(source["batch_id"]).strip()
        instance_id = str(source["instance_id"]).strip()
        reviewer_id = str(source["reviewer_id"]).strip()
        if batch_id != manifest["batch_id"]:
            raise ValueError(f"outcome batch mismatch: {batch_id}")
        if instance_id not in prediction_ids:
            raise ValueError(f"outcome has no frozen prediction: {instance_id}")
        if instance_id in seen:
            raise ValueError(f"duplicate outcome instance_id: {instance_id}")
        seen.add(instance_id)
        if str(source["label"]).strip() not in {"0", "1"}:
            raise ValueError(f"label must be 0 or 1: instance_id={instance_id}")
        if not reviewer_id:
            raise ValueError(f"reviewer_id is required: instance_id={instance_id}")
        adjudicated = _utc(source["adjudicated_at"], "adjudicated_at")
        rows.append(
            {
                "batch_id": batch_id,
                "instance_id": instance_id,
                "label": int(source["label"]),
                "adjudicated_at": adjudicated.isoformat().replace("+00:00", "Z"),
                "reviewer_id": reviewer_id,
            }
        )

    outcomes_db_path = Path(outcomes_db_path)
    outcomes_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(outcomes_db_path) as connection:
        connection.executescript(OUTCOME_SCHEMA)
        try:
            connection.execute("BEGIN IMMEDIATE")
            for row in rows:
                connection.execute(
                    """
                    INSERT INTO shadow_outcomes(
                        batch_id, instance_id, label, adjudicated_at_utc,
                        reviewer_id, prediction_sha256, policy_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["batch_id"],
                        row["instance_id"],
                        row["label"],
                        row["adjudicated_at"],
                        row["reviewer_id"],
                        manifest["prediction_sha256"],
                        manifest["policy_sha256"],
                    ),
                )
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise ValueError(f"outcome already exists: {exc}") from exc
    return {"batch_id": manifest["batch_id"], "imported": len(rows)}


def evaluate_shadow_rows(
    predictions: Iterable[Mapping[str, object]],
    outcome_labels: Mapping[tuple[str, str], int],
    batches: Mapping[str, tuple[str, str]],
) -> dict[str, object]:
    prediction_rows = [dict(row) for row in predictions]
    joined = [
        (row, int(outcome_labels[(str(row["batch_id"]), str(row["instance_id"]))]))
        for row in prediction_rows
        if (str(row["batch_id"]), str(row["instance_id"])) in outcome_labels
    ]
    if batches:
        starts = [_utc(period[0], "collected_start") for period in batches.values()]
        ends = [_utc(period[1], "collected_end") for period in batches.values()]
        observation_days = (max(ends) - min(starts)).total_seconds() / 86400.0
    else:
        observation_days = 0.0
    normal_trials = sum(label == 0 for _, label in joined)
    lane_reports: dict[str, dict[str, object]] = {}
    for lane in LANE_FIELDS:
        candidates = [(row, label) for row, label in joined if bool(row.get(lane, False))]
        tp = sum(label == 1 for _, label in candidates)
        fp = sum(label == 0 for _, label in candidates)
        candidate_count = len(candidates)
        precision = tp / candidate_count if candidate_count else 0.0
        precision_lower = wilson_bound(tp, candidate_count, side="lower")
        false_alert_rate = fp / normal_trials if normal_trials else 0.0
        false_alert_upper = wilson_bound(fp, normal_trials, side="upper")
        profiles = {str(row["profile_id"]) for row, _ in candidates}
        candidate_batches = {str(row["batch_id"]) for row, _ in candidates}
        lane_report: dict[str, object] = {
            "candidate_count": candidate_count,
            "tp": tp,
            "fp": fp,
            "precision": precision,
            "precision_lower": precision_lower,
            "normal_trials": normal_trials,
            "normal_false_alert_rate": false_alert_rate,
            "normal_false_alert_rate_upper": false_alert_upper,
            "false_alerts_per_1000_normals": 1000.0 * false_alert_rate,
            "active_profiles": len(profiles),
            "candidate_batches": len(candidate_batches),
        }
        gate = GATES.get(lane)
        if gate is None:
            lane_report["decision"] = "diagnostic_only"
        else:
            minimums_met = (
                observation_days >= 30.0
                and len(candidate_batches) >= 3
                and candidate_count >= gate["minimum_candidates"]
                and len(profiles) >= gate["minimum_profiles"]
            )
            if not minimums_met:
                decision = "deferred"
            elif (
                precision_lower < gate["precision_lower"]
                or false_alert_upper > gate["false_alert_upper"]
            ):
                decision = "rejected"
            else:
                decision = "promotion_review_eligible"
            lane_report["decision"] = decision
            lane_report["gate"] = gate
        lane_reports[lane] = lane_report
    return {
        "experiment_id": "experiment_223_delayed_outcome_evaluation",
        "adjudicated_instances": len(joined),
        "batch_count": len(batches),
        "observation_days": observation_days,
        "automatic_promotion": False,
        "lanes": lane_reports,
    }


def load_outcome_labels(outcomes_db_path: Path) -> dict[tuple[str, str], int]:
    with sqlite3.connect(Path(outcomes_db_path)) as connection:
        rows = connection.execute(
            "SELECT batch_id, instance_id, label FROM shadow_outcomes"
        ).fetchall()
    return {(str(batch), str(instance)): int(label) for batch, instance, label in rows}


def write_evaluation_report(path: Path, report: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
