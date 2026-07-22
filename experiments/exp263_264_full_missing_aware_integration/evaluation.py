from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from .inputs import EXPECTED_POPULATION, load_population


LANE_FIELDS = {
    "baseline_hard": "baseline_hard_indices",
    "autonomous_hard": "final_hard_indices",
    "standard_review": "final_standard_indices",
    "priority_review": "final_priority_indices",
}
EXPECTED_TOTAL_ANOMALIES = 3918


def verify_prediction_sha256(prediction_payload: bytes, expected_sha256: str) -> str:
    """Verify the frozen prediction bytes before any TEST-label access."""

    actual_sha256 = hashlib.sha256(prediction_payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "prediction SHA256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return actual_sha256


def metrics(selected: set[int], truth: set[int]) -> dict[str, float | int]:
    """Return per-dataset metrics with all undefined ratios recorded as zero."""

    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "alerts": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_denominator": precision_denominator,
        "recall_denominator": recall_denominator,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _load_test_labels(dataset: str) -> object:
    # Kept local so importing this module cannot access TEST labels.
    from experiments.exp226_235_hard_alert_quality.database import (
        load_test_labels_for_evaluation,
    )

    return load_test_labels_for_evaluation(dataset)


def _selected_indices(row: dict[str, object], lane: str) -> set[int]:
    if lane == "review":
        values = [
            *row[LANE_FIELDS["standard_review"]],
            *row[LANE_FIELDS["priority_review"]],
        ]
    else:
        values = row[LANE_FIELDS[lane]]
    return {int(index) for index in values}


def _validate_indices(dataset: str, indices: set[int], n_test: int, lane: str) -> None:
    if any(index < 0 or index >= n_test for index in indices):
        raise ValueError(f"prediction {lane} index is out of range: dataset={dataset}")


def _validate_final_lanes(row: dict[str, object], dataset: str) -> None:
    lanes = {
        "Hard": _selected_indices(row, "autonomous_hard"),
        "Standard": _selected_indices(row, "standard_review"),
        "Priority": _selected_indices(row, "priority_review"),
    }
    names = tuple(lanes)
    if any(
        lanes[names[left]] & lanes[names[right]]
        for left in range(len(names))
        for right in range(left + 1, len(names))
    ):
        raise ValueError(f"final lane overlap: dataset={dataset}")


def _evaluate_row(row: dict[str, object]) -> dict[str, object]:
    dataset = str(row["dataset"])
    n_test = int(row["n_test"])
    _validate_final_lanes(row, dataset)
    labels = _load_test_labels(dataset)
    if len(labels) != n_test:
        raise ValueError(
            f"TEST label length mismatch: dataset={dataset} "
            f"expected={n_test} actual={len(labels)}"
        )
    truth = {index for index, label in enumerate(labels) if int(label) == 1}
    if any(index < 0 or index >= n_test for index in truth):
        raise ValueError(f"TEST labels have an invalid anomaly index: dataset={dataset}")

    output = dict(row)
    output["true_anomaly_count"] = len(truth)
    for lane in (*LANE_FIELDS, "review"):
        selected = _selected_indices(row, lane)
        _validate_indices(dataset, selected, n_test, lane)
        output.update({f"{lane}_{key}": value for key, value in metrics(selected, truth).items()})

    for metric in ("alerts", "tp", "fp", "fn", "f1"):
        output[f"autonomous_hard_{metric}_delta_vs_exp253"] = (
            output[f"autonomous_hard_{metric}"] - output[f"baseline_hard_{metric}"]
        )
    return output


def _parse_prediction_payload(prediction_payload: bytes) -> list[dict[str, object]]:
    try:
        rows = [
            json.loads(line)
            for line in prediction_payload.decode("utf-8").splitlines()
            if line
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prediction payload is not valid JSONL") from exc
    if any(type(row) is not dict for row in rows):
        raise ValueError("prediction payload row is not an object")
    return rows


def _validate_population(
    rows: list[dict[str, object]], expected_population: tuple[str, ...]
) -> None:
    if len(expected_population) != len(set(expected_population)):
        raise ValueError("expected population contains duplicate dataset names")
    if len(rows) != len(expected_population):
        raise ValueError(
            "prediction population size mismatch: "
            f"expected={len(expected_population)} actual={len(rows)}"
        )
    names = [str(row.get("dataset", "")) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("prediction population contains duplicate dataset names")
    if set(names) != set(expected_population):
        raise ValueError("prediction population names mismatch")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _lane_summary(rows: list[dict[str, object]], lane: str) -> dict[str, float | int]:
    tp = sum(int(row[f"{lane}_tp"]) for row in rows)
    fp = sum(int(row[f"{lane}_fp"]) for row in rows)
    fn = sum(int(row[f"{lane}_fn"]) for row in rows)
    alerts = tp + fp
    total_anomalies = sum(int(row["true_anomaly_count"]) for row in rows)
    return {
        "datasets": len(rows),
        "alerts": alerts,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_anomalies": total_anomalies,
        "micro_precision": tp / alerts if alerts else 0.0,
        "micro_recall": tp / total_anomalies if total_anomalies else 0.0,
        "macro_mean_f1": _mean(float(row[f"{lane}_f1"]) for row in rows),
        "zero_f1_datasets": sum(float(row[f"{lane}_f1"]) == 0.0 for row in rows),
    }


def _paired_baseline_comparison(rows: list[dict[str, object]]) -> dict[str, float | int | str]:
    def delta(metric: str) -> float:
        return sum(
            float(row[f"autonomous_hard_{metric}"])
            - float(row[f"baseline_hard_{metric}"])
            for row in rows
        )

    f1_deltas = [
        float(row["autonomous_hard_f1"]) - float(row["baseline_hard_f1"])
        for row in rows
    ]
    return {
        "baseline_experiment": "Exp253",
        "comparison": "paired_dataset_level_autonomous_hard_minus_exp253_baseline_hard",
        "datasets": len(rows),
        "alert_delta": int(delta("alerts")),
        "tp_delta": int(delta("tp")),
        "fp_delta": int(delta("fp")),
        "fn_delta": int(delta("fn")),
        "macro_mean_f1_delta": _mean(f1_deltas),
        "zero_f1_dataset_delta": sum(
            int(float(row["autonomous_hard_f1"]) == 0.0)
            - int(float(row["baseline_hard_f1"]) == 0.0)
            for row in rows
        ),
        "f1_improved_datasets": sum(delta > 0.0 for delta in f1_deltas),
        "f1_unchanged_datasets": sum(delta == 0.0 for delta in f1_deltas),
        "f1_worsened_datasets": sum(delta < 0.0 for delta in f1_deltas),
    }


def _diagnostic_subgroups(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    groups = {
        "missing_override": [row for row in rows if bool(row["missing_override"])],
        "nonmissing": [row for row in rows if not bool(row["missing_override"])],
        "n_train_lt_100": [row for row in rows if int(row["n_train"]) < 100],
        "n_train_ge_100": [row for row in rows if int(row["n_train"]) >= 100],
    }
    return {
        name: {
            "datasets": len(group),
            "used_for_policy_selection": False,
            "autonomous_hard": _lane_summary(group, "autonomous_hard"),
            "review": _lane_summary(group, "review"),
        }
        for name, group in groups.items()
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _lane_transitions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    transitions = []
    for row in rows:
        baseline_hard = _selected_indices(row, "baseline_hard")
        baseline_standard = {int(index) for index in row["baseline_standard_indices"]}
        baseline_priority = {int(index) for index in row["baseline_priority_indices"]}
        final_hard = _selected_indices(row, "autonomous_hard")
        final_priority = _selected_indices(row, "priority_review")
        transitions.append(
            {
            "dataset": row["dataset"],
            "missing_override": row["missing_override"],
            "n_train": row["n_train"],
            "baseline_hard_count": len(baseline_hard),
            "baseline_standard_count": len(baseline_standard),
            "baseline_priority_count": len(baseline_priority),
            "baseline_hard_alerts": row["baseline_hard_alerts"],
            "autonomous_hard_alerts": row["autonomous_hard_alerts"],
            "standard_review_alerts": row["standard_review_alerts"],
            "priority_review_alerts": row["priority_review_alerts"],
            "review_alerts": row["review_alerts"],
            "promoted_alerts": len(row.get("promoted_indices", [])),
            "standard_to_hard_additions": len(baseline_standard & final_hard),
            "standard_to_priority_additions": len(baseline_standard & final_priority),
            "autonomous_hard_tp_delta_vs_exp253": row[
                "autonomous_hard_tp_delta_vs_exp253"
            ],
            "autonomous_hard_fp_delta_vs_exp253": row[
                "autonomous_hard_fp_delta_vs_exp253"
            ],
        }
        )
    return transitions


def _transition_summary(transitions: list[dict[str, object]]) -> dict[str, int]:
    return {
        "datasets": len(transitions),
        "standard_to_hard_additions": sum(
            int(row["standard_to_hard_additions"]) for row in transitions
        ),
        "standard_to_priority_additions": sum(
            int(row["standard_to_priority_additions"]) for row in transitions
        ),
    }


def evaluate_all(
    prediction_path: Path,
    *,
    expected_sha256: str,
    expected_population: tuple[str, ...] | None = None,
    expected_total_anomalies: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate a frozen prediction artifact and write its retrospective outputs."""

    prediction_path = Path(prediction_path)
    prediction_payload = prediction_path.read_bytes()
    actual_sha256 = verify_prediction_sha256(prediction_payload, expected_sha256)
    predictions = _parse_prediction_payload(prediction_payload)
    using_production_defaults = expected_population is None
    expected_population = (
        tuple(load_population()) if using_production_defaults else tuple(expected_population)
    )
    expected_total_anomalies = (
        EXPECTED_TOTAL_ANOMALIES
        if expected_total_anomalies is None
        else int(expected_total_anomalies)
    )
    if using_production_defaults and len(expected_population) != EXPECTED_POPULATION:
        raise ValueError(f"expected production population size mismatch: {len(expected_population)}")
    _validate_population(predictions, expected_population)
    evaluated = [_evaluate_row(row) for row in predictions]
    total_anomalies = sum(int(row["true_anomaly_count"]) for row in evaluated)
    if total_anomalies != expected_total_anomalies:
        raise ValueError(
            "total anomaly mismatch: "
            f"expected={expected_total_anomalies} actual={total_anomalies}"
        )
    transitions = _lane_transitions(evaluated)

    summary: dict[str, object] = {
        "experiment": 264,
        "experiment_id": "experiment_264_full_missing_aware_posthoc_evaluation",
        "prediction_sha256": actual_sha256,
        "datasets": len(evaluated),
        "retrospective": True,
        "selected_from_prior_test_review": True,
        "deployment_eligible": False,
        "labels_loaded_after_sha256_verification": True,
        "policy_selection_performed": False,
        "baseline_hard": _lane_summary(evaluated, "baseline_hard"),
        "autonomous_hard": _lane_summary(evaluated, "autonomous_hard"),
        "standard_review": _lane_summary(evaluated, "standard_review"),
        "priority_review": _lane_summary(evaluated, "priority_review"),
        "review": _lane_summary(evaluated, "review"),
        "paired_baseline_comparison": _paired_baseline_comparison(evaluated),
        "diagnostic_subgroups": _diagnostic_subgroups(evaluated),
        "lane_transitions": _transition_summary(transitions),
    }

    output_root = prediction_path.parent
    _write_csv(output_root / "results.csv", evaluated)
    _write_csv(output_root / "lane_transitions.csv", transitions)
    (output_root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return evaluated, summary
