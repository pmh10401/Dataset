from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from urllib.parse import quote

from .config import DB_PATH, EVIDENCE_GROUPS


PREDICTION_FILENAME = "03_frozen_predictions.jsonl"
PREDICTION_MANIFEST_FILENAME = "04_prediction_manifest.json"
RESULTS_FILENAME = "05_dataset_results.csv"
SUMMARY_FILENAME = "06_global_summary.csv"
DIAGNOSTICS_FILENAME = "07_source_agreement_diagnostics.csv"

BASELINE_LANES = ("hard", "shadow", "priority", "standard")
REVIEW_LANES = ("recovery_shadow", "recovery_priority", "data_quality_review")
ALL_LANES = (*BASELINE_LANES, *REVIEW_LANES)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _csv_text(rows: list[Mapping[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot write an empty evaluation CSV")
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read prediction manifest") from exc
    if type(manifest) is not dict:
        raise ValueError("prediction manifest is not an object")
    return manifest


def _verified_prediction_payload(prediction_path: Path, manifest_path: Path) -> tuple[bytes, str]:
    manifest = _read_manifest(manifest_path)
    try:
        payload = Path(prediction_path).read_bytes()
    except OSError as exc:
        raise ValueError("cannot read frozen prediction") from exc
    actual = hashlib.sha256(payload).hexdigest()
    expected = manifest.get("prediction_sha256")
    if type(expected) is not str or actual != expected:
        raise ValueError(f"prediction SHA256 mismatch: expected={expected!r} actual={actual}")
    return payload, actual


def _parse_rows(payload: bytes) -> list[dict[str, object]]:
    try:
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen prediction is not valid JSONL") from exc
    if not rows or any(type(row) is not dict for row in rows):
        raise ValueError("frozen prediction must contain object rows")
    names = [row.get("dataset") for row in rows]
    if any(type(name) is not str or not name for name in names):
        raise ValueError("prediction dataset name is invalid")
    if len(names) != len(set(names)):
        raise ValueError("prediction contains duplicate datasets")
    return rows


def _indices(row: Mapping[str, object], lane: str) -> set[int]:
    field = f"{lane}_indices"
    values = row.get(field)
    if type(values) is not list:
        raise ValueError(f"prediction row has invalid {field}")
    if any(type(index) is not int for index in values):
        raise ValueError(f"prediction row has non-integer {field}")
    selected = set(values)
    if len(selected) != len(values):
        raise ValueError(f"prediction row has duplicate {field}")
    return selected


def _preflight_row(row: Mapping[str, object]) -> dict[str, set[int]]:
    if row.get("contains_test_labels") is not False:
        raise ValueError("frozen prediction TEST-label boundary mismatch")
    if row.get("contains_evaluation_metrics") is not False:
        raise ValueError("frozen prediction evaluation-metric boundary mismatch")
    lanes = {lane: _indices(row, lane) for lane in ALL_LANES}
    names = tuple(lanes)
    for left, name in enumerate(names):
        for other in names[left + 1 :]:
            if lanes[name] & lanes[other]:
                raise ValueError(f"prediction lane overlap: {name} and {other}")
    return lanes


def labels_for_dataset(connection: sqlite3.Connection, dataset: str) -> dict[int, int]:
    rows = connection.execute(
        """SELECT i.instance_index, CAST(i.label AS INTEGER)
           FROM instances i JOIN datasets d ON d.id=i.dataset_id
           WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index""",
        (dataset,),
    )
    labels = {int(index): int(label) for index, label in rows}
    if not labels:
        raise ValueError(f"TEST labels are missing: dataset={dataset}")
    if set(labels) != set(range(len(labels))):
        raise ValueError(f"TEST instance indices are not contiguous: dataset={dataset}")
    if any(label not in (0, 1) for label in labels.values()):
        raise ValueError(f"TEST labels are not binary: dataset={dataset}")
    return labels


def lane_metrics(indices: Iterable[int], labels: Mapping[int, int]) -> dict[str, float | int]:
    """Match exact dataset instance indices; undefined ratios are always zero."""

    selected = set(indices)
    if any(index not in labels for index in selected):
        raise ValueError("prediction index is outside the TEST instance range")
    truth = {index for index, label in labels.items() if int(label) == 1}
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    count = len(selected)
    precision = tp / count if count else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = list(rows)

    def total(field: str) -> int:
        return sum(int(row.get(field, 0)) for row in rows)

    hard_alerts = total("hard_alerts")
    baseline_hard_alerts = total("baseline_hard_alerts")
    hard_tp = total("hard_tp")
    hard_fp = total("hard_fp")
    total_anomalies = total("true_anomaly_count")
    total_test_runs = total("total_test_runs")
    summary: dict[str, object] = {
        "datasets": len(rows),
        "total_test_runs": total_test_runs,
        "total_anomalies": total_anomalies,
        "baseline_hard_alerts": baseline_hard_alerts,
        "baseline_shadow_count": total("baseline_shadow_count"),
        "baseline_priority_count": total("baseline_priority_count"),
        "baseline_standard_count": total("baseline_standard_count"),
        "hard_alerts": hard_alerts,
        "hard_alert_delta": hard_alerts - baseline_hard_alerts,
        "hard_tp": hard_tp,
        "hard_fp": hard_fp,
        "hard_fn": total("hard_fn"),
        "hard_precision": hard_tp / hard_alerts if hard_alerts else 0.0,
        "hard_recall": hard_tp / total_anomalies if total_anomalies else 0.0,
        "hard_mean_dataset_f1": _mean(float(row.get("hard_f1", 0.0)) for row in rows),
        "hard_zero_f1_datasets": sum(float(row.get("hard_f1", 0.0)) == 0.0 for row in rows),
    }
    summary["baseline_union_count"] = sum(
        int(summary[field])
        for field in (
            "baseline_hard_alerts",
            "baseline_shadow_count",
            "baseline_priority_count",
            "baseline_standard_count",
        )
    )
    for lane in REVIEW_LANES:
        count = total(f"{lane}_count")
        tp = total(f"{lane}_tp")
        summary.update(
            {
                f"{lane}_count": count,
                f"{lane}_tp": tp,
                f"{lane}_fp": total(f"{lane}_fp"),
                f"{lane}_precision": tp / count if count else 0.0,
            }
        )
    review_count = sum(int(summary[f"{lane}_count"]) for lane in REVIEW_LANES)
    summary.update(
        {
            "human_assisted_review_count": review_count,
            "recovered_unique_fn_count": total("recovered_unique_fn_count"),
            "datasets_with_added_candidates": sum(
                int(int(row.get("human_assisted_review_count", 0)) > 0) for row in rows
            ),
            "review_request_rate": review_count / total_test_runs if total_test_runs else 0.0,
            "retrospective": True,
            "deployment_eligible": False,
            "autonomous_scope": "Hard only; unchanged Exp253 baseline",
            "human_assisted_scope": (
                "RecoveryShadow, RecoveryPriority, and DataQualityReview; review only"
            ),
            "point_matching": "exact dataset TEST instance_index; no adjacency merging",
            "zero_denominator_policy": "0.0",
        }
    )
    return summary


def _metric_columns(prefix: str, metrics: Mapping[str, float | int]) -> dict[str, object]:
    count_name = "alerts" if prefix == "hard" else "count"
    output = {f"{prefix}_{count_name}": metrics["count"]}
    output.update({f"{prefix}_{key}": metrics[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")})
    return output


def _diagnostic_row(
    row: Mapping[str, object], lanes: Mapping[str, set[int]], labels: Mapping[int, int]
) -> dict[str, object]:
    dataset = str(row["dataset"])
    support_value = row.get("support_by_index", {})
    if type(support_value) is not dict:
        raise ValueError(f"support_by_index is invalid: dataset={dataset}")
    try:
        support = {int(index): int(count) for index, count in support_value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"support_by_index is invalid: dataset={dataset}") from exc
    if any(index not in labels or count < 1 or count > len(EVIDENCE_GROUPS) for index, count in support.items()):
        raise ValueError(f"support_by_index is outside its contract: dataset={dataset}")
    existing = set().union(*(lanes[lane] for lane in BASELINE_LANES))
    outside = set(labels) - existing
    availability = row.get("source_availability", {})
    if type(availability) is not dict or any(
        type(availability.get(group)) is not bool for group in EVIDENCE_GROUPS
    ):
        raise ValueError(f"source_availability is invalid: dataset={dataset}")
    diagnostic: dict[str, object] = {
        "dataset": dataset,
        "outside_existing_count": len(outside),
        "available_source_count": sum(bool(availability[group]) for group in EVIDENCE_GROUPS),
        "abstained_source_count": sum(not bool(availability[group]) for group in EVIDENCE_GROUPS),
    }
    for count in range(len(EVIDENCE_GROUPS) + 1):
        selected = {index for index in outside if support.get(index, 0) == count}
        metric = lane_metrics(selected, labels)
        diagnostic.update(
            {
                f"agreement_{count}_count": metric["count"],
                f"agreement_{count}_tp": metric["tp"],
                f"agreement_{count}_fp": metric["fp"],
                f"agreement_{count}_precision": metric["precision"],
            }
        )
    for group in EVIDENCE_GROUPS:
        diagnostic[f"{group}_available"] = bool(availability[group])
    return diagnostic


def _evaluate_row(
    row: Mapping[str, object], lanes: Mapping[str, set[int]], labels: Mapping[int, int], sha256: str
) -> tuple[dict[str, object], dict[str, object]]:
    for lane, indices in lanes.items():
        if any(index not in labels for index in indices):
            raise ValueError(f"prediction {lane} index is outside TEST range: dataset={row['dataset']}")
    truth = {index for index, label in labels.items() if label == 1}
    hard = lane_metrics(lanes["hard"], labels)
    review = set().union(*(lanes[lane] for lane in REVIEW_LANES))
    recovered = truth & (lanes["recovery_shadow"] | lanes["recovery_priority"])
    output: dict[str, object] = {
        "dataset": row["dataset"],
        "n_train": int(row.get("n_train", 0)),
        "total_test_runs": len(labels),
        "true_anomaly_count": len(truth),
        "prediction_sha256": sha256,
        "baseline_hard_alerts": hard["count"],
        "baseline_shadow_count": len(lanes["shadow"]),
        "baseline_priority_count": len(lanes["priority"]),
        "baseline_standard_count": len(lanes["standard"]),
    }
    output.update(_metric_columns("hard", hard))
    for lane in REVIEW_LANES:
        output.update(_metric_columns(lane, lane_metrics(lanes[lane], labels)))
    output.update(
        {
            "human_assisted_review_count": len(review),
            "recovered_unique_fn_count": len(recovered),
            "review_request_rate": len(review) / len(labels) if labels else 0.0,
            "retrospective": True,
            "deployment_eligible": False,
        }
    )
    return output, _diagnostic_row(row, lanes, labels)


def _read_only_connection(database_path: Path) -> sqlite3.Connection:
    resolved = Path(database_path).resolve()
    uri = f"file:{quote(str(resolved), safe='/')}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def evaluate_predictions(
    prediction_path: Path,
    manifest_path: Path | None = None,
    *,
    database_path: Path = DB_PATH,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate a verified frozen checkpoint without ever writing prediction artifacts."""

    prediction_path = Path(prediction_path)
    if prediction_path.is_dir():
        output_root = prediction_path
        prediction_path = output_root / PREDICTION_FILENAME
        if manifest_path is None:
            manifest_path = output_root / PREDICTION_MANIFEST_FILENAME
    else:
        output_root = prediction_path.parent
    if manifest_path is None:
        manifest_path = output_root / PREDICTION_MANIFEST_FILENAME

    payload, prediction_sha256 = _verified_prediction_payload(prediction_path, Path(manifest_path))
    prediction_rows = _parse_rows(payload)
    preflight = [(row, _preflight_row(row)) for row in prediction_rows]

    evaluated: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    connection = _read_only_connection(database_path)
    try:
        connection.execute("PRAGMA query_only = ON")
        for row, lanes in preflight:
            labels = labels_for_dataset(connection, str(row["dataset"]))
            result, diagnostic = _evaluate_row(row, lanes, labels, prediction_sha256)
            evaluated.append(result)
            diagnostics.append(diagnostic)
    finally:
        connection.close()

    summary = aggregate(evaluated)
    summary["prediction_sha256"] = prediction_sha256
    _atomic_text(output_root / RESULTS_FILENAME, _csv_text(evaluated))
    _atomic_text(output_root / SUMMARY_FILENAME, _csv_text([summary]))
    _atomic_text(output_root / DIAGNOSTICS_FILENAME, _csv_text(diagnostics))
    return evaluated, summary
