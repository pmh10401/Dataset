from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from urllib.parse import quote

from .config import (
    DB_PATH,
    EVIDENCE_GROUPS,
    EXPECTED_DATASETS,
    EXP253_PREDICTION,
    FROZEN_PREDICTION_HASHES,
    FROZEN_PREDICTION_MANIFEST_HASHES,
)
from .prediction import CheckpointError, verify_prediction_checkpoint


PREDICTION_FILENAME = "03_frozen_predictions.jsonl"
PREDICTION_MANIFEST_FILENAME = "04_prediction_manifest.json"
RESULTS_FILENAME = "05_dataset_results.csv"
SUMMARY_FILENAME = "06_global_summary.csv"
DIAGNOSTICS_FILENAME = "07_source_agreement_diagnostics.csv"

BASELINE_LANES = ("hard", "shadow", "priority", "standard")
REVIEW_LANES = ("recovery_shadow", "recovery_priority", "data_quality_review")
ALL_LANES = (*BASELINE_LANES, *REVIEW_LANES)
HUMAN_REVIEW_LANES = ("shadow", "priority", "standard", *REVIEW_LANES)
COMBINED_METRIC_FORMULA = (
    "user reviews Shadow, Priority, Standard, RecoveryShadow, RecoveryPriority, and "
    "DataQualityReview; confirms TP and rejects review FP; combined TP is Hard TP plus "
    "confirmed unique review TP, combined FP remains Hard FP"
)


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


def _require_full_checkpoint(
    output_root: Path, prediction_path: Path, manifest_path: Path
) -> dict[str, object]:
    if prediction_path != output_root / PREDICTION_FILENAME or manifest_path != output_root / PREDICTION_MANIFEST_FILENAME:
        raise ValueError("production evaluation requires the standard checkpoint paths")
    try:
        manifest = verify_prediction_checkpoint(output_root)
    except CheckpointError as exc:
        raise ValueError(f"strict checkpoint verification failed: {exc}") from exc
    if (
        manifest.get("complete_population") is not True
        or manifest.get("smoke") is not False
        or manifest.get("rows") != EXPECTED_DATASETS
        or manifest.get("datasets") != EXPECTED_DATASETS
    ):
        raise ValueError("production evaluation requires a complete 1,117-dataset checkpoint")
    return manifest


def _load_exp253_hard(
    prediction_path: Path, *, require_pinned_source: bool
) -> tuple[dict[str, set[int]], str]:
    prediction_path = Path(prediction_path)
    try:
        payload = prediction_path.read_bytes()
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load pinned Exp253 frozen source") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if require_pinned_source:
        if prediction_path != EXP253_PREDICTION or actual_sha256 != FROZEN_PREDICTION_HASHES["exp253"]:
            raise ValueError("pinned Exp253 prediction SHA256 mismatch")
        manifest_path = prediction_path.parent / "prediction_manifest.json"
        try:
            manifest_payload = manifest_path.read_bytes()
            manifest = json.loads(manifest_payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot load pinned Exp253 prediction manifest") from exc
        if (
            hashlib.sha256(manifest_payload).hexdigest()
            != FROZEN_PREDICTION_MANIFEST_HASHES["exp253"]
            or type(manifest) is not dict
            or manifest.get("prediction_sha256") != actual_sha256
            or manifest.get("datasets") != EXPECTED_DATASETS
        ):
            raise ValueError("pinned Exp253 prediction manifest mismatch")
    if any(type(row) is not dict for row in rows):
        raise ValueError("Exp253 frozen source contains a non-object row")
    hard_by_dataset: dict[str, set[int]] = {}
    for row in rows:
        dataset = row.get("dataset")
        indices = row.get("final_hard_indices")
        if type(dataset) is not str or not dataset or type(indices) is not list:
            raise ValueError("Exp253 frozen source row contract mismatch")
        if dataset in hard_by_dataset or any(type(index) is not int for index in indices):
            raise ValueError("Exp253 frozen source population mismatch")
        selected = set(indices)
        if len(selected) != len(indices):
            raise ValueError("Exp253 frozen source has duplicate Hard indices")
        hard_by_dataset[dataset] = selected
    if require_pinned_source and len(hard_by_dataset) != EXPECTED_DATASETS:
        raise ValueError("pinned Exp253 source is not the complete 1,117-dataset population")
    return hard_by_dataset, actual_sha256


def _require_hard_parity(
    prediction_rows: list[dict[str, object]],
    preflight: list[tuple[dict[str, object], dict[str, set[int]]]],
    exp253_hard: Mapping[str, set[int]],
    *,
    require_full_population: bool,
) -> int:
    prediction_names = {str(row["dataset"]) for row in prediction_rows}
    population_matches = (
        prediction_names == set(exp253_hard)
        if require_full_population
        else prediction_names <= set(exp253_hard)
    )
    if not population_matches:
        raise ValueError("Hard parity population mismatch with Exp253")
    mismatches = [
        str(row["dataset"])
        for row, lanes in preflight
        if lanes["hard"] != exp253_hard[str(row["dataset"])]
    ]
    if mismatches:
        raise ValueError(f"Hard parity mismatch with Exp253: datasets={mismatches[:5]}")
    return len(prediction_rows)


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
    for lane in HUMAN_REVIEW_LANES:
        count = total(f"{lane}_count")
        tp = total(f"{lane}_tp")
        fp = total(f"{lane}_fp")
        fn = total(f"{lane}_fn")
        precision = tp / count if count else 0.0
        recall = tp / total_anomalies if total_anomalies else 0.0
        summary.update(
            {
                f"{lane}_count": count,
                f"{lane}_tp": tp,
                f"{lane}_fp": fp,
                f"{lane}_fn": fn,
                f"{lane}_precision": precision,
                f"{lane}_recall": recall,
                f"{lane}_f1": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                ),
                f"{lane}_mean_dataset_f1": _mean(
                    float(row.get(f"{lane}_f1", 0.0)) for row in rows
                ),
            }
        )
    review_count = sum(int(summary[f"{lane}_count"]) for lane in HUMAN_REVIEW_LANES)
    combined_tp = total("combined_tp")
    combined_fp = total("combined_fp")
    combined_fn = total("combined_fn")
    combined_count = combined_tp + combined_fp
    summary.update(
        {
            "human_assisted_review_count": review_count,
            "recovery_review_count": total("recovery_review_count"),
            "human_assisted_confirmed_tp": total("combined_review_confirmed_tp"),
            "human_assisted_rejected_fp": total("combined_review_rejected_fp"),
            "recovered_unique_fn_count": total("recovered_unique_fn_count"),
            "datasets_with_added_candidates": sum(
                int(int(row.get("recovery_review_count", 0)) > 0) for row in rows
            ),
            "review_request_rate": review_count / total_test_runs if total_test_runs else 0.0,
            "combined_count": combined_count,
            "combined_tp": combined_tp,
            "combined_fp": combined_fp,
            "combined_fn": combined_fn,
            "combined_precision": combined_tp / combined_count if combined_count else 0.0,
            "combined_recall": combined_tp / total_anomalies if total_anomalies else 0.0,
            "mean_combined_f1": _mean(float(row.get("combined_f1", 0.0)) for row in rows),
            "hard_parity_datasets": total("hard_parity_with_exp253"),
            "hard_parity_mismatches": len(rows) - total("hard_parity_with_exp253"),
            "retrospective": True,
            "deployment_eligible": False,
            "autonomous_scope": "Hard only; unchanged Exp253 baseline",
            "human_assisted_scope": (
                "Shadow, Priority, Standard, RecoveryShadow, RecoveryPriority, and "
                "DataQualityReview; review only"
            ),
            "combined_metric_scope": "human-assisted diagnostic; never an autonomous metric",
            "combined_metric_formula": COMBINED_METRIC_FORMULA,
            "no_test_policy_selection": True,
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
    review = set().union(*(lanes[lane] for lane in HUMAN_REVIEW_LANES))
    recovery_review = set().union(*(lanes[lane] for lane in REVIEW_LANES))
    recovered = truth & (lanes["recovery_shadow"] | lanes["recovery_priority"])
    confirmed_review = review & truth
    rejected_review = review - truth
    combined = lanes["hard"] | confirmed_review
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
    for lane in HUMAN_REVIEW_LANES:
        output.update(_metric_columns(lane, lane_metrics(lanes[lane], labels)))
    output.update(_metric_columns("combined", lane_metrics(combined, labels)))
    output.update(
        {
            "human_assisted_review_count": len(review),
            "recovery_review_count": len(recovery_review),
            "combined_review_requested_count": len(review),
            "combined_review_confirmed_tp": len(confirmed_review),
            "combined_review_rejected_fp": len(rejected_review),
            "recovered_unique_fn_count": len(recovered),
            "review_request_rate": len(review) / len(labels) if labels else 0.0,
            "hard_parity_with_exp253": True,
            "combined_metric_formula": COMBINED_METRIC_FORMULA,
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
    exp253_prediction_path: Path = EXP253_PREDICTION,
    require_full_population: bool = True,
    progress_callback: Callable[[str, int, int], None] | None = None,
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
    manifest_path = Path(manifest_path)

    if require_full_population:
        _require_full_checkpoint(output_root, prediction_path, manifest_path)

    payload, prediction_sha256 = _verified_prediction_payload(prediction_path, manifest_path)
    prediction_rows = _parse_rows(payload)
    if require_full_population and (
        len(prediction_rows) != EXPECTED_DATASETS
        or len({str(row["dataset"]) for row in prediction_rows}) != EXPECTED_DATASETS
    ):
        raise ValueError("production evaluation requires exactly 1,117 unique prediction rows")
    preflight = [(row, _preflight_row(row)) for row in prediction_rows]
    exp253_hard, exp253_sha256 = _load_exp253_hard(
        exp253_prediction_path, require_pinned_source=require_full_population
    )
    hard_parity_datasets = _require_hard_parity(
        prediction_rows,
        preflight,
        exp253_hard,
        require_full_population=require_full_population,
    )

    evaluated: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    connection = _read_only_connection(database_path)
    try:
        connection.execute("PRAGMA query_only = ON")
        for done, (row, lanes) in enumerate(preflight, start=1):
            labels = labels_for_dataset(connection, str(row["dataset"]))
            result, diagnostic = _evaluate_row(row, lanes, labels, prediction_sha256)
            evaluated.append(result)
            diagnostics.append(diagnostic)
            if progress_callback is not None:
                progress_callback(str(row["dataset"]), done, len(preflight))
    finally:
        connection.close()

    summary = aggregate(evaluated)
    summary["prediction_sha256"] = prediction_sha256
    summary["exp253_prediction_sha256"] = exp253_sha256
    summary["hard_parity_datasets"] = hard_parity_datasets
    summary["hard_parity_mismatches"] = 0
    _atomic_text(output_root / RESULTS_FILENAME, _csv_text(evaluated))
    _atomic_text(output_root / SUMMARY_FILENAME, _csv_text([summary]))
    _atomic_text(output_root / DIAGNOSTICS_FILENAME, _csv_text(diagnostics))
    return evaluated, summary
