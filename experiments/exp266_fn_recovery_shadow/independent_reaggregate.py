from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from .config import (
    DB_PATH,
    EXPECTED_DATASETS,
    EXPECTED_LANE_TOTALS,
    EXP253_PREDICTION,
    EXP264_PREDICTION,
    FROZEN_DATABASE_SHA256,
    FROZEN_PREDICTION_HASHES,
    FROZEN_PREDICTION_MANIFEST_HASHES,
)
from .database import DatabaseContractError, verify_database_sha256
from .prediction import CheckpointError, verify_prediction_checkpoint


RESULTS_FILENAME = "05_dataset_results.csv"
SUMMARY_FILENAME = "06_global_summary.csv"
OUTPUT_FILENAME = "10_independent_reaggregation.json"
REVIEW_LANES = ("recovery_shadow", "recovery_priority", "data_quality_review")
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
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{Path(path).name} has no rows")
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_csv(results_path: Path) -> dict[str, object]:
    """Recompute summary fields from persisted rows without importing evaluation code."""

    rows = _read_csv(results_path)

    def integer(row: Mapping[str, str], field: str) -> int:
        try:
            return int(row[field])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"results row has invalid {field}") from exc

    def total(field: str) -> int:
        return sum(integer(row, field) for row in rows)

    hard_alerts = total("hard_alerts")
    baseline_hard_alerts = total("baseline_hard_alerts")
    hard_tp = total("hard_tp")
    hard_fp = total("hard_fp")
    total_anomalies = total("true_anomaly_count")
    total_test_runs = total("total_test_runs")
    result: dict[str, object] = {
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
        "hard_mean_dataset_f1": _mean(float(row["hard_f1"]) for row in rows),
        "hard_zero_f1_datasets": sum(float(row["hard_f1"]) == 0.0 for row in rows),
    }
    result["baseline_union_count"] = sum(
        int(result[field])
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
        result.update(
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
                    float(row[f"{lane}_f1"]) for row in rows
                ),
            }
        )
    review_count = sum(int(result[f"{lane}_count"]) for lane in HUMAN_REVIEW_LANES)
    exp266_incremental_review_count = total("recovery_review_count")
    combined_tp = total("combined_tp")
    combined_fp = total("combined_fp")
    combined_fn = total("combined_fn")
    combined_count = combined_tp + combined_fp
    result.update(
        {
            "human_assisted_review_count": review_count,
            "recovery_review_count": exp266_incremental_review_count,
            "human_assisted_confirmed_tp": total("combined_review_confirmed_tp"),
            "human_assisted_rejected_fp": total("combined_review_rejected_fp"),
            "recovered_unique_fn_count": total("recovered_unique_fn_count"),
            "datasets_with_added_candidates": sum(
                integer(row, "recovery_review_count") > 0 for row in rows
            ),
            "total_existing_plus_exp266_review_request_rate": (
                review_count / total_test_runs if total_test_runs else 0.0
            ),
            "exp266_incremental_review_request_rate": (
                exp266_incremental_review_count / total_test_runs
                if total_test_runs
                else 0.0
            ),
            "combined_count": combined_count,
            "combined_tp": combined_tp,
            "combined_fp": combined_fp,
            "combined_fn": combined_fn,
            "combined_precision": combined_tp / combined_count if combined_count else 0.0,
            "combined_recall": combined_tp / total_anomalies if total_anomalies else 0.0,
            "mean_combined_f1": _mean(float(row["combined_f1"]) for row in rows),
            "hard_parity_datasets": sum(
                row.get("hard_parity_with_exp253") == "True" for row in rows
            ),
            "hard_parity_mismatches": sum(
                row.get("hard_parity_with_exp253") != "True" for row in rows
            ),
            "combined_metric_formula": COMBINED_METRIC_FORMULA,
        }
    )
    for lane, field in (
        ("shadow", "shadow_parity_with_exp264"),
        ("priority", "priority_parity_with_exp264"),
        ("standard", "standard_parity_with_exp264"),
    ):
        result[f"{lane}_parity_datasets"] = sum(
            row.get(field) == "True" for row in rows
        )
        result[f"{lane}_parity_mismatches"] = sum(
            row.get(field) != "True" for row in rows
        )
    prediction_hashes = {row.get("prediction_sha256", "") for row in rows}
    if len(prediction_hashes) != 1 or "" in prediction_hashes:
        raise ValueError("results rows do not share one prediction SHA256")
    result["prediction_sha256"] = prediction_hashes.pop()
    return result


def numerically_equal(stored: object, recomputed: object) -> bool:
    try:
        return math.isclose(float(stored), float(recomputed), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return stored == recomputed


def verify_summary(results_path: Path, summary_path: Path) -> dict[str, object]:
    recomputed = aggregate_csv(results_path)
    summary_rows = _read_csv(summary_path)
    if len(summary_rows) != 1:
        raise ValueError("06_global_summary.csv must contain exactly one row")
    stored = summary_rows[0]
    missing = sorted(set(recomputed) - set(stored))
    mismatches = {
        key: {"stored": stored.get(key), "recomputed": recomputed[key]}
        for key in recomputed
        if key not in stored or not numerically_equal(stored[key], recomputed[key])
    }
    return {
        "passed": not mismatches,
        "datasets": recomputed["datasets"],
        "compared_fields": list(recomputed),
        "missing_fields": missing,
        "mismatches": mismatches,
        "recomputed": recomputed,
    }


def _load_exp253_hard(
    prediction_path: Path, *, require_pinned_source: bool
) -> tuple[dict[str, set[int]], str]:
    prediction_path = Path(prediction_path)
    try:
        payload = prediction_path.read_bytes()
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load pinned Exp253 frozen source") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if require_pinned_source:
        manifest_path = prediction_path.parent / "prediction_manifest.json"
        try:
            manifest_payload = manifest_path.read_bytes()
            manifest = json.loads(manifest_payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot load pinned Exp253 prediction manifest") from exc
        if (
            prediction_path != EXP253_PREDICTION
            or actual != FROZEN_PREDICTION_HASHES["exp253"]
            or hashlib.sha256(manifest_payload).hexdigest()
            != FROZEN_PREDICTION_MANIFEST_HASHES["exp253"]
            or type(manifest) is not dict
            or manifest.get("prediction_sha256") != actual
            or manifest.get("datasets") != EXPECTED_DATASETS
        ):
            raise ValueError("pinned Exp253 prediction contract mismatch")
    hard: dict[str, set[int]] = {}
    for row in rows:
        if type(row) is not dict:
            raise ValueError("Exp253 frozen source row is invalid")
        dataset = row.get("dataset")
        values = row.get("final_hard_indices")
        if type(dataset) is not str or not dataset or type(values) is not list:
            raise ValueError("Exp253 frozen source row contract mismatch")
        if dataset in hard or any(type(index) is not int for index in values):
            raise ValueError("Exp253 frozen source population mismatch")
        hard[dataset] = set(values)
        if len(hard[dataset]) != len(values):
            raise ValueError("Exp253 frozen source has duplicate Hard indices")
    if require_pinned_source and len(hard) != EXPECTED_DATASETS:
        raise ValueError("pinned Exp253 source is not complete")
    return hard, actual


def _load_exp264_lanes(
    prediction_path: Path, *, require_pinned_source: bool
) -> tuple[dict[str, dict[str, set[int]]], str]:
    prediction_path = Path(prediction_path)
    try:
        payload = prediction_path.read_bytes()
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load pinned Exp264 frozen source") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if require_pinned_source:
        manifest_path = prediction_path.parent / "prediction_manifest.json"
        try:
            manifest_payload = manifest_path.read_bytes()
            manifest = json.loads(manifest_payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot load pinned Exp264 prediction manifest") from exc
        if (
            prediction_path != EXP264_PREDICTION
            or actual != FROZEN_PREDICTION_HASHES["exp264"]
            or hashlib.sha256(manifest_payload).hexdigest()
            != FROZEN_PREDICTION_MANIFEST_HASHES["exp264"]
            or type(manifest) is not dict
            or manifest.get("prediction_sha256") != actual
            or manifest.get("datasets") != EXPECTED_DATASETS
        ):
            raise ValueError("pinned Exp264 prediction contract mismatch")
    mappings = {lane: {} for lane in ("shadow", "priority", "standard")}
    fields = {
        "shadow": "promoted_indices",
        "priority": "final_priority_indices",
        "standard": "final_standard_indices",
    }
    for row in rows:
        if type(row) is not dict:
            raise ValueError("Exp264 frozen source row is invalid")
        dataset = row.get("dataset")
        if type(dataset) is not str or not dataset:
            raise ValueError("Exp264 frozen source row contract mismatch")
        for lane, field in fields.items():
            values = row.get(field)
            if type(values) is not list or any(type(index) is not int for index in values):
                raise ValueError("Exp264 frozen source row contract mismatch")
            selected = set(values)
            if dataset in mappings[lane] or len(selected) != len(values):
                raise ValueError("Exp264 frozen source population mismatch")
            mappings[lane][dataset] = selected
    if require_pinned_source and any(
        len(mapping) != EXPECTED_DATASETS for mapping in mappings.values()
    ):
        raise ValueError("pinned Exp264 source is not complete")
    return mappings, actual


def _index_mapping_sha256(mapping: Mapping[str, set[int]]) -> str:
    payload = [
        {"dataset": dataset, "indices": sorted(mapping[dataset])}
        for dataset in sorted(mapping)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _prediction_audit(
    output_root: Path,
    *,
    exp253_prediction_path: Path,
    exp264_prediction_path: Path,
    database_path: Path,
    expected_database_sha256: str,
    require_full_population: bool,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    try:
        database_sha256 = verify_database_sha256(
            database_path, expected_database_sha256
        )
    except DatabaseContractError as exc:
        raise ValueError(str(exc)) from exc
    prediction = output_root / "03_frozen_predictions.jsonl"
    manifest_path = output_root / "04_prediction_manifest.json"
    if require_full_population:
        try:
            strict_manifest = verify_prediction_checkpoint(output_root)
        except CheckpointError as exc:
            raise ValueError(f"strict checkpoint verification failed: {exc}") from exc
        if (
            strict_manifest.get("complete_population") is not True
            or strict_manifest.get("smoke") is not False
            or strict_manifest.get("rows") != EXPECTED_DATASETS
            or strict_manifest.get("datasets") != EXPECTED_DATASETS
        ):
            raise ValueError("independent reaggregation requires complete 1,117 manifest fields")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = prediction.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot verify frozen prediction for reaggregation") from exc
    if type(manifest) is not dict or manifest.get("prediction_sha256") != actual:
        raise ValueError("independent reaggregation prediction SHA256 mismatch")
    if any(type(row) is not dict for row in rows):
        raise ValueError("independent reaggregation prediction row is invalid")
    if require_full_population and (
        len(rows) != EXPECTED_DATASETS
        or len({str(row.get("dataset", "")) for row in rows}) != EXPECTED_DATASETS
    ):
        raise ValueError("independent reaggregation requires exactly 1,117 unique rows")

    exp253_hard, exp253_sha256 = _load_exp253_hard(
        exp253_prediction_path, require_pinned_source=require_full_population
    )
    exp264_lanes, exp264_sha256 = _load_exp264_lanes(
        exp264_prediction_path, require_pinned_source=require_full_population
    )
    expected_by_lane: dict[str, Mapping[str, set[int]]] = {
        "hard": exp253_hard,
        **exp264_lanes,
    }

    result_rows = _read_csv(output_root / RESULTS_FILENAME)
    result_sizes = {row["dataset"]: int(row["total_test_runs"]) for row in result_rows}
    prediction_names = [str(row.get("dataset", "")) for row in rows]
    parity_population_matches = all(
        (
            set(prediction_names) == set(expected)
            if require_full_population
            else set(prediction_names) <= set(expected)
        )
        for expected in expected_by_lane.values()
    )
    expected_rows = manifest.get("rows", len(rows))
    if type(expected_rows) is not int:
        expected_rows = len(rows)
    missing_datasets = max(expected_rows - len(set(prediction_names)), 0)
    missing_datasets += len(set(prediction_names) - set(result_sizes))
    missing_datasets += len(set(result_sizes) - set(prediction_names))
    missing_datasets += len(prediction_names) - len(set(prediction_names))

    lane_names = (
        "hard",
        "shadow",
        "priority",
        "standard",
        "recovery_shadow",
        "recovery_priority",
        "data_quality_review",
    )
    lane_totals = {lane: 0 for lane in lane_names}
    index_errors = 0
    lane_overlaps = 0
    prediction_label_fields = 0
    prediction_evaluation_fields = 0
    parity_datasets = {lane: 0 for lane in ("hard", "shadow", "priority", "standard")}
    parity_mismatches = {lane: 0 for lane in parity_datasets}
    actual_lane_mappings = {lane: {} for lane in parity_datasets}
    label_keys = {"y_test", "label", "labels", "test_labels", "anomaly_positions"}
    metric_keys = {"tp", "fp", "fn", "precision", "recall", "f1"}

    def forbidden_counts(value: object) -> tuple[int, int]:
        labels = metrics = 0
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                labels += lowered in label_keys
                metrics += lowered in metric_keys
                child_labels, child_metrics = forbidden_counts(child)
                labels += child_labels
                metrics += child_metrics
        elif isinstance(value, list):
            for child in value:
                child_labels, child_metrics = forbidden_counts(child)
                labels += child_labels
                metrics += child_metrics
        return labels, metrics

    for done, row in enumerate(rows, start=1):
        dataset = str(row.get("dataset", ""))
        n_test = result_sizes.get(dataset, 0)
        lanes: dict[str, set[int]] = {}
        for lane in lane_names:
            values = row.get(f"{lane}_indices")
            if type(values) is not list:
                index_errors += 1
                values = []
            if any(type(index) is not int for index in values):
                index_errors += 1
            selected = {index for index in values if type(index) is int}
            if len(selected) != len(values):
                index_errors += 1
            index_errors += sum(index < 0 or index >= n_test for index in selected)
            lanes[lane] = selected
            lane_totals[lane] += len(selected)
        for lane in parity_datasets:
            actual_lane_mappings[lane][dataset] = lanes[lane]
            expected = expected_by_lane[lane]
            if dataset in expected and lanes[lane] == expected[dataset]:
                parity_datasets[lane] += 1
            else:
                parity_mismatches[lane] += 1
        for field in ("quality_flag_indices",):
            values = row.get(field, [])
            if type(values) is not list:
                index_errors += 1
                values = []
            index_errors += sum(
                type(index) is not int or index < 0 or index >= n_test for index in values
            )
        support = row.get("support_by_index", {})
        if type(support) is not dict:
            index_errors += 1
        else:
            for index, count in support.items():
                try:
                    parsed_index = int(index)
                    parsed_count = int(count)
                except (TypeError, ValueError):
                    index_errors += 1
                    continue
                index_errors += parsed_index < 0 or parsed_index >= n_test
                index_errors += parsed_count < 1 or parsed_count > 5
        sources = row.get("support_sources")
        if sources is not None:
            if type(sources) is not dict:
                index_errors += 1
            else:
                for values in sources.values():
                    if type(values) is not list:
                        index_errors += 1
                        continue
                    index_errors += sum(
                        type(index) is not int or index < 0 or index >= n_test
                        for index in values
                    )
        for left, lane in enumerate(lane_names):
            for other in lane_names[left + 1 :]:
                lane_overlaps += len(lanes[lane] & lanes[other])
        labels, metrics = forbidden_counts(row)
        prediction_label_fields += labels
        prediction_evaluation_fields += metrics
        prediction_label_fields += row.get("contains_test_labels") is not False
        prediction_evaluation_fields += row.get("contains_evaluation_metrics") is not False
        if progress_callback is not None:
            progress_callback(dataset, done, len(rows))

    baseline = {lane: lane_totals[lane] for lane in ("hard", "shadow", "priority", "standard")}
    baseline["union"] = sum(baseline.values())
    full_contract_passed = True
    if manifest.get("complete_population") is True:
        full_contract_passed = baseline == EXPECTED_LANE_TOTALS
    return {
        "actual_sha256": actual,
        "expected_sha256": str(manifest.get("prediction_sha256")),
        "baseline_lane_totals": baseline,
        "baseline_lane_contract_passed": full_contract_passed,
        "prediction_label_fields": prediction_label_fields,
        "prediction_evaluation_fields": prediction_evaluation_fields,
        "missing_datasets": missing_datasets,
        "index_errors": index_errors,
        "lane_overlaps": lane_overlaps,
        "database_sha256": database_sha256,
        "exp253_prediction_sha256": exp253_sha256,
        "exp264_prediction_sha256": exp264_sha256,
        "immutable_lane_parity_population_matches": parity_population_matches,
        **{
            key: value
            for lane in parity_datasets
            for key, value in (
                (f"{lane}_parity_datasets", parity_datasets[lane]),
                (f"{lane}_parity_mismatches", parity_mismatches[lane]),
                (
                    f"{lane}_actual_indices_sha256",
                    _index_mapping_sha256(actual_lane_mappings[lane]),
                ),
                (
                    f"{lane}_expected_indices_sha256",
                    _index_mapping_sha256(
                        {
                            dataset: expected_by_lane[lane][dataset]
                            for dataset in set(prediction_names)
                            if dataset in expected_by_lane[lane]
                        }
                    ),
                ),
            )
        },
    }


def write_independent_reaggregation(
    output_root: Path,
    *,
    exp253_prediction_path: Path = EXP253_PREDICTION,
    exp264_prediction_path: Path = EXP264_PREDICTION,
    database_path: Path = DB_PATH,
    expected_database_sha256: str = FROZEN_DATABASE_SHA256,
    require_full_population: bool = True,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    output_root = Path(output_root)
    audit = _prediction_audit(
        output_root,
        exp253_prediction_path=exp253_prediction_path,
        exp264_prediction_path=exp264_prediction_path,
        database_path=database_path,
        expected_database_sha256=expected_database_sha256,
        require_full_population=require_full_population,
        progress_callback=progress_callback,
    )
    actual = str(audit["actual_sha256"])
    expected = str(audit["expected_sha256"])
    verification = verify_summary(
        output_root / RESULTS_FILENAME, output_root / SUMMARY_FILENAME
    )
    recomputed = verification["recomputed"]
    evaluation_prediction_sha256_matches = recomputed["prediction_sha256"] == actual
    audit_passed = (
        actual == expected
        and evaluation_prediction_sha256_matches
        and bool(audit["baseline_lane_contract_passed"])
        and int(audit["prediction_label_fields"]) == 0
        and int(audit["prediction_evaluation_fields"]) == 0
        and int(audit["missing_datasets"]) == 0
        and int(audit["index_errors"]) == 0
        and int(audit["lane_overlaps"]) == 0
        and bool(audit["immutable_lane_parity_population_matches"])
        and all(
            int(audit[f"{lane}_parity_datasets"]) == int(recomputed["datasets"])
            and int(audit[f"{lane}_parity_mismatches"]) == 0
            and audit[f"{lane}_actual_indices_sha256"]
            == audit[f"{lane}_expected_indices_sha256"]
            for lane in ("hard", "shadow", "priority", "standard")
        )
    )
    payload = {
        "passed": bool(verification["passed"] and audit_passed),
        "retrospective": True,
        "read_only_reaggregation": True,
        "point_matching": "exact dataset TEST instance_index; no adjacency merging",
        "zero_denominator_policy": 0.0,
        "prediction_sha256": actual,
        "prediction_sha256_verified": actual == expected,
        "evaluation_prediction_sha256_matches": evaluation_prediction_sha256_matches,
        **{
            key: value
            for key, value in audit.items()
            if key not in {"actual_sha256", "expected_sha256"}
        },
        "summary_verification": {
            key: value for key, value in verification.items() if key != "recomputed"
        },
        "recomputed_summary": recomputed,
        "hard_review_separation": {
            "passed": recomputed["hard_alert_delta"] == 0,
            "hard_alerts": recomputed["hard_alerts"],
            "baseline_hard_alerts": recomputed["baseline_hard_alerts"],
            "human_assisted_review_count": recomputed["human_assisted_review_count"],
        },
        "combined_metric_scope": "human-assisted diagnostic; never an autonomous metric",
        "combined_metric_formula": COMBINED_METRIC_FORMULA,
    }
    _atomic_text(
        output_root / OUTPUT_FILENAME,
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independently reaggregate Exp266 results")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = write_independent_reaggregation(args.output_root)
    totals = payload["baseline_lane_totals"]
    print(
        f"Hard={totals['hard']}, Shadow={totals['shadow']}, "
        f"Priority={totals['priority']}, Standard={totals['standard']}"
    )
    print(f"baseline lane delta={0 if payload['baseline_lane_contract_passed'] else 1}")
    print(f"prediction label fields={payload['prediction_label_fields']}")
    print(f"prediction evaluation fields={payload['prediction_evaluation_fields']}")
    print(f"missing datasets={payload['missing_datasets']}")
    print(f"index errors={payload['index_errors']}")
    print(f"lane overlaps={payload['lane_overlaps']}")
    for lane in ("hard", "shadow", "priority", "standard"):
        print(
            f"{lane.title()} parity={payload[f'{lane}_parity_datasets']}/"
            f"{payload['recomputed_summary']['datasets']}, "
            f"mismatches={payload[f'{lane}_parity_mismatches']}, "
            f"hash={payload[f'{lane}_actual_indices_sha256']}"
        )
    print(f"independent reaggregation={'PASS' if payload['passed'] else 'FAIL'}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
