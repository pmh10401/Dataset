from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import (
    load_test_labels_for_evaluation,
    load_values_bundle,
    sha256_file,
)
from experiments.exp226_235_hard_alert_quality.preprocessing import align_bundle
from experiments.exp278_281_train_only_recovery.contracts import write_csv
from experiments.exp285_287_synthetic_shape_verifier.prediction import verify_prediction

from .comparison import (
    NEAR_DUPLICATE_CORRELATION_DISTANCE,
    NEAR_DUPLICATE_RMSE,
    NORMALIZED_EQUIVALENT_RMSE,
    PairComparison,
    compare_pair,
)


EXP286_ROOT = Path("/Users/minho/Documents/Dataset/exp285_287_synthetic_shape_verifier")
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
EXPERIMENT_ID = "experiment_289_recovery_priority_label_consistency_audit"
TIERS = ("raw_exact", "aligned_exact", "normalized_equivalent", "near_duplicate", "distinct", "unavailable")
CONFLICT_FIELDNAMES = (
    "dataset",
    "match_space",
    "fingerprint_sha256",
    "normal_indices",
    "anomaly_indices",
    "normal_count",
    "anomaly_count",
    "cross_label_pair_count",
    "retrospective_diagnostic_only",
)


def write_rows(
    path: Path,
    rows: Sequence[dict[str, object]],
    *,
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_frozen_candidates(
    prediction_rows: Iterable[dict[str, object]], expected: int = 43
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for row in prediction_rows:
        dataset = str(row["dataset"])
        for record in row["candidate_records"]:  # type: ignore[index]
            if (
                record["source_lane"] == "recovery_priority"
                and record["variant_pass"]["synthetic_shape_alpha_005"]
            ):
                index = int(record["candidate_index"])
                key = (dataset, index)
                if key in seen:
                    raise ValueError(f"duplicate frozen candidate: {key}")
                seen.add(key)
                selected.append({"dataset": dataset, **record})
    if len(selected) != expected:
        raise ValueError(f"expected {expected} frozen candidates, found {len(selected)}")
    return selected


def validate_candidate_index(dataset: str, index: int, n_test: int) -> None:
    if index < 0 or index >= n_test:
        raise IndexError(f"candidate index out of range: dataset={dataset} index={index} n_test={n_test}")


def _fingerprint(row: np.ndarray) -> str:
    values = np.asarray(row, dtype=np.float32).reshape(-1)
    digest = hashlib.sha256()
    digest.update(len(values).to_bytes(8, "little", signed=False))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def exact_cross_label_groups(
    normal_rows: Sequence[np.ndarray] | np.ndarray,
    anomaly_rows: Sequence[np.ndarray] | np.ndarray,
    *,
    normal_indices: Sequence[int],
    anomaly_indices: Sequence[int],
) -> list[dict[str, object]]:
    normal: dict[str, list[int]] = {}
    anomaly: dict[str, list[int]] = {}
    for index, row in zip(normal_indices, normal_rows, strict=True):
        normal.setdefault(_fingerprint(np.asarray(row)), []).append(int(index))
    for index, row in zip(anomaly_indices, anomaly_rows, strict=True):
        anomaly.setdefault(_fingerprint(np.asarray(row)), []).append(int(index))
    groups = []
    for fingerprint in sorted(set(normal) & set(anomaly)):
        groups.append(
            {
                "fingerprint_sha256": fingerprint,
                "normal_indices": json.dumps(normal[fingerprint], separators=(",", ":")),
                "anomaly_indices": json.dumps(anomaly[fingerprint], separators=(",", ":")),
                "normal_count": len(normal[fingerprint]),
                "anomaly_count": len(anomaly[fingerprint]),
                "cross_label_pair_count": len(normal[fingerprint]) * len(anomaly[fingerprint]),
            }
        )
    return groups


def summarize_candidate_tiers(rows: Iterable[dict[str, object]]) -> Counter[tuple[str, str, str]]:
    summary: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        outcome = str(row["outcome"])
        summary[(outcome, "train", str(row["train_tier"]))] += 1
        summary[(outcome, "opposite_test", str(row["opposite_test_tier"]))] += 1
    return summary


def _nearest_reference(
    candidate_raw: np.ndarray,
    candidate_aligned: np.ndarray,
    reference_raw: Sequence[np.ndarray],
    reference_aligned: np.ndarray,
    reference_indices: Sequence[int],
) -> tuple[int | None, PairComparison | None]:
    best: tuple[tuple[float, float, float, int], int, PairComparison] | None = None
    for raw, aligned, index in zip(reference_raw, reference_aligned, reference_indices, strict=True):
        comparison = compare_pair(candidate_raw, raw, candidate_aligned, aligned)
        key = (
            comparison.normalized_rmse,
            comparison.correlation_distance,
            comparison.aligned_rmse,
            int(index),
        )
        if best is None or key < best[0]:
            best = (key, int(index), comparison)
    if best is None:
        return None, None
    return best[1], best[2]


def _comparison_fields(prefix: str, comparison: PairComparison | None) -> dict[str, object]:
    if comparison is None:
        return {
            f"{prefix}_tier": "unavailable",
            f"{prefix}_raw_exact": False,
            f"{prefix}_aligned_exact": False,
            f"{prefix}_normalized_rmse": "",
            f"{prefix}_correlation_distance": "",
            f"{prefix}_aligned_rmse": "",
            f"{prefix}_aligned_max_abs": "",
        }
    return {f"{prefix}_{key}": value for key, value in comparison.as_dict().items()}


def _write_waveform(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_index", "value"])
        writer.writerows(enumerate(np.asarray(values, dtype=float)))


def run_audit(output_root: Path) -> dict[str, object]:
    prediction_path = EXP286_ROOT / "exp286_frozen_predictions.jsonl"
    prediction_manifest_path = EXP286_ROOT / "exp286_prediction_manifest.json"
    prediction_rows = verify_prediction(prediction_path, prediction_manifest_path)
    if len(prediction_rows) != 1117:
        raise ValueError(f"Exp286 prediction coverage mismatch: {len(prediction_rows)}/1117")
    frozen = select_frozen_candidates(prediction_rows)
    by_dataset: dict[str, list[dict[str, object]]] = {}
    for record in frozen:
        by_dataset.setdefault(str(record["dataset"]), []).append(record)

    candidate_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    waveform_cache: dict[tuple[str, int, str], np.ndarray] = {}

    for prediction_row in prediction_rows:
        dataset = str(prediction_row["dataset"])
        records = by_dataset.get(dataset, [])
        if not records:
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "candidate_count": 0,
                    "tp": 0,
                    "fp": 0,
                    "candidate_opposite_label_nondistinct": 0,
                    "raw_exact_conflict_groups": 0,
                    "aligned_exact_conflict_groups": 0,
                    "retrospective_diagnostic_only": True,
                }
            )
            continue

        raw_bundle = load_values_bundle(dataset)
        aligned_bundle = align_bundle(raw_bundle)
        labels = load_test_labels_for_evaluation(dataset)
        train_raw = list(raw_bundle.train_normal)
        test_raw = list(raw_bundle.test_values)
        train_aligned = np.asarray(aligned_bundle.train_normal, dtype=np.float32)
        test_aligned = np.asarray(aligned_bundle.test_values, dtype=np.float32)
        if len(labels) != len(test_raw):
            raise ValueError(f"TEST label/value count mismatch: dataset={dataset}")

        normal_indices = np.flatnonzero(labels == 0).astype(int).tolist()
        anomaly_indices = np.flatnonzero(labels == 1).astype(int).tolist()
        for match_space, normal_values, anomaly_values in (
            ("raw", [test_raw[i] for i in normal_indices], [test_raw[i] for i in anomaly_indices]),
            ("aligned", test_aligned[normal_indices], test_aligned[anomaly_indices]),
        ):
            groups = exact_cross_label_groups(
                normal_values,
                anomaly_values,
                normal_indices=normal_indices,
                anomaly_indices=anomaly_indices,
            )
            for group in groups:
                conflict_rows.append(
                    {
                        "dataset": dataset,
                        "match_space": match_space,
                        **group,
                        "retrospective_diagnostic_only": True,
                    }
                )

        local_rows = []
        for record in records:
            index = int(record["candidate_index"])
            validate_candidate_index(dataset, index, len(test_raw))
            label = int(labels[index])
            opposite_indices = normal_indices if label == 1 else anomaly_indices
            train_index, train_comparison = _nearest_reference(
                test_raw[index],
                test_aligned[index],
                train_raw,
                train_aligned,
                list(range(len(train_raw))),
            )
            opposite_index, opposite_comparison = _nearest_reference(
                test_raw[index],
                test_aligned[index],
                [test_raw[i] for i in opposite_indices],
                test_aligned[opposite_indices] if opposite_indices else np.empty((0, test_aligned.shape[1]), dtype=np.float32),
                opposite_indices,
            )
            row = {
                "dataset": dataset,
                "candidate_index": index,
                "true_label": label,
                "outcome": "TP" if label == 1 else "FP",
                "synthetic_shape_score": record["synthetic_shape_score"],
                "pvalue": record["pvalue"],
                "nearest_train_index": "" if train_index is None else train_index,
                "nearest_opposite_test_index": "" if opposite_index is None else opposite_index,
                **_comparison_fields("train", train_comparison),
                **_comparison_fields("opposite_test", opposite_comparison),
                "retrospective_diagnostic_only": True,
                "hard_policy_changed": False,
            }
            candidate_rows.append(row)
            local_rows.append(row)
            waveform_cache[(dataset, index, "candidate")] = test_aligned[index]
            if train_index is not None:
                waveform_cache[(dataset, index, "train")] = train_aligned[train_index]
            if opposite_index is not None:
                waveform_cache[(dataset, index, "opposite")] = test_aligned[opposite_index]

        raw_group_count = sum(1 for row in conflict_rows if row["dataset"] == dataset and row["match_space"] == "raw")
        aligned_group_count = sum(1 for row in conflict_rows if row["dataset"] == dataset and row["match_space"] == "aligned")
        dataset_rows.append(
            {
                "dataset": dataset,
                "candidate_count": len(local_rows),
                "tp": sum(row["outcome"] == "TP" for row in local_rows),
                "fp": sum(row["outcome"] == "FP" for row in local_rows),
                "candidate_opposite_label_nondistinct": sum(row["opposite_test_tier"] != "distinct" for row in local_rows),
                "raw_exact_conflict_groups": raw_group_count,
                "aligned_exact_conflict_groups": aligned_group_count,
                "retrospective_diagnostic_only": True,
            }
        )

    if len(candidate_rows) != 43:
        raise ValueError(f"candidate audit coverage mismatch: {len(candidate_rows)}/43")
    if Counter(row["outcome"] for row in candidate_rows) != Counter({"FP": 32, "TP": 11}):
        raise ValueError("frozen TP/FP population mismatch")
    if len(dataset_rows) != 1117:
        raise ValueError(f"dataset summary coverage mismatch: {len(dataset_rows)}/1117")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "exp289_candidate_label_consistency.csv", candidate_rows)
    write_csv(root / "exp289_dataset_summary.csv", dataset_rows)
    write_rows(
        root / "exp289_exact_cross_label_conflicts.csv",
        conflict_rows,
        fieldnames=CONFLICT_FIELDNAMES,
    )

    summary_counter = summarize_candidate_tiers(candidate_rows)
    summary_rows = []
    for outcome in ("TP", "FP"):
        for reference in ("train", "opposite_test"):
            for tier in TIERS:
                summary_rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "outcome": outcome,
                        "reference": reference,
                        "tier": tier,
                        "count": summary_counter[(outcome, reference, tier)],
                        "retrospective_diagnostic_only": True,
                        "hard_policy_changed": False,
                    }
                )
    write_csv(root / "exp289_tier_summary.csv", summary_rows)

    tier_priority = {name: order for order, name in enumerate(TIERS)}
    representative_rows = []
    for outcome in ("TP", "FP"):
        subset = [row for row in candidate_rows if row["outcome"] == outcome]
        subset.sort(key=lambda row: (tier_priority[str(row["opposite_test_tier"])], float(row["opposite_test_normalized_rmse"] or float("inf"))))
        for order, row in enumerate(subset[:5], 1):
            case_id = f"{outcome.lower()}_{order:02d}"
            dataset = str(row["dataset"])
            index = int(row["candidate_index"])
            for kind in ("candidate", "train", "opposite"):
                values = waveform_cache.get((dataset, index, kind))
                if values is not None:
                    _write_waveform(root / "cases" / f"{case_id}_{kind}.csv", values)
            representative_rows.append({"case_id": case_id, **row})
    write_csv(root / "exp289_representative_cases.csv", representative_rows)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_manifest_path": str(prediction_manifest_path),
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "database_path": str(DB_PATH),
        "database_sha256": sha256_file(DB_PATH),
        "prediction_dataset_count": len(prediction_rows),
        "candidate_count": len(candidate_rows),
        "candidate_dataset_count": len(by_dataset),
        "normalized_equivalent_rmse": NORMALIZED_EQUIVALENT_RMSE,
        "near_duplicate_rmse": NEAR_DUPLICATE_RMSE,
        "near_duplicate_correlation_distance": NEAR_DUPLICATE_CORRELATION_DISTANCE,
        "retrospective_diagnostic_only": True,
        "hard_policy_changed": False,
    }
    (root / "exp289_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "exact_cross_label_groups",
    "run_audit",
    "select_frozen_candidates",
    "summarize_candidate_tiers",
    "validate_candidate_index",
    "write_rows",
]
