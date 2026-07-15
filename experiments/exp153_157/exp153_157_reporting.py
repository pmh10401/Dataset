"""Artifact IO, freezing and label-separated evaluation for Exp153--157."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

import exp153_157_core as core

FORBIDDEN_PREDICTION_KEYS = {
    "y_test",
    "hard_tp",
    "hard_fp",
    "hard_f1",
    "combined_f1",
    "standard_review_tp",
    "priority_review_tp",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = [dict(row) for row in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = dict(row)
            forbidden = sorted(FORBIDDEN_PREDICTION_KEYS & set(payload))
            if forbidden:
                raise ValueError(f"prediction row contains evaluation fields: {forbidden}")
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def freeze_prediction_file(
    prediction_path: Path,
    manifest_path: Path,
    *,
    experiment_ids: Sequence[str],
    row_count: int,
    input_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Freeze a label-free prediction artifact before evaluation."""

    rows = read_jsonl(prediction_path)
    if len(rows) != int(row_count):
        raise ValueError(f"prediction row count mismatch {len(rows)} != {row_count}")
    for row in rows:
        forbidden = sorted(FORBIDDEN_PREDICTION_KEYS & set(row))
        if forbidden:
            raise ValueError(f"prediction contains forbidden keys: {forbidden}")
    manifest = {
        "prediction_path": str(Path(prediction_path)),
        "prediction_sha256": core.sha256_file(prediction_path),
        "row_count": len(rows),
        "experiment_ids": list(experiment_ids),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "input_hashes": dict(input_hashes or {}),
    }
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def total(key: str) -> int:
        return int(sum(float(row.get(key, 0) or 0) for row in rows))

    def mean(key: str) -> float:
        return float(np.mean([float(row.get(key, 0) or 0) for row in rows])) if rows else 0.0

    hard_tp, hard_fp = total("hard_tp"), total("hard_fp")
    standard_tp, standard_fp = total("standard_review_tp"), total("standard_review_fp")
    priority_tp, priority_fp = total("priority_review_tp"), total("priority_review_fp")
    return {
        "datasets": len(rows),
        "hard_alerts": total("hard_count"),
        "hard_tp": hard_tp,
        "hard_fp": hard_fp,
        "hard_precision": hard_tp / max(1, hard_tp + hard_fp),
        "mean_hard_recall": mean("hard_recall"),
        "mean_hard_f1": mean("hard_f1"),
        "standard_review_candidates": total("standard_review_count"),
        "standard_review_tp": standard_tp,
        "standard_review_fp": standard_fp,
        "standard_review_precision": standard_tp / max(1, standard_tp + standard_fp),
        "priority_review_candidates": total("priority_review_count"),
        "priority_review_tp": priority_tp,
        "priority_review_fp": priority_fp,
        "priority_review_precision": priority_tp / max(1, priority_tp + priority_fp),
        "mean_combined_f1": mean("combined_f1"),
        "mean_combined_f1_scope": "human_assisted_diagnostic_only",
        "abstained_datasets": total("abstained"),
    }


def evaluate_prediction_file(
    prediction_path: Path,
    prediction_manifest_path: Path,
    result_csv: Path,
    summary_csv: Path,
    summary_md: Path,
    *,
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    """Evaluate only after verifying the frozen prediction SHA-256."""

    from exp153_157_bridge import load_test_labels

    manifest = json.loads(Path(prediction_manifest_path).read_text(encoding="utf-8"))
    core.verify_frozen_file(prediction_path, manifest["prediction_sha256"])
    predictions = read_jsonl(prediction_path)
    results = []
    labels_cache: dict[str, np.ndarray] = {}
    for row in predictions:
        dataset = str(row["dataset_name"])
        if dataset not in labels_cache:
            labels_cache[dataset] = load_test_labels(dataset)
        labels = labels_cache[dataset]
        hard = core.parse_indices(row.get("hard_indices"))
        standard = core.parse_indices(row.get("standard_review_indices"))
        priority = core.parse_indices(row.get("priority_review_indices"))
        if hard & standard or hard & priority or standard & priority:
            raise ValueError(f"lane overlap dataset={dataset}")
        candidate = core.parse_indices(row.get("candidate_indices"))
        if candidate != hard | standard | priority:
            raise ValueError(f"candidate/lane union mismatch dataset={dataset}")
        hard_m = core.tier_metrics(labels, hard)
        standard_m = core.tier_metrics(labels, standard)
        priority_m = core.tier_metrics(labels, priority)
        combined_m = core.tier_metrics(labels, candidate)
        evaluated = dict(row)
        for prefix, metrics in (
            ("hard", hard_m),
            ("standard_review", standard_m),
            ("priority_review", priority_m),
            ("combined", combined_m),
        ):
            evaluated.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        evaluated["mean_combined_f1_scope"] = "human_assisted_diagnostic_only"
        evaluated["retrospective_counterfactual"] = 1
        evaluated["prospective_validated"] = 0
        results.append(evaluated)

    write_rows(result_csv, results)
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in results:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)
    summaries = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary.update(_aggregate(group))
        summaries.append(summary)
    write_rows(summary_csv, summaries)
    Path(summary_md).write_text(
        "# Evaluation summary\n\n"
        "Hard alert metrics are autonomous. Standard/Priority review and combined F1 "
        "are human-assisted diagnostics. No TEST metric selects a variant.\n\n```json\n"
        + json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    return summaries
