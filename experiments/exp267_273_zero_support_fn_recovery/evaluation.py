"""Post-hash TEST evaluation with autonomous and review metrics separated."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from .recovery_routing import (
    _load_pinned_exp266,
    verify_recovery_prediction_hash,
)


class PredictionHashError(ValueError):
    """Raised before TEST labels when a frozen prediction fails verification."""


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
    finally:
        try:
            os.unlink(raw)
        except FileNotFoundError:
            pass


def _metrics(indices: set[int], truth: set[int]) -> dict[str, int | float]:
    tp = len(indices & truth)
    fp = len(indices - truth)
    fn = len(truth - indices)
    precision = tp / len(indices) if indices else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": len(indices), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def _default_truth_loader(expected: tuple[str, ...]) -> Mapping[str, set[int]]:
    from .diagnostics import load_test_truth

    truth, _ = load_test_truth(expected)
    return truth


def _tier(n_train: int) -> str:
    if n_train < 5:
        return "lt_5"
    if n_train < 20:
        return "5_19"
    if n_train < 100:
        return "20_99"
    if n_train < 200:
        return "100_199"
    if n_train < 500:
        return "200_499"
    return "ge_500"


def _test_tier(n_test: int) -> str:
    if n_test < 100:
        return "lt_100"
    if n_test < 250:
        return "100_249"
    if n_test < 500:
        return "250_499"
    return "ge_500"


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    total = lambda field: sum(int(row[field]) for row in rows)
    hard_count, hard_tp, hard_fp = total("hard_count"), total("hard_tp"), total("hard_fp")
    anomalies = total("true_anomaly_count")
    review_count, review_tp, review_fp = total("review_count"), total("review_tp"), total("review_fp")
    hard_precision = hard_tp / hard_count if hard_count else 0.0
    hard_recall = hard_tp / anomalies if anomalies else 0.0
    review_precision = review_tp / review_count if review_count else 0.0
    confirmed = hard_tp + review_tp
    combined_precision = confirmed / (hard_count + review_tp) if hard_count + review_tp else 0.0
    combined_recall = confirmed / anomalies if anomalies else 0.0
    return {
        "datasets": len(rows),
        "total_anomalies": anomalies,
        "autonomous": {
            "hard_alerts": hard_count, "hard_tp": hard_tp, "hard_fp": hard_fp,
            "hard_fn": anomalies - hard_tp, "hard_precision": hard_precision,
            "hard_recall": hard_recall,
            "mean_hard_f1": sum(float(row["hard_f1"]) for row in rows) / len(rows) if rows else 0.0,
            "hard_alert_delta_from_exp253": hard_count - 2253 if len(rows) == 1117 else 0,
        },
        "human_assisted_review": {
            "review_candidates": review_count, "review_tp": review_tp, "review_fp": review_fp,
            "review_precision": review_precision, "human_assisted_only": True,
        },
        "human_assisted_diagnostic": {
            "combined_confirmed_tp": confirmed,
            "combined_fp": hard_fp,
            "combined_precision": combined_precision,
            "combined_recall": combined_recall,
            "mean_combined_f1": sum(float(row["combined_f1"]) for row in rows) / len(rows) if rows else 0.0,
            "human_assisted_only": True,
        },
        "lane_overlap_count": total("lane_overlap_count"),
        "retrospective": True,
        "prospective": False,
    }


def evaluate_frozen_predictions(
    prediction_path: Path,
    manifest_path: Path,
    *,
    expected_datasets: Iterable[str],
    truth_loader: Callable[[tuple[str, ...]], Mapping[str, set[int]]] | None = None,
    output_dir: Path | None = None,
    source_prediction_path: Path | None = None,
    source_prediction_manifest_path: Path | None = None,
    membership_path: Path | None = None,
    cache_root: Path | None = None,
    source_runtime: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Verify all label-free artifacts, then and only then open TEST truth."""
    expected = tuple(sorted(str(value) for value in expected_datasets))
    try:
        verify_recovery_prediction_hash(
            prediction_path, manifest_path, expected_datasets=expected,
        )
        payload = Path(prediction_path).read_bytes()
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
        if len(expected) == 1117:
            if any(value is None for value in (
                source_prediction_path, source_prediction_manifest_path, membership_path, cache_root,
            )):
                raise ValueError("full evaluation requires source prediction, membership, and cache artifacts")
            from .qualification import _read_prediction_rows, _verify_membership
            from .source_pipeline import verify_prediction_hash

            verify_prediction_hash(
                Path(source_prediction_path), Path(source_prediction_manifest_path),
                expected_datasets=expected, cache_root=Path(cache_root),
                source_runtime=source_runtime,
            )
            _, source_prediction_hash, source_manifest_hash = _read_prediction_rows(
                Path(source_prediction_path), Path(source_prediction_manifest_path),
                expected_datasets=expected,
            )
            membership = _verify_membership(
                Path(membership_path), prediction_hash=source_prediction_hash,
            )
            exp272_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            recorded = exp272_manifest.get("input_sha256")
            if not isinstance(recorded, Mapping) or any(
                recorded.get(key) != value for key, value in {
                    "new_prediction": source_prediction_hash,
                    "new_prediction_manifest": source_manifest_hash,
                    "membership": membership["membership_sha256"],
                }.items()
            ):
                raise ValueError("Exp272 source/membership input hashes do not match")
            frozen, _, _ = _load_pinned_exp266()
            for row in rows:
                item = frozen.datasets[str(row["dataset"])]
                if row["hard_indices"] != sorted(item.hard):
                    raise ValueError("Exp253 Hard parity mismatch")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PredictionHashError(str(error)) from error
    loader = truth_loader or _default_truth_loader
    truth_by_dataset = loader(expected)
    if set(truth_by_dataset) != set(expected):
        raise ValueError("TEST truth population mismatch")
    result_rows: list[dict[str, object]] = []
    review_fields = (
        "shadow_indices", "priority_indices", "standard_indices",
        "recovery_shadow_indices", "recovery_priority_indices", "data_quality_review_indices",
    )
    for row in rows:
        dataset = str(row["dataset"])
        n_test = int(row["n_test"])
        truth = set(truth_by_dataset[dataset])
        if any(type(index) is not int or index < 0 or index >= n_test for index in truth):
            raise ValueError(f"TEST truth index outside range: dataset={dataset}")
        hard = set(row["hard_indices"])
        review_sets = [set(row[field]) for field in review_fields]
        review = set().union(*review_sets)
        overlap = len(hard & review)
        for offset, values in enumerate(review_sets):
            overlap += sum(len(values & other) for other in review_sets[offset + 1 :])
        if overlap:
            raise ValueError(f"lane overlap after freeze: dataset={dataset}")
        hard_metrics = _metrics(hard, truth)
        review_metrics = _metrics(review, truth)
        combined_indices = hard | (review & truth)
        combined_metrics = _metrics(combined_indices, truth)
        output: dict[str, object] = {
            "dataset": dataset, "n_train": int(row["n_train"]), "n_test": n_test,
            "train_tier": _tier(int(row["n_train"])), "test_tier": _test_tier(n_test),
            "true_anomaly_count": len(truth), "lane_overlap_count": overlap,
            **{f"hard_{key}": value for key, value in hard_metrics.items()},
            **{f"review_{key}": value for key, value in review_metrics.items()},
            "combined_f1": combined_metrics["f1"],
        }
        for field, values in zip(review_fields, review_sets, strict=True):
            lane = field.removesuffix("_indices")
            metrics = _metrics(values, truth)
            output.update({f"{lane}_{key}": value for key, value in metrics.items()})
        result_rows.append(output)
    summary = _aggregate(result_rows)
    summary["diagnostic_slices"] = {
        axis: {
            tier: _aggregate([row for row in result_rows if row[axis] == tier])
            for tier in sorted({str(row[axis]) for row in result_rows})
        }
        for axis in ("train_tier", "test_tier")
    }
    output: dict[str, object] = {"rows": result_rows, "summary": summary}
    if output_dir is not None:
        root = Path(output_dir)
        results_path = root / "exp273_dataset_results.csv"
        summary_path = root / "exp273_evaluation_summary.json"
        with tempfile.TemporaryDirectory() as _:
            headers = list(result_rows[0])
            from io import StringIO

            handle = StringIO(newline="")
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(result_rows)
            _atomic_text(results_path, handle.getvalue())
        _atomic_text(summary_path, json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n")
        output["results_path"] = results_path
        output["summary_path"] = summary_path
    return output


__all__ = ["PredictionHashError", "evaluate_frozen_predictions"]
