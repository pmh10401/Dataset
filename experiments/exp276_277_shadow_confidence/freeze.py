from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.calibration import empirical_upper_pvalues
from experiments.exp267_273_zero_support_fn_recovery.recovery_routing import (
    verify_recovery_prediction_hash,
)
from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import _cache_hash

from .policy import ALPHA, confidence_record, normalized_tail_margin


EXP272_ROOT = Path("/Users/minho/Documents/Dataset/exp267_273_zero_support_fn_recovery")
SOURCE_NAMES = ("scattering", "signature", "rff_forecast", "tcn_forecast")
FORECAST_GROUP = "rff_forecast+tcn_seed_intersection+tcn_seed_median"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _cache_index() -> dict[str, dict[str, tuple[Path, dict[str, object]]]]:
    root = EXP272_ROOT / "source_cache" / "source_cache"
    result: dict[str, dict[str, tuple[Path, dict[str, object]]]] = {}
    for source in SOURCE_NAMES:
        by_dataset: dict[str, tuple[Path, dict[str, object]]] = {}
        for metadata_path in (root / source / "cache").glob("*.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            dataset = str(metadata["dataset"])
            if dataset in by_dataset:
                raise ValueError(f"duplicate {source} cache metadata: {dataset}")
            by_dataset[dataset] = (metadata_path.with_suffix(".npz"), metadata)
        if len(by_dataset) != 1117:
            raise ValueError(f"{source} cache coverage mismatch: {len(by_dataset)}")
        result[source] = by_dataset
    return result


def _risk_index() -> dict[tuple[str, str, str], tuple[float, int]]:
    path = EXP272_ROOT / "exp270_train_risk_dataset.csv"
    result: dict[tuple[str, str, str], tuple[float, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["alpha"]) - ALPHA) > 1e-15:
                continue
            key = (row["dataset"], row["source"], row["variant"])
            result[key] = (float(row["wilson_upper_95"]), int(row["reference_count"]))
    return result


def _load_arrays(cache: tuple[Path, dict[str, object]]) -> dict[str, np.ndarray]:
    path, metadata = cache
    if _sha256(path) != metadata.get("archive_sha256"):
        raise ValueError(f"source cache archive hash mismatch: {path}")
    with np.load(path) as archive:
        arrays = {key: np.asarray(archive[key], dtype=float) for key in archive.files}
    if _cache_hash(arrays, metadata) != metadata.get("cache_sha256"):
        raise ValueError(f"source cache semantic hash mismatch: {path}")
    return arrays


def _source_evidence(
    dataset: str,
    index: int,
    support: set[str],
    caches: dict[str, dict[str, tuple[Path, dict[str, object]]]],
    risks: dict[tuple[str, str, str], tuple[float, int]],
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for source in ("scattering", "signature"):
        if source not in support:
            continue
        arrays = _load_arrays(caches[source][dataset])
        pvalue = float(empirical_upper_pvalues(arrays["train_oof"], arrays["test_scores"])[index])
        risk, reference = risks[(dataset, source, "primary")]
        evidence.append({
            "group": source, "pvalue": pvalue, "reference_count": reference,
            "train_risk_upper_95": risk,
            "normalized_margin": normalized_tail_margin(pvalue, reference),
        })

    if FORECAST_GROUP in support:
        variants: list[tuple[str, float, float, int]] = []
        rff = _load_arrays(caches["rff_forecast"][dataset])
        if len(rff.get("test_scores", ())):
            pvalue = float(empirical_upper_pvalues(rff["train_oof"], rff["test_scores"])[index])
            risk, reference = risks[(dataset, "rff_forecast", "primary")]
            if pvalue <= ALPHA + 1e-15:
                variants.append(("rff_primary", pvalue, risk, reference))
        tcn = _load_arrays(caches["tcn_forecast"][dataset])
        if len(tcn.get("test_scores", ())):
            median_p = float(empirical_upper_pvalues(tcn["train_oof"], tcn["test_scores"])[index])
            risk, reference = risks[(dataset, "tcn_forecast", "seed_median")]
            if median_p <= ALPHA + 1e-15:
                variants.append(("tcn_seed_median", median_p, risk, reference))
            seed_pvalues = [
                float(empirical_upper_pvalues(
                    tcn[f"train_oof__seed_{seed}"], tcn[f"test_scores__seed_{seed}"]
                )[index])
                for seed in (20260721, 20261721)
            ]
            intersection_p = max(seed_pvalues)
            risk, reference = risks[(dataset, "tcn_forecast", "seed_intersection")]
            if intersection_p <= ALPHA + 1e-15:
                variants.append(("tcn_seed_intersection", intersection_p, risk, reference))
        if not variants:
            raise ValueError(f"forecast support cannot be reconstructed: {dataset}[{index}]")
        selected = min(variants, key=lambda item: item[1])
        evidence.append({
            "group": "normal_forecast", "pvalue": selected[1],
            "reference_count": selected[3],
            "train_risk_upper_95": max(item[2] for item in variants),
            "normalized_margin": normalized_tail_margin(selected[1], selected[3]),
            "supporting_variants": [item[0] for item in variants],
        })
    return evidence


def freeze_confidence_predictions(
    *, output_root: Path, expected_datasets: int = 1117, dataset_limit: int | None = None,
) -> tuple[Path, Path]:
    prediction_in = EXP272_ROOT / "exp272_frozen_predictions.jsonl"
    manifest_in = EXP272_ROOT / "exp272_prediction_manifest.json"
    source_prediction = EXP272_ROOT / "source_predictions" / "frozen_source_predictions.jsonl"
    source_manifest_path = EXP272_ROOT / "source_predictions" / "source_prediction_manifest.json"
    rows = _read_jsonl(prediction_in)
    full_datasets = tuple(sorted(str(row["dataset"]) for row in rows))
    verify_recovery_prediction_hash(prediction_in, manifest_in, expected_datasets=full_datasets)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if _sha256(source_prediction) != source_manifest.get("prediction_sha256"):
        raise ValueError("source prediction hash mismatch")
    if source_manifest.get("datasets") != 1117 or source_manifest.get("contains_test_labels") is not False:
        raise ValueError("source prediction contract mismatch")
    datasets = full_datasets[:dataset_limit] if dataset_limit is not None else full_datasets
    if len(datasets) != expected_datasets:
        raise ValueError("requested output population mismatch")

    caches, risks = _cache_index(), _risk_index()
    by_dataset = {str(row["dataset"]): row for row in rows}
    output_rows: list[dict[str, object]] = []
    for dataset in datasets:
        source_row = by_dataset[dataset]
        candidate_records = []
        support_by_index = source_row["support_by_index"]
        for index in source_row["recovery_shadow_indices"]:
            support = set(support_by_index[str(index)])
            evidence = _source_evidence(dataset, index, support, caches, risks)
            confidence = confidence_record(
                [float(item["normalized_margin"]) for item in evidence],
                [float(item["train_risk_upper_95"]) for item in evidence],
            )
            candidate_records.append({
                "candidate_index": index, "continuous_evidence": evidence, **confidence,
            })
        output_rows.append({
            "dataset": dataset, "n_train": source_row["n_train"], "n_test": source_row["n_test"],
            "hard_indices": source_row["hard_indices"],
            "recovery_shadow_indices": source_row["recovery_shadow_indices"],
            "candidate_records": candidate_records,
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "hard_policy_changed": False, "review_only": True, "retrospective": True,
        })

    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for row in output_rows
    )
    root = Path(output_root)
    prediction_path = root / "exp276_frozen_confidence.jsonl"
    manifest_path = root / "exp276_prediction_manifest.json"
    _atomic_bytes(prediction_path, payload)
    manifest = {
        "experiment_id": "experiment_276_recovery_shadow_confidence_freeze",
        "datasets": len(output_rows), "rows": len(output_rows),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "contains_test_labels": False, "contains_evaluation_metrics": False,
        "hard_policy_changed": False, "review_only": True, "retrospective": True,
        "input_sha256": {
            "exp272_prediction": _sha256(prediction_in), "exp272_manifest": _sha256(manifest_in),
            "source_prediction": _sha256(source_prediction),
            "source_manifest": _sha256(source_manifest_path),
            "train_risk": _sha256(EXP272_ROOT / "exp270_train_risk_dataset.csv"),
        },
    }
    _atomic_bytes(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    summary = {
        "datasets": len(output_rows),
        "recovery_shadow_candidates": sum(len(row["candidate_records"]) for row in output_rows),
        "guard_all_deep_count": sum(bool(c["guard_all_deep"]) for row in output_rows for c in row["candidate_records"]),
        "guard_median_deep_count": sum(bool(c["guard_median_deep"]) for row in output_rows for c in row["candidate_records"]),
        "guard_consistent_deep_count": sum(bool(c["guard_consistent_deep"]) for row in output_rows for c in row["candidate_records"]),
        "contains_test_labels": False, "contains_evaluation_metrics": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "exp276_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    return prediction_path, manifest_path


def verify_confidence_prediction(
    prediction_path: Path, manifest_path: Path, *, expected_datasets: Iterable[str],
) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if _sha256(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("Exp276 prediction hash mismatch")
    rows = _read_jsonl(prediction_path)
    expected = tuple(sorted(str(value) for value in expected_datasets))
    if tuple(str(row.get("dataset")) for row in rows) != expected or len(rows) != manifest.get("rows"):
        raise ValueError("Exp276 prediction population mismatch")
    forbidden = {"y_test", "tp", "fp", "fn", "precision", "recall", "f1", "combined_f1"}
    for row in rows:
        if forbidden & set(row):
            raise ValueError("Exp276 prediction contains evaluation fields")
        if row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False:
            raise ValueError("Exp276 prediction boundary mismatch")
        indices = [record["candidate_index"] for record in row["candidate_records"]]
        if indices != row["recovery_shadow_indices"]:
            raise ValueError("Exp276 confidence coverage mismatch")
    return rows


__all__ = ["freeze_confidence_predictions", "verify_confidence_prediction"]
