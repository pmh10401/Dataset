from __future__ import annotations

import csv
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.exp201_205.core import (
    ALPHA,
    SOURCE_NAMES,
    TRAIN_RISK_CEILING,
    parse_indices,
    sha256_file,
    validate_prediction_payload,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/Users/minho/Documents/timesries project")
DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"
EXP186_ROOT = DATA_DIR / "exp186_200_unattended"
EXP162_ROOT = SOURCE_ROOT / "outputs/exp162_full_coverage_alternative_score_cache"
EXP164_ROOT = SOURCE_ROOT / "outputs/exp164_train_normal_agreement_risk_audit"


@dataclass(frozen=True)
class FrozenInputs:
    exp137: dict[str, dict[str, object]]
    exp190: dict[str, dict[str, set[int]]]
    exp198: dict[str, set[int]]
    exp162_manifest: dict[str, dict[str, str]]
    exp162_contract: dict[str, object]
    datasets: tuple[str, ...]
    input_hashes: dict[str, str]


def _verified_jsonl(path: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str(manifest["prediction_sha256"])
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"prediction hash mismatch path={path} expected={expected} actual={actual}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        validate_prediction_payload(row)
    return rows


def _load_exp137_allowed() -> tuple[dict[str, dict[str, object]], Path]:
    path = DATA_DIR / "experiment_137_operational_triage_results.csv"
    allowed: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            name = raw["dataset_name"]
            if name in allowed:
                raise ValueError(f"duplicate Exp137 dataset: {name}")
            allowed[name] = {
                "dataset": name,
                "n_train": int(raw["train_normal_count"]),
                "n_test": int(raw["test_size"]),
                "hard_indices": parse_indices(raw["hard_alert_indices"]),
                "standard_indices": parse_indices(raw["standard_review_indices"]),
                "priority_indices": parse_indices(raw["priority_review_indices"]),
            }
    return allowed, path


def _load_exp190() -> tuple[dict[str, dict[str, set[int]]], list[Path]]:
    prediction = EXP186_ROOT / "exp190_frozen_predictions.jsonl"
    manifest = EXP186_ROOT / "exp190_prediction_manifest.json"
    rows = _verified_jsonl(prediction, manifest)
    result: dict[str, dict[str, set[int]]] = {}
    for row in rows:
        name = str(row["dataset"])
        result.setdefault(name, {})[str(row["variant"])] = parse_indices(row["candidate_indices"])
    return result, [prediction, manifest]


def _load_exp198() -> tuple[dict[str, set[int]], list[Path]]:
    prediction = EXP186_ROOT / "exp198_frozen_predictions.jsonl"
    manifest = EXP186_ROOT / "exp198_prediction_manifest.json"
    rows = _verified_jsonl(prediction, manifest)
    result = {str(row["dataset"]): parse_indices(row["candidate_indices"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate Exp198 dataset")
    return result, [prediction, manifest]


def _load_exp162() -> tuple[dict[str, dict[str, str]], dict[str, object], list[Path]]:
    manifest_path = EXP162_ROOT / "02_cache_manifest.csv"
    contract_path = EXP162_ROOT / "01_source_contract.json"
    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    manifest = {row["dataset"]: row for row in rows}
    if len(manifest) != 1117 or len(rows) != 1117:
        raise ValueError("Exp162 manifest must cover 1117 unique datasets")
    bad = [name for name, row in manifest.items() if row["status"] not in {"success", "reused"}]
    if bad:
        raise ValueError(f"Exp162 unsuccessful cache rows: {bad[:10]}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("contains_test_labels") is not False:
        raise ValueError("Exp162 contract contains TEST labels")
    return manifest, contract, [manifest_path, contract_path]


def _validate_train_risk() -> Path:
    path = EXP164_ROOT / "09_n_train_bin_summary.csv"
    matches = [
        row for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
        if row["alpha"] == str(ALPHA)
        and row["source_or_variant"] == "train_audit_3of3"
        and row["n_train_bin"] == "ge100"
    ]
    if len(matches) != 1:
        raise ValueError("Exp164 ge100 three-of-three alpha=0.01 risk row missing")
    if float(matches[0]["wilson_upper_bound"]) > TRAIN_RISK_CEILING:
        raise ValueError("Exp164 TRAIN-normal risk ceiling is not satisfied")
    return path


def load_frozen_inputs(dataset_limit: int | None = None) -> FrozenInputs:
    exp137, exp137_path = _load_exp137_allowed()
    exp190, exp190_paths = _load_exp190()
    exp198, exp198_paths = _load_exp198()
    exp162, contract, exp162_paths = _load_exp162()
    risk_path = _validate_train_risk()

    datasets = tuple(sorted(name for name, row in exp137.items() if int(row["n_train"]) >= 100))
    if len(datasets) != 251:
        raise ValueError(f"eligible population mismatch expected=251 actual={len(datasets)}")
    for name in datasets:
        if name not in exp190 or name not in exp198 or name not in exp162:
            raise ValueError(f"missing frozen source coverage dataset={name}")
        if set(exp190[name]) != {
            "three_tier_high_confidence", "three_tier_standard", "three_tier_exploratory"
        }:
            raise ValueError(f"incomplete Exp190 tiers dataset={name}")

    selected = datasets[: int(dataset_limit)] if dataset_limit else datasets
    hash_paths = [exp137_path, *exp190_paths, *exp198_paths, *exp162_paths, risk_path]
    return FrozenInputs(
        exp137=exp137,
        exp190=exp190,
        exp198=exp198,
        exp162_manifest=exp162,
        exp162_contract=contract,
        datasets=selected,
        input_hashes={str(path): sha256_file(path) for path in hash_paths},
    )


def load_alternative_scores(inputs: FrozenInputs, dataset: str) -> dict[str, np.ndarray]:
    entry = inputs.exp162_manifest[dataset]
    cache_path = SOURCE_ROOT / entry["file_path"]
    actual = sha256_file(cache_path)
    if actual != entry["file_hash"]:
        raise ValueError(f"Exp162 cache hash mismatch dataset={dataset}")
    with np.load(cache_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata["dataset"] != dataset:
            raise ValueError("Exp162 cache dataset mismatch")
        if metadata["source_config_hash"] != inputs.exp162_contract["source_config_hash"]:
            raise ValueError("Exp162 source config hash mismatch")
        arrays = {
            f"{source}_{split}": np.asarray(archive[f"{source}_{split}"], dtype=float)
            for source in SOURCE_NAMES for split in ("train", "test")
        }
    row = inputs.exp137[dataset]
    if any(len(arrays[f"{source}_train"]) != int(row["n_train"]) for source in SOURCE_NAMES):
        raise ValueError(f"Exp162 TRAIN row mismatch dataset={dataset}")
    if any(len(arrays[f"{source}_test"]) != int(row["n_test"]) for source in SOURCE_NAMES):
        raise ValueError(f"Exp162 TEST row mismatch dataset={dataset}")
    return arrays


def freeze_predictions(
    stage: str,
    experiment_id: str,
    rows: list[dict[str, object]],
    output_root: Path,
    input_hashes: dict[str, str],
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / f"exp{stage}_frozen_predictions.jsonl"
    manifest_path = output_root / f"exp{stage}_prediction_manifest.json"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            validate_prediction_payload(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "experiment_id": experiment_id,
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_rows": len(rows),
        "dataset_count": len({str(row["dataset"]) for row in rows}),
        "input_hashes": dict(sorted(input_hashes.items())),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return prediction_path, manifest_path


def load_verified_predictions(prediction_path: Path, manifest_path: Path) -> list[dict[str, object]]:
    return _verified_jsonl(prediction_path, manifest_path)


def load_test_labels_for_evaluation(dataset: str) -> np.ndarray:
    connection = sqlite3.connect(str(DB_PATH))
    try:
        rows = connection.execute(
            """
            SELECT CAST(i.label AS INTEGER)
            FROM instances i JOIN datasets d ON d.id=i.dataset_id
            WHERE d.name=? AND i.split='TEST'
            ORDER BY i.instance_index
            """,
            (dataset,),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"missing TEST labels dataset={dataset}")
    return np.asarray([int(row[0]) for row in rows], dtype=np.int8)
