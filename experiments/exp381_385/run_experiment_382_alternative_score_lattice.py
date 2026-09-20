from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp377_380.feature_combination_reporting import (
    evaluate_indices,
    summarize_detail_rows,
    write_csv,
)
from experiments.exp381_385.alternative_score_core import (
    ALTERNATIVE_SOURCES,
    calibrate_local_empirical,
    evaluate_rule,
    generate_rule_specs,
)
from experiments.exp381_385.coverage_audit_core import sha256_file


REPOSITORY_ROOT = Path("/Users/minho/Documents/timesries project")
DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
EXP162_ROOT = REPOSITORY_ROOT / "outputs/exp162_full_coverage_alternative_score_cache"
EXP162_MANIFEST = EXP162_ROOT / "02_cache_manifest.csv"
EXP381_ROOT = DATA_ROOT / "exp381_385_coverage_first_feature_lattice"
DEFAULT_OUTPUT_ROOT = EXP381_ROOT
ALPHAS = (0.005, 0.01, 0.02, 0.05)
SOURCE_CACHE_PREFIX = {
    "spectral_derivative": "spectral",
    "shapelet_normal_prototype": "shapelet",
    "interval_quantile": "interval",
}
FORBIDDEN_KEYS = {
    "y_test",
    "labels",
    "tp",
    "fp",
    "fn",
    "f1",
    "precision",
    "recall",
    "combined_f1",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_walk_keys(nested))
    return keys


def _freeze_predictions(rows: list[dict], output_root: Path) -> tuple[Path, Path]:
    prediction_path = output_root / "exp382_frozen_predictions.jsonl"
    manifest_path = output_root / "exp382_prediction_manifest.json"
    ordered = sorted(rows, key=lambda row: (row["dataset"], float(row["alpha"])))
    for row in ordered:
        forbidden = _walk_keys(row) & FORBIDDEN_KEYS
        if forbidden:
            raise ValueError(f"prediction contains evaluation fields: {sorted(forbidden)}")
    payload = "".join(_canonical_json(row) + "\n" for row in ordered).encode("utf-8")
    prediction_path.write_bytes(payload)
    manifest = {
        "experiment_id": "experiment_382_alternative_score_lattice",
        "rows": len(ordered),
        "datasets": len({row["dataset"] for row in ordered}),
        "alphas": list(ALPHAS),
        "calibration_variant": "local_train_empirical_tail",
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "prediction_sha256": _sha256_bytes(payload),
        "retrospective_counterfactual": True,
        "selected_for_deployment": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prediction_path, manifest_path


def _load_frozen_predictions(
    prediction_path: Path,
    manifest_path: Path,
) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != manifest["prediction_sha256"]:
        raise ValueError("Exp382 prediction SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != manifest["rows"]:
        raise ValueError("Exp382 prediction row count mismatch")
    return rows


def _verified_cache_paths() -> dict[str, Path]:
    coverage_path = EXP381_ROOT / "exp381_exp162_cache_audit.csv"
    exp381_manifest_path = EXP381_ROOT / "exp381_manifest.json"
    if not coverage_path.is_file() or not exp381_manifest_path.is_file():
        raise ValueError("Exp381 verified cache audit is required")
    exp381_manifest = json.loads(exp381_manifest_path.read_text(encoding="utf-8"))
    if exp381_manifest.get("cache_complete") is not True:
        raise ValueError("Exp381 cache contract is incomplete")
    with coverage_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1117 or any(row["valid"] != "1" for row in rows):
        raise ValueError("Exp381 cache audit does not cover 1,117 valid datasets")
    return {row["dataset"]: Path(row["file_path"]) for row in rows}


def _load_test_labels(dataset_names: set[str]) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = defaultdict(list)
    uri = f"file:{DB_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        for dataset, label in connection.execute(
            """
            SELECT d.name, i.label
            FROM datasets AS d
            JOIN instances AS i ON i.dataset_id = d.id
            WHERE i.split = 'TEST'
            ORDER BY d.name, i.instance_index
            """
        ):
            if dataset in dataset_names:
                labels[str(dataset)].append(str(label))
    if set(labels) != dataset_names:
        raise ValueError("missing TEST labels after prediction freeze")
    return dict(labels)


def _build_predictions(cache_paths: dict[str, Path]) -> list[dict]:
    rules = generate_rule_specs()
    rows: list[dict] = []
    total = len(cache_paths)
    for done, dataset in enumerate(sorted(cache_paths), start=1):
        with np.load(cache_paths[dataset], allow_pickle=False) as cache:
            n_test = int(np.asarray(cache["spectral_test"]).size)
            for alpha in ALPHAS:
                calibrations = {}
                source_indices = {}
                source_available = {}
                for source in ALTERNATIVE_SOURCES:
                    cache_prefix = SOURCE_CACHE_PREFIX[source]
                    result = calibrate_local_empirical(
                        cache[f"{cache_prefix}_train"],
                        cache[f"{cache_prefix}_test"],
                        alpha,
                    )
                    calibrations[source] = {
                        "status": result.status,
                        "n_train": result.n_train,
                        "threshold": result.threshold,
                        "threshold_rank": result.threshold_rank,
                        "distinct_train_scores": result.distinct_train_scores,
                        "indices": list(result.indices),
                    }
                    source_indices[source] = result.indices
                    source_available[source] = result.status == "attainable"
                rule_rows = {}
                for rule in rules:
                    result = evaluate_rule(
                        rule,
                        source_indices=source_indices,
                        source_available=source_available,
                    )
                    rule_rows[rule.rule_id] = {
                        "sources": list(rule.sources),
                        "k": rule.k,
                        "indices": list(result.indices),
                        "available_source_count": result.available_source_count,
                        "selected_source_count": result.selected_source_count,
                        "fully_observable": result.fully_observable,
                        "resolution_blocked": result.resolution_blocked,
                        "unavailable_sources": list(result.unavailable_sources),
                    }
                rows.append(
                    {
                        "experiment_id": "experiment_382_alternative_score_lattice",
                        "dataset": dataset,
                        "alpha": alpha,
                        "n_test": n_test,
                        "calibration_variant": "local_train_empirical_tail",
                        "contains_test_labels": False,
                        "contains_evaluation_metrics": False,
                        "sources": calibrations,
                        "rules": rule_rows,
                    }
                )
        if done == total or done % 50 == 0:
            print(
                f"Progress: [{done}/{total}] rows={done * len(ALPHAS)} "
                f"dataset={dataset} errors=0",
                flush=True,
            )
    return rows


def _evaluate(rows: list[dict], labels_by_dataset: dict[str, list[str]]) -> None:
    detail: list[dict] = []
    for row in rows:
        labels = labels_by_dataset[row["dataset"]]
        if len(labels) != int(row["n_test"]):
            raise ValueError(f"TEST length mismatch: {row['dataset']}")
        for rule_id, rule in row["rules"].items():
            base = {
                "experiment_id": "experiment_382_alternative_score_lattice",
                "dataset": row["dataset"],
                "alpha": row["alpha"],
                "calibration_variant": row["calibration_variant"],
                "rule_id": rule_id,
                "sources": _canonical_json(rule["sources"]),
                "k": rule["k"],
                "available_source_count": rule["available_source_count"],
                "selected_source_count": rule["selected_source_count"],
                "fully_observable": rule["fully_observable"],
                "resolution_blocked": rule["resolution_blocked"],
                "retrospective_counterfactual": True,
                "prospective_validated": False,
                "selected_for_deployment": False,
            }
            detail.append(
                {
                    **base,
                    "scope": "full_population",
                    **evaluate_indices(rule["indices"], labels),
                }
            )
            if rule["fully_observable"]:
                detail.append(
                    {
                        **base,
                        "scope": "eligible_population",
                        **evaluate_indices(rule["indices"], labels),
                    }
                )
    write_csv(DEFAULT_OUTPUT_ROOT / "exp382_detail.csv", detail)
    summary = summarize_detail_rows(
        detail,
        (
            "calibration_variant",
            "alpha",
            "rule_id",
            "scope",
        ),
    )
    write_csv(DEFAULT_OUTPUT_ROOT / "exp382_summary.csv", summary)


def run(output_root: Path) -> None:
    global DEFAULT_OUTPUT_ROOT
    DEFAULT_OUTPUT_ROOT = output_root
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "exp382_status.json"
    status_path.write_text(
        _canonical_json(
            {
                "id": "experiment_382_alternative_score_lattice",
                "experiment_id": "experiment_382_alternative_score_lattice",
                "status": "running",
                "phase": "prediction",
                "datasets_done": 0,
                "datasets_total": 1117,
                "progress_percent": 0.0,
                "errors": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache_paths = _verified_cache_paths()
    rows = _build_predictions(cache_paths)
    prediction_path, prediction_manifest_path = _freeze_predictions(rows, output_root)

    # Evaluation starts only after the label-free prediction file is frozen.
    frozen_rows = _load_frozen_predictions(prediction_path, prediction_manifest_path)
    labels_by_dataset = _load_test_labels({row["dataset"] for row in frozen_rows})
    _evaluate(frozen_rows, labels_by_dataset)

    pooling_status = {
        "variant": "similarity_pooling",
        "status": "blocked_not_executed",
        "reason": (
            "A hash-verified frozen TRAIN descriptor manifest is not part of the "
            "Exp381 verified inputs. The implementation does not approximate donors."
        ),
        "human_review_only": True,
    }
    (output_root / "exp382_similarity_pooling_status.json").write_text(
        json.dumps(pooling_status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_files = [
        prediction_path,
        prediction_manifest_path,
        output_root / "exp382_detail.csv",
        output_root / "exp382_summary.csv",
        output_root / "exp382_similarity_pooling_status.json",
    ]
    manifest = {
        "experiment_id": "experiment_382_alternative_score_lattice",
        "datasets": 1117,
        "prediction_rows": len(frozen_rows),
        "alphas": list(ALPHAS),
        "rules": len(generate_rule_specs()),
        "input_exp162_manifest_sha256": sha256_file(EXP162_MANIFEST),
        "input_exp381_manifest_sha256": sha256_file(
            EXP381_ROOT / "exp381_manifest.json"
        ),
        "prediction_sha256": sha256_file(prediction_path),
        "output_files": {path.name: sha256_file(path) for path in output_files},
        "retrospective_counterfactual": True,
        "contains_test_labels_in_prediction": False,
        "labels_loaded_after_prediction_freeze": True,
        "selected_for_deployment": False,
        "prospective_validated": False,
    }
    (output_root / "exp382_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(
            {
                "id": "experiment_382_alternative_score_lattice",
                "experiment_id": "experiment_382_alternative_score_lattice",
                "status": "completed",
                "phase": "completed",
                "datasets_done": 1117,
                "datasets_total": 1117,
                "progress_percent": 100.0,
                "errors": 0,
                "elapsed_seconds": time.time() - started,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.output_root)


if __name__ == "__main__":
    main()
