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

from experiments.exp377_380.feature_combination_core import collapse_family_evidence
from experiments.exp377_380.feature_combination_reporting import evaluate_indices
from experiments.exp381_385.alternative_score_core import (
    evaluate_rule,
    generate_rule_specs,
)
from experiments.exp381_385.coverage_audit_core import sha256_file
from experiments.exp381_385.exhaustive_lattice_core import (
    NEW_TRAIN_SCORE_SOURCES,
    SEVEN_SOURCES,
    leave_one_out_tail_candidates,
    wilson_upper_bound,
)


REPOSITORY_ROOT = Path("/Users/minho/Documents/timesries project")
DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
OUTPUT_ROOT = DATA_ROOT / "exp381_385_coverage_first_feature_lattice"
EXP357_ROOT = DATA_ROOT / "exp356_359_operational_confidence_tiers"
EXP357_PREDICTIONS = EXP357_ROOT / "exp357_frozen_evidence.jsonl"
EXP357_MANIFEST = EXP357_ROOT / "exp357_prediction_manifest.json"
EXP382_PREDICTIONS = OUTPUT_ROOT / "exp382_frozen_predictions.jsonl"
EXP382_MANIFEST = OUTPUT_ROOT / "exp382_prediction_manifest.json"
EXP383_FAMILY_MAPS = OUTPUT_ROOT / "exp383_source_family_maps.json"
EXP381_CACHE_AUDIT = OUTPUT_ROOT / "exp381_exp162_cache_audit.csv"
ALPHA = 0.005
RISK_TARGETS_PER_100 = (0.5, 1.0, 2.0)
SOURCE_CACHE_PREFIX = {
    "spectral_derivative": "spectral",
    "shapelet_normal_prototype": "shapelet",
    "interval_quantile": "interval",
}


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_verified_jsonl(path: Path, manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["prediction_sha256"]:
        raise ValueError(f"prediction SHA256 mismatch: {path}")
    if manifest.get("contains_test_labels") is not False:
        raise ValueError(f"prediction contains TEST labels: {path}")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError(f"prediction contains evaluation metrics: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _freeze_registry(rows: list[dict], output_root: Path) -> tuple[Path, Path]:
    path = output_root / "exp384_frozen_source_registry.jsonl"
    payload = "".join(_canonical(row) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(payload)
    manifest = {
        "experiment_id": "experiment_384_exhaustive_lattice_risk",
        "rows": len(rows),
        "datasets": len(rows),
        "alpha": ALPHA,
        "sources": list(SEVEN_SOURCES),
        "rules": len(generate_rule_specs(SEVEN_SOURCES)),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "retrospective_counterfactual": True,
    }
    manifest_path = output_root / "exp384_source_registry_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, manifest_path


def _load_labels(dataset_names: set[str]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    uri = f"file:{DB_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        for dataset, label in connection.execute(
            """
            SELECT d.name, i.label
            FROM datasets d JOIN instances i ON i.dataset_id = d.id
            WHERE i.split='TEST'
            ORDER BY d.name, i.instance_index
            """
        ):
            if dataset in dataset_names:
                values[str(dataset)].append(str(label))
    if set(values) != dataset_names:
        raise ValueError("TEST label coverage mismatch")
    return dict(values)


def _cache_paths() -> dict[str, Path]:
    with EXP381_CACHE_AUDIT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1117 or any(row["valid"] != "1" for row in rows):
        raise ValueError("Exp381 cache audit is incomplete")
    return {row["dataset"]: Path(row["file_path"]) for row in rows}


def _build_registry() -> list[dict]:
    exp357_rows = _load_verified_jsonl(EXP357_PREDICTIONS, EXP357_MANIFEST)
    exp382_rows = _load_verified_jsonl(EXP382_PREDICTIONS, EXP382_MANIFEST)
    exp357 = {row["dataset"]: row for row in exp357_rows}
    exp382 = {
        row["dataset"]: row
        for row in exp382_rows
        if float(row["alpha"]) == ALPHA
    }
    if set(exp357) != set(exp382) or len(exp357) != 1117:
        raise ValueError("Exp357/Exp382 population mismatch")

    rows = []
    for dataset in sorted(exp357):
        old = exp357[dataset]
        new = exp382[dataset]
        old_state = collapse_family_evidence(old)
        indices = dict(old_state["indices"])
        available = dict(old_state["available"])
        for source in NEW_TRAIN_SCORE_SOURCES:
            source_row = new["sources"][source]
            indices[source] = [int(index) for index in source_row["indices"]]
            available[source] = source_row["status"] == "attainable"
        rows.append(
            {
                "experiment_id": "experiment_384_exhaustive_lattice_risk",
                "dataset": dataset,
                "alpha": ALPHA,
                "n_test": int(old["n_test"]),
                "source_indices": {
                    source: sorted(set(indices[source])) for source in SEVEN_SOURCES
                },
                "source_available": {
                    source: bool(available[source]) for source in SEVEN_SOURCES
                },
                "baseline_hard_indices": sorted(
                    {int(index) for index in old["baseline_hard_indices"]}
                ),
                "automatic_normal_exact_indices": sorted(
                    {int(index) for index in old["automatic_normal_exact_indices"]}
                ),
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            }
        )
    return rows


def _empty_aggregate() -> dict:
    return {
        "datasets": 0,
        "candidate_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "f1_sum": 0.0,
        "zero_f1_datasets": 0,
        "candidate_zero_datasets": 0,
        "resolution_blocked_datasets": 0,
        "fully_observable_datasets": 0,
    }


def _update_aggregate(aggregate: dict, metrics: dict, rule_result) -> None:
    aggregate["datasets"] += 1
    aggregate["candidate_count"] += int(metrics["candidate_count"])
    aggregate["tp"] += int(metrics["tp"])
    aggregate["fp"] += int(metrics["fp"])
    aggregate["fn"] += int(metrics["fn"])
    aggregate["f1_sum"] += float(metrics["f1"])
    aggregate["zero_f1_datasets"] += float(metrics["f1"]) == 0.0
    aggregate["candidate_zero_datasets"] += int(metrics["candidate_count"]) == 0
    aggregate["resolution_blocked_datasets"] += rule_result.resolution_blocked
    aggregate["fully_observable_datasets"] += rule_result.fully_observable


def _evaluate_registry(rows: list[dict], labels_by_dataset: dict[str, list[str]]) -> list[dict]:
    rules = generate_rule_specs(SEVEN_SOURCES)
    aggregates = {
        (rule.rule_id, scope): _empty_aggregate()
        for rule in rules
        for scope in ("full_population", "common_support")
    }
    for done, row in enumerate(rows, start=1):
        labels = labels_by_dataset[row["dataset"]]
        exact_normal = set(row["automatic_normal_exact_indices"])
        for rule in rules:
            result = evaluate_rule(
                rule,
                source_indices=row["source_indices"],
                source_available=row["source_available"],
            )
            guarded = sorted(set(result.indices) - exact_normal)
            metrics = evaluate_indices(guarded, labels)
            _update_aggregate(
                aggregates[(rule.rule_id, "full_population")],
                metrics,
                result,
            )
            if result.fully_observable:
                _update_aggregate(
                    aggregates[(rule.rule_id, "common_support")],
                    metrics,
                    result,
                )
        if done == len(rows) or done % 50 == 0:
            print(
                f"Progress: [{done}/{len(rows)}] rows={done * len(rules)} "
                f"dataset={row['dataset']} errors=0",
                flush=True,
            )

    summary = []
    rule_map = {rule.rule_id: rule for rule in rules}
    for (rule_id, scope), aggregate in sorted(aggregates.items()):
        if aggregate["datasets"] == 0:
            continue
        tp = aggregate["tp"]
        fp = aggregate["fp"]
        fn = aggregate["fn"]
        rule = rule_map[rule_id]
        common = {
            "experiment_id": "experiment_384_exhaustive_lattice_risk",
            "alpha": ALPHA,
            "lattice_variant": "raw_seven_source",
            "rule_id": rule_id,
            "sources": _canonical(rule.sources),
            "k": rule.k,
            "scope": scope,
            **{key: value for key, value in aggregate.items() if key != "f1_sum"},
            "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
            "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
            "mean_f1": aggregate["f1_sum"] / aggregate["datasets"],
            "selected_for_deployment": False,
            "retrospective_counterfactual": True,
        }
        summary.append(common)
        summary.append({**common, "lattice_variant": "independent_family"})
    return summary


def _train_risk(cache_paths: dict[str, Path]) -> list[dict]:
    all_rules = generate_rule_specs(SEVEN_SOURCES)
    new_rules = {
        rule.rule_id: rule
        for rule in generate_rule_specs(NEW_TRAIN_SCORE_SOURCES)
    }
    totals = {
        rule_id: {"normal_runs": 0, "false_candidates": 0, "eligible_datasets": 0}
        for rule_id in new_rules
    }
    for dataset in sorted(cache_paths):
        with np.load(cache_paths[dataset], allow_pickle=False) as cache:
            source_indices = {
                source: leave_one_out_tail_candidates(
                    cache[f"{SOURCE_CACHE_PREFIX[source]}_train"],
                    ALPHA,
                )
                for source in NEW_TRAIN_SCORE_SOURCES
            }
            n_train = len(cache["spectral_train"])
        source_available = {
            source: bool(n_train >= 5 and int(ALPHA * n_train + 1e-12) >= 1)
            for source in NEW_TRAIN_SCORE_SOURCES
        }
        for rule_id, rule in new_rules.items():
            result = evaluate_rule(
                rule,
                source_indices=source_indices,
                source_available=source_available,
            )
            if result.resolution_blocked:
                continue
            totals[rule_id]["normal_runs"] += n_train
            totals[rule_id]["false_candidates"] += len(result.indices)
            totals[rule_id]["eligible_datasets"] += 1

    rows = []
    for rule in all_rules:
        total = totals.get(rule.rule_id)
        if total is None:
            rows.append(
                {
                    "rule_id": rule.rule_id,
                    "sources": _canonical(rule.sources),
                    "k": rule.k,
                    "risk_status": "provenance_unavailable",
                    "risk_reason": "one_or_more_exp357_sources_lack_raw_train_scores",
                    "eligible_datasets": "",
                    "normal_runs": "",
                    "false_candidates": "",
                    "false_alerts_per_100": "",
                    "wilson_upper_per_100": "",
                    "meets_0_5_per_100": "",
                    "meets_1_0_per_100": "",
                    "meets_2_0_per_100": "",
                }
            )
            continue
        trials = total["normal_runs"]
        successes = total["false_candidates"]
        rate = successes / trials if trials else 0.0
        upper = wilson_upper_bound(successes, trials) if trials else 1.0
        rows.append(
            {
                "rule_id": rule.rule_id,
                "sources": _canonical(rule.sources),
                "k": rule.k,
                "risk_status": "available",
                "risk_reason": "",
                "eligible_datasets": total["eligible_datasets"],
                "normal_runs": trials,
                "false_candidates": successes,
                "false_alerts_per_100": rate * 100.0,
                "wilson_upper_per_100": upper * 100.0,
                "meets_0_5_per_100": int(upper * 100.0 <= 0.5),
                "meets_1_0_per_100": int(upper * 100.0 <= 1.0),
                "meets_2_0_per_100": int(upper * 100.0 <= 2.0),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output_root: Path) -> None:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "exp384_status.json"
    status_path.write_text(
        _canonical(
            {
                "id": "experiment_384_exhaustive_lattice_risk",
                "status": "running",
                "phase": "source_registry",
                "datasets_done": 0,
                "datasets_total": 1117,
                "progress_percent": 0.0,
                "errors": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    family_maps = json.loads(EXP383_FAMILY_MAPS.read_text(encoding="utf-8"))
    if any(
        item["family_count"] != 3
        for item in family_maps["sensitivity_maps"].values()
    ):
        raise ValueError("Exp383 family maps are not equivalent across sensitivities")

    rows = _build_registry()
    registry_path, registry_manifest_path = _freeze_registry(rows, output_root)
    labels = _load_labels({row["dataset"] for row in rows})
    summary = _evaluate_registry(rows, labels)
    risk_rows = _train_risk(_cache_paths())
    coverage_rows = [
        {
            "dataset": row["dataset"],
            "n_test": row["n_test"],
            "available_source_count": sum(row["source_available"].values()),
            "all_seven_sources_available": int(
                all(row["source_available"].values())
            ),
            "errors": 0,
        }
        for row in rows
    ]
    _write_csv(output_root / "exp384_dataset_coverage.csv", coverage_rows)
    _write_csv(output_root / "exp384_lattice_summary.csv", summary)
    _write_csv(output_root / "exp384_train_risk.csv", risk_rows)

    report = [
        "# Exp384 Exhaustive Seven-Source Lattice",
        "",
        "- Datasets: 1,117",
        "- Alpha: 0.005",
        "- Raw rules: 448",
        "- Independent-family rules: 448 (Exp383 did not collapse the new sources)",
        "- Prediction labels: absent",
        "- TEST labels loaded after source-registry SHA256 freeze",
        "- Deployment-selected rule: none",
        "",
        "## TRAIN Risk Limit",
        "",
        "TRAIN pseudo-normal risk is available only for rules made exclusively from "
        "the three Exp162 sources. Exp357 raw TRAIN scores are unavailable, so rules "
        "containing the existing four sources are marked provenance_unavailable.",
    ]
    (output_root / "exp384_summary.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    outputs = [
        registry_path,
        registry_manifest_path,
        output_root / "exp384_dataset_coverage.csv",
        output_root / "exp384_lattice_summary.csv",
        output_root / "exp384_train_risk.csv",
        output_root / "exp384_summary.md",
    ]
    manifest = {
        "experiment_id": "experiment_384_exhaustive_lattice_risk",
        "datasets": 1117,
        "alpha": ALPHA,
        "rules": 448,
        "raw_and_independent_maps_equivalent": True,
        "input_hashes": {
            str(path): sha256_file(path)
            for path in (
                EXP357_PREDICTIONS,
                EXP357_MANIFEST,
                EXP382_PREDICTIONS,
                EXP382_MANIFEST,
                EXP383_FAMILY_MAPS,
            )
        },
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
        "contains_test_labels_in_prediction": False,
        "labels_loaded_after_prediction_freeze": True,
        "selected_for_deployment": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }
    (output_root / "exp384_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(
            {
                "id": "experiment_384_exhaustive_lattice_risk",
                "experiment_id": "experiment_384_exhaustive_lattice_risk",
                "status": "completed",
                "phase": "completed",
                "datasets_done": 1117,
                "datasets_total": 1117,
                "progress_percent": 100.0,
                "errors": 0,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.output_root)


if __name__ == "__main__":
    main()
