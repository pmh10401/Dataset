from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp377_380.feature_combination_reporting import evaluate_indices
from experiments.exp381_385.alternative_score_core import (
    evaluate_rule,
    generate_rule_specs,
)
from experiments.exp381_385.coverage_audit_core import sha256_file
from experiments.exp381_385.exhaustive_lattice_core import SEVEN_SOURCES
from experiments.exp381_385.pipeline_structure_core import build_pipeline_lanes


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
OUTPUT_ROOT = DATA_ROOT / "exp381_385_coverage_first_feature_lattice"
REGISTRY_PATH = OUTPUT_ROOT / "exp384_frozen_source_registry.jsonl"
REGISTRY_MANIFEST = OUTPUT_ROOT / "exp384_source_registry_manifest.json"
RISK_PATH = OUTPUT_ROOT / "exp384_train_risk.csv"
EXP359_ROOT = DATA_ROOT / "exp356_359_operational_confidence_tiers"
EXP359_PATH = EXP359_ROOT / "exp359_frozen_predictions.jsonl"
EXP359_MANIFEST = EXP359_ROOT / "exp359_prediction_manifest.json"
RISK_TARGET_FIELDS = {
    0.5: "meets_0_5_per_100",
    1.0: "meets_1_0_per_100",
    2.0: "meets_2_0_per_100",
}


def _load_jsonl(path: Path, manifest_path: Path) -> list[dict]:
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


def _load_labels(names: set[str]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    uri = f"file:{DB_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        for dataset, label in connection.execute(
            """
            SELECT d.name, i.label
            FROM datasets d JOIN instances i ON i.dataset_id=d.id
            WHERE i.split='TEST'
            ORDER BY d.name, i.instance_index
            """
        ):
            if dataset in names:
                values[str(dataset)].append(str(label))
    if set(values) != names:
        raise ValueError("TEST label coverage mismatch")
    return dict(values)


def _empty() -> dict:
    return {
        "datasets": 0,
        "candidate_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "f1_sum": 0.0,
        "candidate_zero_datasets": 0,
        "zero_f1_datasets": 0,
    }


def _add(aggregate: dict, indices: tuple[int, ...] | list[int], labels: list[str]) -> None:
    metrics = evaluate_indices(indices, labels)
    aggregate["datasets"] += 1
    for key in ("candidate_count", "tp", "fp", "fn"):
        aggregate[key] += int(metrics[key])
    aggregate["f1_sum"] += float(metrics["f1"])
    aggregate["candidate_zero_datasets"] += int(metrics["candidate_count"]) == 0
    aggregate["zero_f1_datasets"] += float(metrics["f1"]) == 0.0


def _summary_row(key: tuple, aggregate: dict) -> dict:
    structure, rule_id, risk_target, lane = key
    tp = aggregate["tp"]
    fp = aggregate["fp"]
    fn = aggregate["fn"]
    return {
        "experiment_id": "experiment_385_pipeline_structures",
        "structure": structure,
        "rule_id": rule_id,
        "risk_target_per_100": risk_target,
        "lane": lane,
        **{key: value for key, value in aggregate.items() if key != "f1_sum"},
        "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
        "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
        "mean_f1": aggregate["f1_sum"] / aggregate["datasets"],
        "autonomous": lane in {"hard", "replacement_research"},
        "human_assisted": lane in {
            "review_addition",
            "review_union",
        },
        "selected_for_deployment": False,
        "retrospective_counterfactual": True,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output_root: Path) -> None:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    registry_rows = _load_jsonl(REGISTRY_PATH, REGISTRY_MANIFEST)
    exp359_rows = _load_jsonl(EXP359_PATH, EXP359_MANIFEST)
    registry = {row["dataset"]: row for row in registry_rows}
    exp359 = {row["dataset"]: row for row in exp359_rows}
    if set(registry) != set(exp359) or len(registry) != 1117:
        raise ValueError("Exp384/Exp359 population mismatch")

    labels_by_dataset = _load_labels(set(registry))
    with RISK_PATH.open(newline="", encoding="utf-8") as handle:
        risk_rows = {row["rule_id"]: row for row in csv.DictReader(handle)}
    rules = generate_rule_specs(SEVEN_SOURCES)
    aggregates: dict[tuple, dict] = defaultdict(_empty)

    for done, dataset in enumerate(sorted(registry), start=1):
        row = registry[dataset]
        review = exp359[dataset].get("review_union_indices", [])
        labels = labels_by_dataset[dataset]
        _add(aggregates[("baseline", "baseline", "", "hard")], row["baseline_hard_indices"], labels)
        for rule in rules:
            result = evaluate_rule(
                rule,
                source_indices=row["source_indices"],
                source_available=row["source_available"],
            )
            lanes = build_pipeline_lanes(
                rule_indices=result.indices,
                baseline_hard_indices=row["baseline_hard_indices"],
                exact_normal_indices=row["automatic_normal_exact_indices"],
                existing_review_indices=review,
            )
            _add(
                aggregates[
                    ("baseline_plus_review", rule.rule_id, "", "review_addition")
                ],
                lanes.review_additions,
                labels,
            )
            _add(
                aggregates[
                    ("baseline_plus_review", rule.rule_id, "", "review_union")
                ],
                lanes.review_union,
                labels,
            )
            _add(
                aggregates[
                    ("confidence_badge", rule.rule_id, "", "hard_confirmed")
                ],
                lanes.baseline_confirmed,
                labels,
            )
            _add(
                aggregates[
                    ("confidence_badge", rule.rule_id, "", "hard_unconfirmed")
                ],
                lanes.baseline_unconfirmed,
                labels,
            )
            _add(
                aggregates[
                    ("replacement_research", rule.rule_id, "", "replacement_research")
                ],
                lanes.replacement,
                labels,
            )
            risk = risk_rows[rule.rule_id]
            if risk["risk_status"] == "available":
                for target, field in RISK_TARGET_FIELDS.items():
                    if risk[field] == "1":
                        _add(
                            aggregates[
                                (
                                    "train_risk_hard_expansion",
                                    rule.rule_id,
                                    target,
                                    "hard",
                                )
                            ],
                            lanes.hard_expansion,
                            labels,
                        )
        if done == len(registry) or done % 50 == 0:
            print(
                f"Progress: [{done}/{len(registry)}] rows={done * len(rules)} "
                f"dataset={dataset} errors=0",
                flush=True,
            )

    summary = [
        _summary_row(key, aggregate)
        for key, aggregate in sorted(
            aggregates.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        )
    ]
    coverage = [
        {
            "dataset": dataset,
            "n_test": registry[dataset]["n_test"],
            "baseline_hard_count": len(registry[dataset]["baseline_hard_indices"]),
            "existing_review_count": len(
                exp359[dataset].get("review_union_indices", [])
            ),
            "errors": 0,
        }
        for dataset in sorted(registry)
    ]
    _write_csv(output_root / "exp385_dataset_coverage.csv", coverage)
    _write_csv(output_root / "exp385_pipeline_summary.csv", summary)

    report = [
        "# Exp385 Pipeline Structure Comparison",
        "",
        "- Baseline Hard is preserved in review and confidence structures.",
        "- Exact-normal candidates cannot be reintroduced.",
        "- Review metrics remain human-assisted and separate from autonomous Hard.",
        "- Hard expansion exists only when the precomputed TRAIN Wilson upper bound "
        "meets a frozen sensitivity target.",
        "- Replacement is research-only.",
        "- No structure or rule is selected from TEST results.",
    ]
    (output_root / "exp385_summary.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    outputs = [
        output_root / "exp385_dataset_coverage.csv",
        output_root / "exp385_pipeline_summary.csv",
        output_root / "exp385_summary.md",
    ]
    manifest = {
        "experiment_id": "experiment_385_pipeline_structures",
        "datasets": 1117,
        "rules": 448,
        "input_hashes": {
            str(path): sha256_file(path)
            for path in (
                REGISTRY_PATH,
                REGISTRY_MANIFEST,
                RISK_PATH,
                EXP359_PATH,
                EXP359_MANIFEST,
            )
        },
        "output_hashes": {path.name: sha256_file(path) for path in outputs},
        "hard_and_review_metrics_separated": True,
        "test_results_select_policy": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }
    (output_root / "exp385_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "exp385_status.json").write_text(
        json.dumps(
            {
                "id": "experiment_385_pipeline_structures",
                "experiment_id": "experiment_385_pipeline_structures",
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
