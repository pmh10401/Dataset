from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp377_380.feature_combination_core import collapse_family_evidence
from experiments.exp381_385.coverage_audit_core import (
    DatasetContract,
    alpha_resolution,
    audit_exp162_cache,
    sha256_file,
)


REPOSITORY_ROOT = Path("/Users/minho/Documents/timesries project")
DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
EXP357_ROOT = DATA_ROOT / "exp356_359_operational_confidence_tiers"
EXP357_EVIDENCE = EXP357_ROOT / "exp357_frozen_evidence.jsonl"
EXP357_MANIFEST = EXP357_ROOT / "exp357_prediction_manifest.json"
EXP162_ROOT = REPOSITORY_ROOT / "outputs/exp162_full_coverage_alternative_score_cache"
EXP162_CONTRACT = EXP162_ROOT / "01_source_contract.json"
EXP162_MANIFEST = EXP162_ROOT / "02_cache_manifest.csv"
OUTPUT_ROOT = DATA_ROOT / "exp381_385_coverage_first_feature_lattice"
ALPHAS = (0.005, 0.01, 0.02, 0.05)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_value(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_exp357() -> tuple[list[dict], dict]:
    manifest = json.loads(EXP357_MANIFEST.read_text(encoding="utf-8"))
    actual_hash = sha256_file(EXP357_EVIDENCE)
    if actual_hash != manifest.get("prediction_sha256"):
        raise ValueError("Exp357 frozen evidence SHA256 mismatch")
    if manifest.get("contains_test_labels") is not False:
        raise ValueError("Exp357 frozen evidence is not label-free")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("Exp357 frozen evidence contains evaluation metrics")
    rows = [
        json.loads(line)
        for line in EXP357_EVIDENCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest["rows"]):
        raise ValueError("Exp357 row count does not match its manifest")
    datasets = [str(row["dataset"]) for row in rows]
    if len(datasets) != len(set(datasets)):
        raise ValueError("Exp357 contains duplicate datasets")
    return sorted(rows, key=lambda row: row["dataset"]), manifest


def _load_database_contracts(
    database_path: Path,
) -> tuple[dict[str, DatasetContract], list[str]]:
    uri = f"file:{database_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT name, train_normal_count, test_total_count
            FROM datasets
            ORDER BY name
            """
        ).fetchall()
    contracts = {
        str(dataset): DatasetContract(
            dataset=str(dataset),
            n_train=int(n_train),
            n_test=int(n_test),
        )
        for dataset, n_train, n_test in rows
    }
    return contracts, sorted(contracts)


def _find_exp357_provenance(repository_root: Path) -> dict:
    exact_id = "experiment_357_full_population_evidence_vector"
    python_hits: list[str] = []
    raw_score_hits: list[str] = []
    this_file = Path(__file__).resolve()
    for path in (repository_root / "experiments").rglob("*.py"):
        if path.resolve() == this_file:
            continue
        if path.name.startswith("test_") or "audit" in path.name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = str(path.relative_to(repository_root))
        if exact_id in text:
            python_hits.append(relative)
        if "exp357" in text.lower() and "raw_score" in text.lower():
            raw_score_hits.append(relative)
    return {
        "experiment_id": exact_id,
        "frozen_evidence_path": str(EXP357_EVIDENCE),
        "frozen_evidence_sha256": sha256_file(EXP357_EVIDENCE),
        "manifest_path": str(EXP357_MANIFEST),
        "manifest_sha256": sha256_file(EXP357_MANIFEST),
        "source_generation_code_identified": bool(python_hits),
        "source_generation_code_hits": sorted(python_hits),
        "raw_score_provenance_identified": bool(raw_score_hits),
        "raw_score_code_hits": sorted(raw_score_hits),
        "available_artifact_scope": "alpha_0.005_final_candidate_indices_only",
        "provenance_limit": (
            "The frozen artifact stores final candidate indices and availability, "
            "not the raw score vectors needed to regenerate other alpha values."
        ),
    }


def run(output_root: Path) -> None:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "exp381_status.json"
    _write_json(
        status_path,
        {
            "id": "experiment_381_coverage_provenance_audit",
            "experiment_id": "experiment_381_coverage_provenance_audit",
            "status": "running",
            "phase": "loading_contracts",
            "datasets_done": 0,
            "datasets_total": 1117,
            "progress_percent": 0.0,
            "errors": 0,
        },
    )

    exp357_rows, exp357_manifest = _load_exp357()
    population = [str(row["dataset"]) for row in exp357_rows]
    population_set = set(population)
    all_contracts, all_database_names = _load_database_contracts(DB_PATH)
    missing_from_database = sorted(population_set - set(all_contracts))
    if missing_from_database:
        raise ValueError(
            f"Exp357 datasets missing from DB: {len(missing_from_database)}"
        )
    contracts = {
        dataset: (all_contracts[dataset].n_train, all_contracts[dataset].n_test)
        for dataset in population
    }

    exp162_contract = json.loads(EXP162_CONTRACT.read_text(encoding="utf-8"))
    actual_db_hash = sha256_file(DB_PATH)
    if actual_db_hash != exp162_contract.get("database_sha256"):
        raise ValueError("Exp162 contract DB SHA256 differs from current DB")
    print(
        f"Progress: [0/{len(population)}] rows=0 "
        "dataset=cache_integrity_audit errors=0",
        flush=True,
    )
    cache_audit = audit_exp162_cache(
        EXP162_MANIFEST,
        contracts,
        expected_dataset_count=len(population),
        repository_root=REPOSITORY_ROOT,
        expected_source_config_hash=str(
            exp162_contract["source_config_hash"]
        ),
    )

    exp357_by_dataset = {str(row["dataset"]): row for row in exp357_rows}
    cache_by_dataset = {row.dataset: row for row in cache_audit.rows}
    coverage_rows: list[dict] = []
    resolution_rows: list[dict] = []
    availability_histogram: Counter[int] = Counter()
    family_available_counts: Counter[str] = Counter()
    train_bin_counts: Counter[str] = Counter()
    for done, dataset in enumerate(population, start=1):
        contract = all_contracts[dataset]
        family_state = collapse_family_evidence(exp357_by_dataset[dataset])
        available_count = sum(bool(value) for value in family_state["available"].values())
        availability_histogram[available_count] += 1
        train_bin_counts[contract.train_bin] += 1
        for family, available in family_state["available"].items():
            if available:
                family_available_counts[family] += 1
        cache_row = cache_by_dataset.get(dataset)
        coverage_rows.append(
            {
                "dataset": dataset,
                "n_train_normal": contract.n_train,
                "n_test": contract.n_test,
                "train_bin": contract.train_bin,
                "minimum_attainable_p": (
                    ""
                    if contract.minimum_attainable_p is None
                    else f"{contract.minimum_attainable_p:.12g}"
                ),
                "exp357_evidence_alpha": exp357_manifest["evidence_alpha"],
                "trend_available": int(family_state["available"]["trend"]),
                "dynamics_available": int(family_state["available"]["dynamics"]),
                "spectral_available": int(family_state["available"]["spectral"]),
                "normal_distance_available": int(
                    family_state["available"]["normal_distance"]
                ),
                "old_source_family_count": available_count,
                "exp162_cache_present": int(cache_row is not None),
                "exp162_cache_valid": int(cache_row.valid if cache_row else False),
                "exp162_cache_errors": (
                    "|".join(cache_row.errors) if cache_row else "missing_manifest_row"
                ),
            }
        )
        for alpha in ALPHAS:
            resolution_rows.append(
                {
                    "dataset": dataset,
                    "n_train_normal": contract.n_train,
                    "train_bin": contract.train_bin,
                    "alpha": alpha,
                    "minimum_attainable_p": (
                        ""
                        if contract.minimum_attainable_p is None
                        else f"{contract.minimum_attainable_p:.12g}"
                    ),
                    "resolution_status": alpha_resolution(contract.n_train, alpha),
                }
            )
        if done == len(population) or done % 50 == 0:
            print(
                f"Progress: [{done}/{len(population)}] rows={done} "
                f"dataset={dataset} errors={len(cache_row.errors) if cache_row else 1}",
                flush=True,
            )

    _write_csv(
        output_root / "exp381_dataset_coverage.csv",
        coverage_rows,
        list(coverage_rows[0]),
    )
    _write_csv(
        output_root / "exp381_alpha_resolution.csv",
        resolution_rows,
        list(resolution_rows[0]),
    )
    provenance = _find_exp357_provenance(REPOSITORY_ROOT)
    _write_json(output_root / "exp381_exp357_provenance.json", provenance)

    cache_rows = [
        {
            "dataset": row.dataset,
            "file_path": row.file_path,
            "declared_hash": row.declared_hash,
            "actual_hash": row.actual_hash,
            "n_train_expected": row.n_train_expected,
            "n_test_expected": row.n_test_expected,
            "source_config_hash": row.source_config_hash,
            "contains_test_labels": row.contains_test_labels,
            "valid": int(row.valid),
            "errors": "|".join(row.errors),
        }
        for row in cache_audit.rows
    ]
    _write_csv(
        output_root / "exp381_exp162_cache_audit.csv",
        cache_rows,
        list(cache_rows[0]),
    )

    excluded = sorted(set(all_database_names) - population_set)
    resolution_counts = Counter(
        (str(row["alpha"]), str(row["resolution_status"]))
        for row in resolution_rows
    )
    summary_lines = [
        "# Exp381 Coverage and Provenance Audit",
        "",
        "- Classification: retrospective read-only audit",
        f"- Database datasets: {len(all_database_names)}",
        f"- Frozen evaluation population: {len(population)}",
        f"- Database datasets outside frozen population: {len(excluded)}",
        f"- Excluded datasets: {', '.join(excluded) if excluded else 'none'}",
        f"- Exp162 valid cache rows: {cache_audit.valid_count}/{len(population)}",
        f"- Exp162 cache complete: {cache_audit.complete}",
        f"- Exp357 evidence alpha: {exp357_manifest['evidence_alpha']}",
        f"- Exp357 raw score provenance identified: {provenance['raw_score_provenance_identified']}",
        f"- Exp357 source generation code identified: {provenance['source_generation_code_identified']}",
        "",
        "## Existing Four-Source Coverage",
        "",
        f"- all four available: {availability_histogram[4]}",
        f"- no source available: {availability_histogram[0]}",
    ]
    for family in ("trend", "dynamics", "spectral", "normal_distance"):
        summary_lines.append(
            f"- {family} available: {family_available_counts[family]}"
        )
    summary_lines.extend(["", "## TRAIN Size Distribution", ""])
    for train_bin in ("1-4", "5-19", "20-49", "50-99", "100-199", "ge200"):
        summary_lines.append(f"- {train_bin}: {train_bin_counts[train_bin]}")
    summary_lines.extend(["", "## Alpha Resolution", ""])
    for alpha in ALPHAS:
        key = str(alpha)
        summary_lines.append(
            f"- alpha={alpha}: attainable={resolution_counts[(key, 'attainable')]}, "
            f"blocked={resolution_counts[(key, 'resolution_blocked')]}, "
            f"abstained={resolution_counts[(key, 'abstained_n_train_lt5')]}"
        )
    summary_lines.extend(
        [
            "",
            "## Interpretation Limit",
            "",
            "Exp357 provides frozen candidate indices at alpha=0.005. "
            "Its raw score vectors and identifiable source-generation code are "
            "not available in the current repository audit, so other alpha values "
            "must not be reconstructed by approximation.",
        ]
    )
    (output_root / "exp381_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    output_files = [
        output_root / "exp381_dataset_coverage.csv",
        output_root / "exp381_alpha_resolution.csv",
        output_root / "exp381_exp357_provenance.json",
        output_root / "exp381_exp162_cache_audit.csv",
        output_root / "exp381_summary.md",
    ]
    manifest = {
        "experiment_id": "experiment_381_coverage_provenance_audit",
        "classification": "retrospective_read_only_audit",
        "git_commit": _git_value(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        "git_branch": _git_value(REPOSITORY_ROOT, "branch", "--show-current"),
        "database_path": str(DB_PATH),
        "database_sha256": actual_db_hash,
        "input_files": {
            str(path): sha256_file(path)
            for path in (
                EXP357_EVIDENCE,
                EXP357_MANIFEST,
                EXP162_CONTRACT,
                EXP162_MANIFEST,
            )
        },
        "code_files": {
            str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path)
            for path in (
                Path(__file__),
                REPOSITORY_ROOT
                / "experiments/exp381_385/coverage_audit_core.py",
            )
        },
        "output_files": {
            path.name: sha256_file(path)
            for path in output_files
        },
        "dataset_count": len(population),
        "cache_complete": cache_audit.complete,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "prospective_validated": False,
    }
    _write_json(output_root / "exp381_manifest.json", manifest)

    errors = sum(not row.valid for row in cache_audit.rows)
    complete = cache_audit.complete and len(population) == 1117
    _write_json(
        status_path,
        {
            "id": "experiment_381_coverage_provenance_audit",
            "experiment_id": "experiment_381_coverage_provenance_audit",
            "status": "completed" if complete else "failed",
            "phase": "completed" if complete else "integrity_failed",
            "datasets_done": len(population),
            "datasets_total": len(population),
            "progress_percent": 100.0,
            "errors": errors,
            "elapsed_seconds": time.time() - started,
        },
    )
    if not complete:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.output_root)


if __name__ == "__main__":
    main()
