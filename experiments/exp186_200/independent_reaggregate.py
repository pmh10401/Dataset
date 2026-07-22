from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from experiments.exp186_200.artifacts import DATA_DIR, load_verified_predictions
from experiments.exp186_200.run_experiment_186_200 import STAGE_META


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_reaggregation(
    results_dir: Path = DATA_DIR,
    output_root: Path = DATA_DIR / "exp186_200_unattended",
) -> dict[str, object]:
    print("Running Exp200 independent reaggregation arithmetic...", flush=True)

    # 1. Load results and summaries for Exp186 to Exp199
    results_by_stage = {}
    summaries_by_stage = {}

    for stage_id in range(186, 200):
        stage = str(stage_id)
        exp_id = STAGE_META[stage][0]
        results_path = results_dir / f"{exp_id}_results.csv"
        summary_path = results_dir / f"{exp_id}_summary.csv"

        if not results_path.exists():
            raise FileNotFoundError(f"Results CSV not found for stage {stage}: {results_path}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary CSV not found for stage {stage}: {summary_path}")

        results_by_stage[stage] = read_csv_rows(results_path)
        summaries_by_stage[stage] = read_csv_rows(summary_path)

    # 2. Verify independent arithmetic (totals and averages matching summaries)
    print("Verifying summary arithmetic for stages 186-199...", flush=True)
    for stage, results in results_by_stage.items():
        summary_rows = summaries_by_stage[stage]
        # group results by config_name
        by_config = {}
        for row in results:
            by_config.setdefault(row["config_name"], []).append(row)

        for s_row in summary_rows:
            config = s_row["config_name"]
            config_results = by_config.get(config, [])

            if not config_results:
                raise ValueError(f"No results found matching config {config} in stage {stage}")

            # Recompute totals
            datasets = len(config_results)
            candidates = sum(int(r["candidate_count"]) for r in config_results)
            tps = sum(int(r["tp"]) for r in config_results)
            fps = sum(int(r["fp"]) for r in config_results)
            
            calc_precision = tps / max(1, tps + fps)
            calc_mean_f1 = float(np.mean([float(r["f1"]) for r in config_results]))
            
            counts = [int(r["candidate_count"]) for r in config_results]
            calc_median = float(np.median(counts))
            calc_p95 = float(np.percentile(counts, 95))
            calc_max = max(counts)

            # Compare with summary row
            if int(s_row["datasets"]) != datasets:
                raise ValueError(f"Mismatch in datasets count for stage {stage} config {config}: {s_row['datasets']} != {datasets}")
            if int(s_row["review_candidates"]) != candidates:
                raise ValueError(f"Mismatch in review_candidates for stage {stage} config {config}: {s_row['review_candidates']} != {candidates}")
            if int(s_row["review_tp"]) != tps:
                raise ValueError(f"Mismatch in review_tp for stage {stage} config {config}: {s_row['review_tp']} != {tps}")
            if int(s_row["review_fp"]) != fps:
                raise ValueError(f"Mismatch in review_fp for stage {stage} config {config}: {s_row['review_fp']} != {fps}")
            if abs(float(s_row["review_precision"]) - calc_precision) > 1e-5:
                raise ValueError(f"Mismatch in review_precision for stage {stage} config {config}: {s_row['review_precision']} != {calc_precision}")
            if abs(float(s_row["mean_f1"]) - calc_mean_f1) > 1e-5:
                raise ValueError(f"Mismatch in mean_f1 for stage {stage} config {config}: {s_row['mean_f1']} != {calc_mean_f1}")
            if abs(float(s_row["workload_median"]) - calc_median) > 1e-5:
                raise ValueError(f"Mismatch in workload_median for stage {stage} config {config}: {s_row['workload_median']} != {calc_median}")
            if abs(float(s_row["workload_p95"]) - calc_p95) > 1e-5:
                raise ValueError(f"Mismatch in workload_p95 for stage {stage} config {config}: {s_row['workload_p95']} != {calc_p95}")
            if int(s_row["workload_max"]) != calc_max:
                raise ValueError(f"Mismatch in workload_max for stage {stage} config {config}: {s_row['workload_max']} != {calc_max}")

    print("All stage summaries successfully verified against detail results.", flush=True)

    # 3. Compute overlap matrix and unique contributions
    print("Computing candidate overlaps and unique contributions...", flush=True)
    candidates_by_config = {}
    for stage_id in range(186, 200):
        stage = str(stage_id)
        jsonl_path = output_root / f"exp{stage}_frozen_predictions.jsonl"
        manifest_path = output_root / f"exp{stage}_prediction_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payloads = load_verified_predictions(
            jsonl_path,
            manifest_path,
            expected_datasets=int(manifest["dataset_count"]),
        )
        for p in payloads:
            config = f"exp{stage}_{p['variant']}"
            dataset = p["dataset"]
            candidates_by_config.setdefault(config, set()).update(
                (dataset, idx) for idx in p["candidate_indices"]
            )

    configs = sorted(candidates_by_config.keys())
    overlaps = {}
    unique_contributions = {}

    for config in configs:
        other_candidates = set()
        for other_config, other_set in candidates_by_config.items():
            if other_config != config:
                other_candidates.update(other_set)
        unique_contributions[config] = len(candidates_by_config[config] - other_candidates)

        overlaps[config] = {}
        for other_config in configs:
            intersection = len(candidates_by_config[config] & candidates_by_config[other_config])
            overlaps[config][other_config] = intersection

    # 4. Generate Exp200 frontier report
    report_lines = [
        "# Exp200 Frozen Frontier Reaggregation Report",
        "",
        "This report reaggregates Exp186-199 into conservative, balanced, and expanded views.",
        "No policy is recommended or selected from TEST performance.",
        "",
        "## Performance and Workload Frontier",
        "",
        "| Configuration | Total Candidates | Unique Candidates |",
        "|---|---:|---:|",
    ]
    for config in configs:
        total = len(candidates_by_config[config])
        unique = unique_contributions[config]
        report_lines.append(f"| {config} | {total} | {unique} |")

    report_lines.append("")
    report_lines.append("## Pairwise Candidate Overlaps")
    report_lines.append("")

    header = "| Configuration | " + " | ".join(configs) + " |"
    divider = "|---| " + " | ".join("---:" for _ in configs) + " |"
    report_lines.append(header)
    report_lines.append(divider)

    for config in configs:
        row = f"| {config} | " + " | ".join(str(overlaps[config][other]) for other in configs) + " |"
        report_lines.append(row)

    report_path = output_root / "exp200_reaggregation_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Reaggregation report written to {report_path}", flush=True)

    return {
        "report_path": str(report_path),
        "configs": configs,
        "unique_contributions": unique_contributions,
        "overlaps": overlaps,
    }


if __name__ == "__main__":
    run_reaggregation()
