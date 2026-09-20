from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp386_388.historical_runtime import (
    OUTPUT_ROOT,
    alpha_output_root,
    run_historical_alpha,
    sha256_file,
)


SWEEP_ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20)


def main() -> None:
    started = time.time()
    exact_report = OUTPUT_ROOT / "exp386_exact_replay_report.json"
    if not exact_report.is_file() or not json.loads(exact_report.read_text()).get(
        "exact_prediction_match"
    ):
        raise SystemExit("Exp386 exact replay must pass before Exp387")
    run_records = []
    for number, alpha in enumerate(SWEEP_ALPHAS, start=1):
        run_records.append(
            run_historical_alpha(
                alpha,
                progress_offset=(number - 1) * 1117,
                progress_total=len(SWEEP_ALPHAS) * 1117,
            )
        )
        print(
            f"Progress: [{number}/{len(SWEEP_ALPHAS)}] rows={number * 1117} "
            f"dataset=alpha_{alpha:.3f} errors=0",
            flush=True,
        )
    rows = []
    for alpha in (0.005, *SWEEP_ALPHAS):
        root = alpha_output_root(alpha)
        with (root / "exp357_source_coverage.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"alpha": alpha, **row})
    coverage_path = OUTPUT_ROOT / "exp387_alpha_coverage.csv"
    with coverage_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for alpha in (0.005, *SWEEP_ALPHAS):
        alpha_rows = [row for row in rows if float(row["alpha"]) == alpha]
        summary_rows.append(
            {
                "alpha": alpha,
                "datasets": len(alpha_rows),
                "all_primary_evidence_available": sum(
                    row["normal_distance_available"] == "True"
                    and row["temporal_available"] == "True"
                    and row["structural_available"] == "True"
                    for row in alpha_rows
                ),
                "selected_for_deployment": False,
            }
        )
    summary_path = OUTPUT_ROOT / "exp387_alpha_coverage_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = {
        "experiment_id": "experiment_387_existing_evidence_alpha_sweep",
        "alphas": [0.005, *SWEEP_ALPHAS],
        "datasets_per_alpha": 1117,
        "alpha_selected_from_test_metrics": False,
        "runs": run_records,
        "coverage_sha256": sha256_file(coverage_path),
        "coverage_summary_sha256": sha256_file(summary_path),
        "elapsed_seconds": time.time() - started,
    }
    (OUTPUT_ROOT / "exp387_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_ROOT / "exp387_status.json").write_text(
        json.dumps(
            {
                "id": "experiment_387_existing_evidence_alpha_sweep",
                "status": "completed",
                "phase": "completed",
                "datasets_done": 1117,
                "datasets_total": 1117,
                "alpha_runs_done": 6,
                "progress_percent": 100.0,
                "errors": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
