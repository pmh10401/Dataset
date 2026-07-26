from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp381_385.alternative_score_core import ALTERNATIVE_SOURCES
from experiments.exp381_385.coverage_audit_core import sha256_file
from experiments.exp381_385.source_dependence_core import (
    collapse_correlated_sources,
    robust_normalize,
    spearman_correlation,
)


REPOSITORY_ROOT = Path("/Users/minho/Documents/timesries project")
DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp381_385_coverage_first_feature_lattice"
EXP381_CACHE_AUDIT = OUTPUT_ROOT / "exp381_exp162_cache_audit.csv"
EXP381_PROVENANCE = OUTPUT_ROOT / "exp381_exp357_provenance.json"
THRESHOLDS = (0.70, 0.85, 0.95)
SOURCE_CACHE_PREFIX = {
    "spectral_derivative": "spectral",
    "shapelet_normal_prototype": "shapelet",
    "interval_quantile": "interval",
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_cache_paths() -> dict[str, Path]:
    with EXP381_CACHE_AUDIT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1117 or any(row["valid"] != "1" for row in rows):
        raise ValueError("Exp381 cache audit is not complete")
    return {row["dataset"]: Path(row["file_path"]) for row in rows}


def run(output_root: Path) -> None:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "exp383_status.json"
    status_path.write_text(
        json.dumps(
            {
                "id": "experiment_383_source_dependence_audit",
                "experiment_id": "experiment_383_source_dependence_audit",
                "status": "running",
                "phase": "train_only_correlation",
                "datasets_done": 0,
                "datasets_total": 1117,
                "progress_percent": 0.0,
                "errors": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    cache_paths = _load_cache_paths()
    pairs = tuple(combinations(ALTERNATIVE_SOURCES, 2))
    per_dataset_rows: list[dict] = []
    pooled: dict[str, list[np.ndarray]] = {
        source: [] for source in ALTERNATIVE_SOURCES
    }
    pair_values: dict[tuple[str, str], list[float]] = {pair: [] for pair in pairs}

    for done, dataset in enumerate(sorted(cache_paths), start=1):
        with np.load(cache_paths[dataset], allow_pickle=False) as cache:
            train_scores = {
                source: np.asarray(
                    cache[f"{SOURCE_CACHE_PREFIX[source]}_train"],
                    dtype=np.float64,
                )
                for source in ALTERNATIVE_SOURCES
            }
        for source, values in train_scores.items():
            pooled[source].append(robust_normalize(values))
        for left, right in pairs:
            correlation = spearman_correlation(
                train_scores[left],
                train_scores[right],
            )
            if correlation is not None:
                pair_values[(left, right)].append(correlation)
            per_dataset_rows.append(
                {
                    "dataset": dataset,
                    "n_train_normal": len(train_scores[left]),
                    "source_left": left,
                    "source_right": right,
                    "spearman": "" if correlation is None else correlation,
                    "correlation_available": int(correlation is not None),
                    "uses_train_scores": 1,
                    "uses_test_scores": 0,
                    "uses_test_labels": 0,
                }
            )
        if done == len(cache_paths) or done % 50 == 0:
            print(
                f"Progress: [{done}/{len(cache_paths)}] rows={len(per_dataset_rows)} "
                f"dataset={dataset} errors=0",
                flush=True,
            )

    global_correlations: dict[tuple[str, str], float | None] = {}
    summary_rows: list[dict] = []
    for left, right in pairs:
        values = np.asarray(pair_values[(left, right)], dtype=np.float64)
        global_value = spearman_correlation(
            np.concatenate(pooled[left]),
            np.concatenate(pooled[right]),
        )
        global_correlations[(left, right)] = global_value
        summary_rows.append(
            {
                "source_left": left,
                "source_right": right,
                "datasets_with_correlation": int(values.size),
                "dataset_spearman_mean": float(values.mean()) if values.size else "",
                "dataset_spearman_median": (
                    float(np.median(values)) if values.size else ""
                ),
                "dataset_abs_spearman_p75": (
                    float(np.quantile(np.abs(values), 0.75)) if values.size else ""
                ),
                "dataset_abs_spearman_p90": (
                    float(np.quantile(np.abs(values), 0.90)) if values.size else ""
                ),
                "global_pooled_train_spearman": (
                    "" if global_value is None else global_value
                ),
                "uses_test_scores": 0,
                "uses_test_labels": 0,
            }
        )

    family_maps = {}
    for threshold in THRESHOLDS:
        families = collapse_correlated_sources(
            ALTERNATIVE_SOURCES,
            global_correlations,
            threshold,
        )
        family_maps[str(threshold)] = {
            "threshold": threshold,
            "families": [list(family) for family in families],
            "family_count": len(families),
        }
    family_map_payload = {
        "experiment_id": "experiment_383_source_dependence_audit",
        "sources": list(ALTERNATIVE_SOURCES),
        "correlation_input": "TRAIN scores only",
        "global_correlations": {
            f"{left}__{right}": value
            for (left, right), value in global_correlations.items()
        },
        "sensitivity_maps": family_maps,
        "test_performance_used_to_select_map": False,
        "selected_map": None,
        "existing_four_source_raw_scores_available": False,
        "existing_four_source_limit": (
            "Exp357 retains alpha=0.005 candidate indices but no raw score vectors."
        ),
    }
    family_map_payload["config_sha256"] = hashlib.sha256(
        json.dumps(
            family_map_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    _write_csv(output_root / "exp383_dataset_correlations.csv", per_dataset_rows)
    _write_csv(output_root / "exp383_correlation_summary.csv", summary_rows)
    (output_root / "exp383_source_family_maps.json").write_text(
        json.dumps(family_map_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report = [
        "# Exp383 TRAIN-only Source Dependence Audit",
        "",
        "- Datasets: 1,117",
        "- Sources: spectral derivative, shapelet normal prototype, interval quantile",
        "- Correlation data: TRAIN score vectors only",
        "- TEST scores used: no",
        "- TEST labels used: no",
        "- Correlation thresholds: 0.70, 0.85, 0.95 (all retained as sensitivities)",
        "- Deployment-selected map: none",
        "",
        "## Provenance Limit",
        "",
        "The existing four Exp357 evidence families cannot be included in raw-score "
        "correlation because only their alpha=0.005 frozen candidate indices remain.",
    ]
    (output_root / "exp383_summary.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    outputs = [
        output_root / "exp383_dataset_correlations.csv",
        output_root / "exp383_correlation_summary.csv",
        output_root / "exp383_source_family_maps.json",
        output_root / "exp383_summary.md",
    ]
    manifest = {
        "experiment_id": "experiment_383_source_dependence_audit",
        "datasets": 1117,
        "input_cache_audit_sha256": sha256_file(EXP381_CACHE_AUDIT),
        "input_exp357_provenance_sha256": sha256_file(EXP381_PROVENANCE),
        "output_files": {path.name: sha256_file(path) for path in outputs},
        "contains_test_scores": False,
        "contains_test_labels": False,
        "retrospective_diagnostic": True,
        "selected_for_deployment": False,
    }
    (output_root / "exp383_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(
            {
                "id": "experiment_383_source_dependence_audit",
                "experiment_id": "experiment_383_source_dependence_audit",
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
