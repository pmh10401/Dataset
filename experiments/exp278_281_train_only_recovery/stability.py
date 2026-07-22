from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.calibration import empirical_upper_pvalues
from experiments.exp276_277_shadow_confidence.freeze import FORECAST_GROUP, _cache_index, _load_arrays

from .contracts import ALPHA, BOOTSTRAP_SEEDS, CONTINUOUS_GROUPS, candidate_rows, write_csv


def bootstrap_recurrence(reference: np.ndarray, query_score: float, *, identity: str) -> float:
    values = np.asarray(reference, dtype=float).reshape(-1)
    if not len(values):
        return 0.0
    hits = 0
    for seed in BOOTSTRAP_SEEDS:
        derived = int.from_bytes(hashlib.sha256(f"{identity}:{seed}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(derived)
        sample = values[rng.integers(0, len(values), size=len(values))]
        pvalue = float(empirical_upper_pvalues(sample, np.asarray([query_score]))[0])
        hits += pvalue <= ALPHA + 1e-15
    return hits / len(BOOTSTRAP_SEEDS)


def _simple_stability(dataset: str, index: int, source: str, caches) -> float:
    arrays = _load_arrays(caches[source][dataset])
    if not len(arrays.get("test_scores", ())):
        return 0.0
    return bootstrap_recurrence(
        arrays["train_oof"], float(arrays["test_scores"][index]),
        identity=f"{dataset}:{index}:{source}:primary",
    )


def _forecast_stability(dataset: str, index: int, caches) -> float:
    values: list[float] = []
    rff = _load_arrays(caches["rff_forecast"][dataset])
    if len(rff.get("test_scores", ())):
        baseline_p = float(empirical_upper_pvalues(rff["train_oof"], rff["test_scores"])[index])
        if baseline_p <= ALPHA + 1e-15:
            values.append(bootstrap_recurrence(
                rff["train_oof"], float(rff["test_scores"][index]),
                identity=f"{dataset}:{index}:rff_forecast:primary",
            ))
    arrays = _load_arrays(caches["tcn_forecast"][dataset])
    if len(arrays.get("test_scores", ())):
        median_p = float(empirical_upper_pvalues(arrays["train_oof"], arrays["test_scores"])[index])
        if median_p <= ALPHA + 1e-15:
            values.append(bootstrap_recurrence(
                arrays["train_oof"], float(arrays["test_scores"][index]),
                identity=f"{dataset}:{index}:tcn:median",
            ))
        seed_pvalues = [float(empirical_upper_pvalues(
            arrays[f"train_oof__seed_{seed}"], arrays[f"test_scores__seed_{seed}"]
        )[index]) for seed in (20260721, 20261721)]
        if max(seed_pvalues) <= ALPHA + 1e-15:
            seed_values = [bootstrap_recurrence(
                arrays[f"train_oof__seed_{seed}"], float(arrays[f"test_scores__seed_{seed}"][index]),
                identity=f"{dataset}:{index}:tcn:{seed}",
            ) for seed in (20260721, 20261721)]
            values.append(min(seed_values))
    if not values:
        raise ValueError(f"forecast support cannot be reconstructed: {dataset}[{index}]")
    return max(values)


def run_calibration_stability(output_root: Path) -> tuple[Path, Path]:
    rows, datasets = candidate_rows(); caches = _cache_index(); detail = []
    for row in rows:
        for lane in ("recovery_shadow_indices", "recovery_priority_indices"):
            for index in row[lane]:
                support = set(row["support_by_index"][str(index)])
                group_values: dict[str, float] = {}
                for source in ("scattering", "signature"):
                    if source in support:
                        group_values[source] = _simple_stability(str(row["dataset"]), index, source, caches)
                if FORECAST_GROUP in support:
                    group_values["normal_forecast"] = _forecast_stability(str(row["dataset"]), index, caches)
                detail.append({
                    "dataset": row["dataset"], "candidate_index": index,
                    "source_lane": lane.removesuffix("_indices"),
                    "continuous_group_count": len(support & CONTINUOUS_GROUPS),
                    "group_stability": json.dumps(group_values, sort_keys=True, separators=(",", ":")),
                    "mean_calibration_stability": sum(group_values.values()) / len(group_values) if group_values else 0.0,
                    "minimum_calibration_stability": min(group_values.values(), default=0.0),
                    "bootstrap_replicates": len(BOOTSTRAP_SEEDS),
                    "contains_test_labels": False, "contains_evaluation_metrics": False,
                })
    summary = [{
        "experiment_id": "experiment_279_train_calibration_stability",
        "datasets": len(datasets), "candidate_rows": len(detail),
        "stability_ge_080": sum(float(row["mean_calibration_stability"]) >= 0.8 for row in detail),
        "stability_ge_060": sum(float(row["mean_calibration_stability"]) >= 0.6 for row in detail),
        "stability_type": "full_reference_size_train_oof_bootstrap",
        "contains_test_labels": False, "contains_evaluation_metrics": False,
        "hard_policy_changed": False, "retrospective": True,
    }]
    root = Path(output_root); detail_path = root / "exp279_candidate_stability.csv"; summary_path = root / "exp279_summary.csv"
    write_csv(detail_path, detail); write_csv(summary_path, summary)
    return detail_path, summary_path


__all__ = ["bootstrap_recurrence", "run_calibration_stability"]
