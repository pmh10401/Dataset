from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation, load_values_bundle
from experiments.exp226_235_hard_alert_quality.preprocessing import align_bundle
from experiments.exp278_281_train_only_recovery.contracts import write_csv
from experiments.exp282_284_prototype_distance_verifier.distance_engine import fixed_features
from experiments.exp285_287_synthetic_shape_verifier.prediction import verify_prediction


EXP286_ROOT = Path("/Users/minho/Documents/Dataset/exp285_287_synthetic_shape_verifier")


def category(correlation: float, amplitude_log_ratio: float, mean_shift: float, spectral: float) -> str:
    if correlation <= 0.05: return "normal_shape_match"
    if correlation <= 0.15 and (amplitude_log_ratio >= 0.5 or mean_shift >= 2.0): return "amplitude_or_level_only"
    if spectral >= 0.05 and correlation > 0.15: return "spectral_shift"
    return "complex_difference"


def _normalize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float); centered = x - x.mean(axis=1, keepdims=True); scale = x.std(axis=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)


def _spectral_distance(left: np.ndarray, right: np.ndarray) -> float:
    spectra = []
    for row in (left, right):
        power = np.abs(np.fft.rfft(row - row.mean())) ** 2; power[0] = 0.0
        spectra.append(power / power.sum() if power.sum() > 1e-12 else np.zeros_like(power))
    if spectra[0].sum() <= 1e-12 and spectra[1].sum() <= 1e-12: return 0.0
    if spectra[0].sum() <= 1e-12 or spectra[1].sum() <= 1e-12: return 1.0
    freq = np.linspace(0.0, 1.0, len(spectra[0]))
    return float(wasserstein_distance(freq, freq, u_weights=spectra[0], v_weights=spectra[1]))


def _representatives(rows: list[dict[str, object]], outcome: str, count: int = 5) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        if row["outcome"] == outcome: grouped[row["category"]].append(row)
    for values in grouped.values(): values.sort(key=lambda row: float(row["synthetic_shape_score"]), reverse=True)
    selected = []; categories = sorted(grouped)
    while len(selected) < count and any(grouped.values()):
        for name in categories:
            if grouped[name] and len(selected) < count: selected.append(grouped[name].pop(0))
    return selected


def run_audit(output_root: Path) -> tuple[Path, Path]:
    prediction_path = EXP286_ROOT / "exp286_frozen_predictions.jsonl"; manifest_path = EXP286_ROOT / "exp286_prediction_manifest.json"
    prediction_rows = verify_prediction(prediction_path, manifest_path)
    comparisons = []; waveforms = {}
    for row in prediction_rows:
        selected = [record for record in row["candidate_records"] if record["source_lane"] == "recovery_priority" and record["variant_pass"]["synthetic_shape_alpha_005"]]
        if not selected: continue
        labels = load_test_labels_for_evaluation(str(row["dataset"])); bundle = align_bundle(load_values_bundle(str(row["dataset"])))
        train = np.asarray(bundle.train_normal, dtype=float); test = np.asarray(bundle.test_values, dtype=float)
        train_norm = _normalize(train); train_features = fixed_features(train)
        feature_median = np.median(train_features, axis=0); feature_scale = np.median(np.abs(train_features - feature_median), axis=0) * 1.4826
        feature_scale = np.where(feature_scale > 1e-9, feature_scale, 1.0)
        for record in selected:
            index = int(record["candidate_index"]); candidate = test[index]; candidate_norm = _normalize(candidate[None, :])[0]
            similarities = (train_norm @ candidate_norm) / len(candidate_norm)
            if np.linalg.norm(candidate_norm) <= 1e-12:
                similarities[np.linalg.norm(train_norm, axis=1) <= 1e-12] = 1.0
            nearest = int(np.argmax(similarities)); normal = train[nearest]
            correlation = float(np.clip(1.0 - similarities[nearest], 0.0, 2.0))
            candidate_std, normal_std = candidate.std(), normal.std()
            amplitude = abs(math.log((candidate_std + 1e-9) / (normal_std + 1e-9)))
            normal_scale = np.median(np.abs(normal - np.median(normal))) * 1.4826
            if normal_scale <= 1e-9: normal_scale = normal_std if normal_std > 1e-9 else 1.0
            mean_shift = abs(float(candidate.mean() - normal.mean())) / normal_scale
            spectral = _spectral_distance(candidate, normal)
            feature_distance = float(np.linalg.norm((fixed_features(candidate[None, :])[0] - train_features[nearest]) / feature_scale))
            result = {
                "dataset": row["dataset"], "candidate_index": index, "nearest_train_index": nearest,
                "true_label": int(labels[index]), "outcome": "TP" if int(labels[index]) == 1 else "FP",
                "synthetic_shape_score": record["synthetic_shape_score"], "pvalue": record["pvalue"],
                "correlation_distance": correlation, "amplitude_log_ratio": amplitude,
                "normalized_mean_shift": mean_shift, "spectral_wasserstein": spectral,
                "robust_feature_distance": feature_distance,
                "category": category(correlation, amplitude, mean_shift, spectral),
                "retrospective_diagnostic_only": True,
            }
            comparisons.append(result); waveforms[(str(row["dataset"]), index)] = (candidate, normal)
    if Counter(row["outcome"] for row in comparisons) != Counter({"FP": 32, "TP": 11}):
        raise ValueError("Exp288 frozen diagnostic population mismatch")
    root = Path(output_root); detail_path = root / "exp288_counterexample_results.csv"; summary_path = root / "exp288_summary.csv"
    write_csv(detail_path, comparisons)
    summary = []
    for outcome in ("TP", "FP"):
        subset = [row for row in comparisons if row["outcome"] == outcome]
        counts = Counter(row["category"] for row in subset)
        summary.append({"experiment_id": "experiment_288_recovery_priority_counterexample_audit", "outcome": outcome,
                        "count": len(subset), **{name: counts[name] for name in ("normal_shape_match", "amplitude_or_level_only", "spectral_shift", "complex_difference")},
                        "retrospective_diagnostic_only": True, "hard_policy_changed": False})
    write_csv(summary_path, summary)
    cases = _representatives(comparisons, "TP") + _representatives(comparisons, "FP")
    case_rows = []
    for order, row in enumerate(cases, 1):
        case_id = f"{row['outcome'].lower()}_{order:02d}"; candidate, normal = waveforms[(str(row["dataset"]), int(row["candidate_index"]))]
        case_dir = root / "cases"; case_dir.mkdir(parents=True, exist_ok=True)
        for kind, values in (("candidate", candidate), ("nearest_train_normal", normal)):
            with (case_dir / f"{case_id}_{kind}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle); writer.writerow(["time_index", "value"]); writer.writerows(enumerate(values))
        case_rows.append({"case_id": case_id, **row})
    write_csv(root / "exp288_representative_cases.csv", case_rows)
    return detail_path, summary_path


__all__ = ["category", "run_audit"]
