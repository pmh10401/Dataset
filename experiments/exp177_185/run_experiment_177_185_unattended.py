from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp177_185 import core
from experiments.exp151_152.run_experiment_151_152_virtual_run_conformal_policy_v2 import (
    load_dataset_series_only,
)


DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"
DEFAULT_OUTPUT_ROOT = DATA_DIR / "exp177_185_unattended"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
BASELINE_PATH = ROOT / "results/exp137/experiment_137_operational_triage_results.csv"
CACHE_MANIFEST_PATH = ROOT / "outputs/exp153_157_train_only_improvements/01_score_cache_manifest.csv"
EXISTING_CACHE_ROOT = Path(
    "/Users/minho/Documents/timesries project/outputs/exp153_157_train_only_improvements/cache"
)
EXPECTED_DATASETS = 639
STAGE_TIMEOUT_SECONDS = {
    "177": 1800,
    "178": 1800,
    "179": 10800,
    "180": 3600,
    "181": 3600,
    "182": 1800,
    "183": 14400,
    "184": 5400,
    "185": 1800,
}


STAGE_META = {
    "177": ("experiment_177_train_oof_review_policy_freeze", EXPECTED_DATASETS),
    "178": ("experiment_178_frozen_policy_review_prediction", EXPECTED_DATASETS),
    "179": ("experiment_179_canonical_multiscale_feature_cache", EXPECTED_DATASETS),
    "180": ("experiment_180_canonical_score_backend_prediction", EXPECTED_DATASETS),
    "181": ("experiment_181_existing_new_source_fusion_prediction", EXPECTED_DATASETS),
    "182": ("experiment_182_five_hour_checkpoint_evaluation", EXPECTED_DATASETS),
    "183": ("experiment_183_wavelet_multiscale_feature_prediction", EXPECTED_DATASETS),
    "184": ("experiment_184_extension_source_fusion_prediction", EXPECTED_DATASETS),
    "185": ("experiment_185_final_independent_reaggregation", EXPECTED_DATASETS),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def parse_indices(value: object) -> set[int]:
    text = str(value or "").replace(",", " ").strip()
    return {int(token) for token in text.split()} if text else set()


def indices_text(values: set[int] | list[int]) -> str:
    return " ".join(str(value) for value in sorted({int(item) for item in values}))


def dataset_key(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()


def eligible_baseline() -> dict[str, dict[str, str]]:
    selected = sorted(
        (row for row in read_rows(BASELINE_PATH) if int(row["train_normal_count"]) >= 20),
        key=lambda row: row["dataset_name"],
    )
    if EXPECTED_DATASETS < 639:
        selected = selected[:EXPECTED_DATASETS]
    rows = {row["dataset_name"]: row for row in selected}
    if len(rows) != EXPECTED_DATASETS:
        raise ValueError(f"eligible dataset mismatch: {len(rows)} != {EXPECTED_DATASETS}")
    return dict(sorted(rows.items()))


def cache_manifest() -> dict[str, dict[str, str]]:
    rows = {row["dataset_name"]: row for row in read_rows(CACHE_MANIFEST_PATH)}
    if len(rows) != 1117:
        raise ValueError(f"existing score cache coverage mismatch: {len(rows)}")
    return rows


def load_existing_scores(dataset: str, manifest: dict[str, dict[str, str]]) -> dict[str, object]:
    row = manifest[dataset]
    path = EXISTING_CACHE_ROOT / row["cache_file"]
    if core.sha256_file(path) != row["cache_sha256"]:
        raise ValueError(f"score cache hash mismatch: {dataset}")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        sources = {
            name: {
                "train": np.asarray(archive[f"{key}__train_oof"], dtype=float),
                "test": np.asarray(archive[f"{key}__test"], dtype=float),
            }
            for name, key in core.SOURCE_CACHE_KEYS.items()
        }
    return {"metadata": metadata, "sources": sources, "path": path}


def existing_lane_union(row: dict[str, str]) -> set[int]:
    return (
        parse_indices(row["hard_alert_indices"])
        | parse_indices(row["standard_review_indices"])
        | parse_indices(row["priority_review_indices"])
    )


def prediction_paths(stage: str) -> tuple[Path, Path]:
    return (
        OUTPUT_ROOT / f"exp{stage}_frozen_predictions.jsonl",
        OUTPUT_ROOT / f"exp{stage}_prediction_manifest.json",
    )


def stage_csv_paths(stage: str) -> tuple[Path, Path]:
    exp_id, _ = STAGE_META[stage]
    root = DATA_DIR if OUTPUT_ROOT == DEFAULT_OUTPUT_ROOT else OUTPUT_ROOT / "dashboard"
    return root / f"{exp_id}_results.csv", root / f"{exp_id}_summary.csv"


def progress(done: int, expected: int, rows: int, dataset: str, errors: int = 0) -> None:
    print(
        f"Progress: [{done} / {expected}] rows={rows} dataset={dataset} errors={errors}",
        flush=True,
    )


def write_completion(stage: str, payload: dict[str, object]) -> None:
    write_json(
        OUTPUT_ROOT / f"exp{stage}_completion.json",
        {
            "stage": stage,
            "experiment_id": STAGE_META[stage][0],
            "completed_at": utc_now(),
            "retrospective": True,
            "prospective_validated": False,
            **payload,
        },
    )


def train_pvalues_by_source(existing: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        name: core.leave_one_out_pvalues(existing["sources"][name]["train"])
        for name in core.SOURCE_NAMES
    }


def test_pvalues_by_source(existing: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        name: core.empirical_upper_tail_pvalues(
            existing["sources"][name]["train"], existing["sources"][name]["test"]
        )
        for name in core.SOURCE_NAMES
    }


def stage177() -> None:
    baseline = eligible_baseline()
    manifest = cache_manifest()
    false_counts = {alpha: 0 for alpha in core.ALPHAS}
    trials = 0
    dataset_rows = []
    for number, dataset in enumerate(baseline, 1):
        existing = load_existing_scores(dataset, manifest)
        pvalues = train_pvalues_by_source(existing)
        n_train = int(existing["metadata"]["n_train"])
        trials += n_train
        row: dict[str, object] = {
            "experiment_id": STAGE_META["177"][0],
            "dataset_name": dataset,
            "config_name": "train_oof_policy_audit",
            "n_train": n_train,
            "mean_f1": "",
            "metric_scope": "policy_freeze_no_test_evaluation",
        }
        for alpha in core.ALPHAS:
            count = len(core.agreement_indices(pvalues, alpha, 2))
            false_counts[alpha] += count
            row[f"normal_candidates_alpha_{alpha:g}"] = count
        dataset_rows.append(row)
        if number % 25 == 0 or number == len(baseline):
            progress(number, len(baseline), number, dataset)

    policy_variants = []
    summary_rows = []
    for target in core.RISK_TARGETS:
        selected = core.select_alpha_from_train_risk(false_counts, trials, target)
        policy_variants.append({
            "risk_target": target,
            "selected_alpha": selected,
            "selection_source": "TRAIN-normal OOF only",
            "test_metric_selection": False,
        })
        summary_rows.append({
            "experiment_id": STAGE_META["177"][0],
            "config_name": f"risk_target_{target:g}",
            "risk_target": target,
            "selected_alpha": "" if selected is None else selected,
            "train_normal_trials": trials,
            "selected_train_false_alerts": "" if selected is None else false_counts[selected],
            "selected_wilson_upper": "" if selected is None else core.wilson_upper(false_counts[selected], trials),
            "mean_f1": "",
            "metric_scope": "policy_freeze_no_test_evaluation",
        })
    policy = {
        "experiment_id": STAGE_META["177"][0],
        "alphas": list(core.ALPHAS),
        "risk_targets": list(core.RISK_TARGETS),
        "normal_trials": trials,
        "false_alerts_by_alpha": {str(key): value for key, value in false_counts.items()},
        "wilson_upper_by_alpha": {
            str(key): core.wilson_upper(value, trials) for key, value in false_counts.items()
        },
        "policy_variants": policy_variants,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
        "end_to_end_train_only_provenance_validated": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    policy_path = OUTPUT_ROOT / "exp177_frozen_policy.json"
    write_json(policy_path, policy)
    detail_path, summary_path = stage_csv_paths("177")
    write_rows(detail_path, dataset_rows)
    write_rows(summary_path, summary_rows)
    write_rows(OUTPUT_ROOT / "exp177_train_oof_dataset_audit.csv", dataset_rows)
    write_completion("177", {"policy_sha256": core.sha256_file(policy_path), "datasets": len(baseline)})


def load_frozen_policy() -> dict[str, object]:
    path = OUTPUT_ROOT / "exp177_frozen_policy.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_predictions(stage: str, payloads: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    prediction_path, manifest_path = prediction_paths(stage)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            core.validate_prediction_payload(payload)
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "experiment_id": STAGE_META[stage][0],
        "prediction_file": prediction_path.name,
        "prediction_sha256": core.sha256_file(prediction_path),
        "prediction_rows": len(payloads),
        "dataset_count": len({str(item["dataset"]) for item in payloads}),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "created_at": utc_now(),
    }
    write_json(manifest_path, manifest)
    detail_path, summary_path = stage_csv_paths(stage)
    write_rows(detail_path, rows)
    write_rows(summary_path, [{
        "experiment_id": STAGE_META[stage][0],
        "config_name": "frozen_prediction",
        "datasets": manifest["dataset_count"],
        "rows": len(payloads),
        "mean_f1": "",
        "metric_scope": "prediction_only_no_test_labels",
        "prediction_sha256": manifest["prediction_sha256"],
    }])
    write_completion(stage, manifest)


def stage178() -> None:
    baseline = eligible_baseline()
    manifest = cache_manifest()
    policy = load_frozen_policy()
    payloads, rows = [], []
    for number, (dataset, baseline_row) in enumerate(baseline.items(), 1):
        existing = load_existing_scores(dataset, manifest)
        pvalues = test_pvalues_by_source(existing)
        excluded = existing_lane_union(baseline_row)
        for variant in policy["policy_variants"]:
            alpha = variant["selected_alpha"]
            candidates = set() if alpha is None else core.agreement_indices(pvalues, alpha, 2)
            candidates -= excluded
            payload = {
                "experiment_id": STAGE_META["178"][0],
                "variant": f"risk_target_{variant['risk_target']:g}",
                "risk_target": variant["risk_target"],
                "selected_alpha": alpha,
                "dataset": dataset,
                "n_test": int(existing["metadata"]["n_test"]),
                "candidate_indices": sorted(candidates),
                "final_lane": "standard_review_only",
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            }
            payloads.append(payload)
            rows.append({
                "experiment_id": STAGE_META["178"][0], "dataset_name": dataset,
                "config_name": payload["variant"], "candidate_count": len(candidates),
                "candidate_indices": indices_text(candidates), "mean_f1": "",
                "metric_scope": "prediction_only_no_test_labels",
            })
        if number % 25 == 0 or number == len(baseline):
            progress(number, len(baseline), len(rows), dataset)
    freeze_predictions("178", payloads, rows)


def _feature_worker(dataset: str, feature_kind: str, cache_dir_text: str) -> dict[str, object]:
    cache_dir = Path(cache_dir_text)
    x_train, x_test, record = load_dataset_series_only(dataset, DB_PATH)
    extractor = core.canonical_features if feature_kind == "canonical" else core.wavelet_features
    train_features = extractor(x_train)
    test_features = extractor(x_test)
    path = cache_dir / f"{dataset_key(dataset)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_features=train_features,
        test_features=test_features,
        metadata_json=np.asarray(json.dumps({
            "dataset": dataset, "n_train": len(x_train), "n_test": len(x_test),
            "train_target_len": int(record["train_target_len"]), "feature_kind": feature_kind,
        }, sort_keys=True)),
    )
    return {
        "dataset_name": dataset, "n_train": len(x_train), "n_test": len(x_test),
        "train_target_len": int(record["train_target_len"]),
        "feature_count": train_features.shape[1], "cache_file": path.name,
        "cache_sha256": core.sha256_file(path), "status": "completed", "error": "",
    }


def build_feature_cache(stage: str, feature_kind: str) -> None:
    baseline = eligible_baseline()
    workers = int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7"))
    cache_dir = OUTPUT_ROOT / f"{feature_kind}_feature_cache"
    device = core.choose_compute_device()
    # Catch22 and PyWavelets are native CPU libraries. The MPS benchmark is
    # recorded, but mixing hardware paths would change the frozen feature contract.
    device["feature_device"] = "cpu"
    device["feature_device_reason"] = f"{feature_kind} extractor is CPU-native"
    write_json(OUTPUT_ROOT / f"exp{stage}_device_decision.json", device)
    pending = list(baseline)
    completed: dict[str, dict[str, object]] = {}
    errors: dict[str, str] = {}
    for attempt in (1, 2):
        if not pending:
            break
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_feature_worker, dataset, feature_kind, str(cache_dir)): dataset
                for dataset in pending
            }
            pending = []
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    completed[dataset] = future.result()
                    errors.pop(dataset, None)
                except Exception as exc:
                    errors[dataset] = f"{type(exc).__name__}: {exc}"
                    pending.append(dataset)
                progress(len(completed), len(baseline), len(completed), dataset, len(errors))
    rows = [completed[name] for name in sorted(completed)]
    for dataset in sorted(errors):
        rows.append({
            "dataset_name": dataset, "status": "failed", "error": errors[dataset],
            "n_train": baseline[dataset]["train_normal_count"], "n_test": baseline[dataset]["test_size"],
            "train_target_len": "", "feature_count": "", "cache_file": "", "cache_sha256": "",
        })
    detail_path, summary_path = stage_csv_paths(stage)
    write_rows(detail_path, rows)
    write_rows(summary_path, [{
        "experiment_id": STAGE_META[stage][0], "config_name": feature_kind,
        "datasets": len(completed), "errors": len(errors), "rows": len(rows),
        "mean_f1": "", "metric_scope": "feature_cache_no_test_labels",
    }])
    manifest_path = OUTPUT_ROOT / f"exp{stage}_{feature_kind}_feature_manifest.csv"
    write_rows(manifest_path, rows)
    write_completion(stage, {
        "datasets": len(completed), "errors": len(errors),
        "manifest_sha256": core.sha256_file(manifest_path), "workers": workers,
        "device_decision": device,
    })
    if errors or len(completed) != EXPECTED_DATASETS:
        raise SystemExit(2)


def load_feature_manifest(stage: str, feature_kind: str) -> dict[str, dict[str, str]]:
    path = OUTPUT_ROOT / f"exp{stage}_{feature_kind}_feature_manifest.csv"
    rows = {row["dataset_name"]: row for row in read_rows(path) if row["status"] == "completed"}
    if len(rows) != EXPECTED_DATASETS:
        raise ValueError(f"{feature_kind} feature coverage mismatch: {len(rows)}")
    return rows


def load_feature_cache(
    feature_kind: str, row: dict[str, str], cache_root: Path | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    root = Path(cache_root) if cache_root is not None else OUTPUT_ROOT / f"{feature_kind}_feature_cache"
    path = root / row["cache_file"]
    if core.sha256_file(path) != row["cache_sha256"]:
        raise ValueError(f"feature cache hash mismatch: {row['dataset_name']}")
    with np.load(path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["train_features"], dtype=float),
            np.asarray(archive["test_features"], dtype=float),
            json.loads(str(archive["metadata_json"].item())),
        )


def _score_worker(
    dataset: str,
    feature_kind: str,
    manifest_row: dict[str, str],
    feature_dir_text: str,
    score_dir_text: str,
) -> dict[str, object]:
    train_features, test_features, metadata = load_feature_cache(
        feature_kind, manifest_row, Path(feature_dir_text)
    )
    arrays = {}
    for backend in ("knn_local_gap", "robust_diagonal"):
        train_scores, test_scores = core.crossfit_scores(
            train_features, test_features, backend=backend, seed=core.SEED
        )
        arrays[f"{backend}__train_oof"] = train_scores
        arrays[f"{backend}__test"] = test_scores
    path = Path(score_dir_text) / f"{dataset_key(dataset)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    return {
        "dataset_name": dataset, "n_train": metadata["n_train"], "n_test": metadata["n_test"],
        "score_file": path.name, "score_sha256": core.sha256_file(path),
    }


def load_new_scores(feature_kind: str, row: dict[str, str]) -> dict[str, object]:
    path = OUTPUT_ROOT / f"{feature_kind}_score_cache" / row["score_file"]
    if core.sha256_file(path) != row["score_sha256"]:
        raise ValueError(f"score cache hash mismatch: {row['dataset_name']}")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        return {
            "metadata": metadata,
            "knn_local_gap": {
                "train": np.asarray(archive["knn_local_gap__train_oof"], dtype=float),
                "test": np.asarray(archive["knn_local_gap__test"], dtype=float),
            },
            "robust_diagonal": {
                "train": np.asarray(archive["robust_diagonal__train_oof"], dtype=float),
                "test": np.asarray(archive["robust_diagonal__test"], dtype=float),
            },
        }


def score_feature_stage(stage: str, feature_stage: str, feature_kind: str) -> None:
    baseline = eligible_baseline()
    feature_manifest = load_feature_manifest(feature_stage, feature_kind)
    workers = int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7"))
    score_dir = OUTPUT_ROOT / f"{feature_kind}_score_cache"
    feature_dir = OUTPUT_ROOT / f"{feature_kind}_feature_cache"
    score_rows: dict[str, dict[str, object]] = {}
    errors: dict[str, str] = {}
    pending = list(baseline)
    for attempt in (1, 2):
        if not pending:
            break
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _score_worker,
                    dataset,
                    feature_kind,
                    feature_manifest[dataset],
                    str(feature_dir),
                    str(score_dir),
                ): dataset
                for dataset in pending
            }
            pending = []
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    score_rows[dataset] = future.result()
                    errors.pop(dataset, None)
                except Exception as exc:
                    errors[dataset] = f"{type(exc).__name__}: {exc}"
                    pending.append(dataset)
                progress(len(score_rows), len(baseline), len(score_rows), dataset, len(errors))
    if errors or len(score_rows) != EXPECTED_DATASETS:
        error_rows = [
            {"dataset_name": name, "status": "failed", "error": message}
            for name, message in sorted(errors.items())
        ]
        write_rows(OUTPUT_ROOT / f"exp{stage}_score_errors.csv", error_rows)
        raise SystemExit(2)
    score_manifest_path = OUTPUT_ROOT / f"exp{stage}_{feature_kind}_score_manifest.csv"
    write_rows(score_manifest_path, [score_rows[name] for name in sorted(score_rows)])

    payloads, detail_rows = [], []
    for number, (dataset, baseline_row) in enumerate(baseline.items(), 1):
        scores = load_new_scores(feature_kind, score_rows[dataset])
        excluded = existing_lane_union(baseline_row)
        for backend in ("knn_local_gap", "robust_diagonal"):
            pvalues = core.empirical_upper_tail_pvalues(
                scores[backend]["train"], scores[backend]["test"]
            )
            for alpha in core.ALPHAS:
                candidates = set(np.flatnonzero(pvalues <= alpha).astype(int).tolist()) - excluded
                variant = f"{feature_kind}_{backend}_alpha_{alpha:g}"
                payloads.append({
                    "experiment_id": STAGE_META[stage][0], "variant": variant,
                    "dataset": dataset, "n_test": int(scores["metadata"]["n_test"]),
                    "candidate_indices": sorted(candidates), "final_lane": "standard_review_only",
                    "contains_test_labels": False, "contains_evaluation_metrics": False,
                })
                detail_rows.append({
                    "experiment_id": STAGE_META[stage][0], "dataset_name": dataset,
                    "config_name": variant, "candidate_count": len(candidates),
                    "candidate_indices": indices_text(candidates), "mean_f1": "",
                    "metric_scope": "prediction_only_no_test_labels",
                })
        if number % 50 == 0 or number == len(baseline):
            progress(number, len(baseline), len(detail_rows), dataset)
    freeze_predictions(stage, payloads, detail_rows)


def load_score_manifest(stage: str, feature_kind: str) -> dict[str, dict[str, str]]:
    path = OUTPUT_ROOT / f"exp{stage}_{feature_kind}_score_manifest.csv"
    rows = {row["dataset_name"]: row for row in read_rows(path)}
    if len(rows) != EXPECTED_DATASETS:
        raise ValueError(f"score manifest coverage mismatch: {len(rows)}")
    return rows


def fusion_stage(
    stage: str,
    canonical_stage: str,
    wavelet_stage: str | None = None,
) -> None:
    baseline = eligible_baseline()
    existing_manifest = cache_manifest()
    canonical_manifest = load_score_manifest(canonical_stage, "canonical")
    wavelet_manifest = load_score_manifest(wavelet_stage, "wavelet") if wavelet_stage else None
    payloads, detail_rows = [], []
    for number, (dataset, baseline_row) in enumerate(baseline.items(), 1):
        existing = load_existing_scores(dataset, existing_manifest)
        existing_p = test_pvalues_by_source(existing)
        canonical = load_new_scores("canonical", canonical_manifest[dataset])
        canonical_p = {
            backend: core.empirical_upper_tail_pvalues(
                canonical[backend]["train"], canonical[backend]["test"]
            )
            for backend in ("knn_local_gap", "robust_diagonal")
        }
        wavelet_p = None
        if wavelet_manifest:
            wavelet = load_new_scores("wavelet", wavelet_manifest[dataset])
            wavelet_p = {
                backend: core.empirical_upper_tail_pvalues(
                    wavelet[backend]["train"], wavelet[backend]["test"]
                )
                for backend in ("knn_local_gap", "robust_diagonal")
            }
        excluded = existing_lane_union(baseline_row)
        for alpha in core.ALPHAS:
            existing_support = np.sum(np.vstack([values <= alpha for values in existing_p.values()]), axis=0)
            c_knn = canonical_p["knn_local_gap"] <= alpha
            c_robust = canonical_p["robust_diagonal"] <= alpha
            variants = {
                "existing1_plus_canonical_knn": (existing_support >= 1) & c_knn,
                "existing1_plus_canonical_robust": (existing_support >= 1) & c_robust,
                "canonical_backend_agreement": c_knn & c_robust,
                "expanded_2of6": (existing_support + c_knn.astype(int) + c_robust.astype(int)) >= 2,
            }
            if wavelet_p is not None:
                w_knn = wavelet_p["knn_local_gap"] <= alpha
                w_robust = wavelet_p["robust_diagonal"] <= alpha
                variants.update({
                    "canonical_any_plus_wavelet_any": (c_knn | c_robust) & (w_knn | w_robust),
                    "existing1_plus_wavelet_knn": (existing_support >= 1) & w_knn,
                    "expanded_2of8": (
                        existing_support + c_knn.astype(int) + c_robust.astype(int)
                        + w_knn.astype(int) + w_robust.astype(int)
                    ) >= 2,
                })
            for name, mask in variants.items():
                candidates = set(np.flatnonzero(mask).astype(int).tolist()) - excluded
                variant = f"{name}_alpha_{alpha:g}"
                payloads.append({
                    "experiment_id": STAGE_META[stage][0], "variant": variant,
                    "dataset": dataset, "n_test": int(existing["metadata"]["n_test"]),
                    "candidate_indices": sorted(candidates), "final_lane": "standard_review_only",
                    "contains_test_labels": False, "contains_evaluation_metrics": False,
                })
                detail_rows.append({
                    "experiment_id": STAGE_META[stage][0], "dataset_name": dataset,
                    "config_name": variant, "candidate_count": len(candidates),
                    "candidate_indices": indices_text(candidates), "mean_f1": "",
                    "metric_scope": "prediction_only_no_test_labels",
                })
        if number % 25 == 0 or number == len(baseline):
            progress(number, len(baseline), len(detail_rows), dataset)
    freeze_predictions(stage, payloads, detail_rows)


def load_test_labels(dataset: str) -> np.ndarray:
    connection = sqlite3.connect(str(DB_PATH))
    try:
        rows = connection.execute(
            """
            SELECT CAST(i.label AS INTEGER)
            FROM instances i JOIN datasets d ON d.id = i.dataset_id
            WHERE d.name = ? AND i.split = 'TEST'
            ORDER BY i.instance_index
            """,
            (dataset,),
        ).fetchall()
    finally:
        connection.close()
    return np.asarray([int(row[0]) for row in rows], dtype=np.int8)


def load_frozen_predictions(stage: str) -> list[dict[str, object]]:
    prediction_path, manifest_path = prediction_paths(stage)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if core.sha256_file(prediction_path) != manifest["prediction_sha256"]:
        raise ValueError(f"prediction hash mismatch: Exp{stage}")
    payloads = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    for payload in payloads:
        core.validate_prediction_payload(payload)
    return payloads


def evaluate_stages(stage: str, source_stages: tuple[str, ...], report_name: str) -> None:
    baseline = eligible_baseline()
    label_map = {dataset: load_test_labels(dataset) for dataset in baseline}
    output_rows = []
    for source_stage in source_stages:
        for payload in load_frozen_predictions(source_stage):
            labels = label_map[payload["dataset"]]
            truth = set(np.flatnonzero(labels != 0).astype(int).tolist())
            candidates = {int(value) for value in payload["candidate_indices"]}
            tp = len(candidates & truth)
            fp = len(candidates - truth)
            fn = len(truth - candidates)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-15, precision + recall)
            output_rows.append({
                "experiment_id": payload["experiment_id"], "source_stage": source_stage,
                "dataset_name": payload["dataset"], "config_name": payload["variant"],
                "review_candidate_count": len(candidates), "review_tp": tp, "review_fp": fp,
                "review_fn": fn, "review_precision": precision, "review_recall": recall,
                "review_f1": f1, "mean_f1": f1,
                "metric_scope": "human_assisted_review_sensitivity_only",
                "retrospective": 1, "prospective_validated": 0,
            })
    by_variant: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in output_rows:
        by_variant[(str(row["source_stage"]), str(row["config_name"]))].append(row)
    summary_rows = []
    for (source_stage, variant), rows in sorted(by_variant.items()):
        tp = sum(int(row["review_tp"]) for row in rows)
        fp = sum(int(row["review_fp"]) for row in rows)
        counts = [int(row["review_candidate_count"]) for row in rows]
        summary_rows.append({
            "experiment_id": STAGE_META[stage][0], "source_stage": source_stage,
            "config_name": variant, "datasets": len(rows), "review_candidates": sum(counts),
            "review_tp": tp, "review_fp": fp, "review_precision": tp / max(1, tp + fp),
            "mean_f1": float(np.mean([float(row["review_f1"]) for row in rows])),
            "workload_median": float(np.median(counts)), "workload_p95": float(np.percentile(counts, 95)),
            "workload_max": max(counts), "metric_scope": "human_assisted_review_sensitivity_only",
            "retrospective": 1, "prospective_validated": 0, "selected_for_operation": 0,
        })
    detail_path, summary_path = stage_csv_paths(stage)
    write_rows(detail_path, output_rows)
    write_rows(summary_path, summary_rows)
    report_lines = [
        f"# {report_name}", "",
        "All rows are retrospective review-only sensitivity. Exp137 autonomous Hard alerts are unchanged.",
        "No alpha, score backend, or fusion rule is selected for operation from TEST metrics.", "",
        "| Source | Variant | Candidates | TP | FP | Precision | Mean F1 | p95 workload |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| Exp{row['source_stage']} | {row['config_name']} | {row['review_candidates']} | "
            f"{row['review_tp']} | {row['review_fp']} | {float(row['review_precision']):.4f} | "
            f"{float(row['mean_f1']):.4f} | {float(row['workload_p95']):.1f} |"
        )
    report_path = OUTPUT_ROOT / f"exp{stage}_{report_name.lower().replace(' ', '_')}.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_completion(stage, {
        "datasets": len(baseline), "detail_rows": len(output_rows),
        "summary_rows": len(summary_rows), "report_sha256": core.sha256_file(report_path),
    })


def stage185() -> None:
    evaluate_stages("185", ("183", "184"), "Final Extension Evaluation")
    core_summary = read_rows(stage_csv_paths("182")[1])
    extension_summary = read_rows(stage_csv_paths("185")[1])
    combined = core_summary + extension_summary
    write_rows(OUTPUT_ROOT / "exp185_all_variant_summary.csv", combined)
    counts = defaultdict(int)
    for row in combined:
        counts[row["source_stage"]] += 1
    audit = {
        "core_summary_sha256": core.sha256_file(stage_csv_paths("182")[1]),
        "extension_summary_sha256": core.sha256_file(stage_csv_paths("185")[1]),
        "variant_counts_by_source_stage": dict(sorted(counts.items())),
        "all_results_retrospective": True,
        "autonomous_hard_alert_changed": False,
        "test_metric_policy_selection": False,
        "prospective_validated": False,
    }
    write_json(OUTPUT_ROOT / "exp185_independent_reaggregation.json", audit)


def main() -> None:
    global OUTPUT_ROOT, EXPECTED_DATASETS
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=tuple(STAGE_META))
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.dataset_limit:
        EXPECTED_DATASETS = int(args.dataset_limit)
    if args.output_root:
        OUTPUT_ROOT = args.output_root.resolve()
    stage_limit = STAGE_TIMEOUT_SECONDS[args.stage]
    batch_deadline = float(os.environ.get("EXP177_185_BATCH_DEADLINE_UNIX", "0") or 0)
    if batch_deadline:
        stage_limit = min(stage_limit, max(1, int(batch_deadline - time.time())))

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Exp{args.stage} exceeded its unattended runtime limit")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(stage_limit)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"START Exp{args.stage} {STAGE_META[args.stage][0]}", flush=True)
    start = time.time()
    if args.stage == "177":
        stage177()
    elif args.stage == "178":
        stage178()
    elif args.stage == "179":
        build_feature_cache("179", "canonical")
    elif args.stage == "180":
        score_feature_stage("180", "179", "canonical")
    elif args.stage == "181":
        fusion_stage("181", "180")
    elif args.stage == "182":
        evaluate_stages("182", ("178", "180", "181"), "Five Hour Checkpoint Evaluation")
    elif args.stage == "183":
        build_feature_cache("183", "wavelet")
        score_feature_stage("183", "183", "wavelet")
    elif args.stage == "184":
        fusion_stage("184", "180", "183")
    elif args.stage == "185":
        stage185()
    signal.alarm(0)
    print(f"END Exp{args.stage} elapsed_seconds={time.time() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
