from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from experiments.exp186_200.core import empirical_upper_tail_pvalues
from experiments.exp206_210.artifacts import (
    freeze_predictions,
    load_test_labels_for_evaluation,
    load_verified_predictions,
)
from experiments.exp206_210.core import sha256_file
from experiments.exp224.policy import load_verified_exp224_policy
from experiments.exp225.core import EXPERIMENT_ID, build_prediction_payload


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
REPO_ROOT = Path(__file__).resolve().parents[2]
EXP219_ROOT = DATA_ROOT / "exp215_217_shapelet_stability"
EXP224_ROOT = DATA_ROOT / "exp224_prospective"
EXP153_ROOT = Path(
    "/Users/minho/Documents/timesries project/outputs/exp153_157_train_only_improvements"
)
EXP177_ROOT = DATA_ROOT / "exp177_185_unattended"
OUTPUT_ROOT = DATA_ROOT / "exp225_fixed_dictionary_confirmation"

EXP219_ID = "experiment_219_fixed_dictionary_frozen_candidate_audit"
EXP219_PRIMARY_VARIANT = "fixed_dictionary_control_conservative_max__0.005"
EXPECTED_EXP219_DATASETS = 251
EXPECTED_EXP224_PROFILES = 639
EXPECTED_EXP153_MANIFEST_SHA = (
    "bf73796c046a6102fce53d3b6fc606afa1ed0a5161ebb6e92d94698ab56a68a0"
)
EXPECTED_CANONICAL_MANIFEST_SHA = (
    "36810c14a5cca10d9758aa55144bcd785e1bae4d12278039cac83af2b96901ce"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_verified_exp219_primary(
    prediction_path: Path,
    manifest_path: Path,
    *,
    expected_datasets: int = EXPECTED_EXP219_DATASETS,
) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual_hash = sha256_file(Path(prediction_path))
    if actual_hash != manifest.get("prediction_sha256"):
        raise ValueError(
            "prediction hash mismatch: "
            f"expected={manifest.get('prediction_sha256')} actual={actual_hash}"
        )
    if manifest.get("experiment_id") != EXP219_ID:
        raise ValueError("unexpected Exp219 experiment ID")
    if manifest.get("contains_test_labels") is not False:
        raise ValueError("Exp219 manifest contains TEST labels")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("Exp219 manifest contains evaluation metrics")
    rows = [
        json.loads(line)
        for line in Path(prediction_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("prediction_rows", -1)):
        raise ValueError("Exp219 prediction row count mismatch")
    selected = [row for row in rows if row.get("variant") == EXP219_PRIMARY_VARIANT]
    if len(selected) != int(expected_datasets):
        raise ValueError(
            f"Exp219 primary coverage mismatch: {len(selected)}/{expected_datasets}"
        )
    datasets = [str(row["dataset"]) for row in selected]
    if len(datasets) != len(set(datasets)):
        raise ValueError("duplicate Exp219 primary dataset")
    for row in selected:
        if row.get("contains_test_labels") is not False:
            raise ValueError("Exp219 prediction contains TEST labels")
        if row.get("contains_evaluation_metrics") is not False:
            raise ValueError("Exp219 prediction contains evaluation metrics")
        if row.get("autonomous_alert") is not False:
            raise ValueError("Exp219 primary must remain review-only")
    return sorted(selected, key=lambda row: str(row["dataset"]))


def _verified_exp224_contract() -> tuple[dict[str, object], dict[str, object]]:
    policy_path = EXP224_ROOT / "policy_bundle.json"
    report_path = EXP224_ROOT / "raw_replay_report.json"
    # Exp224's completed raw replay already rehashed all 639 artifacts. Exp225
    # verifies the immutable policy body and replay hash, then independently
    # verifies every score cache it actually consumes below.
    policy = load_verified_exp224_policy(policy_path, verify_artifacts=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("policy_hash") != policy.get("policy_hash"):
        raise ValueError("Exp224 policy/raw replay hash mismatch")
    if int(report.get("profile_count", -1)) != EXPECTED_EXP224_PROFILES:
        raise ValueError("Exp224 raw replay coverage mismatch")
    if int(report.get("mismatch_count", -1)) != 0:
        raise ValueError("Exp224 raw replay has score mismatches")
    if report.get("shadow_execution_gate") != "passed":
        raise ValueError("Exp224 raw replay gate is not passed")
    alpha = policy.get("selected_alpha")
    if alpha is None or not 0.0 < float(alpha) < 1.0:
        raise ValueError("Exp224 TRAIN-only selected alpha is unavailable")
    profiles = {str(row["profile_id"]) for row in policy.get("profiles", [])}
    if len(profiles) != EXPECTED_EXP224_PROFILES:
        raise ValueError("Exp224 policy profile coverage mismatch")
    return policy, report


def _manifest_map(path: Path, *, expected_sha: str) -> dict[str, dict[str, str]]:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError(
            f"score manifest hash mismatch: path={path} expected={expected_sha} actual={actual}"
        )
    rows = _read_csv(path)
    mapped = {str(row["dataset_name"]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f"duplicate score manifest dataset: {path}")
    return mapped


def _confirmation_pvalues(
    dataset: str,
    source_manifest: dict[str, dict[str, str]],
    canonical_manifest: dict[str, dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    source_row = source_manifest[dataset]
    source_path = EXP153_ROOT / "cache" / source_row["cache_file"]
    if sha256_file(source_path) != source_row["cache_sha256"]:
        raise ValueError(f"temporal score cache hash mismatch: {dataset}")
    with np.load(source_path, allow_pickle=False) as archive:
        temporal_train = np.asarray(archive["rocket_exp40__train_oof"], dtype=float)
        temporal_test = np.asarray(archive["rocket_exp40__test"], dtype=float)

    canonical_row = canonical_manifest[dataset]
    canonical_path = EXP177_ROOT / "canonical_score_cache" / canonical_row["score_file"]
    if sha256_file(canonical_path) != canonical_row["score_sha256"]:
        raise ValueError(f"canonical score cache hash mismatch: {dataset}")
    with np.load(canonical_path, allow_pickle=False) as archive:
        canonical_train = np.asarray(archive["knn_local_gap__train_oof"], dtype=float)
        canonical_test = np.asarray(archive["knn_local_gap__test"], dtype=float)
    if len(temporal_test) != len(canonical_test):
        raise ValueError(f"confirmation TEST score length mismatch: {dataset}")
    return (
        empirical_upper_tail_pvalues(temporal_train, temporal_test),
        empirical_upper_tail_pvalues(canonical_train, canonical_test),
    )


def run_prediction(output_root: Path) -> tuple[Path, Path]:
    exp219_prediction = EXP219_ROOT / "exp219_frozen_predictions.jsonl"
    exp219_manifest = EXP219_ROOT / "exp219_prediction_manifest.json"
    primary_rows = load_verified_exp219_primary(exp219_prediction, exp219_manifest)
    policy, report = _verified_exp224_contract()
    profile_ids = {str(row["profile_id"]) for row in policy["profiles"]}
    if not {str(row["dataset"]) for row in primary_rows} <= profile_ids:
        raise ValueError("Exp219 primary datasets are outside Exp224 coverage")

    source_manifest_path = EXP153_ROOT / "01_score_cache_manifest.csv"
    canonical_manifest_path = EXP177_ROOT / "exp180_canonical_score_manifest.csv"
    source_manifest = _manifest_map(
        source_manifest_path, expected_sha=EXPECTED_EXP153_MANIFEST_SHA
    )
    canonical_manifest = _manifest_map(
        canonical_manifest_path, expected_sha=EXPECTED_CANONICAL_MANIFEST_SHA
    )
    alpha = float(policy["selected_alpha"])
    prediction_rows: list[dict[str, object]] = []
    for done, source in enumerate(primary_rows, 1):
        dataset = str(source["dataset"])
        temporal_p, canonical_p = _confirmation_pvalues(
            dataset, source_manifest, canonical_manifest
        )
        if len(temporal_p) != int(source["n_test"]):
            raise ValueError(f"Exp219/confirmation TEST length mismatch: {dataset}")
        payload = build_prediction_payload(
            dataset=dataset,
            n_test=int(source["n_test"]),
            candidate_indices=source["candidate_indices"],
            temporal_p=temporal_p,
            canonical_p=canonical_p,
            alpha=alpha,
        )
        payload.update(
            {
                "source_candidate_variant": EXP219_PRIMARY_VARIANT,
                "confirmation_policy_hash": policy["policy_hash"],
                "confirmation_sources": [
                    "temporal_random_kernel",
                    "canonical_normal_state",
                ],
            }
        )
        prediction_rows.append(payload)
        if done % 25 == 0 or done == len(primary_rows):
            print(f"Exp225 predict: [{done}/{len(primary_rows)}] {dataset}", flush=True)
    input_hashes = {
        "exp219_prediction_manifest": sha256_file(exp219_manifest),
        "exp219_predictions": sha256_file(exp219_prediction),
        "exp224_policy": sha256_file(EXP224_ROOT / "policy_bundle.json"),
        "exp224_raw_replay_report": sha256_file(
            EXP224_ROOT / "raw_replay_report.json"
        ),
        "exp153_score_manifest": sha256_file(source_manifest_path),
        "canonical_score_manifest": sha256_file(canonical_manifest_path),
    }
    return freeze_predictions(
        "225", EXPERIMENT_ID, prediction_rows, output_root, input_hashes
    )


def _metric(indices: set[int], labels: np.ndarray) -> tuple[int, int, int, float]:
    tp = sum(int(labels[index] == 1) for index in indices)
    fp = len(indices) - tp
    total_anomalies = int(np.sum(labels == 1))
    fn = total_anomalies - tp
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return tp, fp, fn, f1


def evaluate_predictions(
    rows: Iterable[dict[str, object]],
    label_loader: Callable[[str], np.ndarray] = load_test_labels_for_evaluation,
) -> list[dict[str, object]]:
    evaluated: list[dict[str, object]] = []
    for row in rows:
        dataset = str(row["dataset"])
        labels = np.asarray(label_loader(dataset), dtype=np.int8)
        if len(labels) != int(row["n_test"]):
            raise ValueError(f"evaluation label length mismatch: {dataset}")
        candidates = {int(value) for value in row["candidate_indices"]}
        dual = {int(value) for value in row["dual_confirmed_review_indices"]}
        single = {int(value) for value in row["single_confirmed_review_indices"]}
        if (dual & single) or not ((dual | single) <= candidates):
            raise ValueError(f"review lane contract mismatch: {dataset}")
        candidate_tp, candidate_fp, candidate_fn, candidate_f1 = _metric(
            candidates, labels
        )
        dual_tp, dual_fp, dual_fn, dual_f1 = _metric(dual, labels)
        single_tp, single_fp, single_fn, single_f1 = _metric(single, labels)
        confirmed = dual | single
        confirmed_tp, confirmed_fp, confirmed_fn, confirmed_f1 = _metric(
            confirmed, labels
        )
        evaluated.append(
            {
                "dataset": dataset,
                "n_test": len(labels),
                "true_anomaly_count": int(np.sum(labels == 1)),
                "candidate_count": len(candidates),
                "candidate_tp": candidate_tp,
                "candidate_fp": candidate_fp,
                "candidate_fn": candidate_fn,
                "candidate_f1": candidate_f1,
                "dual_review_count": len(dual),
                "dual_review_tp": dual_tp,
                "dual_review_fp": dual_fp,
                "dual_review_fn": dual_fn,
                "dual_review_f1": dual_f1,
                "single_review_count": len(single),
                "single_review_tp": single_tp,
                "single_review_fp": single_fp,
                "single_review_fn": single_fn,
                "single_review_f1": single_f1,
                "confirmed_review_count": len(confirmed),
                "confirmed_review_tp": confirmed_tp,
                "confirmed_review_fp": confirmed_fp,
                "confirmed_review_fn": confirmed_fn,
                "confirmed_review_f1": confirmed_f1,
                "autonomous_alert_count": 0,
                "metric_scope": "retrospective_human_assisted_review",
            }
        )
    return evaluated


def summarize_evaluated(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("evaluated rows are empty")

    def total(name: str) -> int:
        return sum(int(row[name]) for row in rows)

    candidate_count = total("candidate_count")
    candidate_tp = total("candidate_tp")
    candidate_fp = total("candidate_fp")
    confirmed_count = total("dual_review_count") + total("single_review_count")
    confirmed_tp = total("dual_review_tp") + total("single_review_tp")
    confirmed_fp = total("dual_review_fp") + total("single_review_fp")
    return {
        "experiment_id": EXPERIMENT_ID,
        "datasets": len(rows),
        "source_variant": EXP219_PRIMARY_VARIANT,
        "candidate_count": candidate_count,
        "candidate_tp": candidate_tp,
        "candidate_fp": candidate_fp,
        "candidate_precision": candidate_tp / max(1, candidate_count),
        "dual_review_count": total("dual_review_count"),
        "dual_review_tp": total("dual_review_tp"),
        "dual_review_fp": total("dual_review_fp"),
        "single_review_count": total("single_review_count"),
        "single_review_tp": total("single_review_tp"),
        "single_review_fp": total("single_review_fp"),
        "confirmed_review_count": confirmed_count,
        "confirmed_review_tp": confirmed_tp,
        "confirmed_review_fp": confirmed_fp,
        "confirmed_review_precision": confirmed_tp / max(1, confirmed_count),
        "mean_confirmed_review_f1": float(
            np.mean([float(row.get("confirmed_review_f1", 0.0)) for row in rows])
        ),
        "zero_confirmed_f1_datasets": sum(
            float(row.get("confirmed_review_f1", 0.0)) == 0.0 for row in rows
        ),
        "autonomous_alert_count": 0,
        "autonomous_exp137_unchanged": 1,
        "selected_for_operation": 0,
        "metric_scope": "retrospective_human_assisted_review",
    }


def run_evaluation(output_root: Path) -> dict[str, object]:
    prediction_path = output_root / "exp225_frozen_predictions.jsonl"
    manifest_path = output_root / "exp225_prediction_manifest.json"
    rows = load_verified_predictions(prediction_path, manifest_path)
    evaluated = evaluate_predictions(rows)
    summary = summarize_evaluated(evaluated)
    _write_csv(output_root / f"{EXPERIMENT_ID}_results.csv", evaluated)
    _write_csv(output_root / f"{EXPERIMENT_ID}_summary.csv", [summary])
    (output_root / "exp225_completion.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "completed",
                "datasets": len(evaluated),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in {"predict", "all"}:
        run_prediction(args.output_dir)
    if args.phase in {"evaluate", "all"}:
        summary = run_evaluation(args.output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
