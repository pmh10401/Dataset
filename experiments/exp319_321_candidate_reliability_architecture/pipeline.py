from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.exp309_316_fn_recovery.contracts import verify_prediction_rows
from experiments.exp309_316_fn_recovery.reporting import write_csv, write_result_report
from experiments.exp317_318_train_reliability_fusion.config import (
    EXP290_ROOT, EXP291_MANIFEST, EXP291_PREDICTION, SOURCE_GROUPS,
)
from experiments.exp317_318_train_reliability_fusion.contracts import (
    freeze_jsonl, sha256_file, verify_frozen_jsonl,
)
from experiments.exp317_318_train_reliability_fusion.source_loader import load_verified_sources

from .config import (
    ALPHAS, CONFIDENCE_THRESHOLDS, EXP317_ROOT, EXPERIMENT_IDS, OUTPUT_ROOTS, TOP_KS,
)
from .scoring import calibration_stability, candidate_confidence, robust_margin_strength


def _key(value: float) -> str:
    return str(value).replace(".", "p")


def _threshold_key(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _publish_status(
    root: Path, experiment: int, phase: str, done: int, total: int, started: float,
    dataset: str | None = None, status: str = "running",
) -> None:
    elapsed = max(time.time() - started, 1e-9)
    rate = done / elapsed * 60.0 if done else None
    _atomic_json(root / "status.json", {
        "id": EXPERIMENT_IDS[experiment], "experiment_id": EXPERIMENT_IDS[experiment],
        "status": status, "phase": phase, "datasets_done": done, "datasets_total": total,
        "progress_percent": 100.0 * done / total if total else 0.0,
        "current_dataset": dataset, "elapsed_seconds": elapsed,
        "recent_rate_per_minute": rate,
        "eta_seconds": (total - done) / rate * 60.0 if rate else None,
        "updated_at_unix": time.time(), "automatic_policy_changed": False,
        "retrospective": True,
    })


def _load_baselines():
    exp291 = {
        str(row["dataset"]): row
        for row in verify_prediction_rows(EXP291_PREDICTION, EXP291_MANIFEST)
    }
    exp290_prediction = EXP290_ROOT / "exp290_frozen_predictions.jsonl"
    exp290 = {
        str(row["dataset"]): row
        for row in verify_prediction_rows(
            exp290_prediction, EXP290_ROOT / "exp290_prediction_manifest.json",
        )
    }
    return exp291, exp290, exp290_prediction


def _load_reliability(root: Path):
    path = Path(root) / "exp317_frozen_reliability.jsonl"
    manifest_path = Path(root) / "exp317_reliability_manifest.json"
    rows = verify_frozen_jsonl(path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        (str(row["dataset"]), str(row["source"]), float(row["alpha"])): row
        for row in rows
    }, manifest, path


def run_exp319(root: Path, dataset_limit: int | None = None) -> dict[str, object]:
    root = Path(root); root.mkdir(parents=True, exist_ok=True); started = time.time()
    sources, source_hashes, frozen_datasets = load_verified_sources()
    reliability, reliability_manifest, reliability_path = _load_reliability(EXP317_ROOT)
    if reliability_manifest.get("source_prediction_sha256") != source_hashes:
        raise ValueError("Exp317 and source hashes differ")
    exp291, exp290, exp290_prediction = _load_baselines()
    datasets = sorted(frozen_datasets)
    if dataset_limit is not None:
        datasets = datasets[: int(dataset_limit)]
    _publish_status(root, 319, "predict", 0, len(datasets), started)
    outputs = []
    for position, dataset in enumerate(datasets, 1):
        base = exp291[dataset]
        hard = set(map(int, base["automatic_hard_indices"]))
        review = set(map(int, base["default_review_indices"]))
        priority = set(map(int, base["research_only_recovery_priority_indices"]))
        exact = set(map(int, exp290[dataset]["train_raw_exact_normal_indices"]))
        row: dict[str, object] = {
            "experiment_id": EXPERIMENT_IDS[319], "dataset": dataset,
            "n_test": int(base["n_test"]), "automatic_hard_indices": sorted(hard),
            "existing_review_indices": sorted(review), "existing_priority_indices": sorted(priority),
            "exact_normal_guard_indices": sorted(exact), "contains_test_labels": False,
            "contains_evaluation_metrics": False, "automatic_policy_changed": False,
            "research_shadow_only": True, "primary_policy_selected": False,
        }
        for alpha in ALPHAS:
            alpha_key = _key(alpha)
            weights = {}
            evidence = {}
            for source, by_dataset in sources.items():
                audit = reliability[(dataset, source, float(alpha))]
                weights[source] = float(audit["weight"]) if audit["available"] else 0.0
                evidence[source] = set(map(int, by_dataset[dataset][f"alpha_{alpha_key}_indices"]))
            source_universe = set().union(*evidence.values()) if evidence else set()
            score_universe = source_universe | review
            confidence_by_index = {}
            support_by_index = {}
            selected = {threshold: set() for threshold in CONFIDENCE_THRESHOLDS}
            for index in sorted(score_universe):
                strengths = {}
                supporting_sources = []
                for source, indices in evidence.items():
                    if index not in indices or weights[source] <= 0.0:
                        continue
                    source_row = sources[source][dataset]
                    train_scores = source_row["train_oof_scores"]
                    test_scores = source_row["test_scores"]
                    if index >= len(test_scores):
                        raise ValueError(f"source score index out of range: {dataset}/{source}/{index}")
                    stability = calibration_stability(train_scores, float(test_scores[index]), alpha)
                    margin = robust_margin_strength(train_scores, float(test_scores[index]), alpha)
                    strengths[source] = 0.5 * stability + 0.5 * margin
                    supporting_sources.append(source)
                confidence = candidate_confidence(weights, strengths, groups=SOURCE_GROUPS)
                confidence_by_index[str(index)] = confidence.confidence
                support_by_index[str(index)] = {
                    "supporting_sources": sorted(supporting_sources),
                    "supporting_groups": list(confidence.supporting_group_names),
                    "available_groups": confidence.available_groups,
                    "supporting_groups_count": confidence.supporting_groups,
                    "confidence": confidence.confidence,
                }
                if index in source_universe and index not in hard and index not in exact:
                    for threshold in CONFIDENCE_THRESHOLDS:
                        if (
                            confidence.available_groups >= 3
                            and confidence.supporting_groups >= 2
                            and confidence.confidence >= threshold
                        ):
                            selected[threshold].add(index)
            row[f"alpha_{alpha_key}_confidence_by_index"] = confidence_by_index
            row[f"alpha_{alpha_key}_support_by_index"] = support_by_index
            for threshold, indices in selected.items():
                row[f"alpha_{alpha_key}_confidence_{_threshold_key(threshold)}_indices"] = sorted(indices)
        outputs.append(row)
        _publish_status(root, 319, "predict", position, len(datasets), started, dataset)
    manifest = freeze_jsonl(
        outputs, root / "exp319_frozen_predictions.jsonl", root / "exp319_prediction_manifest.json",
        experiment_id=EXPERIMENT_IDS[319], metadata={
            "datasets": len(outputs), "dataset_limit": dataset_limit,
            "alphas": list(ALPHAS), "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
            "candidate_strength": "0.5*five_split_stability+0.5*robust_margin_strength",
            "source_prediction_sha256": source_hashes,
            "exp317_sha256": sha256_file(reliability_path),
            "exp291_sha256": sha256_file(EXP291_PREDICTION),
            "exp290_sha256": sha256_file(exp290_prediction),
            "automatic_policy_changed": False, "research_shadow_only": True,
            "primary_policy_selected": False, "retrospective": True,
        },
    )
    _publish_status(root, 319, "predict", len(datasets), len(datasets), started, status="completed")
    return manifest


def run_exp320(root: Path, exp319_root: Path, dataset_limit: int | None = None) -> dict[str, object]:
    root = Path(root); root.mkdir(parents=True, exist_ok=True); started = time.time()
    rows = verify_frozen_jsonl(
        Path(exp319_root) / "exp319_frozen_predictions.jsonl",
        Path(exp319_root) / "exp319_prediction_manifest.json",
    )
    if dataset_limit is not None:
        rows = rows[: int(dataset_limit)]
    _publish_status(root, 320, "predict", 0, len(rows), started)
    outputs = []
    for position, source in enumerate(rows, 1):
        review = list(map(int, source["existing_review_indices"]))
        row: dict[str, object] = {
            "experiment_id": EXPERIMENT_IDS[320], "dataset": source["dataset"],
            "n_test": source["n_test"], "automatic_hard_indices": source["automatic_hard_indices"],
            "existing_review_indices": sorted(review), "contains_test_labels": False,
            "contains_evaluation_metrics": False, "automatic_policy_changed": False,
            "review_membership_changed": False, "primary_policy_selected": False,
        }
        for alpha in ALPHAS:
            alpha_key = _key(alpha)
            confidence = source[f"alpha_{alpha_key}_confidence_by_index"]
            ranked = sorted(review, key=lambda index: (-float(confidence.get(str(index), 0.0)), index))
            row[f"alpha_{alpha_key}_ranked_review_indices"] = ranked
            row[f"alpha_{alpha_key}_review_confidence"] = {
                str(index): float(confidence.get(str(index), 0.0)) for index in ranked
            }
        outputs.append(row)
        _publish_status(root, 320, "predict", position, len(rows), started, str(source["dataset"]))
    manifest = freeze_jsonl(
        outputs, root / "exp320_frozen_predictions.jsonl", root / "exp320_prediction_manifest.json",
        experiment_id=EXPERIMENT_IDS[320], metadata={
            "datasets": len(outputs), "dataset_limit": dataset_limit,
            "exp319_sha256": sha256_file(Path(exp319_root) / "exp319_frozen_predictions.jsonl"),
            "review_membership_changed": False, "automatic_policy_changed": False,
            "ranking_only": True, "primary_policy_selected": False, "retrospective": True,
        },
    )
    _publish_status(root, 320, "predict", len(rows), len(rows), started, status="completed")
    return manifest


def run_exp321(
    root: Path, exp319_root: Path, exp320_root: Path, dataset_limit: int | None = None,
) -> dict[str, object]:
    root = Path(root); root.mkdir(parents=True, exist_ok=True); started = time.time()
    rows319 = verify_frozen_jsonl(
        Path(exp319_root) / "exp319_frozen_predictions.jsonl",
        Path(exp319_root) / "exp319_prediction_manifest.json",
    )
    rows320 = {str(row["dataset"]): row for row in verify_frozen_jsonl(
        Path(exp320_root) / "exp320_frozen_predictions.jsonl",
        Path(exp320_root) / "exp320_prediction_manifest.json",
    )}
    if dataset_limit is not None:
        rows319 = rows319[: int(dataset_limit)]
    _publish_status(root, 321, "predict", 0, len(rows319), started)
    outputs = []
    for position, source in enumerate(rows319, 1):
        dataset = str(source["dataset"]); ranked = rows320[dataset]
        hard = set(map(int, source["automatic_hard_indices"]))
        review = set(map(int, source["existing_review_indices"]))
        priority = set(map(int, source["existing_priority_indices"]))
        row: dict[str, object] = {
            "experiment_id": EXPERIMENT_IDS[321], "dataset": dataset, "n_test": source["n_test"],
            "automatic_hard_indices": sorted(hard), "existing_review_indices": sorted(review),
            "existing_priority_indices": sorted(priority), "contains_test_labels": False,
            "contains_evaluation_metrics": False, "automatic_policy_changed": False,
            "review_membership_changed": False, "research_shadow_autonomous": False,
            "primary_policy_selected": False,
        }
        existing = hard | review | priority
        for alpha in ALPHAS:
            alpha_key = _key(alpha)
            row[f"alpha_{alpha_key}_ranked_review_indices"] = ranked[f"alpha_{alpha_key}_ranked_review_indices"]
            for threshold in CONFIDENCE_THRESHOLDS:
                field = f"alpha_{alpha_key}_confidence_{_threshold_key(threshold)}_indices"
                row[f"{field}_research_shadow"] = sorted(set(map(int, source[field])) - existing)
        outputs.append(row)
        _publish_status(root, 321, "predict", position, len(rows319), started, dataset)
    manifest = freeze_jsonl(
        outputs, root / "exp321_frozen_predictions.jsonl", root / "exp321_prediction_manifest.json",
        experiment_id=EXPERIMENT_IDS[321], metadata={
            "datasets": len(outputs), "dataset_limit": dataset_limit,
            "exp319_sha256": sha256_file(Path(exp319_root) / "exp319_frozen_predictions.jsonl"),
            "exp320_sha256": sha256_file(Path(exp320_root) / "exp320_frozen_predictions.jsonl"),
            "automatic_policy_changed": False, "review_membership_changed": False,
            "research_shadow_autonomous": False, "primary_policy_selected": False,
            "retrospective": True, "prospective_validated": False,
        },
    )
    _publish_status(root, 321, "predict", len(rows319), len(rows319), started, status="completed")
    return manifest


def _load_labels(rows):
    from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation

    return {
        str(row["dataset"]): np.asarray(
            load_test_labels_for_evaluation(str(row["dataset"])), dtype=np.int8,
        )
        for row in rows
    }


def _candidate_metrics(candidates: set[int], truth: set[int]) -> tuple[int, int, float]:
    tp = len(candidates & truth); fp = len(candidates - truth)
    return tp, fp, tp / len(candidates) if candidates else 0.0


def _write_simple_report(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "# " + title + "\n\n"
        "Retrospective evaluation only. Exp291 automatic Hard is unchanged and no primary policy is selected.\n\n"
        "```json\n" + json.dumps(rows, indent=2, sort_keys=True) + "\n```\n",
        encoding="utf-8",
    )


def evaluate_exp319(root: Path) -> list[dict[str, object]]:
    root = Path(root); prediction = root / "exp319_frozen_predictions.jsonl"
    rows = verify_frozen_jsonl(prediction, root / "exp319_prediction_manifest.json")
    prediction_hash = sha256_file(prediction)
    labels = _load_labels(rows)
    detail = []; summary = []
    for alpha in ALPHAS:
        alpha_key = _key(alpha)
        for threshold in CONFIDENCE_THRESHOLDS:
            variant = f"alpha={alpha}:confidence={threshold:.2f}"
            field = f"alpha_{alpha_key}_confidence_{_threshold_key(threshold)}_indices"
            current = []
            for row in rows:
                dataset = str(row["dataset"]); truth = set(np.flatnonzero(labels[dataset] == 1).tolist())
                candidates = set(map(int, row[field])); tp, fp, precision = _candidate_metrics(candidates, truth)
                existing = set(map(int, row["automatic_hard_indices"])) | set(map(int, row["existing_review_indices"])) | set(map(int, row["existing_priority_indices"]))
                record = {
                    "experiment_id": EXPERIMENT_IDS[319], "dataset": dataset, "variant": variant,
                    "candidate_count": len(candidates), "tp": tp, "fp": fp, "precision": precision,
                    "outside_all_existing_tp": len((candidates - existing) & truth),
                    "retrospective": True, "autonomous": False,
                }
                detail.append(record); current.append(record)
            count = sum(int(row["candidate_count"]) for row in current)
            tp = sum(int(row["tp"]) for row in current); fp = sum(int(row["fp"]) for row in current)
            summary.append({
                "experiment_id": EXPERIMENT_IDS[319], "variant": variant, "datasets": len(current),
                "candidate_count": count, "tp": tp, "fp": fp,
                "precision": tp / count if count else 0.0,
                "outside_all_existing_tp": sum(int(row["outside_all_existing_tp"]) for row in current),
                "automatic_policy_changed": False, "research_shadow_only": True,
                "primary_policy_selected": False, "retrospective": True,
            })
    write_csv(root / "exp319_results.csv", detail); write_csv(root / "exp319_summary.csv", summary)
    write_result_report(root / "RESULT_REPORT.md", EXPERIMENT_IDS[319], summary)
    _atomic_json(root / "exp319_evaluation_manifest.json", {
        "prediction_sha256": prediction_hash, "labels_loaded_after_prediction_freeze": True,
        "primary_policy_selected": False, "automatic_policy_changed": False, "retrospective": True,
    })
    return summary


def evaluate_exp320(root: Path) -> list[dict[str, object]]:
    root = Path(root); prediction = root / "exp320_frozen_predictions.jsonl"
    rows = verify_frozen_jsonl(prediction, root / "exp320_prediction_manifest.json")
    prediction_hash = sha256_file(prediction); labels = _load_labels(rows)
    summary = []
    for alpha in ALPHAS:
        alpha_key = _key(alpha); ranked_global = []
        total_review = 0; total_tp = 0
        for row in rows:
            dataset = str(row["dataset"]); truth = set(np.flatnonzero(labels[dataset] == 1).tolist())
            ranked = list(map(int, row[f"alpha_{alpha_key}_ranked_review_indices"]))
            scores = row[f"alpha_{alpha_key}_review_confidence"]
            total_review += len(ranked); total_tp += len(set(ranked) & truth)
            ranked_global.extend(
                (float(scores[str(index)]), dataset, index, index in truth) for index in ranked
            )
        ranked_global.sort(key=lambda item: (-item[0], item[1], item[2]))
        for k in (*TOP_KS, len(ranked_global)):
            selected = ranked_global[:k]; tp = sum(bool(item[3]) for item in selected)
            summary.append({
                "experiment_id": EXPERIMENT_IDS[320], "variant": f"alpha={alpha}:top_k={k}",
                "alpha": alpha, "top_k": k, "review_total": total_review,
                "selected_count": len(selected), "tp": tp, "fp": len(selected) - tp,
                "precision_at_k": tp / len(selected) if selected else 0.0,
                "full_review_tp": total_tp, "review_membership_changed": False,
                "automatic_policy_changed": False, "ranking_only": True,
                "primary_policy_selected": False, "retrospective": True,
            })
    write_csv(root / "exp320_results.csv", summary); write_csv(root / "exp320_summary.csv", summary)
    _write_simple_report(root / "RESULT_REPORT.md", EXPERIMENT_IDS[320], summary)
    _atomic_json(root / "exp320_evaluation_manifest.json", {
        "prediction_sha256": prediction_hash, "labels_loaded_after_prediction_freeze": True,
        "primary_policy_selected": False, "review_membership_changed": False, "retrospective": True,
    })
    return summary


def evaluate_exp321(root: Path) -> list[dict[str, object]]:
    root = Path(root); prediction = root / "exp321_frozen_predictions.jsonl"
    rows = verify_frozen_jsonl(prediction, root / "exp321_prediction_manifest.json")
    prediction_hash = sha256_file(prediction); labels = _load_labels(rows)
    baseline = {str(row["dataset"]): row for row in verify_prediction_rows(EXP291_PREDICTION, EXP291_MANIFEST)}
    summary = []
    for row in rows:
        if row["automatic_hard_indices"] != baseline[str(row["dataset"])]["automatic_hard_indices"]:
            raise ValueError(f"Exp321 changed automatic Hard: {row['dataset']}")
    for alpha in ALPHAS:
        alpha_key = _key(alpha)
        for threshold in CONFIDENCE_THRESHOLDS:
            field = f"alpha_{alpha_key}_confidence_{_threshold_key(threshold)}_indices_research_shadow"
            count = tp = fp = 0
            for row in rows:
                dataset = str(row["dataset"]); truth = set(np.flatnonzero(labels[dataset] == 1).tolist())
                candidates = set(map(int, row[field])); row_tp, row_fp, _ = _candidate_metrics(candidates, truth)
                count += len(candidates); tp += row_tp; fp += row_fp
            summary.append({
                "experiment_id": EXPERIMENT_IDS[321],
                "variant": f"alpha={alpha}:confidence={threshold:.2f}",
                "datasets": len(rows), "research_shadow_count": count,
                "research_shadow_tp": tp, "research_shadow_fp": fp,
                "research_shadow_precision": tp / count if count else 0.0,
                "automatic_hard_count": sum(len(row["automatic_hard_indices"]) for row in rows),
                "automatic_policy_changed": False, "review_membership_changed": False,
                "research_shadow_autonomous": False, "primary_policy_selected": False,
                "retrospective": True,
            })
    write_csv(root / "exp321_results.csv", summary); write_csv(root / "exp321_summary.csv", summary)
    _write_simple_report(root / "RESULT_REPORT.md", EXPERIMENT_IDS[321], summary)
    _atomic_json(root / "exp321_evaluation_manifest.json", {
        "prediction_sha256": prediction_hash, "labels_loaded_after_prediction_freeze": True,
        "automatic_hard_exact_match_exp291": True, "review_membership_changed": False,
        "primary_policy_selected": False, "retrospective": True,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, choices=(319, 320, 321), required=True)
    parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--exp319-root", type=Path, default=OUTPUT_ROOTS[319])
    parser.add_argument("--exp320-root", type=Path, default=OUTPUT_ROOTS[320])
    args = parser.parse_args(); root = args.output_root or OUTPUT_ROOTS[args.experiment]
    if args.phase in ("predict", "all"):
        if args.experiment == 319:
            run_exp319(root, args.dataset_limit)
        elif args.experiment == 320:
            run_exp320(root, args.exp319_root, args.dataset_limit)
        else:
            run_exp321(root, args.exp319_root, args.exp320_root, args.dataset_limit)
    if args.phase in ("evaluate", "all"):
        {319: evaluate_exp319, 320: evaluate_exp320, 321: evaluate_exp321}[args.experiment](root)


if __name__ == "__main__":
    main()
