from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.preprocessing import preprocessing_contract

from .calibration import calibration_resolution, candidate_indices, empirical_pvalues
from .change_point import score_change_point
from .config import (
    ALPHAS, CPU_WORKERS, DB_PATH, EXPECTED_DATASETS, EXPERIMENT_IDS,
    EXP290_ROOT, EXP291_MANIFEST, EXP291_PREDICTION, OUTPUT_ROOTS, SEED,
)
from .contracts import ScoreArtifact, freeze_prediction_rows, sha256_file, verify_prediction_rows
from .discord import score_discord
from .deep_svdd import score_deep_svdd
from .evidence_fusion import (
    FULL_SOURCE_GROUPS, close_normal_guard_indices, fused_evidence_arrays,
    fuse_indices,
)
from .evaluation import evaluate_experiment
from .foundation import (
    list_datasets_small_to_large,
    load_preprocessed_bundle,
    select_frozen_evaluation_universe,
)
from .contrastive_memory import score_contrastive_memory
from .predictive_residual import score_predictive_residual
from .wavelet_phase import score_wavelet_phase


SOURCE_FUNCTIONS = {
    309: ("multiscale_patch_discord", score_discord),
    310: ("multiwindow_change_point", score_change_point),
    311: ("wavelet_phase_coherence", score_wavelet_phase),
}

NEURAL_SOURCE_FUNCTIONS = {
    313: ("predictive_residual", score_predictive_residual),
    314: ("contrastive_normal_memory", score_contrastive_memory),
    315: ("deep_svdd_normal_boundary", score_deep_svdd),
}


def _frozen_evaluation_datasets(limit: int | None) -> tuple[list[str], list[str]]:
    frozen_rows = verify_prediction_rows(EXP291_PREDICTION, EXP291_MANIFEST)
    frozen_names = {str(row["dataset"]) for row in frozen_rows}
    datasets, excluded = select_frozen_evaluation_universe(
        list_datasets_small_to_large(), frozen_names, expected=EXPECTED_DATASETS,
    )
    return (datasets[: int(limit)] if limit is not None else datasets), excluded


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _publish_status(
    root: Path, experiment_id: str, status: str, phase: str,
    done: int, expected: int, started: float, *, current_dataset: str | None = None,
    errors: list[str] | None = None,
) -> None:
    elapsed = max(0.0, time.time() - started)
    rate = done / elapsed * 60.0 if done and elapsed else None
    eta = (expected - done) / rate * 60.0 if rate else None
    _atomic_json(root / "status.json", {
        "id": experiment_id,
        "experiment_id": experiment_id,
        "status": status,
        "phase": phase,
        "datasets_done": done,
        "done": done,
        "expected_datasets": expected,
        "expected": expected,
        "progress_percent": done / expected * 100.0 if expected else 0.0,
        "current_dataset": current_dataset,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "recent_datasets_per_minute": rate,
        "error_count": len(errors or []),
        "errors": errors or [],
        "retrospective": True,
        "prospective": False,
        "autonomous_metric_scope": "Exp291 automatic Hard unchanged",
        "review_metric_scope": "research Shadow only",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_unix": time.time(),
    })


def _artifact_row(experiment: int, dataset: str, bundle, source_name: str, artifact: ScoreArtifact) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": EXPERIMENT_IDS[int(experiment)],
        "source": source_name,
        "dataset": str(dataset),
        "n_train": int(len(bundle.train_normal)),
        "n_test": int(len(bundle.test_values)),
        "target_length": int(bundle.target_length),
        "available": bool(artifact.available),
        "abstention_reason": artifact.reason,
        "candidate_grain": "test_instance_virtual_run",
        "train_oof_scores": artifact.train_oof_scores.tolist(),
        "test_scores": artifact.test_scores.tolist(),
        "diagnostics": artifact.diagnostics,
        "contains_test_values": False,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "automatic_policy_changed": False,
        "shadow_autonomous": False,
    }
    if artifact.available:
        pvalues = empirical_pvalues(artifact.train_oof_scores, artifact.test_scores)
        row["test_pvalues"] = pvalues.tolist()
        for alpha in ALPHAS:
            key = str(alpha).replace(".", "p")
            resolution = calibration_resolution(len(artifact.train_oof_scores), alpha)
            row[f"alpha_{key}_indices"] = candidate_indices(
                pvalues, alpha, n_reference=len(artifact.train_oof_scores),
            )
            row[f"alpha_{key}_resolution_blocked"] = resolution["resolution_blocked"]
            row[f"alpha_{key}_attainable_p"] = resolution["attainable_p"]
    else:
        row["test_pvalues"] = []
        for alpha in ALPHAS:
            key = str(alpha).replace(".", "p")
            row[f"alpha_{key}_indices"] = []
            row[f"alpha_{key}_resolution_blocked"] = True
            row[f"alpha_{key}_attainable_p"] = None
    return row


def _score_dataset(experiment: int, dataset: str) -> dict[str, object]:
    source_name, function = SOURCE_FUNCTIONS[int(experiment)]
    bundle = load_preprocessed_bundle(dataset)
    artifact: ScoreArtifact = function(bundle.train_normal, bundle.test_values, SEED)
    return _artifact_row(experiment, dataset, bundle, source_name, artifact)


def _score_dataset_neural(experiment: int, dataset: str, device: str) -> dict[str, object]:
    source_name, function = NEURAL_SOURCE_FUNCTIONS[int(experiment)]
    bundle = load_preprocessed_bundle(dataset)
    artifact: ScoreArtifact = function(bundle.train_normal, bundle.test_values, device, SEED)
    return _artifact_row(experiment, dataset, bundle, source_name, artifact)


def run_prediction(
    experiment: int, root: Path, *, dataset_limit: int | None = None,
    workers: int = CPU_WORKERS, source_roots: dict[int, Path] | None = None,
    device: str = "auto",
) -> dict[str, object]:
    if int(experiment) == 312:
        return run_nonlearning_fusion(root, source_roots=source_roots)
    if int(experiment) == 316:
        return run_full_fusion(root, source_roots=source_roots)
    if int(experiment) in NEURAL_SOURCE_FUNCTIONS:
        return run_neural_prediction(
            int(experiment), root, dataset_limit=dataset_limit, device=device,
        )
    if int(experiment) not in SOURCE_FUNCTIONS:
        raise NotImplementedError(f"prediction is not implemented yet for Exp{experiment}")
    datasets, excluded_database_datasets = _frozen_evaluation_datasets(dataset_limit)
    expected = len(datasets)
    started = time.time()
    experiment_id = EXPERIMENT_IDS[int(experiment)]
    root.mkdir(parents=True, exist_ok=True)
    _publish_status(root, experiment_id, "running", "predict", 0, expected, started)
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    if int(workers) <= 1:
        iterator = ((dataset, _score_dataset(experiment, dataset), None) for dataset in datasets)
        for dataset, row, error in iterator:
            if error: errors.append(error)
            else: rows.append(row)
            _publish_status(root, experiment_id, "running", "predict", len(rows), expected, started, current_dataset=dataset, errors=errors)
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), expected)) as pool:
            futures = {pool.submit(_score_dataset, int(experiment), dataset): dataset for dataset in datasets}
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    errors.append(f"{dataset}: {type(exc).__name__}: {exc}")
                _publish_status(root, experiment_id, "running", "predict", len(rows), expected, started, current_dataset=dataset, errors=errors)
    if errors or len(rows) != expected:
        _publish_status(root, experiment_id, "failed", "predict", len(rows), expected, started, errors=errors)
        raise RuntimeError(f"Exp{experiment} prediction failed: rows={len(rows)}/{expected}, errors={errors[:3]}")
    rows.sort(key=lambda row: str(row["dataset"]))
    prediction_path = root / f"exp{experiment}_frozen_predictions.jsonl"
    manifest_path = root / f"exp{experiment}_prediction_manifest.json"
    manifest = freeze_prediction_rows(
        rows, prediction_path, manifest_path, experiment_id=experiment_id,
        metadata={
            "datasets": expected,
            "dataset_limit": dataset_limit,
            "seed": SEED,
            "alphas": list(ALPHAS),
            "database_path": str(DB_PATH),
            "evaluation_universe": "Exp291 frozen 1,117-dataset population",
            "evaluation_universe_prediction_sha256": sha256_file(EXP291_PREDICTION),
            "database_datasets_outside_evaluation_universe": excluded_database_datasets,
            "preprocessing_contract": preprocessing_contract(),
            "candidate_grain": "test_instance_virtual_run",
            "automatic_policy_changed": False,
            "research_shadow_only": True,
            "retrospective": True,
            "prospective_validated": False,
        },
    )
    _publish_status(root, experiment_id, "completed", "predict", expected, expected, started)
    return manifest


def run_neural_prediction(
    experiment: int, root: Path, *, dataset_limit: int | None, device: str,
) -> dict[str, object]:
    datasets, excluded_database_datasets = _frozen_evaluation_datasets(dataset_limit)
    expected = len(datasets); started = time.time(); experiment_id = EXPERIMENT_IDS[int(experiment)]
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []; errors: list[str] = []
    _publish_status(root, experiment_id, "running", "predict", 0, expected, started)
    for dataset in datasets:
        try:
            rows.append(_score_dataset_neural(experiment, dataset, device))
        except Exception as exc:
            errors.append(f"{dataset}: {type(exc).__name__}: {exc}")
        _publish_status(
            root, experiment_id, "running", "predict", len(rows), expected,
            started, current_dataset=dataset, errors=errors,
        )
        if errors:
            break
    if errors or len(rows) != expected:
        _publish_status(root, experiment_id, "failed", "predict", len(rows), expected, started, errors=errors)
        raise RuntimeError(f"Exp{experiment} prediction failed: rows={len(rows)}/{expected}, errors={errors[:3]}")
    rows.sort(key=lambda row: str(row["dataset"]))
    manifest = freeze_prediction_rows(
        rows,
        root / f"exp{experiment}_frozen_predictions.jsonl",
        root / f"exp{experiment}_prediction_manifest.json",
        experiment_id=experiment_id,
        metadata={
            "datasets": expected, "dataset_limit": dataset_limit, "seed": SEED,
            "alphas": list(ALPHAS), "requested_device": device,
            "evaluation_universe": "Exp291 frozen 1,117-dataset population",
            "evaluation_universe_prediction_sha256": sha256_file(EXP291_PREDICTION),
            "database_datasets_outside_evaluation_universe": excluded_database_datasets,
            "preprocessing_contract": preprocessing_contract(),
            "calibration_method": "three_way_train_validation_calibration_split",
            "automatic_policy_changed": False, "research_shadow_only": True,
            "retrospective": True, "prospective_validated": False,
        },
    )
    _publish_status(root, experiment_id, "completed", "predict", expected, expected, started)
    return manifest


def _load_source_rows(number: int, root: Path) -> tuple[dict[str, dict[str, object]], str]:
    prediction = root / f"exp{number}_frozen_predictions.jsonl"
    manifest = root / f"exp{number}_prediction_manifest.json"
    rows = verify_prediction_rows(prediction, manifest)
    return {str(row["dataset"]): row for row in rows}, sha256_file(prediction)


def run_nonlearning_fusion(
    root: Path, *, source_roots: dict[int, Path] | None = None,
) -> dict[str, object]:
    source_roots = source_roots or {number: OUTPUT_ROOTS[number] for number in (309, 310, 311)}
    sources = {number: _load_source_rows(number, source_roots[number]) for number in (309, 310, 311)}
    dataset_sets = [set(rows) for rows, _ in sources.values()]
    if not dataset_sets or any(values != dataset_sets[0] for values in dataset_sets[1:]):
        raise ValueError("Exp309-311 source coverage mismatch")
    datasets = sorted(dataset_sets[0])
    exp291 = {str(row["dataset"]): row for row in verify_prediction_rows(EXP291_PREDICTION, EXP291_MANIFEST)}
    exp290 = {
        str(row["dataset"]): row
        for row in verify_prediction_rows(
            EXP290_ROOT / "exp290_frozen_predictions.jsonl",
            EXP290_ROOT / "exp290_prediction_manifest.json",
        )
    }
    source_names = {
        309: "multiscale_patch_discord",
        310: "multiwindow_change_point",
        311: "wavelet_phase_coherence",
    }
    outputs = []
    for dataset in datasets:
        hard = set(map(int, exp291[dataset]["automatic_hard_indices"]))
        exact_normal = set(map(int, exp290[dataset]["train_raw_exact_normal_indices"]))
        row: dict[str, object] = {
            "experiment_id": EXPERIMENT_IDS[312],
            "dataset": dataset,
            "n_test": int(exp291[dataset]["n_test"]),
            "automatic_hard_indices": sorted(hard),
            "exact_normal_guard_indices": sorted(exact_normal),
            "candidate_grain": "test_instance_virtual_run",
            "contains_test_values": False,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "automatic_policy_changed": False,
            "shadow_autonomous": False,
        }
        for alpha in ALPHAS:
            key = str(alpha).replace(".", "p")
            evidence: dict[str, set[int]] = {}
            pvalues: dict[str, np.ndarray] = {}
            attainable: dict[str, float] = {}
            for number in (309, 310, 311):
                source_row = sources[number][0][dataset]
                name = source_names[number]
                evidence[name] = set(map(int, source_row[f"alpha_{key}_indices"]))
                source_pvalues = np.asarray(source_row["test_pvalues"], dtype=float)
                if len(source_pvalues) == int(row["n_test"]):
                    pvalues[name] = source_pvalues
                value = source_row.get(f"alpha_{key}_attainable_p")
                attainable[name] = float(value) if value is not None else 1.0
            shadow = fuse_indices(
                evidence, exact_normal=exact_normal, excluded=hard, minimum_groups=2,
            )
            row[f"alpha_{key}_shadow_indices"] = sorted(shadow)
            row[f"alpha_{key}_support_by_index"] = {
                str(index): sorted(name for name, indices in evidence.items() if index in indices)
                for index in sorted(set().union(*evidence.values()))
            }
            arrays = fused_evidence_arrays(
                pvalues,
                attainable_by_source={name: attainable[name] for name in pvalues},
            )
            row[f"alpha_{key}_bonferroni_p"] = arrays["bonferroni_p"]
            row[f"alpha_{key}_mean_e"] = arrays["mean_e"]
        outputs.append(row)
    prediction = root / "exp312_frozen_predictions.jsonl"
    manifest_path = root / "exp312_prediction_manifest.json"
    started = time.time()
    _publish_status(root, EXPERIMENT_IDS[312], "running", "predict", 0, len(outputs), started)
    manifest = freeze_prediction_rows(
        outputs, prediction, manifest_path, experiment_id=EXPERIMENT_IDS[312],
        metadata={
            "datasets": len(outputs),
            "source_prediction_sha256": {str(number): values[1] for number, values in sources.items()},
            "exp291_prediction_sha256": sha256_file(EXP291_PREDICTION),
            "exp290_prediction_sha256": sha256_file(EXP290_ROOT / "exp290_frozen_predictions.jsonl"),
            "alphas": list(ALPHAS),
            "minimum_independent_groups": 2,
            "automatic_policy_changed": False,
            "research_shadow_only": True,
            "primary_policy_selected": False,
            "retrospective": True,
            "prospective_validated": False,
        },
    )
    _publish_status(root, EXPERIMENT_IDS[312], "completed", "predict", len(outputs), len(outputs), started)
    return manifest


def run_full_fusion(
    root: Path, *, source_roots: dict[int, Path] | None = None,
) -> dict[str, object]:
    source_roots = source_roots or {number: OUTPUT_ROOTS[number] for number in range(309, 316)}
    sources = {number: _load_source_rows(number, source_roots[number]) for number in range(309, 316) if number != 312}
    dataset_sets = [set(rows) for rows, _ in sources.values()]
    if not dataset_sets or any(values != dataset_sets[0] for values in dataset_sets[1:]):
        raise ValueError("Exp309-315 source coverage mismatch")
    datasets = sorted(dataset_sets[0])
    exp291 = {str(row["dataset"]): row for row in verify_prediction_rows(EXP291_PREDICTION, EXP291_MANIFEST)}
    exp290 = {
        str(row["dataset"]): row
        for row in verify_prediction_rows(
            EXP290_ROOT / "exp290_frozen_predictions.jsonl",
            EXP290_ROOT / "exp290_prediction_manifest.json",
        )
    }
    source_names = {
        309: "multiscale_patch_discord", 310: "multiwindow_change_point",
        311: "wavelet_phase_coherence", 313: "predictive_residual",
        314: "contrastive_normal_memory", 315: "deep_svdd_normal_boundary",
    }
    outputs = []
    for dataset in datasets:
        hard = set(map(int, exp291[dataset]["automatic_hard_indices"]))
        exact_normal = set(map(int, exp290[dataset]["train_raw_exact_normal_indices"]))
        row: dict[str, object] = {
            "experiment_id": EXPERIMENT_IDS[316], "dataset": dataset,
            "n_test": int(exp291[dataset]["n_test"]),
            "automatic_hard_indices": sorted(hard),
            "exact_normal_guard_indices": sorted(exact_normal),
            "contains_test_values": False, "contains_test_labels": False,
            "contains_evaluation_metrics": False, "automatic_policy_changed": False,
            "shadow_autonomous": False, "primary_policy_selected": False,
        }
        for alpha in ALPHAS:
            key = str(alpha).replace(".", "p")
            evidence: dict[str, set[int]] = {}; pvalues: dict[str, np.ndarray] = {}; attainable = {}
            for number, name in source_names.items():
                source_row = sources[number][0][dataset]
                evidence[name] = set(map(int, source_row[f"alpha_{key}_indices"]))
                values = np.asarray(source_row["test_pvalues"], dtype=float)
                if len(values) == int(row["n_test"]):
                    pvalues[name] = values
                    raw_attainable = source_row.get(f"alpha_{key}_attainable_p")
                    attainable[name] = float(raw_attainable) if raw_attainable is not None else 1.0
            close_normal = close_normal_guard_indices(pvalues)
            two_raw = fuse_indices(
                evidence, exact_normal=exact_normal, excluded=hard,
                minimum_groups=2, groups=FULL_SOURCE_GROUPS,
            )
            three_raw = fuse_indices(
                evidence, exact_normal=exact_normal, excluded=hard,
                minimum_groups=3, groups=FULL_SOURCE_GROUPS,
            )
            row[f"alpha_{key}_close_normal_guard_indices"] = sorted(close_normal)
            row[f"alpha_{key}_two_group_shadow_raw_indices"] = sorted(two_raw)
            row[f"alpha_{key}_two_group_shadow_guarded_indices"] = sorted(two_raw - close_normal)
            row[f"alpha_{key}_three_group_shadow_raw_indices"] = sorted(three_raw)
            row[f"alpha_{key}_three_group_shadow_guarded_indices"] = sorted(three_raw - close_normal)
            row[f"alpha_{key}_support_by_index"] = {
                str(index): sorted(name for name, indices in evidence.items() if index in indices)
                for index in sorted(set().union(*evidence.values()))
            }
            arrays = fused_evidence_arrays(pvalues, attainable_by_source=attainable)
            row[f"alpha_{key}_bonferroni_p"] = arrays["bonferroni_p"]
            row[f"alpha_{key}_mean_e"] = arrays["mean_e"]
        outputs.append(row)
    started = time.time(); root.mkdir(parents=True, exist_ok=True)
    _publish_status(root, EXPERIMENT_IDS[316], "running", "predict", 0, len(outputs), started)
    manifest = freeze_prediction_rows(
        outputs, root / "exp316_frozen_predictions.jsonl",
        root / "exp316_prediction_manifest.json", experiment_id=EXPERIMENT_IDS[316],
        metadata={
            "datasets": len(outputs),
            "source_prediction_sha256": {str(number): values[1] for number, values in sources.items()},
            "exp291_prediction_sha256": sha256_file(EXP291_PREDICTION),
            "exp290_prediction_sha256": sha256_file(EXP290_ROOT / "exp290_frozen_predictions.jsonl"),
            "alphas": list(ALPHAS), "method_groups": FULL_SOURCE_GROUPS,
            "shadow_variants": ["two_group_raw", "two_group_guarded", "three_group_raw", "three_group_guarded"],
            "automatic_policy_changed": False, "research_shadow_only": True,
            "primary_policy_selected": False, "retrospective": True,
            "prospective_validated": False,
        },
    )
    _publish_status(root, EXPERIMENT_IDS[316], "completed", "predict", len(outputs), len(outputs), started)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, required=True, choices=tuple(EXPERIMENT_IDS))
    parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=CPU_WORKERS)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-root-309", type=Path)
    parser.add_argument("--source-root-310", type=Path)
    parser.add_argument("--source-root-311", type=Path)
    parser.add_argument("--source-root-313", type=Path)
    parser.add_argument("--source-root-314", type=Path)
    parser.add_argument("--source-root-315", type=Path)
    args = parser.parse_args()
    root = args.output_root or OUTPUT_ROOTS[args.experiment]
    overall_started = time.time()
    try:
        if args.phase in ("predict", "all"):
            source_roots = None
            if args.experiment == 312 and all((args.source_root_309, args.source_root_310, args.source_root_311)):
                source_roots = {309: args.source_root_309, 310: args.source_root_310, 311: args.source_root_311}
            if args.experiment == 316 and all((
                args.source_root_309, args.source_root_310, args.source_root_311,
                args.source_root_313, args.source_root_314, args.source_root_315,
            )):
                source_roots = {
                    309: args.source_root_309, 310: args.source_root_310,
                    311: args.source_root_311, 313: args.source_root_313,
                    314: args.source_root_314, 315: args.source_root_315,
                }
            run_prediction(
                args.experiment, root, dataset_limit=args.dataset_limit,
                workers=args.workers, source_roots=source_roots, device=args.device,
            )
        if args.phase in ("evaluate", "all"):
            prediction_manifest = json.loads((root / f"exp{args.experiment}_prediction_manifest.json").read_text(encoding="utf-8"))
            expected = int(prediction_manifest.get("datasets") or prediction_manifest.get("rows") or 0)
            _publish_status(root, EXPERIMENT_IDS[args.experiment], "running", "evaluate", 0, expected, overall_started)
            evaluate_experiment(args.experiment, root)
            _publish_status(root, EXPERIMENT_IDS[args.experiment], "completed", "evaluate", expected, expected, overall_started)
    except Exception as exc:
        _publish_status(
            root, EXPERIMENT_IDS[args.experiment], "failed", args.phase, 0,
            int(args.dataset_limit or EXPECTED_DATASETS), overall_started,
            errors=[f"{type(exc).__name__}: {exc}"],
        )
        raise


if __name__ == "__main__":
    main()
