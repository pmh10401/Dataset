from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr

from experiments.exp220_223.evaluation import wilson_bound
from experiments.exp226_235_hard_alert_quality.baseline_exp137 import load_exp137_baseline
from experiments.exp226_235_hard_alert_quality.database import (
    audit_database,
    load_test_labels_for_evaluation,
    load_values_bundle,
    sha256_file,
)
from experiments.exp226_235_hard_alert_quality.exp226_train_risk import fold_local_pvalues
from experiments.exp226_235_hard_alert_quality.preprocessing import align_bundle
from experiments.exp226_235_hard_alert_quality.score_cache import (
    FrozenScoreRepository,
    load_verified_score_bundle,
    write_score_bundle,
)
from experiments.exp237_new_score_source_qualification.config import (
    ALPHAS,
    DB_PATH,
    HEARTBEAT_PATH,
    OUTPUT_ROOT,
    REPO_OUTPUT_ROOT,
    SEED,
    SOURCE_MAX_LENGTH,
    SOURCE_NAMES,
)
from experiments.exp237_new_score_source_qualification.core import (
    build_score_bundle,
    deterministic_fold_plan,
)
from experiments.exp237_new_score_source_qualification.prediction import (
    build_prediction_rows,
    freeze_predictions,
    load_verified_predictions,
)
from experiments.exp237_new_score_source_qualification.reporting import (
    evaluate_rows,
    summarize_rows,
    write_csv,
)


EXPERIMENT_ID = "experiment_237_new_score_source_qualification"
EXTERNAL_LIVE_PATH = DB_PATH.parent / "rank_dashboard_external_live_run.json"


def _atomic_text(path: Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, mode="w", encoding="utf-8", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def cache_file_stem(dataset: str) -> str:
    readable = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(dataset))[:80]
    digest = hashlib.sha256(str(dataset).encode("utf-8")).hexdigest()[:16]
    return f"{readable}__{digest}"


def _dataset_names(dataset_limit: int | None) -> tuple[str, ...]:
    names = tuple(sorted(load_exp137_baseline()))
    if len(names) != 1117:
        raise ValueError(f"Exp237 evaluation population mismatch: {len(names)}/1117")
    return names[: int(dataset_limit)] if dataset_limit is not None else names


def _cache_paths(output_root: Path, dataset: str) -> tuple[Path, Path]:
    stem = cache_file_stem(dataset)
    return output_root / "cache" / f"{stem}.npz", output_root / "cache" / f"{stem}.json"


def _cache_manifest_row(output_root: Path, dataset: str) -> dict[str, object]:
    cache_path, manifest_path = _cache_paths(output_root, dataset)
    bundle = load_verified_score_bundle(cache_path, manifest_path)
    provenance = dict(bundle.provenance)
    return {
        "dataset": dataset,
        "cache_file": str(cache_path.relative_to(output_root)),
        "manifest_file": str(manifest_path.relative_to(output_root)),
        "cache_sha256": sha256_file(cache_path),
        "manifest_sha256": sha256_file(manifest_path),
        "n_train": int(provenance.get("n_train", 0)),
        "n_test": int(provenance.get("n_test", 0)),
        "train_derived_target_length": int(provenance.get("train_derived_target_length", 0)),
        "score_length": int(provenance.get("score_length", 0)),
        "calibration_abstained": int(bool(provenance.get("calibration_abstained", False))),
        "abstention_reason": str(provenance.get("abstention_reason", "")),
        "seed": int(provenance.get("seed", SEED)),
        "contains_test_labels": int(bool(provenance.get("contains_test_labels", False))),
    }


def _build_dataset_cache(dataset: str, output_root_text: str, reset: bool) -> dict[str, object]:
    output_root = Path(output_root_text)
    cache_path, manifest_path = _cache_paths(output_root, dataset)
    if not reset and cache_path.exists() and manifest_path.exists():
        return _cache_manifest_row(output_root, dataset)
    raw = load_values_bundle(dataset, DB_PATH)
    aligned = align_bundle(raw)
    scores = build_score_bundle(
        dataset,
        np.asarray(aligned.train_normal),
        np.asarray(aligned.test_values),
        seed=SEED,
    )
    provenance = {
        **dict(scores.provenance),
        "database_sha256": sha256_file(DB_PATH),
        "train_derived_target_length": int(aligned.target_length or 0),
        "score_length": min(int(aligned.target_length or 0), SOURCE_MAX_LENGTH),
        "source_names": SOURCE_NAMES,
        "source_uses_test_labels": False,
        "source_uses_family_name": False,
        "source_uses_test_length_for_policy": False,
    }
    scores = type(scores)(
        dataset=scores.dataset,
        train_oof=scores.train_oof,
        test_scores=scores.test_scores,
        provenance=provenance,
    )
    write_score_bundle(cache_path, manifest_path, scores)
    return _cache_manifest_row(output_root, dataset)


def _status_payload(
    *,
    status: str,
    phase: str,
    done: int,
    expected: int,
    current_dataset: str | None,
    output_root: Path,
    started_unix: float,
    errors: int = 0,
) -> dict[str, object]:
    now = time.time()
    return {
        "id": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "label": "Exp237 - New score-source qualification",
        "status": status,
        "phase": phase,
        "pid": os.getpid(),
        "child_pid": os.getpid(),
        "expected_datasets": expected,
        "datasets_done": done,
        "done": done,
        "expected": expected,
        "progress_percent": 100.0 * done / max(1, expected),
        "rows": done,
        "current_dataset": current_dataset,
        "errors": errors,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_unix)),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "updated_unix": now,
        "detail_csv": str(output_root / "experiment_237_results.csv"),
        "stdout_log": str(output_root / "exp237_full.log"),
        "scope": "retrospective_research_score_source_qualification",
        "autonomous_metric": "not_evaluated",
        "review_metric": "research_candidates_only",
    }


def _write_status(payload: dict[str, object], output_root: Path) -> None:
    _write_json(output_root / "status.json", payload)
    _write_json(EXTERNAL_LIVE_PATH, payload)
    _write_json(HEARTBEAT_PATH, payload)


def _write_manifest_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, mode="w", newline="", encoding="utf-8", delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["dataset"])))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def cache_scores(
    output_root: Path,
    *,
    dataset_limit: int | None,
    workers: int,
    reset_cache: bool,
) -> Path:
    names = _dataset_names(dataset_limit)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    _write_status(
        _status_payload(status="running", phase="cache", done=0, expected=len(names), current_dataset=None,
                        output_root=output_root, started_unix=started), output_root
    )
    if int(workers) == 1:
        futures = None
        iterator = ((name, None) for name in names)
    else:
        executor = ProcessPoolExecutor(max_workers=int(workers))
        future_map = {
            executor.submit(_build_dataset_cache, name, str(output_root), reset_cache): name
            for name in names
        }
        futures = (executor, future_map)
        iterator = ((future_map[future], future) for future in as_completed(future_map))
    try:
        for dataset, future in iterator:
            try:
                row = (
                    _build_dataset_cache(dataset, str(output_root), reset_cache)
                    if future is None
                    else future.result()
                )
                rows.append(row)
            except Exception as exc:
                failures.append({"dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
            done = len(rows) + len(failures)
            _write_manifest_csv(output_root / "score_cache_manifest.csv", rows)
            payload = _status_payload(
                status="running", phase="cache", done=done, expected=len(names), current_dataset=dataset,
                output_root=output_root, started_unix=started, errors=len(failures),
            )
            _write_status(payload, output_root)
            print(f"progress {done}/{len(names)} last={dataset} errors={len(failures)}", flush=True)
    finally:
        if futures is not None:
            futures[0].shutdown(wait=True, cancel_futures=False)
    if failures:
        write_csv(output_root / "cache_errors.csv", failures)
        payload = _status_payload(
            status="failed", phase="cache", done=len(rows), expected=len(names), current_dataset=None,
            output_root=output_root, started_unix=started, errors=len(failures),
        )
        _write_status(payload, output_root)
        raise RuntimeError(f"Exp237 cache failures: {len(failures)}")
    if len(rows) != len(names):
        raise ValueError(f"Exp237 cache coverage mismatch: {len(rows)}/{len(names)}")
    return output_root / "score_cache_manifest.csv"


def validate_prediction_coverage(
    rows: Iterable[dict[str, object]], expected_datasets: Iterable[str]
) -> None:
    expected = set(str(value) for value in expected_datasets)
    counts: dict[str, set[tuple[str, float]]] = {name: set() for name in expected}
    for row in rows:
        dataset = str(row["dataset"])
        if dataset not in counts:
            raise ValueError(f"prediction coverage has unexpected dataset: {dataset}")
        key = (str(row["source"]), float(row["alpha"]))
        if key in counts[dataset]:
            raise ValueError(f"duplicate prediction coverage: {dataset} {key}")
        counts[dataset].add(key)
        indices = [int(value) for value in row.get("candidate_indices", ())]
        if indices and (min(indices) < 0 or max(indices) >= int(row["n_test"])):
            raise ValueError(f"candidate index outside TEST range: {dataset}")
    required = {(source, float(alpha)) for source in SOURCE_NAMES for alpha in ALPHAS}
    missing = [dataset for dataset, actual in counts.items() if actual != required]
    if missing:
        raise ValueError(f"prediction coverage mismatch: {len(missing)} datasets")


def _train_qualification_rows(scores, dataset: str) -> list[dict[str, object]]:
    n_train = int(scores.provenance.get("n_train", 0))
    output: list[dict[str, object]] = []
    for source in SOURCE_NAMES:
        if bool(scores.provenance.get("calibration_abstained", False)):
            for alpha in ALPHAS:
                output.append({
                    "dataset": dataset, "source": source, "alpha": alpha, "n_train": n_train,
                    "train_alerts": 0, "train_alert_rate": 0.0, "wilson_upper_95": 1.0,
                    "minimum_attainable_p": 1.0, "resolution_blocked": 1,
                    "calibration_abstained": 1,
                })
            continue
        plan = deterministic_fold_plan(n_train, seed=SEED)
        pvalues = fold_local_pvalues(scores.train_oof[source], plan)
        minimum = min(1.0 / (len(fit) + 1.0) for fit, _ in plan)
        for alpha in ALPHAS:
            alerts = int(np.count_nonzero(pvalues <= alpha))
            output.append({
                "dataset": dataset, "source": source, "alpha": alpha, "n_train": n_train,
                "train_alerts": alerts, "train_alert_rate": alerts / n_train,
                "wilson_upper_95": wilson_bound(alerts, n_train, side="upper"),
                "minimum_attainable_p": minimum,
                "resolution_blocked": int(minimum > alpha),
                "calibration_abstained": 0,
            })
    return output


def predict(output_root: Path, *, dataset_limit: int | None) -> tuple[Path, Path]:
    names = _dataset_names(dataset_limit)
    rows = []
    qualification: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    frozen_repository = FrozenScoreRepository.from_frozen_manifests()
    for done, dataset in enumerate(names, start=1):
        cache_path, manifest_path = _cache_paths(output_root, dataset)
        scores = load_verified_score_bundle(cache_path, manifest_path)
        rows.extend(build_prediction_rows(scores))
        qualification.extend(_train_qualification_rows(scores, dataset))
        if not bool(scores.provenance.get("calibration_abstained", False)):
            old = frozen_repository.load(dataset)
            for new_name in SOURCE_NAMES:
                for old_name, old_values in sorted(old.train_oof.items()):
                    if len(old_values) != len(scores.train_oof[new_name]) or len(old_values) < 3:
                        continue
                    correlation = float(spearmanr(scores.train_oof[new_name], old_values).statistic)
                    correlation_rows.append({
                        "dataset": dataset,
                        "new_source": new_name,
                        "frozen_source": old_name,
                        "n_train": len(old_values),
                        "train_oof_spearman": correlation if np.isfinite(correlation) else 0.0,
                        "uses_test_labels": 0,
                    })
        payload = _status_payload(
            status="running", phase="predict", done=done, expected=len(names), current_dataset=dataset,
            output_root=output_root, started_unix=time.time(),
        )
        _write_status(payload, output_root)
        if done % 10 == 0 or done == len(names):
            print(f"progress {done}/{len(names)} last={dataset}", flush=True)
    validate_prediction_coverage((vars(row) for row in rows), names)
    prediction_path = output_root / "exp237_frozen_predictions.jsonl"
    manifest_path = output_root / "exp237_prediction_manifest.json"
    freeze_predictions(rows, prediction_path, manifest_path)
    write_csv(output_root / "exp237_train_qualification.csv", qualification)
    if correlation_rows:
        write_csv(output_root / "exp237_train_source_correlations.csv", correlation_rows)
    return prediction_path, manifest_path


def evaluate(output_root: Path, *, dataset_limit: int | None) -> tuple[Path, Path]:
    names = _dataset_names(dataset_limit)
    prediction_path = output_root / "exp237_frozen_predictions.jsonl"
    manifest_path = output_root / "exp237_prediction_manifest.json"
    predictions = load_verified_predictions(prediction_path, manifest_path)
    validate_prediction_coverage(predictions, names)
    evaluated = evaluate_rows(predictions, load_test_labels_for_evaluation)
    summaries = summarize_rows(evaluated)
    result_path = output_root / "experiment_237_results.csv"
    summary_path = output_root / "experiment_237_summary.csv"
    write_csv(result_path, evaluated)
    write_csv(summary_path, summaries)
    payload = _status_payload(
        status="completed", phase="completed", done=len(names), expected=len(names), current_dataset=None,
        output_root=output_root, started_unix=time.time(),
    )
    payload["rows"] = len(evaluated)
    payload["result_summary"] = {"datasets": len(names), "variants": len(summaries)}
    _write_status(payload, output_root)
    return result_path, summary_path


def copy_report_artifacts(output_root: Path, repo_output_root: Path) -> None:
    repo_output_root.mkdir(parents=True, exist_ok=True)
    names = (
        "score_cache_manifest.csv",
        "exp237_prediction_manifest.json",
        "exp237_train_qualification.csv",
        "exp237_train_source_correlations.csv",
        "experiment_237_results.csv",
        "experiment_237_summary.csv",
        "status.json",
    )
    for name in names:
        source = output_root / name
        if source.exists():
            (repo_output_root / name).write_bytes(source.read_bytes())
    audit = audit_database(DB_PATH)
    manifest_rows = []
    for path in sorted(repo_output_root.iterdir()):
        if path.is_file():
            manifest_rows.append({
                "file": path.name,
                "sha256": sha256_file(path),
                "experiment_id": EXPERIMENT_ID,
                "database_sha256": audit.database_sha256,
                "seed": SEED,
                "retrospective": 1,
                "autonomous": 0,
            })
    if manifest_rows:
        write_csv(repo_output_root / "MANIFEST.csv", manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cache", "predict", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--repo-output-root", type=Path, default=REPO_OUTPUT_ROOT)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--reset-cache", action="store_true")
    args = parser.parse_args()
    if args.stage in {"cache", "all"}:
        cache_scores(
            args.output_root,
            dataset_limit=args.dataset_limit,
            workers=args.workers,
            reset_cache=args.reset_cache,
        )
    if args.stage in {"predict", "all"}:
        prediction_path, manifest_path = predict(args.output_root, dataset_limit=args.dataset_limit)
        print(prediction_path)
        print(manifest_path)
    if args.stage in {"evaluate", "all"}:
        result_path, summary_path = evaluate(args.output_root, dataset_limit=args.dataset_limit)
        print(result_path)
        print(summary_path)
        copy_report_artifacts(args.output_root, args.repo_output_root)


if __name__ == "__main__":
    main()
