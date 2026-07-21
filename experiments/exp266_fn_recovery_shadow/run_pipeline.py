from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .config import EXPECTED_DATASETS, EXPERIMENT_ID, FULL_RUN_WORKERS, OUTPUT_ROOT
from .evaluation import evaluate_predictions
from .inputs import load_frozen_inputs
from .prediction import predict_all, verify_prediction_checkpoint
from .reporting import write_reports
from .train_risk import audit_train_risk


STAGE_ORDER = ("audit", "predict", "risk", "evaluate", "report")
STATUS_FILENAME = "status.json"


def publication_root(output_root: Path, dataset_limit: int | None) -> Path:
    root = Path(output_root)
    return root if dataset_limit is None else root / "smoke" / f"dataset_limit_{dataset_limit}"


def output_paths(output_root: Path) -> dict[str, str]:
    root = Path(output_root)
    return {
        "status": str(root / STATUS_FILENAME),
        "audit": str(root / "00_contract_audit.json"),
        "source_manifest": str(root / "01_source_manifest.csv"),
        "risk_audit": str(root / "02_train_normal_risk_audit.csv"),
        "predictions": str(root / "03_frozen_predictions.jsonl"),
        "prediction_manifest": str(root / "04_prediction_manifest.json"),
        "results": str(root / "05_dataset_results.csv"),
        "summary": str(root / "06_global_summary.csv"),
        "diagnostics": str(root / "07_source_agreement_diagnostics.csv"),
        "invariance_report": str(root / "08_invariance_report.md"),
        "result_report": str(root / "09_result_report.md"),
        "independent_reaggregation": str(root / "10_independent_reaggregation.json"),
        "manifest": str(root / "MANIFEST.csv"),
        "stdout_log": str(root / "full.log"),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write((json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def write_status(
    status_path: Path,
    *,
    status: str,
    stage: str,
    datasets_done: int,
    expected_datasets: int,
    current_dataset: str | None,
    workers: int,
    output_root: Path,
    stdout_log: Path,
    started_unix: float,
    warnings: Sequence[str],
    errors: Sequence[str],
) -> None:
    now = time.time()
    elapsed = max(0.0, now - started_unix)
    rate = datasets_done / elapsed * 60.0 if elapsed > 0 and datasets_done else 0.0
    eta = (expected_datasets - datasets_done) / rate * 60.0 if rate > 0 else None
    root = Path(output_root)
    _atomic_json(
        Path(status_path),
        {
            "id": EXPERIMENT_ID,
            "experiment_id": EXPERIMENT_ID,
            "label": "Exp266 - lane 밖 FN 복구 Shadow",
            "status": status,
            "phase": stage,
            "stage": stage,
            "pid": os.getpid(),
            "workers": workers,
            "done": datasets_done,
            "datasets_done": datasets_done,
            "expected": expected_datasets,
            "expected_datasets": expected_datasets,
            "progress_percent": datasets_done / expected_datasets * 100 if expected_datasets else 0.0,
            "current_dataset": current_dataset,
            "elapsed_seconds": elapsed,
            "recent_datasets_per_minute": rate,
            "eta_seconds": eta,
            "warnings": list(warnings),
            "errors": list(errors),
            "error_count": len(errors),
            "output_root": str(root),
            "output_paths": output_paths(root),
            "detail_csv": str(root / "05_dataset_results.csv"),
            "summary_csv": str(root / "06_global_summary.csv"),
            "stdout_log": str(stdout_log),
            "started_unix": started_unix,
            "updated_unix": now,
            "heartbeat": {"updated_unix": now, "status": status},
        },
    )


def _stage_complete(stage: str, root: Path, dataset_limit: int | None) -> bool:
    if stage == "audit":
        return all((root / name).is_file() for name in ("00_contract_audit.json", "01_source_manifest.csv"))
    if stage == "predict":
        try:
            manifest = verify_prediction_checkpoint(root)
        except (OSError, ValueError):
            return False
        expected = EXPECTED_DATASETS if dataset_limit is None else dataset_limit
        return manifest.get("datasets") == expected and manifest.get("complete_population") is (dataset_limit is None)
    if stage == "risk":
        return (root / "02_train_normal_risk_audit.csv").is_file()
    if stage == "evaluate":
        required = ("05_dataset_results.csv", "06_global_summary.csv", "07_source_agreement_diagnostics.csv")
        if not all((root / name).is_file() for name in required):
            return False
        try:
            with (root / "06_global_summary.csv").open(newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))
            expected = EXPECTED_DATASETS if dataset_limit is None else dataset_limit
            return int(summary["datasets"]) == expected
        except (OSError, KeyError, StopIteration, ValueError, csv.Error):
            return False
    if stage == "report":
        return all(
            (root / name).is_file()
            for name in ("08_invariance_report.md", "09_result_report.md", "10_independent_reaggregation.json", "MANIFEST.csv")
        )
    raise ValueError(f"unknown stage: {stage}")


def first_incomplete_stage(output_root: Path, dataset_limit: int | None) -> str | None:
    root = publication_root(output_root, dataset_limit)
    for stage in STAGE_ORDER:
        if not _stage_complete(stage, root, dataset_limit):
            return stage
    return None


def run_stage(stage: str, output_root: Path, dataset_limit: int | None) -> None:
    root = publication_root(output_root, dataset_limit)
    root.mkdir(parents=True, exist_ok=True)
    actions: dict[str, Callable[[], object]] = {
        "audit": lambda: load_frozen_inputs(output_root=root),
        "predict": lambda: predict_all(output_root, dataset_limit),
        "risk": lambda: audit_train_risk(output_root=root),
        "evaluate": lambda: evaluate_predictions(root, require_full_population=dataset_limit is None),
        "report": lambda: write_reports(root, require_full_population=dataset_limit is None),
    }
    if stage not in actions:
        raise ValueError(f"unknown stage: {stage}")
    if stage == "risk" and dataset_limit is not None:
        raise ValueError("TRAIN-risk audit requires the complete 1,117-dataset population")
    actions[stage]()
    if not _stage_complete(stage, root, dataset_limit):
        raise RuntimeError(f"stage did not publish a complete artifact: {stage}")


def run_all(
    output_root: Path,
    dataset_limit: int | None,
    workers: int = FULL_RUN_WORKERS,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[str, ...]:
    del workers
    first = first_incomplete_stage(output_root, dataset_limit)
    if first is None:
        return ()
    stages = STAGE_ORDER[STAGE_ORDER.index(first):]
    for stage in stages:
        if on_stage is not None:
            on_stage(stage)
        run_stage(stage, output_root, dataset_limit)
    return stages


def completed_output_is_valid(output_root: Path, dataset_limit: int | None) -> bool:
    return first_incomplete_stage(output_root, dataset_limit) is None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the resumable Exp266 recovery-shadow pipeline")
    parser.add_argument("--stage", choices=(*STAGE_ORDER, "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=FULL_RUN_WORKERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset_limit is not None and not 0 < args.dataset_limit < EXPECTED_DATASETS:
        raise SystemExit("dataset-limit must be a strict positive subset of 1,117")
    if not 1 <= args.workers <= FULL_RUN_WORKERS:
        raise SystemExit(f"workers must be between 1 and {FULL_RUN_WORKERS}")

    base_root = args.output_root.expanduser().resolve()
    root = publication_root(base_root, args.dataset_limit)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / STATUS_FILENAME
    started = time.time()
    expected = args.dataset_limit or EXPECTED_DATASETS
    log_path = root / "full.log"
    current_stage = args.stage if args.stage != "all" else first_incomplete_stage(base_root, args.dataset_limit) or "report"
    write_status(status_path, status="queued", stage=current_stage, datasets_done=0, expected_datasets=expected,
                 current_dataset=None, workers=args.workers, output_root=root, stdout_log=log_path,
                 started_unix=started, warnings=[], errors=[])

    def stage_started(stage: str) -> None:
        nonlocal current_stage
        current_stage = stage
        write_status(status_path, status="running", stage=stage, datasets_done=0, expected_datasets=expected,
                     current_dataset=None, workers=args.workers, output_root=root, stdout_log=log_path,
                     started_unix=started, warnings=[], errors=[])

    try:
        if args.stage == "all":
            stages = run_all(base_root, args.dataset_limit, args.workers, stage_started)
            completed_stage = stages[-1] if stages else "report"
        else:
            stage_started(args.stage)
            run_stage(args.stage, base_root, args.dataset_limit)
            completed_stage = args.stage
        if args.stage == "all" and not completed_output_is_valid(base_root, args.dataset_limit):
            raise RuntimeError("pipeline did not publish complete Exp266 outputs")
        done = expected if args.stage == "all" else 0
        write_status(status_path, status="completed", stage=completed_stage, datasets_done=done,
                     expected_datasets=expected, current_dataset=None, workers=args.workers,
                     output_root=root, stdout_log=log_path, started_unix=started, warnings=[], errors=[])
        return 0
    except BaseException as exc:
        write_status(status_path, status="failed", stage=current_stage, datasets_done=0,
                     expected_datasets=expected, current_dataset=None, workers=args.workers,
                     output_root=root, stdout_log=log_path, started_unix=started, warnings=[],
                     errors=[f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)])
        if isinstance(exc, KeyboardInterrupt):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
