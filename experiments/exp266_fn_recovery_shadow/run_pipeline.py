from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .config import EXPECTED_DATASETS, EXPECTED_LANE_TOTALS, EXPERIMENT_ID, FROZEN_DATABASE_EXCLUSIONS, FULL_RUN_WORKERS, OUTPUT_ROOT, DB_PATH
from .evaluation import evaluate_predictions
from .independent_reaggregate import verify_summary
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
    recent_datasets_per_minute: float | None = None,
) -> None:
    now = time.time()
    elapsed = max(0.0, now - started_unix)
    average_rate = datasets_done / elapsed * 60.0 if elapsed > 0 and datasets_done else 0.0
    recent_rate = average_rate if recent_datasets_per_minute is None else recent_datasets_per_minute
    eta = (expected_datasets - datasets_done) / recent_rate * 60.0 if recent_rate > 0 else None
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
            "recent_datasets_per_minute": recent_rate,
            "average_datasets_per_minute": average_rate,
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


class StatusHeartbeat:
    def __init__(self, status_path: Path, *, output_root: Path, stdout_log: Path,
                 workers: int, expected_datasets: int, interval_seconds: float = 5.0) -> None:
        self.status_path = Path(status_path)
        self.output_root = Path(output_root)
        self.stdout_log = Path(stdout_log)
        self.workers = workers
        self.expected_datasets = expected_datasets
        self.interval_seconds = interval_seconds
        self.started_unix = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stage = "queued"
        self.datasets_done = 0
        self.current_dataset: str | None = None
        self._last_progress_done = 0
        self._last_progress_unix = self.started_unix
        self._recent_rate = 0.0

    def _emit(self) -> None:
        with self._lock:
            write_status(
                self.status_path, status="running", stage=self.stage,
                datasets_done=self.datasets_done, expected_datasets=self.expected_datasets,
                current_dataset=self.current_dataset, workers=self.workers,
                output_root=self.output_root, stdout_log=self.stdout_log,
                started_unix=self.started_unix, warnings=[], errors=[],
                recent_datasets_per_minute=self._recent_rate,
            )

    def start(self, stage: str) -> None:
        with self._lock:
            self.stage = stage
            self.datasets_done = 0
            self.current_dataset = None
        self._emit()
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="exp266-status-heartbeat", daemon=True)
            self._thread.start()

    def progress(self, dataset: str, done: int, _expected: int | None = None) -> None:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_progress_unix
            if done > self._last_progress_done and elapsed > 0:
                self._recent_rate = (done - self._last_progress_done) / elapsed * 60.0
            self._last_progress_done = done
            self._last_progress_unix = now
            self.current_dataset = dataset
            self.datasets_done = done
        self._emit()

    def complete_stage(self) -> None:
        with self._lock:
            self.datasets_done = self.expected_datasets
            self.current_dataset = None
        self._emit()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._emit()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))


def _expected_dataset_names() -> set[str]:
    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as connection:
            names = {str(row[0]) for row in connection.execute("SELECT name FROM datasets")}
    except sqlite3.Error:
        return set()
    return names - set(FROZEN_DATABASE_EXCLUSIONS)


def _csv_dataset_names(path: Path) -> set[str] | None:
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    names = [row.get("dataset") for row in rows]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(rows):
        return None
    return set(names)


def _full_population_matches(path: Path) -> bool:
    expected = _expected_dataset_names()
    names = _csv_dataset_names(path)
    return bool(expected) and names == expected and len(names) == EXPECTED_DATASETS


def _stage_complete(stage: str, root: Path, dataset_limit: int | None) -> bool:
    if dataset_limit is not None:
        return False
    if stage == "audit":
        try:
            audit = json.loads((root / "00_contract_audit.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            audit.get("experiment_id") == EXPERIMENT_ID
            and audit.get("datasets") == EXPECTED_DATASETS
            and audit.get("lane_totals") == EXPECTED_LANE_TOTALS
            and _full_population_matches(root / "01_source_manifest.csv")
        )
    if stage == "predict":
        try:
            manifest = verify_prediction_checkpoint(root)
        except (OSError, ValueError):
            return False
        try:
            rows = [json.loads(line) for line in (root / "03_frozen_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
        except (OSError, json.JSONDecodeError):
            return False
        names = {row.get("dataset") for row in rows if isinstance(row, dict)}
        return (
            manifest.get("datasets") == EXPECTED_DATASETS
            and manifest.get("complete_population") is True
            and len(rows) == EXPECTED_DATASETS
            and names == _expected_dataset_names()
        )
    if stage == "risk":
        return _full_population_matches(root / "02_train_normal_risk_audit.csv")
    if stage == "evaluate":
        required = ("05_dataset_results.csv", "06_global_summary.csv", "07_source_agreement_diagnostics.csv")
        if not all((root / name).is_file() for name in required):
            return False
        try:
            with (root / "06_global_summary.csv").open(newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))
            return (
                int(summary["datasets"]) == EXPECTED_DATASETS
                and _full_population_matches(root / "05_dataset_results.csv")
                and _full_population_matches(root / "07_source_agreement_diagnostics.csv")
                and bool(verify_summary(root / "05_dataset_results.csv", root / "06_global_summary.csv")["passed"])
            )
        except (OSError, KeyError, StopIteration, ValueError, csv.Error):
            return False
    if stage == "report":
        try:
            reaggregation = json.loads((root / "10_independent_reaggregation.json").read_text(encoding="utf-8"))
            with (root / "MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
                manifest_rows = list(csv.DictReader(handle))
        except (OSError, json.JSONDecodeError, csv.Error):
            return False
        required = {
            "00_contract_audit.json", "01_source_manifest.csv", "02_train_normal_risk_audit.csv",
            "03_frozen_predictions.jsonl", "04_prediction_manifest.json", "05_dataset_results.csv",
            "06_global_summary.csv", "07_source_agreement_diagnostics.csv", "08_invariance_report.md",
            "09_result_report.md", "10_independent_reaggregation.json",
        }
        manifest_files = {row.get("file") for row in manifest_rows}
        hashes_match = all(
            isinstance(row.get("sha256"), str)
            and (root / str(row["file"])).is_file()
            and hashlib.sha256((root / str(row["file"])).read_bytes()).hexdigest() == row["sha256"]
            for row in manifest_rows
        )
        return (
            required <= manifest_files
            and hashes_match
            and reaggregation.get("passed") is True
            and reaggregation.get("missing_datasets") == 0
            and reaggregation.get("index_errors") == 0
            and reaggregation.get("lane_overlaps") == 0
            and reaggregation.get("hard_parity_mismatches") == 0
            and reaggregation.get("summary_verification", {}).get("passed") is True
        )
    raise ValueError(f"unknown stage: {stage}")


def first_incomplete_stage(output_root: Path, dataset_limit: int | None) -> str | None:
    root = publication_root(output_root, dataset_limit)
    for stage in STAGE_ORDER:
        if not _stage_complete(stage, root, dataset_limit):
            return stage
    return None


def run_stage(stage: str, output_root: Path, dataset_limit: int | None, workers: int = FULL_RUN_WORKERS,
              progress_callback: Callable[[str, int, int], None] | None = None) -> None:
    root = publication_root(output_root, dataset_limit)
    root.mkdir(parents=True, exist_ok=True)
    actions: dict[str, Callable[[], object]] = {
        "audit": lambda: load_frozen_inputs(output_root=root),
        "predict": lambda: predict_all(output_root, dataset_limit, workers=workers, progress_callback=progress_callback),
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
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[str, ...]:
    first = first_incomplete_stage(output_root, dataset_limit)
    if first is None:
        return ()
    stages = STAGE_ORDER[STAGE_ORDER.index(first):]
    for stage in stages:
        if on_stage is not None:
            on_stage(stage)
        run_stage(stage, output_root, dataset_limit, workers, progress_callback)
    return stages


def completed_output_is_valid(output_root: Path, dataset_limit: int | None) -> bool:
    if first_incomplete_stage(output_root, dataset_limit) is not None:
        return False
    status_path = publication_root(output_root, dataset_limit) / STATUS_FILENAME
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return True
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("error_count", 0) == 0 and not status.get("errors")


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
    heartbeat = StatusHeartbeat(
        status_path, output_root=root, stdout_log=log_path, workers=args.workers,
        expected_datasets=expected,
    )

    def stage_started(stage: str) -> None:
        nonlocal current_stage
        current_stage = stage
        heartbeat.start(stage)

    try:
        if args.stage == "all":
            stages = run_all(base_root, args.dataset_limit, args.workers, stage_started, heartbeat.progress)
            completed_stage = stages[-1] if stages else "report"
        else:
            stage_started(args.stage)
            run_stage(args.stage, base_root, args.dataset_limit, args.workers, heartbeat.progress)
            completed_stage = args.stage
        if args.stage == "all" and not completed_output_is_valid(base_root, args.dataset_limit):
            raise RuntimeError("pipeline did not publish complete Exp266 outputs")
        heartbeat.complete_stage()
        heartbeat.close()
        done = expected if args.stage == "all" else heartbeat.datasets_done
        write_status(status_path, status="completed", stage=completed_stage, datasets_done=done,
                     expected_datasets=expected, current_dataset=None, workers=args.workers,
                     output_root=root, stdout_log=log_path, started_unix=started, warnings=[], errors=[])
        return 0
    except BaseException as exc:
        heartbeat.close()
        write_status(status_path, status="failed", stage=current_stage, datasets_done=0,
                     expected_datasets=expected, current_dataset=None, workers=args.workers,
                     output_root=root, stdout_log=log_path, started_unix=started, warnings=[],
                     errors=[f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)])
        if isinstance(exc, KeyboardInterrupt):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
