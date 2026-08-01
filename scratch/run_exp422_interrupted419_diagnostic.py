#!/usr/bin/env python3
"""Run the separately preregistered Exp422 diagnostic after an interrupted Exp419."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import sys
import threading
import time
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp422_mps_numeric_delta.capability import detect_capability
from experiments.exp422_mps_numeric_delta.config import FROZEN_CONFIG
from experiments.exp422_mps_numeric_delta.core_parity import run_core_parity
from experiments.exp422_mps_numeric_delta.manifest import (
    build_manifest,
    source_manifest_git_commit,
    verify_source_manifest,
    write_manifest,
)
from experiments.exp422_mps_numeric_delta.score_parity import (
    run_score_parity,
    select_train_sample,
)
from scratch.run_exp422_mps_numeric_delta import (
    _pid_is_alive,
    _provenance,
    current_git_commit,
    preflight_database,
    validate_output_root,
)


_INTERRUPTED_EXIT_CODE = 143
_HEARTBEAT_SECONDS = 60.0


def _read_control_file(path: Path, description: str) -> str:
    control = Path(path)
    if control.is_symlink() or not control.is_file():
        raise ValueError(f"Exp419 {description} must be a regular file")
    try:
        raw = control.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Exp419 {description} is unreadable") from exc
    if len(raw) > 128 or not text.strip() or text.strip() != text.strip(" \t\r\n"):
        raise ValueError(f"Exp419 {description} is malformed")
    return text.strip()


def require_exp419_interrupted(pidfile: Path, exitfile: Path) -> int:
    """Require a dead official PID and the explicitly preregistered exit 143."""
    pid_text = _read_control_file(Path(pidfile), "PID file")
    try:
        pid = int(pid_text, 10)
    except ValueError as exc:
        raise ValueError("Exp419 PID file is malformed") from exc
    if pid < 1:
        raise ValueError("Exp419 PID file is malformed")
    if _pid_is_alive(pid):
        raise ValueError("Exp419 is still running")

    exit_text = _read_control_file(Path(exitfile), "exit file")
    try:
        exit_code = int(exit_text, 10)
    except ValueError as exc:
        raise ValueError("Exp419 exit file is malformed") from exc
    if exit_code != _INTERRUPTED_EXIT_CODE:
        raise ValueError("Exp419 interruption policy requires exit code 143")
    return exit_code


def _bounded_repeats(value: str) -> int:
    try:
        repeats = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("repeats must be an integer from 1 through 7") from exc
    if not 1 <= repeats <= 7:
        raise argparse.ArgumentTypeError("repeats must be from 1 through 7")
    return repeats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the separately preregistered Exp422-IR TRAIN-only diagnostic."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exp419-pidfile", type=Path, required=True)
    parser.add_argument("--exp419-exitfile", type=Path, required=True)
    parser.add_argument("--repeats", type=_bounded_repeats, required=True)
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProgressLogger:
    """Durable JSONL progress evidence for live status checks."""

    def __init__(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.path = target
        self._handle = target.open("x", encoding="utf-8")
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: object) -> None:
        if not isinstance(event, str) or not event.strip():
            raise ValueError("progress event must be a nonempty string")
        payload: dict[str, object] = {
            "event": event,
            "pid": os.getpid(),
            "timestamp_utc": _utc_now(),
        }
        payload.update(fields)
        line = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                os.fsync(self._handle.fileno())
                self._handle.close()


class _Heartbeat:
    def __init__(self, logger: ProgressLogger) -> None:
        self._logger = logger
        self._started = time.monotonic()
        self._phase = "starting"
        self._completed = 0
        self._total = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="exp422-ir-heartbeat", daemon=True)

    def set_state(self, phase: str, *, completed: int = 0, total: int = 0) -> None:
        with self._lock:
            self._phase = phase
            self._completed = completed
            self._total = total

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(_HEARTBEAT_SECONDS):
            with self._lock:
                phase, completed, total = self._phase, self._completed, self._total
            self._logger.emit(
                "heartbeat",
                phase=phase,
                completed=completed,
                total=total,
                elapsed_seconds=round(time.monotonic() - self._started, 3),
            )


def _write_context(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    data = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n"
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_fields(stage: object) -> dict[str, object]:
    return {
        "state": getattr(stage, "state", "unknown"),
        "reason": getattr(stage, "reason", None),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    predecessor_exit_code = require_exp419_interrupted(
        args.exp419_pidfile, args.exp419_exitfile
    )
    output_root = validate_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(output_root / "progress.jsonl")
    heartbeat = _Heartbeat(logger)
    try:
        logger.emit(
            "run_started",
            experiment="Exp422-IR",
            predecessor_policy="exp419_interrupted_exit_143_only",
            predecessor_exit_code=predecessor_exit_code,
            repeats=args.repeats,
        )
        heartbeat.start()
        heartbeat.set_state("preflight", total=1)
        logger.emit("preflight_started")
        capability = detect_capability()
        quick_check = preflight_database(args.database)
        source_digest = verify_source_manifest(REPO_ROOT)
        source_git_commit = source_manifest_git_commit(REPO_ROOT)
        git_commit = current_git_commit(REPO_ROOT)
        _write_context(
            output_root / "execution_context.json",
            {
                "experiment": "Exp422-IR",
                "git_commit": git_commit,
                "predecessor_exp419_exit_code": predecessor_exit_code,
                "predecessor_policy": "exp419_interrupted_exit_143_only",
                "progress_log": str((output_root / "progress.jsonl").resolve()),
                "run_log": "stdout_stderr_redirected_by_launcher",
                "source_manifest_sha256": source_digest,
                "source_git_commit": source_git_commit,
                "started_at_utc": _utc_now(),
            },
        )
        heartbeat.set_state("ready", completed=1, total=1)
        logger.emit(
            "preflight_finished",
            state="passed",
            reason=None,
            database_quick_check=quick_check,
            mps_available=capability.mps_available,
            mps_built=capability.mps_built,
            torch_version=capability.torch_version,
        )

        core = None
        score = None
        sample = None
        for repeat in range(1, args.repeats + 1):
            logger.emit("repeat_started", repeat=repeat, repeats=args.repeats)
            heartbeat.set_state("core", total=60)
            logger.emit("core_started", repeat=repeat, total_rows=60)
            core = run_core_parity(FROZEN_CONFIG, capability)
            heartbeat.set_state("core_finished", completed=len(core.reports), total=60)
            logger.emit(
                "core_finished",
                repeat=repeat,
                completed_rows=len(core.reports),
                total_rows=60,
                **_stage_fields(core.stage),
            )
            if core.stage.state != "passed":
                score = None
                break
            if sample is None:
                heartbeat.set_state("train_sample", total=1)
                logger.emit("train_sample_started", repeat=repeat)
                sample = select_train_sample(args.database, FROZEN_CONFIG)
                heartbeat.set_state("train_sample_finished", completed=1, total=1)
                logger.emit(
                    "train_sample_finished",
                    dataset=sample.dataset,
                    length=sample.length,
                    train_count=sample.train_count,
                    bounded_series_count=sample.bounded_series_count,
                )
            heartbeat.set_state("score", total=len(FROZEN_CONFIG.candidate_batch_sizes))
            logger.emit(
                "score_started",
                repeat=repeat,
                batch_sizes=list(FROZEN_CONFIG.candidate_batch_sizes),
                total_batches=len(FROZEN_CONFIG.candidate_batch_sizes),
            )
            score = run_score_parity(sample, capability, FROZEN_CONFIG)
            heartbeat.set_state(
                "score_finished",
                completed=len(score.reports),
                total=len(FROZEN_CONFIG.candidate_batch_sizes),
            )
            logger.emit(
                "score_finished",
                repeat=repeat,
                completed_batches=len(score.reports),
                total_batches=len(FROZEN_CONFIG.candidate_batch_sizes),
                **_stage_fields(score.stage),
            )
            if score.stage.state != "passed":
                break

        if core is None:
            raise RuntimeError("at least one bounded repeat is required")
        manifest = build_manifest(
            core,
            score,
            _provenance(
                output_root=output_root,
                repeats=args.repeats,
                git_commit=git_commit,
                source_git_commit=source_git_commit,
                source_manifest_sha256=source_digest,
                sqlite_quick_check=quick_check,
                capability=capability,
            ),
        )
        path = write_manifest(output_root / "manifest.json", manifest)
        logger.emit("manifest_written", path=str(path), state=manifest.state)
        print(path, flush=True)
        return 0
    except Exception as exc:
        try:
            logger.emit("run_failed", error_type=type(exc).__name__, error=str(exc))
        finally:
            raise
    finally:
        heartbeat.stop()
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
