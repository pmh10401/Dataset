"""Atomic status publication and concise Exp267-273 report helpers."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path


DATA_DIR = Path("/Users/minho/Documents/Dataset")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_status(
    experiment_id: str,
    status: str,
    phase: str,
    done: int,
    expected: int,
    current_dataset: str | None,
    *,
    errors: Sequence[str],
    timestamps: Sequence[tuple[float, int]],
    stdout_log: Path,
    started_unix: float,
    now_unix: float | None = None,
    prediction_sha256: str | None = None,
    learner_eligible: int | None = None,
    abstained: int | None = None,
) -> dict[str, object]:
    now_value = time.time() if now_unix is None else float(now_unix)
    rate: float | None = None
    if len(timestamps) >= 2:
        first_time, first_done = timestamps[0]
        last_time, last_done = timestamps[-1]
        elapsed = last_time - first_time
        if elapsed > 0 and last_done >= first_done:
            rate = (last_done - first_done) / elapsed * 60.0
    eta = (expected - done) / rate * 60.0 if rate and done < expected else (0.0 if done >= expected else None)
    return {
        "id": str(experiment_id), "experiment_id": str(experiment_id),
        "status": str(status), "phase": str(phase), "stage": str(phase),
        "done": int(done), "datasets_done": int(done),
        "rows": int(done),
        "expected": int(expected), "expected_datasets": int(expected),
        "progress_percent": (100.0 * done / expected) if expected else 0.0,
        "current_dataset": current_dataset,
        "errors": list(errors), "error_count": len(errors),
        "started_unix": float(started_unix), "updated_unix": now_value,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_value)),
        "elapsed_seconds": max(0.0, now_value - float(started_unix)),
        "recent_datasets_per_minute": rate, "eta_seconds": eta,
        "stdout_log": str(stdout_log), "prediction_sha256": prediction_sha256,
        "learner_eligible": learner_eligible, "abstained": abstained,
        "autonomous_metric_scope": "Exp253 Hard only; unchanged through Exp272",
        "review_metric_scope": "human-assisted review only",
        "retrospective": True, "prospective": False,
    }


def publish_status(
    status: Mapping[str, object],
    *,
    output_root: Path,
    data_dir: Path = DATA_DIR,
) -> tuple[Path, Path, Path]:
    """Publish all dashboard surfaces atomically; callers decide error policy."""
    paths = (
        Path(output_root) / "status.json",
        Path(data_dir) / "rank_dashboard_external_live_run.json",
        Path(data_dir) / "rank_experiments_dashboard_heartbeat.json",
    )
    for path in paths:
        _atomic_json(path, status)
    return paths


__all__ = ["build_status", "publish_status"]
