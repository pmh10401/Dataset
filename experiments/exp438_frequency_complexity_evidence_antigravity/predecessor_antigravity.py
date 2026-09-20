"""Strict read-only Exp437 execution evidence for Exp438 readiness."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Callable, Mapping

from experiments.exp436_missing_safe_strict_consensus_antigravity.checkpoint_antigravity import (
    Exp436CheckpointStore,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import (
    EXP437_PROFILE,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.progress_antigravity import (
    inspect_progress,
)


CONCURRENT_HEARTBEAT_MAX_AGE_SECONDS_ANTIGRAVITY = 300.0
EXP437_CONCURRENCY_AUTHORIZED_ANTIGRAVITY = True
_EXPECTED_IDENTITY_ANTIGRAVITY = "exp437-metadata-roundtrip-antigravity-20260807"
_EXPECTED_TOTAL_ANTIGRAVITY = 1117


def _process_is_live_antigravity(pid: int) -> bool:
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _utc_now_antigravity() -> datetime:
    return datetime.now(timezone.utc)


def _heartbeat_age_antigravity(value: object, now: datetime) -> float:
    if not isinstance(value, str):
        raise ValueError("Exp437 heartbeat is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Exp437 heartbeat is invalid") from error
    if parsed.tzinfo is None or now.tzinfo is None:
        raise ValueError("Exp437 heartbeat must be timezone-aware")
    age = (now.astimezone(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
    if not math.isfinite(age) or age < 0.0:
        raise ValueError("Exp437 heartbeat age is invalid")
    return float(age)


def read_exp437_snapshot_antigravity(
    run_parent: Path = EXP437_PROFILE.run_parent,
    *,
    now_utc: Callable[[], datetime] = _utc_now_antigravity,
    process_is_live: Callable[[int], bool] = _process_is_live_antigravity,
) -> dict[str, object]:
    """Validate checkpoint/runtime bytes and return decision-only evidence."""
    parent = Path(run_parent)
    if parent.resolve() != EXP437_PROFILE.run_parent.resolve():
        raise ValueError("Exp437 predecessor root is invalid")
    if not callable(now_utc) or not callable(process_is_live):
        raise ValueError("Exp437 predecessor dependency is invalid")
    output = parent / "output_antigravity"
    checkpoint = parent / "checkpoint_antigravity"
    store = Exp436CheckpointStore.open_classified_read_only(
        checkpoint, profile=EXP437_PROFILE
    )
    if store.run_identity != _EXPECTED_IDENTITY_ANTIGRAVITY:
        raise ValueError("Exp437 predecessor identity is invalid")
    inspection = store.inspect()
    record_digests = store.completed_digests()
    progress_view = inspect_progress(
        output,
        run_identity=store.run_identity,
        checkpoint=checkpoint,
        context_fingerprint=store.context_fingerprint,
        checkpoint_inspection={
            "completed": inspection.completed,
            "datasets": tuple(record_digests),
            "record_digests": record_digests,
        },
        process_is_live=None,
    )
    progress = dict(progress_view["progress"])
    pid = dict(progress_view["pid"])
    now = now_utc()
    if not isinstance(now, datetime):
        raise ValueError("Exp437 predecessor clock is invalid")
    recorded_pid = pid.get("pid")
    pid_active = pid.get("pid_active")
    if type(recorded_pid) is not int or type(pid_active) is not bool:
        raise ValueError("Exp437 predecessor PID is invalid")
    snapshot: dict[str, object] = {
        "run_identity": store.run_identity,
        "run_parent": str(parent.resolve()),
        "state": progress.get("state"),
        "pid": recorded_pid,
        "pid_active": pid_active,
        "pid_os_live": bool(process_is_live(recorded_pid)) if pid_active else False,
        "backend": progress.get("backend"),
        "completed": progress.get("completed_datasets"),
        "checkpoint_completed": len(record_digests),
        "total": progress.get("total_datasets"),
        "error": progress.get("error"),
        "heartbeat": progress.get("heartbeat"),
        "heartbeat_age_seconds": _heartbeat_age_antigravity(
            progress.get("heartbeat"), now
        ),
        "publication_exists": (output / "publication_antigravity").exists(),
        "evaluation_exists": (parent / "evaluation_antigravity").exists(),
        "concurrency_authorized": EXP437_CONCURRENCY_AUTHORIZED_ANTIGRAVITY,
    }
    return snapshot


def _validate_common_antigravity(snapshot: Mapping[str, object]) -> None:
    if not isinstance(snapshot, Mapping):
        raise ValueError("Exp437 execution snapshot is invalid")
    expected_parent = str(EXP437_PROFILE.run_parent.resolve())
    if (
        snapshot.get("run_identity") != _EXPECTED_IDENTITY_ANTIGRAVITY
        or snapshot.get("run_parent") != expected_parent
        or snapshot.get("backend") != "cpu"
        or snapshot.get("total") != _EXPECTED_TOTAL_ANTIGRAVITY
        or type(snapshot.get("completed")) is not int
        or type(snapshot.get("checkpoint_completed")) is not int
        or type(snapshot.get("error")) is not int
        or type(snapshot.get("pid")) is not int
        or int(snapshot["pid"]) < 1
        or type(snapshot.get("pid_active")) is not bool
        or type(snapshot.get("pid_os_live")) is not bool
        or type(snapshot.get("publication_exists")) is not bool
        or type(snapshot.get("evaluation_exists")) is not bool
        or snapshot.get("error") != 0
    ):
        raise ValueError("Exp437 execution binding is invalid")
    completed = int(snapshot["completed"])
    checkpoint_completed = int(snapshot["checkpoint_completed"])
    if not (0 <= completed <= checkpoint_completed <= _EXPECTED_TOTAL_ANTIGRAVITY):
        raise ValueError("Exp437 execution counters are invalid")


def classify_exp437_snapshot_antigravity(
    snapshot: Mapping[str, object],
) -> str:
    """Return the only readiness mode allowed by a validated snapshot."""
    _validate_common_antigravity(snapshot)
    state = snapshot.get("state")
    if state in {"complete", "checkpoint_complete"}:
        if (
            snapshot.get("pid_active") is not False
            or snapshot.get("completed") != _EXPECTED_TOTAL_ANTIGRAVITY
            or snapshot.get("checkpoint_completed") != _EXPECTED_TOTAL_ANTIGRAVITY
            or snapshot.get("evaluation_exists") is not False
        ):
            raise ValueError("Exp437 terminal execution is invalid")
        return "terminal"
    if state != "running":
        raise ValueError("Exp437 execution state is invalid")
    age = snapshot.get("heartbeat_age_seconds")
    if (
        snapshot.get("pid_active") is not True
        or snapshot.get("pid_os_live") is not True
        or type(age) not in {int, float}
        or not math.isfinite(float(age))
        or not 0.0 <= float(age) <= CONCURRENT_HEARTBEAT_MAX_AGE_SECONDS_ANTIGRAVITY
        or snapshot.get("completed") == _EXPECTED_TOTAL_ANTIGRAVITY
        or snapshot.get("publication_exists") is not False
        or snapshot.get("evaluation_exists") is not False
        or snapshot.get("concurrency_authorized") is not True
    ):
        raise ValueError("Exp437 concurrent execution is invalid")
    return "concurrent"


def revalidate_exp437_snapshot_antigravity(
    bound: Mapping[str, object], current: Mapping[str, object]
) -> str:
    """Allow only monotonic progress or a valid terminal transition."""
    bound_mode = classify_exp437_snapshot_antigravity(bound)
    current_mode = classify_exp437_snapshot_antigravity(current)
    for name in ("run_identity", "run_parent", "backend", "total"):
        if current.get(name) != bound.get(name):
            raise ValueError("Exp437 execution binding changed")
    if (
        int(current["completed"]) < int(bound["completed"])
        or int(current["checkpoint_completed"]) < int(bound["checkpoint_completed"])
    ):
        raise ValueError("Exp437 execution counters regressed")
    if bound_mode == "terminal" and current_mode != "terminal":
        raise ValueError("Exp437 terminal execution regressed")
    return current_mode


__all__ = [
    "CONCURRENT_HEARTBEAT_MAX_AGE_SECONDS_ANTIGRAVITY",
    "EXP437_CONCURRENCY_AUTHORIZED_ANTIGRAVITY",
    "classify_exp437_snapshot_antigravity",
    "read_exp437_snapshot_antigravity",
    "revalidate_exp437_snapshot_antigravity",
]
