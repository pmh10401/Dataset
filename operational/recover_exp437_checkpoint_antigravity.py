#!/usr/bin/env python3
"""One-shot exact-source recovery launcher for the frozen Exp437 incident."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping


EXP437_SOURCE_WORKTREE_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery"
)
REVIEWED_EXP437_COMMIT_ANTIGRAVITY = (
    "cc0f507f290dc7f613763d3653498f8f54cad88a"
)
EXP437_RUN_IDENTITY_ANTIGRAVITY = (
    "exp437-metadata-roundtrip-antigravity-20260807"
)


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


def _sha256_file_antigravity(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_state_antigravity() -> tuple[str, bool]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=EXP437_SOURCE_WORKTREE_ANTIGRAVITY,
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=EXP437_SOURCE_WORKTREE_ANTIGRAVITY,
        text=True,
    )
    return head, status == ""


def _bootstrap_exp437_antigravity() -> None:
    head, clean = _source_state_antigravity()
    if head != REVIEWED_EXP437_COMMIT_ANTIGRAVITY or not clean:
        raise ValueError("Exp437 recovery source is not the exact clean commit")
    source = str(EXP437_SOURCE_WORKTREE_ANTIGRAVITY)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)


def _validate_incident_antigravity(
    evidence: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "source_head": REVIEWED_EXP437_COMMIT_ANTIGRAVITY,
        "source_clean": True,
        "run_identity": EXP437_RUN_IDENTITY_ANTIGRAVITY,
        "state": "running",
        "recorded_pid": 81739,
        "pid_active": True,
        "pid_os_live": False,
        "backend": "cpu",
        "error": 0,
        "progress_completed": 938,
        "event_completed": 938,
        "checkpoint_completed": 938,
        "physical_record_files": 939,
        "total": 1117,
        "reconciliation_reasons": ["intent_with_record"],
        "label_access_count": 0,
        "test_values_opened": 0,
        "test_labels_opened": 0,
        "publication_exists": False,
        "evaluation_exists": False,
    }
    if not isinstance(evidence, Mapping) or any(
        evidence.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("Exp437 recovery incident no longer matches frozen evidence")
    return dict(evidence)


def preflight_exp437_recovery_antigravity() -> dict[str, object]:
    """Validate the exact stale-PID crash window without changing a byte."""
    _bootstrap_exp437_antigravity()
    from experiments.exp403_408.artifact_io import canonical_json_bytes
    from experiments.exp436_missing_safe_strict_consensus_antigravity.checkpoint_antigravity import (
        Exp436CheckpointStore,
    )
    from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import (
        EXP437_PROFILE,
    )
    from experiments.exp436_missing_safe_strict_consensus_antigravity.progress_antigravity import (
        ProgressWriter,
    )

    head, clean = _source_state_antigravity()
    readiness_raw = EXP437_PROFILE.readiness.read_bytes()
    readiness = json.loads(readiness_raw)
    if canonical_json_bytes(readiness) != readiness_raw:
        raise ValueError("Exp437 readiness is not canonical")
    store = Exp436CheckpointStore.open_classified_read_only(
        EXP437_PROFILE.checkpoint, profile=EXP437_PROFILE
    )
    inspection = store.inspect()
    record_digests = store.completed_digests()
    progress_view = ProgressWriter.preflight_existing(
        EXP437_PROFILE.output,
        run_identity=store.run_identity,
        checkpoint=EXP437_PROFILE.checkpoint,
        backend="cpu",
        total_datasets=len(store.datasets),
        context_fingerprint=store.context_fingerprint,
        checkpoint_inspection={
            "completed": inspection.completed,
            "datasets": tuple(record_digests),
            "record_digests": record_digests,
        },
        process_is_live=_process_is_live_antigravity,
    )
    progress = progress_view["progress"]
    pid = progress_view["pid"]
    records = EXP437_PROFILE.checkpoint / "records_antigravity"
    physical_records = len(tuple(records.glob("*.json")))
    publication = EXP437_PROFILE.output / "publication_antigravity"
    evidence: dict[str, object] = {
        "source_head": head,
        "source_clean": clean,
        "run_identity": store.run_identity,
        "state": progress["state"],
        "recorded_pid": pid["pid"],
        "pid_active": pid["pid_active"],
        "pid_os_live": _process_is_live_antigravity(int(pid["pid"])),
        "backend": progress["backend"],
        "error": progress["error"],
        "progress_completed": progress["completed_datasets"],
        "event_completed": progress_view["event_completed"],
        "checkpoint_completed": progress_view["checkpoint_completed"],
        "physical_record_files": physical_records,
        "total": progress["total_datasets"],
        "reconciliation_reasons": list(inspection.reconciliation_reasons),
        "label_access_count": readiness.get("label_access_count"),
        "test_values_opened": readiness.get("test_values_opened"),
        "test_labels_opened": readiness.get("test_labels_opened"),
        "publication_exists": publication.exists(),
        "evaluation_exists": EXP437_PROFILE.evaluation.exists(),
        "readiness_sha256": _sha256_file_antigravity(EXP437_PROFILE.readiness),
        "progress_sha256": _sha256_file_antigravity(
            EXP437_PROFILE.output / "progress_antigravity.json"
        ),
        "pid_sha256": _sha256_file_antigravity(
            EXP437_PROFILE.output / "run_antigravity.pid"
        ),
        "events_sha256": _sha256_file_antigravity(
            EXP437_PROFILE.output / "events_antigravity.jsonl"
        ),
        "record_set_sha256": _sha256_file_antigravity(
            EXP437_PROFILE.checkpoint / "record_set_antigravity.json"
        ),
    }
    return _validate_incident_antigravity(evidence)


def recover_exp437_checkpoint_antigravity() -> dict[str, object]:
    """Resume through the frozen public runtime and publish only on completion."""
    preflight_exp437_recovery_antigravity()
    from experiments.exp436_missing_safe_strict_consensus_antigravity.cli_antigravity import (
        _input_factory,
        _publish_after_complete,
    )
    from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import (
        EXP437_PROFILE,
    )
    from experiments.exp436_missing_safe_strict_consensus_antigravity.runtime_antigravity import (
        resume_exp436,
    )

    summary = resume_exp436(
        readiness_path=EXP437_PROFILE.readiness,
        run_root=EXP437_PROFILE.output,
        checkpoint_root=EXP437_PROFILE.checkpoint,
        reviewed_commit=REVIEWED_EXP437_COMMIT_ANTIGRAVITY,
        input_session_factory=_input_factory(
            REVIEWED_EXP437_COMMIT_ANTIGRAVITY, EXP437_PROFILE
        ),
        profile=EXP437_PROFILE,
    )
    if summary.status == "checkpoint_complete":
        return dict(_publish_after_complete(summary, EXP437_PROFILE))
    return dict(summary.to_payload())


def main_antigravity(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen Exp437 checkpoint recovery")
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args(argv)
    result = (
        preflight_exp437_recovery_antigravity()
        if arguments.preflight
        else recover_exp437_checkpoint_antigravity()
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
