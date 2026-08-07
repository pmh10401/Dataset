#!/usr/bin/env python3
"""Collect review-only TRAIN score evidence after an Exp422 core parity failure."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import sys
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp422_mps_numeric_delta.capability import detect_capability
from experiments.exp422_mps_numeric_delta.config import FROZEN_CONFIG
from experiments.exp422_mps_numeric_delta.core_parity import run_core_parity
from experiments.exp422_mps_numeric_delta.manifest import (
    _core_payload,
    _score_payload,
    build_manifest,
    source_manifest_git_commit,
    verify_source_manifest,
    write_manifest,
)
from experiments.exp422_mps_numeric_delta.score_parity import (
    run_score_parity,
    select_train_sample,
)
from scratch.run_exp422_interrupted419_diagnostic import (
    ProgressLogger,
    _Heartbeat,
    _provenance,
    _stage_fields,
    _utc_now,
    require_exp419_interrupted,
)
from scratch.run_exp422_mps_numeric_delta import (
    current_git_commit,
    preflight_database,
    validate_output_root,
)


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
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


def _exploratory_payload(
    core: object,
    score: object,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    core_stage = getattr(core, "stage")
    score_stage = getattr(score, "stage")
    reasons: list[str] = []
    for reason in (
        core_stage.reason,
        score_stage.reason,
        *(report.fallback_reason for report in getattr(score, "reports")),
    ):
        if reason is not None and reason not in reasons:
            reasons.append(reason)
    return {
        "artifact_kind": "exp422_ir_exploratory_score_evidence",
        "schema_version": 1,
        "experiment_id": "Exp422-IR-E",
        "state": "review_only",
        "core_gate_overridden": True,
        "core_stage": _core_payload(core),
        "score_stage": _score_payload(score),
        "fallback_reasons": reasons,
        "provenance": dict(provenance),
        "recorded_at_utc": _utc_now(),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one review-only TRAIN score pass after Exp422 core failure."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exp419-pidfile", type=Path, required=True)
    parser.add_argument("--exp419-exitfile", type=Path, required=True)
    args = parser.parse_args()

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
            experiment="Exp422-IR-E",
            predecessor_policy="exp419_interrupted_exit_143_only",
            predecessor_exit_code=predecessor_exit_code,
            review_only=True,
            core_gate_overridden=True,
        )
        heartbeat.start()
        heartbeat.set_state("preflight", total=1)
        logger.emit("preflight_started")
        capability = detect_capability()
        quick_check = preflight_database(args.database)
        source_digest = verify_source_manifest(REPO_ROOT)
        source_git_commit = source_manifest_git_commit(REPO_ROOT)
        git_commit = current_git_commit(REPO_ROOT)
        _write_json_exclusive(
            output_root / "execution_context.json",
            {
                "experiment": "Exp422-IR-E",
                "git_commit": git_commit,
                "pid": os.getpid(),
                "predecessor_exp419_exit_code": predecessor_exit_code,
                "predecessor_policy": "exp419_interrupted_exit_143_only",
                "core_gate_overridden": True,
                "review_only": True,
                "progress_log": str((output_root / "progress.jsonl").resolve()),
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

        heartbeat.set_state("core", total=60)
        logger.emit("core_started", total_rows=60)
        core = run_core_parity(FROZEN_CONFIG, capability)
        heartbeat.set_state("core_finished", completed=len(core.reports), total=60)
        logger.emit(
            "core_finished",
            completed_rows=len(core.reports),
            total_rows=60,
            **_stage_fields(core.stage),
        )

        logger.emit(
            "core_failure_ignored_for_review_only_score",
            reason=core.stage.reason,
            operational_adoption=False,
        )
        heartbeat.set_state("train_sample", total=1)
        logger.emit("train_sample_started")
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
            "score_started_after_core_failure",
            batch_sizes=list(FROZEN_CONFIG.candidate_batch_sizes),
            total_batches=len(FROZEN_CONFIG.candidate_batch_sizes),
            review_only=True,
        )
        score = run_score_parity(sample, capability, FROZEN_CONFIG)
        heartbeat.set_state(
            "score_finished",
            completed=len(score.reports),
            total=len(FROZEN_CONFIG.candidate_batch_sizes),
        )
        logger.emit(
            "score_finished",
            completed_batches=len(score.reports),
            total_batches=len(FROZEN_CONFIG.candidate_batch_sizes),
            **_stage_fields(score.stage),
        )

        provenance = _provenance(
            output_root=output_root,
            repeats=1,
            git_commit=git_commit,
            source_git_commit=source_git_commit,
            source_manifest_sha256=source_digest,
            sqlite_quick_check=quick_check,
            capability=capability,
        )
        core_manifest = build_manifest(core, None, provenance)
        core_manifest_path = write_manifest(output_root / "core_manifest.json", core_manifest)
        exploratory_provenance = {
            **provenance,
            "core_gate_overridden": True,
            "review_only": True,
        }
        exploratory_path = output_root / "exploratory_manifest.json"
        _write_json_exclusive(
            exploratory_path,
            _exploratory_payload(core, score, exploratory_provenance),
        )
        logger.emit(
            "exploratory_manifest_written",
            core_manifest=str(core_manifest_path),
            exploratory_manifest=str(exploratory_path),
            state="review_only",
        )
        print(exploratory_path, flush=True)
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
