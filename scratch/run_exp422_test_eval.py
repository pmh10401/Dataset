#!/usr/bin/env python3
"""Run the separate prediction-first Exp422 TEST evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp422_test_eval.audit import compute_confusion_audit
from experiments.exp422_test_eval.config import FROZEN_CONFIG
from experiments.exp422_test_eval.inputs import (
    freeze_prediction_inputs,
    load_frozen_test_labels,
    sha256_file,
    sqlite_quick_check,
    verify_database_reference,
    verify_input_snapshots,
)
from experiments.exp422_test_eval.reporting import (
    MANIFEST_FILE,
    STATUS_FILE,
    artifact_hashes,
    render_manifest,
    render_payloads,
    verify_rendered_payloads,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the separate, post-hoc Exp422 TEST confusion evaluation.",
        allow_abbrev=False,
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--frozen-test-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _fresh_output(path: Path) -> Path:
    output = Path(path).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError("output-root must be a new directory")
    if not output.parent.exists() or not output.parent.is_dir():
        raise FileNotFoundError("output-root parent does not exist")
    output.mkdir()
    return output


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _emit(log: Path, event: str, **fields: object) -> None:
    body = {"event": event, "pid": os.getpid(), "timestamp_utc": _utc_now(), **fields}
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _provenance(output: Path, *, state: str, reason: str | None, test_labels_opened: bool) -> dict[str, object]:
    return {
        "attempt_root": str(output.resolve()),
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "source_experiment_id": FROZEN_CONFIG.source_experiment_id,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "state": state,
        "reason": reason,
        "database_path": None,
        "prediction_path": None,
        "prediction_manifest_path": None,
        "test_labels_opened": test_labels_opened,
        "test_informed_reselection": False,
        "operational_policy_changed": False,
    }


def _publish_blocked(output: Path, log: Path, state: str, reason: str) -> None:
    status = {
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "state": state,
        "reason": reason,
        "test_labels_opened": False,
        "git_commit": _git_commit(),
    }
    _write_json(output / STATUS_FILE, status)
    manifest = render_manifest(
        artifact_sha256={},
        provenance={
            **status,
            "attempt_root": str(output.resolve()),
            "test_informed_reselection": False,
            "operational_policy_changed": False,
        },
    )
    (output / MANIFEST_FILE).write_bytes(manifest)
    _emit(log, "completed", state=state, reason=reason, test_labels_opened=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = _fresh_output(args.output_root)
    pid_file = output / "pid.txt"
    progress = output / "progress.jsonl"
    pid_file.write_text(str(os.getpid()) + "\n", encoding="ascii")
    _emit(progress, "preflight_started", total_datasets=FROZEN_CONFIG.expected_datasets)

    prediction = Path(args.prediction).expanduser()
    manifest = Path(args.prediction_manifest).expanduser()
    labels = Path(args.frozen_test_labels).expanduser()
    if not prediction.is_file() or not manifest.is_file():
        _publish_blocked(output, progress, "blocked_no_prediction", "frozen Exp422 prediction or manifest is absent")
        return 0

    try:
        database_sha256 = verify_database_reference(args.database, config=FROZEN_CONFIG)
        database_quick_check = sqlite_quick_check(args.database, config=FROZEN_CONFIG)
        frozen_prediction = freeze_prediction_inputs(prediction, manifest, config=FROZEN_CONFIG)
    except (OSError, ValueError) as error:
        _publish_blocked(output, progress, "blocked_invalid_prediction", str(error))
        return 0
    _emit(
        progress,
        "prediction_frozen",
        prediction_sha256=frozen_prediction.prediction_sha256,
        prediction_manifest_sha256=frozen_prediction.manifest_sha256,
        prediction_coordinate_count=len(frozen_prediction.coordinates),
    )

    if not labels.is_file():
        _publish_blocked(output, progress, "blocked_no_labels", "frozen TEST label snapshot is absent")
        return 0
    _emit(progress, "labels_read_started", label_path=str(labels.resolve()))
    try:
        frozen_labels = load_frozen_test_labels(labels, frozen_prediction, config=FROZEN_CONFIG)
        result = compute_confusion_audit(frozen_prediction, frozen_labels)
        payloads = render_payloads(
            result,
            frozen_prediction,
            frozen_labels,
            database_sha256,
            _git_commit(),
            config=FROZEN_CONFIG,
        )
        verify_rendered_payloads(payloads)
        verify_input_snapshots(
            args.database,
            prediction,
            manifest,
            labels,
            database_sha256,
            frozen_prediction,
            frozen_labels,
            config=FROZEN_CONFIG,
        )
    except (OSError, ValueError) as error:
        _publish_blocked(output, progress, "blocked_invalid_labels", str(error))
        return 0

    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    hashes = artifact_hashes(payloads)
    (output / MANIFEST_FILE).write_bytes(
        render_manifest(
            artifact_sha256=hashes,
            provenance={
                "experiment_id": FROZEN_CONFIG.experiment_id,
                "source_experiment_id": FROZEN_CONFIG.source_experiment_id,
                "state": "completed",
                "git_commit": _git_commit(),
                "database_sha256": database_sha256,
                "sqlite_quick_check": database_quick_check,
                "prediction_sha256": frozen_prediction.prediction_sha256,
                "prediction_manifest_sha256": frozen_prediction.manifest_sha256,
                "test_labels_sha256": frozen_labels.labels_sha256,
                "prediction_identity_sha256": frozen_prediction.identity_sha256,
                "test_labels_opened": True,
                "test_labels_loaded_after_prediction_freeze": True,
                "test_informed_reselection": False,
                "operational_policy_changed": False,
            },
        )
    )
    _write_json(
        output / STATUS_FILE,
        {
            "experiment_id": FROZEN_CONFIG.experiment_id,
            "state": "completed",
            "reason": None,
            "test_labels_opened": True,
            "prediction_coordinate_count": result.prediction_coordinates,
            "label_row_count": result.label_rows,
        },
    )
    _emit(progress, "matrix_written", prediction_coordinate_count=result.prediction_coordinates, label_row_count=result.label_rows)
    _emit(progress, "completed", state="completed", reason=None, test_labels_opened=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
