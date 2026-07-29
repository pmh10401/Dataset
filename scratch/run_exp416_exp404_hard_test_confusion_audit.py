"""One-shot clean-commit runner for the frozen Exp416 TEST audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.exp416_exp404_hard_test_confusion_audit.audit import (
    compute_confusion_audit,
)
from experiments.exp416_exp404_hard_test_confusion_audit.inputs import (
    freeze_prediction_inputs,
    load_frozen_test_labels,
    verify_database_reference,
    verify_input_snapshots,
)
from experiments.exp416_exp404_hard_test_confusion_audit.publication import (
    publish_payloads,
)
from experiments.exp416_exp404_hard_test_confusion_audit.reporting import (
    render_payloads,
    verify_rendered_payloads,
)


_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the one-shot Exp416 Exp404 Hard TEST confusion audit.",
        allow_abbrev=False,
    )
    parser.add_argument("--database-reference", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--frozen-test-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _resolve_input_file(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"Exp416 input file does not exist: {resolved}")
    return resolved


def _resolve_new_output(path: Path) -> Path:
    expanded = path.expanduser()
    try:
        parent = expanded.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError("Exp416 output parent does not exist") from error
    if not parent.is_dir():
        raise FileNotFoundError("Exp416 output parent does not exist")
    output = parent / expanded.name
    if os.path.lexists(output):
        raise FileExistsError("Exp416 output directory must not exist")
    return output


def _git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def _require_clean_git_commit() -> str:
    status = _git(["status", "--porcelain"])
    if status.returncode != 0:
        raise ValueError("Exp416 cannot inspect Git worktree")
    if status.stdout:
        raise ValueError("Exp416 requires a clean Git worktree")
    revision = _git(["rev-parse", "HEAD"])
    if revision.returncode != 0:
        raise ValueError("Exp416 cannot resolve Git commit")
    commit = revision.stdout.strip()
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("Exp416 resolved Git commit is invalid")
    return commit


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    database_path = _resolve_input_file(args.database_reference)
    prediction_path = _resolve_input_file(args.prediction)
    manifest_path = _resolve_input_file(args.prediction_manifest)
    labels_path = _resolve_input_file(args.frozen_test_labels)
    output_dir = _resolve_new_output(args.output_dir)

    git_commit = _require_clean_git_commit()
    database_sha256 = verify_database_reference(database_path)
    frozen_prediction = freeze_prediction_inputs(prediction_path, manifest_path)
    frozen_labels = load_frozen_test_labels(labels_path, frozen_prediction)
    result = compute_confusion_audit(frozen_prediction, frozen_labels)
    payloads = render_payloads(
        result,
        frozen_prediction,
        frozen_labels,
        database_sha256,
        git_commit,
    )
    verify_rendered_payloads(payloads)

    provenance = {
        "database_reference_sha256": database_sha256,
        "exp415_prediction_created": False,
        "frozen_test_labels_sha256": frozen_labels.labels_sha256,
        "git_commit": git_commit,
        "operational_policy_changed": False,
        "prediction_identity_sha256": frozen_prediction.identity_sha256,
        "prediction_manifest_sha256": frozen_prediction.manifest_sha256,
        "prediction_modified": False,
        "prediction_sha256": frozen_prediction.prediction_sha256,
        "test_informed_reselection": False,
        "test_labels_loaded_after_prediction_freeze": True,
    }
    hashes = publish_payloads(
        payloads,
        output_dir,
        provenance,
        pre_publish_verify=lambda: verify_input_snapshots(
            database_path,
            prediction_path,
            manifest_path,
            labels_path,
            database_sha256,
            frozen_prediction,
            frozen_labels,
        ),
    )
    print(
        json.dumps(
            {
                "artifact_sha256": dict(hashes),
                "micro_counts": asdict(result.micro_counts),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
