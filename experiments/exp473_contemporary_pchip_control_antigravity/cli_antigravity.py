"""Closed, read-only observation surface for Exp473 D-prime."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import TextIO

from experiments.exp472_makima_upsampling_ablation_antigravity.config_antigravity import (
    assess_exp472_backend_antigravity,
)

from .config_antigravity import (
    EXP473_OFFICIAL_ROOT_ANTIGRAVITY,
    EXP473_SMOKE_ROOT_ANTIGRAVITY,
    FROZEN_EXP473_CONFIG_ANTIGRAVITY,
)
from .contracts_antigravity import canonical_json_bytes_exp473_antigravity


def _status_exp473_antigravity(mode: str) -> dict[str, object]:
    if mode not in {"smoke", "full"}:
        raise ValueError("Exp473 status mode differs")
    root = (
        EXP473_SMOKE_ROOT_ANTIGRAVITY
        if mode == "smoke"
        else EXP473_OFFICIAL_ROOT_ANTIGRAVITY
    )
    total = (
        len(FROZEN_EXP473_CONFIG_ANTIGRAVITY.smoke_datasets)
        if mode == "smoke"
        else FROZEN_EXP473_CONFIG_ANTIGRAVITY.dataset_count
    )
    shard_root = root / "score_shards_antigravity"
    completed = 0
    if shard_root.is_dir() and not shard_root.is_symlink():
        completed = sum(
            1
            for path in shard_root.glob("*_antigravity/score_shard_antigravity.json")
            if path.is_file() and not path.is_symlink()
        )
    manifest = root / "score_manifest_antigravity.json"
    return {
        "experiment": "Exp473",
        "mode": mode,
        "root": str(root),
        "scored": completed,
        "failed": 0,
        "remaining": max(0, total - completed),
        "manifest_present": manifest.is_file() and not manifest.is_symlink(),
        "read_only": True,
    }


def main_antigravity(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    runtime: object | None = None,
) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    command: str
    mode: str
    if arguments == ("status",):
        command = "status"
        mode = "full"
    elif (
        len(arguments) == 3
        and arguments[0] in {"status", "verify"}
        and arguments[1] == "--mode"
        and arguments[2] in {"smoke", "full"}
    ):
        command = arguments[0]
        mode = arguments[2]
    elif arguments in {("readiness",), ("smoke",), ("run",), ("verify",)}:
        command = arguments[0]
        mode = "smoke" if command == "smoke" else "full"
    else:
        raise ValueError(
            "Exp473 command must be readiness|smoke|run|status|verify, "
            "or status|verify --mode smoke|full"
        )
    stream = sys.stdout if stdout is None else stdout
    if command == "status":
        record: object = _status_exp473_antigravity(mode)
    elif command == "readiness":
        record = {
            "experiment": "Exp473",
            "database_sha256": FROZEN_EXP473_CONFIG_ANTIGRAVITY.database_sha256,
            "dataset_count": FROZEN_EXP473_CONFIG_ANTIGRAVITY.dataset_count,
            "test_row_count": FROZEN_EXP473_CONFIG_ANTIGRAVITY.test_row_count,
            "backend": assess_exp472_backend_antigravity().artifact_record(),
            "labels_opened": False,
        }
    else:
        typed = runtime
        if typed is None:
            from .runtime_antigravity import build_runtime_exp473_antigravity

            typed = build_runtime_exp473_antigravity(mode)
        if getattr(typed, "mode", None) != mode:
            raise ValueError("Exp473 command runtime mode differs")
        if command in {"smoke", "run"}:
            run_method = getattr(typed, "run_exp473_antigravity", None)
            if not callable(run_method):
                raise ValueError("Exp473 command requires a typed runtime")
            record = {
                "experiment": "Exp473",
                "command": command,
                "result": run_method(),
            }
        else:
            verify_method = getattr(typed, "verify_exp473_antigravity", None)
            if not callable(verify_method):
                raise ValueError("Exp473 command requires a typed runtime")
            record = {
                "experiment": "Exp473",
                "command": "verify",
                "result": verify_method(),
                "read_only": True,
            }
    stream.write(canonical_json_bytes_exp473_antigravity(record).decode("utf-8"))
    stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main_antigravity())


__all__ = ["main_antigravity"]
