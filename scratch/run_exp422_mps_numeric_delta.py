#!/usr/bin/env python3
"""Run one guarded, bounded Exp422 TRAIN-only diagnostic attempt."""

from __future__ import annotations

import argparse
import errno
import os
from pathlib import Path
import subprocess
from typing import Sequence

from experiments.exp417_causal_multihorizon_shadow.inputs import (
    verified_read_only_connection,
)
from experiments.exp422_mps_numeric_delta.capability import detect_capability
from experiments.exp422_mps_numeric_delta.config import FROZEN_CONFIG, frozen_config_sha256
from experiments.exp422_mps_numeric_delta.core_parity import run_core_parity
from experiments.exp422_mps_numeric_delta.manifest import (
    build_manifest,
    verify_source_manifest,
    write_manifest,
)
from experiments.exp422_mps_numeric_delta.score_parity import (
    run_score_parity,
    select_train_sample,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_OUTPUT_COMPONENTS = ("exp419", "checkpoint")


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
        description="Run the bounded TRAIN-only Exp422 CPU/MPS numeric-delta diagnostic."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exp419-pidfile", type=Path, required=True)
    parser.add_argument("--exp419-exitfile", type=Path, required=True)
    parser.add_argument("--repeats", type=_bounded_repeats, required=True)
    return parser.parse_args(argv)


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


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        raise ValueError("unable to inspect the Exp419 PID") from exc
    return True


def require_exp419_finished(pidfile: Path, exitfile: Path) -> None:
    """Require no live official PID and an explicit successful exit record."""
    pid_path = Path(pidfile)
    if pid_path.exists() or pid_path.is_symlink():
        try:
            pid = int(_read_control_file(pid_path, "PID file"), 10)
        except ValueError as exc:
            if "PID file" in str(exc):
                raise
            raise ValueError("Exp419 PID file is malformed") from exc
        if pid < 1:
            raise ValueError("Exp419 PID file is malformed")
        if _pid_is_alive(pid):
            raise ValueError("Exp419 is still running")
    exit_path = Path(exitfile)
    if not exit_path.exists():
        raise ValueError("Exp419 exit file is absent")
    try:
        exit_code = int(_read_control_file(exit_path, "exit file"), 10)
    except ValueError as exc:
        if "exit file" in str(exc):
            raise
        raise ValueError("Exp419 exit file is malformed") from exc
    if exit_code != 0:
        raise ValueError("Exp419 exit code is nonzero")


def validate_output_root(path: Path) -> Path:
    """Accept only a non-Exp419 output directory that is absent or empty."""
    output = Path(path)
    resolved = output.resolve(strict=False)
    components = tuple(part.casefold() for part in (*output.parts, *resolved.parts))
    if any(token in component for component in components for token in _FORBIDDEN_OUTPUT_COMPONENTS):
        raise ValueError("Exp419 and checkpoint output paths are forbidden")
    if output.is_symlink():
        raise ValueError("output root must not be a symlink")
    if output.exists():
        if not output.is_dir():
            raise ValueError("output root must be a directory")
        try:
            if any(output.iterdir()):
                raise ValueError("output root must be fresh or empty")
        except OSError as exc:
            raise ValueError("output root is not readable") from exc
    return output


def current_git_commit(repo_root: Path = REPO_ROOT) -> str:
    """Return the exact checked-out Git object identity for provenance."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("unable to bind Exp422 to a Git commit") from exc
    commit = result.stdout.strip()
    if len(commit) not in {40, 64} or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Git commit is not canonical")
    return commit


def preflight_database(database: Path) -> str:
    """Verify the frozen database identity and SQLite integrity read-only."""
    with verified_read_only_connection(
        Path(database), FROZEN_CONFIG.exp417_config.database_sha256
    ) as connection:
        try:
            row = connection.execute("PRAGMA quick_check").fetchone()
        except Exception as exc:
            raise ValueError("SQLite quick-check failed") from exc
    if row != ("ok",):
        raise ValueError("SQLite quick-check did not return ok")
    return "ok"


def _provenance(
    *,
    output_root: Path,
    repeats: int,
    git_commit: str,
    source_manifest_sha256: str,
    sqlite_quick_check: str,
    capability: object,
) -> dict[str, object]:
    return {
        "attempt_root": str(output_root.resolve()),
        "config_sha256": frozen_config_sha256(),
        "database_sha256": FROZEN_CONFIG.exp417_config.database_sha256,
        "device_name": capability.device_name,
        "fresh_no_clobber_attempt": True,
        "git_commit": git_commit,
        "mlx_availability": capability.mlx_availability,
        "mlx_eligibility_reason": (
            "algorithm_rewrite_required"
            if capability.mlx_availability == "available"
            else "package_absent"
        ),
        "mlx_metal_version": capability.mlx_metal_version,
        "mlx_version": capability.mlx_version,
        "mps_available": capability.mps_available,
        "mps_built": capability.mps_built,
        "repeats": repeats,
        "source_manifest_sha256": source_manifest_sha256,
        "sqlite_quick_check": sqlite_quick_check,
        "torch_version": capability.torch_version,
        "train_only": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_exp419_finished(args.exp419_pidfile, args.exp419_exitfile)
    output_root = validate_output_root(args.output_root)
    capability = detect_capability()
    quick_check = preflight_database(args.database)
    source_digest = verify_source_manifest(REPO_ROOT)
    git_commit = current_git_commit(REPO_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    core = None
    score = None
    sample = None
    for _repeat in range(args.repeats):
        core = run_core_parity(FROZEN_CONFIG, capability)
        if core.stage.state != "passed":
            score = None
            break
        if sample is None:
            sample = select_train_sample(args.database, FROZEN_CONFIG)
        score = run_score_parity(sample, capability, FROZEN_CONFIG)
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
            source_manifest_sha256=source_digest,
            sqlite_quick_check=quick_check,
            capability=capability,
        ),
    )
    path = write_manifest(output_root / "manifest.json", manifest)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
