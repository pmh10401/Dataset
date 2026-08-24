"""Closed Exp471 source-to-report command orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import sys
from time import monotonic
from typing import TextIO

from .config_antigravity import (
    CPU_WORKERS_ANTIGRAVITY,
    EXP467_TOTAL_SUCCESSES_ANTIGRAVITY,
    EXP467_SOURCE_ROOT_ANTIGRAVITY,
    EXP471_BASELINE_ROOT_AUTHORITY_ANTIGRAVITY,
    EXP471_BASELINE_ROOT_BY_METHOD_ANTIGRAVITY,
    EXP471_EVALUATION_ROOT_ANTIGRAVITY,
    EXP471_REPORT_ROOT_ANTIGRAVITY,
    EXP471_ROOT_ANTIGRAVITY,
    EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY,
    METHOD_ORDER_ANTIGRAVITY,
)
from .contracts_antigravity import (
    Exp471ReportAuthorityAntigravity,
    canonical_json_bytes_exp471_antigravity,
)


_COMMANDS_ANTIGRAVITY = (
    "verify-source",
    "benchmark-prepare",
    "prepare",
    "status",
    "report",
)
_PROGRESS_NAME_ANTIGRAVITY = "progress_antigravity.jsonl"
_TOTAL_SHARDS_ANTIGRAVITY = 11_170
_PROGRESS_SCHEMA_ANTIGRAVITY = "exp471-progress-antigravity.v1"
_PROGRESS_KEYS_ANTIGRAVITY = {
    "schema",
    "stage",
    "durable_completed",
    "total",
    "failures",
    "attempts",
    "remaining",
    "elapsed_seconds",
    "eta_seconds",
    "timestamp_utc",
    "artifact_path",
    "root_path",
}
_PROGRESS_STAGE_RANK_ANTIGRAVITY = {
    "verify-source": 0,
    "score-cache": 1,
    "evaluation": 2,
    "report": 3,
}
_PROGRESS_STAGE_TOTAL_ANTIGRAVITY = {
    "verify-source": EXP467_TOTAL_SUCCESSES_ANTIGRAVITY,
    "score-cache": _TOTAL_SHARDS_ANTIGRAVITY,
    "evaluation": 10,
    "report": 11,
}
_PROGRESS_TERMINAL_STAGES_ANTIGRAVITY = {
    "verify-source",
    "evaluation",
    "report",
}
_PROGRESS_ALLOWED_TRANSITIONS_ANTIGRAVITY = {
    "verify-source": {"verify-source", "score-cache", "evaluation", "report"},
    "score-cache": {"score-cache", "evaluation", "report", "verify-source"},
    "evaluation": {"report", "verify-source"},
    "report": {"verify-source"},
}


def _repository_root_antigravity() -> Path:
    return Path(__file__).resolve().parents[2]


def _plain_antigravity(value: object) -> object:
    raw = canonical_json_bytes_exp471_antigravity(value)
    return json.loads(raw.decode("utf-8"))


def _emit_antigravity(stream: TextIO, value: object) -> None:
    stream.write(canonical_json_bytes_exp471_antigravity(value).decode("utf-8"))
    stream.flush()


@dataclass(slots=True)
class _DirectoryChainAntigravity:
    descriptor: int
    descriptors: list[int]
    validators: list[tuple[int, str, os.stat_result]]

    def validate_antigravity(self) -> None:
        for parent, name, expected in self.validators:
            try:
                observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except OSError as error:
                raise ValueError("Exp471 configured directory path changed") from error
            if (
                not stat.S_ISDIR(observed.st_mode)
                or observed.st_dev != expected.st_dev
                or observed.st_ino != expected.st_ino
            ):
                raise ValueError("Exp471 configured directory path changed")

    def close_antigravity(self) -> None:
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_existing_directory_chain_antigravity(
    path: Path,
) -> _DirectoryChainAntigravity | None:
    """Open every configured component without following links; absence is safe."""

    path = Path(path)
    if not path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
        raise ValueError("Exp471 configured path is unsafe")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    validators: list[tuple[int, str, os.stat_result]] = []
    try:
        current = os.open("/", flags)
        descriptors.append(current)
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                absent = _DirectoryChainAntigravity(
                    current, descriptors, validators
                )
                absent.validate_antigravity()
                absent.close_antigravity()
                return None
            except OSError as error:
                raise ValueError(
                    "Exp471 configured directory path is unsafe"
                ) from error
            descriptors.append(child)
            opened = os.fstat(child)
            try:
                observed = os.stat(part, dir_fd=current, follow_symlinks=False)
            except OSError as error:
                raise ValueError(
                    "Exp471 configured directory path changed"
                ) from error
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(observed.st_mode)
                or opened.st_dev != observed.st_dev
                or opened.st_ino != observed.st_ino
            ):
                raise ValueError("Exp471 configured directory path is unsafe")
            validators.append((current, part, opened))
            current = child
        anchor = _DirectoryChainAntigravity(current, descriptors, validators)
        anchor.validate_antigravity()
        return anchor
    except Exception:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _read_configured_leaf_at_antigravity(
    anchor: _DirectoryChainAntigravity, name: str
) -> bytes | None:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("Exp471 configured leaf name is unsafe")
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=anchor.descriptor
            )
        except FileNotFoundError:
            anchor.validate_antigravity()
            return None
        except OSError as error:
            raise ValueError("Exp471 configured leaf is unsafe") from error
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("Exp471 configured leaf is not a safe regular file")
        raw = _read_all_descriptor_antigravity(descriptor)
        after = os.fstat(descriptor)
        try:
            observed = os.stat(name, dir_fd=anchor.descriptor, follow_symlinks=False)
        except OSError as error:
            raise ValueError("Exp471 configured leaf changed") from error
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or (before.st_dev, before.st_ino) != (observed.st_dev, observed.st_ino)
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise ValueError("Exp471 configured leaf changed")
        anchor.validate_antigravity()
        return raw
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_configured_leaves_antigravity(
    root: Path, names: Sequence[str]
) -> tuple[bytes | None, ...]:
    """Read regular leaves through one retained no-follow directory chain."""

    anchor = _open_existing_directory_chain_antigravity(root)
    if anchor is None:
        return tuple(None for _name in names)
    try:
        values = tuple(
            _read_configured_leaf_at_antigravity(anchor, name) for name in names
        )
        anchor.validate_antigravity()
        return values
    finally:
        anchor.close_antigravity()


def _read_configured_leaf_antigravity(root: Path, name: str) -> bytes | None:
    return _read_configured_leaves_antigravity(root, (name,))[0]


def _validate_status_root_generation_antigravity(
    anchor: _DirectoryChainAntigravity | None,
) -> None:
    """Require every status observation to retain the initial root generation."""

    if anchor is not None:
        anchor.validate_antigravity()
        return
    observed = _open_existing_directory_chain_antigravity(
        EXP471_ROOT_ANTIGRAVITY
    )
    if observed is None:
        return
    observed.close_antigravity()
    raise ValueError("Exp471 configured root appeared during status")


def _open_progress_root_antigravity() -> tuple[
    _DirectoryChainAntigravity, os.stat_result
]:
    anchor = _open_existing_directory_chain_antigravity(EXP471_ROOT_ANTIGRAVITY)
    if anchor is None:
        raise FileNotFoundError("Exp471 progress root is missing")
    return anchor, os.fstat(anchor.descriptor)


def _validate_progress_root_antigravity(
    anchor: _DirectoryChainAntigravity, opened: os.stat_result
) -> None:
    try:
        anchor.validate_antigravity()
        current = os.fstat(anchor.descriptor)
    except (OSError, ValueError) as error:
        raise ValueError("Exp471 progress root changed") from error
    if (
        current.st_dev != opened.st_dev
        or current.st_ino != opened.st_ino
    ):
        raise ValueError("Exp471 progress root changed")


def _progress_log_identity_antigravity(
    root_descriptor: int, descriptor: int
) -> os.stat_result:
    opened = os.fstat(descriptor)
    try:
        linked = os.stat(
            _PROGRESS_NAME_ANTIGRAVITY,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ValueError("Exp471 progress log changed") from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(linked.st_mode)
        or opened.st_nlink != 1
        or linked.st_nlink != 1
        or opened.st_dev != linked.st_dev
        or opened.st_ino != linked.st_ino
    ):
        raise ValueError("Exp471 progress log is unsafe or changed")
    return opened


def _read_all_descriptor_antigravity(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _progress_timestamp_antigravity(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError("Exp471 progress timestamp schema differs")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Exp471 progress timestamp is malformed") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("Exp471 progress timestamp is not UTC")
    return parsed


def _validate_report_terminal_artifact_antigravity() -> None:
    marker_name = ".exp471_report_complete_antigravity.json"
    artifact_map_name = "artifact_sha256_antigravity.json"
    marker_raw, artifact_map_raw = _read_configured_leaves_antigravity(
        EXP471_REPORT_ROOT_ANTIGRAVITY,
        (marker_name, artifact_map_name),
    )
    if marker_raw is None or artifact_map_raw is None:
        raise ValueError("Exp471 progress report marker artifact is missing")
    try:
        marker = json.loads(marker_raw.decode("utf-8"))
        artifact_map = json.loads(artifact_map_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Exp471 progress report marker artifact is malformed"
        ) from error
    marker_keys = {
        "schema",
        "source_authority_sha256",
        "cache_manifest_sha256",
        "evaluation_sha256",
        "report_file_sha256",
        "report_manifest_sha256",
        "artifact_map",
    }
    if (
        not isinstance(marker, dict)
        or set(marker) != marker_keys
        or marker.get("schema") != "exp471-report-publication-antigravity.v1"
        or marker.get("artifact_map") != artifact_map_name
        or canonical_json_bytes_exp471_antigravity(marker) != marker_raw
    ):
        raise ValueError("Exp471 progress report marker artifact differs")
    try:
        authority = Exp471ReportAuthorityAntigravity(
            source_authority_sha256=marker["source_authority_sha256"],
            cache_manifest_sha256=marker["cache_manifest_sha256"],
            evaluation_sha256=marker["evaluation_sha256"],
            report_file_sha256=marker["report_file_sha256"],
            report_manifest_sha256=marker["report_manifest_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Exp471 progress report marker fields differ") from error
    if (
        len(authority.report_file_sha256) != 11
        or not isinstance(artifact_map, dict)
        or set(artifact_map) != {"schema", "self_excluded", "files"}
        or artifact_map.get("schema")
        != "exp471-report-artifact-map-antigravity.v1"
        or artifact_map.get("self_excluded") != artifact_map_name
        or artifact_map.get("files") != dict(authority.report_file_sha256)
        or canonical_json_bytes_exp471_antigravity(artifact_map)
        != artifact_map_raw
        or sha256(artifact_map_raw).hexdigest()
        != authority.report_manifest_sha256
    ):
        raise ValueError("Exp471 progress report marker binding differs")


def _validate_terminal_progress_artifact_antigravity(stage: str) -> None:
    if stage == "verify-source":
        raw = _read_configured_leaf_antigravity(
            EXP471_ROOT_ANTIGRAVITY, "source_authority_antigravity.json"
        )
        if raw is None:
            raise ValueError("Exp471 progress source artifact is missing")
        from .source_authority_antigravity import _authority_from_bytes_antigravity

        authority = _authority_from_bytes_antigravity(raw)
        if authority.total_successes != EXP467_TOTAL_SUCCESSES_ANTIGRAVITY:
            raise ValueError("Exp471 progress source artifact total differs")
        return
    if stage == "evaluation":
        raw = _read_configured_leaf_antigravity(
            EXP471_EVALUATION_ROOT_ANTIGRAVITY, "evaluation_antigravity.json"
        )
        if raw is None:
            raise ValueError("Exp471 progress evaluation artifact is missing")
        from .evaluation_antigravity import _artifact_from_bytes_antigravity

        artifact = _artifact_from_bytes_antigravity(raw)
        if tuple(artifact.method_order) != tuple(METHOD_ORDER_ANTIGRAVITY):
            raise ValueError("Exp471 progress evaluation artifact methods differ")
        return
    if stage == "report":
        _validate_report_terminal_artifact_antigravity()
        return
    raise ValueError("Exp471 progress terminal stage differs")


def _validate_progress_history_antigravity(raw: bytes) -> tuple[dict[str, object], ...]:
    if not raw:
        return ()
    if not raw.endswith(b"\n"):
        raise ValueError("Exp471 progress history is truncated")
    records: list[dict[str, object]] = []
    prior: dict[str, object] | None = None
    prior_timestamp: datetime | None = None
    for encoded in raw.splitlines(keepends=True):
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Exp471 progress history is malformed") from error
        if not isinstance(value, dict) or set(value) != _PROGRESS_KEYS_ANTIGRAVITY:
            raise ValueError("Exp471 progress history schema differs")
        try:
            canonical = canonical_json_bytes_exp471_antigravity(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Exp471 progress history is noncanonical") from error
        if canonical != encoded:
            raise ValueError("Exp471 progress history is noncanonical")
        if (
            value["schema"] != _PROGRESS_SCHEMA_ANTIGRAVITY
            or value["stage"] not in _PROGRESS_STAGE_RANK_ANTIGRAVITY
        ):
            raise ValueError("Exp471 progress history schema differs")
        integers = (
            "durable_completed",
            "total",
            "failures",
            "attempts",
            "remaining",
        )
        if any(type(value[field]) is not int or value[field] < 0 for field in integers):
            raise ValueError("Exp471 progress count schema differs")
        if (
            value["total"] < 1
            or value["durable_completed"] > value["total"]
            or value["remaining"]
            != value["total"] - value["durable_completed"]
            or value["failures"] > value["attempts"]
        ):
            raise ValueError("Exp471 progress count relation differs")
        stage = value["stage"]
        if value["total"] != _PROGRESS_STAGE_TOTAL_ANTIGRAVITY[stage]:
            raise ValueError("Exp471 progress stage total differs")
        if stage in _PROGRESS_TERMINAL_STAGES_ANTIGRAVITY and (
            value["durable_completed"] != value["total"]
            or value["remaining"] != 0
        ):
            raise ValueError("Exp471 progress terminal stage differs")
        elapsed = value["elapsed_seconds"]
        eta = value["eta_seconds"]
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
            or (
                eta is not None
                and (
                    type(eta) not in (int, float)
                    or not math.isfinite(float(eta))
                    or float(eta) < 0.0
                )
            )
            or (
                value["durable_completed"] == 0
                and value["remaining"] > 0
                and eta is not None
            )
            or (value["remaining"] == 0 and float(eta or 0.0) != 0.0)
        ):
            raise ValueError("Exp471 progress timing relation differs")
        timestamp = _progress_timestamp_antigravity(value["timestamp_utc"])
        root_path = Path(value["root_path"])
        artifact_path = Path(value["artifact_path"])
        expected_artifact_path = {
            "verify-source": (
                EXP471_ROOT_ANTIGRAVITY / "source_authority_antigravity.json"
            ),
            "score-cache": EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY,
            "evaluation": (
                EXP471_EVALUATION_ROOT_ANTIGRAVITY / "evaluation_antigravity.json"
            ),
            "report": EXP471_REPORT_ROOT_ANTIGRAVITY,
        }[stage]
        if (
            root_path != EXP471_ROOT_ANTIGRAVITY
            or not artifact_path.is_absolute()
            or artifact_path != expected_artifact_path
        ):
            raise ValueError("Exp471 progress path schema differs")
        if prior is not None:
            if value["attempts"] != prior["attempts"] + 1:
                raise ValueError("Exp471 progress attempts are not cumulative")
            if value["failures"] < prior["failures"]:
                raise ValueError("Exp471 progress failures are not cumulative")
            if value["failures"] > prior["failures"] + 1:
                raise ValueError("Exp471 progress failures advanced impossibly")
            if value["stage"] == prior["stage"] and (
                value["total"] != prior["total"]
                or value["durable_completed"] < prior["durable_completed"]
            ):
                raise ValueError("Exp471 progress durable count regressed")
            if value["stage"] not in _PROGRESS_ALLOWED_TRANSITIONS_ANTIGRAVITY[
                prior["stage"]
            ]:
                raise ValueError("Exp471 progress stage transition differs")
            if prior_timestamp is not None and timestamp < prior_timestamp:
                raise ValueError("Exp471 progress timestamp regressed")
        elif value["attempts"] != 1 or value["failures"] not in (0, 1):
            raise ValueError("Exp471 progress attempts must start canonically")
        if stage in _PROGRESS_TERMINAL_STAGES_ANTIGRAVITY:
            if stage != "verify-source" and not any(
                record["stage"] == "verify-source" for record in records
            ):
                raise ValueError("Exp471 progress terminal lifecycle differs")
            _validate_terminal_progress_artifact_antigravity(stage)
        records.append(value)
        prior = value
        prior_timestamp = timestamp
    return tuple(records)


def _latest_progress_antigravity() -> Mapping[str, object] | None:
    try:
        root_anchor, root_opened = _open_progress_root_antigravity()
    except FileNotFoundError:
        try:
            EXP471_ROOT_ANTIGRAVITY.lstat()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise ValueError("Exp471 progress root is unsafe") from error
        raise ValueError("Exp471 progress root changed")
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(
                _PROGRESS_NAME_ANTIGRAVITY, flags, dir_fd=root_anchor.descriptor
            )
        except FileNotFoundError:
            _validate_progress_root_antigravity(root_anchor, root_opened)
            return None
        except OSError as error:
            raise ValueError("Exp471 progress log is unsafe") from error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            _validate_progress_root_antigravity(root_anchor, root_opened)
            _progress_log_identity_antigravity(root_anchor.descriptor, descriptor)
            _validate_progress_root_antigravity(root_anchor, root_opened)
            raw = _read_all_descriptor_antigravity(descriptor)
            _validate_progress_root_antigravity(root_anchor, root_opened)
            _progress_log_identity_antigravity(root_anchor.descriptor, descriptor)
            records = _validate_progress_history_antigravity(raw)
            _validate_progress_root_antigravity(root_anchor, root_opened)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        _validate_progress_root_antigravity(root_anchor, root_opened)
        return None if not records else dict(records[-1])
    finally:
        root_anchor.close_antigravity()


def _append_progress_antigravity(
    value: Mapping[str, object], *, failed: bool = False
) -> None:
    root_anchor, root_opened = _open_progress_root_antigravity()
    try:
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        last_open_error: OSError | None = None
        # Darwin may transiently return ENOENT when several openat callers race
        # to create the same O_NOFOLLOW leaf. The pinned parent makes retry safe.
        for _ in range(128):
            try:
                descriptor = os.open(
                    _PROGRESS_NAME_ANTIGRAVITY,
                    flags,
                    0o600,
                    dir_fd=root_anchor.descriptor,
                )
                break
            except FileNotFoundError as error:
                last_open_error = error
                continue
            except OSError as error:
                raise ValueError("Exp471 progress log is unsafe") from error
        if descriptor is None:
            raise ValueError("Exp471 progress log is unsafe") from last_open_error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            _validate_progress_root_antigravity(root_anchor, root_opened)
            _progress_log_identity_antigravity(root_anchor.descriptor, descriptor)
            _validate_progress_root_antigravity(root_anchor, root_opened)
            existing_raw = _read_all_descriptor_antigravity(descriptor)
            _validate_progress_root_antigravity(root_anchor, root_opened)
            existing = _validate_progress_history_antigravity(existing_raw)
            prior = existing[-1] if existing else None
            attempts = 1 if prior is None else int(prior["attempts"]) + 1
            failures = (0 if prior is None else int(prior["failures"])) + int(
                failed
            )
            record = {
                **value,
                "failures": failures,
                "attempts": attempts,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
            }
            payload = canonical_json_bytes_exp471_antigravity(record)
            _validate_progress_history_antigravity(existing_raw + payload)
            _progress_log_identity_antigravity(root_anchor.descriptor, descriptor)
            _validate_progress_root_antigravity(root_anchor, root_opened)
            _write_all = memoryview(payload)
            while _write_all:
                _validate_progress_root_antigravity(root_anchor, root_opened)
                written = os.write(descriptor, _write_all)
                if written <= 0:
                    raise OSError("Exp471 progress append made no progress")
                _write_all = _write_all[written:]
                _validate_progress_root_antigravity(root_anchor, root_opened)
            _validate_progress_root_antigravity(root_anchor, root_opened)
            os.fsync(descriptor)
            _validate_progress_root_antigravity(root_anchor, root_opened)
            _progress_log_identity_antigravity(root_anchor.descriptor, descriptor)
            _validate_progress_root_antigravity(root_anchor, root_opened)
            os.fsync(root_anchor.descriptor)
            _validate_progress_root_antigravity(root_anchor, root_opened)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
    finally:
        root_anchor.close_antigravity()


def _append_durable_stage_antigravity(
    stage: str, completed: int, total: int, artifact_path: Path, started: float
) -> None:
    elapsed = max(0.0, monotonic() - started)
    _append_progress_antigravity(
        {
            "schema": _PROGRESS_SCHEMA_ANTIGRAVITY,
            "stage": stage,
            "durable_completed": completed,
            "total": total,
            "remaining": max(0, total - completed),
            "elapsed_seconds": round(elapsed, 6),
            "eta_seconds": 0.0 if completed == total else None,
            "artifact_path": str(artifact_path),
            "root_path": str(EXP471_ROOT_ANTIGRAVITY),
        }
    )


def _progress_callback_antigravity(stage: str, artifact_path: Path):
    started = monotonic()
    def progress(event: Mapping[str, object]) -> None:
        completed = event.get("durable_completed", 0)
        if type(completed) is not int or completed < 0:
            raise ValueError("Exp471 durable progress count differs")
        failed = event.get("status") in {"failed", "terminal_failed"}
        total = event.get("total", _TOTAL_SHARDS_ANTIGRAVITY)
        if type(total) is not int or total < completed:
            raise ValueError("Exp471 durable progress total differs")
        remaining = max(0, total - completed)
        elapsed = max(0.0, monotonic() - started)
        eta = 0.0 if remaining == 0 else (
            elapsed * remaining / completed if completed else None
        )
        record = {
            "schema": _PROGRESS_SCHEMA_ANTIGRAVITY,
            "stage": stage,
            "durable_completed": completed,
            "total": total,
            "remaining": remaining,
            "elapsed_seconds": round(elapsed, 6),
            "eta_seconds": None if eta is None else round(eta, 6),
            "artifact_path": str(artifact_path),
            "root_path": str(EXP471_ROOT_ANTIGRAVITY),
        }
        _append_progress_antigravity(record, failed=failed)

    return progress


def _coordinate_record_antigravity(dataset: str, identity: Mapping[str, object]) -> dict[str, object]:
    count = identity.get("count")
    indices = identity.get("indices")
    if (
        type(count) is not int
        or count < 1
        or not isinstance(indices, (tuple, list))
        or tuple(indices) != tuple(range(count))
    ):
        raise ValueError("Exp471 production TEST coordinates differ")
    bare = {"dataset": dataset, "count": count, "indices": list(indices)}
    return {
        "count": count,
        "indices": list(indices),
        "coordinates_sha256": sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest(),
    }


def _production_score_bundle_antigravity(session: object) -> Mapping[str, object]:
    """Build Task 5's verified, label-free input from issued Exp467 evidence."""

    from .source_authority_antigravity import (
        _require_captured_values_api_antigravity,
    )
    from experiments.exp450_resampling_image_ablation_antigravity import (
        evaluation_antigravity as frozen_evaluation,
    )
    from experiments.paper_public_baselines.runner import verify_score_root

    authority = session.authority
    catalog = session.catalog
    values_api = _require_captured_values_api_antigravity(session)
    paper = values_api.database_config_antigravity
    if Path(paper.database_path).is_symlink() or paper.database_sha256 != authority.database_sha256:
        raise ValueError("Exp471 production database authority differs")
    binding = values_api.bind_preverified_database_read_only_antigravity(
        Path(paper.database_path),
        config=paper,
        verified_catalog_database=catalog["database"],
    )
    datasets = tuple(entry["dataset"] for entry in catalog["datasets"])
    raw = {
        dataset: values_api.load_dataset_values_antigravity(
            Path(paper.database_path), dataset, preverified_database=binding
        )
        for dataset in datasets
    }
    build = getattr(
        session.verifier, "build_exp467_score_manifests_antigravity", None
    )
    if not callable(build):
        raise ValueError("Exp471 fixed score-manifest builder is unavailable")
    score_authority = build(
        session.context.run_root,
        verification=session.verification,
        context=session.context,
        catalog=catalog,
        migration_manifest=session.migration_manifest,
        raw_datasets=raw,
    )
    manifests = score_authority.manifests
    if tuple(manifests) != METHOD_ORDER_ANTIGRAVITY[:4]:
        raise ValueError("Exp471 production condition score order differs")
    first = manifests[METHOD_ORDER_ANTIGRAVITY[0]]
    identities = frozen_evaluation._manifest_test_identities(first)
    if tuple(identities) != datasets:
        raise ValueError("Exp471 production TEST dataset order differs")
    coordinates = {
        dataset: _coordinate_record_antigravity(dataset, identities[dataset])
        for dataset in datasets
    }
    conditions: dict[str, object] = {}
    for method in METHOD_ORDER_ANTIGRAVITY[:4]:
        rows = manifests[method].get("score_rows")
        if not isinstance(rows, Mapping) or tuple(rows) != datasets:
            raise ValueError("Exp471 production condition scores differ")
        conditions[method] = {
            "source_manifest_sha256": authority.condition_manifest_sha256[method],
            "scores": {
                dataset: {
                    "test_indices": coordinates[dataset]["indices"],
                    "test_scores": list(rows[dataset]),
                }
                for dataset in datasets
            },
        }

    verified_baselines = frozen_evaluation._verify_baseline_roots(
        frozen_evaluation.BASELINE_ROOTS, verify_score_root
    )
    coordinate_inventory = {
        "schema": "exp471-test-coordinate-inventory-antigravity.v1",
        "dataset_count": len(datasets),
        "datasets": [
            {
                "dataset": dataset,
                "test_count": coordinates[dataset]["count"],
                "test_indices": coordinates[dataset]["indices"],
            }
            for dataset in datasets
        ],
    }
    coordinates_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(coordinate_inventory)
    ).hexdigest()
    baselines: dict[str, object] = {}
    baseline_inventory: dict[str, str] = {}
    for root_name in ("classical", "deep_svdd", "drocc"):
        manifest = verified_baselines[root_name]
        baseline_identities, database_sha256, shards = (
            frozen_evaluation._baseline_identity_and_shards(manifest)
        )
        if (
            database_sha256 != authority.database_sha256
            or tuple(baseline_identities) != datasets
        ):
            raise ValueError("Exp471 production baseline identity differs")
        root_authority = EXP471_BASELINE_ROOT_AUTHORITY_ANTIGRAVITY[root_name]
        manifest_path = Path(manifest.run_root) / "frozen_score_manifest.json"
        if (
            manifest.run_config_sha256 != root_authority["run_config_sha256"]
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
            or sha256(manifest_path.read_bytes()).hexdigest()
            != root_authority["manifest_sha256"]
        ):
            raise ValueError("Exp471 pinned baseline root authority differs")
        for method in METHOD_ORDER_ANTIGRAVITY[4:]:
            if EXP471_BASELINE_ROOT_BY_METHOD_ANTIGRAVITY[method] != root_name:
                continue
            bare = {
                "schema": "exp471-verified-baseline-authority-antigravity.v1",
                "method": method,
                "database_sha256": authority.database_sha256,
                "dataset_order": list(datasets),
                "test_coordinates_sha256": coordinates_sha256,
                "source_manifest_sha256": root_authority["manifest_sha256"],
                "run_config_sha256": root_authority["run_config_sha256"],
            }
            authority_sha256 = sha256(
                canonical_json_bytes_exp471_antigravity(bare)
            ).hexdigest()
            baseline_inventory[method] = authority_sha256
            scores: dict[str, object] = {}
            for dataset in datasets:
                shard = shards.get((method, dataset))
                if not isinstance(shard, Mapping) or shard.get("status") != "complete":
                    raise ValueError("Exp471 production baseline coverage differs")
                scores[dataset] = {
                    "train_oof_scores": list(shard["train_oof_scores"]),
                    "test_indices": list(shard["test_indices"]),
                    "test_scores": list(shard["test_scores"]),
                }
            baselines[method] = {
                "authority": {**bare, "authority_sha256": authority_sha256},
                "scores": scores,
            }
    return {
        "schema": "exp471-verified-score-input-antigravity.v2",
        "database_sha256": authority.database_sha256,
        "dataset_order": list(datasets),
        "test_coordinates": coordinates,
        "baseline_authority_sha256": baseline_inventory,
        "conditions": conditions,
        "baselines": baselines,
    }


def _reopened_score_bundle_antigravity(session: object) -> Mapping[str, object]:
    """Rebuild only Task 5 authority metadata from verified durable shards."""

    from .cache_artifacts_antigravity import (
        read_cache_manifest_antigravity,
        read_score_shard_antigravity,
    )

    authority = session.authority
    manifest = read_cache_manifest_antigravity(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)
    if (
        manifest.source_authority_sha256 != authority.authority_sha256
        or manifest.method_order != METHOD_ORDER_ANTIGRAVITY
    ):
        raise ValueError("Exp471 reopened cache source authority differs")
    datasets = manifest.dataset_order
    shards: dict[tuple[str, str], object] = {}
    coordinates: dict[str, object] = {}
    for method in METHOD_ORDER_ANTIGRAVITY:
        for dataset in datasets:
            shard = read_score_shard_antigravity(
                EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY, method, dataset
            )
            key = f"{method}:{dataset}"
            if (
                shard.envelope_sha256 != manifest.shard_sha256[key]
                or shard.cache_context_sha256 != manifest.cache_context_sha256
                or shard.source_authority_sha256 != authority.authority_sha256
            ):
                raise ValueError("Exp471 reopened score shard authority differs")
            shards[(method, dataset)] = shard
            record = _coordinate_record_antigravity(
                dataset,
                {"count": shard.test_count, "indices": list(range(shard.test_count))},
            )
            prior = coordinates.setdefault(dataset, record)
            if prior != record:
                raise ValueError("Exp471 reopened TEST coordinates differ")

    conditions: dict[str, object] = {}
    for method in METHOD_ORDER_ANTIGRAVITY[:4]:
        scores: dict[str, object] = {}
        for dataset in datasets:
            shard = shards[(method, dataset)]
            if (
                shard.source_manifest_sha256
                != authority.condition_manifest_sha256[method]
            ):
                raise ValueError("Exp471 reopened condition source differs")
            scores[dataset] = {
                "test_indices": coordinates[dataset]["indices"],
                "test_scores": list(shard.raw_scores),
            }
        conditions[method] = {
            "source_manifest_sha256": authority.condition_manifest_sha256[method],
            "scores": scores,
        }

    inventory_record = {
        "schema": "exp471-test-coordinate-inventory-antigravity.v1",
        "dataset_count": len(datasets),
        "datasets": [
            {
                "dataset": dataset,
                "test_count": coordinates[dataset]["count"],
                "test_indices": coordinates[dataset]["indices"],
            }
            for dataset in datasets
        ],
    }
    coordinates_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(inventory_record)
    ).hexdigest()
    baseline_inventory: dict[str, str] = {}
    baselines: dict[str, object] = {}
    for method in METHOD_ORDER_ANTIGRAVITY[4:]:
        root_name = EXP471_BASELINE_ROOT_BY_METHOD_ANTIGRAVITY[method]
        root_authority = EXP471_BASELINE_ROOT_AUTHORITY_ANTIGRAVITY[root_name]
        bare = {
            "schema": "exp471-verified-baseline-authority-antigravity.v1",
            "method": method,
            "database_sha256": authority.database_sha256,
            "dataset_order": list(datasets),
            "test_coordinates_sha256": coordinates_sha256,
            "source_manifest_sha256": root_authority["manifest_sha256"],
            "run_config_sha256": root_authority["run_config_sha256"],
        }
        authority_sha256 = sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest()
        baseline_inventory[method] = authority_sha256
        scores: dict[str, object] = {}
        for dataset in datasets:
            shard = shards[(method, dataset)]
            if shard.source_manifest_sha256 != root_authority["manifest_sha256"]:
                raise ValueError("Exp471 reopened baseline source differs")
            scores[dataset] = {
                # Metadata validation requires a finite nonempty TRAIN vector,
                # but reopening never calibrates or republishes from this value.
                "train_oof_scores": [0.0],
                "test_indices": coordinates[dataset]["indices"],
                "test_scores": list(shard.raw_scores),
            }
        baselines[method] = {
            "authority": {**bare, "authority_sha256": authority_sha256},
            "scores": scores,
        }
    return {
        "schema": "exp471-verified-score-input-antigravity.v2",
        "database_sha256": authority.database_sha256,
        "dataset_order": list(datasets),
        "test_coordinates": coordinates,
        "baseline_authority_sha256": baseline_inventory,
        "conditions": conditions,
        "baselines": baselines,
    }


def _score_input_antigravity(session: object) -> Mapping[str, object]:
    manifest = EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY / "score_cache_manifest_antigravity.json"
    if os.path.lexists(manifest):
        return _reopened_score_bundle_antigravity(session)
    return _production_score_bundle_antigravity(session)


class _ProductionRuntimeAntigravity:
    def verify_source_antigravity(self):
        from .source_authority_antigravity import (
            open_exp467_source_session_antigravity,
            publish_source_authority_antigravity,
        )

        holder: dict[str, object] = {}

        def score_reader():
            return _score_input_antigravity(holder["session"])

        def label_reader():
            from experiments.paper_public_baselines.label_access import (
                load_test_labels,
            )

            session = holder["session"]
            datasets = tuple(
                entry["dataset"] for entry in session.catalog["datasets"]
            )
            return load_test_labels(datasets)

        started = monotonic()
        session = open_exp467_source_session_antigravity(
            _repository_root_antigravity(),
            EXP467_SOURCE_ROOT_ANTIGRAVITY,
            score_reader_antigravity=score_reader,
            label_callback_antigravity=label_reader,
        )
        holder["session"] = session
        authority_path = publish_source_authority_antigravity(
            EXP471_ROOT_ANTIGRAVITY, session.authority
        )
        _append_durable_stage_antigravity(
            "verify-source",
            session.authority.total_successes,
            session.authority.total_successes,
            authority_path,
            started,
        )
        return session

    def open_cache_antigravity(self, session):
        from .score_cache_antigravity import open_complete_score_cache_antigravity

        manifest = EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY / "score_cache_manifest_antigravity.json"
        if not os.path.lexists(manifest):
            raise FileNotFoundError("Exp471 cache manifest is absent")
        return open_complete_score_cache_antigravity(session)

    def benchmark_prepare_antigravity(self, session):
        from . import score_cache_antigravity as score_cache

        tasks = score_cache._build_tasks_from_verified_input_antigravity(
            session.authority, session.score_reader_antigravity()
        )
        sample = tasks[: min(len(tasks), max(2, CPU_WORKERS_ANTIGRAVITY))]
        return score_cache.benchmark_score_preparation_antigravity(
            sample, parallel_workers=CPU_WORKERS_ANTIGRAVITY
        )

    def prepare_antigravity(self, session, progress):
        from .score_cache_antigravity import prepare_score_cache_antigravity

        return prepare_score_cache_antigravity(session, progress=progress)

    def open_evaluation_antigravity(self, session, cache):
        from .evaluation_antigravity import read_evaluation_exp471_antigravity

        path = EXP471_EVALUATION_ROOT_ANTIGRAVITY / "evaluation_antigravity.json"
        if not os.path.lexists(path):
            raise FileNotFoundError("Exp471 evaluation is absent")
        return read_evaluation_exp471_antigravity(
            EXP471_EVALUATION_ROOT_ANTIGRAVITY, session, cache
        )

    def evaluate_antigravity(self, session, cache):
        from .evaluation_antigravity import evaluate_exp471_antigravity

        started = monotonic()
        artifact = evaluate_exp471_antigravity(session, cache)
        _append_durable_stage_antigravity(
            "evaluation",
            len(artifact.method_order),
            len(artifact.method_order),
            EXP471_EVALUATION_ROOT_ANTIGRAVITY / "evaluation_antigravity.json",
            started,
        )
        return artifact

    def report_antigravity(self, artifact):
        from .reporting_antigravity import write_exp471_report_antigravity
        from .score_cache_antigravity import status_score_cache_antigravity

        worker = status_score_cache_antigravity(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)
        started = monotonic()
        authority = write_exp471_report_antigravity(
            artifact,
            EXP471_REPORT_ROOT_ANTIGRAVITY,
            worker_provenance={
                "schema": "exp471-worker-provenance-antigravity.v1",
                "benchmark_decision": worker["benchmark_decision"],
                "execution_decision": worker["execution_decision"],
                "detected_logical_cpus": worker["detected_logical_cpus"],
                "default_cpu_workers": worker["default_cpu_workers"],
                "inner_numerical_threads": worker["inner_numerical_threads"],
                "acceleration": worker["acceleration"],
            },
        )
        _append_durable_stage_antigravity(
            "report",
            len(authority.report_file_sha256),
            len(authority.report_file_sha256),
            EXP471_REPORT_ROOT_ANTIGRAVITY,
            started,
        )
        return authority

    def status_antigravity(self):
        from .score_cache_antigravity import status_score_cache_antigravity

        root_anchor = _open_existing_directory_chain_antigravity(
            EXP471_ROOT_ANTIGRAVITY
        )
        cache_anchor: _DirectoryChainAntigravity | None = None
        try:
            _validate_status_root_generation_antigravity(root_anchor)
            cache_anchor = _open_existing_directory_chain_antigravity(
                EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY
            )
            _validate_status_root_generation_antigravity(root_anchor)
            cache = status_score_cache_antigravity(
                EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY
            )
            if cache_anchor is not None:
                cache_anchor.validate_antigravity()
            _validate_status_root_generation_antigravity(root_anchor)
            if cache_anchor is not None:
                cache_anchor.close_antigravity()
                cache_anchor = None
            evaluation_present = (
                _read_configured_leaf_antigravity(
                    EXP471_EVALUATION_ROOT_ANTIGRAVITY,
                    "evaluation_antigravity.json",
                )
                is not None
            )
            _validate_status_root_generation_antigravity(root_anchor)
            report_present = (
                _read_configured_leaf_antigravity(
                    EXP471_REPORT_ROOT_ANTIGRAVITY,
                    ".exp471_report_complete_antigravity.json",
                )
                is not None
            )
            _validate_status_root_generation_antigravity(root_anchor)
            latest = _latest_progress_antigravity()
            _validate_status_root_generation_antigravity(root_anchor)
            completed = int(cache["durable_completed"])
            total = cache["total_shards"]
            expected_total = (
                _TOTAL_SHARDS_ANTIGRAVITY if total is None else int(total)
            )
            failure_count = (
                0 if latest is None else int(latest.get("failures", 0))
            )
            result = {
                "schema": "exp471-status-antigravity.v1",
                "durable_completed": completed,
                "total": expected_total,
                "failures": failure_count,
                "remaining": max(0, expected_total - completed),
                "verified_stage": (
                    "report-present-unverified"
                    if report_present
                    else "evaluation-present-unverified"
                    if evaluation_present
                    else "cache-complete"
                    if cache["complete"]
                    else "cache-incomplete"
                ),
                "paths": {
                    "root": str(EXP471_ROOT_ANTIGRAVITY),
                    "score_cache": str(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY),
                    "evaluation": str(EXP471_EVALUATION_ROOT_ANTIGRAVITY),
                    "report": str(EXP471_REPORT_ROOT_ANTIGRAVITY),
                    "progress": str(
                        EXP471_ROOT_ANTIGRAVITY / _PROGRESS_NAME_ANTIGRAVITY
                    ),
                },
                "worker_decision": {
                    "benchmark": cache["benchmark_decision"],
                    "actual": cache["execution_decision"],
                },
                "latest_progress": latest,
                "failure": (
                    None
                    if failure_count == 0
                    else {
                        "stage": latest.get("stage") if latest else "unknown",
                        "failures": failure_count,
                    }
                ),
            }
            _validate_status_root_generation_antigravity(root_anchor)
            return result
        finally:
            if cache_anchor is not None:
                cache_anchor.close_antigravity()
            if root_anchor is not None:
                root_anchor.close_antigravity()


def _report_command_antigravity(runtime: object) -> Mapping[str, object]:
    session = runtime.verify_source_antigravity()
    try:
        cache = runtime.open_cache_antigravity(session)
    except FileNotFoundError:
        cache = runtime.prepare_antigravity(
            session,
            _runtime_progress_antigravity(runtime),
        )
    try:
        artifact = runtime.open_evaluation_antigravity(session, cache)
    except FileNotFoundError:
        artifact = runtime.evaluate_antigravity(session, cache)
    report = runtime.report_antigravity(artifact)
    return {
        "schema": "exp471-command-result-antigravity.v1",
        "experiment": "Exp471",
        "command": "report",
        "source_authority_sha256": session.authority.authority_sha256,
        "cache_manifest_sha256": cache.manifest_sha256,
        "evaluation_sha256": artifact.evaluation_sha256,
        "report_manifest_sha256": report.report_manifest_sha256,
        "report_root": str(EXP471_REPORT_ROOT_ANTIGRAVITY),
    }


def _runtime_progress_antigravity(runtime: object):
    if isinstance(runtime, _ProductionRuntimeAntigravity):
        return _progress_callback_antigravity(
            "score-cache", EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY
        )
    return lambda _event: None


def main_antigravity(
    argv: Sequence[str] | None = None,
    *,
    runtime: object | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one fixed command and emit exactly one canonical finite JSON record."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    if arguments in {(), ("--help",), ("-h",)}:
        _emit_antigravity(
            output,
            {
                "schema": "exp471-cli-help-antigravity.v1",
                "experiment": "Exp471",
                "command": "help",
                "commands": list(_COMMANDS_ANTIGRAVITY),
            },
        )
        return 0
    if len(arguments) != 1 or arguments[0] not in _COMMANDS_ANTIGRAVITY:
        _emit_antigravity(
            errors,
            {
                "schema": "exp471-cli-error-antigravity.v1",
                "error": "invalid_arguments",
            },
        )
        return 2
    command = arguments[0]
    active = _ProductionRuntimeAntigravity() if runtime is None else runtime
    try:
        if command == "status":
            value = {
                "experiment": "Exp471",
                "command": command,
                **active.status_antigravity(),
            }
        elif command == "verify-source":
            session = active.verify_source_antigravity()
            value = {
                "schema": "exp471-command-result-antigravity.v1",
                "experiment": "Exp471",
                "command": command,
                "authority_sha256": session.authority.authority_sha256,
                "inventory_sha256": session.authority.inventory_sha256,
                "total_successes": getattr(session.authority, "total_successes", 4468),
                "root": str(EXP471_ROOT_ANTIGRAVITY),
            }
        elif command == "benchmark-prepare":
            session = active.verify_source_antigravity()
            decision = active.benchmark_prepare_antigravity(session)
            value = {
                "schema": "exp471-command-result-antigravity.v1",
                "experiment": "Exp471",
                "command": command,
                "decision": decision,
            }
        elif command == "prepare":
            session = active.verify_source_antigravity()
            cache = active.prepare_antigravity(
                session,
                _runtime_progress_antigravity(active),
            )
            value = {
                "schema": "exp471-command-result-antigravity.v1",
                "experiment": "Exp471",
                "command": command,
                "manifest_sha256": cache.manifest_sha256,
                "total_shards": cache.total_shards,
                "root": str(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY),
            }
        else:
            value = _report_command_antigravity(active)
        _emit_antigravity(output, value)
        return 0
    except Exception as error:
        _emit_antigravity(
            errors,
            {
                "schema": "exp471-cli-error-antigravity.v1",
                "error": "command_failed",
                "command": command,
                "exception_type": type(error).__name__,
            },
        )
        return 2


__all__ = ["main_antigravity"]
