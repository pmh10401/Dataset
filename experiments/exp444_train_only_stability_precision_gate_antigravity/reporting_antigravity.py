"""Create-only, label-free Exp444 stability-lane publication and inspection.

The checkpoint is the prediction authority.  Publishing may re-open only the
already approved values-only source boundary to reproduce the original-control
replay; inspection deliberately never imports that boundary (or any runtime or
evaluation entrypoint), opens no database, and mutates no attempt authority.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from uuid import uuid4

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    canonical_jsonl_bytes_antigravity,
    parse_canonical_json_bytes_antigravity,
)
from .config_antigravity import FROZEN_CONFIG, scientific_config_sha256_antigravity
from .contracts_antigravity import (
    BackendDecisionAntigravity,
    CandidateInputAntigravity,
    CandidateStabilityResultAntigravity,
    CheckpointContextAntigravity,
    DepthViewScoreAntigravity,
    SourceCandidateAntigravity,
    StabilitySelectionAntigravity,
    validate_prediction_payload_antigravity,
)


PUBLICATION_DIRECTORY_ANTIGRAVITY = "publication_antigravity"
DATASET_SUMMARY_FILE_ANTIGRAVITY = "dataset_summary_antigravity.jsonl"
SOURCE_CANDIDATES_FILE_ANTIGRAVITY = "source_candidates_antigravity.jsonl"
CANDIDATE_STABILITY_FILE_ANTIGRAVITY = "candidate_stability_antigravity.jsonl"
VIEW_EVIDENCE_FILE_ANTIGRAVITY = "view_evidence_antigravity.jsonl"
STABLE_REVIEW_FILE_ANTIGRAVITY = "stable_review_prediction_antigravity.jsonl"
UNSTABLE_REVIEW_FILE_ANTIGRAVITY = "unstable_review_prediction_antigravity.jsonl"
CONTROL_REPLAY_FILE_ANTIGRAVITY = "control_replay_antigravity.json"
SOURCE_AUDIT_FILE_ANTIGRAVITY = "source_audit_antigravity.json"
BACKEND_DECISION_FILE_ANTIGRAVITY = "backend_decision_antigravity.json"
RUNTIME_FILE_ANTIGRAVITY = "runtime_antigravity.json"
MANIFEST_FILE_ANTIGRAVITY = "manifest_antigravity.json"
STATUS_FILE_ANTIGRAVITY = "status_antigravity.json"
ARTIFACT_SHA256_FILE_ANTIGRAVITY = "artifact_sha256_antigravity.json"

_LEAF_FILES_ANTIGRAVITY = (
    DATASET_SUMMARY_FILE_ANTIGRAVITY,
    SOURCE_CANDIDATES_FILE_ANTIGRAVITY,
    CANDIDATE_STABILITY_FILE_ANTIGRAVITY,
    VIEW_EVIDENCE_FILE_ANTIGRAVITY,
    STABLE_REVIEW_FILE_ANTIGRAVITY,
    UNSTABLE_REVIEW_FILE_ANTIGRAVITY,
    CONTROL_REPLAY_FILE_ANTIGRAVITY,
    SOURCE_AUDIT_FILE_ANTIGRAVITY,
    BACKEND_DECISION_FILE_ANTIGRAVITY,
    RUNTIME_FILE_ANTIGRAVITY,
)
_PUBLICATION_FILES_ANTIGRAVITY = frozenset((*_LEAF_FILES_ANTIGRAVITY, MANIFEST_FILE_ANTIGRAVITY, STATUS_FILE_ANTIGRAVITY, ARTIFACT_SHA256_FILE_ANTIGRAVITY))
_RECORD_NAMES_ANTIGRAVITY = tuple(f"{index:04d}_record_antigravity.json" for index in range(FROZEN_CONFIG.source_candidate_count))
_PINNED_PUBLICATION_HOOK_ANTIGRAVITY = None  # deterministic test-only race seam
_STAGING_BEFORE_RENAME_HOOK_ANTIGRAVITY = None  # deterministic test-only race seam
_ATTEMPT_CAPTURE_HOOK_ANTIGRAVITY = None  # deterministic test-only race seam
_QUARANTINE_BEFORE_RENAME_HOOK_ANTIGRAVITY = None  # deterministic test-only race seam

_ATTEMPT_FILES_ANTIGRAVITY = ("progress_antigravity.json", "pid_antigravity.json", "events_antigravity.jsonl", "text_antigravity.log")
_CHECKPOINT_FILES_ANTIGRAVITY = ("context_antigravity.json", "state_antigravity.json")


@dataclass(frozen=True)
class _AttemptSnapshotAntigravity:
    attempt_name: str
    payloads: Mapping[str, bytes]
    attempt_fd: int
    attempt_identity: tuple[int, int, int, int, int]

    def close_antigravity(self) -> None:
        """Release the pinned attempt authority after publication finishes."""
        if self.attempt_fd >= 0:
            os.close(self.attempt_fd)
            object.__setattr__(self, "attempt_fd", -1)


@dataclass(frozen=True)
class _PublicationQuarantineAntigravity:
    """Non-destructive evidence for a rejected post-rename publication."""

    name: str | None
    parent_fsynced: bool
    error: str | None


def _sha256_antigravity(value: bytes) -> str:
    return sha256(value).hexdigest()


def _quarantine_name_antigravity(destination_name: str, nonce: str) -> str:
    return f".{destination_name}.{nonce}.quarantine_antigravity"


def _is_quarantine_name_antigravity(name: str, destination_name: str = PUBLICATION_DIRECTORY_ANTIGRAVITY) -> bool:
    prefix = f".{destination_name}."
    suffix = ".quarantine_antigravity"
    nonce = name.removeprefix(prefix).removesuffix(suffix)
    return name.startswith(prefix) and name.endswith(suffix) and len(nonce) == 32 and all(character in "0123456789abcdef" for character in nonce)


def _validated_attempt_quarantine_names_antigravity(attempt_fd: int, names: set[str]) -> set[str]:
    """Accept only directory-backed Exp444 quarantine names beside frozen inputs."""
    expected = set((*_ATTEMPT_FILES_ANTIGRAVITY, "checkpoint_antigravity"))
    quarantines = names - expected
    if names != expected | quarantines or any(not _is_quarantine_name_antigravity(name) for name in quarantines):
        raise ValueError("Exp444 attempt inventory is invalid")
    for name in quarantines:
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=attempt_fd)
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode):
                raise ValueError("Exp444 attempt quarantine is invalid")
        finally:
            os.close(descriptor)
    return quarantines


def _capture_attempt_antigravity(attempt: Path) -> _AttemptSnapshotAntigravity:
    """Capture the complete terminal authority through retained no-follow dirfds."""
    attempt_fd = os.open(attempt, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptors: list[int] = []
    try:
        attempt_status = os.fstat(attempt_fd)
        if not stat.S_ISDIR(attempt_status.st_mode):
            raise ValueError("Exp444 attempt authority is invalid")
        _validated_attempt_quarantine_names_antigravity(attempt_fd, set(os.listdir(attempt_fd)))
        checkpoint_fd = os.open("checkpoint_antigravity", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=attempt_fd); descriptors.append(checkpoint_fd)
        if set(os.listdir(checkpoint_fd)) != set((*_CHECKPOINT_FILES_ANTIGRAVITY, "records_antigravity")):
            raise ValueError("Exp444 checkpoint inventory is invalid")
        records_fd = os.open("records_antigravity", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=checkpoint_fd); descriptors.append(records_fd)
        if tuple(sorted(os.listdir(records_fd))) != _RECORD_NAMES_ANTIGRAVITY:
            raise ValueError("Exp444 checkpoint record inventory is invalid")
        payloads: dict[str, bytes] = {}
        identities: list[tuple[int, str, tuple[int, int, int, int, int]]] = []
        for directory_fd, names, prefix in ((attempt_fd, _ATTEMPT_FILES_ANTIGRAVITY, ""), (checkpoint_fd, _CHECKPOINT_FILES_ANTIGRAVITY, "checkpoint/"), (records_fd, _RECORD_NAMES_ANTIGRAVITY, "records/")):
            for name in names:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd); descriptors.append(fd)
                first = os.fstat(fd)
                if not stat.S_ISREG(first.st_mode) or first.st_nlink != 1:
                    raise ValueError("Exp444 attempt authority must be a single-linked regular file")
                raw = b"".join(iter(lambda: os.read(fd, 1024 * 1024), b""))
                last = os.fstat(fd); named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns, first.st_ctime_ns) != (last.st_dev, last.st_ino, last.st_size, last.st_mtime_ns, last.st_ctime_ns) or (first.st_dev, first.st_ino) != (named.st_dev, named.st_ino):
                    raise ValueError("Exp444 attempt authority changed during capture")
                payloads[prefix + name] = raw
                identities.append((directory_fd, name, (first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns, first.st_ctime_ns)))
        if _ATTEMPT_CAPTURE_HOOK_ANTIGRAVITY is not None:
            _ATTEMPT_CAPTURE_HOOK_ANTIGRAVITY(attempt)
        _validated_attempt_quarantine_names_antigravity(attempt_fd, set(os.listdir(attempt_fd)))
        if set(os.listdir(checkpoint_fd)) != set((*_CHECKPOINT_FILES_ANTIGRAVITY, "records_antigravity")) or tuple(sorted(os.listdir(records_fd))) != _RECORD_NAMES_ANTIGRAVITY:
            raise ValueError("Exp444 attempt inventory changed during capture")
        for directory_fd, name, identity in identities:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns):
                raise ValueError("Exp444 attempt entry changed during capture")
        return _AttemptSnapshotAntigravity(
            attempt.name,
            MappingProxyType(payloads),
            attempt_fd,
            (attempt_status.st_dev, attempt_status.st_ino, attempt_status.st_mode, attempt_status.st_mtime_ns, attempt_status.st_ctime_ns),
        )
    except BaseException:
        os.close(attempt_fd)
        raise
    finally:
        for fd in reversed(descriptors): os.close(fd)


def _assert_attempt_namespace_antigravity(attempt: Path, snapshot: _AttemptSnapshotAntigravity, *, allow_owned_mutation: bool = False) -> None:
    """Require the caller's attempt pathname to remain the captured authority."""
    try:
        descriptor = os.fstat(snapshot.attempt_fd)
        named = os.stat(attempt, follow_symlinks=False)
    except OSError as error:
        raise ValueError("Exp444 attempt namespace changed during publication") from error
    captured = snapshot.attempt_identity
    current = (descriptor.st_dev, descriptor.st_ino, descriptor.st_mode, descriptor.st_mtime_ns, descriptor.st_ctime_ns)
    namespace = (named.st_dev, named.st_ino, named.st_mode, named.st_mtime_ns, named.st_ctime_ns)
    if allow_owned_mutation:
        current = current[:3]
        namespace = namespace[:3]
        captured = captured[:3]
    if current != captured or namespace != captured or not stat.S_ISDIR(descriptor.st_mode):
        raise ValueError("Exp444 attempt namespace changed during publication")


def _read_canonical_json_bytes_antigravity(encoded: bytes, name: str) -> Mapping[str, object]:
    try:
        payload = parse_canonical_json_bytes_antigravity(encoded)
    except ValueError as error:
        raise ValueError(f"Exp444 {name} is invalid") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes_antigravity(payload) != encoded:
        raise ValueError(f"Exp444 {name} is not canonical")
    return dict(payload)


def _read_canonical_json_antigravity(path: Path, name: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Exp444 {name} is absent or invalid")
    return _read_canonical_json_bytes_antigravity(path.read_bytes(), name)


def _read_canonical_jsonl_bytes_antigravity(encoded: bytes, name: str) -> tuple[dict[str, object], ...]:
    if not encoded:
        return ()
    if not encoded.endswith(b"\n"):
        raise ValueError(f"Exp444 {name} is not canonical")
    rows: list[dict[str, object]] = []
    for line in encoded.splitlines():
        try:
            value = parse_canonical_json_bytes_antigravity(line + b"\n")
        except ValueError as error:
            raise ValueError(f"Exp444 {name} is invalid") from error
        if not isinstance(value, Mapping) or canonical_json_bytes_antigravity(value) != line + b"\n":
            raise ValueError(f"Exp444 {name} is not canonical")
        rows.append(dict(value))
    if canonical_jsonl_bytes_antigravity(rows) != encoded:
        raise ValueError(f"Exp444 {name} is not canonical")
    return tuple(rows)


def _read_canonical_jsonl_antigravity(path: Path, name: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Exp444 {name} is absent or invalid")
    return _read_canonical_jsonl_bytes_antigravity(path.read_bytes(), name)


@contextmanager
def _pinned_publication_payloads_antigravity(root: Path | None = None, *, root_fd: int | None = None):
    """Hold every publication leaf on one root descriptor through inspection."""
    if (root is None) == (root_fd is None):
        raise ValueError("Exp444 publication root authority is invalid")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW) if root_fd is None else os.dup(root_fd)
    descriptors: list[int] = []
    try:
        before = os.fstat(root_fd)
        expected = set(_PUBLICATION_FILES_ANTIGRAVITY)
        if set(os.listdir(root_fd)) != expected:
            raise ValueError("Exp444 publication inventory is invalid")
        payloads: dict[str, bytes] = {}
        identities: dict[str, tuple[int, int, int, int, int]] = {}
        for name in sorted(expected):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
            descriptors.append(fd)
            first = os.fstat(fd)
            if not stat.S_ISREG(first.st_mode) or first.st_nlink != 1:
                raise ValueError("Exp444 publication leaf must be a single-linked regular file")
            chunks: list[bytes] = []
            while block := os.read(fd, 1024 * 1024):
                chunks.append(block)
            payloads[name] = b"".join(chunks)
            last = os.fstat(fd)
            named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            identity = (first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns, first.st_ctime_ns)
            if identity != (last.st_dev, last.st_ino, last.st_size, last.st_mtime_ns, last.st_ctime_ns) or (first.st_dev, first.st_ino) != (named.st_dev, named.st_ino):
                raise ValueError("Exp444 publication leaf changed during pinned read")
            identities[name] = identity
        if _PINNED_PUBLICATION_HOOK_ANTIGRAVITY is not None:
            _PINNED_PUBLICATION_HOOK_ANTIGRAVITY(root)
        yield payloads
        if set(os.listdir(root_fd)) != expected:
            raise ValueError("Exp444 publication inventory changed during inspection")
        for name, identity in identities.items():
            named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns, named.st_ctime_ns) != identity:
                raise ValueError("Exp444 publication entry changed during inspection")
        after = os.fstat(root_fd)
        if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Exp444 publication root changed during inspection")
    finally:
        for fd in descriptors:
            os.close(fd)
        os.close(root_fd)


def _ensure_label_free_antigravity(value: object, name: str) -> None:
    """Apply the prediction contract to scientific rows, never governance."""
    try:
        validate_prediction_payload_antigravity(value)
    except ValueError as error:
        raise ValueError(f"Exp444 {name} contains a forbidden prediction field") from error


@dataclass(frozen=True)
class FrozenPublicationAntigravity:
    """Immutable publication handoff for the later, separately gated evaluator."""

    publication_root: Path
    manifest: Mapping[str, object]
    selection: StabilitySelectionAntigravity

    def __post_init__(self) -> None:
        root = Path(self.publication_root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Exp444 publication root is absent or invalid")
        manifest = dict(self.manifest)
        if type(manifest.get("label_access_count")) is not int or manifest["label_access_count"] != 0:
            raise ValueError("Exp444 publication label governance is invalid")
        if not isinstance(self.selection, StabilitySelectionAntigravity):
            raise ValueError("Exp444 publication selection is invalid")
        object.__setattr__(self, "publication_root", root)
        object.__setattr__(self, "manifest", MappingProxyType(manifest))

    @property
    def stable_count(self) -> int:
        return self.selection.stable_count

    @property
    def unstable_count(self) -> int:
        return self.selection.unstable_count

    @property
    def source_coordinates(self) -> frozenset[tuple[str, int]]:
        return frozenset(item.coordinate for item in self.selection.source)

    @property
    def stable_coordinates(self) -> frozenset[tuple[str, int]]:
        return frozenset(item.coordinate for item in self.selection.stable)

    @property
    def unstable_coordinates(self) -> frozenset[tuple[str, int]]:
        return frozenset(item.coordinate for item in self.selection.unstable)

    @property
    def label_access_count(self) -> int:
        return 0

    @property
    def prediction_sha256(self) -> str:
        value = self.manifest.get("prediction_sha256")
        if type(value) is not str:
            raise ValueError("Exp444 publication prediction SHA256 is invalid")
        return value


@dataclass(frozen=True)
class _PublishSourceAuthorityAntigravity:
    candidates: tuple[SourceCandidateAntigravity, ...]
    dataset_summaries: tuple[dict[str, object], ...]
    source_hashes: Mapping[str, str]
    control_rows: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        if len(self.candidates) != FROZEN_CONFIG.source_candidate_count:
            raise ValueError("Exp444 source authority candidate count is invalid")
        if len(self.dataset_summaries) != FROZEN_CONFIG.dataset_count:
            raise ValueError("Exp444 dataset summary count is invalid")
        if any(not isinstance(item, SourceCandidateAntigravity) for item in self.candidates):
            raise ValueError("Exp444 source authority candidates are invalid")
        names = tuple(row.get("dataset") for row in self.dataset_summaries)
        if any(type(name) is not str or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Exp444 dataset summaries are invalid")
        if tuple(sorted(names)) != names:
            raise ValueError("Exp444 dataset summaries are not in canonical order")
        for value, name in ((self.dataset_summaries, "dataset summary"), (self.control_rows, "control replay")):
            _ensure_label_free_antigravity(value, name)
        if len(self.control_rows) != FROZEN_CONFIG.source_candidate_count:
            raise ValueError("Exp444 original-control replay count is invalid")


def _checkpoint_context_antigravity() -> CheckpointContextAntigravity:
    return CheckpointContextAntigravity(
        FROZEN_CONFIG.experiment_id,
        FROZEN_CONFIG.run_identity,
        scientific_config_sha256_antigravity(),
        FROZEN_CONFIG.environment_lock_sha256,
        FROZEN_CONFIG.source_candidate_count,
        FROZEN_CONFIG.cpu_workers,
        FROZEN_CONFIG.inner_threads,
    )


def _complete_checkpoint_results_antigravity(attempt_root: Path, snapshot: _AttemptSnapshotAntigravity | None = None) -> tuple[CandidateStabilityResultAntigravity, ...]:
    attempt = Path(attempt_root)
    owns_snapshot = snapshot is None
    snapshot = _capture_attempt_antigravity(attempt) if snapshot is None else snapshot
    try:
        _assert_attempt_namespace_antigravity(attempt, snapshot)
        context = _read_canonical_json_bytes_antigravity(snapshot.payloads["checkpoint/context_antigravity.json"], "checkpoint context")
        if context != _checkpoint_context_antigravity().to_payload():
            raise ValueError("Exp444 checkpoint context changed")
        state = _read_canonical_json_bytes_antigravity(snapshot.payloads["checkpoint/state_antigravity.json"], "checkpoint state")
        expected_state = {"context_sha256", "status", "completed_units", "total_units", "error_count", "current_unit", "current_view", "payload_sha256", "record_sha256_prefix", "process", "error", "updated_at"}
        if set(state) != expected_state or state["status"] != "complete" or state["completed_units"] != 30 or state["total_units"] != 30 or state["error_count"] != 0 or state["current_unit"] is not None or state["current_view"] is not None or state["error"] is not None:
            raise ValueError("Exp444 checkpoint is not a complete inactive 30-record authority")
        progress = _read_canonical_json_bytes_antigravity(snapshot.payloads["progress_antigravity.json"], "progress")
        pid = _read_canonical_json_bytes_antigravity(snapshot.payloads["pid_antigravity.json"], "PID")
        if progress.get("stage") != "complete" or progress.get("active") is not False or progress.get("completed") != 30 or progress.get("total") != 30 or progress.get("backend") != "cpu" or pid.get("active") is not False:
            raise ValueError("Exp444 attempt is not complete and inactive")
        prefix = state["record_sha256_prefix"]
        if not isinstance(prefix, list) or len(prefix) != 30 or any(type(item) is not str or len(item) != 64 for item in prefix):
            raise ValueError("Exp444 checkpoint record prefix is invalid")
        if state["payload_sha256"] != _sha256_antigravity(canonical_json_bytes_antigravity(prefix)):
            raise ValueError("Exp444 checkpoint prefix hash changed")
        results: list[CandidateStabilityResultAntigravity] = []
        for sequence, name in enumerate(_RECORD_NAMES_ANTIGRAVITY):
            encoded = snapshot.payloads[f"records/{name}"]
            payload = _read_canonical_json_bytes_antigravity(encoded, f"checkpoint record {sequence}")
            if set(payload) != {"sequence", "result"} or payload["sequence"] != sequence or not isinstance(payload["result"], Mapping):
                raise ValueError("Exp444 checkpoint record is invalid")
            if _sha256_antigravity(encoded) != prefix[sequence]:
                raise ValueError("Exp444 checkpoint record hash changed")
            try:
                result = CandidateStabilityResultAntigravity.from_payload(payload["result"])
            except ValueError as error:
                raise ValueError("Exp444 checkpoint stability result is invalid") from error
            if result.candidate.original_rank != sequence + 1:
                raise ValueError("Exp444 checkpoint rank order changed")
            results.append(result)
        return tuple(results)
    finally:
        if owns_snapshot:
            snapshot.close_antigravity()


def _exp442_dataset_summaries_antigravity() -> tuple[dict[str, object], ...]:
    """Read the already frozen Exp442 summary, never an unbounded database query."""
    source = Path(FROZEN_CONFIG.exp442_prediction_path).with_name(DATASET_SUMMARY_FILE_ANTIGRAVITY)
    rows = _read_canonical_jsonl_antigravity(source, "frozen source dataset summary")
    return rows


def _publish_source_authority_antigravity() -> _PublishSourceAuthorityAntigravity:
    """Lazy values-only source replay used only before first publication creation."""
    from .sources_antigravity import load_candidate_values_antigravity, verify_frozen_sources_antigravity
    from .stability_antigravity import replay_original_candidates_antigravity

    snapshot = verify_frozen_sources_antigravity()
    session = snapshot.session
    try:
        inputs = load_candidate_values_antigravity(snapshot)
        controls = replay_original_candidates_antigravity(inputs)
        if snapshot.label_access_count != 0 or getattr(session, "label_access_count", None) != 0:
            raise ValueError("Exp444 publication source replay accessed labels")
        if tuple(item.candidate for item in inputs) != snapshot.candidates or tuple(item.candidate for item in controls) != snapshot.candidates:
            raise ValueError("Exp444 original-control source coordinates changed")
        control_rows = tuple(
            {
                "source": control.source.to_payload(),
                "depth_scores": [score.to_payload() for score in control.depth_scores],
                "replayed_depths": list(control.replayed_depths),
                "replayed_candidate": control.replayed_candidate.to_payload(),
                "matches_source": control.matches_source,
                "replayed_rank": control.replayed_rank,
            }
            for control in controls
        )
        return _PublishSourceAuthorityAntigravity(snapshot.candidates, _exp442_dataset_summaries_antigravity(), dict(snapshot.source_hashes), control_rows)
    finally:
        if session is not None:
            session.close()


def _result_rows_antigravity(results: Sequence[CandidateStabilityResultAntigravity]) -> tuple[dict[str, object], ...]:
    rows = tuple(item.to_payload() for item in results)
    _ensure_label_free_antigravity(rows, "candidate stability")
    return rows


def _view_rows_antigravity(results: Sequence[CandidateStabilityResultAntigravity]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for result in results:
        for view in result.views:
            rows.append({"dataset": result.candidate.dataset, "instance_index": result.candidate.instance_index, "original_rank": result.candidate.original_rank, "lane": result.lane, "view": view.to_payload()})
    _ensure_label_free_antigravity(rows, "view evidence")
    return tuple(rows)


def _source_audit_payload_antigravity(authority: _PublishSourceAuthorityAntigravity) -> dict[str, object]:
    return {"schema": "exp444.source-audit.v1", "source_hashes": dict(sorted(authority.source_hashes.items())), "source_candidate_count": 30, "dataset_count": 1117, "label_access_count": 0}


def _terminal_authority_surface_antigravity(path: Path, name: str) -> dict[str, object]:
    """Capture one terminal authority verbatim, with an in-leaf integrity seal."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Exp444 terminal {name} is absent or invalid")
    encoded = path.read_bytes()
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Exp444 terminal {name} is not UTF-8") from error
    return {"byte_size": len(encoded), "sha256": _sha256_antigravity(encoded), "utf8": text}


def _runtime_payload_antigravity(attempt: Path, results: Sequence[CandidateStabilityResultAntigravity], snapshot: _AttemptSnapshotAntigravity) -> dict[str, object]:
    state = _read_canonical_json_bytes_antigravity(snapshot.payloads["checkpoint/state_antigravity.json"], "checkpoint state")
    authorities = {
        "context": {"byte_size": len(snapshot.payloads["checkpoint/context_antigravity.json"]), "sha256": _sha256_antigravity(snapshot.payloads["checkpoint/context_antigravity.json"]), "utf8": snapshot.payloads["checkpoint/context_antigravity.json"].decode()},
        "state": {"byte_size": len(snapshot.payloads["checkpoint/state_antigravity.json"]), "sha256": _sha256_antigravity(snapshot.payloads["checkpoint/state_antigravity.json"]), "utf8": snapshot.payloads["checkpoint/state_antigravity.json"].decode()},
        "progress": {"byte_size": len(snapshot.payloads["progress_antigravity.json"]), "sha256": _sha256_antigravity(snapshot.payloads["progress_antigravity.json"]), "utf8": snapshot.payloads["progress_antigravity.json"].decode()},
        "pid": {"byte_size": len(snapshot.payloads["pid_antigravity.json"]), "sha256": _sha256_antigravity(snapshot.payloads["pid_antigravity.json"]), "utf8": snapshot.payloads["pid_antigravity.json"].decode()},
        "events": {"byte_size": len(snapshot.payloads["events_antigravity.jsonl"]), "sha256": _sha256_antigravity(snapshot.payloads["events_antigravity.jsonl"]), "utf8": snapshot.payloads["events_antigravity.jsonl"].decode()},
        "text": {"byte_size": len(snapshot.payloads["text_antigravity.log"]), "sha256": _sha256_antigravity(snapshot.payloads["text_antigravity.log"]), "utf8": snapshot.payloads["text_antigravity.log"].decode()},
    }
    records = [
        {"sequence": sequence, "byte_size": len(snapshot.payloads[f"records/{name}"]), "sha256": _sha256_antigravity(snapshot.payloads[f"records/{name}"]), "utf8": snapshot.payloads[f"records/{name}"].decode()}
        for sequence, name in enumerate(_RECORD_NAMES_ANTIGRAVITY)
    ]
    return {"schema": "exp444.runtime.v2", "attempt_name": attempt.name, "backend": "cpu", "workers": 11, "completed": len(results), "record_sha256_prefix": list(state["record_sha256_prefix"]), "authorities": authorities, "records": records, "label_access_count": 0}


def _backend_payload_antigravity() -> dict[str, object]:
    return BackendDecisionAntigravity("cpu", 11, (), "cpu_fallback_until_parity_speed_gate").to_payload()


def _parse_control_replays_antigravity(rows: object, source: Sequence[SourceCandidateAntigravity]) -> tuple[object, ...]:
    """Reconstitute the complete OriginalReplay contract and recompute ranks."""
    if not isinstance(rows, list) or len(rows) != 30:
        raise ValueError("Exp444 original-control replay count is invalid")
    from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.contracts_antigravity import ProvisionalCandidateAntigravity
    from .stability_antigravity import OriginalReplayAntigravity, verify_original_replay_ranks_antigravity

    expected_keys = {"source", "depth_scores", "replayed_depths", "replayed_candidate", "matches_source", "replayed_rank"}
    replays: list[OriginalReplayAntigravity] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_keys or not isinstance(row["source"], Mapping) or not isinstance(row["depth_scores"], list) or not isinstance(row["replayed_depths"], list) or not isinstance(row["replayed_candidate"], Mapping):
            raise ValueError("Exp444 original-control replay is invalid")
        try:
            replay = OriginalReplayAntigravity(
                SourceCandidateAntigravity.from_payload(row["source"]),
                tuple(DepthViewScoreAntigravity.from_payload(score) for score in row["depth_scores"]),
                tuple(row["replayed_depths"]),
                ProvisionalCandidateAntigravity.from_payload(row["replayed_candidate"]),
                row["matches_source"],
                row["replayed_rank"],
            )
        except ValueError as error:
            raise ValueError("Exp444 original-control replay is invalid") from error
        candidate_scores = tuple(score.to_payload() for score in replay.replayed_candidate.depth_scores)
        if tuple(score.to_payload() for score in replay.depth_scores) != candidate_scores or replay.source.coordinate != replay.replayed_candidate.coordinate or replay.source.raw_length != replay.replayed_candidate.raw_length or replay.replayed_depths != replay.source.original_depths or replay.replayed_candidate.depths != replay.source.original_depths or replay.replayed_candidate.agreement_count != len(replay.replayed_depths) or replay.matches_source is not True:
            raise ValueError("Exp444 original-control replay is contradictory")
        agreeing = tuple(score for score in replay.depth_scores if score.exceeds)
        if tuple(score.depth for score in agreeing) != replay.replayed_depths or len(agreeing) < 2:
            raise ValueError("Exp444 original-control agreeing depths changed")
        ordered = tuple(sorted(agreeing, key=lambda score: (-score.excess_ratio, score.empirical_rank, score.depth)))
        if replay.replayed_candidate.second_excess_ratio != ordered[1].excess_ratio or replay.replayed_candidate.second_empirical_rank != ordered[1].empirical_rank:
            raise ValueError("Exp444 original-control scalar evidence changed")
        replays.append(replay)
    if tuple(item.source for item in replays) != tuple(source) or any(not item.matches_source or item.replayed_candidate.coordinate != item.source.coordinate or item.replayed_depths != item.source.original_depths for item in replays):
        raise ValueError("Exp444 original-control parity failed")
    try:
        verified = verify_original_replay_ranks_antigravity(tuple(replays))
    except ValueError as error:
        raise ValueError("Exp444 original-control rank verification failed") from error
    if tuple(item.replayed_rank for item in replays) != tuple(item.replayed_rank for item in verified):
        raise ValueError("Exp444 original-control rank changed")
    return tuple(replays)


def _payloads_antigravity(attempt: Path, authority: _PublishSourceAuthorityAntigravity, snapshot: _AttemptSnapshotAntigravity, *, summary_bytes: bytes | None = None, backend_authority: Mapping[str, object] | None = None) -> tuple[dict[str, bytes], StabilitySelectionAntigravity]:
    results = _complete_checkpoint_results_antigravity(attempt, snapshot)
    source = tuple(authority.candidates)
    if tuple(item.candidate for item in results) != source:
        raise ValueError("Exp444 checkpoint coordinates differ from frozen source")
    stable = tuple(item for item in results if item.lane == "stable_review")
    unstable = tuple(item for item in results if item.lane == "unstable_review")
    selection = StabilitySelectionAntigravity(source, stable, unstable)
    _parse_control_replays_antigravity(list(authority.control_rows), source)
    result_rows = _result_rows_antigravity(results)
    stable_rows = _result_rows_antigravity(stable)
    unstable_rows = _result_rows_antigravity(unstable)
    runtime_payload = _runtime_payload_antigravity(attempt, results, snapshot)
    # This is deliberately before staging: every terminal binding must reject
    # while the create-only destination is still absent.
    _inspect_terminal_runtime_antigravity(runtime_payload, selection)
    payloads: dict[str, bytes] = {
        DATASET_SUMMARY_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(authority.dataset_summaries) if summary_bytes is None else bytes(summary_bytes),
        SOURCE_CANDIDATES_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(item.to_payload() for item in source),
        CANDIDATE_STABILITY_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(result_rows),
        VIEW_EVIDENCE_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(_view_rows_antigravity(results)),
        STABLE_REVIEW_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(stable_rows),
        UNSTABLE_REVIEW_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(unstable_rows),
        CONTROL_REPLAY_FILE_ANTIGRAVITY: canonical_json_bytes_antigravity({"schema": "exp444.control-replay.v1", "rows": list(authority.control_rows)}),
        SOURCE_AUDIT_FILE_ANTIGRAVITY: canonical_json_bytes_antigravity(_source_audit_payload_antigravity(authority)),
        BACKEND_DECISION_FILE_ANTIGRAVITY: canonical_json_bytes_antigravity(_backend_payload_antigravity()) if backend_authority is None else canonical_json_bytes_antigravity({"schema": "exp444.backend-decision-wrapper.v1", "authority": dict(backend_authority), "decision": dict(backend_authority["decision"])}),
        RUNTIME_FILE_ANTIGRAVITY: canonical_json_bytes_antigravity(runtime_payload),
    }
    return payloads, selection


def _final_payloads_antigravity(leaf_payloads: Mapping[str, bytes], selection: StabilitySelectionAntigravity) -> dict[str, bytes]:
    if set(leaf_payloads) != set(_LEAF_FILES_ANTIGRAVITY):
        raise ValueError("Exp444 publication leaf inventory is invalid")
    leaf_hashes = {name: _sha256_antigravity(leaf_payloads[name]) for name in _LEAF_FILES_ANTIGRAVITY}
    prediction_sha = _sha256_antigravity(leaf_payloads[STABLE_REVIEW_FILE_ANTIGRAVITY] + leaf_payloads[UNSTABLE_REVIEW_FILE_ANTIGRAVITY])
    manifest = {"schema": "exp444.publication.v1", "experiment_id": "Exp444", "run_identity": FROZEN_CONFIG.run_identity, "source_candidate_count": 30, "dataset_count": 1117, "stable_count": selection.stable_count, "unstable_count": selection.unstable_count, "prediction_sha256": prediction_sha, "label_access_count": 0, "leaf_sha256": leaf_hashes}
    manifest_bytes = canonical_json_bytes_antigravity(manifest)
    status = {"schema": "exp444.status.v1", "status": "descriptive_complete_no_promotion", "review_only": True, "operational_alert_eligible": False, "scientific_winner_selected": False, "source_candidate_count": 30, "stable_count": selection.stable_count, "unstable_count": selection.unstable_count, "label_access_count": 0}
    status_bytes = canonical_json_bytes_antigravity(status)
    artifact_map = {"schema": "exp444.artifact-sha256.v1", "files": {**leaf_hashes, MANIFEST_FILE_ANTIGRAVITY: _sha256_antigravity(manifest_bytes), STATUS_FILE_ANTIGRAVITY: _sha256_antigravity(status_bytes)}}
    return {**leaf_payloads, MANIFEST_FILE_ANTIGRAVITY: manifest_bytes, STATUS_FILE_ANTIGRAVITY: status_bytes, ARTIFACT_SHA256_FILE_ANTIGRAVITY: canonical_json_bytes_antigravity(artifact_map)}


def _rename_no_replace_at_antigravity(parent_fd: int, source_name: str, destination_name: str) -> None:
    """Perform Darwin's dirfd-relative no-replace rename without reopening authority."""
    renameatx_np = ctypes.CDLL(None, use_errno=True).renameatx_np
    renameatx_np.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameatx_np.restype = ctypes.c_int
    if renameatx_np(parent_fd, os.fsencode(source_name), parent_fd, os.fsencode(destination_name), ctypes.c_uint(0x00000004)) != 0:
        code = ctypes.get_errno() or errno.EIO
        raise OSError(code, os.strerror(code), destination_name)


def _exclusive_rename_at_antigravity(parent_fd: int, source_name: str, destination_name: str) -> None:
    """Publish a staging name through Darwin's no-replace rename primitive."""
    _rename_no_replace_at_antigravity(parent_fd, source_name, destination_name)


def _exclusive_rename_aside_at_antigravity(parent_fd: int, source_name: str, quarantine_name: str) -> None:
    """Move a rejected destination aside with the same Darwin no-replace primitive."""
    _rename_no_replace_at_antigravity(parent_fd, source_name, quarantine_name)


def _quarantine_rejected_destination_antigravity(parent_fd: int, destination_name: str) -> _PublicationQuarantineAntigravity:
    """Preserve a contested destination under a unique no-replace quarantine name.

    No pathname deletion is attempted.  Darwin only provides pathname-based
    renameatx_np, so a same-user race can change the source before this call;
    the operation deliberately preserves whichever directory that atomic
    rename moves and leaves the canonical destination absent on success.
    """
    for _ in range(8):
        try:
            quarantine_name = _quarantine_name_antigravity(destination_name, uuid4().hex)
        except BaseException as error:
            return _PublicationQuarantineAntigravity(None, False, f"quarantine_name={type(error).__name__}")
        try:
            if _QUARANTINE_BEFORE_RENAME_HOOK_ANTIGRAVITY is not None:
                _QUARANTINE_BEFORE_RENAME_HOOK_ANTIGRAVITY(parent_fd, destination_name, quarantine_name)
            _exclusive_rename_aside_at_antigravity(parent_fd, destination_name, quarantine_name)
        except BaseException as error:
            if isinstance(error, OSError) and error.errno == errno.EEXIST:
                continue
            code = error.errno if isinstance(error, OSError) and error.errno is not None else type(error).__name__
            return _PublicationQuarantineAntigravity(None, False, f"rename_aside={code}")
        try:
            os.fsync(parent_fd)
        except BaseException as error:
            code = error.errno if isinstance(error, OSError) and error.errno is not None else type(error).__name__
            return _PublicationQuarantineAntigravity(quarantine_name, False, f"parent_fsync={code}")
        return _PublicationQuarantineAntigravity(quarantine_name, True, None)
    return _PublicationQuarantineAntigravity(None, False, "rename_aside=collision_limit")


def _add_quarantine_note_antigravity(error: BaseException, destination: Path, evidence: _PublicationQuarantineAntigravity, observed_identity: tuple[int, int] | None, staging_identity: tuple[int, int] | None) -> None:
    """Keep the original rejection while attaching bounded quarantine evidence."""
    identity_known = observed_identity is not None
    identity_match = "unknown" if not identity_known else str(observed_identity == staging_identity)
    if evidence.name is None:
        note = f"Exp444 rejected publication quarantine requires operator intervention; canonical={destination}; identity_known={identity_known}; identity_match={identity_match}; {evidence.error}"
    else:
        note = f"Exp444 rejected publication quarantined at {destination.parent / evidence.name}; identity_known={identity_known}; identity_match={identity_match}; parent_fsynced={evidence.parent_fsynced}; error={evidence.error or 'none'}"
    error.add_note(note)


def _write_staging_antigravity(destination: Path, payloads: Mapping[str, bytes], *, parent_fd: int | None = None) -> None:
    """Durably publish only through a pinned attempt descriptor when supplied."""
    staging = destination.with_name(f".{destination.name}.{uuid4().hex}.staging_antigravity")
    parent = destination.parent
    uses_pinned_parent = parent_fd is not None
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW) if parent_fd is None else os.dup(parent_fd)
    staging_fd = -1
    staging_identity: tuple[int, int] | None = None
    staging_children: dict[str, tuple[int, int, int, int, int]] = {}
    rename_succeeded = False
    published_identity: tuple[int, int] | None = None
    try:
        try:
            os.stat(staging.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("Exp444 publication staging path already exists")
        os.mkdir(staging.name, 0o700, dir_fd=parent_fd)
        staging_fd = os.open(staging.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        status = os.fstat(staging_fd)
        if not stat.S_ISDIR(status.st_mode) or stat.S_IMODE(status.st_mode) != 0o700:
            raise ValueError("Exp444 publication staging directory is invalid")
        staging_identity = (status.st_dev, status.st_ino)
        names = (*_LEAF_FILES_ANTIGRAVITY, MANIFEST_FILE_ANTIGRAVITY, STATUS_FILE_ANTIGRAVITY, ARTIFACT_SHA256_FILE_ANTIGRAVITY)
        for name in names:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=staging_fd)
            try:
                offset = 0
                while offset < len(payloads[name]):
                    written = os.write(fd, payloads[name][offset:])
                    if written == 0:
                        raise OSError("Exp444 publication staging write made no progress")
                    offset += written
                os.fsync(fd)
                child = os.fstat(fd)
                staging_children[name] = (child.st_dev, child.st_ino, child.st_size, child.st_mtime_ns, child.st_ctime_ns)
            finally:
                os.close(fd)
        os.fsync(staging_fd)
        with _pinned_publication_payloads_antigravity(root_fd=staging_fd) as staged:
            if staged != payloads:
                raise ValueError("Exp444 publication staging bytes changed")
        if _STAGING_BEFORE_RENAME_HOOK_ANTIGRAVITY is not None:
            # The seam is test-only.  Production publication keeps every I/O
            # operation below relative to the retained parent descriptor.
            _STAGING_BEFORE_RENAME_HOOK_ANTIGRAVITY(staging if not uses_pinned_parent else destination.parent / staging.name)
        current_staging = os.stat(staging.name, dir_fd=parent_fd, follow_symlinks=False)
        if staging_identity != (current_staging.st_dev, current_staging.st_ino):
            raise ValueError("Exp444 publication staging directory changed")
        for name, identity in staging_children.items():
            current = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
            if identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns):
                raise ValueError("Exp444 publication staging entry changed")
        with _pinned_publication_payloads_antigravity(root_fd=staging_fd) as staged:
            if staged != payloads:
                raise ValueError("Exp444 publication staging bytes changed")
        _exclusive_rename_at_antigravity(parent_fd, staging.name, destination.name)
        rename_succeeded = True
        os.fsync(parent_fd)
        published_fd = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            published = os.fstat(published_fd)
            published_identity = (published.st_dev, published.st_ino)
            if staging_identity != (published.st_dev, published.st_ino):
                raise ValueError("Exp444 publication destination changed")
            with _pinned_publication_payloads_antigravity(root_fd=published_fd) as published_payloads:
                if published_payloads != payloads:
                    raise ValueError("Exp444 publication destination bytes changed")
        finally:
            os.close(published_fd)
    except BaseException as error:
        if rename_succeeded:
            evidence = _quarantine_rejected_destination_antigravity(parent_fd, destination.name)
            _add_quarantine_note_antigravity(error, destination, evidence, published_identity, staging_identity)
        if staging_identity is not None:
            try:
                current = os.stat(staging.name, dir_fd=parent_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == staging_identity and stat.S_ISDIR(current.st_mode):
                    names = set(os.listdir(staging_fd)) if staging_fd >= 0 else set()
                    if names.issubset(set((*_LEAF_FILES_ANTIGRAVITY, MANIFEST_FILE_ANTIGRAVITY, STATUS_FILE_ANTIGRAVITY, ARTIFACT_SHA256_FILE_ANTIGRAVITY))):
                        for name in names:
                            os.unlink(name, dir_fd=staging_fd)
                        os.rmdir(staging.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)
        os.close(parent_fd)


def _parse_selection_antigravity(payloads: Mapping[str, bytes]) -> StabilitySelectionAntigravity:
    source = tuple(SourceCandidateAntigravity.from_payload(row) for row in _read_canonical_jsonl_bytes_antigravity(payloads[SOURCE_CANDIDATES_FILE_ANTIGRAVITY], "source candidates"))
    results = tuple(CandidateStabilityResultAntigravity.from_payload(row) for row in _read_canonical_jsonl_bytes_antigravity(payloads[CANDIDATE_STABILITY_FILE_ANTIGRAVITY], "candidate stability"))
    stable = tuple(CandidateStabilityResultAntigravity.from_payload(row) for row in _read_canonical_jsonl_bytes_antigravity(payloads[STABLE_REVIEW_FILE_ANTIGRAVITY], "stable review"))
    unstable = tuple(CandidateStabilityResultAntigravity.from_payload(row) for row in _read_canonical_jsonl_bytes_antigravity(payloads[UNSTABLE_REVIEW_FILE_ANTIGRAVITY], "unstable review"))
    if tuple(item.candidate for item in results) != source:
        raise ValueError("Exp444 publication lane ordering changed")
    expected_stable = tuple(item for item in results if item.lane == "stable_review")
    expected_unstable = tuple(item for item in results if item.lane == "unstable_review")
    if stable != expected_stable or unstable != expected_unstable:
        raise ValueError("Exp444 publication lane membership changed")
    return StabilitySelectionAntigravity(source, stable, unstable)


def _inspect_terminal_runtime_antigravity(runtime: Mapping[str, object], selection: StabilitySelectionAntigravity) -> None:
    """Validate verbatim terminal surfaces without reopening the attempt path."""
    required = {"schema", "attempt_name", "backend", "workers", "completed", "record_sha256_prefix", "authorities", "records", "label_access_count"}
    if set(runtime) != required or runtime["schema"] != "exp444.runtime.v2" or runtime["backend"] != "cpu" or runtime["workers"] != 11 or runtime["completed"] != 30 or runtime["label_access_count"] != 0:
        raise ValueError("Exp444 publication runtime evidence is invalid")
    prefix = runtime["record_sha256_prefix"]
    authorities = runtime["authorities"]
    if not isinstance(prefix, list) or len(prefix) != 30 or any(type(item) is not str or len(item) != 64 for item in prefix) or not isinstance(authorities, Mapping) or set(authorities) != {"context", "state", "progress", "pid", "events", "text"}:
        raise ValueError("Exp444 publication runtime evidence is invalid")
    raw: dict[str, bytes] = {}
    for name, surface in authorities.items():
        if not isinstance(surface, Mapping) or set(surface) != {"byte_size", "sha256", "utf8"} or type(surface["byte_size"]) is not int or surface["byte_size"] < 0 or type(surface["sha256"]) is not str or type(surface["utf8"]) is not str:
            raise ValueError("Exp444 publication terminal authority surface is invalid")
        encoded = surface["utf8"].encode("utf-8")
        if len(encoded) != surface["byte_size"] or _sha256_antigravity(encoded) != surface["sha256"]:
            raise ValueError("Exp444 publication terminal authority surface changed")
        raw[name] = encoded
    context = parse_canonical_json_bytes_antigravity(raw["context"])
    state = parse_canonical_json_bytes_antigravity(raw["state"])
    progress = parse_canonical_json_bytes_antigravity(raw["progress"])
    pid = parse_canonical_json_bytes_antigravity(raw["pid"])
    if not all(isinstance(value, Mapping) for value in (context, state, progress, pid)) or dict(context) != _checkpoint_context_antigravity().to_payload() or state.get("context_sha256") != _sha256_antigravity(raw["context"]) or state.get("status") != "complete" or state.get("completed_units") != 30 or state.get("total_units") != 30 or state.get("error_count") != 0 or state.get("current_unit") is not None or state.get("current_view") is not None or state.get("error") is not None:
        raise ValueError("Exp444 publication terminal checkpoint state is invalid")
    if state.get("record_sha256_prefix") != prefix or state.get("payload_sha256") != _sha256_antigravity(canonical_json_bytes_antigravity(prefix)):
        raise ValueError("Exp444 publication terminal checkpoint prefix changed")
    if progress.get("stage") != "complete" or progress.get("active") is not False or progress.get("completed") != 30 or progress.get("total") != 30 or progress.get("backend") != "cpu" or pid.get("active") is not False:
        raise ValueError("Exp444 publication terminal progress is invalid")
    identity = (progress.get("pid"), progress.get("start_identity"))
    if type(identity[0]) is not int or type(identity[1]) is not str or not identity[1] or (pid.get("pid"), pid.get("start_identity")) != identity or not isinstance(state.get("process"), Mapping) or (state["process"].get("pid"), state["process"].get("start_identity")) != identity:
        raise ValueError("Exp444 publication terminal process identity is invalid")
    event_lines = raw["events"].splitlines(keepends=True)
    if not event_lines:
        raise ValueError("Exp444 publication terminal events are invalid")
    try:
        events = [parse_canonical_json_bytes_antigravity(line) for line in event_lines]
    except ValueError as error:
        raise ValueError("Exp444 publication terminal events are invalid") from error
    if any(not isinstance(event, Mapping) or canonical_json_bytes_antigravity(event) != line for event, line in zip(events, event_lines, strict=True)):
        raise ValueError("Exp444 publication terminal events are not canonical")
    terminal = events[-1]
    if terminal.get("event") != "terminal" or terminal.get("stage") != "complete" or terminal.get("pid") != identity[0] or terminal.get("completed") != 30 or terminal.get("total") != 30 or terminal.get("latest_error") != progress.get("latest_error"):
        raise ValueError("Exp444 publication terminal event is invalid")
    if not raw["text"].endswith(b"\n") or "종료 stage=complete" not in raw["text"].decode("utf-8").splitlines()[-1]:
        raise ValueError("Exp444 publication terminal text is invalid")
    records = runtime["records"]
    if not isinstance(records, list) or len(records) != 30:
        raise ValueError("Exp444 publication terminal records are invalid")
    reconstructed: list[CandidateStabilityResultAntigravity] = []
    for sequence, surface in enumerate(records):
        if not isinstance(surface, Mapping) or set(surface) != {"sequence", "byte_size", "sha256", "utf8"} or surface.get("sequence") != sequence or type(surface.get("utf8")) is not str or type(surface.get("byte_size")) is not int or type(surface.get("sha256")) is not str:
            raise ValueError("Exp444 publication terminal record is invalid")
        encoded = surface["utf8"].encode("utf-8")
        if len(encoded) != surface["byte_size"] or _sha256_antigravity(encoded) != surface["sha256"] or surface["sha256"] != prefix[sequence]:
            raise ValueError("Exp444 publication terminal record changed")
        payload = parse_canonical_json_bytes_antigravity(encoded)
        if not isinstance(payload, Mapping) or canonical_json_bytes_antigravity(payload) != encoded or set(payload) != {"sequence", "result"} or payload.get("sequence") != sequence or not isinstance(payload.get("result"), Mapping):
            raise ValueError("Exp444 publication terminal record is invalid")
        reconstructed.append(CandidateStabilityResultAntigravity.from_payload(payload["result"]))
    if tuple(item.candidate for item in reconstructed) != selection.source or tuple(item.candidate.original_rank for item in reconstructed) != tuple(range(1, 31)) or tuple(reconstructed) != tuple(sorted((*selection.stable, *selection.unstable), key=lambda item: item.candidate.original_rank)):
        raise ValueError("Exp444 publication terminal record reconstruction changed")


def _inspect_publication_payloads_antigravity(payloads: Mapping[str, bytes]) -> tuple[Mapping[str, object], StabilitySelectionAntigravity]:
    """Replay publication bytes already held under one trusted directory descriptor."""
    selection = _parse_selection_antigravity(payloads)
    summaries = _read_canonical_jsonl_bytes_antigravity(payloads[DATASET_SUMMARY_FILE_ANTIGRAVITY], "dataset summary")
    if _sha256_antigravity(payloads[DATASET_SUMMARY_FILE_ANTIGRAVITY]) != FROZEN_CONFIG.exp440_summary_sha256 or len(summaries) != 1117 or tuple(row.get("dataset") for row in summaries) != tuple(sorted(row.get("dataset") for row in summaries)) or len({row.get("dataset") for row in summaries}) != 1117:
        raise ValueError("Exp444 publication dataset summaries are invalid")
    _ensure_label_free_antigravity(summaries, "dataset summary")
    _ensure_label_free_antigravity(_read_canonical_jsonl_bytes_antigravity(payloads[VIEW_EVIDENCE_FILE_ANTIGRAVITY], VIEW_EVIDENCE_FILE_ANTIGRAVITY), VIEW_EVIDENCE_FILE_ANTIGRAVITY)
    ordered_results = tuple(sorted((*selection.stable, *selection.unstable), key=lambda item: item.candidate.original_rank))
    if payloads[VIEW_EVIDENCE_FILE_ANTIGRAVITY] != canonical_jsonl_bytes_antigravity(_view_rows_antigravity(ordered_results)):
        raise ValueError("Exp444 publication view evidence changed")
    control = _read_canonical_json_bytes_antigravity(payloads[CONTROL_REPLAY_FILE_ANTIGRAVITY], "control replay")
    if set(control) != {"schema", "rows"} or control["schema"] != "exp444.control-replay.v1" or not isinstance(control["rows"], list):
        raise ValueError("Exp444 publication control replay is invalid")
    _parse_control_replays_antigravity(control["rows"], selection.source)
    audit = _read_canonical_json_bytes_antigravity(payloads[SOURCE_AUDIT_FILE_ANTIGRAVITY], "source audit")
    if set(audit) != {"schema", "source_hashes", "source_candidate_count", "dataset_count", "label_access_count"} or audit["schema"] != "exp444.source-audit.v1" or audit["source_candidate_count"] != 30 or audit["dataset_count"] != 1117 or audit["label_access_count"] != 0 or not isinstance(audit["source_hashes"], Mapping) or not audit["source_hashes"]:
        raise ValueError("Exp444 publication source audit is invalid")
    backend_payload = _read_canonical_json_bytes_antigravity(payloads[BACKEND_DECISION_FILE_ANTIGRAVITY], "backend decision")
    if backend_payload.get("schema") == "exp444.backend-decision-wrapper.v1":
        if set(backend_payload) != {"schema", "authority", "decision"} or not isinstance(backend_payload["authority"], Mapping) or backend_payload.get("decision") != backend_payload["authority"].get("decision") or _sha256_antigravity(canonical_json_bytes_antigravity(backend_payload["authority"])) != FROZEN_CONFIG.backend_decision_authority_sha256:
            raise ValueError("Exp444 publication backend authority changed")
        from .provenance_antigravity import validate_backend_authority_payload_antigravity
        validate_backend_authority_payload_antigravity(backend_payload["authority"])
        backend_payload = backend_payload["decision"]
    backend = BackendDecisionAntigravity.from_payload(backend_payload)
    if backend.backend != "cpu" or backend.workers != 11:
        raise ValueError("Exp444 publication backend decision is invalid")
    _inspect_terminal_runtime_antigravity(_read_canonical_json_bytes_antigravity(payloads[RUNTIME_FILE_ANTIGRAVITY], "runtime"), selection)
    manifest = _read_canonical_json_bytes_antigravity(payloads[MANIFEST_FILE_ANTIGRAVITY], "manifest")
    status = _read_canonical_json_bytes_antigravity(payloads[STATUS_FILE_ANTIGRAVITY], "status")
    expected = _final_payloads_antigravity({name: payloads[name] for name in _LEAF_FILES_ANTIGRAVITY}, selection)
    if any(payloads[name] != payload for name, payload in expected.items()):
        raise ValueError("Exp444 publication hash or contract changed")
    if manifest["label_access_count"] != 0 or status.get("label_access_count") != 0:
        raise ValueError("Exp444 publication label governance is invalid")
    return manifest, selection


def inspect_publication_antigravity(publication_root: Path) -> FrozenPublicationAntigravity:
    """Fully replay a frozen publication without database, worker, or evaluator access."""
    root = Path(publication_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Exp444 publication inventory is invalid")
    with _pinned_publication_payloads_antigravity(root) as payloads:
        manifest, selection = _inspect_publication_payloads_antigravity(payloads)
    return FrozenPublicationAntigravity(root, manifest, selection)


def publish_prediction_antigravity(attempt_root: Path) -> FrozenPublicationAntigravity:
    """Freeze a complete checkpoint into immutable stable/unstable review lanes."""
    attempt = Path(attempt_root)
    destination = attempt / PUBLICATION_DIRECTORY_ANTIGRAVITY
    # Re-entry is strictly offline: it cannot re-open source/DB/checkpoint or
    # rewrite prediction evidence that was already frozen.
    if destination.exists() or destination.is_symlink():
        return inspect_publication_antigravity(destination)
    snapshot = _capture_attempt_antigravity(attempt)
    try:
        from .provenance_antigravity import verify_exp444_provenance_antigravity
        provenance = verify_exp444_provenance_antigravity()
        authority = _publish_source_authority_antigravity()
        payloads, selection = _payloads_antigravity(attempt, authority, snapshot, summary_bytes=provenance.dataset_summary_bytes, backend_authority=provenance.backend_authority)
        expected = _final_payloads_antigravity(payloads, selection)
        _assert_attempt_namespace_antigravity(attempt, snapshot)
        _write_staging_antigravity(destination, expected, parent_fd=snapshot.attempt_fd)
        _assert_attempt_namespace_antigravity(attempt, snapshot, allow_owned_mutation=True)
        publication_fd = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=snapshot.attempt_fd)
        try:
            with _pinned_publication_payloads_antigravity(root_fd=publication_fd) as published:
                manifest, inspected_selection = _inspect_publication_payloads_antigravity(published)
        finally:
            os.close(publication_fd)
        _assert_attempt_namespace_antigravity(attempt, snapshot, allow_owned_mutation=True)
        return FrozenPublicationAntigravity(destination, manifest, inspected_selection)
    finally:
        snapshot.close_antigravity()


__all__ = [
    "FrozenPublicationAntigravity", "publish_prediction_antigravity", "inspect_publication_antigravity",
    "PUBLICATION_DIRECTORY_ANTIGRAVITY", "DATASET_SUMMARY_FILE_ANTIGRAVITY",
    "SOURCE_CANDIDATES_FILE_ANTIGRAVITY", "CANDIDATE_STABILITY_FILE_ANTIGRAVITY",
    "VIEW_EVIDENCE_FILE_ANTIGRAVITY", "STABLE_REVIEW_FILE_ANTIGRAVITY",
    "UNSTABLE_REVIEW_FILE_ANTIGRAVITY", "CONTROL_REPLAY_FILE_ANTIGRAVITY",
    "SOURCE_AUDIT_FILE_ANTIGRAVITY", "BACKEND_DECISION_FILE_ANTIGRAVITY",
    "RUNTIME_FILE_ANTIGRAVITY", "MANIFEST_FILE_ANTIGRAVITY", "STATUS_FILE_ANTIGRAVITY",
    "ARTIFACT_SHA256_FILE_ANTIGRAVITY",
]
