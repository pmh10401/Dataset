"""Pinned, label-free authorities required before Exp444 can mutate state."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Mapping

from .artifacts_antigravity import parse_canonical_json_bytes_antigravity
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import BackendDecisionAntigravity


def _read_pinned_child_antigravity(directory_fd: int, name: str, expected_sha256: str) -> bytes:
    """Read one regular, single-linked authority child through its parent FD."""
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    try:
        first = os.fstat(fd)
        if not stat.S_ISREG(first.st_mode) or first.st_nlink != 1:
            raise ValueError("provenance child must be a single-linked regular file")
        chunks: list[bytes] = []
        while block := os.read(fd, 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        last = os.fstat(fd)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns, first.st_ctime_ns) != (last.st_dev, last.st_ino, last.st_size, last.st_mtime_ns, last.st_ctime_ns) or (first.st_dev, first.st_ino) != (named.st_dev, named.st_ino):
            raise ValueError("provenance child changed during pinned read")
    finally:
        os.close(fd)
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("provenance SHA256 changed")
    return raw


def _read_pinned_antigravity(directory: Path, name: str, expected_sha256: str) -> bytes:
    """Read one authority through a retained directory descriptor."""
    return _read_pinned_bundle_antigravity(directory, ((name, expected_sha256),))[name]


def _read_pinned_bundle_antigravity(directory: Path, entries: tuple[tuple[str, str], ...]) -> dict[str, bytes]:
    """Pin the complete Exp440 authority set to one publication-directory FD."""
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        before = os.fstat(directory_fd)
        output = {name: _read_pinned_child_antigravity(directory_fd, name, digest) for name, digest in entries}
        after = os.fstat(directory_fd)
        if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("provenance directory changed during pinned read")
        return output
    finally:
        os.close(directory_fd)


def _canonical_mapping_antigravity(raw: bytes, name: str) -> Mapping[str, object]:
    try:
        payload = parse_canonical_json_bytes_antigravity(raw)
    except ValueError as error:
        raise ValueError(f"{name} is not canonical") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    return payload


@dataclass(frozen=True)
class Exp444ProvenanceAntigravity:
    dataset_summary_bytes: bytes
    backend_authority: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_summary_bytes", bytes(self.dataset_summary_bytes))
        object.__setattr__(self, "backend_authority", MappingProxyType(dict(self.backend_authority)))

    @property
    def backend_decision(self) -> BackendDecisionAntigravity:
        value = self.backend_authority.get("decision")
        if not isinstance(value, Mapping):
            raise ValueError("backend authority decision is invalid")
        return BackendDecisionAntigravity.from_payload(value)


def verify_exp444_provenance_antigravity() -> Exp444ProvenanceAntigravity:
    """Authenticate tracked Task5 evidence and Exp440's exact 1,117-row summary."""
    external = Path(FROZEN_CONFIG.exp440_publication_path)
    manifest_name = Path(FROZEN_CONFIG.exp440_manifest_path).name
    artifact_name = Path(FROZEN_CONFIG.exp440_artifact_map_path).name
    summary_name = Path(FROZEN_CONFIG.exp440_summary_path).name
    external_bundle = _read_pinned_bundle_antigravity(external, ((manifest_name, FROZEN_CONFIG.exp440_manifest_sha256), (artifact_name, FROZEN_CONFIG.exp440_artifact_map_sha256), (summary_name, FROZEN_CONFIG.exp440_summary_sha256)))
    manifest_raw = external_bundle[manifest_name]
    artifact_raw = external_bundle[artifact_name]
    summary = external_bundle[summary_name]
    manifest = _canonical_mapping_antigravity(manifest_raw, "Exp440 manifest")
    artifact = _canonical_mapping_antigravity(artifact_raw, "Exp440 artifact map")
    if manifest.get("experiment_id") != "Exp440" or manifest.get("dataset_count") != 1117 or manifest.get("contains_test_labels") is not False:
        raise ValueError("Exp440 manifest identity/governance is invalid")
    files = artifact.get("files")
    manifest_files = manifest.get("artifact_sha256")
    if not isinstance(files, Mapping) or not isinstance(manifest_files, Mapping) or files.get("dataset_summary_antigravity.jsonl") != FROZEN_CONFIG.exp440_summary_sha256 or dict(manifest_files) != {name: digest for name, digest in files.items() if name != "manifest_antigravity.json"}:
        raise ValueError("Exp440 summary is not cross-attested")
    rows = summary.splitlines(keepends=True)
    if len(rows) != 1117 or any(not row.endswith(b"\n") for row in rows):
        raise ValueError("Exp440 summary count is invalid")
    names: list[str] = []
    for row in rows:
        parsed = _canonical_mapping_antigravity(row, "Exp440 dataset summary")
        if set(parsed) != {"dataset", "candidate_count"} or type(parsed["dataset"]) is not str or type(parsed["candidate_count"]) is not int:
            raise ValueError("Exp440 summary row is invalid")
        names.append(parsed["dataset"])
    if names != sorted(names) or len(set(names)) != 1117:
        raise ValueError("Exp440 summary order is invalid")
    local = Path(__file__).with_name(FROZEN_CONFIG.backend_decision_authority_identifier)
    authority_raw = _read_pinned_antigravity(local.parent, local.name, FROZEN_CONFIG.backend_decision_authority_sha256)
    authority = _canonical_mapping_antigravity(authority_raw, "Task5 backend authority")
    required = {"schema", "task5_log_utf8", "task5_log_byte_size", "task5_log_sha256", "decision_line_index", "benchmark_head", "start", "inputs", "decision", "complete"}
    if set(authority) != required or authority.get("schema") != "exp444.backend-decision-authority.v1" or authority.get("task5_log_sha256") != "74c54a40405dd40e5f3f297b1878aff233b5c2a3c217c06453767b624205e090" or authority.get("task5_log_byte_size") != 12139 or authority.get("decision_line_index") != 3 or authority.get("benchmark_head") != "cf57bcf9ef234ffa626d5216da988fb00c12c29e":
        raise ValueError("Task5 backend authority identity is invalid")
    log = authority["task5_log_utf8"]
    if type(log) is not str or len(log.encode()) != 12139 or sha256(log.encode()).hexdigest() != authority["task5_log_sha256"]:
        raise ValueError("Task5 backend log bytes changed")
    decision = Exp444ProvenanceAntigravity(summary, authority).backend_decision
    if decision.backend != "cpu" or decision.workers != 11 or len(decision.evidence) != 5 or authority.get("inputs", {}).get("label_access_count") != 0 or authority.get("complete", {}).get("label_access_count") != 0:
        raise ValueError("Task5 backend authority evidence is invalid")
    return Exp444ProvenanceAntigravity(summary, authority)


__all__ = ["Exp444ProvenanceAntigravity", "verify_exp444_provenance_antigravity"]
