"""Descriptor-pinned, pre-label authority for immutable Exp446/Exp404 inputs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Mapping, Sequence

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import PublicationIdentityAntigravity


_DATASET_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset")
_EXP446_ROOT_ANTIGRAVITY = _DATASET_ROOT_ANTIGRAVITY / "exp446_exp445_publication_materialization_repair_antigravity_20260811"
_EXP446_PARTS_ANTIGRAVITY = ("attempts_antigravity", "attempt_0000_antigravity", "publication_antigravity")
_EXP404_HARD_ANTIGRAVITY = _DATASET_ROOT_ANTIGRAVITY / "exp403_408_feature_ml_validation" / "exp404_hard_confidence.jsonl"
_DATABASE_ANTIGRAVITY = _DATASET_ROOT_ANTIGRAVITY / "univariate_ts.db"
_DATABASE_SHA256_ANTIGRAVITY = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
_DATABASE_INODE_ANTIGRAVITY = 1048138
_LEAVES_ANTIGRAVITY = (
    "artifact_sha256_antigravity.json", "backend_decision_antigravity.json",
    "candidate_stability_antigravity.jsonl", "control_replay_antigravity.json",
    "dataset_summary_antigravity.jsonl", "manifest_antigravity.json",
    "runtime_antigravity.json", "source_audit_antigravity.json",
    "source_candidates_antigravity.jsonl", "stable_review_prediction_antigravity.jsonl",
    "status_antigravity.json", "unstable_review_prediction_antigravity.jsonl",
    "view_evidence_antigravity.jsonl",
)
_DIRECTORY_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_LEAF_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


@dataclass(frozen=True, slots=True)
class HardPredictionsAntigravity:
    """The byte-pinned Exp404 Hard coordinates, with no database capability."""

    sha256: str
    coordinates: frozenset[tuple[str, int]]


def _identity_antigravity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_directory_antigravity(parent_fd: int | None, name: str | Path) -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError("frozen source directory is absent or unsafe") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode) or _identity_antigravity(before) != _identity_antigravity(opened):
            raise ValueError("frozen source directory changed during open")
        return descriptor, _identity_antigravity(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _assert_directory_antigravity(parent_fd: int | None, name: str | Path, descriptor: int, expected: tuple[int, int, int, int, int, int, int]) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(descriptor)
    except OSError as error:
        raise ValueError("frozen source directory disappeared") from error
    if any(not stat.S_ISDIR(item.st_mode) or _identity_antigravity(item) != expected for item in (current, opened)):
        raise ValueError("frozen source directory changed during verification")


def _exact_names_antigravity(directory_fd: int, expected: Sequence[str]) -> None:
    try:
        names = tuple(sorted(os.listdir(directory_fd)))
    except OSError as error:
        raise ValueError("frozen source inventory is unreadable") from error
    if names != tuple(sorted(expected)):
        raise ValueError("frozen source inventory is not exact")


def _read_leaf_antigravity(directory_fd: int, name: str) -> tuple[bytes, str, tuple[int, int, int, int, int, int, int]]:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        descriptor = os.open(name, _LEAF_FLAGS_ANTIGRAVITY, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"frozen source leaf is absent or unsafe: {name}") from error
    try:
        opened = os.fstat(descriptor)
        expected = _identity_antigravity(before)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or _identity_antigravity(opened) != expected:
            raise ValueError(f"frozen source leaf is not one regular inode: {name}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
            digest.update(block)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        after = os.fstat(descriptor)
        if any(
            not stat.S_ISREG(item.st_mode) or item.st_nlink != 1 or _identity_antigravity(item) != expected
            for item in (current, after)
        ):
            raise ValueError(f"frozen source leaf changed during read: {name}")
        return b"".join(chunks), digest.hexdigest(), expected
    except OSError as error:
        raise ValueError(f"frozen source leaf read failed: {name}") from error
    finally:
        os.close(descriptor)


def _strict_json_antigravity(payload: bytes, name: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} has duplicate JSON keys")
            result[key] = value
        return result
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not a JSON object")
    return value


def _strict_jsonl_antigravity(payload: bytes, name: str) -> tuple[Mapping[str, object], ...]:
    if not payload.endswith(b"\n") or not payload:
        raise ValueError(f"{name} is not complete JSONL")
    return tuple(_strict_json_antigravity(line, f"{name} row {number}") for number, line in enumerate(payload.splitlines(), 1) if line)


def _coordinate_antigravity(value: object, name: str) -> tuple[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"dataset", "instance_index", "original_depths", "original_rank", "raw_length"}:
        raise ValueError(f"{name} candidate schema is invalid")
    dataset, index, rank = value["dataset"], value["instance_index"], value["original_rank"]
    if type(dataset) is not str or not dataset or type(index) is not int or index < 0 or type(rank) is not int or rank < 1:
        raise ValueError(f"{name} candidate is invalid")
    return dataset, index


def _validate_exp446_semantics_antigravity(leaves: Mapping[str, bytes], digests: Mapping[str, str], root_identity: tuple[int, int, int, int, int, int, int]) -> PublicationIdentityAntigravity:
    if digests["manifest_antigravity.json"] != FROZEN_CONFIG_ANTIGRAVITY.exp446_manifest_file_sha256:
        raise ValueError("Exp446 manifest digest differs from frozen authority")
    if digests["artifact_sha256_antigravity.json"] != FROZEN_CONFIG_ANTIGRAVITY.exp446_artifact_map_file_sha256:
        raise ValueError("Exp446 artifact map digest differs from frozen authority")
    artifact_map = _strict_json_antigravity(leaves["artifact_sha256_antigravity.json"], "Exp446 artifact map")
    if set(artifact_map) != {"schema", "files"} or artifact_map.get("schema") != "exp446.artifact-sha256.v1" or not isinstance(artifact_map.get("files"), Mapping):
        raise ValueError("Exp446 artifact map schema is invalid")
    expected_map_names = set(_LEAVES_ANTIGRAVITY) - {"artifact_sha256_antigravity.json"}
    if set(artifact_map["files"]) != expected_map_names or any(artifact_map["files"].get(name) != digests[name] for name in expected_map_names):
        raise ValueError("Exp446 artifact map does not bind the exact leaves")
    manifest = _strict_json_antigravity(leaves["manifest_antigravity.json"], "Exp446 manifest")
    if (
        set(manifest) != {"dataset_count", "experiment_id", "label_access_count", "leaf_sha256", "prediction_sha256", "run_identity", "schema", "source_candidate_count", "stable_count", "unstable_count"}
        or manifest.get("experiment_id") != "Exp446" or manifest.get("schema") != "exp446.publication.v1"
        or manifest.get("run_identity") != "exp446-exp445-publication-materialization-repair-antigravity-20260811"
        or (manifest.get("dataset_count"), manifest.get("source_candidate_count"), manifest.get("stable_count"), manifest.get("unstable_count"), manifest.get("label_access_count")) != (1117, 30, 26, 4, 0)
        or not isinstance(manifest.get("leaf_sha256"), Mapping)
    ):
        raise ValueError("Exp446 manifest semantics are invalid")
    manifest_leaves = manifest["leaf_sha256"]
    expected_manifest_names = expected_map_names - {"manifest_antigravity.json", "status_antigravity.json"}
    if set(manifest_leaves) != expected_manifest_names or any(manifest_leaves.get(name) != digests[name] for name in expected_manifest_names):
        raise ValueError("Exp446 manifest leaf identity is invalid")
    stable = _strict_jsonl_antigravity(leaves["stable_review_prediction_antigravity.jsonl"], "stable review")
    unstable = _strict_jsonl_antigravity(leaves["unstable_review_prediction_antigravity.jsonl"], "unstable review")
    source = _strict_jsonl_antigravity(leaves["source_candidates_antigravity.jsonl"], "source candidates")
    summary = _strict_jsonl_antigravity(leaves["dataset_summary_antigravity.jsonl"], "dataset summary")
    if len(summary) != 1117 or any(set(row) != {"candidate_count", "dataset"} or type(row["dataset"]) is not str or not row["dataset"] for row in summary):
        raise ValueError("Exp446 dataset order is invalid")
    dataset_order = tuple(row["dataset"] for row in summary)
    if len(set(dataset_order)) != 1117:
        raise ValueError("Exp446 dataset order is not unique")
    source_coordinates = tuple(_coordinate_antigravity(row, "source") for row in source)
    if len(source_coordinates) != 30 or len(set(source_coordinates)) != 30 or tuple(row["original_rank"] for row in source) != tuple(range(1, 31)):
        raise ValueError("Exp446 source candidates are invalid")
    def lane(rows: Sequence[Mapping[str, object]], name: str, count: int) -> tuple[tuple[str, int], ...]:
        if len(rows) != count:
            raise ValueError(f"Exp446 {name} population is invalid")
        result: list[tuple[str, int]] = []
        for row in rows:
            if not isinstance(row.get("candidate"), Mapping) or row.get("lane") != name:
                raise ValueError(f"Exp446 {name} lane is invalid")
            result.append(_coordinate_antigravity(row["candidate"], name))
        return tuple(result)
    stable_coordinates = lane(stable, "stable_review", 26)
    unstable_coordinates = lane(unstable, "unstable_review", 4)
    if set(source_coordinates) != set(stable_coordinates) | set(unstable_coordinates) or set(stable_coordinates) & set(unstable_coordinates):
        raise ValueError("Exp446 lane partition is not exact")
    if any(dataset not in dataset_order for dataset, _ in source_coordinates):
        raise ValueError("Exp446 source coordinate dataset is absent")
    if hashlib.sha256(leaves["stable_review_prediction_antigravity.jsonl"] + leaves["unstable_review_prediction_antigravity.jsonl"]).hexdigest() != FROZEN_CONFIG_ANTIGRAVITY.exp446_prediction_sha256:
        raise ValueError("Exp446 frozen prediction digest differs")
    source_audit = _strict_json_antigravity(leaves["source_audit_antigravity.json"], "Exp446 source audit")
    status = _strict_json_antigravity(leaves["status_antigravity.json"], "Exp446 status")
    if source_audit.get("label_access_count") != 0 or not isinstance(source_audit.get("source_hashes"), Mapping) or source_audit["source_hashes"].get("database") != _DATABASE_SHA256_ANTIGRAVITY:
        raise ValueError("Exp446 source audit violates the label/database contract")
    if status.get("label_access_count") != 0 or status.get("review_only") is not True or status.get("operational_alert_eligible") is not False:
        raise ValueError("Exp446 status violates review-only governance")
    return PublicationIdentityAntigravity(
        root_device=root_identity[0], root_inode=root_identity[1], leaf_sha256=dict(digests),
        manifest_sha256=digests["manifest_antigravity.json"], prediction_sha256=FROZEN_CONFIG_ANTIGRAVITY.exp446_prediction_sha256,
        artifact_map_sha256=digests["artifact_sha256_antigravity.json"], dataset_order=dataset_order,
        source_coordinates=source_coordinates, stable_coordinates=stable_coordinates,
        unstable_coordinates=unstable_coordinates, label_access_count=0,
    )


def _verify_database_antigravity() -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    try:
        before = os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)
        descriptor = os.open(_DATABASE_ANTIGRAVITY, _LEAF_FLAGS_ANTIGRAVITY)
    except OSError as error:
        raise ValueError("source database is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        expected = _identity_antigravity(before)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_ino != _DATABASE_INODE_ANTIGRAVITY or _identity_antigravity(opened) != expected:
            raise ValueError("source database inode/link contract is invalid")
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        while block := os.read(descriptor, 1 << 20):
            digest.update(block)
        current = os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)
        if digest.hexdigest() != _DATABASE_SHA256_ANTIGRAVITY or any(_identity_antigravity(item) != expected for item in (opened, current)) or current.st_nlink != 1:
            raise ValueError("source database changed during digest verification")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, expected
    except BaseException:
        os.close(descriptor)
        raise


def capture_exp446_publication_antigravity() -> PublicationIdentityAntigravity:
    """Capture every Exp446 publication byte under descriptor pins, without labels."""
    root_fd = attempts_fd = attempt_fd = publication_fd = -1
    try:
        root_fd, root_identity = _open_directory_antigravity(None, _EXP446_ROOT_ANTIGRAVITY)
        attempts_fd, attempts_identity = _open_directory_antigravity(root_fd, _EXP446_PARTS_ANTIGRAVITY[0])
        attempt_fd, attempt_identity = _open_directory_antigravity(attempts_fd, _EXP446_PARTS_ANTIGRAVITY[1])
        publication_fd, publication_identity = _open_directory_antigravity(attempt_fd, _EXP446_PARTS_ANTIGRAVITY[2])
        _exact_names_antigravity(publication_fd, _LEAVES_ANTIGRAVITY)
        leaves: dict[str, bytes] = {}
        digests: dict[str, str] = {}
        for name in _LEAVES_ANTIGRAVITY:
            payload, digest, _ = _read_leaf_antigravity(publication_fd, name)
            leaves[name], digests[name] = payload, digest
        result = _validate_exp446_semantics_antigravity(leaves, digests, publication_identity)
        _exact_names_antigravity(publication_fd, _LEAVES_ANTIGRAVITY)
        _assert_directory_antigravity(attempt_fd, _EXP446_PARTS_ANTIGRAVITY[2], publication_fd, publication_identity)
        _assert_directory_antigravity(attempts_fd, _EXP446_PARTS_ANTIGRAVITY[1], attempt_fd, attempt_identity)
        _assert_directory_antigravity(root_fd, _EXP446_PARTS_ANTIGRAVITY[0], attempts_fd, attempts_identity)
        _assert_directory_antigravity(None, _EXP446_ROOT_ANTIGRAVITY, root_fd, root_identity)
        return result
    finally:
        for descriptor in (publication_fd, attempt_fd, attempts_fd, root_fd):
            if descriptor >= 0:
                os.close(descriptor)


def capture_hard_predictions_antigravity(identity: PublicationIdentityAntigravity) -> HardPredictionsAntigravity:
    """Capture Exp404 Hard only after a verified Exp446 identity is supplied."""
    if not isinstance(identity, PublicationIdentityAntigravity):
        raise ValueError("verified Exp446 identity is required")
    parent_fd = file_fd = -1
    try:
        parent_fd, parent_identity = _open_directory_antigravity(None, _EXP404_HARD_ANTIGRAVITY.parent)
        payload, digest, file_identity = _read_leaf_antigravity(parent_fd, _EXP404_HARD_ANTIGRAVITY.name)
        if digest != FROZEN_CONFIG_ANTIGRAVITY.exp404_hard_prediction_sha256:
            raise ValueError("Exp404 Hard digest differs from frozen authority")
        rows = _strict_jsonl_antigravity(payload, "Exp404 Hard")
        coordinates: set[tuple[str, int]] = set()
        required = {"abstention_reason", "candidate_index", "confidence_by_alpha", "contains_evaluation_metrics", "contains_test_labels", "dataset", "eligibility", "scores_by_alpha"}
        for row in rows:
            if set(row) != required or row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False or type(row.get("dataset")) is not str or not row["dataset"] or type(row.get("candidate_index")) is not int or row["candidate_index"] < 0:
                raise ValueError("Exp404 Hard row violates frozen label-free schema")
            coordinate = (row["dataset"], row["candidate_index"])
            if coordinate in coordinates:
                raise ValueError("Exp404 Hard contains duplicate coordinates")
            coordinates.add(coordinate)
        if len(coordinates) != 2218:
            raise ValueError("Exp404 Hard coordinate population differs")
        _assert_directory_antigravity(None, _EXP404_HARD_ANTIGRAVITY.parent, parent_fd, parent_identity)
        return HardPredictionsAntigravity(digest, frozenset(coordinates))
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)


def open_test_labels_antigravity(identity: PublicationIdentityAntigravity, independently_confirmed: PublicationIdentityAntigravity | None):
    """Open the one-use TEST-label session only after byte/root-identical captures."""
    if not isinstance(identity, PublicationIdentityAntigravity) or independently_confirmed is None or identity != independently_confirmed:
        raise ValueError("two byte/root-identical Exp446 captures are required before labels")
    descriptor, expected = _verify_database_antigravity()
    try:
        import sqlite3
        uri = f"file:/dev/fd/{descriptor}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
    except BaseException:
        os.close(descriptor)
        raise
    expected_order = identity.dataset_order
    next_index = 0
    closed = False
    def authorizer(action: int, table: str | None, column: str | None, _database: str | None, _trigger: str | None) -> int:
        permitted = {
            ("instances", "instance_index"), ("instances", "label"), ("instances", "split"), ("instances", "dataset_id"),
            ("datasets", "id"), ("datasets", "name"),
        }
        if action == sqlite3.SQLITE_SELECT or (action == sqlite3.SQLITE_READ and (table, column) in permitted):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    connection.set_authorizer(authorizer)
    class LabelSessionAntigravity:
        __slots__ = ()
        def labels_for_dataset_antigravity(self, dataset: str) -> tuple[tuple[int, int], ...]:
            nonlocal next_index
            if closed or next_index >= len(expected_order) or dataset != expected_order[next_index]:
                raise ValueError("TEST labels must be read once in frozen dataset order")
            rows = tuple(connection.execute("SELECT i.instance_index, i.label FROM instances AS i JOIN datasets AS d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index", (dataset,)))
            if any(type(index) is not int or type(label) is not int or label not in (0, 1) for index, label in rows):
                raise ValueError("TEST label row is invalid")
            if tuple(index for index, _ in rows) != tuple(range(len(rows))):
                raise ValueError("TEST indices are not contiguous")
            next_index += 1
            return rows
        def close_antigravity(self) -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                connection.close()
                current = os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)
                opened = os.fstat(descriptor)
                if current.st_nlink != 1 or any(_identity_antigravity(item) != expected for item in (current, opened)):
                    raise ValueError("source database changed during TEST-label session")
            finally:
                os.close(descriptor)
    return LabelSessionAntigravity()


__all__ = ["HardPredictionsAntigravity", "capture_exp446_publication_antigravity", "capture_hard_predictions_antigravity", "open_test_labels_antigravity"]
