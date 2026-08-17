"""Byte-preserving single-method views over completed mixed score roots."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .artifacts import (
    canonical_json_bytes,
    read_strict_json,
    sha256_file,
    validate_score_payload,
    write_create_only_bytes,
)
from .runner import FrozenScoreManifest, freeze_score_root


DERIVED_VIEW_SCHEMA_ANTIGRAVITY = (
    "paper-public-baselines.derived-single-method-view-antigravity.v1"
)
DERIVED_VIEW_PROVENANCE_NAME_ANTIGRAVITY = (
    "derived_view_provenance_antigravity.json"
)


def freeze_single_method_view_antigravity(
    source_root: Path, destination_root: Path, method: str
) -> FrozenScoreManifest:
    """Freeze one completed method without changing any scientific shard byte."""
    source = _confined_root_antigravity(Path(source_root), must_exist=True)
    destination = _confined_root_antigravity(
        Path(destination_root), must_exist=False
    )
    if method != "deep_svdd_1d":
        raise ValueError("derived method identity is invalid")

    source_config_path = source / "run_config.json"
    source_digest_path = source / "run_config.sha256"
    source_config = read_strict_json(source_config_path)
    source_config_sha = sha256_file(source_config_path)
    if (
        source_digest_path.read_text(encoding="ascii").strip()
        != source_config_sha
    ):
        raise ValueError("source run config SHA256 mismatch")
    if source_config.get("schema") != "paper-public-baselines.score-run.v1":
        raise ValueError("source run config schema is invalid")
    datasets = _string_tuple_antigravity(source_config.get("datasets"), "datasets")
    methods = _string_tuple_antigravity(source_config.get("methods"), "methods")
    if method not in methods:
        raise ValueError("derived method is absent from source run")
    database_sha = source_config.get("database_sha256")
    if type(database_sha) is not str or len(database_sha) != 64:
        raise ValueError("source database SHA256 is invalid")

    source_method_root = source / "shards" / method
    if not source_method_root.is_dir() or source_method_root.is_symlink():
        raise ValueError("source method shard directory is invalid")
    expected_names = {
        f"{hashlib.sha256(dataset.encode('utf-8')).hexdigest()}{suffix}"
        for dataset in datasets
        for suffix in (".json", ".meta")
    }
    actual_names = {
        path.name for path in source_method_root.iterdir() if path.is_file()
    }
    if actual_names != expected_names:
        raise ValueError("source method shard inventory is not exact")

    target_config = dict(source_config)
    target_config["methods"] = [method]
    target_config_bytes = canonical_json_bytes(target_config)
    target_config_sha = hashlib.sha256(target_config_bytes).hexdigest()
    write_create_only_bytes(destination / "run_config.json", target_config_bytes)
    write_create_only_bytes(
        destination / "run_config.sha256",
        (target_config_sha + "\n").encode("ascii"),
    )

    shard_set_digest = hashlib.sha256()
    for dataset in datasets:
        token = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
        source_shard = source_method_root / f"{token}.json"
        source_meta = source_method_root / f"{token}.meta"
        if source_shard.is_symlink() or source_meta.is_symlink():
            raise ValueError("source shard symlink is forbidden")
        shard_bytes = source_shard.read_bytes()
        shard_sha = hashlib.sha256(shard_bytes).hexdigest()
        metadata = read_strict_json(source_meta)
        expected_metadata = {
            "schema": "paper-public-baselines.score-shard-meta.v1",
            "run_config_sha256": source_config_sha,
            "database_sha256": database_sha,
            "shard_sha256": shard_sha,
        }
        if metadata != expected_metadata:
            if metadata.get("shard_sha256") != shard_sha:
                raise ValueError("source shard SHA256 mismatch")
            raise ValueError("source shard metadata is invalid")
        shard = validate_score_payload(read_strict_json(source_shard))
        if shard["dataset"] != dataset or shard["method"] != method:
            raise ValueError("source shard coordinate is invalid")

        target_method_root = destination / "shards" / method
        write_create_only_bytes(target_method_root / source_shard.name, shard_bytes)
        write_create_only_bytes(
            target_method_root / source_meta.name,
            canonical_json_bytes(
                {
                    "schema": "paper-public-baselines.score-shard-meta.v1",
                    "run_config_sha256": target_config_sha,
                    "database_sha256": database_sha,
                    "shard_sha256": shard_sha,
                }
            ),
        )
        shard_set_digest.update(dataset.encode("utf-8"))
        shard_set_digest.update(b"\0")
        shard_set_digest.update(shard_sha.encode("ascii"))
        shard_set_digest.update(b"\0")

    provenance = {
        "schema": DERIVED_VIEW_SCHEMA_ANTIGRAVITY,
        "source_root": str(source),
        "destination_root": str(destination),
        "method": method,
        "dataset_count": len(datasets),
        "database_sha256": database_sha,
        "source_run_config_sha256": source_config_sha,
        "destination_run_config_sha256": target_config_sha,
        "ordered_shard_set_sha256": shard_set_digest.hexdigest(),
        "scientific_shard_bytes_changed": False,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    write_create_only_bytes(
        destination / DERIVED_VIEW_PROVENANCE_NAME_ANTIGRAVITY,
        canonical_json_bytes(provenance),
    )
    return freeze_score_root(destination)


def _confined_root_antigravity(path: Path, *, must_exist: bool) -> Path:
    absolute = path.absolute()
    if absolute.is_symlink():
        raise ValueError("score root path contains a symlink")
    resolved = absolute.resolve(strict=False)
    if must_exist and not resolved.is_dir():
        raise ValueError("source score root is absent")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("score root is not a real directory")
    return resolved


def _string_tuple_antigravity(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not str or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"source {name} is invalid")
    return tuple(value)


__all__ = [
    "DERIVED_VIEW_PROVENANCE_NAME_ANTIGRAVITY",
    "DERIVED_VIEW_SCHEMA_ANTIGRAVITY",
    "freeze_single_method_view_antigravity",
]
