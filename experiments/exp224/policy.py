from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

from experiments.exp224.artifacts import _runtime_metadata, sha256_file
from experiments.exp224.core import ALPHAS, RISK_TARGET, SEED, select_largest_safe_alpha

POLICY_SCHEMA = "exp224_policy_bundle_v1"


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _policy_hash(payload: Mapping[str, object]) -> str:
    content = {key: value for key, value in payload.items() if key != "policy_hash"}
    return hashlib.sha256(_canonical_json(content)).hexdigest()


def _normalize_counts(counts: Mapping[object, object]) -> dict[str, int]:
    normalized = {str(alpha): int(counts.get(str(alpha), counts.get(alpha, 0))) for alpha in ALPHAS}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("false counts must be non-negative")
    return normalized


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("artifact manifest is empty")
    profile_ids = [row["profile_id"] for row in rows]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("duplicate profile_id in artifact manifest")
    return sorted(rows, key=lambda row: row["profile_id"])


def freeze_exp224_policy(
    manifest: Path,
    risk_summary: Mapping[str, object],
    paths: Mapping[str, object],
    output_path: Path,
) -> dict[str, object]:
    """Freeze immutable artifact, risk, source, DB, and runtime provenance."""
    manifest_path = Path(manifest).resolve()
    db_path = Path(paths["db_path"]).resolve()
    source_paths = {
        str(name): Path(path).resolve()
        for name, path in dict(paths.get("source_paths", {})).items()
    }
    rows = _load_manifest(manifest_path)
    profiles: list[dict[str, object]] = []
    derived_counts = {str(alpha): 0 for alpha in ALPHAS}
    derived_trials = 0
    for row in rows:
        artifact_path = Path(row["artifact_path"]).resolve()
        actual_hash = sha256_file(artifact_path)
        if actual_hash != row["artifact_sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch: {row['profile_id']}")
        row_counts = _normalize_counts(json.loads(row["train_false_counts_json"]))
        row_trials = int(row["train_trials"])
        for alpha in ALPHAS:
            derived_counts[str(alpha)] += row_counts[str(alpha)]
        derived_trials += row_trials
        profiles.append(
            {
                "profile_id": row["profile_id"],
                "artifact_path": str(artifact_path),
                "artifact_sha256": actual_hash,
                "n_train": int(row["n_train"]),
                "train_target_length": int(row["train_target_length"]),
                "train_trials": row_trials,
                "train_false_counts": row_counts,
            }
        )

    supplied_counts = _normalize_counts(
        dict(risk_summary.get("false_counts", derived_counts))
    )
    supplied_trials = int(risk_summary.get("trials", derived_trials))
    if supplied_counts != derived_counts or supplied_trials != derived_trials:
        raise ValueError("risk summary does not match artifact manifest")
    expected_count = int(risk_summary.get("expected_profile_count", 639))
    if len(profiles) != expected_count:
        raise ValueError(
            f"profile count mismatch: {len(profiles)}/{expected_count}"
        )
    selected_alpha = select_largest_safe_alpha(
        {float(key): value for key, value in derived_counts.items()},
        derived_trials,
        RISK_TARGET,
    )
    profile_list_hash = hashlib.sha256(
        _canonical_json({"profile_ids": [row["profile_id"] for row in profiles]})
    ).hexdigest()
    payload: dict[str, object] = {
        "schema": POLICY_SCHEMA,
        "experiment": "Exp224",
        "seed": SEED,
        "alphas": list(ALPHAS),
        "risk_target": RISK_TARGET,
        "selected_alpha": selected_alpha,
        "shadow_execution_gate": "raw_replay_required",
        "expected_profile_count": expected_count,
        "profile_list_sha256": profile_list_hash,
        "artifact_manifest_path": str(manifest_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "database_path": str(db_path),
        "database_sha256": sha256_file(db_path),
        "source_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(source_paths.items())
        },
        "runtime": _runtime_metadata(),
        "train_risk": {
            "false_counts": derived_counts,
            "trials": derived_trials,
        },
        "profiles": profiles,
    }
    payload["policy_hash"] = _policy_hash(payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = _canonical_json(payload) + b"\n"
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    return payload


def load_verified_exp224_policy(
    path: Path,
    verify_artifacts: bool = True,
) -> dict[str, object]:
    policy_path = Path(path)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if payload.get("schema") != POLICY_SCHEMA:
        raise ValueError("unsupported Exp224 policy schema")
    if payload.get("policy_hash") != _policy_hash(payload):
        raise ValueError("policy hash mismatch")
    if payload.get("runtime") != _runtime_metadata():
        raise ValueError("policy runtime mismatch")
    if verify_artifacts:
        manifest_path = Path(str(payload["artifact_manifest_path"]))
        if sha256_file(manifest_path) != payload["artifact_manifest_sha256"]:
            raise ValueError("artifact manifest SHA-256 mismatch")
        db_path = Path(str(payload["database_path"]))
        if sha256_file(db_path) != payload["database_sha256"]:
            raise ValueError("database SHA-256 mismatch")
        for name, source in dict(payload["source_files"]).items():
            if sha256_file(Path(str(source["path"]))) != source["sha256"]:
                raise ValueError(f"source SHA-256 mismatch: {name}")
        for profile in payload["profiles"]:
            if sha256_file(Path(str(profile["artifact_path"]))) != profile["artifact_sha256"]:
                raise ValueError(
                    f"artifact SHA-256 mismatch: {profile['profile_id']}"
                )
    return payload

