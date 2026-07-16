from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.exp220_223.policy import load_verified_policy, sha256_file


SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_batches (
    batch_id TEXT PRIMARY KEY,
    collected_start_utc TEXT NOT NULL,
    collected_end_utc TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_instances (
    batch_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    observed_at_utc TEXT,
    values_blob BLOB NOT NULL,
    original_length INTEGER NOT NULL CHECK (original_length > 0),
    PRIMARY KEY (batch_id, instance_id),
    FOREIGN KEY (batch_id) REFERENCES shadow_batches(batch_id)
);
CREATE TABLE IF NOT EXISTS shadow_rejections (
    batch_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (batch_id, instance_id),
    FOREIGN KEY (batch_id) REFERENCES shadow_batches(batch_id)
);
"""


def _utc_text(value: str, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} is required")
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {text}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_constant(value: str) -> object:
    raise ValueError(f"values must be finite: {value}")


def _load_source(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("source contains no instances")
    return rows


def _validate_rows(
    rows: list[dict[str, object]], known_profiles: set[str]
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        instance_id = str(row.get("instance_id", "")).strip()
        profile_id = str(row.get("profile_id", "")).strip()
        if not instance_id or not profile_id:
            raise ValueError(f"instance_id and profile_id are required: row={row_number}")
        if instance_id in seen:
            raise ValueError(f"duplicate instance_id: {instance_id}")
        seen.add(instance_id)
        raw_values = row.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"empty values: instance_id={instance_id}")
        try:
            values = np.asarray(raw_values, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"values must be numeric: instance_id={instance_id}") from exc
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"values must be finite 1-D: instance_id={instance_id}")
        if profile_id not in known_profiles:
            rejected.append(
                {
                    "instance_id": instance_id,
                    "profile_id": profile_id,
                    "status": "onboarding_required",
                }
            )
            continue
        observed = row.get("observed_at")
        observed_at = None if observed in (None, "") else _utc_text(str(observed), "observed_at")
        accepted.append(
            {
                "instance_id": instance_id,
                "profile_id": profile_id,
                "observed_at": observed_at,
                "values": values,
            }
        )
    return accepted, rejected


def _normalize_ingest_policy(
    policy: dict[str, object],
) -> tuple[str, set[str]]:
    if policy.get("schema") == "exp224_policy_bundle_v1":
        policy_hash = str(policy.get("policy_hash", ""))
        profiles = {
            str(profile["profile_id"])
            for profile in policy.get("profiles", [])
        }
    else:
        policy_hash = str(policy.get("policy_sha256", ""))
        profiles = {str(profile_id) for profile_id in policy.get("profiles", {})}
    if not policy_hash or not profiles:
        raise ValueError("ingest policy is missing hash or profiles")
    return policy_hash, profiles


def _load_ingest_policy(path: Path) -> tuple[str, set[str]]:
    candidate = json.loads(Path(path).read_text(encoding="utf-8"))
    if candidate.get("schema") == "exp224_policy_bundle_v1":
        from experiments.exp224.policy import load_verified_exp224_policy

        policy = load_verified_exp224_policy(path, verify_artifacts=True)
    else:
        policy = load_verified_policy(path, verify_artifacts=True)
    return _normalize_ingest_policy(policy)


def ingest_values(
    *,
    db_path: Path,
    source_path: Path,
    policy_path: Path,
    batch_id: str,
    collected_start: str,
    collected_end: str,
) -> dict[str, object]:
    batch_id = str(batch_id).strip()
    if not batch_id:
        raise ValueError("batch_id is required")
    start = _utc_text(collected_start, "collected_start")
    end = _utc_text(collected_end, "collected_end")
    if start > end:
        raise ValueError("collected_start must not be after collected_end")
    policy_hash, known_profiles = _load_ingest_policy(policy_path)
    rows = _load_source(source_path)
    accepted, rejected = _validate_rows(rows, known_profiles)

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO shadow_batches(
                    batch_id, collected_start_utc, collected_end_utc,
                    source_sha256, policy_sha256, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    start,
                    end,
                    sha256_file(source_path),
                    policy_hash,
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                ),
            )
            for row in accepted:
                connection.execute(
                    """
                    INSERT INTO shadow_instances(
                        batch_id, instance_id, profile_id, observed_at_utc,
                        values_blob, original_length
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        row["instance_id"],
                        row["profile_id"],
                        row["observed_at"],
                        sqlite3.Binary(row["values"].tobytes(order="C")),
                        len(row["values"]),
                    ),
                )
            for row in rejected:
                connection.execute(
                    """
                    INSERT INTO shadow_rejections(
                        batch_id, instance_id, profile_id, status
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (batch_id, row["instance_id"], row["profile_id"], row["status"]),
                )
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise ValueError(f"batch or instance already exists: {exc}") from exc

    return {
        "batch_id": batch_id,
        "accepted": len(accepted),
        "onboarding_required": len(rejected),
    }


def load_batch_instances(db_path: Path, batch_id: str) -> list[dict[str, object]]:
    with sqlite3.connect(Path(db_path)) as connection:
        rows = connection.execute(
            """
            SELECT instance_id, profile_id, observed_at_utc, values_blob, original_length
            FROM shadow_instances
            WHERE batch_id = ?
            ORDER BY instance_id
            """,
            (batch_id,),
        ).fetchall()
    return [
        {
            "instance_id": instance_id,
            "profile_id": profile_id,
            "observed_at": observed_at,
            "values": np.frombuffer(blob, dtype=np.float32, count=int(length)).copy(),
            "original_length": int(length),
        }
        for instance_id, profile_id, observed_at, blob, length in rows
    ]


def load_batch_metadata(db_path: Path, batch_id: str) -> dict[str, str]:
    with sqlite3.connect(Path(db_path)) as connection:
        row = connection.execute(
            """
            SELECT batch_id, collected_start_utc, collected_end_utc,
                   source_sha256, policy_sha256
            FROM shadow_batches WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"batch not found: {batch_id}")
    keys = (
        "batch_id",
        "collected_start_utc",
        "collected_end_utc",
        "source_sha256",
        "policy_sha256",
    )
    return dict(zip(keys, row))
