from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from experiments.exp220_223.policy import freeze_policy
from experiments.exp220_223.storage import (
    _normalize_ingest_policy,
    ingest_values,
    load_batch_instances,
)


def _policy(tmp_path: Path) -> Path:
    asset = tmp_path / "asset.txt"
    asset.write_text("frozen\n", encoding="utf-8")
    path = tmp_path / "policy.json"
    freeze_policy(
        path,
        artifacts={"asset": asset},
        profiles={
            "profile-a": {"n_train": 20, "train_target_len": 3},
            "profile-b": {"n_train": 30, "train_target_len": 4},
        },
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_ingest_separates_unregistered_profiles_and_sorts_instances(tmp_path: Path) -> None:
    source = tmp_path / "values.jsonl"
    _write_jsonl(
        source,
        [
            {"instance_id": "z", "profile_id": "profile-a", "observed_at": "2026-07-02T00:00:00Z", "values": [1, 2, 3]},
            {"instance_id": "a", "profile_id": "unknown", "observed_at": "2026-07-01T00:00:00Z", "values": [4, 5]},
            {"instance_id": "b", "profile_id": "profile-b", "observed_at": "2026-07-01T01:00:00Z", "values": [6, 7, 8, 9]},
        ],
    )

    summary = ingest_values(
        db_path=tmp_path / "values.db",
        source_path=source,
        policy_path=_policy(tmp_path),
        batch_id="batch-1",
        collected_start="2026-07-01T00:00:00Z",
        collected_end="2026-07-02T00:00:00Z",
    )

    assert summary == {"batch_id": "batch-1", "accepted": 2, "onboarding_required": 1}
    instances = load_batch_instances(tmp_path / "values.db", "batch-1")
    assert [row["instance_id"] for row in instances] == ["b", "z"]
    assert instances[0]["values"].tolist() == [6.0, 7.0, 8.0, 9.0]
    with sqlite3.connect(tmp_path / "values.db") as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(shadow_instances)")
        }
        rejected = connection.execute(
            "SELECT status FROM shadow_rejections"
        ).fetchone()[0]
    assert "label" not in columns
    assert rejected == "onboarding_required"


@pytest.mark.parametrize("bad_values", [[], [1.0, float("nan")], [float("inf")]])
def test_ingest_rejects_invalid_values_atomically(
    tmp_path: Path, bad_values: list[float]
) -> None:
    source = tmp_path / "bad.jsonl"
    _write_jsonl(
        source,
        [
            {"instance_id": "ok", "profile_id": "profile-a", "values": [1, 2]},
            {"instance_id": "bad", "profile_id": "profile-a", "values": bad_values},
        ],
    )
    db_path = tmp_path / "values.db"

    with pytest.raises(ValueError, match="finite|empty"):
        ingest_values(
            db_path=db_path,
            source_path=source,
            policy_path=_policy(tmp_path),
            batch_id="batch-1",
            collected_start="2026-07-01T00:00:00Z",
            collected_end="2026-07-02T00:00:00Z",
        )

    if db_path.exists():
        with sqlite3.connect(db_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM shadow_batches").fetchone()[0] == 0


def test_ingest_rejects_duplicate_instance_ids(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.jsonl"
    row = {"instance_id": "same", "profile_id": "profile-a", "values": [1, 2]}
    _write_jsonl(source, [row, row])

    with pytest.raises(ValueError, match="duplicate instance_id"):
        ingest_values(
            db_path=tmp_path / "values.db",
            source_path=source,
            policy_path=_policy(tmp_path),
            batch_id="batch-1",
            collected_start="2026-07-01T00:00:00Z",
            collected_end="2026-07-02T00:00:00Z",
        )


def test_ingest_normalizes_exp224_policy_contract() -> None:
    policy_hash, profiles = _normalize_ingest_policy(
        {
            "schema": "exp224_policy_bundle_v1",
            "policy_hash": "hash-224",
            "profiles": [
                {"profile_id": "profile-b"},
                {"profile_id": "profile-a"},
            ],
        }
    )

    assert policy_hash == "hash-224"
    assert profiles == {"profile-a", "profile-b"}
