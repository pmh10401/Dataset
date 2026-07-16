from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from experiments.exp224.artifacts import _runtime_metadata, write_deterministic_npz
from experiments.exp224.scoring import (
    Exp224EvidenceProvider,
    assert_shadow_gate,
    build_exp224_rows,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    train_db = tmp_path / "train.db"
    with sqlite3.connect(train_db) as connection:
        connection.executescript(
            """
            CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT UNIQUE, series_length INTEGER);
            CREATE TABLE instances (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER,
                split TEXT,
                instance_index INTEGER,
                label TEXT,
                values_blob BLOB
            );
            INSERT INTO datasets VALUES (1, 'profile-a', 8);
            """
        )
        for index in range(10):
            values = np.full(8, index, dtype=np.float32)
            connection.execute(
                "INSERT INTO instances(dataset_id, split, instance_index, label, values_blob) VALUES (1, 'TRAIN', ?, 'normal', ?)",
                (index, values.tobytes()),
            )

    artifact = tmp_path / "artifact.npz"
    write_deterministic_npz(
        artifact,
        {
            "temporal_oof": np.arange(10, dtype=float),
            "canonical_oof": np.arange(10, dtype=float),
        },
    )
    upstream = tmp_path / "exp137.jsonl"
    upstream_rows = [
        {
            "batch_id": "batch-1",
            "instance_id": "a",
            "profile_id": "profile-a",
            "exp137_hard": False,
            "exp137_standard": False,
            "exp137_priority": False,
        },
        {
            "batch_id": "batch-1",
            "instance_id": "z",
            "profile_id": "profile-a",
            "exp137_hard": True,
            "exp137_standard": False,
            "exp137_priority": False,
        },
    ]
    upstream.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in upstream_rows),
        encoding="utf-8",
    )
    upstream_manifest = tmp_path / "exp137.manifest.json"
    upstream_manifest.write_text(
        json.dumps(
            {
                "batch_id": "batch-1",
                "contains_test_labels": False,
                "prediction_sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    policy = {
        "policy_hash": "policy-224",
        "selected_alpha": 0.1,
        "runtime": _runtime_metadata(),
        "profiles": [
            {
                "profile_id": "profile-a",
                "artifact_path": str(artifact),
                "train_target_length": 8,
            }
        ],
    }
    return train_db, upstream, upstream_manifest, policy


def _fake_backend(profile_id, train, query, record):
    del profile_id, train, record
    scores = np.full(len(query), 10.0, dtype=float)
    return scores, scores


def _instances() -> list[dict[str, object]]:
    return [
        {
            "instance_id": "z",
            "profile_id": "profile-a",
            "values": np.full(8, 2.0, dtype=np.float32),
        },
        {
            "instance_id": "a",
            "profile_id": "profile-a",
            "values": np.full(8, 1.0, dtype=np.float32),
        },
    ]


def test_fake_scoring_is_order_invariant_and_exp137_is_excluded(tmp_path: Path) -> None:
    train_db, upstream, upstream_manifest, policy = _fixture(tmp_path)
    provider = Exp224EvidenceProvider(
        train_db_path=train_db,
        exp137_predictions_path=upstream,
        exp137_manifest_path=upstream_manifest,
        batch_id="batch-1",
        score_backend=_fake_backend,
    )

    forward = provider(_instances(), policy)
    reverse = provider(list(reversed(_instances())), policy)
    assert forward == reverse

    rows = build_exp224_rows(
        batch_id="batch-1",
        instances=_instances(),
        policy=policy,
        evidence=forward,
    )
    assert [row["instance_id"] for row in rows] == ["a", "z"]
    assert rows[0]["exp224_shadow"] is True
    assert rows[1]["exp224_shadow"] is False
    assert rows[1]["exp137_excluded"] is True


def test_runtime_mismatch_rejects_scoring(tmp_path: Path) -> None:
    train_db, upstream, upstream_manifest, policy = _fixture(tmp_path)
    policy["runtime"] = {**policy["runtime"], "python": "mutated"}
    provider = Exp224EvidenceProvider(
        train_db_path=train_db,
        exp137_predictions_path=upstream,
        exp137_manifest_path=upstream_manifest,
        batch_id="batch-1",
        score_backend=_fake_backend,
    )

    with pytest.raises(ValueError, match="runtime mismatch"):
        provider(_instances(), policy)


def test_nonpassed_raw_report_rejects_real_shadow() -> None:
    policy = {"policy_hash": "policy-224"}
    report = {
        "policy_hash": "policy-224",
        "profile_count": 639,
        "expected_profile_count": 639,
        "mismatch_count": 0,
        "shadow_execution_gate": "blocked_raw_replay_mismatch",
    }

    with pytest.raises(ValueError, match="raw replay gate"):
        assert_shadow_gate(policy, report)


def test_passed_report_must_cover_exactly_639_profiles() -> None:
    policy = {"policy_hash": "policy-224"}
    report = {
        "policy_hash": "policy-224",
        "profile_count": 638,
        "expected_profile_count": 639,
        "mismatch_count": 0,
        "shadow_execution_gate": "passed",
    }

    with pytest.raises(ValueError, match="639/639"):
        assert_shadow_gate(policy, report)

