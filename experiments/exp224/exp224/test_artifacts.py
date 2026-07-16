from __future__ import annotations

import sqlite3
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.exp224.artifacts import (
    _runtime_metadata,
    build_artifacts,
    compute_profile_artifact,
    load_train_profile,
    write_deterministic_npz,
)


def _make_profile_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                series_length INTEGER
            );
            CREATE TABLE instances (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER NOT NULL,
                split TEXT NOT NULL,
                instance_index INTEGER NOT NULL,
                label TEXT NOT NULL,
                values_blob BLOB NOT NULL
            );
            INSERT INTO datasets(id, name, series_length)
            VALUES (1, 'profile-a', 16);
            """
        )
        x = np.linspace(0.0, 2.0 * np.pi, 16, dtype=np.float32)
        rows = []
        for index in range(10):
            values = (np.sin(x + index / 10.0) + index / 100.0).astype(np.float32)
            rows.append((1, "TRAIN", index, "normal", values.tobytes()))
        rows.append(
            (1, "TEST", 0, "anomaly", np.full(16, 999.0, dtype=np.float32).tobytes())
        )
        connection.executemany(
            """
            INSERT INTO instances(dataset_id, split, instance_index, label, values_blob)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


def test_load_train_profile_ignores_test_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "profile.db"
    _make_profile_db(db_path)

    before, metadata_before = load_train_profile(db_path, "profile-a")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE instances SET values_blob = ? WHERE split = 'TEST'",
            (np.full(16, -999.0, dtype=np.float32).tobytes(),),
        )
    after, metadata_after = load_train_profile(db_path, "profile-a")

    np.testing.assert_array_equal(before, after)
    assert {
        key: value
        for key, value in metadata_before.items()
        if key != "train_series"
    } == {
        key: value
        for key, value in metadata_after.items()
        if key != "train_series"
    }
    for before_series, after_series in zip(
        metadata_before["train_series"], metadata_after["train_series"], strict=True
    ):
        np.testing.assert_array_equal(before_series, after_series)
    assert before.shape == (10, 16)


def test_artifact_probe_is_train_median_and_test_invariant(tmp_path: Path) -> None:
    db_path = tmp_path / "profile.db"
    _make_profile_db(db_path)
    train, _ = load_train_profile(db_path, "profile-a")

    before = compute_profile_artifact("profile-a", db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE instances SET values_blob = ? WHERE split = 'TEST'",
            (np.arange(16, dtype=np.float32).tobytes(),),
        )
    after = compute_profile_artifact("profile-a", db_path)

    np.testing.assert_array_equal(before["probe"], np.median(train, axis=0))
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


def test_deterministic_npz_is_byte_stable(tmp_path: Path) -> None:
    arrays = {
        "z_values": np.asarray([3.0, 1.0], dtype=np.float64),
        "a_metadata": np.asarray('{"profile_id":"profile-a"}'),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    first_hash = write_deterministic_npz(first, arrays)
    second_hash = write_deterministic_npz(second, arrays)

    assert first.read_bytes() == second.read_bytes()
    assert first_hash == second_hash


def test_train_loader_uses_frozen_linear_interpolation_for_nonfinite_values(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "profile.db"
    _make_profile_db(db_path)
    damaged = np.arange(16, dtype=np.float32)
    damaged[4:7] = np.nan
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE instances SET values_blob = ? WHERE split = 'TRAIN' AND instance_index = 0",
            (damaged.tobytes(),),
        )

    train, _ = load_train_profile(db_path, "profile-a")

    assert np.all(np.isfinite(train))
    np.testing.assert_array_equal(train[0, 4:7], np.asarray([4.0, 5.0, 6.0]))


def test_build_reuses_a_valid_existing_artifact(tmp_path: Path, monkeypatch) -> None:
    artifacts_dir = tmp_path / "artifacts"
    filename = f"{hashlib.sha256(b'profile-a').hexdigest()[:24]}.npz"
    artifact_path = artifacts_dir / filename
    metadata = {
        "profile_id": "profile-a",
        "n_train": 10,
        "train_target_length": 16,
        "train_false_counts": {
            "0.005": 0,
            "0.01": 0,
            "0.02": 0,
            "0.05": 0,
        },
        "train_trials": 10,
        "runtime": _runtime_metadata(),
    }
    write_deterministic_npz(
        artifact_path,
        {
            "canonical_oof": np.arange(10, dtype=float),
            "canonical_probe": np.asarray([0.0]),
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
            "probe": np.arange(16, dtype=np.float32),
            "temporal_oof": np.arange(10, dtype=float),
            "temporal_probe": np.asarray([0.0]),
        },
    )

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("valid artifact must be reused")

    monkeypatch.setattr(
        "experiments.exp224.artifacts.compute_profile_artifact", fail_if_recomputed
    )

    rows = build_artifacts(
        ["profile-a"], tmp_path / "unused.db", tmp_path, workers=1
    )

    assert rows[0]["artifact_path"] == str(artifact_path.resolve())
