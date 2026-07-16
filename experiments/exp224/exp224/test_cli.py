from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from experiments.exp224.run_exp224 import _eligible_profile_ids, build_parser


@pytest.mark.parametrize(
    "command",
    ["build-policy", "raw-replay-audit", "ingest-values", "predict-shadow"],
)
def test_cli_exposes_exp224_commands(command: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as result:
        parser.parse_args([command, "--help"])
    assert result.value.code == 0


def test_profile_registry_excludes_db_only_profiles(tmp_path: Path) -> None:
    db_path = tmp_path / "profiles.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE datasets(name TEXT PRIMARY KEY, train_normal_count INTEGER)"
        )
        connection.executemany(
            "INSERT INTO datasets VALUES (?, ?)",
            [("profile-a", 20), ("profile-b", 30), ("db-only", 40)],
        )
    registry = tmp_path / "exp137.csv"
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset_name", "train_normal_count", "hard_f1"),
        )
        writer.writeheader()
        writer.writerow(
            {"dataset_name": "profile-a", "train_normal_count": 20, "hard_f1": 0.0}
        )
        writer.writerow(
            {"dataset_name": "profile-b", "train_normal_count": 30, "hard_f1": 1.0}
        )

    assert _eligible_profile_ids(
        db_path, registry_path=registry, expected_count=2
    ) == ["profile-a", "profile-b"]
