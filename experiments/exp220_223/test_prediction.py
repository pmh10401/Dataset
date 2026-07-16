from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from experiments.exp220_223.policy import freeze_policy
from experiments.exp220_223.prediction import (
    freeze_shadow_predictions,
    predict_batch,
    verify_prediction_manifest,
)
from experiments.exp220_223.storage import ingest_values


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    asset = tmp_path / "asset.txt"
    asset.write_text("frozen\n", encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    freeze_policy(
        policy_path,
        artifacts={"asset": asset},
        profiles={
            "profile-a": {"n_train": 20, "train_target_len": 3},
        },
    )
    source = tmp_path / "values.jsonl"
    source.write_text(
        json.dumps({"instance_id": "z", "profile_id": "profile-a", "values": [1, 2, 3]}) + "\n"
        + json.dumps({"instance_id": "a", "profile_id": "profile-a", "values": [4, 5, 6]}) + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "values.db"
    ingest_values(
        db_path=db_path,
        source_path=source,
        policy_path=policy_path,
        batch_id="batch-1",
        collected_start="2026-07-01T00:00:00Z",
        collected_end="2026-07-02T00:00:00Z",
    )
    return db_path, policy_path


def _provider(instances, policy):
    del policy
    return {
        row["instance_id"]: {
            "exp137_hard": row["instance_id"] == "z",
            "exp137_standard": False,
            "exp137_priority": False,
            "source_pvalues": {
                "temporal_random_kernel": 0.001,
                "spectral_image": 0.002,
                "texture_image": 0.5,
                "interval_statistical": 0.8,
            },
            "canonical_knn_p": 0.004,
            "exp195_zero_train_guard": True,
        }
        for row in instances
    }


def test_prediction_is_label_free_sorted_and_byte_stable(tmp_path: Path) -> None:
    db_path, policy_path = _fixture(tmp_path)
    first = predict_batch(
        db_path=db_path,
        batch_id="batch-1",
        policy_path=policy_path,
        output_dir=tmp_path / "one",
        evidence_provider=_provider,
    )
    second = predict_batch(
        db_path=db_path,
        batch_id="batch-1",
        policy_path=policy_path,
        output_dir=tmp_path / "two",
        evidence_provider=_provider,
    )

    assert first["prediction_sha256"] == second["prediction_sha256"]
    rows = [json.loads(line) for line in Path(first["prediction_path"]).read_text().splitlines()]
    assert [row["instance_id"] for row in rows] == ["a", "z"]
    assert rows[0]["exp190_high"] is True
    assert rows[0]["exp195_shadow"] is True
    assert rows[1]["exp190_high"] is False  # excluded by the frozen Exp137 lane
    forbidden = {"label", "tp", "fp", "fn", "f1", "test_position", "test_length"}
    assert all(not (forbidden & set(row)) for row in rows)
    assert "outcomes" not in inspect.signature(predict_batch).parameters
    verify_prediction_manifest(first["prediction_path"], first["manifest_path"])


def test_prediction_manifest_detects_tampering(tmp_path: Path) -> None:
    db_path, policy_path = _fixture(tmp_path)
    frozen = predict_batch(
        db_path=db_path,
        batch_id="batch-1",
        policy_path=policy_path,
        output_dir=tmp_path / "out",
        evidence_provider=_provider,
    )
    Path(frozen["prediction_path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction hash mismatch"):
        verify_prediction_manifest(frozen["prediction_path"], frozen["manifest_path"])


def test_freeze_rejects_evaluation_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evaluation field"):
        freeze_shadow_predictions(
            rows=[{"batch_id": "b", "instance_id": "i", "label": 1}],
            batch_id="b",
            policy_sha256="abc",
            source_sha256="def",
            output_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_freeze_accepts_exp224_row_without_weakening_legacy_contract(tmp_path: Path) -> None:
    frozen = freeze_shadow_predictions(
        rows=[
            {
                "batch_id": "batch-224",
                "instance_id": "i-1",
                "profile_id": "profile-a",
                "policy_hash": "policy-224",
                "exp137_hard": False,
                "exp137_standard": False,
                "exp137_priority": False,
                "temporal_p": 0.001,
                "canonical_p": 0.002,
                "selected_alpha": 0.005,
                "exp137_excluded": False,
                "exp224_shadow": True,
                "autonomous_promotion": False,
            }
        ],
        batch_id="batch-224",
        policy_sha256="policy-224",
        source_sha256="source",
        output_dir=tmp_path,
    )

    _, rows = verify_prediction_manifest(
        Path(frozen["prediction_path"]), Path(frozen["manifest_path"])
    )
    assert rows[0]["exp224_shadow"] is True
