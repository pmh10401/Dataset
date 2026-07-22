from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.exp220_223.scoring import (
    RawEvidenceProvider,
    align_to_train_length,
    exp195_zero_train_guard,
    run_source_crossfit,
)


def test_alignment_uses_only_frozen_train_target_length() -> None:
    assert align_to_train_length(np.asarray([0.0, 10.0]), 3).tolist() == [0.0, 5.0, 10.0]
    assert align_to_train_length(np.asarray([2.0]), 4).tolist() == [2.0, 2.0, 2.0, 2.0]


def test_exp195_guard_uses_crossfit_train_normal_candidates_only() -> None:
    source_scores = {
        name: np.zeros(300, dtype=float)
        for name in ("a", "b", "c", "d")
    }
    canonical = np.zeros(300, dtype=float)
    assert exp195_zero_train_guard(source_scores, canonical, conservative_alpha=0.01) is True

    source_scores["a"][-1] = 10_000.0
    source_scores["b"][-1] = 10_000.0
    canonical[-1] = 10_000.0
    assert exp195_zero_train_guard(source_scores, canonical, conservative_alpha=0.01) is False


def test_raw_provider_requires_complete_external_exp137_lanes(tmp_path: Path) -> None:
    upstream = tmp_path / "exp137.jsonl"
    upstream.write_text(
        json.dumps(
            {
                "batch_id": "batch-1",
                "instance_id": "one",
                "profile_id": "profile-a",
                "exp137_hard": False,
                "exp137_standard": False,
                "exp137_priority": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provider = RawEvidenceProvider(
        train_db_path=tmp_path / "unused.db",
        exp137_predictions_path=upstream,
        exp137_manifest_path=tmp_path / "exp137_manifest.json",
        batch_id="batch-1",
    )
    (tmp_path / "exp137_manifest.json").write_text(
        json.dumps(
            {
                "batch_id": "batch-1",
                "prediction_sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
                "contains_test_labels": False,
            }
        ),
        encoding="utf-8",
    )
    instances = [
        {"instance_id": "one", "profile_id": "profile-a", "values": np.ones(3)},
        {"instance_id": "two", "profile_id": "profile-a", "values": np.ones(3)},
    ]
    policy = {"profiles": {"profile-a": {"train_target_len": 3}}}

    with pytest.raises(ValueError, match="Exp137 upstream coverage"):
        provider(instances, policy)


def test_raw_provider_rejects_tampered_exp137_upstream(tmp_path: Path) -> None:
    upstream = tmp_path / "exp137.jsonl"
    upstream.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "batch_id": "batch-1",
                "prediction_sha256": "0" * 64,
                "contains_test_labels": False,
            }
        ),
        encoding="utf-8",
    )
    provider = RawEvidenceProvider(
        train_db_path=tmp_path / "unused.db",
        exp137_predictions_path=upstream,
        exp137_manifest_path=manifest,
        batch_id="batch-1",
    )
    with pytest.raises(ValueError, match="Exp137 upstream hash mismatch"):
        provider([], {"profiles": {}})


def test_source_crossfit_uses_exp152_owner_exposed_by_bridge() -> None:
    class FakeExp152:
        @staticmethod
        def crossfit_source(*args):
            return args

    class FakeBridge:
        exp152 = FakeExp152()

    result = run_source_crossfit(FakeBridge(), "source", "train", "query", "record")
    assert result == ("source", "train", "query", "record")
