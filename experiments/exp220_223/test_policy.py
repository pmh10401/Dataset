from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.exp220_223.policy import (
    default_artifact_paths,
    freeze_policy,
    load_verified_policy,
    runtime_versions,
)


def test_freeze_policy_hashes_assets_and_detects_tampering(tmp_path: Path) -> None:
    asset = tmp_path / "frozen.json"
    asset.write_text('{"alpha": 0.005}\n', encoding="utf-8")
    bundle_path = tmp_path / "policy_bundle.json"

    frozen = freeze_policy(
        bundle_path,
        artifacts={"frozen_policy": asset},
        profiles={
            "profile-a": {
                "n_train": 20,
                "train_target_len": 16,
                "exp195_zero_train_guard": True,
            }
        },
    )

    assert frozen["policy_sha256"] == load_verified_policy(bundle_path)["policy_sha256"]
    assert frozen["artifacts"]["frozen_policy"]["sha256"]
    assert json.loads(bundle_path.read_text(encoding="utf-8"))["profiles"] == frozen["profiles"]

    asset.write_text('{"alpha": 0.01}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_verified_policy(bundle_path, verify_artifacts=True)


def test_default_policy_includes_shadow_scoring_and_prediction_code(tmp_path: Path) -> None:
    artifacts = default_artifact_paths(tmp_path, tmp_path, tmp_path / "db.sqlite")
    assert artifacts["exp220_scoring_code"] == tmp_path / "experiments/exp220_223/scoring.py"
    assert artifacts["exp220_prediction_code"] == tmp_path / "experiments/exp220_223/prediction.py"
    assert artifacts["exp223_evaluation_code"] == tmp_path / "experiments/exp220_223/evaluation.py"
    assert artifacts["exp221_storage_code"] == tmp_path / "experiments/exp220_223/storage.py"


def test_runtime_versions_freeze_python_numpy_sklearn_and_torch() -> None:
    versions = runtime_versions()
    assert {"python", "numpy", "scikit-learn", "torch"} <= set(versions)
