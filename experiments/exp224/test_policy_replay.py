from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp224.artifacts import write_deterministic_npz
from experiments.exp224.policy import freeze_exp224_policy, load_verified_exp224_policy
from experiments.exp224.replay import raw_replay_gate


def _write_manifest(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact_path = tmp_path / "artifact.npz"
    artifact_hash = write_deterministic_npz(
        artifact_path,
        {
            "canonical_oof": np.arange(10, dtype=float),
            "canonical_probe": np.asarray([0.5]),
            "metadata_json": np.asarray(
                json.dumps(
                    {
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
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "probe": np.arange(16, dtype=np.float32),
            "temporal_oof": np.arange(10, dtype=float),
            "temporal_probe": np.asarray([0.5]),
        },
    )
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "profile_id",
                "artifact_path",
                "artifact_sha256",
                "n_train",
                "train_target_length",
                "train_trials",
                "train_false_counts_json",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "profile_id": "profile-a",
                "artifact_path": str(artifact_path),
                "artifact_sha256": artifact_hash,
                "n_train": 10,
                "train_target_length": 16,
                "train_trials": 10,
                "train_false_counts_json": (
                    '{"0.005":0,"0.01":0,"0.02":0,"0.05":0}'
                ),
            }
        )
    db_path = tmp_path / "train.db"
    db_path.write_bytes(b"TRAIN-only database fixture")
    return manifest_path, artifact_path, db_path


def _freeze(tmp_path: Path) -> tuple[Path, Path]:
    manifest_path, artifact_path, db_path = _write_manifest(tmp_path)
    policy_path = tmp_path / "policy.json"
    freeze_exp224_policy(
        manifest_path,
        {
            "expected_profile_count": 1,
            "false_counts": {"0.005": 0, "0.01": 0, "0.02": 0, "0.05": 0},
            "trials": 10,
        },
        {
            "db_path": db_path,
            "source_paths": {
                "core": Path(__file__).with_name("core.py"),
                "artifacts": Path(__file__).with_name("artifacts.py"),
            },
        },
        policy_path,
    )
    return policy_path, artifact_path


def test_policy_rejects_artifact_tampering(tmp_path: Path) -> None:
    policy_path, artifact_path = _freeze(tmp_path)
    load_verified_exp224_policy(policy_path)

    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_verified_exp224_policy(policy_path)


def test_policy_rejects_runtime_mutation(tmp_path: Path) -> None:
    policy_path, _ = _freeze(tmp_path)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    payload["runtime"]["python"] = "0.0-mutated"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="policy hash mismatch"):
        load_verified_exp224_policy(policy_path, verify_artifacts=False)


def test_replay_gate_never_opens_on_partial_coverage() -> None:
    assert raw_replay_gate(638, 639, mismatch_count=0, risk_match=True) == (
        "blocked_incomplete_raw_replay"
    )


def test_replay_gate_never_accepts_a_reduced_expected_profile_count() -> None:
    assert raw_replay_gate(1, 1, mismatch_count=0, risk_match=True) == (
        "blocked_incomplete_raw_replay"
    )


def test_replay_gate_blocks_any_score_mismatch() -> None:
    assert raw_replay_gate(639, 639, mismatch_count=1, risk_match=True) == (
        "blocked_raw_replay_mismatch"
    )


def test_replay_gate_opens_only_for_complete_exact_replay() -> None:
    assert raw_replay_gate(639, 639, mismatch_count=0, risk_match=True) == "passed"
    assert raw_replay_gate(639, 639, mismatch_count=0, risk_match=False) == (
        "blocked_raw_replay_mismatch"
    )
