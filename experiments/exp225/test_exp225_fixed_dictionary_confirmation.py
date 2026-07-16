from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_core():
    spec = importlib.util.find_spec("experiments.exp225.core")
    assert spec is not None, "Exp225 core module must exist"
    return importlib.import_module("experiments.exp225.core")


def load_runner():
    spec = importlib.util.find_spec("experiments.exp225.run_exp225")
    assert spec is not None, "Exp225 runner module must exist"
    return importlib.import_module("experiments.exp225.run_exp225")


def test_confirmation_lanes_are_exclusive_and_candidate_bounded() -> None:
    core = load_core()
    lanes = core.confirmation_lanes(
        candidate_indices={0, 1, 2, 3},
        temporal_p=np.asarray([0.001, 0.20, 0.005, 0.30, 0.001]),
        canonical_p=np.asarray([0.002, 0.004, 0.50, 0.40, 0.001]),
        alpha=0.01,
    )

    assert lanes["dual_confirmed_review"] == {0}
    assert lanes["single_confirmed_review"] == {1, 2}
    assert lanes["unconfirmed"] == {3}
    assert set().union(*lanes.values()) == {0, 1, 2, 3}
    assert not (
        lanes["dual_confirmed_review"] & lanes["single_confirmed_review"]
    )


def test_confirmation_rejects_out_of_range_candidate() -> None:
    core = load_core()
    with pytest.raises(ValueError, match="outside score range"):
        core.confirmation_lanes(
            candidate_indices={2},
            temporal_p=np.asarray([0.1, 0.2]),
            canonical_p=np.asarray([0.1, 0.2]),
            alpha=0.01,
        )


def test_prediction_payload_is_review_only_and_label_free() -> None:
    core = load_core()
    payload = core.build_prediction_payload(
        dataset="profile-a",
        n_test=3,
        candidate_indices={0, 2},
        temporal_p=np.asarray([0.001, 0.5, 0.002]),
        canonical_p=np.asarray([0.002, 0.5, 0.2]),
        alpha=0.01,
    )

    assert payload["autonomous_alert"] is False
    assert payload["contains_test_labels"] is False
    assert payload["contains_evaluation_metrics"] is False
    assert payload["dual_confirmed_review_indices"] == [0]
    assert payload["single_confirmed_review_indices"] == [2]
    forbidden = {"y_test", "tp", "fp", "f1", "precision", "recall"}
    assert forbidden.isdisjoint({key.lower() for key in payload})


def test_prediction_manifest_hash_is_verified(tmp_path: Path) -> None:
    runner = load_runner()
    prediction = tmp_path / "predictions.jsonl"
    prediction.write_text(
        json.dumps(
            {
                "experiment_id": runner.EXP219_ID,
                "dataset": "profile-a",
                "variant": runner.EXP219_PRIMARY_VARIANT,
                "n_test": 2,
                "candidate_indices": [1],
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "experiment_id": runner.EXP219_ID,
                "prediction_file": prediction.name,
                "prediction_sha256": "0" * 64,
                "prediction_rows": 1,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prediction hash mismatch"):
        runner.load_verified_exp219_primary(prediction, manifest, expected_datasets=1)


def test_summary_reports_review_metrics_without_autonomous_promotion() -> None:
    runner = load_runner()
    summary = runner.summarize_evaluated(
        [
            {
                "candidate_count": 3,
                "candidate_tp": 1,
                "candidate_fp": 2,
                "dual_review_count": 1,
                "dual_review_tp": 1,
                "dual_review_fp": 0,
                "single_review_count": 1,
                "single_review_tp": 0,
                "single_review_fp": 1,
            }
        ]
    )

    assert summary["candidate_count"] == 3
    assert summary["confirmed_review_count"] == 2
    assert summary["confirmed_review_tp"] == 1
    assert summary["confirmed_review_fp"] == 1
    assert summary["confirmed_review_precision"] == 0.5
    assert summary["autonomous_alert_count"] == 0
    assert summary["selected_for_operation"] == 0
