from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from experiments.exp220_223.evaluation import (
    GATES,
    evaluate_shadow_rows,
    import_outcomes,
    wilson_bound,
)
from experiments.exp220_223.prediction import freeze_shadow_predictions


def _prediction_row(instance_id: str) -> dict[str, object]:
    return {
        "batch_id": "batch-1",
        "instance_id": instance_id,
        "profile_id": "profile-a",
        "policy_hash": "policy",
        "exp137_hard": False,
        "exp137_standard": False,
        "exp137_priority": False,
        "exp190_high": True,
        "exp190_standard": False,
        "exp195_shadow": True,
        "source_support_count": 2,
        "source_pvalues": {},
        "canonical_knn_p": 0.004,
        "exp195_zero_train_guard": True,
        "autonomous_promotion": False,
    }


def test_import_outcomes_requires_verified_frozen_prediction_and_separate_db(
    tmp_path: Path,
) -> None:
    frozen = freeze_shadow_predictions(
        rows=[_prediction_row("i-1")],
        batch_id="batch-1",
        policy_sha256="policy",
        source_sha256="source",
        output_dir=tmp_path / "predictions",
    )
    csv_path = tmp_path / "outcomes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("batch_id", "instance_id", "label", "adjudicated_at", "reviewer_id"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "batch_id": "batch-1",
                "instance_id": "i-1",
                "label": 1,
                "adjudicated_at": "2026-08-01T00:00:00Z",
                "reviewer_id": "reviewer",
            }
        )

    summary = import_outcomes(
        outcomes_db_path=tmp_path / "outcomes.db",
        outcomes_csv_path=csv_path,
        prediction_path=Path(frozen["prediction_path"]),
        manifest_path=Path(frozen["manifest_path"]),
    )
    assert summary == {"batch_id": "batch-1", "imported": 1}
    with sqlite3.connect(tmp_path / "outcomes.db") as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(shadow_outcomes)")
        }
    assert "values_blob" not in columns

    Path(frozen["prediction_path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction hash mismatch"):
        import_outcomes(
            outcomes_db_path=tmp_path / "other.db",
            outcomes_csv_path=csv_path,
            prediction_path=Path(frozen["prediction_path"]),
            manifest_path=Path(frozen["manifest_path"]),
        )


def test_evaluation_applies_preregistered_minimums_and_wilson_gates() -> None:
    predictions = []
    labels: dict[tuple[str, str], int] = {}
    batches = {
        "batch-1": ("2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"),
        "batch-2": ("2026-07-15T00:00:00Z", "2026-07-16T00:00:00Z"),
        "batch-3": ("2026-07-31T00:00:00Z", "2026-08-01T00:00:00Z"),
    }
    for index in range(100):
        batch_id = f"batch-{index % 3 + 1}"
        instance_id = f"anomaly-{index}"
        predictions.append(
            {
                **_prediction_row(instance_id),
                "batch_id": batch_id,
                "profile_id": f"profile-{index % 30}",
            }
        )
        labels[(batch_id, instance_id)] = 1
    for index in range(1000):
        batch_id = f"batch-{index % 3 + 1}"
        instance_id = f"normal-{index}"
        predictions.append(
            {
                **_prediction_row(instance_id),
                "batch_id": batch_id,
                "exp190_high": False,
                "exp195_shadow": False,
            }
        )
        labels[(batch_id, instance_id)] = 0

    report = evaluate_shadow_rows(predictions, labels, batches)
    assert report["lanes"]["exp190_high"]["decision"] == "promotion_review_eligible"
    assert report["lanes"]["exp195_shadow"]["decision"] == "promotion_review_eligible"
    assert report["lanes"]["exp190_standard"]["decision"] == "deferred"
    assert report["lanes"]["exp190_high"]["normal_false_alert_rate_upper"] <= 0.005
    assert wilson_bound(100, 100, side="lower") > 0.9


def _exp224_evaluation_fixture(
    *, candidates: int, true_positives: int
) -> tuple[list[dict[str, object]], dict[tuple[str, str], int], dict[str, tuple[str, str]]]:
    batches = {
        "batch-1": ("2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"),
        "batch-2": ("2026-07-15T00:00:00Z", "2026-07-16T00:00:00Z"),
        "batch-3": ("2026-07-31T00:00:00Z", "2026-08-01T00:00:00Z"),
    }
    predictions: list[dict[str, object]] = []
    labels: dict[tuple[str, str], int] = {}
    for index in range(candidates):
        batch_id = f"batch-{index % 3 + 1}"
        instance_id = f"candidate-{index}"
        predictions.append(
            {
                **_prediction_row(instance_id),
                "batch_id": batch_id,
                "profile_id": f"profile-{index % 30}",
                "exp190_high": False,
                "exp195_shadow": False,
                "exp224_shadow": True,
            }
        )
        labels[(batch_id, instance_id)] = int(index < true_positives)
    for index in range(10_000):
        batch_id = f"batch-{index % 3 + 1}"
        instance_id = f"normal-{index}"
        predictions.append(
            {
                **_prediction_row(instance_id),
                "batch_id": batch_id,
                "exp190_high": False,
                "exp195_shadow": False,
                "exp224_shadow": False,
            }
        )
        labels[(batch_id, instance_id)] = 0
    return predictions, labels, batches


def test_exp224_gate_is_deferred_below_preregistered_sample() -> None:
    predictions, labels, batches = _exp224_evaluation_fixture(
        candidates=99, true_positives=99
    )

    report = evaluate_shadow_rows(predictions, labels, batches)

    assert report["lanes"]["exp224_shadow"]["decision"] == "deferred"


def test_exp224_gate_rejects_when_precision_lower_misses_threshold() -> None:
    predictions, labels, batches = _exp224_evaluation_fixture(
        candidates=100, true_positives=90
    )

    report = evaluate_shadow_rows(predictions, labels, batches)

    assert report["lanes"]["exp224_shadow"]["decision"] == "rejected"


def test_exp224_gate_is_only_promotion_review_eligible_at_all_boundaries() -> None:
    predictions, labels, batches = _exp224_evaluation_fixture(
        candidates=100, true_positives=100
    )

    report = evaluate_shadow_rows(predictions, labels, batches)
    lane = report["lanes"]["exp224_shadow"]

    assert lane["decision"] == "promotion_review_eligible"
    assert lane["precision_lower"] >= 0.90
    assert lane["normal_false_alert_rate_upper"] <= 0.005
    assert GATES["exp190_high"]["precision_lower"] == 0.80
    assert GATES["exp195_shadow"]["false_alert_upper"] == 0.005
