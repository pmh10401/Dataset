"""Task 3 tests for four-method Exp479 performance evaluation."""

from __future__ import annotations

import importlib
import json
import importlib.util
from html.parser import HTMLParser
import inspect
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import (
    METHODS_ANTIGRAVITY,
)
from experiments.exp479_missing_score_recovery_antigravity.contracts_antigravity import (
    PreparedDatasetAntigravity,
)


class _CheckboxParserAntigravity(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.checkboxes: list[dict[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if tag == "input" and values.get("type") == "checkbox":
            self.checkboxes.append(values)


def test_evaluation_module_exposes_the_task3_api_antigravity() -> None:
    """Deleting the Task 3 module or a required entry point must fail."""
    module_name = (
        "experiments.exp479_missing_score_recovery_antigravity."
        "evaluation_antigravity"
    )
    specification = importlib.util.find_spec(module_name)
    assert specification is not None, "Exp479 evaluation module is missing"
    module = importlib.import_module(module_name)
    assert hasattr(module, "score_prepared_methods_antigravity")
    assert hasattr(module, "load_retained_test_labels_antigravity")
    assert hasattr(module, "evaluate_score_vector_antigravity")
    assert hasattr(module, "evaluate_scored_dataset_antigravity")


def test_scoring_api_has_no_test_label_parameter_antigravity() -> None:
    """Adding TEST labels to feature or score generation must fail this boundary."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "evaluation_antigravity"
    )
    parameters = inspect.signature(module.score_prepared_methods_antigravity).parameters
    assert tuple(parameters) == ("prepared",)


def test_auc_and_strict_zero_false_positive_tp_match_hand_result_antigravity() -> None:
    """Using a non-strict threshold or a different AUC orientation must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "evaluation_antigravity"
    )
    metrics = module.evaluate_score_vector_antigravity(
        np.asarray([0.1, 0.4, 0.5, 0.2], dtype=np.float64),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
    )
    assert metrics == {"auc": 0.75, "zf": 1}
    tied = module.evaluate_score_vector_antigravity(
        np.asarray([0.4, 0.1, 0.4, 0.5], dtype=np.float64),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
    )
    assert tied["zf"] == 1


def _label_connection_antigravity(labels: tuple[str, ...]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
        "CREATE TABLE instances ("
        "id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL, split TEXT NOT NULL, "
        "label TEXT NOT NULL, instance_index INTEGER NOT NULL, values_blob BLOB NOT NULL);"
        "INSERT INTO datasets(id,name) VALUES (1,'Demo_normal_3');"
    )
    for index, label in enumerate(labels):
        connection.execute(
            "INSERT INTO instances(dataset_id,split,label,instance_index,values_blob) "
            "VALUES (1,'TEST',?,?,?)",
            (label, index, np.asarray([float(index)], dtype="<f4").tobytes()),
        )
    return connection


def test_test_labels_open_only_for_retained_indices_antigravity() -> None:
    """Changing index order, accepting other labels, or returning excluded rows must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "evaluation_antigravity"
    )
    connection = _label_connection_antigravity(("0", "1", "1"))
    try:
        labels = module.load_retained_test_labels_antigravity(
            connection, "Demo_normal_3", (0, 2)
        )
    finally:
        connection.close()
    np.testing.assert_array_equal(labels, np.asarray([0, 1], dtype=np.int64))
    assert labels.flags.writeable is False

    bad = _label_connection_antigravity(("0", "unexpected"))
    try:
        with pytest.raises(ValueError, match="binary"):
            module.load_retained_test_labels_antigravity(
                bad, "Demo_normal_3", (0, 1)
            )
    finally:
        bad.close()


def test_four_method_metrics_include_official_baseline_and_oracle_antigravity() -> None:
    """Dropping a method or selecting a method outside the score map must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "evaluation_antigravity"
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    scores = {
        "minirocket_local_gap": np.asarray([0.1, 0.2, 0.4, 0.3]),
        "multirocket_local_gap": np.asarray([0.2, 0.3, 0.1, 0.4]),
        "hydra_local_gap": np.asarray([0.1, 0.2, 0.8, 0.7]),
        "hydra_multirocket_local_gap": np.asarray([0.3, 0.4, 0.6, 0.5]),
    }
    result = module.evaluate_scored_dataset_antigravity(
        "Demo_normal_3", scores, labels, "hydra_local_gap"
    )
    assert tuple(result["scores"]) == METHODS_ANTIGRAVITY
    assert result["official_method"] == "hydra_local_gap"
    assert result["baseline_method"] == "minirocket_local_gap"
    assert result["official"] == result["scores"]["hydra_local_gap"]
    assert result["baseline"] == result["scores"]["minirocket_local_gap"]
    assert result["oracle_auc"] == 1.0


@pytest.mark.filterwarnings("ignore:Could not find the number of physical cores")
def test_real_exp449_scorer_returns_all_four_finite_score_vectors_antigravity() -> None:
    """Returning an empty placeholder or skipping any real candidate must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "evaluation_antigravity"
    )
    rng = np.random.default_rng(479)
    train = rng.normal(size=(10, 32)).astype(np.float64)
    test = rng.normal(size=(4, 32)).astype(np.float64)
    prepared = PreparedDatasetAntigravity(
        dataset="Synthetic_normal_0",
        train_rows=train,
        test_rows=test,
        test_indices=(0, 1, 2, 3),
        target_length=32,
        raw_train_rows=10,
        excluded_train_rows=0,
        raw_test_rows=4,
        excluded_test_rows=0,
    )

    results = module.score_prepared_methods_antigravity(prepared)

    assert tuple(results) == METHODS_ANTIGRAVITY
    assert all(result.seed == 20260721 for result in results.values())
    assert all(result.train_oof.shape == (10,) for result in results.values())
    assert all(result.test_scores.shape == (4,) for result in results.values())
    assert all(np.isfinite(result.train_oof).all() for result in results.values())
    assert all(np.isfinite(result.test_scores).all() for result in results.values())


def test_dashboard_marks_task3_and_the_actual_task6_state_antigravity() -> None:
    """Task 3 and the actual official-run lifecycle state must stay visible."""
    dashboard = (
        Path(__file__).resolve().parents[3]
        / "docs/EXP479_DASHBOARD_antigravity.html"
    ).read_text(encoding="utf-8")
    parser = _CheckboxParserAntigravity()
    parser.feed(dashboard)
    assert len(parser.checkboxes) == 10
    assert all("checked" in item for item in parser.checkboxes[:7])
    assert "Task 3 — 네 후보 성능 평가 코드 구현" in dashboard
    result_path = (
        Path(__file__).resolve().parents[3]
        / "experiments/exp479_missing_score_recovery_antigravity/results_antigravity.json"
    )
    complete = result_path.exists() and json.loads(
        result_path.read_text(encoding="utf-8")
    ).get("state") == "complete"
    assert ("checked" in parser.checkboxes[-1]) is complete
    assert ("공식 실행 완료" in dashboard) is complete
