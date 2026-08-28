"""Task 2 tests for the Exp479 Numba preprocessing path."""

from __future__ import annotations

import importlib
import importlib.util
from html.parser import HTMLParser
import json
from pathlib import Path
import sqlite3

from numba.core.registry import CPUDispatcher
import numpy as np
import pytest

from experiments.exp474_conservative_fallback_routing_antigravity.run_antigravity import (
    CHECKPOINT_DIR_ANTIGRAVITY,
)
from experiments.exp478_full_1117_official_routing_antigravity.run_antigravity import (
    DATABASE_PATH_ANTIGRAVITY,
    RESULT_PATH_ANTIGRAVITY as EXP478_RESULT_PATH_ANTIGRAVITY,
)
from experiments.exp479_missing_score_recovery_antigravity.population_antigravity import (
    build_population_manifest_antigravity,
    select_recovery_population_antigravity,
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


def test_preprocessing_module_exposes_the_task2_api_antigravity() -> None:
    """Deleting the Task 2 module or a required entry point must fail."""
    module_name = (
        "experiments.exp479_missing_score_recovery_antigravity."
        "preprocessing_antigravity"
    )
    specification = importlib.util.find_spec(module_name)
    assert specification is not None, "Exp479 preprocessing module is missing"
    module = importlib.import_module(module_name)
    assert hasattr(module, "lower_median_length_antigravity")
    assert hasattr(module, "resample_row_linear_antigravity")
    assert hasattr(module, "prepare_dataset_values_antigravity")
    assert hasattr(module, "audit_recovery_preprocessing_antigravity")


def test_lower_median_uses_the_smaller_middle_length_antigravity() -> None:
    """Averaging the two central lengths or choosing the upper one must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "preprocessing_antigravity"
    )
    assert module.lower_median_length_antigravity((32, 4, 16, 8)) == 8
    assert module.lower_median_length_antigravity((9, 3, 6)) == 6
    with pytest.raises(ValueError, match="non-empty"):
        module.lower_median_length_antigravity(())


def test_numba_linear_resampling_preserves_equal_rows_and_endpoints_antigravity() -> None:
    """Using a NumPy fallback, changing equal rows, or moving endpoints must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "preprocessing_antigravity"
    )
    assert isinstance(module.resample_row_linear_antigravity, CPUDispatcher)

    source = np.asarray([0.0, 10.0], dtype=np.float64)
    np.testing.assert_array_equal(
        module.resample_row_linear_antigravity(source, 3),
        np.asarray([0.0, 5.0, 10.0], dtype=np.float64),
    )
    equal = np.asarray([1.25, -2.5, 7.0], dtype=np.float64)
    actual = module.resample_row_linear_antigravity(equal, 3)
    np.testing.assert_array_equal(actual, equal)
    assert actual is not equal


def _connection_antigravity(test_labels: tuple[str, str, str]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
        "CREATE TABLE instances ("
        "id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL, split TEXT NOT NULL, "
        "label TEXT NOT NULL, instance_index INTEGER NOT NULL, values_blob BLOB NOT NULL);"
        "INSERT INTO datasets(id,name) VALUES (1,'Demo_normal_7');"
    )
    train_rows = (
        (0, "0", [0.0, 1.0, 2.0, 3.0]),
        (1, "0", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (2, "0", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        (3, "0", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]),
        (4, "0", [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0]),
        (5, "0", [0.0, 1.0, np.nan, 3.0]),
        (6, "1", [100.0, 101.0, 102.0, 103.0]),
    )
    test_rows = (
        (0, test_labels[0], [0.0, 2.0]),
        (1, test_labels[1], [1.0, 3.0, 5.0]),
        (2, test_labels[2], [0.0, np.inf, 2.0]),
    )
    for split, rows in (("TRAIN", train_rows), ("TEST", test_rows)):
        for index, label, values in rows:
            blob = np.asarray(values, dtype="<f4").tobytes()
            connection.execute(
                "INSERT INTO instances(dataset_id,split,label,instance_index,values_blob) "
                "VALUES (1,?,?,?,?)",
                (split, label, index, blob),
            )
    return connection


def test_preparation_excludes_whole_nonfinite_rows_and_ignores_test_labels_antigravity() -> None:
    """Imputing a value, retaining a TRAIN anomaly, or reading TEST labels must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "preprocessing_antigravity"
    )
    first_connection = _connection_antigravity(("0", "1", "1"))
    second_connection = _connection_antigravity(("changed", "labels", "ignored"))
    try:
        first = module.prepare_dataset_values_antigravity(
            first_connection, "Demo_normal_7"
        )
        second = module.prepare_dataset_values_antigravity(
            second_connection, "Demo_normal_7"
        )
    finally:
        first_connection.close()
        second_connection.close()

    assert first.target_length == 8
    assert first.raw_train_rows == 6
    assert first.excluded_train_rows == 1
    assert first.raw_test_rows == 3
    assert first.excluded_test_rows == 1
    assert first.test_indices == (0, 1)
    assert first.train_rows.shape == (5, 8)
    assert first.test_rows.shape == (2, 8)
    np.testing.assert_allclose(first.train_rows.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(first.train_rows.std(axis=1), 1.0, atol=1e-12)
    np.testing.assert_array_equal(first.train_rows, second.train_rows)
    np.testing.assert_array_equal(first.test_rows, second.test_rows)
    assert first.test_indices == second.test_indices
    assert first.train_rows.flags.writeable is False
    assert first.test_rows.flags.writeable is False


def test_all_155_recovery_datasets_pass_the_value_only_preprocessing_audit_antigravity() -> None:
    """Leaving any recovery dataset unprepared or hiding a failure must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "preprocessing_antigravity"
    )
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
    )
    try:
        population = select_recovery_population_antigravity(
            connection, CHECKPOINT_DIR_ANTIGRAVITY
        )
        audit = module.audit_recovery_preprocessing_antigravity(
            connection, population.datasets
        )
    finally:
        connection.close()

    assert audit["schema"] == "exp479_preprocessing_audit_v1"
    assert audit["summary"]["dataset_count"] == 155
    assert audit["summary"]["success"] == 155
    assert audit["summary"]["failures"] == 0
    assert len(audit["rows"]) == 155
    assert [row["dataset"] for row in audit["rows"]] == list(population.datasets)
    assert all(row["retained_train_rows"] >= 5 for row in audit["rows"])
    assert all(row["retained_test_rows"] >= 2 for row in audit["rows"])
    assert all(row["target_length"] >= 2 for row in audit["rows"])


def test_task2_artifacts_show_the_real_audit_and_only_completed_checks_antigravity() -> None:
    """A stale audit, hidden exclusions, or prematurely checked future task must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "preprocessing_antigravity"
    )
    assert hasattr(module, "render_task2_dashboard_antigravity")
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
    )
    try:
        population = select_recovery_population_antigravity(
            connection, CHECKPOINT_DIR_ANTIGRAVITY
        )
        manifest = build_population_manifest_antigravity(
            population,
            checkpoint_dir=CHECKPOINT_DIR_ANTIGRAVITY,
            exp478_result_path=EXP478_RESULT_PATH_ANTIGRAVITY,
            database_path=DATABASE_PATH_ANTIGRAVITY,
        )
        audit = module.audit_recovery_preprocessing_antigravity(
            connection, population.datasets
        )
    finally:
        connection.close()

    html = module.render_task2_dashboard_antigravity(population, manifest, audit)
    parser = _CheckboxParserAntigravity()
    parser.feed(html)
    assert "성공 155" in html and "실패 0" in html
    assert "TRAIN 비유한 행 제외" in html and "164" in html
    assert "TEST 비유한 행 제외" in html and "235" in html
    assert len(parser.checkboxes) == 10
    assert sum("checked" in item for item in parser.checkboxes) == 6

    root = Path(__file__).resolve().parents[3]
    audit_path = (
        root
        / "experiments/exp479_missing_score_recovery_antigravity/"
        "preprocessing_audit_antigravity.json"
    )
    dashboard_path = root / "docs/EXP479_DASHBOARD_antigravity.html"
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit
    current_dashboard = dashboard_path.read_text(encoding="utf-8")
    current_parser = _CheckboxParserAntigravity()
    current_parser.feed(current_dashboard)
    assert "성공 155" in current_dashboard and "실패 0" in current_dashboard
    assert all("checked" in item for item in current_parser.checkboxes[:6])
