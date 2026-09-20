"""Task 1 tests for the exact Exp479 recovery population."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import FrozenInstanceError
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sqlite3

import pytest

from experiments.exp474_conservative_fallback_routing_antigravity.run_antigravity import (
    CHECKPOINT_DIR_ANTIGRAVITY,
)
from experiments.exp478_full_1117_official_routing_antigravity.run_antigravity import (
    DATABASE_PATH_ANTIGRAVITY,
    RESULT_PATH_ANTIGRAVITY as EXP478_RESULT_PATH_ANTIGRAVITY,
)


def test_population_module_exposes_the_task1_api_antigravity() -> None:
    """Deleting the Task 1 module or either public entry point must fail."""
    module_name = (
        "experiments.exp479_missing_score_recovery_antigravity."
        "population_antigravity"
    )
    specification = importlib.util.find_spec(module_name)
    assert specification is not None, "Exp479 population module is missing"

    module = importlib.import_module(module_name)
    assert hasattr(module, "select_recovery_population_antigravity")
    assert hasattr(module, "build_population_manifest_antigravity")


def test_real_population_is_the_exact_sorted_155_dataset_difference_antigravity() -> None:
    """Wrong set subtraction, ordering, or exclusion handling must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "population_antigravity"
    )
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
    )
    try:
        audit = module.select_recovery_population_antigravity(
            connection, CHECKPOINT_DIR_ANTIGRAVITY
        )
    finally:
        connection.close()

    assert audit.included_count == 1117
    assert audit.existing_score_count == 962
    assert audit.checkpoint_count == 23
    assert len(audit.datasets) == 155
    assert len(set(audit.datasets)) == 155
    assert audit.datasets == tuple(sorted(audit.datasets))
    assert audit.datasets[:2] == (
        "AllGestureWiimoteX_normal_1",
        "AllGestureWiimoteX_normal_10",
    )
    assert audit.datasets[-2:] == (
        "ShakeGestureWiimoteZ_normal_8",
        "ShakeGestureWiimoteZ_normal_9",
    )
    assert audit.excluded_datasets == (
        "CornellWhaleChallenge",
        "Phoneme_normal_39",
    )
    with pytest.raises(FrozenInstanceError):
        audit.datasets = ()


def test_population_manifest_pins_the_exact_sources_and_target_antigravity() -> None:
    """Dropping source hashes or changing the exact target list must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "population_antigravity"
    )
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
    )
    try:
        audit = module.select_recovery_population_antigravity(
            connection, CHECKPOINT_DIR_ANTIGRAVITY
        )
    finally:
        connection.close()

    manifest = module.build_population_manifest_antigravity(
        audit,
        checkpoint_dir=CHECKPOINT_DIR_ANTIGRAVITY,
        exp478_result_path=EXP478_RESULT_PATH_ANTIGRAVITY,
        database_path=DATABASE_PATH_ANTIGRAVITY,
    )

    assert manifest["schema"] == "exp479_population_manifest_v1"
    assert manifest["counts"] == {
        "included": 1117,
        "existing_scores": 962,
        "recovery": 155,
        "checkpoint_files": 23,
    }
    assert manifest["recovery_dataset_sha256"] == (
        "ce85b1626061c50b6ee92e57948442bc565b84c08c219dff7f2b45a40e453065"
    )
    assert manifest["source_hashes"]["exp478_result"] == (
        "7af31a8849c00af9a3021fea60f9cedc4e0b537b0a1ac483d4f5c5f7ec0d26cc"
    )
    checkpoint_hashes = manifest["source_hashes"]["score_checkpoints"]
    assert len(checkpoint_hashes) == 23
    assert list(checkpoint_hashes) == sorted(checkpoint_hashes)
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in checkpoint_hashes.values()
    )
    assert manifest["database"]["path"] == str(DATABASE_PATH_ANTIGRAVITY)
    assert manifest["database"]["size_bytes"] > 0
    assert manifest["database"]["expected_sha256"] == (
        "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
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


def test_task1_dashboard_checks_only_completed_work_antigravity() -> None:
    """Checking future tasks or hiding the measured Task 1 counts must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "population_antigravity"
    )
    assert hasattr(module, "render_task1_dashboard_antigravity")

    connection = sqlite3.connect(
        f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
    )
    try:
        audit = module.select_recovery_population_antigravity(
            connection, CHECKPOINT_DIR_ANTIGRAVITY
        )
    finally:
        connection.close()
    manifest = module.build_population_manifest_antigravity(
        audit,
        checkpoint_dir=CHECKPOINT_DIR_ANTIGRAVITY,
        exp478_result_path=EXP478_RESULT_PATH_ANTIGRAVITY,
        database_path=DATABASE_PATH_ANTIGRAVITY,
    )

    html = module.render_task1_dashboard_antigravity(audit, manifest)
    parser = _CheckboxParserAntigravity()
    parser.feed(html)

    assert "155" in html and "962" in html and "23" in html
    assert "아직 성능 수치를 계산하지 않았습니다" in html
    assert len(parser.checkboxes) == 10
    assert sum("checked" in item for item in parser.checkboxes) == 5
    assert all("disabled" in item for item in parser.checkboxes)


def test_task1_artifacts_match_the_verified_manifest_and_dashboard_antigravity() -> None:
    """A stale or manually divergent Task 1 artifact must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "population_antigravity"
    )
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
    )
    try:
        audit = module.select_recovery_population_antigravity(
            connection, CHECKPOINT_DIR_ANTIGRAVITY
        )
    finally:
        connection.close()
    manifest = module.build_population_manifest_antigravity(
        audit,
        checkpoint_dir=CHECKPOINT_DIR_ANTIGRAVITY,
        exp478_result_path=EXP478_RESULT_PATH_ANTIGRAVITY,
        database_path=DATABASE_PATH_ANTIGRAVITY,
    )
    root = Path(__file__).resolve().parents[3]
    manifest_path = (
        root
        / "experiments/exp479_missing_score_recovery_antigravity/"
        "population_manifest_antigravity.json"
    )
    dashboard_path = root / "docs/EXP479_DASHBOARD_antigravity.html"

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    dashboard = dashboard_path.read_text(encoding="utf-8")
    parser = _CheckboxParserAntigravity()
    parser.feed(dashboard)
    assert manifest["recovery_dataset_sha256"] in dashboard
    assert "155" in dashboard and "Task 1" in dashboard
    assert len(parser.checkboxes) >= 5
    assert all("checked" in item for item in parser.checkboxes[:5])
