"""Task 2 tests for the Exp480 derived evidence database."""

from __future__ import annotations

import math
from pathlib import Path
import sqlite3

import pytest

from experiments.exp480_family_balanced_evidence_routing_antigravity.config_antigravity import (
    METHODS_ANTIGRAVITY,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.evidence_antigravity import (
    build_evidence_database_antigravity,
    family_from_dataset_antigravity,
    load_evidence_database_antigravity,
    load_evidence_rows_antigravity,
    source_provenance_antigravity,
)


@pytest.fixture(scope="module")
def real_rows_antigravity():
    return load_evidence_rows_antigravity()


def test_family_is_only_prefix_before_first_normal_antigravity() -> None:
    """Adding name aliases or using the suffix as a feature must fail."""
    assert family_from_dataset_antigravity("ShapesAll_normal_60") == "ShapesAll"
    assert (
        family_from_dataset_antigravity("FreezerRegularTrain_normal_1")
        == "FreezerRegularTrain"
    )
    with pytest.raises(ValueError, match="_normal_"):
        family_from_dataset_antigravity("ShapesAll")


def test_candidate_metrics_reject_nonfinite_or_unknown_values_antigravity() -> None:
    """Malformed frozen performance must never enter routing evidence."""
    with pytest.raises(ValueError, match="method"):
        CandidateMetricAntigravity("unknown", 0.8, 1)
    with pytest.raises(ValueError, match="finite"):
        CandidateMetricAntigravity(METHODS_ANTIGRAVITY[0], math.nan, 1)
    with pytest.raises(ValueError, match="nonnegative"):
        CandidateMetricAntigravity(METHODS_ANTIGRAVITY[0], 0.8, -1)


def test_real_sources_cover_exactly_1117_and_128_families_antigravity(
    real_rows_antigravity,
) -> None:
    """Dropping or duplicating a frozen score row must fail the census."""
    rows = real_rows_antigravity
    assert len(rows) == 1117
    assert len({row.dataset for row in rows}) == 1117
    assert len({row.source_family for row in rows}) == 128
    assert sum(row.dataset.startswith("ShapesAll_normal_") for row in rows) == 60
    assert all(tuple(metric.method for metric in row.scores) == METHODS_ANTIGRAVITY for row in rows)
    assert all(len(row.features) == 7 for row in rows)


def test_sqlite_round_trip_preserves_all_evidence_antigravity(
    tmp_path: Path, real_rows_antigravity
) -> None:
    """Changing insert order, schema counts or values must fail round-trip equality."""
    target = tmp_path / "evidence_antigravity.sqlite3"
    digest = build_evidence_database_antigravity(
        real_rows_antigravity,
        target,
        source_provenance_antigravity(),
    )
    restored = load_evidence_database_antigravity(target)

    assert len(digest) == 64
    assert restored == real_rows_antigravity
    with sqlite3.connect(f"file:{target}?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT count(*) FROM dataset_evidence").fetchone()[0] == 1117
        assert connection.execute("SELECT count(*) FROM candidate_metrics").fetchone()[0] == 4468

