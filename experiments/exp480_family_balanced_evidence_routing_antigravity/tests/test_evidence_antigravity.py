"""Task 2 tests for the Exp480 derived evidence database."""

from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
import shutil
import sqlite3

import pytest

from experiments.exp480_family_balanced_evidence_routing_antigravity import (
    evidence_antigravity as evidence_module_antigravity,
)
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


def test_official_evidence_validation_checks_the_complete_frozen_contract_antigravity() -> None:
    """Skipping census, provenance, family, integrity, or FK checks must fail preflight."""
    validation = evidence_module_antigravity.validate_evidence_database_antigravity()

    assert validation == {
        "datasets": 1117,
        "families": 128,
        "candidate_metrics": 4468,
        "train_features": 1117,
        "provenance_equal": True,
        "integrity": "ok",
        "foreign_keys_valid": True,
    }


def test_official_evidence_validation_rejects_provenance_drift_antigravity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A derived DB copied from different frozen sources must not reach evaluation."""
    target = tmp_path / "evidence_provenance_drift_antigravity.sqlite3"
    source = Path(__file__).resolve().parents[1] / "evidence_antigravity.sqlite3"
    shutil.copy2(source, target)
    with sqlite3.connect(target) as connection:
        connection.execute(
            "UPDATE provenance SET value=? WHERE key='exp479_result_sha256'",
            ("0" * 64,),
        )
        connection.commit()
    monkeypatch.setattr(
        evidence_module_antigravity,
        "EXPECTED_EVIDENCE_DATABASE_SHA256_ANTIGRAVITY",
        sha256(target.read_bytes()).hexdigest(),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="provenance"):
        evidence_module_antigravity.validate_evidence_database_antigravity(target)


def test_official_evidence_validation_rejects_family_mapping_drift_antigravity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A family label not derived from the exact first `_normal_` marker must fail."""
    target = tmp_path / "evidence_family_drift_antigravity.sqlite3"
    source = Path(__file__).resolve().parents[1] / "evidence_antigravity.sqlite3"
    shutil.copy2(source, target)
    with sqlite3.connect(target) as connection:
        connection.execute(
            "UPDATE dataset_evidence SET source_family='wrong_family' "
            "WHERE dataset=(SELECT min(dataset) FROM dataset_evidence)"
        )
        connection.commit()
    monkeypatch.setattr(
        evidence_module_antigravity,
        "EXPECTED_EVIDENCE_DATABASE_SHA256_ANTIGRAVITY",
        sha256(target.read_bytes()).hexdigest(),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="family mapping"):
        evidence_module_antigravity.validate_evidence_database_antigravity(target)


def test_official_evidence_validation_rejects_foreign_key_drift_antigravity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An orphaned feature row must fail even when all table row counts still match."""
    target = tmp_path / "evidence_foreign_key_drift_antigravity.sqlite3"
    source = Path(__file__).resolve().parents[1] / "evidence_antigravity.sqlite3"
    shutil.copy2(source, target)
    with sqlite3.connect(target) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "UPDATE train_features SET dataset='orphan_normal_1' "
            "WHERE dataset=(SELECT min(dataset) FROM train_features)"
        )
        connection.commit()
    monkeypatch.setattr(
        evidence_module_antigravity,
        "EXPECTED_EVIDENCE_DATABASE_SHA256_ANTIGRAVITY",
        sha256(target.read_bytes()).hexdigest(),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="foreign-key"):
        evidence_module_antigravity.validate_evidence_database_antigravity(target)


def _tampered_evidence_database_antigravity(
    tmp_path: Path, filename: str, statement: str
) -> Path:
    target = tmp_path / filename
    source = Path(__file__).resolve().parents[1] / "evidence_antigravity.sqlite3"
    shutil.copy2(source, target)
    with sqlite3.connect(target) as connection:
        connection.execute(statement)
        connection.commit()
    return target


def test_canonical_evidence_sha_rejects_coherent_proposal_tamper_antigravity(
    tmp_path: Path,
) -> None:
    """A valid alternative proposal must still fail when it is not canonical evidence."""
    target = _tampered_evidence_database_antigravity(
        tmp_path,
        "proposal_tamper_antigravity.sqlite3",
        "UPDATE dataset_evidence SET proposal=CASE "
        "WHEN proposal='minirocket_local_gap' THEN 'hydra_local_gap' "
        "ELSE 'minirocket_local_gap' END "
        "WHERE dataset=(SELECT min(dataset) FROM dataset_evidence)",
    )

    with pytest.raises(RuntimeError, match="canonical evidence SHA"):
        evidence_module_antigravity.validate_evidence_database_antigravity(target)


def test_canonical_evidence_sha_rejects_coherent_feature_tamper_antigravity(
    tmp_path: Path,
) -> None:
    """A finite seven-feature mutation must not pass structural validation."""
    target = _tampered_evidence_database_antigravity(
        tmp_path,
        "feature_tamper_antigravity.sqlite3",
        "UPDATE train_features SET width=width+0.125 "
        "WHERE dataset=(SELECT min(dataset) FROM train_features)",
    )

    with pytest.raises(RuntimeError, match="canonical evidence SHA"):
        evidence_module_antigravity.validate_evidence_database_antigravity(target)


def test_canonical_evidence_sha_rejects_coherent_auc_tamper_antigravity(
    tmp_path: Path,
) -> None:
    """An in-range AUC mutation must not pass the frozen metric contract."""
    target = _tampered_evidence_database_antigravity(
        tmp_path,
        "auc_tamper_antigravity.sqlite3",
        "UPDATE candidate_metrics SET auc=CASE WHEN auc<0.99 THEN auc+0.001 "
        "ELSE auc-0.001 END WHERE rowid=(SELECT min(rowid) FROM candidate_metrics)",
    )

    with pytest.raises(RuntimeError, match="canonical evidence SHA"):
        evidence_module_antigravity.validate_evidence_database_antigravity(target)


def test_canonical_evidence_sha_rejects_coherent_zf_tamper_antigravity(
    tmp_path: Path,
) -> None:
    """A nonnegative integer ZF mutation must not pass the frozen metric contract."""
    target = _tampered_evidence_database_antigravity(
        tmp_path,
        "zf_tamper_antigravity.sqlite3",
        "UPDATE candidate_metrics SET zf=zf+1 "
        "WHERE rowid=(SELECT min(rowid) FROM candidate_metrics)",
    )

    with pytest.raises(RuntimeError, match="canonical evidence SHA"):
        evidence_module_antigravity.validate_evidence_database_antigravity(target)


def test_source_provenance_hashes_database_and_invalidates_changed_cache_antigravity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording a literal or reusing a digest after same-size mutation must fail."""
    database = tmp_path / "original_database_antigravity.db"
    database.write_bytes(b"alpha")
    alpha_sha = "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8"
    omega_sha = "304b4a90a76a1cbe4c112e074b30e75181f54df43d60f883597457844293b341"
    monkeypatch.setattr(evidence_module_antigravity, "DATABASE_PATH_ANTIGRAVITY", database)
    monkeypatch.setattr(
        evidence_module_antigravity, "EXPECTED_DATABASE_SHA256_ANTIGRAVITY", alpha_sha
    )
    assert evidence_module_antigravity.source_provenance_antigravity()[
        "database_expected_sha256"
    ] == alpha_sha

    database.write_bytes(b"omega")
    monkeypatch.setattr(
        evidence_module_antigravity, "EXPECTED_DATABASE_SHA256_ANTIGRAVITY", omega_sha
    )
    assert evidence_module_antigravity.source_provenance_antigravity()[
        "database_expected_sha256"
    ] == omega_sha

    monkeypatch.setattr(
        evidence_module_antigravity, "EXPECTED_DATABASE_SHA256_ANTIGRAVITY", alpha_sha
    )
    with pytest.raises(RuntimeError, match="database source hash changed"):
        evidence_module_antigravity.source_provenance_antigravity()
