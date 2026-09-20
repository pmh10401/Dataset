"""Exp490 TRAIN-only extraction, mocked encoders, and constant/tie behavior."""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
    _REQUESTED_KERNELS_ANTIGRAVITY,
    _reference_indices_antigravity,
    crossfit_plan_antigravity,
)
from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import (
    METHODS_ANTIGRAVITY,
    SEED_ANTIGRAVITY,
)
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import (
    prepare_dataset_values_antigravity,
)
from experiments.exp490_behavior_descriptors_antigravity.behavior_descriptors_antigravity import (
    BEHAVIOR_KEYS_ANTIGRAVITY,
    assemble_behavior_antigravity,
    b_reference_indices_antigravity,
    extract_behavior_antigravity,
    prepare_normal_train_matrix_antigravity,
    spearman_or_zero_antigravity,
)


class RecordingEncoderAntigravity:
    def __init__(self, name, seed, journal):
        self.name = name
        self.seed = seed
        self.journal = journal

    def fit_transform(self, rows):
        self.journal.append((self.name, self.seed, np.asarray(rows)[:, 0].astype(int).tolist()))
        features = np.column_stack([np.asarray(rows, dtype=float)[:, 0], np.arange(len(rows), dtype=float)])
        return features

    def transform(self, rows):
        return np.column_stack([np.asarray(rows, dtype=float)[:, 0], np.arange(len(rows), dtype=float)])


def test_behavior_keys_are_eighteen_and_canonical():
    assert len(BEHAVIOR_KEYS_ANTIGRAVITY) == 18
    assert BEHAVIOR_KEYS_ANTIGRAVITY[:3] == (
        "minirocket_local_gap_tail_q95",
        "minirocket_local_gap_tail_q99",
        "minirocket_local_gap_ab_spearman",
    )
    assert BEHAVIOR_KEYS_ANTIGRAVITY[-1].startswith("spearman_hydra_local_gap_")


def test_constant_correlation_is_zero_and_counted():
    constant = np.ones(8)
    varying = np.arange(8, dtype=float)
    value, was_constant = spearman_or_zero_antigravity(constant, varying)
    assert value == 0.0 and was_constant
    ties_left = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
    ties_right = np.array([1.0, 2.0, 2.0, 4.0, 5.0])
    tied, tied_constant = spearman_or_zero_antigravity(ties_left, ties_right)
    assert tied_constant is False and np.isfinite(tied)
    oof_a = {method: (constant if method == METHODS_ANTIGRAVITY[0] else varying) for method in METHODS_ANTIGRAVITY}
    oof_b = {method: varying for method in METHODS_ANTIGRAVITY}
    values, diagnostics = assemble_behavior_antigravity(oof_a, oof_b)
    assert len(values) == 18
    assert all(np.isfinite(values))
    assert values[2] == 0.0
    assert diagnostics["constant_correlation_count"] >= 4


def test_extract_behavior_is_deterministic_and_keeps_b_inside_a(monkeypatch):
    journal = []

    def factory(name, seed):
        return RecordingEncoderAntigravity(name, seed, journal)

    rng = np.random.default_rng(SEED_ANTIGRAVITY)
    train = rng.normal(size=(12, 9))
    train[:, 0] = np.arange(12)
    monkeypatch.setattr(
        "experiments.exp490_behavior_descriptors_antigravity.behavior_descriptors_antigravity.make_representation_antigravity",
        factory,
    )
    first = extract_behavior_antigravity(train, seed=SEED_ANTIGRAVITY)
    second = extract_behavior_antigravity(train, seed=SEED_ANTIGRAVITY)
    assert first["values"] == second["values"]
    assert first["diagnostics"]["representation_fits"] == 40
    assert all(entry[1] == SEED_ANTIGRAVITY for entry in journal)
    folds = crossfit_plan_antigravity(len(train), seed=SEED_ANTIGRAVITY)
    for method in METHODS_ANTIGRAVITY:
        requested = _REQUESTED_KERNELS_ANTIGRAVITY[method]
        for fold_index, (fit_indices, query_indices) in enumerate(folds):
            reference_a = _reference_indices_antigravity(
                fit_indices, seed=SEED_ANTIGRAVITY + 1009 * (fold_index + 1) + requested
            )
            reference_b = b_reference_indices_antigravity(
                reference_a, seed=SEED_ANTIGRAVITY, fold_index=fold_index
            )
            assert np.isin(reference_b, reference_a).all()
            assert np.intersect1d(reference_b, query_indices).size == 0
            assert np.intersect1d(reference_a, query_indices).size == 0


def test_prepare_normal_train_never_reads_test_or_calls_exp479_prepare(tmp_path):
    path = tmp_path / "input_antigravity.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE datasets(id INTEGER, name TEXT);"
            "CREATE TABLE instances(dataset_id INTEGER, split TEXT, label TEXT, instance_index INTEGER, values_blob BLOB);"
            "INSERT INTO datasets VALUES(1, 'Wafer_normal_0');"
        )
        for index in range(6):
            blob = (np.arange(16, dtype="<f4") + index).tobytes()
            connection.execute(
                "INSERT INTO instances VALUES(1, 'TRAIN', '0', ?, ?)",
                (index, blob),
            )
        connection.execute(
            "INSERT INTO instances VALUES(1, 'TEST', '1', 0, ?)",
            (np.full(16, 99, dtype="<f4").tobytes(),),
        )
        connection.execute(
            "INSERT INTO instances VALUES(1, 'TRAIN', '1', 9, ?)",
            (np.full(16, 77, dtype="<f4").tobytes(),),
        )
    with patch(
        "experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity.prepare_dataset_values_antigravity",
        side_effect=AssertionError("TEST loader must not run"),
    ):
        train, provenance = prepare_normal_train_matrix_antigravity(path, "Wafer_normal_0")
    assert train.shape[0] == 6
    assert provenance["finite_rows"] == 6
    assert provenance["total_rows"] == 6
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE instances SET values_blob=? WHERE split='TEST' OR label='1'",
            (np.arange(400, dtype="<f4").tobytes(),),
        )
    repeated, repeated_provenance = prepare_normal_train_matrix_antigravity(path, "Wafer_normal_0")
    np.testing.assert_array_equal(train, repeated)
    assert provenance["train_membership_sha256"] == repeated_provenance["train_membership_sha256"]
    with sqlite3.connect(path) as connection:
        prepared = prepare_dataset_values_antigravity(connection, "Wafer_normal_0")
    assert prepared.raw_test_rows == 1


def test_official_result_cannot_be_overwritten(tmp_path):
    from experiments.exp490_behavior_descriptors_antigravity.run_antigravity import run

    path = tmp_path / "results_antigravity.json"
    path.write_text("frozen")
    with pytest.raises(RuntimeError, match="already exists"):
        run("official", output=tmp_path)
    assert path.read_text() == "frozen"


def test_behavior_cache_rejects_wrong_population_and_report_renders():
    from types import SimpleNamespace
    from experiments.exp490_behavior_descriptors_antigravity import run_antigravity as runner
    row = SimpleNamespace(dataset="F_normal_0", features=(16., 6.))
    record = dict(values=[0.]*18, keys=list(BEHAVIOR_KEYS_ANTIGRAVITY), finite_rows=6,
                  total_rows=6, diagnostics={"n_train":6}, train_membership_sha256="a"*64)
    runner.validate_behavior_cache_antigravity({row.dataset:record}, [row], complete=True)
    with pytest.raises(ValueError, match="unexpected dataset"):
        runner.validate_behavior_cache_antigravity({"Other_normal_0":record}, [row])
    with pytest.raises(ValueError, match="invalid behavior"):
        runner.validate_behavior_cache_antigravity({row.dataset:{**record,"values":[float("inf")]*18}}, [row])
    assert "Exp490" in runner.render_report_antigravity({"n_rows":1,"comparison":{}}, {})
