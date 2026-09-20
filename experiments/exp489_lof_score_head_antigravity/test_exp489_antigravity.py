"""Exp489 paired-head isolation, sklearn parity, and population filters."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import (
    MethodCandidateAntigravity,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
    _reference_indices_antigravity,
    crossfit_plan_antigravity,
    score_candidate_antigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import METHODS_ANTIGRAVITY
from experiments.exp489_lof_score_head_antigravity.lof_score_head_antigravity import score_heads_antigravity
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import (
    EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY,
    representative_rows_antigravity,
)


class IdentityEncoderAntigravity:
    def __init__(self) -> None:
        self.fit_calls = 0
        self.transform_calls = 0

    def fit_transform(self, rows):
        self.fit_calls += 1
        return np.asarray(rows, dtype=float)

    def transform(self, rows):
        self.transform_calls += 1
        return np.asarray(rows, dtype=float)


def _scores(method, auc, zf):
    return tuple(CandidateMetricAntigravity(name, float(auc), int(zf)) for name in METHODS_ANTIGRAVITY)


def _row(dataset, n_train, family=None):
    family = family or dataset.split("_normal_", 1)[0]
    return EvidenceRowAntigravity(
        dataset=dataset,
        source_family=family,
        proposal=METHODS_ANTIGRAVITY[0],
        features=(64.0, float(n_train), 0.4, 0.1, 2.0, 0.8, 0.3),
        scores=_scores(METHODS_ANTIGRAVITY[0], 0.8, 2),
    )


def test_representatives_drop_families_with_fewer_than_five_train_rows():
    rows = (
        _row("PigAirwayPressure_normal_0", 4),
        _row("PigArtPressure_normal_0", 4),
        _row("PigCVP_normal_1", 4),
        _row("PigCVP_normal_0", 4),
        _row("Wafer_normal_9", 8),
        _row("Wafer_normal_0", 12),
        _row("Yoga_normal_0", 20),
    )
    chosen, excluded = representative_rows_antigravity(rows, expected_families=2)
    assert tuple(row.dataset for row in chosen) == ("Wafer_normal_0", "Yoga_normal_0")
    assert excluded == EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY


def test_score_heads_share_one_encoder_and_match_sklearn_lof(monkeypatch):
    encoders = []

    def factory(name, seed):
        assert name == "minirocket_local_gap"
        assert seed == 20260721
        encoder = IdentityEncoderAntigravity()
        encoders.append(encoder)
        return encoder

    rng = np.random.default_rng(20260721)
    train = rng.normal(size=(10, 6)).astype(np.float32)
    test = rng.normal(size=(4, 6)).astype(np.float32)
    monkeypatch.setattr(
        "experiments.exp489_lof_score_head_antigravity.lof_score_head_antigravity.make_representation_antigravity",
        factory,
    )
    first = score_heads_antigravity(train, test, seed=20260721)
    second = score_heads_antigravity(train, test, seed=20260721)
    assert first["representation_fits"] == 6
    assert len(encoders) == 12
    assert all(encoder.fit_calls == 1 for encoder in encoders)
    np.testing.assert_array_equal(first["lof"]["test_scores"], second["lof"]["test_scores"])
    np.testing.assert_array_equal(first["control"]["train_oof"], second["control"]["train_oof"])

    folds = crossfit_plan_antigravity(len(train), seed=20260721)
    expected_oof = np.empty(len(train), dtype=float)
    for fold_index, (fit_indices, query_indices) in enumerate(folds):
        reference = _reference_indices_antigravity(
            fit_indices, seed=20260721 + 1009 * (fold_index + 1) + 4096
        )
        ref_features = np.asarray(train[reference], dtype=float)
        query_features = np.asarray(train[query_indices], dtype=float)
        scaler = RobustScaler(quantile_range=(10, 90)).fit(ref_features)
        nref = len(reference)
        lof = LocalOutlierFactor(
            n_neighbors=min(3, nref - 1),
            novelty=True,
            metric="minkowski",
            p=2,
            contamination="auto",
            n_jobs=1,
        )
        lof.fit(scaler.transform(ref_features))
        expected_oof[query_indices] = -lof.score_samples(scaler.transform(query_features))
    np.testing.assert_allclose(first["lof"]["train_oof"], expected_oof, rtol=0, atol=1e-12)

    candidate = MethodCandidateAntigravity(
        "minirocket_local_gap", "minirocket_local_gap", (20260721,), (0.005,)
    )
    with patch(
        "experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity.make_representation_antigravity",
        side_effect=lambda *_: IdentityEncoderAntigravity(),
    ):
        control = score_candidate_antigravity(train, test, candidate, seed=20260721)
    np.testing.assert_allclose(first["control"]["test_scores"], control.test_scores, rtol=0, atol=1e-12)
    np.testing.assert_allclose(first["control"]["train_oof"], control.train_oof, rtol=0, atol=1e-12)


def test_score_heads_reject_bad_and_constant_nonfinite_inputs():
    train = np.arange(30, dtype=float).reshape(6, 5)
    test = np.arange(10, dtype=float).reshape(2, 5)
    with pytest.raises(ValueError, match="seed"):
        score_heads_antigravity(train, test, seed=True)
    with pytest.raises(ValueError, match="shape"):
        score_heads_antigravity(train, test[:, :-1], seed=20260721)
    bad = train.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_heads_antigravity(bad, test, seed=20260721)
    constant = np.ones((6, 8), dtype=float)
    constant[:, 0] = np.arange(6)
    with patch(
        "experiments.exp489_lof_score_head_antigravity.lof_score_head_antigravity.make_representation_antigravity",
        side_effect=lambda *_: IdentityEncoderAntigravity(),
    ):
        scores = score_heads_antigravity(constant, constant[:2], seed=20260721)
    assert np.isfinite(scores["lof"]["test_scores"]).all()
    assert np.isfinite(scores["control"]["test_scores"]).all()


def test_nref_below_two_fails(monkeypatch):
    monkeypatch.setattr(
        "experiments.exp489_lof_score_head_antigravity.lof_score_head_antigravity._reference_indices_antigravity",
        lambda indices, *, seed: np.asarray(indices[:1], dtype=int),
    )
    monkeypatch.setattr(
        "experiments.exp489_lof_score_head_antigravity.lof_score_head_antigravity.make_representation_antigravity",
        lambda *_: IdentityEncoderAntigravity(),
    )
    train = np.arange(40, dtype=float).reshape(8, 5)
    with pytest.raises(ValueError, match="nref"):
        score_heads_antigravity(train, train[:2], seed=20260721)


def test_official_result_cannot_be_overwritten(tmp_path):
    from experiments.exp489_lof_score_head_antigravity.run_antigravity import run

    path = tmp_path / "results_antigravity.json"
    path.write_text("frozen")
    with pytest.raises(RuntimeError, match="already exists"):
        run("official", output=tmp_path)
    assert path.read_text() == "frozen"


def test_score_checkpoint_is_written_before_labels(tmp_path, monkeypatch):
    from experiments.exp489_lof_score_head_antigravity import run_antigravity as runner

    order = []
    heads = {
        "feature_dimension": 2,
        "representation_fits": 6,
        "control": {"train_oof": np.arange(6, dtype=float), "test_scores": np.arange(3, dtype=float)},
        "lof": {"train_oof": np.arange(6, dtype=float) + 1, "test_scores": np.arange(3, dtype=float) + 1},
    }

    class Prepared:
        train_rows = np.arange(24, dtype=float).reshape(6, 4)
        test_rows = np.arange(12, dtype=float).reshape(3, 4)
        test_indices = (0, 1, 2)
        target_length = 4

    monkeypatch.setattr(runner, "prepare_dataset_values_antigravity", lambda *_: Prepared)
    monkeypatch.setattr(runner, "score_heads_antigravity", lambda *args, **kwargs: order.append("score") or heads)

    original_persist = runner.persist_score_checkpoint_antigravity

    def persist(*args, **kwargs):
        order.append("checkpoint")
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(runner, "persist_score_checkpoint_antigravity", persist)
    monkeypatch.setattr(
        runner,
        "load_retained_test_labels_antigravity",
        lambda *args, **kwargs: order.append("labels") or np.asarray([0, 1, 0], dtype=np.int64),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_score_vector_antigravity",
        lambda *args, **kwargs: {"auc": 0.75, "zf": 1},
    )
    monkeypatch.setattr(runner, "readonly_sqlite_antigravity", lambda database: __import__("sqlite3").connect(":memory:"))
    payload = {
        "database": str(tmp_path / "unused_antigravity.db"),
        "dataset": "Wafer_normal_0",
        "source_family": "Wafer",
        "score_dir": str(tmp_path / "scores"),
        "binding": {"k": "v"},
    }
    runner.score_dataset_worker_antigravity(payload)
    assert order == ["score", "checkpoint", "labels"]
    assert (tmp_path / "scores" / "Wafer_normal_0_antigravity_scores_antigravity.npz").exists()


def test_resume_records_reject_duplicates_and_changed_score_artifact(tmp_path):
    from types import SimpleNamespace
    from experiments.exp489_lof_score_head_antigravity import run_antigravity as runner
    row = SimpleNamespace(dataset="F_normal_0", source_family="F")
    heads = {"feature_dimension":2,"representation_fits":6,
             "control":{"train_oof":np.arange(5.),"test_scores":np.arange(2.)},
             "lof":{"train_oof":np.arange(5.),"test_scores":np.arange(2.)}}
    binding={"code":"fixed"}
    meta=runner.persist_score_checkpoint_antigravity(tmp_path,row.dataset,heads,
        {"binding":binding,"source_family":"F","test_indices":[0,1]})
    record={"dataset":row.dataset,"source_family":"F","score_sha256":meta["score_sha256"]}
    runner.validate_score_records_antigravity([record],[row],tmp_path,binding,complete=True)
    with pytest.raises(ValueError,match="duplicate"):
        runner.validate_score_records_antigravity([record,record],[row],tmp_path,binding)
    npz,_=runner._score_paths_antigravity(tmp_path,row.dataset)
    npz.write_bytes(npz.read_bytes()+b"tampered")
    with pytest.raises(ValueError,match="stale or mixed"):
        runner.validate_score_records_antigravity([record],[row],tmp_path,binding)
