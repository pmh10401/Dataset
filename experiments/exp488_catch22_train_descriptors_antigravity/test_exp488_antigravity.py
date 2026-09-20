"""Exp488 boundaries: preserve the control and isolate learned preprocessing."""
from dataclasses import replace

import numpy as np
import pytest

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import CandidateMetricAntigravity, EvidenceRowAntigravity
from experiments.exp481_strict_evidence_residual_correction_antigravity.config_antigravity import FEATURE_KEYS_ANTIGRAVITY, METHODS_ANTIGRAVITY
from experiments.exp481_strict_evidence_residual_correction_antigravity.evaluation_antigravity import evaluate_nested_out_of_family_antigravity
from experiments.exp481_strict_evidence_residual_correction_antigravity.policy_antigravity import StrictResidualPolicyAntigravity


def evidence(count=6):
    return tuple(EvidenceRowAntigravity(
        dataset=f"F{i:03}_normal_0", source_family=f"F{i:03}", proposal=METHODS_ANTIGRAVITY[0],
        features=(64.0, 20.0 + i, .4 + i / 100, .1, 2.0, .8, .3),
        scores=tuple(CandidateMetricAntigravity(method, float(.8 + (.05 if j == 1 else -.02 * j)), 3 + (1 if j == 1 else -min(j, 2))) for j, method in enumerate(METHODS_ANTIGRAVITY)),
    ) for i in range(count))


def test_opt_in_seven_features_preserves_every_nested_decision():
    rows = evidence()
    kwargs = dict(residual_policy_kwargs={"k": 3}, inner_min_changed_families=2)
    baseline = evaluate_nested_out_of_family_antigravity(rows, **kwargs)
    actual = evaluate_nested_out_of_family_antigravity(rows, residual_policy_kwargs={
        "k": 3, "feature_keys": FEATURE_KEYS_ANTIGRAVITY,
        "descriptor_values": {row.dataset: row.features for row in rows},
    }, inner_min_changed_families=2)
    assert actual == baseline


def test_descriptor_fit_uses_donors_only_and_keeps_query_numeric():
    rows = evidence(5)
    keys = ("width", "n_train", "catch22_test")
    values = {r.dataset: (64.0, 20.0, float(i)) for i, r in enumerate(rows)}
    values[rows[0].dataset] = (64.0, 20.0, float("nan"))
    first = StrictResidualPolicyAntigravity(k=3, feature_keys=keys, descriptor_values=values).fit(rows[:4])
    values[rows[4].dataset] = (1e9, 1e9, 1e9)
    second = StrictResidualPolicyAntigravity(k=3, feature_keys=keys, descriptor_values=values).fit(rows[:4])
    np.testing.assert_array_equal(first._matrix, second._matrix)
    assert first._medians[2] == 2.0
    query = dict(zip(keys, (64.0, 20.0, float("nan"))))
    a = first.route(query, row_knn_method=METHODS_ANTIGRAVITY[0])
    b = second.route(query, row_knn_method=METHODS_ANTIGRAVITY[0])
    assert a == b
    with pytest.raises(ValueError):
        first.route({**query, "dataset": 1.0}, row_knn_method=METHODS_ANTIGRAVITY[0])


def test_thirty_four_families_exercise_unmodified_inner_gate():
    rows = evidence(34)
    rows += (replace(rows[0], dataset="F000_normal_1"),)
    keys = ("width", "n_train", "catch22_test")
    values = {r.dataset: (r.features[0], r.features[1], float(i)) for i, r in enumerate(rows)}
    kwargs = {"feature_keys": keys, "descriptor_values": values}
    result = evaluate_nested_out_of_family_antigravity(rows, residual_policy_kwargs=kwargs)
    assert result["family_count"] == 34
    assert result["summaries"]["changed"]["count"] > 0
    assert any(g["enabled"] and g["changed_families"] >= 20 for gates in result["inner_gates_by_outer_family"].values() for g in gates)
    assert result["leakage"]["family_overlap_count"] == 0
    assert result["leakage"]["route_input_key_count"] == 3


def test_train_loader_excludes_test_and_nonnormal_rows(tmp_path):
    import sqlite3
    from .features_antigravity import load_training_rows
    path = tmp_path / "input_antigravity.sqlite3"
    with sqlite3.connect(path) as c:
        c.executescript("CREATE TABLE datasets(id INTEGER,name TEXT); CREATE TABLE instances(dataset_id INTEGER,split TEXT,label TEXT,instance_index INTEGER,values_blob BLOB); INSERT INTO datasets VALUES(1,'F000_normal_0');")
        for split, label, index, offset in [("TRAIN", "0", 0, 0), ("TRAIN", "0", 1, 1), ("TEST", "0", 0, 99), ("TRAIN", "1", 2, 99)]:
            c.execute("INSERT INTO instances VALUES(1,?,?,?,?)", (split, label, index, (np.arange(15, dtype='<f4') + offset).tobytes()))
    rows, identity = load_training_rows(path, "F000_normal_0")
    assert len(rows) == 2
    np.testing.assert_array_equal(rows[1], np.arange(15) + 1)
    with sqlite3.connect(path) as c:
        c.execute("UPDATE instances SET values_blob=? WHERE split='TEST' OR label='1'", (np.ones(200, dtype='<f4').tobytes(),))
    repeated, other_identity = load_training_rows(path, "F000_normal_0")
    assert identity == other_identity
    for a, b in zip(rows, repeated, strict=True):
        np.testing.assert_array_equal(a, b)


def test_descriptors_keep_ragged_rows_and_undefined_values():
    from aeon.transformations.collection.feature_based import Catch22
    from .features_antigravity import describe_rows, FEATURE_KEYS
    rows = [np.sin(np.arange(15)), np.cos(np.arange(64) * .13), np.ones(257), np.array([0.0, np.nan])]
    record = describe_rows(rows, (64.0, 3.0))
    normalized = [((r-r.mean())/r.std())[None, :] for r in rows[:2]]
    raw = Catch22(catch24=False, outlier_norm=False, replace_nans=False, use_pycatch22=False, n_jobs=1).fit_transform(normalized)
    expected = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else np.nan for col in raw.T])
    values = np.array([np.nan if x is None else x for x in record["values"]])
    assert len(FEATURE_KEYS) == 24
    np.testing.assert_allclose(values[2:], expected, rtol=0, atol=0, equal_nan=True)
    assert record["finite_rows"] == 3 and record["eligible_rows"] == 2
    assert record == describe_rows(rows, (64.0, 3.0))
    constant = describe_rows([np.ones(15), np.ones(15)], (15.0, 2.0))
    assert constant["values"][2:] == [None] * 22
    with pytest.raises(ValueError, match="frozen"):
        describe_rows(rows, (64.0, 99.0))


def test_all_missing_and_constant_donor_columns_stay_finite():
    rows = evidence(5)
    keys = ("width", "n_train", "catch22_missing", "catch22_constant")
    values = {r.dataset: (64.0, 20.0, np.nan, 3.0) for r in rows}
    policy = StrictResidualPolicyAntigravity(k=3, feature_keys=keys, descriptor_values=values).fit(rows[:4])
    assert np.isfinite(policy._matrix).all()
    assert tuple(policy._scale[2:]) == (1.0, 1.0)
    assert policy.route(dict(zip(keys, values[rows[4].dataset])), row_knn_method=METHODS_ANTIGRAVITY[0]).selected_candidate == METHODS_ANTIGRAVITY[1]


def test_query_outcomes_do_not_change_routing():
    from experiments.exp481_strict_evidence_residual_correction_antigravity.evaluation_antigravity import evaluate_outer_row_antigravity, _row_policy_antigravity
    rows = evidence(6)
    keys = ("width", "n_train", "catch22_test")
    values = {r.dataset: (64.0, 20.0, float(i)) for i, r in enumerate(rows)}
    residual = StrictResidualPolicyAntigravity(k=3, feature_keys=keys, descriptor_values=values).fit(rows[:-1])
    control = _row_policy_antigravity(rows[:-1], None)
    first = evaluate_outer_row_antigravity(rows[-1], control, residual, METHODS_ANTIGRAVITY[1:])
    changed = replace(rows[-1], scores=tuple(CandidateMetricAntigravity(m, .1, 0) for m in METHODS_ANTIGRAVITY))
    second = evaluate_outer_row_antigravity(changed, control, residual, METHODS_ANTIGRAVITY[1:])
    assert first.method == second.method and first.reason == second.reason


def test_cache_rejects_tampering_stale_binding_and_misaligned_values(tmp_path):
    import json
    from .run_antigravity import read_cache, save_cache
    row = evidence(1)[0]
    path = tmp_path / "cache_antigravity.json"
    binding = {"source": "frozen"}
    record = dict(values=[*row.features[:2], *([None] * 22)], total_rows=20, finite_rows=20,
                  eligible_rows=0, feature_valid_counts=[0] * 22, train_membership_sha256="a" * 64)
    records = {row.dataset: record}
    save_cache(path, binding, records)
    assert read_cache(path, binding, [row]) == records
    with pytest.raises(ValueError, match="binding"):
        read_cache(path, {"source": "different"}, [row])
    payload = json.loads(path.read_text())
    payload["records"][row.dataset]["values"][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        read_cache(path, binding, [row])
    save_cache(path, binding, {row.dataset: {**record, "values": [65.0, *record["values"][1:]]}})
    with pytest.raises(ValueError, match="values"):
        read_cache(path, binding, [row])
    save_cache(path, binding, {row.dataset: {**record, "feature_valid_counts": [1] * 22}})
    with pytest.raises(ValueError, match="provenance"):
        read_cache(path, binding, [row])


def test_official_result_cannot_be_overwritten(tmp_path):
    from .run_antigravity import run
    path = tmp_path / "results_antigravity.json"
    path.write_text("frozen")
    with pytest.raises(RuntimeError, match="already exists"):
        run("official", output=tmp_path)
    assert path.read_text() == "frozen"


def test_implementation_change_invalidates_prior_smoke_binding(tmp_path, monkeypatch):
    from . import run_antigravity as runner
    database = tmp_path / "raw_antigravity.db"
    changed = set()
    def digest(path):
        if path == database:
            return runner.RAW_SHA
        if path.parent.name == "exp484_distance_qualification_antigravity":
            return runner.EXP484_SHA
        return "new" if path in changed else "old"
    monkeypatch.setattr(runner, "file_sha", digest)
    baseline = runner.identity(database, {})
    cache = tmp_path / "cache_antigravity.json"
    runner.save_cache(cache, baseline, {})
    for path in runner.CODE_PATHS:
        changed.clear()
        changed.add(path)
        updated = runner.identity(database, {})
        assert updated != baseline
        with pytest.raises(ValueError, match="binding"):
            runner.read_cache(cache, updated, [])
