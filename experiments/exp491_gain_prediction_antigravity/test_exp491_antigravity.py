"""Exp491 donor isolation, missing/constant finite paths, and family weights."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import METHODS_ANTIGRAVITY
from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp491_gain_prediction_antigravity.gain_prediction_antigravity import (
    TARGET_KEYS_ANTIGRAVITY,
    evaluate_meta_antigravity,
    target_vector_antigravity,
)



def _scores(aucs, zfs):
    return tuple(
        CandidateMetricAntigravity(method, float(auc), int(zf))
        for method, auc, zf in zip(METHODS_ANTIGRAVITY, aucs, zfs, strict=True)
    )


def test_smoke_to_official_binds_inputs_and_publishes_complete_outputs(tmp_path, monkeypatch):
    import copy
    import hashlib
    import json
    import shutil
    from experiments.exp491_gain_prediction_antigravity import run_antigravity as runner
    monkeypatch.setattr(runner, "assess_acceleration_antigravity", lambda: {"selected": "test_cpu"})
    database = tmp_path / "fixture_antigravity.db"
    database.touch()
    rows = runner.synthetic_evidence_rows_antigravity()
    values = runner.synthetic_descriptors_antigravity(rows)
    output = tmp_path / "run_antigravity"
    smoke = runner.run("smoke", database, output, rows=rows, descriptor_values=values)
    result = runner.run("official", database, output, rows=rows, descriptor_values=values)
    assert result["binding"] == smoke["binding"]
    manifest = json.loads((output / "run_manifest_antigravity.json").read_text())
    assert manifest["result_sha256"] == hashlib.sha256((output / "results_antigravity.json").read_bytes()).hexdigest()
    assert manifest["evaluation_workers"] == 1
    assert "Exp491" in (output / "report_antigravity.html").read_text()
    changed = copy.deepcopy(values)
    changed[rows[0].dataset][0] += 1
    stale_output = tmp_path / "stale_antigravity"
    stale_output.mkdir()
    shutil.copy2(output / "smoke_results_antigravity.json", stale_output)
    with pytest.raises(RuntimeError, match="matching smoke"):
        runner.run("official", database, stale_output, rows=rows, descriptor_values=changed)
    with pytest.raises(RuntimeError, match="already exists"):
        runner.run("official", database, output, rows=rows, descriptor_values=values)


def _row(family, member, aucs, zfs, features=None):
    return EvidenceRowAntigravity(
        dataset=f"{family}_normal_{member}",
        source_family=family,
        proposal=METHODS_ANTIGRAVITY[0],
        features=features or (64.0, 20.0 + member, 0.4, 0.1, 2.0, 0.8, 0.3),
        scores=_scores(aucs, zfs),
    )


def _population():
    rows = []
    values = {}
    for family_index in range(5):
        family = f"F{family_index:03}"
        for member in range(2):
            aucs = [0.6, 0.62 + 0.03 * family_index, 0.5, 0.7 - 0.02 * member]
            zfs = [2, 4 + family_index, 2, 1 + member]
            row = _row(family, member, aucs, zfs)
            rows.append(row)
            values[row.dataset] = [float(family_index), float(member), float(family_index - member)]
    return tuple(rows), values, ("f0", "f1", "f2")


def test_held_family_target_perturbation_does_not_change_its_predictions():
    rows, values, keys = _population()
    base = evaluate_meta_antigravity(rows, values, keys)
    held = rows[0].source_family
    perturbed = []
    for row in rows:
        if row.source_family != held:
            perturbed.append(row)
            continue
        scores = tuple(
            replace(metric, auc=0.11, zf=0) for metric in row.scores
        )
        perturbed.append(replace(row, scores=scores))
    changed = evaluate_meta_antigravity(tuple(perturbed), values, keys)
    base_fold = next(fold for fold in base["folds"] if fold["held_family"] == held)
    changed_fold = next(fold for fold in changed["folds"] if fold["held_family"] == held)
    assert base_fold["predictions"] == changed_fold["predictions"]
    other_base = next(fold for fold in base["folds"] if fold["held_family"] != held)
    other_changed = next(fold for fold in changed["folds"] if fold["held_family"] == other_base["held_family"])
    assert other_base["predictions"] != other_changed["predictions"] or other_base["targets"] != other_changed["targets"]


def test_held_out_feature_outlier_does_not_affect_donor_preprocessing():
    rows, values, keys = _population()
    base = evaluate_meta_antigravity(rows, values, keys)
    held = rows[0].source_family
    outlier = {dataset: list(vector) for dataset, vector in values.items()}
    for row in rows:
        if row.source_family == held:
            outlier[row.dataset] = [1e9, 1e9, 1e9]
    changed = evaluate_meta_antigravity(rows, outlier, keys)
    base_fold = next(fold for fold in base["folds"] if fold["held_family"] == held)
    changed_fold = next(fold for fold in changed["folds"] if fold["held_family"] == held)
    assert base_fold["preprocessing"]["medians"] == changed_fold["preprocessing"]["medians"]
    assert base_fold["preprocessing"]["mean"] == changed_fold["preprocessing"]["mean"]
    assert base_fold["preprocessing"]["std"] == changed_fold["preprocessing"]["std"]
    assert held not in base_fold["donor_families"]


def test_all_missing_and_constant_columns_stay_finite():
    rows, _values, _keys = _population()
    keys = ("missing", "constant")
    values = {row.dataset: [None, 3.0] for row in rows}
    result = evaluate_meta_antigravity(rows, values, keys)
    for fold in result["folds"]:
        for key in TARGET_KEYS_ANTIGRAVITY:
            assert all(np.isfinite(value) for value in fold["predictions"][key])
        assert fold["preprocessing"]["std"][0] == 1.0
        assert fold["ridge_alpha"] == 10.0
        assert abs(fold["family_weight_sum"] - fold["n_donor_families"]) < 1e-12
        assert all(abs(total - 1.0) < 1e-12 for total in fold["family_weight_totals"].values())


def test_all_same_class_positive_gain_auroc_is_null():
    rows = []
    values = {}
    for family_index in range(4):
        family = f"G{family_index:03}"
        for member in range(2):
            row = _row(family, member, [0.5, 0.9, 0.9, 0.9], [1, 4, 4, 4])
            rows.append(row)
            values[row.dataset] = [float(family_index), float(member)]
    result = evaluate_meta_antigravity(tuple(rows), values, ("f0", "f1"))
    for key in TARGET_KEYS_ANTIGRAVITY:
        assert result["summary"]["positive_gain_auroc"][key]["auroc"] is None
        assert result["summary"]["positive_gain_auroc"][key]["reason"] == "all_same_class"


def test_family_weights_sum_to_one_when_a_family_has_duplicate_feature_rows():
    rows, values, keys = _population()
    extra = replace(rows[1], dataset="F000_normal_9")
    widened = rows + (extra,)
    values = dict(values)
    values[extra.dataset] = list(values[rows[1].dataset])
    result = evaluate_meta_antigravity(widened, values, keys)
    for fold in result["folds"]:
        assert all(abs(total - 1.0) < 1e-12 for total in fold["family_weight_totals"].values())
        assert abs(fold["family_weight_sum"] - fold["n_donor_families"]) < 1e-12
        assert fold["ridge_alpha"] == 10.0


def test_infinite_descriptors_and_duplicate_ids_are_rejected():
    rows, values, keys = _population()
    values[rows[0].dataset] = [float("inf"), 0.0, 1.0]
    with pytest.raises(ValueError, match="infinite"):
        evaluate_meta_antigravity(rows, values, keys)
    values[rows[0].dataset] = [0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="unique"):
        evaluate_meta_antigravity(rows, values, ("f0", "f0", "f2"))
    with pytest.raises(ValueError, match="dataset IDs"):
        evaluate_meta_antigravity(rows + (rows[0],), values, keys)


def test_query_targets_are_not_required_to_build_predictions():
    rows, values, keys = _population()
    result = evaluate_meta_antigravity(rows, values, keys)
    held = result["folds"][0]
    expected = target_vector_antigravity(rows[0]).tolist()
    assert held["targets"][TARGET_KEYS_ANTIGRAVITY[0]][0] == pytest.approx(expected[0])
    assert "F000" not in held["donor_families"]


def test_official_result_cannot_be_overwritten(tmp_path):
    from experiments.exp491_gain_prediction_antigravity.run_antigravity import run

    path = tmp_path / "results_antigravity.json"
    path.write_text("frozen")
    with pytest.raises(RuntimeError, match="already exists"):
        run("official", output=tmp_path)
    assert path.read_text() == "frozen"


def test_missing_preregistration_fails(monkeypatch, tmp_path):
    from experiments.exp491_gain_prediction_antigravity import runtime_antigravity as runtime

    monkeypatch.setattr(runtime, "PREREG_PATH_ANTIGRAVITY", tmp_path / "missing_antigravity.html")
    with pytest.raises(RuntimeError, match="preregistration"):
        runtime.preregistration_sha256_antigravity()
    with pytest.raises(RuntimeError, match="does not exist"):
        runtime.source_binding_antigravity(database=tmp_path / "missing_antigravity.db")


@pytest.mark.parametrize("experiment,mutation", [(489, "source"), (490, "source"), (491, "source"), (490, "cache"), (491, "cache")])
def test_input_replacement_during_loading_is_rejected(tmp_path, monkeypatch, experiment, mutation):
    import importlib
    names = {489: "lof_score_head", 490: "behavior_descriptors", 491: "gain_prediction"}
    runner = importlib.import_module(f"experiments.exp{experiment}_{names[experiment]}_antigravity.run_antigravity")
    from experiments.exp491_gain_prediction_antigravity.run_antigravity import synthetic_evidence_rows_antigravity, synthetic_descriptors_antigravity
    rows = synthetic_evidence_rows_antigravity()
    values = synthetic_descriptors_antigravity(rows)
    database = tmp_path / "fixture_antigravity.db"
    database.touch()
    cache = tmp_path / "descriptor_antigravity.json"
    cache.write_text("version A")
    calls = []
    def sources():
        calls.append(None)
        return rows, None, {"source": "B" if mutation == "source" and len(calls) > 1 else "A"}
    monkeypatch.setattr(runner, "load_frozen_sources_antigravity", sources)
    monkeypatch.setattr(runner, "validate_database_antigravity", lambda *a, **k: None)
    monkeypatch.setattr(runner, "EXPECTED_REPRESENTATIVE_FAMILIES_ANTIGRAVITY", 6, raising=False)
    monkeypatch.setattr(runner, "EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY", (), raising=False)
    monkeypatch.setattr(runner, "EXPECTED_EXP491_DATASETS_ANTIGRAVITY", len(rows), raising=False)
    monkeypatch.setattr(runner, "EXPECTED_EXP491_FAMILIES_ANTIGRAVITY", 6, raising=False)
    if experiment != 489:
        def load_descriptors(*args):
            if mutation == "cache":
                cache.write_text("version B")
            return values
        name = "_load_24d_antigravity" if experiment == 490 else "load_exp488_descriptors_antigravity"
        monkeypatch.setattr(runner, name, load_descriptors)
    kwargs = {} if experiment == 489 else {"descriptor_cache": cache}
    with pytest.raises(RuntimeError, match="changed while loading inputs"):
        runner.run("smoke", database, tmp_path / "output_antigravity", **kwargs)
    assert not (tmp_path / "output_antigravity/smoke_results_antigravity.json").exists()
