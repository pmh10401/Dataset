"""Task 3 tests for nested family evaluation and frozen performance gates."""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.contracts_antigravity import (
    ResidualDecisionAntigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.evaluation_antigravity import (
    MeasuredDecisionAntigravity,
    aggregate_inner_gate_antigravity,
    apply_exp481_gates_antigravity,
    evaluate_nested_out_of_family_antigravity,
    evaluate_outer_row_antigravity,
)


def _row_antigravity(family: str, index: int) -> EvidenceRowAntigravity:
    baseline_auc = 0.80 + index * 0.001
    baseline_zf = 2 + index
    return EvidenceRowAntigravity(
        dataset=f"{family}_normal_{index}",
        source_family=family,
        proposal="minirocket_local_gap",
        features=(float(index + 1), 20.0, 0.4, 0.1, 3.0, 1.0, 0.2),
        scores=(
            CandidateMetricAntigravity("minirocket_local_gap", baseline_auc, baseline_zf),
            CandidateMetricAntigravity("multirocket_local_gap", baseline_auc + 0.01, baseline_zf + 1),
            CandidateMetricAntigravity("hydra_local_gap", baseline_auc - 0.01, baseline_zf - 1),
            CandidateMetricAntigravity("hydra_multirocket_local_gap", baseline_auc - 0.02, baseline_zf),
        ),
    )


def _decision_antigravity(
    family: str,
    *,
    method: str = "multirocket_local_gap",
    auc_gain: float = 0.01,
    zf_gain: int = 1,
) -> MeasuredDecisionAntigravity:
    return MeasuredDecisionAntigravity(
        dataset=f"{family}_normal_1",
        held_family=family,
        row_knn_method="minirocket_local_gap",
        method=method,
        changed=True,
        auc=0.91,
        zf=4,
        row_knn_auc=0.91 - auc_gain,
        row_knn_zf=4 - zf_gain,
        auc_gain=auc_gain,
        zf_gain=zf_gain,
        reason="strict_candidate_accept",
        inner_enabled_methods=(method,),
    )


def _summary_antigravity() -> dict[str, object]:
    return {
        "row_knn": {
            "dataset_count": 1117,
            "dataset_mean_auc": 0.9832866538430738,
            "zf_total": 2111,
            "family_mean_auc": 0.9718432393318419,
            "family_mean_zf": 2.9440228392325434,
        },
        "exp481": {
            "dataset_count": 1117,
            "dataset_mean_auc": 0.9833,
            "zf_total": 2112,
            "family_mean_auc": 0.972,
            "family_mean_zf": 2.95,
            "added_loss_0_05": 0,
            "added_loss_0_10": 0,
        },
        "changed": {
            "count": 3,
            "wins": 2,
            "losses": 0,
            "ties": 1,
            "tie_rate": 1.0 / 3.0,
            "method_counts": {"multirocket_local_gap": 3},
            "dominance_rate": 1.0,
            "dominant_method": "multirocket_local_gap",
            "dominance_control": {"method": "multirocket_local_gap"},
        },
    }


def test_inner_gate_requires_twenty_changed_families_antigravity() -> None:
    nineteen = tuple(_decision_antigravity(f"F{i}") for i in range(19))
    twenty = tuple(_decision_antigravity(f"F{i}") for i in range(20))
    assert aggregate_inner_gate_antigravity("multirocket_local_gap", nineteen).enabled is False
    gate = aggregate_inner_gate_antigravity("multirocket_local_gap", twenty)
    assert gate.enabled is True
    assert (gate.changed_families, gate.winning_families, gate.losing_families) == (20, 20, 0)


def test_inner_gate_uses_family_zf_and_actual_changed_auc_antigravity() -> None:
    records = (
        _decision_antigravity("A", zf_gain=1),
        _decision_antigravity("A", zf_gain=-1),
        _decision_antigravity("B", zf_gain=-1),
        _decision_antigravity("C", zf_gain=1, auc_gain=0.0),
    )
    gate = aggregate_inner_gate_antigravity(
        "multirocket_local_gap", records, min_changed_families=3
    )
    assert (gate.winning_families, gate.losing_families, gate.tied_families) == (1, 1, 1)
    assert gate.min_auc_gain == 0.0
    assert gate.enabled is False


def test_empty_inner_gate_has_finite_zero_sentinels_antigravity() -> None:
    gate = aggregate_inner_gate_antigravity("hydra_local_gap", ())
    assert gate.enabled is False
    assert gate.changed_families == 0
    assert gate.min_auc_gain == gate.q20_auc_gain == 0.0


def test_outer_row_reads_scores_only_after_both_routes_antigravity(monkeypatch) -> None:
    events: list[str] = []

    class RowPolicy:
        def route(self, features, proposal):
            events.append("row_route")
            return type("Decision", (), {"method": "minirocket_local_gap"})()

    class ResidualPolicy:
        def route(self, features, *, row_knn_method, enabled_methods):
            events.append("residual_route")
            return ResidualDecisionAntigravity(
                method="multirocket_local_gap",
                changed=True,
                reason="strict_candidate_accept",
                selected_candidate="multirocket_local_gap",
                assessments=(),
            )

    from experiments.exp481_strict_evidence_residual_correction_antigravity import evaluation_antigravity

    original = evaluation_antigravity._score_mapping_after_decision_antigravity

    def score_spy(row):
        events.append("score_read")
        return original(row)

    monkeypatch.setattr(evaluation_antigravity, "_score_mapping_after_decision_antigravity", score_spy)
    measured = evaluate_outer_row_antigravity(
        _row_antigravity("A", 1), RowPolicy(), ResidualPolicy(), ("multirocket_local_gap",)
    )
    assert events == ["row_route", "residual_route", "score_read"]
    assert measured.changed is True
    assert measured.auc_gain == pytest.approx(0.01)


def test_inner_qualification_rechecks_runner_up_under_actual_enabled_set_antigravity(
    monkeypatch,
) -> None:
    from experiments.exp481_strict_evidence_residual_correction_antigravity import evaluation_antigravity

    class FittedPolicy:
        def fit(self, rows):
            return self

    monkeypatch.setattr(
        evaluation_antigravity,
        "_row_policy_antigravity",
        lambda rows, kwargs: FittedPolicy(),
    )
    monkeypatch.setattr(
        evaluation_antigravity,
        "StrictResidualPolicyAntigravity",
        lambda **kwargs: FittedPolicy(),
    )

    def measured_for_subset(row, row_policy, residual_policy, enabled_methods):
        enabled = tuple(enabled_methods)
        if "multirocket_local_gap" in enabled:
            return _decision_antigravity(
                row.source_family,
                method="multirocket_local_gap",
                auc_gain=-0.01,
                zf_gain=1,
            )
        if "hydra_local_gap" in enabled:
            return _decision_antigravity(
                row.source_family,
                method="hydra_local_gap",
                auc_gain=0.01,
                zf_gain=1,
            )
        return _decision_antigravity(
            row.source_family,
            method="hydra_multirocket_local_gap",
            auc_gain=-0.01,
            zf_gain=-1,
        )

    monkeypatch.setattr(
        evaluation_antigravity,
        "evaluate_outer_row_antigravity",
        measured_for_subset,
    )
    rows = tuple(_row_antigravity(family, index) for index, family in enumerate(("A", "B", "C"), 1))
    gates = evaluation_antigravity.qualify_inner_methods_antigravity(
        rows, min_changed_families=1
    )
    enabled = tuple(gate.method for gate in gates if gate.enabled)
    assert enabled == ("hydra_local_gap",)


def test_final_gates_use_exact_strict_and_inclusive_boundaries_antigravity() -> None:
    passing = apply_exp481_gates_antigravity(_summary_antigravity())
    assert passing["overall_pass"] is True
    assert passing["zf_total"]["required"] == "> 2111"
    assert passing["dataset_mean_auc"]["required"] == ">= 0.9832866538430738"

    at_zf = _summary_antigravity()
    at_zf["exp481"] = {**at_zf["exp481"], "zf_total": 2111}
    assert apply_exp481_gates_antigravity(at_zf)["zf_total"]["pass"] is False

    loss = _summary_antigravity()
    loss["exp481"] = {**loss["exp481"], "added_loss_0_05": 1}
    assert apply_exp481_gates_antigravity(loss)["added_loss_0_05"]["pass"] is False


def test_no_change_fails_and_retires_feature_space_antigravity() -> None:
    summary = _summary_antigravity()
    summary["changed"] = {
        "count": 0,
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "tie_rate": 0.0,
        "method_counts": {},
        "dominance_rate": 0.0,
        "dominant_method": None,
        "dominance_control": None,
    }
    gates = apply_exp481_gates_antigravity(summary)
    assert gates["changed_count"]["pass"] is False
    assert gates["overall_pass"] is False
    assert gates["retire_seven_feature_space"] is True


def test_dominance_control_is_reported_but_not_an_automatic_failure_antigravity() -> None:
    gates = apply_exp481_gates_antigravity(_summary_antigravity())
    assert gates["dominance_control"]["triggered"] is True
    assert gates["dominance_control"]["available"] is True
    assert gates["overall_pass"] is True


def test_nested_evaluation_is_sorted_and_has_zero_family_leakage_antigravity() -> None:
    rows = tuple(_row_antigravity(family, index) for index, family in enumerate(("C", "A", "B"), 1))
    result = evaluate_nested_out_of_family_antigravity(
        rows,
        row_policy_kwargs={"k": 1, "min_support": 1},
        residual_policy_kwargs={"k": 1},
        inner_min_changed_families=1,
    )
    assert [item["dataset"] for item in result["decisions"]] == sorted(row.dataset for row in rows)
    assert result["backend"] == "numpy_reference"
    assert result["leakage"]["family_overlap_count"] == 0
    assert result["leakage"]["outer_family_seen_by_inner_count"] == 0
    assert result["leakage"]["dataset_name_feature_count"] == 0
    assert result["leakage"]["held_out_score_feature_count"] == 0
    assert result["leakage"]["held_out_scores_read_after_routing"] is True
    assert result["leakage"]["route_input_key_count"] == 7


def test_measured_contract_rejects_inconsistent_change_antigravity() -> None:
    with pytest.raises(ValueError, match="changed"):
        replace(_decision_antigravity("A"), method="minirocket_local_gap")
