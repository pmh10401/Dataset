"""Task 4 tests for identical out-of-family qualification."""

from __future__ import annotations

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.evaluation_antigravity import (
    apply_qualification_gates_antigravity,
    evaluate_out_of_family_antigravity,
    iter_family_splits_antigravity,
    summarise_method_rows_antigravity,
)


def _row_antigravity(
    dataset: str,
    offset: float,
    *,
    hydra_auc: float = 0.81,
    hydra_zf: int = 2,
) -> EvidenceRowAntigravity:
    return EvidenceRowAntigravity(
        dataset=dataset,
        source_family=dataset.split("_normal_", 1)[0],
        proposal="hydra_local_gap",
        features=(offset, 10.0, 0.5, 0.0, 3.0, 1.0, 0.2),
        scores=(
            CandidateMetricAntigravity("minirocket_local_gap", 0.80, 1),
            CandidateMetricAntigravity("multirocket_local_gap", 0.80, 1),
            CandidateMetricAntigravity("hydra_local_gap", hydra_auc, hydra_zf),
            CandidateMetricAntigravity("hydra_multirocket_local_gap", 0.80, 1),
        ),
    )


def _synthetic_rows_antigravity() -> tuple[EvidenceRowAntigravity, ...]:
    return tuple(
        _row_antigravity(f"{family}_normal_1", index / 10)
        for index, family in enumerate(("A", "B", "C", "D"))
    )


def test_family_splits_exclude_every_held_out_row_antigravity() -> None:
    """Allowing even one held-family row into fit must fail the split contract."""
    rows = _synthetic_rows_antigravity()
    splits = tuple(iter_family_splits_antigravity(rows))

    assert tuple(held for held, _, _ in splits) == ("A", "B", "C", "D")
    for held, train_rows, test_rows in splits:
        assert {row.source_family for row in test_rows} == {held}
        assert held not in {row.source_family for row in train_rows}
        assert len(train_rows) == 3
        assert len(test_rows) == 1


def test_out_of_family_evaluation_has_zero_overlap_and_full_coverage_antigravity() -> None:
    """Dropping a target or fitting its family must fail the evaluation result."""
    rows = _synthetic_rows_antigravity()
    result = evaluate_out_of_family_antigravity(
        rows,
        family_policy_kwargs={"k": 3, "min_support": 3},
        row_policy_kwargs={"k": 3, "min_support": 3},
    )

    assert result["dataset_count"] == 4
    assert result["family_count"] == 4
    assert result["leakage"] == {
        "family_overlap_count": 0,
        "dataset_name_feature_count": 0,
        "held_out_score_feature_count": 0,
    }
    assert {row["dataset"] for row in result["decisions"]} == {
        row.dataset for row in rows
    }
    assert all(
        decision["held_family"] not in decision["neighbour_families"]
        for decision in result["decisions"]
    )


def test_family_macro_summary_does_not_weight_large_families_more_antigravity() -> None:
    """Replacing family macro with the dataset mean must fail this uneven fixture."""
    summary = summarise_method_rows_antigravity(
        (
            {"held_family": "A", "auc": 0.50, "zf": 0, "baseline_auc": 0.60},
            {"held_family": "A", "auc": 0.70, "zf": 2, "baseline_auc": 0.70},
            {"held_family": "B", "auc": 0.90, "zf": 4, "baseline_auc": 0.90},
        ),
        accepted_non_baseline=2,
    )

    assert summary == {
        "dataset_count": 3,
        "dataset_mean_auc": 0.7000000000000001,
        "zf_total": 6,
        "family_mean_auc": 0.75,
        "family_mean_zf": 2.5,
        "loss_over_0_05": 1,
        "loss_over_0_10": 0,
        "accepted_non_baseline": 2,
    }


def test_gate_requires_strict_zf_improvement_without_auc_regression_antigravity() -> None:
    """Using non-strict ZF or allowing either AUC mean to regress must fail."""
    measured = {
        "mini": {
            "zf_total": 100, "family_mean_zf": 1.0,
            "dataset_mean_auc": 0.90, "family_mean_auc": 0.90,
            "loss_over_0_05": 0, "loss_over_0_10": 0,
        },
        "row_knn": {
            "zf_total": 101, "family_mean_zf": 1.01,
            "dataset_mean_auc": 0.90, "family_mean_auc": 0.90,
            "loss_over_0_05": 2, "loss_over_0_10": 1,
        },
        "exp480": {
            "zf_total": 102, "family_mean_zf": 1.02,
            "dataset_mean_auc": 0.91, "family_mean_auc": 0.91,
            "loss_over_0_05": 2, "loss_over_0_10": 1,
            "accepted_non_baseline": 1,
        },
    }
    gates = apply_qualification_gates_antigravity(measured)

    assert gates["zf_total"]["pass"] is True
    assert gates["family_mean_zf"]["pass"] is True
    assert gates["dataset_mean_auc"]["pass"] is True
    assert gates["family_mean_auc"]["pass"] is True
    assert gates["overall_pass"] is True

    measured["exp480"]["zf_total"] = 101
    assert apply_qualification_gates_antigravity(measured)["overall_pass"] is False

