"""Result and dashboard behavior tests for Exp484."""

from __future__ import annotations

from hashlib import sha256
import json


def _evaluation_antigravity(
    *, changed: int, passed: bool, zf_total: int, family_mean_zf: float
) -> dict[str, object]:
    return {
        "dataset_count": 1117,
        "family_count": 128,
        "summaries": {
            "exp481": {
                "zf_total": zf_total,
                "family_mean_zf": family_mean_zf,
                "dataset_mean_auc": 0.984,
                "family_mean_auc": 0.973,
                "added_loss_0_05": 0,
                "added_loss_0_10": 0,
            },
            "changed": {
                "count": changed,
                "wins": changed,
                "losses": 0,
                "ties": 0,
                "tie_rate": 0.0,
                "method_counts": {"hydra_local_gap": changed} if changed else {},
            },
        },
        "gates": {"overall_pass": passed},
        "leakage": {
            "family_overlap_count": 0,
            "outer_family_seen_by_inner_count": 0,
            "dataset_name_feature_count": 0,
            "held_out_score_feature_count": 0,
            "route_input_keys": (
                "width", "n_train", "spectral_entropy", "skewness", "kurtosis",
                "diff_std_ratio", "zero_crossing_rate",
            ),
        },
        "inner_gates_by_outer_family": {
            "F1": (
                {
                    "method": "hydra_local_gap",
                    "enabled": bool(changed),
                    "changed_families": changed,
                },
            ),
        },
        "decisions": (),
    }


def test_choose_distance_prefers_the_stronger_qualified_variant_antigravity() -> None:
    """Break caught: picking a failed or lower-performing distance variant."""
    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        choose_distance_variant_antigravity,
    )

    decision = choose_distance_variant_antigravity(
        (
            {
                "distance_variant": "robust_manhattan",
                "overall_pass": True,
                "family_mean_zf": 3.10,
                "zf_total": 2120,
                "family_mean_auc": 0.972,
            },
            {
                "distance_variant": "robust_shrinkage_mahalanobis",
                "overall_pass": True,
                "family_mean_zf": 3.20,
                "zf_total": 2130,
                "family_mean_auc": 0.973,
            },
        )
    )

    assert decision == {
        "decision": "qualify_distance_variant",
        "selected_variant": "robust_shrinkage_mahalanobis",
        "reason": "highest_qualified_family_mean_zf",
    }


def test_choose_distance_retires_distance_only_change_when_all_fail_antigravity() -> None:
    """Break caught: promoting a distance that fails the frozen performance gates."""
    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        choose_distance_variant_antigravity,
    )

    decision = choose_distance_variant_antigravity(
        (
            {
                "distance_variant": "robust_manhattan",
                "overall_pass": False,
                "family_mean_zf": 3.30,
                "zf_total": 2200,
                "family_mean_auc": 0.970,
            },
            {
                "distance_variant": "robust_shrinkage_mahalanobis",
                "overall_pass": False,
                "family_mean_zf": 3.40,
                "zf_total": 2250,
                "family_mean_auc": 0.969,
            },
        )
    )

    assert decision == {
        "decision": "replace_features",
        "selected_variant": None,
        "reason": "no_distance_variant_passed_frozen_gates",
    }


def test_summarize_evaluation_keeps_performance_and_inner_family_counts_antigravity() -> None:
    """Break caught: reporting changes without the frozen gates or family denominator."""
    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        summarize_evaluation_antigravity,
    )

    evaluation = {
        "dataset_count": 1117,
        "family_count": 128,
        "summaries": {
            "exp481": {
                "zf_total": 2120,
                "family_mean_zf": 3.1,
                "dataset_mean_auc": 0.984,
                "family_mean_auc": 0.973,
                "added_loss_0_05": 0,
                "added_loss_0_10": 0,
            },
            "changed": {
                "count": 10,
                "wins": 8,
                "losses": 1,
                "ties": 1,
                "tie_rate": 0.1,
                "method_counts": {"hydra_local_gap": 10},
            },
        },
        "gates": {"overall_pass": True},
        "inner_gates_by_outer_family": {
            "F1": (
                {"method": "hydra_local_gap", "enabled": True},
                {"method": "multirocket_local_gap", "enabled": False},
            ),
            "F2": (
                {"method": "hydra_local_gap", "enabled": True},
                {"method": "multirocket_local_gap", "enabled": True},
            ),
        },
    }

    summary = summarize_evaluation_antigravity(
        "robust_manhattan", "numba_cpu", evaluation
    )

    assert summary["distance_variant"] == "robust_manhattan"
    assert summary["backend"] == "numba_cpu"
    assert summary["dataset_count"] == 1117
    assert summary["family_count"] == 128
    assert summary["changed_count"] == 10
    assert summary["changed_wins"] == 8
    assert summary["zf_total"] == 2120
    assert summary["family_mean_zf"] == 3.1
    assert summary["overall_pass"] is True
    assert summary["inner_any_enabled_outer_family_count"] == 2
    assert summary["inner_enabled_outer_family_counts"] == {
        "hydra_local_gap": 2,
        "multirocket_local_gap": 1,
    }


def test_failure_summary_distinguishes_local_gate_failure_antigravity() -> None:
    """Break caught: blaming the outer family threshold when no local route passed."""
    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        summarize_failure_structure_antigravity,
    )

    evaluation = _evaluation_antigravity(
        changed=0, passed=False, zf_total=2111, family_mean_zf=2.944
    )
    evaluation["decisions"] = (
        {"reason": "strict_local_gate_failed"},
        {"reason": "strict_local_gate_failed"},
        {"reason": "row_knn_alternative_preserved"},
    )

    summary = summarize_failure_structure_antigravity(
        {"robust_manhattan": evaluation}
    )

    assert summary == {
        "robust_manhattan": {
            "max_inner_changed_families": 0,
            "enabled_outer_family_count": 0,
            "decision_reason_counts": {
                "row_knn_alternative_preserved": 1,
                "strict_local_gate_failed": 2,
            },
        }
    }


def test_dashboard_shows_side_by_side_results_and_checked_tasks_antigravity() -> None:
    """Break caught: hiding a variant or leaving completed task boxes unchecked."""
    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        render_dashboard_antigravity,
    )

    rows = tuple(
        {
            "distance_variant": name,
            "backend": "numba_cpu",
            "changed_count": changed,
            "changed_wins": changed,
            "changed_losses": 0,
            "zf_total": 2120 + changed,
            "family_mean_zf": 3.0 + changed / 100,
            "dataset_mean_auc": 0.984,
            "family_mean_auc": 0.973,
            "added_loss_0_05": 0,
            "overall_pass": passed,
            "inner_any_enabled_outer_family_count": changed,
        }
        for name, changed, passed in (
            ("standard_euclidean", 0, False),
            ("robust_manhattan", 21, True),
            ("robust_shrinkage_mahalanobis", 25, True),
        )
    )
    html = render_dashboard_antigravity(
        {
            "state": "verified_complete",
            "variant_summaries": rows,
            "decision": {
                "decision": "qualify_distance_variant",
                "selected_variant": "robust_shrinkage_mahalanobis",
            },
            "route_input_audit": {
                "dataset_name_feature_count": 0,
                "family_name_feature_count": 0,
            },
        }
    )

    assert html.count("checked disabled") == 5
    assert "standard_euclidean" in html
    assert "robust_manhattan" in html
    assert "robust_shrinkage_mahalanobis" in html
    assert "선택: robust_shrinkage_mahalanobis" in html
    assert "이름 특징 0개" in html


def test_dashboard_leaves_verification_unchecked_until_verified_antigravity() -> None:
    """Break caught: marking final verification complete when only evaluation finished."""
    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        render_dashboard_antigravity,
    )

    result = {
        "variant_summaries": (),
        "decision": {"selected_variant": None},
        "route_input_audit": {
            "dataset_name_feature_count": 0,
            "family_name_feature_count": 0,
        },
    }

    html = render_dashboard_antigravity(result)

    assert html.count("checked disabled") == 4
    assert "Task 5 — 결과 해석 및 검증" in html


def test_build_result_keeps_control_and_two_comparable_variants_antigravity() -> None:
    """Break caught: dropping the control or mixing different evaluation populations."""
    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        build_result_antigravity,
    )

    control = {
        "selected_backend": "numpy_reference",
        "evaluation": _evaluation_antigravity(
            changed=0, passed=False, zf_total=2111, family_mean_zf=2.944
        ),
    }
    evaluations = {
        "robust_manhattan": _evaluation_antigravity(
            changed=21, passed=True, zf_total=2120, family_mean_zf=3.10
        ),
        "robust_shrinkage_mahalanobis": _evaluation_antigravity(
            changed=25, passed=True, zf_total=2130, family_mean_zf=3.20
        ),
    }
    acceleration = {
        "schema": "exp484_acceleration_v1",
        "variants": {
            name: {"selected_backend": "numba_cpu"} for name in evaluations
        },
        "mlx": {"assessed": True, "available": False},
    }

    result = build_result_antigravity(control, evaluations, acceleration)

    assert result["schema"] == "exp484_distance_qualification_v1"
    assert [row["distance_variant"] for row in result["variant_summaries"]] == [
        "standard_euclidean",
        "robust_manhattan",
        "robust_shrinkage_mahalanobis",
    ]
    assert result["decision"]["selected_variant"] == "robust_shrinkage_mahalanobis"
    assert result["route_input_audit"] == {
        "dataset_name_feature_count": 0,
        "family_name_feature_count": 0,
        "held_out_score_feature_count": 0,
    }
    assert result["evaluations"] == evaluations


def test_report_explains_the_frozen_scope_and_next_step_antigravity() -> None:
    """Break caught: describing the comparison as a feature or gate change."""
    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        render_report_antigravity,
    )

    html = render_report_antigravity(
        {
            "decision": {
                "decision": "replace_features",
                "selected_variant": None,
                "reason": "no_distance_variant_passed_frozen_gates",
            },
            "variant_summaries": (
                {"distance_variant": "standard_euclidean", "overall_pass": False},
                {"distance_variant": "robust_manhattan", "overall_pass": False},
                {
                    "distance_variant": "robust_shrinkage_mahalanobis",
                    "overall_pass": False,
                },
            ),
            "acceleration": {
                "variants": {
                    "robust_manhattan": {
                        "selected_backend": "numba_cpu",
                        "speedup": 1.5,
                    },
                    "robust_shrinkage_mahalanobis": {
                        "selected_backend": "numpy_reference",
                        "speedup": 0.8,
                    },
                },
                "mlx": {
                    "available": False,
                    "fallback_reason": "mlx_metal_device_unavailable",
                },
            },
            "failure_structure": {
                "robust_manhattan": {
                    "max_inner_changed_families": 0,
                    "decision_reason_counts": {"strict_local_gate_failed": 1028},
                }
            },
        }
    )

    assert "7개 특징은 바꾸지 않았습니다" in html
    assert "K=31" in html
    assert "다음에는 특징 자체를 바꿉니다" in html
    assert "mlx_metal_device_unavailable" in html
    assert "이번에 시험한 두 거리 안에서는" in html
    assert "정보가 부족하다는 뜻" not in html


def test_publish_writes_result_manifest_and_checked_html_antigravity(tmp_path) -> None:
    """Break caught: publishing partial files or a manifest for different result bytes."""
    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        build_result_antigravity,
        publish_result_antigravity,
    )

    control = {
        "selected_backend": "numpy_reference",
        "evaluation": _evaluation_antigravity(
            changed=0, passed=False, zf_total=2111, family_mean_zf=2.944
        ),
    }
    evaluations = {
        "robust_manhattan": _evaluation_antigravity(
            changed=21, passed=True, zf_total=2120, family_mean_zf=3.10
        ),
        "robust_shrinkage_mahalanobis": _evaluation_antigravity(
            changed=25, passed=True, zf_total=2130, family_mean_zf=3.20
        ),
    }
    acceleration = {
        "schema": "exp484_acceleration_v1",
        "variants": {
            name: {"selected_backend": "numba_cpu", "speedup": 1.5}
            for name in evaluations
        },
        "mlx": {
            "assessed": True,
            "available": False,
            "fallback_reason": "mlx_metal_device_unavailable",
        },
    }
    result = build_result_antigravity(control, evaluations, acceleration)
    result["state"] = "verified_complete"

    paths = publish_result_antigravity(
        result,
        root=tmp_path,
        started_at_utc="2026-09-02T00:00:00+00:00",
        completed_at_utc="2026-09-02T00:00:02+00:00",
        elapsed_seconds=2.0,
    )

    assert set(paths) == {"result", "manifest", "acceleration", "dashboard", "report"}
    result_bytes = paths["result"].read_bytes()
    assert json.loads(result_bytes)["schema"] == "exp484_distance_qualification_v1"
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["result_sha256"] == sha256(result_bytes).hexdigest()
    assert paths["dashboard"].read_text(encoding="utf-8").count(
        "checked disabled"
    ) == 5
    assert paths["report"].exists()


def test_evaluate_variants_runs_the_same_population_for_both_distances_antigravity() -> None:
    """Break caught: evaluating one variant on a different family population."""
    from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
        CandidateMetricAntigravity,
        EvidenceRowAntigravity,
    )
    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        evaluate_variants_antigravity,
    )

    rows = tuple(
        EvidenceRowAntigravity(
            dataset=f"F{index}_normal_1",
            source_family=f"F{index}",
            proposal="minirocket_local_gap",
            features=(10.0, 20.0, 0.5, float(index), 3.0, 1.0, 0.2),
            scores=(
                CandidateMetricAntigravity("minirocket_local_gap", 0.70, 2),
                CandidateMetricAntigravity("multirocket_local_gap", 0.80, 3),
                CandidateMetricAntigravity("hydra_local_gap", 0.70, 2),
                CandidateMetricAntigravity("hydra_multirocket_local_gap", 0.70, 2),
            ),
        )
        for index in range(4)
    )
    acceleration = {
        "variants": {
            "robust_manhattan": {"selected_backend": "numpy_reference"},
            "robust_shrinkage_mahalanobis": {
                "selected_backend": "numpy_reference"
            },
        }
    }

    evaluations = evaluate_variants_antigravity(rows, acceleration)

    assert tuple(evaluations) == (
        "robust_manhattan",
        "robust_shrinkage_mahalanobis",
    )
    assert {
        (evaluation["dataset_count"], evaluation["family_count"])
        for evaluation in evaluations.values()
    } == {(4, 4)}
    assert all(
        evaluation["leakage"]["family_overlap_count"] == 0
        for evaluation in evaluations.values()
    )


def test_source_hash_gate_rejects_changed_input_antigravity(tmp_path) -> None:
    """Break caught: running the comparison after a frozen source changed."""
    import pytest

    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        require_sha256_antigravity,
    )

    source = tmp_path / "source_antigravity.json"
    source.write_bytes(b"frozen\n")
    expected = sha256(b"frozen\n").hexdigest()

    assert require_sha256_antigravity(source, expected) == expected
    source.write_bytes(b"changed\n")
    with pytest.raises(RuntimeError, match="source SHA changed"):
        require_sha256_antigravity(source, expected)


def test_load_frozen_sources_keeps_1117_by_128_population_antigravity() -> None:
    """Break caught: running Exp484 on a different evidence or control snapshot."""
    from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
        load_frozen_sources_antigravity,
    )

    rows, control, source_sha = load_frozen_sources_antigravity()

    assert len(rows) == 1117
    assert len({row.source_family for row in rows}) == 128
    assert control["dataset_count"] == 1117
    assert control["family_count"] == 128
    assert source_sha == {
        "evidence_database": "77c53e1afd0af071edf71ef692bd978458dda6a5f8a0fec110736b787fe1b998",
        "exp481_result": "7038b994e422dc93a18927c810ce5e3bdb7601039ba01d2b388895444e09956f",
        "exp483_result": "09205ac93d4a4baa8fa0992a20d89d811d13fbbb2f2a55d51c9be758946ef9cc",
    }
