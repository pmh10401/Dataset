"""Task 5 execution boundaries for Exp481."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from experiments.exp481_strict_evidence_residual_correction_antigravity.run_antigravity import (
    build_official_result_antigravity,
    build_run_manifest_antigravity,
    json_bytes_antigravity,
    render_dashboard_antigravity,
    run_smoke_antigravity,
    run_validate_antigravity,
    validate_run_manifest_antigravity,
    write_json_atomic_antigravity,
)


def _evaluation_antigravity() -> dict[str, object]:
    return {
        "backend": "numpy_reference",
        "dataset_count": 8,
        "family_count": 8,
        "summaries": {
            "row_knn": {"zf_total": 10, "dataset_mean_auc": 0.9},
            "exp481": {"zf_total": 10, "dataset_mean_auc": 0.9},
            "changed": {"count": 0},
        },
        "gates": {"overall_pass": False},
        "leakage": {
            "family_overlap_count": 0,
            "outer_family_seen_by_inner_count": 0,
            "route_input_key_count": 7,
            "dataset_name_feature_count": 0,
            "held_out_score_feature_count": 0,
            "held_out_scores_read_after_routing": True,
        },
        "inner_gates_by_outer_family": {},
        "decisions": [],
    }


def _backend_antigravity() -> dict[str, object]:
    return {
        "selected_backend": "numpy_reference",
        "fallback_reason": "numba_exact_frozen_stage_not_benchmarked",
        "candidate_index_parity": True,
        "decision_parity": True,
        "finite_outputs": True,
        "exact_frozen_stage_benchmarked": False,
        "numba_cold_jit_verified": False,
    }


def test_validate_reports_frozen_inputs_and_backend_antigravity() -> None:
    result = run_validate_antigravity()
    assert result["state"] == "ready_for_official"
    assert (result["datasets"], result["families"]) == (1117, 128)
    assert result["selected_backend"] == "numpy_reference"
    assert result["fallback_reason"] == "numba_exact_frozen_stage_not_benchmarked"


def test_smoke_never_writes_official_result_antigravity(tmp_path: Path) -> None:
    result = run_smoke_antigravity(family_count=8, output_dir=tmp_path)
    assert result["state"] == "smoke_nonofficial"
    assert result["family_count"] == 8
    assert result["evaluation"]["leakage"]["family_overlap_count"] == 0
    assert result["evaluation"]["leakage"]["route_input_key_count"] == 7
    assert (tmp_path / "smoke_results_antigravity.json").exists()
    assert not (tmp_path / "results_antigravity.json").exists()
    assert not (tmp_path / "run_manifest_antigravity.json").exists()


def test_same_input_produces_identical_result_bytes_antigravity() -> None:
    first = build_official_result_antigravity(
        _evaluation_antigravity(), backend_assessment=_backend_antigravity()
    )
    second = build_official_result_antigravity(
        _evaluation_antigravity(), backend_assessment=_backend_antigravity()
    )
    assert json_bytes_antigravity(first) == json_bytes_antigravity(second)
    assert b"elapsed_seconds" not in json_bytes_antigravity(first)


def test_manifest_keeps_runtime_outside_result_antigravity() -> None:
    result_bytes = json_bytes_antigravity({"schema": "demo"})
    manifest = build_run_manifest_antigravity(
        result_bytes=result_bytes,
        backend_assessment=_backend_antigravity(),
        started_at_utc="2026-09-01T00:00:00+00:00",
        completed_at_utc="2026-09-01T00:00:01+00:00",
        elapsed_seconds=1.0,
    )
    assert manifest["result_sha256"] == sha256(result_bytes).hexdigest()
    assert manifest["elapsed_seconds"] == 1.0
    validate_run_manifest_antigravity(
        manifest, expected_result_sha256=sha256(result_bytes).hexdigest()
    )


def test_manifest_rejects_result_hash_drift_antigravity() -> None:
    result_bytes = json_bytes_antigravity({"schema": "demo"})
    manifest = build_run_manifest_antigravity(
        result_bytes=result_bytes,
        backend_assessment=_backend_antigravity(),
        started_at_utc="2026-09-01T00:00:00+00:00",
        completed_at_utc="2026-09-01T00:00:01+00:00",
        elapsed_seconds=1.0,
    )
    import pytest

    with pytest.raises(RuntimeError, match="result SHA"):
        validate_run_manifest_antigravity(manifest, expected_result_sha256="0" * 64)


def test_dashboard_checks_task5_not_task6_after_smoke_antigravity() -> None:
    html = render_dashboard_antigravity(
        {
            "state": "smoke_nonofficial",
            "task5_complete": True,
            "family_count": 8,
            "dataset_count": 8,
            "selected_backend": "numpy_reference",
            "fallback_reason": "numba_exact_frozen_stage_not_benchmarked",
            "evaluation": _evaluation_antigravity(),
        }
    )
    assert '<input type="checkbox" checked disabled> Task 5' in html
    assert '<input type="checkbox" disabled> Task 6' in html
    assert "공식 성능 결과가 아닙니다" in html


def test_atomic_writer_leaves_no_temporary_file_antigravity(tmp_path: Path) -> None:
    target = tmp_path / "result_antigravity.json"
    write_json_atomic_antigravity(target, {"value": 1})
    assert target.read_bytes() == json_bytes_antigravity({"value": 1})
    assert tuple(tmp_path.glob("*temporary_antigravity*")) == ()
