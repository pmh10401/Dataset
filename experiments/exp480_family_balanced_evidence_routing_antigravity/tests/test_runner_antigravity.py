"""Task 5 tests for Exp480 execution and dashboard boundaries."""

from __future__ import annotations

from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path

import pytest

from experiments.exp480_family_balanced_evidence_routing_antigravity import (
    run_antigravity as runner_module_antigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.run_antigravity import (
    render_dashboard_antigravity,
    run_smoke_antigravity,
    run_validate_antigravity,
    write_json_atomic_antigravity,
)


def _row_antigravity(dataset: str, offset: float) -> EvidenceRowAntigravity:
    return EvidenceRowAntigravity(
        dataset=dataset,
        source_family=dataset.split("_normal_", 1)[0],
        proposal="hydra_local_gap",
        features=(offset, 10.0, 0.5, 0.0, 3.0, 1.0, 0.2),
        scores=(
            CandidateMetricAntigravity("minirocket_local_gap", 0.80, 1),
            CandidateMetricAntigravity("multirocket_local_gap", 0.80, 1),
            CandidateMetricAntigravity("hydra_local_gap", 0.81, 2),
            CandidateMetricAntigravity("hydra_multirocket_local_gap", 0.80, 1),
        ),
    )


def _qualification_result_antigravity(*, passed: bool = True) -> dict[str, object]:
    summaries = {
        "mini": {
            "zf_total": 100,
            "family_mean_zf": 1.0,
            "dataset_mean_auc": 0.80,
            "family_mean_auc": 0.79,
        },
        "row_knn": {
            "zf_total": 101,
            "family_mean_zf": 1.1,
            "dataset_mean_auc": 0.81,
            "family_mean_auc": 0.80,
        },
        "exp480": {
            "zf_total": 102 if passed else 99,
            "family_mean_zf": 1.2 if passed else 0.9,
            "dataset_mean_auc": 0.82 if passed else 0.78,
            "family_mean_auc": 0.81 if passed else 0.77,
            "accepted_non_baseline": 7,
        },
    }
    return {
        "schema": "exp480_official_qualification_v1",
        "state": "official_pending_verification",
        "task5_complete": True,
        "family_count": 128,
        "dataset_count": 1117,
        "selected_backend": "cpu_numpy",
        "evidence_database_sha256": "a" * 64,
        "evaluation": {
            "backend": "cpu_numpy",
            "family_count": 128,
            "dataset_count": 1117,
            "summaries": summaries,
            "gates": {"overall_pass": passed},
            "decisions": [],
        },
    }


def _mlx_recommendation_antigravity() -> dict[str, object]:
    return {
        "schema": "exp480_mlx_assessment_v1",
        "mlx_version": "0.32.0",
        "mlx_metal_version": "0.32.0",
        "device": "Device(gpu, 0)",
        "cpu_seconds": 0.22,
        "mlx_seconds": 0.10,
        "finite": True,
        "candidate_indices_equal": True,
        "tolerance": 1e-10,
        "max_abs_error": 1e-12,
        "speedup": 2.2,
        "selected_backend": "mlx",
        "fallback_reason": None,
        "fallback_detail": None,
    }


def _complete_cpu_manifest_antigravity(result_bytes: bytes) -> dict[str, object]:
    return {
        "schema": "exp480_run_manifest_v1",
        "started_at_utc": "2026-09-01T00:00:00+00:00",
        "completed_at_utc": "2026-09-01T00:00:02+00:00",
        "elapsed_seconds": 2.0,
        "result_sha256": sha256(result_bytes).hexdigest(),
        "acceleration_evidence": {
            "mlx_version": "0.32.0",
            "mlx_metal_version": "0.32.0",
            "device": "Device(gpu, 0)",
            "cpu_seconds": 0.22,
            "mlx_seconds": None,
            "finite": False,
            "candidate_indices_equal": False,
            "tolerance": 1e-10,
            "max_abs_error": None,
            "speedup": None,
            "recommended_backend": "cpu_numpy",
            "actual_result_backend": "cpu_numpy",
            "fallback_reason": "mlx_unavailable:ValueError",
            "fallback_detail": "float64 is not supported on the GPU",
        },
    }


def test_atomic_json_write_leaves_no_temporary_file_antigravity(tmp_path: Path) -> None:
    """A partial JSON artifact must never look like a finished result."""
    target = tmp_path / "result_antigravity.json"
    write_json_atomic_antigravity(target, {"schema": "demo", "value": 7})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "schema": "demo",
        "value": 7,
    }
    assert tuple(tmp_path.glob("*temporary_antigravity*")) == ()


def test_dashboard_checks_task5_but_not_task6_after_smoke_antigravity() -> None:
    """A smoke result must never appear as an approved official run."""
    html = render_dashboard_antigravity(
        {
            "state": "smoke_complete",
            "task5_complete": True,
            "family_count": 8,
            "dataset_count": 8,
            "selected_backend": "cpu_numpy",
            "backend_fallback_reason": "mlx_unavailable:ValueError",
        }
    )
    assert '<input type="checkbox" checked disabled> Task 5 — 실행기와 smoke 검증' in html
    assert '<input type="checkbox" disabled> Task 6 — 공식 1,117개 실행 승인' in html
    assert "공식 성능 결과가 아닙니다" in html
    assert "MLX 정확 연산 불가" in html


def test_validate_reports_the_exact_derived_population_antigravity() -> None:
    """A stale or partial evidence DB must not report ready."""
    assert run_validate_antigravity() == {
        "status": "ready",
        "datasets": 1117,
        "families": 128,
        "candidate_metrics": 4468,
    }


def test_real_eight_family_smoke_stays_nonofficial_antigravity(tmp_path: Path) -> None:
    """Smoke must evaluate eight real families without writing official results."""
    result = run_smoke_antigravity(
        family_count=8,
        result_path=tmp_path / "smoke_results_antigravity.json",
        dashboard_path=tmp_path / "dashboard_antigravity.html",
        assessment_path=tmp_path / "mlx_assessment_antigravity.json",
    )
    assert result["schema"] == "exp480_smoke_result_v1"
    assert result["state"] == "smoke_complete"
    assert result["family_count"] == 8
    assert result["task5_complete"] is True
    assert not (tmp_path / "results_antigravity.json").exists()


def test_official_result_is_truthfully_fixed_to_numpy_cpu_antigravity() -> None:
    """An MLX benchmark must never relabel the NumPy-only evaluation result."""
    rows = tuple(
        _row_antigravity(f"{family}_normal_1", index / 10)
        for index, family in enumerate(("A", "B", "C", "D"))
    )
    result = runner_module_antigravity.build_official_result_antigravity(
        rows,
        evidence_database_sha256="a" * 64,
    )

    assert result["selected_backend"] == "cpu_numpy"
    assert result["evaluation"]["backend"] == "cpu_numpy"


def test_manifest_records_full_acceleration_evidence_and_unwired_mlx_antigravity() -> None:
    """Dropping parity/timing/device evidence or pretending MLX computed results must fail."""
    result_bytes = b'{"selected_backend":"cpu_numpy"}\n'
    manifest = runner_module_antigravity.build_run_manifest_antigravity(
        result_bytes=result_bytes,
        acceleration=_mlx_recommendation_antigravity(),
        started_at_utc="2026-09-01T00:00:00+00:00",
        completed_at_utc="2026-09-01T00:00:02+00:00",
        elapsed_seconds=2.0,
    )

    assert manifest["result_sha256"] == sha256(result_bytes).hexdigest()
    assert manifest["started_at_utc"] == "2026-09-01T00:00:00+00:00"
    assert manifest["completed_at_utc"] == "2026-09-01T00:00:02+00:00"
    assert manifest["elapsed_seconds"] == 2.0
    assert manifest["acceleration_evidence"] == {
        "mlx_version": "0.32.0",
        "mlx_metal_version": "0.32.0",
        "device": "Device(gpu, 0)",
        "cpu_seconds": 0.22,
        "mlx_seconds": 0.10,
        "finite": True,
        "candidate_indices_equal": True,
        "tolerance": 1e-10,
        "max_abs_error": 1e-12,
        "speedup": 2.2,
        "recommended_backend": "mlx",
        "actual_result_backend": "cpu_numpy",
        "fallback_reason": "mlx_recommendation_unwired_numpy_evaluation",
        "fallback_detail": (
            "The official evaluation is NumPy-only; MLX benchmark output is not "
            "wired into result computation."
        ),
    }


def test_official_entry_fails_before_evaluation_or_outputs_antigravity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid evidence must stop before any official evaluation or artifact write."""
    def reject_antigravity(path: Path) -> dict[str, object]:
        del path
        raise RuntimeError("evidence source provenance changed")

    def forbidden_load_antigravity(path: Path):
        del path
        pytest.fail("evaluation input loaded after failed preflight")

    monkeypatch.setattr(
        runner_module_antigravity,
        "validate_evidence_database_antigravity",
        reject_antigravity,
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "load_evidence_database_antigravity",
        forbidden_load_antigravity,
    )
    with pytest.raises(RuntimeError, match="provenance"):
        runner_module_antigravity.run_official_antigravity(
            evidence_path=tmp_path / "evidence_antigravity.sqlite3",
            result_path=tmp_path / "results_antigravity.json",
            manifest_path=tmp_path / "run_manifest_antigravity.json",
            dashboard_path=tmp_path / "dashboard_antigravity.html",
            assessment_path=tmp_path / "assessment_antigravity.json",
        )

    assert tuple(tmp_path.iterdir()) == ()


def test_official_entry_atomically_writes_interim_state_without_report_antigravity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Official execution approval must remain distinct from result verification."""
    evidence_path = tmp_path / "evidence_antigravity.sqlite3"
    evidence_path.write_bytes(b"controlled-test-evidence")
    result_path = tmp_path / "results_antigravity.json"
    manifest_path = tmp_path / "run_manifest_antigravity.json"
    dashboard_path = tmp_path / "dashboard_antigravity.html"
    report_path = tmp_path / "implementation_report_antigravity.html"
    assessment_path = tmp_path / "assessment_antigravity.json"
    controlled_result = _qualification_result_antigravity()
    controlled_result["evidence_database_sha256"] = sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        runner_module_antigravity,
        "validate_evidence_database_antigravity",
        lambda path: {"datasets": 1117, "families": 128, "candidate_metrics": 4468},
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "load_evidence_database_antigravity",
        lambda path: (),
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "assess_mlx_antigravity",
        lambda rows, assessment_path: _mlx_recommendation_antigravity(),
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "build_official_result_antigravity",
        lambda rows, evidence_database_sha256: controlled_result,
    )

    runner_module_antigravity.run_official_antigravity(
        evidence_path=evidence_path,
        result_path=result_path,
        manifest_path=manifest_path,
        dashboard_path=dashboard_path,
        assessment_path=assessment_path,
    )

    stored = result_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dashboard = dashboard_path.read_text(encoding="utf-8")
    assert manifest["result_sha256"] == sha256(stored).hexdigest()
    assert manifest["acceleration_evidence"]["actual_result_backend"] == "cpu_numpy"
    assert '<input type="checkbox" checked disabled> Task 6 — 공식 1,117개 실행 승인' in dashboard
    assert '<input type="checkbox" disabled> Task 6 — 공식 결과 검증' in dashboard
    assert not report_path.exists()
    assert tuple(tmp_path.glob("*temporary_antigravity*")) == ()


def test_manifest_validator_rejects_missing_top_level_fields_antigravity() -> None:
    """A result SHA alone must never qualify as a complete execution manifest."""
    manifest = _complete_cpu_manifest_antigravity(b"official-result\n")
    del manifest["completed_at_utc"]

    with pytest.raises(RuntimeError, match="manifest fields"):
        runner_module_antigravity.validate_run_manifest_antigravity(
            manifest,
            expected_result_sha256=manifest["result_sha256"],
        )


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    (
        ("top", "schema", "wrong_schema", "schema"),
        ("top", "elapsed_seconds", float("nan"), "elapsed_seconds"),
        ("acceleration", "actual_result_backend", "mlx", "actual result backend"),
        ("acceleration", "fallback_reason", "", "fallback reason"),
        (
            "acceleration",
            "fallback_reason",
            "speedup_below_1_1x",
            "null relationship",
        ),
        ("acceleration", "cpu_seconds", True, "cpu_seconds"),
        ("acceleration", "speedup", 1.0, "null relationship"),
    ),
)
def test_manifest_validator_rejects_malformed_fields_antigravity(
    section: str, key: str, value: object, message: str
) -> None:
    """Malformed types, backend claims, fallback, or null evidence must fail."""
    manifest = _complete_cpu_manifest_antigravity(b"official-result\n")
    target = manifest if section == "top" else manifest["acceleration_evidence"]
    assert isinstance(target, dict)
    target[key] = value

    with pytest.raises(RuntimeError, match=message):
        runner_module_antigravity.validate_run_manifest_antigravity(
            manifest,
            expected_result_sha256=manifest["result_sha256"],
        )


def test_verify_rejects_incomplete_manifest_before_publication_antigravity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification must not publish HTML from a hash-only manifest."""
    evidence_path = tmp_path / "evidence_antigravity.sqlite3"
    evidence_path.write_bytes(b"controlled-test-evidence")
    result_path = tmp_path / "results_antigravity.json"
    manifest_path = tmp_path / "run_manifest_antigravity.json"
    dashboard_path = tmp_path / "dashboard_antigravity.html"
    report_path = tmp_path / "implementation_report_antigravity.html"
    stored_result = _qualification_result_antigravity()
    stored_result["evidence_database_sha256"] = sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    write_json_atomic_antigravity(result_path, stored_result)
    write_json_atomic_antigravity(
        manifest_path,
        {
            "schema": "exp480_run_manifest_v1",
            "result_sha256": sha256(result_path.read_bytes()).hexdigest(),
        },
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "validate_evidence_database_antigravity",
        lambda path: {"datasets": 1117},
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "load_evidence_database_antigravity",
        lambda path: (),
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "build_official_result_antigravity",
        lambda rows, evidence_database_sha256: stored_result,
    )

    with pytest.raises(RuntimeError, match="manifest fields"):
        runner_module_antigravity.verify_official_antigravity(
            evidence_path=evidence_path,
            result_path=result_path,
            manifest_path=manifest_path,
            dashboard_path=dashboard_path,
            report_path=report_path,
        )
    assert not dashboard_path.exists()
    assert not report_path.exists()


def test_verify_rejects_result_byte_or_manifest_sha_mismatch_antigravity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Either deterministic byte drift or a stale manifest must make verify fail."""
    result_path = tmp_path / "results_antigravity.json"
    manifest_path = tmp_path / "run_manifest_antigravity.json"
    dashboard_path = tmp_path / "dashboard_antigravity.html"
    report_path = tmp_path / "implementation_report_antigravity.html"
    evidence_path = tmp_path / "evidence_antigravity.sqlite3"
    evidence_path.write_bytes(b"controlled-test-evidence")
    stored_result = _qualification_result_antigravity()
    write_json_atomic_antigravity(result_path, stored_result)
    write_json_atomic_antigravity(
        manifest_path,
        {
            **_complete_cpu_manifest_antigravity(result_path.read_bytes()),
            "result_sha256": "0" * 64,
        },
    )
    recomputed = _qualification_result_antigravity()
    recomputed["dataset_count"] = 1116
    monkeypatch.setattr(
        runner_module_antigravity,
        "validate_evidence_database_antigravity",
        lambda path: {"datasets": 1117},
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "load_evidence_database_antigravity",
        lambda path: (),
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "build_official_result_antigravity",
        lambda rows, evidence_database_sha256: recomputed,
    )

    with pytest.raises(RuntimeError, match="result bytes"):
        runner_module_antigravity.verify_official_antigravity(
            evidence_path=evidence_path,
            result_path=result_path,
            manifest_path=manifest_path,
            dashboard_path=dashboard_path,
            report_path=report_path,
        )
    assert not dashboard_path.exists()
    assert not report_path.exists()

    monkeypatch.setattr(
        runner_module_antigravity,
        "build_official_result_antigravity",
        lambda rows, evidence_database_sha256: stored_result,
    )
    with pytest.raises(RuntimeError, match="manifest result SHA"):
        runner_module_antigravity.verify_official_antigravity(
            evidence_path=evidence_path,
            result_path=result_path,
            manifest_path=manifest_path,
            dashboard_path=dashboard_path,
            report_path=report_path,
        )
    assert not dashboard_path.exists()
    assert not report_path.exists()


def test_verify_success_writes_final_dashboard_and_easy_korean_report_antigravity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a byte- and SHA-verified result may publish final checked artifacts."""
    evidence_path = tmp_path / "evidence_antigravity.sqlite3"
    evidence_path.write_bytes(b"controlled-test-evidence")
    result_path = tmp_path / "results_antigravity.json"
    manifest_path = tmp_path / "run_manifest_antigravity.json"
    dashboard_path = tmp_path / "dashboard_antigravity.html"
    report_path = tmp_path / "implementation_report_antigravity.html"
    stored_result = _qualification_result_antigravity()
    stored_result["evidence_database_sha256"] = sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    write_json_atomic_antigravity(result_path, stored_result)
    write_json_atomic_antigravity(
        manifest_path,
        _complete_cpu_manifest_antigravity(result_path.read_bytes()),
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "validate_evidence_database_antigravity",
        lambda path: {"datasets": 1117},
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "load_evidence_database_antigravity",
        lambda path: (),
    )
    monkeypatch.setattr(
        runner_module_antigravity,
        "build_official_result_antigravity",
        lambda rows, evidence_database_sha256: stored_result,
    )

    verification = runner_module_antigravity.verify_official_antigravity(
        evidence_path=evidence_path,
        result_path=result_path,
        manifest_path=manifest_path,
        dashboard_path=dashboard_path,
        report_path=report_path,
    )

    dashboard = dashboard_path.read_text(encoding="utf-8")
    report = report_path.read_text(encoding="utf-8")
    assert verification["match"] is True
    assert '<input type="checkbox" checked disabled> Task 6 — 공식 1,117개 실행 승인' in dashboard
    assert '<input type="checkbox" checked disabled> Task 6 — 공식 결과 검증' in dashboard
    assert "공식 반영 준비" in dashboard
    assert "ZF 합계" in dashboard
    assert "평균 AUC" in dashboard
    assert "공식 반영 준비" in report
    assert "FN" in report and "ZF" in report and "AUC" in report
    assert "공식 라우터는 바꾸지 않았습니다" in report
    parser = HTMLParser()
    parser.feed(report)


def test_easy_korean_report_preserves_failed_qualification_antigravity() -> None:
    """A failed gate must be explained as preserved failure, never as readiness."""
    report = runner_module_antigravity.render_implementation_report_antigravity(
        _qualification_result_antigravity(passed=False)
    )

    assert report.startswith("<!doctype html>")
    assert "개선 실패 보존" in report
    assert "공식 반영 준비" not in report
    assert "공식 라우터는 바꾸지 않았습니다" in report
