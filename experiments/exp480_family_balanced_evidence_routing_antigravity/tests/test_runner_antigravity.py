"""Task 5 tests for Exp480 execution and dashboard boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.exp480_family_balanced_evidence_routing_antigravity.run_antigravity import (
    render_dashboard_antigravity,
    run_smoke_antigravity,
    run_validate_antigravity,
    write_json_atomic_antigravity,
)


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
