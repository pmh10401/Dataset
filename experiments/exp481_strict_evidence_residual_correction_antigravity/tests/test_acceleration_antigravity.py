"""Task 4 selection tests for Numba and MLX acceleration gates."""

from __future__ import annotations

from experiments.exp481_strict_evidence_residual_correction_antigravity import acceleration_antigravity


def _parity_antigravity(**overrides):
    values = {
        "candidate_index_parity": True,
        "statistics_parity": True,
        "decision_parity": True,
        "numerical_agreement": True,
        "finite_outputs": True,
        "max_abs_error": 0.0,
    }
    values.update(overrides)
    return values


def test_numba_parity_failure_keeps_numpy_reference_antigravity(monkeypatch) -> None:
    monkeypatch.setattr(
        acceleration_antigravity,
        "_assess_parity_antigravity",
        lambda: _parity_antigravity(decision_parity=False),
    )
    monkeypatch.setattr(
        acceleration_antigravity,
        "_benchmark_complete_stage_antigravity",
        lambda repeats=1: {
            "numpy_seconds": 2.0,
            "numba_warmup_seconds": 0.1,
            "numba_execution_seconds": 0.5,
            "numba_total_seconds": 0.6,
            "numba_speedup_including_warmup": 3.0,
            "exact_frozen_stage_benchmarked": True,
            "numba_cold_jit_verified": True,
        },
    )
    result = acceleration_antigravity.assess_acceleration_antigravity()
    assert result["selected_backend"] == "numpy_reference"
    assert "parity" in result["fallback_reason"]


def test_numba_needs_full_stage_speed_benefit_antigravity(monkeypatch) -> None:
    monkeypatch.setattr(acceleration_antigravity, "_assess_parity_antigravity", _parity_antigravity)
    monkeypatch.setattr(
        acceleration_antigravity,
        "_benchmark_complete_stage_antigravity",
        lambda repeats=1: {
            "numpy_seconds": 1.0,
            "numba_warmup_seconds": 0.8,
            "numba_execution_seconds": 0.4,
            "numba_total_seconds": 1.2,
            "numba_speedup_including_warmup": 1.0 / 1.2,
            "exact_frozen_stage_benchmarked": True,
            "numba_cold_jit_verified": True,
        },
    )
    result = acceleration_antigravity.assess_acceleration_antigravity()
    assert result["selected_backend"] == "numpy_reference"
    assert result["fallback_reason"] == "numba_no_full_stage_speed_benefit"


def test_numba_selected_only_when_all_gates_pass_antigravity(monkeypatch) -> None:
    monkeypatch.setattr(acceleration_antigravity, "_assess_parity_antigravity", _parity_antigravity)
    monkeypatch.setattr(
        acceleration_antigravity,
        "_benchmark_complete_stage_antigravity",
        lambda repeats=1: {
            "numpy_seconds": 2.0,
            "numba_warmup_seconds": 0.1,
            "numba_execution_seconds": 0.5,
            "numba_total_seconds": 0.6,
            "numba_speedup_including_warmup": 2.0 / 0.6,
            "exact_frozen_stage_benchmarked": True,
            "numba_cold_jit_verified": True,
        },
    )
    result = acceleration_antigravity.assess_acceleration_antigravity()
    assert result["selected_backend"] == "numba_cpu"
    assert result["fallback_reason"] in (
        "mlx_unavailable_in_frozen_runtime",
        "mlx_exact_stage_not_implemented",
    )
    assert result["mlx_eligible"] is False
    assert result["mlx_candidate_index_parity"] is None
    assert "mlx" in result["mlx_fallback_reason"]
    assert "mlx_metal_version" in result
    assert result["mlx_device"]


def test_complete_stage_benchmark_uses_nested_evaluator_antigravity(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        acceleration_antigravity,
        "_nested_stage_antigravity",
        lambda backend: calls.append(backend) or {},
    )
    result = acceleration_antigravity._benchmark_complete_stage_antigravity(repeats=2)
    assert calls == [
        "numpy_reference",
        "numpy_reference",
        "numba_cpu",
        "numba_cpu",
        "numba_cpu",
    ]
    assert result["numba_total_seconds"] == result["numba_warmup_seconds"]
    assert result["exact_frozen_stage_benchmarked"] is False
    assert result["numba_cold_jit_verified"] is False


def test_assessment_benchmarks_before_parity_antigravity(monkeypatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        acceleration_antigravity,
        "_benchmark_complete_stage_antigravity",
        lambda repeats=1: order.append("benchmark") or {
            "numpy_seconds": 1.0,
            "numba_warmup_seconds": 2.0,
            "numba_execution_seconds": 1.0,
            "numba_total_seconds": 2.0,
            "numba_speedup_including_warmup": 0.5,
            "exact_frozen_stage_benchmarked": True,
            "numba_cold_jit_verified": True,
        },
    )
    monkeypatch.setattr(
        acceleration_antigravity,
        "_assess_parity_antigravity",
        lambda: order.append("parity") or _parity_antigravity(),
    )
    acceleration_antigravity.assess_acceleration_antigravity()
    assert order == ["benchmark", "parity"]


def test_diagnostic_workload_can_never_select_numba_antigravity(monkeypatch) -> None:
    monkeypatch.setattr(
        acceleration_antigravity,
        "_assess_parity_antigravity",
        _parity_antigravity,
    )
    monkeypatch.setattr(
        acceleration_antigravity,
        "_benchmark_complete_stage_antigravity",
        lambda repeats=1: {
            "numpy_seconds": 10.0,
            "numba_warmup_seconds": 0.1,
            "numba_execution_seconds": 0.1,
            "numba_total_seconds": 0.1,
            "numba_speedup_including_warmup": 100.0,
            "exact_frozen_stage_benchmarked": False,
            "numba_cold_jit_verified": False,
        },
    )
    result = acceleration_antigravity.assess_acceleration_antigravity()
    assert result["selected_backend"] == "numpy_reference"
    assert result["fallback_reason"] == "numba_exact_frozen_stage_not_benchmarked"
