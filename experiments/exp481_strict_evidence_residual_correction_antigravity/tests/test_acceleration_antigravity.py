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
        },
    )
    result = acceleration_antigravity.assess_acceleration_antigravity()
    assert result["selected_backend"] == "numba_cpu"
    assert result["fallback_reason"] == "mlx_not_exact_for_sequential_family_selection"
    assert result["mlx_eligible"] is False
    assert result["mlx_fallback_reason"] == "sequential_stable_family_dedup_not_exactly_expressible"
