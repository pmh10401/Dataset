"""Task 5 tests for the Exp480 MLX selection gate."""

from __future__ import annotations

from experiments.exp480_family_balanced_evidence_routing_antigravity.acceleration_antigravity import (
    choose_backend_antigravity,
)


def test_mlx_requires_at_least_1_1x_end_to_end_speed_antigravity() -> None:
    """Selecting MLX at only 1.05x must fail the speed contract."""
    result = choose_backend_antigravity(
        finite=True,
        indices_equal=True,
        max_abs_error=1e-12,
        tolerance=1e-10,
        cpu_seconds=1.0,
        mlx_seconds=0.95,
    )
    assert result["selected_backend"] == "cpu_numpy"
    assert result["fallback_reason"] == "speedup_below_1_1x"


def test_mlx_requires_exact_candidate_indices_antigravity() -> None:
    """A faster backend with one changed neighbor must stay on CPU."""
    result = choose_backend_antigravity(
        finite=True,
        indices_equal=False,
        max_abs_error=0.0,
        tolerance=1e-10,
        cpu_seconds=2.0,
        mlx_seconds=1.0,
    )
    assert result["selected_backend"] == "cpu_numpy"
    assert result["fallback_reason"] == "candidate_indices_changed"


def test_mlx_is_selected_only_when_every_gate_passes_antigravity() -> None:
    """Breaking any one parity or timing input must change this decision."""
    result = choose_backend_antigravity(
        finite=True,
        indices_equal=True,
        max_abs_error=1e-12,
        tolerance=1e-10,
        cpu_seconds=2.0,
        mlx_seconds=1.0,
    )
    assert result == {
        "selected_backend": "mlx",
        "fallback_reason": None,
        "speedup": 2.0,
    }

