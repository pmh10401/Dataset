"""MLX eligibility and backend decision for Exp479."""

from __future__ import annotations

from importlib import metadata, util


def _version_antigravity(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def assess_mlx_acceleration_antigravity() -> dict[str, object]:
    """Record the mandatory MLX gate without changing the approved Numba path."""
    mlx_available = util.find_spec("mlx") is not None
    versions = {
        "numpy": _version_antigravity("numpy"),
        "numba": _version_antigravity("numba"),
        "mlx": _version_antigravity("mlx"),
        "mlx_metal": _version_antigravity("mlx-metal"),
    }
    if mlx_available:
        return {
            "schema": "exp479_mlx_assessment_v1",
            "stage": "ragged_linear_resampling",
            "mlx_available": True,
            "eligible": False,
            "parity": "not_run",
            "speedup": None,
            "selected_backend": "numba_cpu",
            "fallback_reason": "mlx_parity_and_speed_smoke_not_completed",
            "versions": versions,
            "device": None,
        }
    return {
        "schema": "exp479_mlx_assessment_v1",
        "stage": "ragged_linear_resampling",
        "mlx_available": False,
        "eligible": False,
        "parity": "not_run",
        "speedup": None,
        "selected_backend": "numba_cpu",
        "fallback_reason": "mlx_not_installed_in_fixed_interpreter",
        "versions": versions,
        "device": None,
    }


__all__ = ["assess_mlx_acceleration_antigravity"]
