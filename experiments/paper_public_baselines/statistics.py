"""Deterministic shared-draw paired bootstrap statistics."""

from __future__ import annotations

import hashlib

import numpy as np


def bootstrap_draws(
    population_size: int,
    *,
    seed: int = 20260812,
    samples: int = 2_000,
) -> tuple[np.ndarray, str]:
    """Create shared dataset-level bootstrap indices and their byte digest."""
    if type(population_size) is not int or population_size < 1:
        raise ValueError("bootstrap population size must be positive")
    if type(samples) is not int or samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(
        0, population_size, size=(samples, population_size), dtype=np.int64
    )
    draws.setflags(write=False)
    digest = hashlib.sha256(draws.tobytes(order="C")).hexdigest()
    return draws, digest


def paired_bootstrap(
    baseline: np.ndarray,
    proposal: np.ndarray,
    *,
    draws: np.ndarray,
    draw_sha256: str,
    seed: int = 20260812,
) -> dict[str, object]:
    """Estimate baseline-minus-proposal mean difference with shared draws."""
    baseline_values = np.asarray(baseline, dtype=np.float64)
    proposal_values = np.asarray(proposal, dtype=np.float64)
    draw_values = np.asarray(draws)
    if (
        baseline_values.ndim != 1
        or proposal_values.ndim != 1
        or baseline_values.shape != proposal_values.shape
        or not len(baseline_values)
    ):
        raise ValueError("paired bootstrap values must be non-empty and aligned")
    if not np.isfinite(baseline_values).all() or not np.isfinite(proposal_values).all():
        raise ValueError("paired bootstrap values must be finite")
    if (
        draw_values.ndim != 2
        or draw_values.shape[1] != len(baseline_values)
        or not np.issubdtype(draw_values.dtype, np.integer)
        or draw_values.min() < 0
        or draw_values.max() >= len(baseline_values)
    ):
        raise ValueError("bootstrap draws are not aligned with paired values")
    actual_sha = hashlib.sha256(
        np.asarray(draw_values, dtype=np.int64).tobytes(order="C")
    ).hexdigest()
    if draw_sha256 != actual_sha:
        raise ValueError("bootstrap draw SHA256 mismatch")
    differences = baseline_values - proposal_values
    estimates = differences[draw_values].mean(axis=1)
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return {
        "point_estimate": float(differences.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "draw_count": int(len(draw_values)),
        "seed": int(seed),
        "draw_sha256": actual_sha,
    }


__all__ = ["bootstrap_draws", "paired_bootstrap"]
