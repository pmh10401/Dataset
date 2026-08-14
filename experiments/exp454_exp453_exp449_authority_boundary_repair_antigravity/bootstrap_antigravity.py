"""Fixed Task-6 paired bootstrap, independent of sources and evaluation."""
from __future__ import annotations

from hashlib import sha256
import numpy as np


BOOTSTRAP_SEED_ANTIGRAVITY = 44920260812
BOOTSTRAP_SAMPLES_ANTIGRAVITY = 20_000
BOOTSTRAP_POPULATION_ANTIGRAVITY = 48
BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY = "264c780d7e5949994f1de3053158ea20a7cc3d0ae858f29770d9849b18ad4d6a"


def bootstrap_draws_antigravity() -> np.ndarray:
    """Return the one immutable PCG64 draw matrix bound into Task 6 artifacts."""
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED_ANTIGRAVITY)).integers(
        0, BOOTSTRAP_POPULATION_ANTIGRAVITY,
        size=(BOOTSTRAP_SAMPLES_ANTIGRAVITY, BOOTSTRAP_POPULATION_ANTIGRAVITY),
        dtype=np.int64, endpoint=False,
    )
    if sha256(draws.tobytes(order="C")).hexdigest() != BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY:
        raise ValueError("canonical bootstrap draw bytes differ from Task-6 authority")
    draws.setflags(write=False)
    return draws


def paired_delta_antigravity(baseline: object, candidate: object) -> dict[str, object]:
    """Compute descriptive FN/FP/F1 paired uncertainty; it never selects a candidate."""
    base = np.asarray(baseline, dtype=np.int64)
    compared = np.asarray(candidate, dtype=np.int64)
    if base.shape != (48, 4) or compared.shape != (48, 4):
        raise ValueError("paired bootstrap rows differ from the fixed 48-dataset population")
    if np.any(base < 0) or np.any(compared < 0):
        raise ValueError("paired bootstrap counts must be non-negative")
    draws = bootstrap_draws_antigravity()
    sampled_base = base[draws].sum(axis=1, dtype=np.int64)
    sampled_compared = compared[draws].sum(axis=1, dtype=np.int64)
    def f1(counts: np.ndarray) -> np.ndarray:
        denominator = 2 * counts[:, 0] + counts[:, 1] + counts[:, 2]
        return np.divide(2.0 * counts[:, 0], denominator, out=np.zeros(len(counts), dtype=np.float64), where=denominator != 0)
    fn_delta = sampled_compared[:, 2] - sampled_base[:, 2]
    fp_delta = sampled_compared[:, 1] - sampled_base[:, 1]
    f1_delta = f1(sampled_compared) - f1(sampled_base)
    linear = lambda values: [float(item) for item in np.percentile(values, (2.5, 97.5), method="linear")]
    total_base, total_compared = base.sum(axis=0, dtype=np.int64), compared.sum(axis=0, dtype=np.int64)
    total_f1 = lambda counts: 0.0 if (denominator := 2 * int(counts[0]) + int(counts[1]) + int(counts[2])) == 0 else (2.0 * int(counts[0])) / denominator
    return {
        "draw_sha256": BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY, "seed": BOOTSTRAP_SEED_ANTIGRAVITY,
        "samples": BOOTSTRAP_SAMPLES_ANTIGRAVITY, "population": BOOTSTRAP_POPULATION_ANTIGRAVITY,
        "dtype": "int64", "endpoint": False, "percentile_method": "linear",
        "fn_delta": int(total_compared[2] - total_base[2]), "fp_delta": int(total_compared[1] - total_base[1]),
        "f1_delta": float(total_f1(total_compared) - total_f1(total_base)),
        "fn_interval": linear(fn_delta), "fp_interval": linear(fp_delta), "f1_interval": linear(f1_delta),
    }


__all__ = ["BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY", "bootstrap_draws_antigravity", "paired_delta_antigravity"]
