"""Draft label-free routing rules derived from Exp471-473 evidence."""

from __future__ import annotations

from typing import Mapping


def route_dataset_antigravity(features: Mapping[str, float]) -> str:
    """Return the recommended method for a dataset given its features.

    Rules are ordered; first match wins. All thresholds are label-free.
    """
    width = features.get("width", 0)
    entropy = features.get("spectral_entropy", 0.5)
    skewness = features.get("skewness", 0.0)

    # Very long sequences with strong periodicity → hydra_local_gap.
    if width >= 400 and 0.35 <= entropy <= 0.65:
        return "hydra_local_gap"

    # Smooth, slowly-varying short signals (very low diff_std_ratio) benefit
    # from minirocket's fine-grained kernels. Distinguishes GunPoint-like
    # smooth postures from ECG-like oscillations at similar widths.
    # Uses only diff_std_ratio (GunPoint=0.067 vs TwoLeadECG=0.211).
    if width < 200 and features.get("diff_std_ratio", 0.5) < 0.10:
        return "minirocket_local_gap"

    # Short sequences with heavy tails AND low-to-moderate spectral entropy
    # are ECG-like; Hydra's multi-scale receptive fields help.
    if width < 200 and features.get("kurtosis", 0.0) > 1.5 and entropy < 0.70:
        return "hydra_local_gap"

    # Short sequences with moderately negative skew (left-tail heavy) are
    # ECG-like oscillatory signals; Hydra's multi-scale receptive fields help.
    # Distinguishes TwoLeadECG (skew ≈ -0.95) from GunPoint (+0.62) and Wafer (+0.49).
    # For mild negative skew (>-1.5) with low kurtosis, minirocket's fine-grain
    # kernels recover more zero-FP true positives; for strong skew (<-1.5)
    # Hydra's broader receptive fields are better (ECG5000 case).
    if width < 200 and -1.50 <= skewness <= -0.30:
        return "minirocket_local_gap"

    if width < 200 and skewness < -1.50:
        return "hydra_local_gap"

    # Extremely long sequences regardless of periodicity → hydra_local_gap.
    if width >= 800:
        return "hydra_local_gap"

    # Long sequences with strong periodic structure → multirocket_local_gap.
    if width >= 400 and entropy < 0.25:
        return "multirocket_local_gap"

    # Strongly skewed distributions → multirocket family handles amplitude better.
    if abs(skewness) >= 2.0:
        return "multirocket_local_gap"

    # High spectral entropy (near-random) with mild asymmetry → minirocket_local_gap.
    if entropy > 0.70:
        return "minirocket_local_gap"

    # Strong negative skew (left tail heavy) → hydra_local_gap.
    if skewness < -1.0:
        return "hydra_local_gap"

    # Default fallback: balanced mid-range signals.
    return "multirocket_local_gap"
