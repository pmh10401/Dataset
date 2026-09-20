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

    # Very long sequences with very low diff_std_ratio (smooth, slowly-varying)
    # benefit from minirocket's fine-grained kernels. EOGHorizontalSignal case:
    # hydra alone fails (AUC=0.0) while minirocket achieves 1.0. The very low
    # zero_crossing_rate (< 0.005) is a signature of such smooth drifts.
    # Requires entropy > 0.15 to exclude strongly periodic signals like
    # StarLightCurves (entropy=0.065) which still need Hydra.
    if (
        width >= 1000
        and features.get("diff_std_ratio", 0.5) < 0.05
        and features.get("zero_crossing_rate", 0.05) < 0.005
        and entropy > 0.15
    ):
        return "minirocket_local_gap"

    # Very short sequences (width < 50) with near-zero kurtosis are hard for
    # multirocket's dilation-based kernels; minirocket handles them better.
    # Chinatown case: multirocket AUC=0.45 vs minirocket=0.99.
    if width < 50 and abs(features.get("kurtosis", 0.0)) < 1.5:
        return "minirocket_local_gap"

    # Short sequences with near-zero skew AND near-zero kurtosis (balanced,
    # non-spike, non-tailed) favour minirocket over multirocket on AUC.
    # ECG200 case: multirocket AUC=0.76 vs minirocket/hydra=0.92.
    if width < 200 and abs(skewness) < 0.20 and abs(features.get("kurtosis", 0.0)) < 0.50:
        return "minirocket_local_gap"

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
