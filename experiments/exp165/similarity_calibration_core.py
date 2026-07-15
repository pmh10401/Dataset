from __future__ import annotations

import numpy as np


SOURCES = ("spectral", "shapelet", "interval")


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        raise ValueError("at least two finite scores are required")
    location = float(np.median(finite))
    mad = float(np.median(np.abs(finite - location))) * 1.4826
    if mad > 1e-12:
        return location, mad
    q25, q75 = np.quantile(finite, (0.25, 0.75))
    iqr_scale = float(q75 - q25) / 1.349
    if iqr_scale > 1e-12:
        return location, iqr_scale
    std = float(np.std(finite))
    return location, std if std > 1e-12 else 1.0


def leave_one_out_location_scale(values: np.ndarray, query_index: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not 0 <= query_index < values.size:
        raise IndexError(query_index)
    return robust_location_scale(np.delete(values, query_index))


def robust_normalize(values: np.ndarray) -> np.ndarray:
    location, scale = robust_location_scale(values)
    return (np.asarray(values, dtype=np.float64) - location) / scale


def build_descriptor(source_scores: dict[str, np.ndarray]) -> np.ndarray:
    descriptor = []
    for source in SOURCES:
        normalized = robust_normalize(source_scores[source])
        finite = normalized[np.isfinite(normalized)]
        if finite.size < 2:
            raise ValueError(f"insufficient descriptor scores for {source}")
        descriptor.extend(np.quantile(finite, (0.10, 0.25, 0.50, 0.75, 0.90)).tolist())
        descriptor.extend((float(np.mean(finite)), float(np.std(finite))))
    value = np.asarray(descriptor, dtype=np.float64)
    if value.shape != (21,) or not np.isfinite(value).all():
        raise ValueError("descriptor must contain 21 finite values")
    return value


def descriptor_scaler(donor_descriptors: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not donor_descriptors:
        raise ValueError("donor descriptors are empty")
    matrix = np.vstack([donor_descriptors[name] for name in sorted(donor_descriptors)])
    center = np.median(matrix, axis=0)
    q25, q75 = np.quantile(matrix, (0.25, 0.75), axis=0)
    scale = q75 - q25
    scale = np.where(scale > 1e-12, scale, 1.0)
    return center.astype(np.float64), scale.astype(np.float64)


def scale_descriptor(descriptor: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(descriptor, dtype=np.float64) - center) / scale


def select_donors(
    target_descriptor: np.ndarray,
    donor_descriptors: dict[str, np.ndarray],
    donor_counts: dict[str, dict[str, int]],
    minimum_reference: int,
) -> list[str]:
    distances = []
    for name, descriptor in donor_descriptors.items():
        distance = float(np.linalg.norm(np.asarray(descriptor) - np.asarray(target_descriptor)))
        distances.append((distance, name))
    distances.sort(key=lambda item: (item[0], item[1]))

    selected = []
    counts = {source: 0 for source in SOURCES}
    cutoff = None
    for distance, name in distances:
        if cutoff is not None and not np.isclose(distance, cutoff, rtol=0.0, atol=1e-12):
            break
        selected.append(name)
        for source in SOURCES:
            counts[source] += int(donor_counts[name][source])
        if all(counts[source] >= minimum_reference for source in SOURCES):
            cutoff = distance
    if not selected or not all(counts[source] >= minimum_reference for source in SOURCES):
        raise ValueError("minimum pooled reference is unavailable")
    return sorted(selected)


def higher_threshold(reference: np.ndarray, alpha: float) -> float:
    finite = np.asarray(reference, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("reference has no finite values")
    return float(np.quantile(finite, 1.0 - alpha, method="higher"))


def similarity_pooled_flags(
    query_scores: dict[str, float],
    references: dict[str, np.ndarray],
    alphas: tuple[float, ...],
) -> tuple[dict[float, dict[str, float]], dict[float, dict[str, bool]]]:
    thresholds: dict[float, dict[str, float]] = {}
    flags: dict[float, dict[str, bool]] = {}
    for alpha in alphas:
        thresholds[alpha] = {}
        flags[alpha] = {}
        for source in SOURCES:
            threshold = higher_threshold(references[source], alpha)
            thresholds[alpha][source] = threshold
            value = float(query_scores[source])
            flags[alpha][source] = bool(np.isfinite(value) and value > threshold)
    return thresholds, flags


def agreement_flags(source_flags: dict[str, bool], required_sources: int) -> bool:
    if required_sources not in (2, 3):
        raise ValueError("required_sources must be 2 or 3")
    return sum(bool(source_flags[source]) for source in SOURCES) >= required_sources
