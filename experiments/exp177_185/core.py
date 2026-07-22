from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping

import numpy as np


SEED = 20260718
ALPHAS = (0.005, 0.01, 0.02, 0.05)
RISK_TARGETS = (0.005, 0.01, 0.02)
SOURCE_NAMES = (
    "temporal_random_kernel",
    "spectral_image",
    "texture_image",
    "interval_statistical",
)
SOURCE_CACHE_KEYS = {
    "temporal_random_kernel": "rocket_exp40",
    "spectral_image": "exp55_best",
    "texture_image": "exp56_best",
    "interval_statistical": "exp84",
}
FORBIDDEN_PREDICTION_FIELDS = {
    "y_test", "labels", "test_labels", "tp", "fp", "fn", "f1",
    "hard_tp", "hard_fp", "hard_fn", "hard_f1", "combined_f1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empirical_upper_tail_pvalues(reference: Iterable[float], query: Iterable[float]) -> np.ndarray:
    ref = np.asarray(list(reference), dtype=float)
    values = np.asarray(list(query), dtype=float)
    ref = ref[np.isfinite(ref)]
    if not len(ref):
        return np.ones(len(values), dtype=float)
    ordered = np.sort(ref)
    counts = len(ordered) - np.searchsorted(ordered, values, side="left")
    result = (1.0 + counts.astype(float)) / (len(ordered) + 1.0)
    result[~np.isfinite(values)] = 1.0
    return result


def leave_one_out_pvalues(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    result = np.ones(len(values), dtype=float)
    for index, value in enumerate(values):
        result[index] = empirical_upper_tail_pvalues(
            np.delete(values, index), np.asarray([value])
        )[0]
    return result


def agreement_indices(
    pvalues_by_source: Mapping[str, np.ndarray], alpha: float, required: int
) -> set[int]:
    if not pvalues_by_source:
        return set()
    arrays = [np.asarray(values, dtype=float) for values in pvalues_by_source.values()]
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1:
        raise ValueError("source p-value lengths differ")
    support = np.sum(np.vstack([values <= float(alpha) for values in arrays]), axis=0)
    return set(np.flatnonzero(support >= int(required)).astype(int).tolist())


def exclusive_lanes(
    hard: Iterable[int], standard: Iterable[int], priority: Iterable[int]
) -> tuple[set[int], set[int], set[int]]:
    hard_set = {int(value) for value in hard}
    standard_set = {int(value) for value in standard} - hard_set
    priority_set = {int(value) for value in priority} - hard_set - standard_set
    return hard_set, standard_set, priority_set


def validate_prediction_payload(payload: Mapping[str, object]) -> None:
    forbidden = FORBIDDEN_PREDICTION_FIELDS & {str(key).lower() for key in payload}
    if forbidden:
        raise ValueError(f"evaluation fields in frozen prediction: {sorted(forbidden)}")
    if payload.get("contains_test_labels") is not False:
        raise ValueError("prediction must declare contains_test_labels=false")
    if payload.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction must declare contains_evaluation_metrics=false")
    n_test = int(payload["n_test"])
    indices = [int(value) for value in payload.get("candidate_indices", [])]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate candidate indices")
    if indices and (min(indices) < 0 or max(indices) >= n_test):
        raise ValueError("candidate index outside TEST range")


def wilson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return 1.0
    p = successes / trials
    z = NormalDist().inv_cdf(confidence)
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return float((center + radius) / denominator)


def select_alpha_from_train_risk(
    false_alerts_by_alpha: Mapping[float, int], trials: int, risk_target: float
) -> float | None:
    eligible = [
        alpha for alpha in ALPHAS
        if wilson_upper(int(false_alerts_by_alpha.get(alpha, trials)), trials) <= risk_target
    ]
    return max(eligible) if eligible else None


def _sanitize_matrix(x: np.ndarray) -> np.ndarray:
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("feature input must be a matrix")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def _multiscale_stat_features(x: np.ndarray) -> np.ndarray:
    matrix = _sanitize_matrix(x)
    rows = []
    for series in matrix:
        centered = series - np.mean(series)
        scale = float(np.std(centered)) or 1.0
        normalized = centered / scale
        diff1 = np.diff(normalized)
        diff2 = np.diff(diff1)
        values = [
            float(np.mean(series)), float(np.std(series)),
            float(np.quantile(series, 0.05)), float(np.quantile(series, 0.25)),
            float(np.median(series)), float(np.quantile(series, 0.75)),
            float(np.quantile(series, 0.95)),
            float(np.mean(np.abs(diff1))) if len(diff1) else 0.0,
            float(np.std(diff1)) if len(diff1) else 0.0,
            float(np.std(diff2)) if len(diff2) else 0.0,
            float(np.mean(np.signbit(normalized[:-1]) != np.signbit(normalized[1:])))
            if len(normalized) > 1 else 0.0,
        ]
        for lag in (1, 2, 4, 8, 16, 32):
            if len(normalized) <= lag:
                values.append(0.0)
            else:
                values.append(float(np.mean(normalized[:-lag] * normalized[lag:])))
        power = np.abs(np.fft.rfft(normalized)) ** 2
        power_sum = float(np.sum(power)) or 1.0
        probabilities = power / power_sum
        frequencies = np.linspace(0.0, 1.0, len(power))
        values.extend([
            float(np.sum(probabilities * frequencies)),
            float(-np.sum(probabilities * np.log(probabilities + 1e-12))),
        ])
        for lower, upper in ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01)):
            mask = (frequencies >= lower) & (frequencies < upper)
            values.append(float(np.sum(probabilities[mask])))
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def canonical_features(x: np.ndarray) -> np.ndarray:
    from pycatch22 import catch22_all

    matrix = _sanitize_matrix(x)
    catch_rows = []
    for series in matrix:
        result = catch22_all(series.tolist(), catch24=True)
        catch_rows.append(np.asarray(result["values"], dtype=float))
    features = np.hstack([np.vstack(catch_rows), _multiscale_stat_features(matrix)])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def wavelet_features(x: np.ndarray) -> np.ndarray:
    import pywt

    matrix = _sanitize_matrix(x)
    rows = []
    for series in matrix:
        centered = series - np.mean(series)
        scale = float(np.std(centered)) or 1.0
        normalized = centered / scale
        max_level = max(1, min(4, pywt.dwt_max_level(len(normalized), pywt.Wavelet("db4").dec_len)))
        coeffs = pywt.wavedec(normalized, "db4", level=max_level)
        values = []
        total_energy = sum(float(np.sum(coeff * coeff)) for coeff in coeffs) or 1.0
        for coeff in coeffs:
            energy = float(np.sum(coeff * coeff)) / total_energy
            values.extend([
                energy,
                float(np.mean(np.abs(coeff))),
                float(np.std(coeff)),
                float(np.max(np.abs(coeff))) if len(coeff) else 0.0,
            ])
        while len(values) < 20:
            values.append(0.0)
        rows.append(values[:20])
    return np.nan_to_num(np.asarray(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _fold_plan(n: int, seed: int, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    if n < 5:
        raise ValueError("crossfit requires at least five TRAIN-normal instances")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    chunks = [chunk for chunk in np.array_split(order, min(folds, n)) if len(chunk)]
    result = []
    all_indices = np.arange(n)
    for query in chunks:
        fit = np.setdiff1d(all_indices, query, assume_unique=True)
        result.append((fit, np.sort(query)))
    return result


def _robust_scale_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(x, axis=0)
    scale = 1.4826 * np.median(np.abs(x - center), axis=0)
    std = np.std(x, axis=0)
    scale = np.where(scale > 1e-9, scale, np.where(std > 1e-9, std, 1.0))
    return center, scale


def _score_fit_query(fit: np.ndarray, query: np.ndarray, backend: str) -> np.ndarray:
    center, scale = _robust_scale_fit(fit)
    fit_z = (fit - center) / scale
    query_z = (query - center) / scale
    if backend == "robust_diagonal":
        return np.sqrt(np.mean(query_z * query_z, axis=1))
    if backend != "knn_local_gap":
        raise ValueError(f"unknown score backend: {backend}")
    k = max(1, min(5, len(fit_z)))
    distances = np.sqrt(np.mean((query_z[:, None, :] - fit_z[None, :, :]) ** 2, axis=2))
    query_distance = np.mean(np.partition(distances, k - 1, axis=1)[:, :k], axis=1)
    if len(fit_z) <= 1:
        return query_distance
    fit_distances = np.sqrt(np.mean((fit_z[:, None, :] - fit_z[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(fit_distances, np.inf)
    fit_k = max(1, min(k, len(fit_z) - 1))
    local = np.mean(np.partition(fit_distances, fit_k - 1, axis=1)[:, :fit_k], axis=1)
    baseline = float(np.median(local)) or 1.0
    return query_distance / baseline


def crossfit_scores(
    train_features: np.ndarray,
    test_features: np.ndarray,
    *,
    backend: str,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_features, dtype=float)
    test = np.asarray(test_features, dtype=float)
    oof = np.full(len(train), np.nan, dtype=float)
    for fit_indices, query_indices in _fold_plan(len(train), seed):
        oof[query_indices] = _score_fit_query(train[fit_indices], train[query_indices], backend)
    test_scores = _score_fit_query(train, test, backend)
    if not np.all(np.isfinite(oof)) or not np.all(np.isfinite(test_scores)):
        raise ValueError("non-finite crossfit score")
    return oof, test_scores


def choose_compute_device(force: str = "auto") -> dict[str, object]:
    if force == "cpu":
        return {"device": "cpu", "reason": "forced", "mps_available": False}
    try:
        import torch
    except ImportError:
        return {"device": "cpu", "reason": "torch unavailable", "mps_available": False}
    available = bool(torch.backends.mps.is_available())
    if force == "mps":
        return {
            "device": "mps" if available else "cpu",
            "reason": "forced" if available else "MPS unavailable",
            "mps_available": available,
        }
    if not available:
        return {"device": "cpu", "reason": "MPS unavailable", "mps_available": False}
    rng = np.random.default_rng(SEED)
    sample = rng.normal(size=(256, 1024)).astype(np.float32)
    timings = {}
    for device in ("cpu", "mps"):
        try:
            tensor = torch.as_tensor(sample, device=device)
            if device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            for _ in range(3):
                spectrum = torch.fft.rfft(tensor, dim=1)
                _ = torch.sum(torch.abs(spectrum) ** 2, dim=1)
            if device == "mps":
                torch.mps.synchronize()
            timings[device] = time.perf_counter() - start
        except Exception as exc:
            return {
                "device": "cpu", "reason": f"MPS benchmark failed: {exc}",
                "mps_available": available,
            }
    use_mps = timings["mps"] < timings["cpu"] * 0.9
    return {
        "device": "mps" if use_mps else "cpu",
        "reason": "fixed synthetic FFT benchmark",
        "mps_available": available,
        "cpu_seconds": timings["cpu"],
        "mps_seconds": timings["mps"],
        "minimum_speedup": 1.1,
    }
