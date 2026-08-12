"""Exact NumPy Exp55/56 imaging transforms, PCA, and KNN scoring closure.

Provenance: run_model_hard_research_experiments.py SHA256
f9f06973009259590b4afd165dc275cff9b834a628ccb17e248d015d55680194 (resample_rows, rp_images,
spectrogram_images, glcm_image_features, pca_pair, fixed_width_features,
imaging_features, score_pair_for_config) plus
run_experiment_60_62_rocket_imaging_selector_variants.py SHA256
4ab38ef8a0ad152ccfc18c1297c40aca31ba79ac29348b9f5da029682507b2ed
(IMAGING_CONFIGS).
"""
from __future__ import annotations

import logging
import numpy as np
from sklearn.decomposition import PCA

from .knn_paper import scale_feature_pair_paper
from .preprocessing_paper import resample_series_paper


EXP55_CONFIG_PAPER = {"name": "train_global_minmax_clip_spectrogram_32_pca32_knn3", "kind": "imaging_knn", "image": "spectrogram", "series_scale": "train_global_minmax_clip", "size": 32, "pca": 32, "neighbors": 3}
EXP56_CONFIG_PAPER = {"name": "glcm_rp_32_pca32_knn3", "kind": "imaging_knn", "image": "rp", "series_scale": "per_series_z", "feature_extractor": "glcm", "size": 32, "pca": 32, "neighbors": 3}
RNG_SEED_PAPER = 20260707
PROVENANCE_PAPER = {"run_model_hard_research_experiments.py": ("resample_rows", "rp_images", "spectrogram_images", "glcm_image_features", "pca_pair", "fixed_width_features", "imaging_features", "score_pair_for_config"), "run_experiment_60_62_rocket_imaging_selector_variants.py": ("IMAGING_CONFIGS",)}


def _resample_rows_paper(matrix: np.ndarray, target_len: int) -> np.ndarray:
    return np.vstack([resample_series_paper(row, target_len) for row in np.asarray(matrix, dtype=np.float32)])


def _spectrogram_images_paper(matrix: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    images = np.empty((len(values), size, size), dtype=np.float32)
    for index, row in enumerate(values):
        nperseg = max(8, min(len(row), size * 2)); hop = max(1, nperseg // 2)
        starts = list(range(0, max(1, len(row) - nperseg + 1), hop)) or [0]
        window = np.hanning(nperseg).astype(np.float32); frames = []
        for start in starts:
            segment = row[start:start + nperseg]
            if len(segment) < nperseg: segment = np.pad(segment, (0, nperseg - len(segment)), mode="edge")
            frames.append(np.log1p(np.abs(np.fft.rfft(segment * window))).astype(np.float32))
        spectrum = np.vstack(frames).T
        row_axis, col_axis = np.linspace(0, spectrum.shape[0] - 1, size), np.linspace(0, spectrum.shape[1] - 1, size)
        images[index] = np.vstack([np.interp(col_axis, np.arange(spectrum.shape[1]), spectrum[int(round(item))]) for item in row_axis])
    images -= images.min(axis=(1, 2), keepdims=True); images /= images.max(axis=(1, 2), keepdims=True) + 1e-6
    return images.astype(np.float32)


def _rp_images_paper(matrix: np.ndarray, size: int) -> np.ndarray:
    values = _resample_rows_paper(matrix, size); images = np.empty((len(values), size, size), dtype=np.float32)
    for index, row in enumerate(values):
        distance = np.abs(row[:, None] - row[None, :])
        images[index] = np.exp(-distance / (np.percentile(distance, 75) + 1e-6))
    return images


def _glcm_features_paper(images: np.ndarray, levels: int = 8) -> np.ndarray:
    features, offsets = [], ((0, 1), (1, 0), (1, 1), (1, -1)); yy, xx = np.mgrid[0:levels, 0:levels].astype(np.float32)
    for image in np.asarray(images, dtype=np.float32):
        normalized = image - np.nanmin(image); normalized /= np.nanmax(normalized) + 1e-6
        quantized = np.clip((normalized * levels).astype(np.int32), 0, levels - 1); row_features = []
        for dy, dx in offsets:
            ay = slice(0, quantized.shape[0] - dy) if dy >= 0 else slice(-dy, quantized.shape[0]); by = slice(dy, quantized.shape[0]) if dy >= 0 else slice(0, quantized.shape[0] + dy)
            ax = slice(0, quantized.shape[1] - dx) if dx >= 0 else slice(-dx, quantized.shape[1]); bx = slice(dx, quantized.shape[1]) if dx >= 0 else slice(0, quantized.shape[1] + dx)
            matrix = np.zeros((levels, levels), dtype=np.float32); np.add.at(matrix, (quantized[ay, ax].ravel(), quantized[by, bx].ravel()), 1.0); matrix += matrix.T; matrix /= matrix.sum() + 1e-6
            contrast = np.sum((yy - xx) ** 2 * matrix); dissimilarity = np.sum(np.abs(yy - xx) * matrix); homogeneity = np.sum(matrix / (1.0 + np.abs(yy - xx))); asm = np.sum(matrix * matrix); entropy = -np.sum(matrix * np.log(matrix + 1e-6))
            mi, mj = np.sum(yy * matrix), np.sum(xx * matrix); si = np.sqrt(np.sum((yy - mi) ** 2 * matrix)) + 1e-6; sj = np.sqrt(np.sum((xx - mj) ** 2 * matrix)) + 1e-6
            row_features.extend((contrast, dissimilarity, homogeneity, asm, np.sqrt(asm), entropy, np.sum((yy - mi) * (xx - mj) * matrix) / (si * sj)))
        features.append(row_features)
    return np.asarray(features, dtype=np.float32)


def _fixed_width_paper(features: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32); values = values.reshape(-1, 1) if values.ndim == 1 else values
    if values.shape[1] >= width: return values[:, :width].astype(np.float32)
    return np.hstack((values, np.zeros((len(values), width - values.shape[1]), dtype=np.float32)))


def _pca_pair_paper(train: np.ndarray, test: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray]:
    train, test = np.asarray(train, dtype=np.float32), np.asarray(test, dtype=np.float32)
    center = train.mean(axis=0, keepdims=True); centered, test_centered = train - center, test - center
    count = max(1, min(int(components), centered.shape[0], centered.shape[1]))
    large_matrix = centered.shape[0] * centered.shape[1] >= 1_000_000
    if large_matrix and count < min(centered.shape):
        try:
            pca = PCA(n_components=count, svd_solver="randomized", random_state=RNG_SEED_PAPER)
            return _fixed_width_paper(pca.fit_transform(centered), components), _fixed_width_paper(pca.transform(test_centered), components)
        except Exception:
            logging.getLogger("model_hard_research").exception("Randomized PCA failed; falling back to SVD.")
    try: _, _, vt = np.linalg.svd(centered, full_matrices=False); basis = vt[:count].T
    except np.linalg.LinAlgError: basis = np.eye(centered.shape[1], count, dtype=np.float32)
    return _fixed_width_paper(centered @ basis, components), _fixed_width_paper(test_centered @ basis, components)


def _prepare_series_pair_paper(train: np.ndarray, test: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "per_series_z":
        def normalize(values: np.ndarray) -> np.ndarray:
            mean = values.mean(axis=1, keepdims=True); std = values.std(axis=1, keepdims=True)
            return (values - mean) / np.where(std == 0.0, 1.0, std)
        return normalize(train).astype(np.float32), normalize(test).astype(np.float32)
    if mode == "train_global_minmax_clip":
        low, high = np.min(train), np.max(train); scale = high - low if high - low > 1e-6 else 1.0
        return (np.clip((train - low) / scale, 0.0, 1.0) * 2.0 - 1.0).astype(np.float32), (np.clip((test - low) / scale, 0.0, 1.0) * 2.0 - 1.0).astype(np.float32)
    raise ValueError("unknown frozen series scale")


def imaging_score_pair_paper(train: np.ndarray, test: np.ndarray, config: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    train, test = _prepare_series_pair_paper(np.asarray(train, dtype=np.float32), np.asarray(test, dtype=np.float32), str(config["series_scale"]))
    if config["image"] == "spectrogram": features = _spectrogram_images_paper
    elif config["image"] == "rp": features = _rp_images_paper
    else: raise ValueError("unknown frozen image transform")
    train_images, test_images = features(train, int(config["size"])), features(test, int(config["size"]))
    train_features, test_features = (_glcm_features_paper(train_images), _glcm_features_paper(test_images)) if config.get("feature_extractor") == "glcm" else (train_images.reshape(len(train), -1), test_images.reshape(len(test), -1))
    train_features, test_features = _pca_pair_paper(train_features, test_features, int(config["pca"]))
    train_features, test_features = scale_feature_pair_paper(train_features, test_features)
    from .knn_paper import density_knn_score_pair_paper
    # Frozen imaging configs use standard KNN rather than local-gap density.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=max(1, min(int(config["neighbors"]) + 1, len(train_features))), metric="euclidean").fit(train_features)
    train_distance, _ = nn.kneighbors(train_features); test_distance, _ = nn.kneighbors(test_features)
    return (train_distance[:, 1:].mean(axis=1) if train_distance.shape[1] > 1 else train_distance[:, 0], test_distance[:, :max(1, min(int(config["neighbors"]), test_distance.shape[1]))].mean(axis=1))
