"""Variant-aware Exp55/56 imaging with frozen downstream scoring semantics.

``linear`` delegates directly to the retained paper scorer.  ``area`` replaces
only its fixed-size resize: spectrograms use two-dimensional adaptive local
means and recurrence plots pool the one-dimensional series before recomputing
their symmetric distance matrices.  The copied scoring closure below records
its immutable source module and SHA-256 for Exp450 provenance.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from paper.equipment_anomaly_detection.sources_paper.imaging_paper import (
    _glcm_features_paper,
    _pca_pair_paper,
    _prepare_series_pair_paper,
    imaging_score_pair_paper,
)
from paper.equipment_anomaly_detection.sources_paper.knn_paper import (
    scale_feature_pair_paper,
)

from .transforms_antigravity import (
    adaptive_average_1d_antigravity,
    adaptive_average_2d_antigravity,
)


FROZEN_IMAGING_SOURCE_ANTIGRAVITY = {
    "path": "paper/equipment_anomaly_detection/sources_paper/imaging_paper.py",
    "sha256": "1c18b6c6e3f70c922f9bba9689d26b99d2eb6320864e9c6425727af6e95d61e2",
}
PROVENANCE_ANTIGRAVITY = {"frozen_imaging_source": FROZEN_IMAGING_SOURCE_ANTIGRAVITY}


def spectrogram_images_exp450_antigravity(
    matrix: np.ndarray, size: int, *, device: str
) -> np.ndarray:
    """Build frozen spectrogram features, then replace resize with area pooling."""
    values = np.asarray(matrix, dtype=np.float32)
    spectra: list[np.ndarray] = []
    for row in values:
        nperseg = max(8, min(len(row), size * 2))
        hop = max(1, nperseg // 2)
        starts = list(range(0, max(1, len(row) - nperseg + 1), hop)) or [0]
        window = np.hanning(nperseg).astype(np.float32)
        frames: list[np.ndarray] = []
        for start in starts:
            segment = row[start : start + nperseg]
            if len(segment) < nperseg:
                segment = np.pad(segment, (0, nperseg - len(segment)), mode="edge")
            frames.append(np.log1p(np.abs(np.fft.rfft(segment * window))).astype(np.float32))
        spectra.append(np.vstack(frames).T)
    images = adaptive_average_2d_antigravity(np.stack(spectra), size, device=device)
    images -= images.min(axis=(1, 2), keepdims=True)
    images /= images.max(axis=(1, 2), keepdims=True) + 1e-6
    return images.astype(np.float32, copy=False)


def recurrence_images_exp450_antigravity(
    matrix: np.ndarray, size: int, *, device: str
) -> np.ndarray:
    """Pool each series, then recompute its frozen recurrence-distance image."""
    fixed = adaptive_average_1d_antigravity(
        np.asarray(matrix, dtype=np.float32), size, device=device
    )
    tensor = torch.as_tensor(fixed, device=device, dtype=torch.float32)
    distance = torch.abs(tensor[:, :, None] - tensor[:, None, :])
    scale = torch.quantile(distance.flatten(1), 0.75, dim=1).clamp_min(1e-6)
    images = torch.exp(-distance / scale[:, None, None])
    if device == "mps":
        torch.mps.synchronize()
    return images.detach().cpu().numpy().astype(np.float32, copy=False)


def imaging_score_pair_exp450_antigravity(
    train: np.ndarray,
    test: np.ndarray,
    config: dict[str, object],
    *,
    image_mode: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a frozen Exp55/56 config with the declared Exp450 image factor."""
    if image_mode == "linear":
        return imaging_score_pair_paper(train, test, config)
    if image_mode != "area":
        raise ValueError("unknown Exp450 image mode")
    train, test = _prepare_series_pair_paper(
        np.asarray(train, dtype=np.float32),
        np.asarray(test, dtype=np.float32),
        str(config["series_scale"]),
    )
    if config["image"] == "spectrogram":
        image_function = spectrogram_images_exp450_antigravity
    elif config["image"] == "rp":
        image_function = recurrence_images_exp450_antigravity
    else:
        raise ValueError("unknown frozen image transform")
    train_images = image_function(train, int(config["size"]), device=device)
    test_images = image_function(test, int(config["size"]), device=device)
    if config.get("feature_extractor") == "glcm":
        train_features, test_features = (
            _glcm_features_paper(train_images),
            _glcm_features_paper(test_images),
        )
    else:
        train_features, test_features = (
            train_images.reshape(len(train), -1),
            test_images.reshape(len(test), -1),
        )
    train_features, test_features = _pca_pair_paper(
        train_features, test_features, int(config["pca"])
    )
    train_features, test_features = scale_feature_pair_paper(train_features, test_features)
    neighbors = NearestNeighbors(
        n_neighbors=max(1, min(int(config["neighbors"]) + 1, len(train_features))),
        metric="euclidean",
    ).fit(train_features)
    train_distance, _ = neighbors.kneighbors(train_features)
    test_distance, _ = neighbors.kneighbors(test_features)
    return (
        train_distance[:, 1:].mean(axis=1)
        if train_distance.shape[1] > 1
        else train_distance[:, 0],
        test_distance[
            :, : max(1, min(int(config["neighbors"]), test_distance.shape[1]))
        ].mean(axis=1),
    )


__all__ = [
    "FROZEN_IMAGING_SOURCE_ANTIGRAVITY",
    "PROVENANCE_ANTIGRAVITY",
    "imaging_score_pair_exp450_antigravity",
    "recurrence_images_exp450_antigravity",
    "spectrogram_images_exp450_antigravity",
]
