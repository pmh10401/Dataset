"""Exact legacy ROCKET primitives for Exp40 and the independent Block-B pass.

Provenance: run_experiment_26_rocket.py SHA256
e11e1e7ed19d4d14112b9af9e72e36f3e1bc746b3bba5a74335bf713773b2eb6
(make_kernels, dilated_convolution, rocket_transform);
run_experiment_29_train_normal_threshold_calibration.py SHA256
6f7018ead920254bf00145639f9cdb8ffd39612bb2248980415bf6a6c736c3de
(rocket_feature_pair); run_experiment_131_rocket_block_b_calibration.py SHA256
32be94dc3fa015df334991a4c301106cb78a2f62cf83b38e505ccfe90544fe5d
(fixed_reference, score_block_b).
"""
from __future__ import annotations

import math

import numpy as np


ROCKET_SEED_PAPER = 20260706
MAX_TRAIN_REFERENCE_PAPER = 1000
PROVENANCE_PAPER = {
    "run_experiment_26_rocket.py": ("make_kernels", "dilated_convolution", "rocket_transform"),
    "run_experiment_29_train_normal_threshold_calibration.py": ("rocket_feature_pair",),
    "run_experiment_131_rocket_block_b_calibration.py": ("fixed_reference", "score_block_b"),
}


def make_kernels_paper(series_length: int, num_kernels: int = 256, seed: int = ROCKET_SEED_PAPER):
    rng = np.random.default_rng(seed + int(series_length))
    lengths = np.array([7, 9, 11], dtype=np.int64)
    kernels = []
    power = max(0, int(math.log2(max(1, (series_length - 1) / 8))))
    for _ in range(int(num_kernels)):
        length = int(rng.choice(lengths))
        weights = rng.normal(0, 1, size=length).astype(np.float32)
        weights -= weights.mean()
        dilation = int(2 ** rng.integers(0, power + 1)) if power else 1
        kernels.append((weights, dilation, bool(rng.integers(0, 2)), float(rng.uniform(-1, 1))))
    return kernels


def _dilated_convolution_paper(row, weights, dilation, padding, bias):
    if dilation > 1:
        expanded = np.zeros((len(weights) - 1) * dilation + 1, dtype=np.float32)
        expanded[::dilation] = weights
        weights = expanded
    if padding:
        row = np.pad(row, (len(weights) // 2, len(weights) // 2), mode="constant")
    if len(row) < len(weights):
        row = np.pad(row, (0, len(weights) - len(row)), mode="constant")
    return np.convolve(row, weights[::-1], mode="valid") + bias


def rocket_transform_paper(matrix: np.ndarray, kernels) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    features = np.empty((len(matrix), len(kernels) * 2), dtype=np.float32)
    for row_index, row in enumerate(matrix):
        for kernel_index, (weights, dilation, padding, bias) in enumerate(kernels):
            convolution = _dilated_convolution_paper(row, weights, dilation, padding, bias)
            features[row_index, 2 * kernel_index] = np.max(convolution)
            features[row_index, 2 * kernel_index + 1] = np.mean(convolution > 0)
    return features


def reference_train_paper(matrix: np.ndarray, seq_len: int, seed_offset: int = 0) -> np.ndarray:
    if len(matrix) <= MAX_TRAIN_REFERENCE_PAPER:
        return matrix
    rng = np.random.default_rng(ROCKET_SEED_PAPER + int(seed_offset) + int(seq_len) + len(matrix))
    return matrix[rng.choice(len(matrix), size=MAX_TRAIN_REFERENCE_PAPER, replace=False)]


def rocket_feature_pair_paper(train: np.ndarray, test: np.ndarray, seq_len: int, num_kernels: int) -> tuple[np.ndarray, np.ndarray]:
    reference = reference_train_paper(train, seq_len, seed_offset=num_kernels)
    kernels = make_kernels_paper(seq_len, num_kernels=num_kernels)
    return rocket_transform_paper(reference, kernels), rocket_transform_paper(test, kernels)


def block_b_feature_pair_paper(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = reference_train_paper(train, train.shape[1], seed_offset=130)
    kernels = make_kernels_paper(train.shape[1], num_kernels=512)
    return rocket_transform_paper(reference, kernels)[:, 512:], rocket_transform_paper(test, kernels)[:, 512:]
