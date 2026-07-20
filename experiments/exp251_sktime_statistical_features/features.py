from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sktime.transformations.panel.catch22 import Catch22
from sktime.transformations.panel.summarize import SummaryTransformer
from sktime.transformations.series.hurst import HurstExponentTransformer
from sktime.transformations.series.signature import SignatureMoments


FEATURE_DIMENSIONS = {
    "catch22_22": 22,
    "summary_9": 9,
    "hurst_3": 3,
    "signature_6": 6,
    "combined_40": 40,
}
BASE_BLOCKS = ("catch22_22", "summary_9", "hurst_3", "signature_6")


def _panel(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("aligned time-series rows must be two-dimensional")
    return values[:, np.newaxis, :]


def _array(values, *, name: str, rows: int) -> np.ndarray:
    output = np.asarray(values, dtype=float)
    expected = FEATURE_DIMENSIONS[name]
    if output.shape != (rows, expected):
        raise ValueError(
            f"feature shape mismatch: block={name} expected={(rows, expected)} actual={output.shape}"
        )
    return output


def extract_feature_blocks(
    rows: np.ndarray, *, target_length: int
) -> dict[str, np.ndarray]:
    panel = _panel(rows)
    max_lag = max(4, min(100, int(target_length) // 2))
    transformers = {
        "catch22_22": Catch22(catch24=False, replace_nans=True),
        "summary_9": SummaryTransformer(),
        "hurst_3": HurstExponentTransformer(method="rs", min_lag=2, max_lag=max_lag),
        "signature_6": SignatureMoments(degree=2, use_index=True),
    }
    blocks: dict[str, np.ndarray] = {}
    for name, transformer in transformers.items():
        if transformer.get_tag("requires_y", False):
            raise ValueError(f"label-dependent transformer prohibited: {name}")
        blocks[name] = _array(
            transformer.fit_transform(panel), name=name, rows=len(panel)
        )
    blocks["combined_40"] = np.concatenate(
        [blocks[name] for name in BASE_BLOCKS], axis=1
    )
    return blocks


@dataclass(frozen=True)
class TrainFeatureSanitizer:
    replacements: dict[str, np.ndarray]

    def transform(self, blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if set(blocks) != set(self.replacements):
            raise ValueError("feature block set differs from fitted TRAIN blocks")
        cleaned = {}
        for name, values in blocks.items():
            array = np.asarray(values, dtype=float).copy()
            replacement = self.replacements[name]
            if array.ndim != 2 or array.shape[1] != len(replacement):
                raise ValueError(f"feature width differs from TRAIN: block={name}")
            invalid = ~np.isfinite(array)
            if invalid.any():
                array[invalid] = replacement[np.nonzero(invalid)[1]]
            cleaned[name] = array
        return cleaned


def fit_train_sanitizer(
    train_blocks: dict[str, np.ndarray]
) -> TrainFeatureSanitizer:
    replacements = {}
    for name, values in train_blocks.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"feature block must be two-dimensional: {name}")
        columns = []
        for index in range(array.shape[1]):
            finite = array[np.isfinite(array[:, index]), index]
            columns.append(float(np.median(finite)) if len(finite) else 0.0)
        replacements[name] = np.asarray(columns, dtype=float)
    return TrainFeatureSanitizer(replacements)
