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
HURST_FAILURE_TEXT = "Unable to calculate valid R/S values"


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


def _is_expected_hurst_failure(error: ValueError) -> bool:
    current: BaseException | None = error
    while current is not None:
        if HURST_FAILURE_TEXT in str(current):
            return True
        current = current.__cause__
    return False


class SafeHurstExponentTransformer(HurstExponentTransformer):
    """Keep sktime's estimator, mapping only its known invalid R/S case to NaN."""

    def _fit(self, X, y=None):
        try:
            return super()._fit(X, y)
        except ValueError as error:
            if not _is_expected_hurst_failure(error):
                raise
            self.hurst_estimate_ = np.nan
            self.confidence_interval_ = (np.nan, np.nan)
            return self


def _extract_hurst_block(panel: np.ndarray, *, max_lag: int) -> np.ndarray:
    transformer = SafeHurstExponentTransformer(
        method="rs", min_lag=2, max_lag=max_lag
    )
    return _array(
        transformer.fit_transform(panel), name="hurst_3", rows=len(panel)
    )


def extract_feature_blocks(
    rows: np.ndarray, *, target_length: int
) -> dict[str, np.ndarray]:
    panel = _panel(rows)
    max_lag = max(4, min(100, int(target_length) // 2))
    transformers = {
        "catch22_22": Catch22(catch24=False, replace_nans=True),
        "summary_9": SummaryTransformer(),
        "signature_6": SignatureMoments(degree=2, use_index=True),
    }
    blocks: dict[str, np.ndarray] = {}
    for name, transformer in transformers.items():
        if transformer.get_tag("requires_y", False):
            raise ValueError(f"label-dependent transformer prohibited: {name}")
        blocks[name] = _array(
            transformer.fit_transform(panel), name=name, rows=len(panel)
        )
    blocks["hurst_3"] = _extract_hurst_block(panel, max_lag=max_lag)
    blocks["combined_40"] = np.concatenate(
        [blocks[name] for name in BASE_BLOCKS], axis=1
    )
    return blocks


@dataclass(frozen=True)
class TrainFeatureSanitizer:
    replacements: dict[str, np.ndarray]
    disabled_columns: dict[str, np.ndarray]

    def transform(self, blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if set(blocks) != set(self.replacements) or set(blocks) != set(self.disabled_columns):
            raise ValueError("feature block set differs from fitted TRAIN blocks")
        cleaned = {}
        for name, values in blocks.items():
            array = np.asarray(values, dtype=float).copy()
            replacement = self.replacements[name]
            disabled = self.disabled_columns[name]
            if array.ndim != 2 or array.shape[1] != len(replacement):
                raise ValueError(f"feature width differs from TRAIN: block={name}")
            invalid = ~np.isfinite(array)
            if invalid.any():
                array[invalid] = replacement[np.nonzero(invalid)[1]]
            array[:, disabled] = 0.0
            cleaned[name] = array
        return cleaned


def fit_train_sanitizer(
    train_blocks: dict[str, np.ndarray]
) -> TrainFeatureSanitizer:
    replacements = {}
    disabled_columns = {}
    for name, values in train_blocks.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"feature block must be two-dimensional: {name}")
        columns = []
        disabled = []
        for index in range(array.shape[1]):
            finite = array[np.isfinite(array[:, index]), index]
            columns.append(float(np.median(finite)) if len(finite) else 0.0)
            disabled.append(not len(finite))
        replacements[name] = np.asarray(columns, dtype=float)
        disabled_columns[name] = np.asarray(disabled, dtype=bool)
    return TrainFeatureSanitizer(replacements, disabled_columns)
