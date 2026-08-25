"""Load only the official ECG200 TRAIN normals for Exp465."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


TRAIN_LABEL_QUERY_COUNT_ANTIGRAVITY = 0
_TRAIN_MD5_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.train_md5
_EXPECTED_TRAIN_ROWS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.train_rows
_EXPECTED_WIDTH_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.width
_EXPECTED_NORMAL_ROWS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.normal_rows
_EXPECTED_INFARCT_ROWS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.infarct_rows


def _reject_test_path_antigravity(path: Path) -> None:
    if any("TEST" in component.upper() for component in path.parts):
        raise ValueError("TEST paths are forbidden; only ECG200 TRAIN may be read")
    name = path.name.upper()
    if "TRAIN" not in name:
        raise ValueError(f"input file name must contain TRAIN: {path.name}")


def load_ecg200_normal_train_antigravity(path: Path) -> np.ndarray:
    """Return the owned read-only float64 69x96 normal TRAIN matrix."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    _reject_test_path_antigravity(path)
    resolved = path.resolve(strict=True)
    _reject_test_path_antigravity(resolved)

    digest = hashlib.md5()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    file_md5 = digest.hexdigest()
    if file_md5 != _TRAIN_MD5_ANTIGRAVITY:
        raise ValueError(f"TRAIN MD5 drift: {file_md5} != {_TRAIN_MD5_ANTIGRAVITY}")

    rows = []
    labels = []
    with resolved.open("r", encoding="utf-8") as handle:
        in_data = False
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if not in_data:
                if line.startswith("@data"):
                    in_data = True
                continue
            values_text, label_text = line.rsplit(":", 1)
            values = np.array([float(item) for item in values_text.split(",")], dtype=np.float64)
            if values.ndim != 1:
                raise ValueError("each data line must decode to a 1-D float64 row")
            label_text = label_text.strip()
            if label_text not in {"1", "-1"}:
                raise ValueError(f"data line label must be the exact integer +1 or -1, got {label_text!r}")
            labels.append(int(label_text))
            rows.append(values)

    if len(rows) != _EXPECTED_TRAIN_ROWS_ANTIGRAVITY:
        raise ValueError(f"TRAIN must contain exactly {_EXPECTED_TRAIN_ROWS_ANTIGRAVITY} data lines")
    for row_index, row in enumerate(rows):
        if row.shape != (_EXPECTED_WIDTH_ANTIGRAVITY,):
            raise ValueError(f"data line {row_index} width must be {_EXPECTED_WIDTH_ANTIGRAVITY}, got {row.shape[0]}")
    matrix = np.array(rows, dtype=np.float64, copy=True, order="C")
    if matrix.shape != (_EXPECTED_TRAIN_ROWS_ANTIGRAVITY, _EXPECTED_WIDTH_ANTIGRAVITY):
        raise ValueError(
            f"TRAIN shape must be {_EXPECTED_TRAIN_ROWS_ANTIGRAVITY}x{_EXPECTED_WIDTH_ANTIGRAVITY}, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("TRAIN values must be finite")
    normal_count = labels.count(1)
    infarct_count = labels.count(-1)
    if (normal_count, infarct_count) != (
        _EXPECTED_NORMAL_ROWS_ANTIGRAVITY,
        _EXPECTED_INFARCT_ROWS_ANTIGRAVITY,
    ) and (infarct_count, normal_count) != (
        _EXPECTED_NORMAL_ROWS_ANTIGRAVITY,
        _EXPECTED_INFARCT_ROWS_ANTIGRAVITY,
    ):
        raise ValueError(
            "TRAIN labels must contain exactly 69 (+1) and 31 (-1) integer rows"
        )
    majority_label = 1 if labels.count(1) >= labels.count(-1) else -1
    result = np.array(
        matrix[[index for index, label in enumerate(labels) if label == majority_label]],
        dtype=np.float64, copy=True, order="C",
    )
    if result.shape != (_EXPECTED_NORMAL_ROWS_ANTIGRAVITY, _EXPECTED_WIDTH_ANTIGRAVITY):
        raise ValueError("normal selection must return exactly 69x96")
    result.setflags(write=False)
    return result


__all__ = ("load_ecg200_normal_train_antigravity",)
