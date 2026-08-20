"""Pure Exp450 length conversion and adaptive local-mean pooling primitives.

The ``linear`` mode delegates directly to the frozen paper control.  The only
new length rule is SciPy polyphase downsampling plus normalized-coordinate
PCHIP upsampling.  These functions do not repair nonfinite source values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import scipy.interpolate
import scipy.signal
import torch
import torch.nn.functional as torch_functional

from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import (
    resample_series_paper,
)


def resample_series_exp450_antigravity(
    values: np.ndarray, target_len: int, *, mode: str
) -> np.ndarray:
    """Return one exact-length float32 row using the selected frozen factor."""
    row = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_len <= 0 or not len(row):
        raise ValueError("Exp450 resampling requires nonempty input and positive target")
    if mode == "linear":
        return resample_series_paper(row, target_len)
    if mode != "polyphase_pchip":
        raise ValueError("unknown Exp450 length mode")
    if len(row) == target_len:
        return row.astype(np.float32, copy=False)
    if len(row) == 1:
        return np.full(target_len, row[0], dtype=np.float32)
    if target_len < len(row):
        divisor = math.gcd(len(row), target_len)
        result = scipy.signal.resample_poly(row, target_len // divisor, len(row) // divisor)
    else:
        source = np.linspace(0.0, 1.0, len(row), dtype=np.float64)
        target = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
        result = scipy.interpolate.PchipInterpolator(source, row, extrapolate=False)(target)
    output = np.asarray(result, dtype=np.float32)
    if output.shape != (target_len,) or not np.isfinite(output).all():
        raise ValueError("Exp450 resampler output differs")
    return output


def align_rows_exp450_antigravity(
    rows: Sequence[np.ndarray], target_len: int, *, mode: str
) -> np.ndarray:
    """Stack independently resampled rows without changing their target rule."""
    if target_len <= 0:
        raise ValueError("Exp450 resampling requires nonempty input and positive target")
    materialized_rows = tuple(rows)
    if not materialized_rows:
        return np.empty((0, target_len), dtype=np.float32)
    return np.stack(
        [resample_series_exp450_antigravity(row, target_len, mode=mode) for row in materialized_rows]
    ).astype(np.float32, copy=False)


def _pooling_input_antigravity(values: np.ndarray, dimensions: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != dimensions or 0 in array.shape or not np.isfinite(array).all():
        raise ValueError("Exp450 adaptive pooling requires a nonempty finite matrix batch")
    return np.ascontiguousarray(array)


def _pooling_device_antigravity(device: str) -> str:
    if device not in {"cpu", "mps"}:
        raise ValueError("Exp450 adaptive pooling device is invalid")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Exp450 MPS adaptive pooling is unavailable")
    return device


def _pooling_size_1d_antigravity(size: int) -> int:
    if type(size) is not int or size <= 0:
        raise ValueError("Exp450 adaptive pooling size is invalid")
    return size


def _pooling_size_2d_antigravity(size: int | tuple[int, int]) -> tuple[int, int]:
    if type(size) is int:
        if size <= 0:
            raise ValueError("Exp450 adaptive pooling size is invalid")
        return size, size
    if (
        type(size) is tuple
        and len(size) == 2
        and all(type(dimension) is int and dimension > 0 for dimension in size)
    ):
        return size
    raise ValueError("Exp450 adaptive pooling size is invalid")


def _cpu_float32_output_antigravity(tensor: torch.Tensor, device: str) -> np.ndarray:
    if device == "mps":
        torch.mps.synchronize()
    output = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(output).all():
        raise ValueError("Exp450 adaptive pooling output differs")
    return output


def _adaptive_pool_windows_antigravity(
    tensor: torch.Tensor, output_size: tuple[int, ...]
) -> torch.Tensor:
    """Execute PyTorch adaptive-average windows on MPS when its kernel lacks a shape.

    MPS currently rejects non-divisible adaptive-pooling dimensions.  The
    window bounds below are PyTorch's adaptive-average definition and every
    operation remains on the already-selected MPS tensor.
    """
    if len(output_size) == 1:
        length = tensor.shape[-1]
        target_length = output_size[0]
        return torch.stack(
            [
                tensor[..., (index * length) // target_length : ((index + 1) * length + target_length - 1) // target_length].mean(dim=-1)
                for index in range(target_length)
            ],
            dim=-1,
        )
    height, width = tensor.shape[-2:]
    target_height, target_width = output_size
    rows = []
    for row_index in range(target_height):
        start_height = (row_index * height) // target_height
        end_height = ((row_index + 1) * height + target_height - 1) // target_height
        columns = []
        for column_index in range(target_width):
            start_width = (column_index * width) // target_width
            end_width = ((column_index + 1) * width + target_width - 1) // target_width
            columns.append(
                tensor[..., start_height:end_height, start_width:end_width].mean(dim=(-2, -1))
            )
        rows.append(torch.stack(columns, dim=-1))
    return torch.stack(rows, dim=-2)


def _adaptive_pool_1d_antigravity(
    tensor: torch.Tensor, output_size: int, device: str
) -> torch.Tensor:
    try:
        return torch_functional.adaptive_avg_pool1d(tensor, output_size)
    except RuntimeError as error:
        if (
            device != "mps"
            or tensor.shape[-1] % output_size == 0
            or "input sizes must be divisible by output sizes" not in str(error)
        ):
            raise
        return _adaptive_pool_windows_antigravity(tensor, (output_size,))


def _adaptive_pool_2d_antigravity(
    tensor: torch.Tensor, output_size: tuple[int, int], device: str
) -> torch.Tensor:
    try:
        return torch_functional.adaptive_avg_pool2d(tensor, output_size)
    except RuntimeError as error:
        if device != "mps" or (
            tensor.shape[-2] % output_size[0] == 0
            and tensor.shape[-1] % output_size[1] == 0
        ) or "input sizes must be divisible by output sizes" not in str(error):
            raise
        return _adaptive_pool_windows_antigravity(tensor, output_size)


def adaptive_average_1d_antigravity(
    values: np.ndarray, size: int, *, device: str
) -> np.ndarray:
    """Apply PyTorch adaptive average pooling to an ``(rows, length)`` batch."""
    matrix = _pooling_input_antigravity(values, 2)
    output_size = _pooling_size_1d_antigravity(size)
    selected_device = _pooling_device_antigravity(device)
    tensor = torch.as_tensor(matrix, device=selected_device, dtype=torch.float32)
    pooled = _adaptive_pool_1d_antigravity(tensor.unsqueeze(1), output_size, selected_device).squeeze(1)
    output = _cpu_float32_output_antigravity(pooled, selected_device)
    if output.shape != (matrix.shape[0], output_size):
        raise ValueError("Exp450 adaptive pooling output differs")
    return output


def adaptive_average_2d_antigravity(
    values: np.ndarray, size: int | tuple[int, int], *, device: str
) -> np.ndarray:
    """Apply PyTorch adaptive average pooling to an ``(rows, height, width)`` batch."""
    matrix = _pooling_input_antigravity(values, 3)
    output_size = _pooling_size_2d_antigravity(size)
    selected_device = _pooling_device_antigravity(device)
    tensor = torch.as_tensor(matrix, device=selected_device, dtype=torch.float32)
    pooled = _adaptive_pool_2d_antigravity(tensor.unsqueeze(1), output_size, selected_device).squeeze(1)
    output = _cpu_float32_output_antigravity(pooled, selected_device)
    if output.shape != (matrix.shape[0], *output_size):
        raise ValueError("Exp450 adaptive pooling output differs")
    return output


__all__ = [
    "adaptive_average_1d_antigravity",
    "adaptive_average_2d_antigravity",
    "align_rows_exp450_antigravity",
    "resample_series_exp450_antigravity",
]
