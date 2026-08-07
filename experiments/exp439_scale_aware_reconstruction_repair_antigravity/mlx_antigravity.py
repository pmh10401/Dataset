"""Optional, lazily-created MLX dense-distance adapter for Exp439."""

from __future__ import annotations

from importlib import metadata
import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class MlxUnavailableAntigravity(RuntimeError):
    pass


class MlxDenseEngineAntigravity:
    backend = "mlx"

    def __init__(
        self,
        scorer: Callable[[NDArray[np.float32], NDArray[np.float32], int, int], object],
        mlx_version: str,
        mlx_metal_version: str,
        device: str,
        synchronize: Callable[[], None] = lambda: None,
    ) -> None:
        if not callable(scorer) or not callable(synchronize):
            raise ValueError("MLX engine callable is invalid")
        if any(type(value) is not str or not value for value in (mlx_version, mlx_metal_version, device)):
            raise ValueError("MLX engine provenance is invalid")
        self._scorer = scorer
        self._synchronize = synchronize
        self.mlx_version = mlx_version
        self.mlx_metal_version = mlx_metal_version
        self.device = device

    def score_batch(self, queries: object, references: object, k: int, *, batch_size: int) -> NDArray[np.float32]:
        query = np.asarray(queries, dtype=np.float32)
        reference = np.asarray(references, dtype=np.float32)
        if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1] or reference.shape[0] < k:
            raise ValueError("MLX dense input is invalid")
        if type(k) is not int or k < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("MLX dense parameter is invalid")
        if not np.isfinite(query).all() or not np.isfinite(reference).all():
            raise ValueError("MLX dense input must be finite")
        result = np.asarray(self._scorer(query, reference, k, batch_size), dtype=np.float32)
        self._synchronize()
        if result.shape != (query.shape[0],) or not np.isfinite(result).all():
            raise ValueError("MLX dense output is invalid")
        result = np.ascontiguousarray(result).copy()
        result.flags.writeable = False
        return result

    def synchronize(self) -> None:
        self._synchronize()


def create_mlx_dense_engine_antigravity() -> MlxDenseEngineAntigravity:
    try:
        import mlx.core as mx
    except (ImportError, OSError) as exc:
        raise MlxUnavailableAntigravity("mlx framework unavailable") from exc
    try:
        mlx_version = metadata.version("mlx")
    except metadata.PackageNotFoundError:
        mlx_version = str(getattr(mx, "__version__", "unknown"))
    try:
        mlx_metal_version = metadata.version("mlx-metal")
    except metadata.PackageNotFoundError:
        mlx_metal_version = "bundled-or-unreported"
    try:
        device = str(mx.metal.device_info())
    except (AttributeError, RuntimeError):
        device = str(mx.default_device())
    if "cpu" in device.casefold():
        raise MlxUnavailableAntigravity("mlx Metal device unavailable")

    def scorer(queries, references, k, batch_size):
        reference = mx.array(references)
        outputs = []
        for start in range(0, queries.shape[0], batch_size):
            query = mx.array(queries[start : start + batch_size])
            delta = query[:, None, :] - reference[None, :, :]
            distances = mx.sqrt(mx.mean(delta * delta, axis=2))
            nearest = mx.sort(distances, axis=1)[:, :k]
            if k % 2:
                result = nearest[:, k // 2]
            else:
                result = (nearest[:, k // 2 - 1] + nearest[:, k // 2]) / 2.0
            mx.eval(result)
            outputs.append(np.asarray(result, dtype=np.float32))
        return np.concatenate(outputs)

    def synchronize() -> None:
        try:
            mx.synchronize()
        except AttributeError:
            mx.eval(mx.array([0.0]))

    return MlxDenseEngineAntigravity(
        scorer,
        mlx_version,
        mlx_metal_version,
        device,
        synchronize,
    )


__all__ = [
    "MlxDenseEngineAntigravity",
    "MlxUnavailableAntigravity",
    "create_mlx_dense_engine_antigravity",
]
