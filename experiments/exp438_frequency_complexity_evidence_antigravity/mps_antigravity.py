"""Optional, lazily-created PyTorch MPS dense-distance adapter for Exp438."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


class MpsUnavailableAntigravity(RuntimeError):
    pass


class MpsDenseEngineAntigravity:
    backend = "mps"

    def __init__(
        self,
        scorer: Callable[[NDArray[np.float32], NDArray[np.float32], int, int], object],
        pytorch_version: str,
        device: str,
        synchronize: Callable[[], None] = lambda: None,
    ) -> None:
        if not callable(scorer) or not callable(synchronize):
            raise ValueError("MPS engine callable is invalid")
        if type(pytorch_version) is not str or not pytorch_version or type(device) is not str or not device:
            raise ValueError("MPS engine provenance is invalid")
        self._scorer = scorer
        self._synchronize = synchronize
        self.pytorch_version = pytorch_version
        self.device = device

    def score_batch(self, queries: object, references: object, k: int, *, batch_size: int) -> NDArray[np.float32]:
        query = np.asarray(queries, dtype=np.float32)
        reference = np.asarray(references, dtype=np.float32)
        if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1] or reference.shape[0] < k:
            raise ValueError("MPS dense input is invalid")
        if type(k) is not int or k < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("MPS dense parameter is invalid")
        if not np.isfinite(query).all() or not np.isfinite(reference).all():
            raise ValueError("MPS dense input must be finite")
        result = np.asarray(self._scorer(query, reference, k, batch_size), dtype=np.float32)
        self._synchronize()
        if result.shape != (query.shape[0],) or not np.isfinite(result).all():
            raise ValueError("MPS dense output is invalid")
        result = np.ascontiguousarray(result).copy()
        result.flags.writeable = False
        return result

    def synchronize(self) -> None:
        self._synchronize()


def create_mps_dense_engine_antigravity() -> MpsDenseEngineAntigravity:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise MpsUnavailableAntigravity("PyTorch framework unavailable") from exc
    if not torch.backends.mps.is_available():
        raise MpsUnavailableAntigravity("PyTorch MPS unavailable")

    def scorer(queries, references, k, batch_size):
        reference = torch.as_tensor(references, dtype=torch.float32, device="mps")
        outputs = []
        for start in range(0, queries.shape[0], batch_size):
            query = torch.as_tensor(
                queries[start : start + batch_size],
                dtype=torch.float32,
                device="mps",
            )
            distances = torch.cdist(query, reference, p=2) / (query.shape[1] ** 0.5)
            nearest = torch.topk(distances, k, dim=1, largest=False).values
            result = torch.median(nearest, dim=1).values
            outputs.append(result.detach().cpu().numpy())
        return np.concatenate(outputs)

    return MpsDenseEngineAntigravity(
        scorer,
        str(torch.__version__),
        "mps",
        torch.mps.synchronize,
    )


__all__ = [
    "MpsDenseEngineAntigravity",
    "MpsUnavailableAntigravity",
    "create_mps_dense_engine_antigravity",
]
