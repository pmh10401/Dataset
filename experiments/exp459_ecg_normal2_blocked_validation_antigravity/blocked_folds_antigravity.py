"""Deterministic contiguous blocked folds for Exp459 Task 2."""

from __future__ import annotations

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import BlockedFoldAntigravity


def blocked_plan_antigravity(n_rows: int) -> tuple[BlockedFoldAntigravity, ...]:
    if type(n_rows) is not int or n_rows < FROZEN_CONFIG_ANTIGRAVITY.folds:
        raise ValueError("n_rows must be an exact integer with at least one row per fold")
    indices = np.arange(n_rows, dtype=np.int64)
    queries = tuple(
        np.array(part, dtype=np.int64, copy=True)
        for part in np.array_split(indices, FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    return tuple(
        BlockedFoldAntigravity(
            n_rows=n_rows,
            fold_index=index,
            fit_indices=np.concatenate(queries[:index] + queries[index + 1 :]),
            query_indices=query,
        )
        for index, query in enumerate(queries)
    )


__all__ = ("blocked_plan_antigravity",)
