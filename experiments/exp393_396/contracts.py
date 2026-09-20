from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GrammarConfig:
    alphas: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
    state_count: int = 8
    random_seed: int = 20260727
    n_init: int = 10
    max_iter: int = 200
    high_support_rate: float = 0.05
    transition_tail_rate: float = 0.95
    component_tail_rate: float = 0.95


@dataclass(frozen=True)
class PreparedDataset:
    train: np.ndarray
    test: np.ndarray
    target_length: int
    train_center: np.ndarray
    train_scale: np.ndarray


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def config_sha256(config: GrammarConfig) -> str:
    return hashlib.sha256(canonical_json(asdict(config)).encode("utf-8")).hexdigest()
