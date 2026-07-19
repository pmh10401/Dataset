from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class DatasetBundle:
    dataset: str
    train_normal: np.ndarray
    test_values: np.ndarray
    target_length: int


@dataclass(frozen=True)
class SourceConfig:
    experiment: int
    name: str
    minimum_train: int
    seed: int
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class ScoreResult:
    source: str
    train_oof: np.ndarray
    test_scores: np.ndarray
    abstained: bool
    abstention_reason: str | None
    provenance: Mapping[str, object]
    artifacts: Mapping[str, np.ndarray] | None = None


@dataclass(frozen=True)
class CalibrationResult:
    pvalues: np.ndarray
    indices: tuple[int, ...]
    attainable_p: float
    resolution_blocked: bool


class ScoreSource(Protocol):
    name: str
    minimum_train: int

    def score(
        self, bundle: DatasetBundle, seed: int, artifact_dir: Path
    ) -> ScoreResult: ...

