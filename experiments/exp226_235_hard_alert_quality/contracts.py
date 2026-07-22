from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


SeriesRows = Sequence[np.ndarray] | np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    dataset: str
    train_normal: SeriesRows
    test_values: SeriesRows
    target_length: int | None = None


@dataclass(frozen=True)
class ScoreBundle:
    dataset: str
    train_oof: Mapping[str, np.ndarray]
    test_scores: Mapping[str, np.ndarray]
    provenance: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionRow:
    experiment_id: str
    dataset: str
    variant: str
    hard_indices: tuple[int, ...] = ()
    review_indices: tuple[int, ...] = ()
    research_indices: tuple[int, ...] = ()
    evidence: Mapping[str, object] = field(default_factory=dict)
    autonomous_alert: bool = False
    contains_test_labels: bool = False
    contains_evaluation_metrics: bool = False


@dataclass(frozen=True)
class DatabaseAudit:
    database_path: str
    database_sha256: str
    dataset_count: int
    train_instance_count: int
    test_instance_count: int
    tables: tuple[str, ...]

