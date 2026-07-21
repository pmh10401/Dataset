from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FrozenDatasetInputs:
    dataset: str
    n_train: int
    n_test: int
    hard: frozenset[int]
    shadow: frozenset[int]
    priority: frozenset[int]
    standard: frozenset[int]
    evidence: Mapping[str, frozenset[int]]
    available_groups: frozenset[str]
    quality_flags: frozenset[int]
    target_length: int = 0


@dataclass(frozen=True)
class FrozenRecoveryInputs:
    population: tuple[str, ...]
    datasets: Mapping[str, FrozenDatasetInputs]
    input_sha256: Mapping[str, str]
    lane_totals: Mapping[str, int]
