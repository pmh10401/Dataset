"""Exact immutable configuration preregistered for Exp421."""

from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real

from experiments.exp417_causal_multihorizon_shadow.config import (
    Exp417FrozenConfig,
    FROZEN_CONFIG as EXP417_FROZEN_CONFIG,
)


@dataclass(frozen=True)
class Exp421Config:
    """The fixed scientific and resource boundaries for Exp421."""

    experiment_id: str = "Exp421"
    exp417_config: Exp417FrozenConfig = EXP417_FROZEN_CONFIG
    candidate_batch_sizes: tuple[int, ...] = (1, 4, 8, 16, 32)
    max_normalized_error: float = 1e-4
    min_spearman: float = 0.999
    min_speedup: float = 1.1
    frozen_model_seeds: tuple[int, ...] = (41701, 41702, 41703)
    frozen_perturbation_seeds: tuple[int, ...] = (41711, 41712, 41713)
    max_dataset_workers: int = 7
    max_mlx_heavy_processes: int = 1

    def validate(self) -> None:
        if type(self.exp417_config) is not Exp417FrozenConfig:
            raise ValueError("Exp421 frozen config mismatch: exp417_config")
        self.exp417_config.validate()
        expected = type(self)()
        for item in fields(self):
            actual = getattr(self, item.name)
            wanted = getattr(expected, item.name)
            if type(actual) is not type(wanted) or actual != wanted:
                raise ValueError(f"Exp421 frozen config mismatch: {item.name}")
        if self.frozen_model_seeds != self.exp417_config.model_seeds:
            raise ValueError("Exp421 model seeds must bind to Exp417")
        if self.frozen_perturbation_seeds != self.exp417_config.perturbation_seeds:
            raise ValueError("Exp421 perturbation seeds must bind to Exp417")
        if (
            self.candidate_batch_sizes != tuple(sorted(self.candidate_batch_sizes))
            or len(self.candidate_batch_sizes) != len(set(self.candidate_batch_sizes))
            or any(type(size) is not int or size < 1 for size in self.candidate_batch_sizes)
        ):
            raise ValueError("Exp421 batch sizes must be sorted, unique positive integers")
        for name in ("max_normalized_error", "min_spearman", "min_speedup"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"Exp421 frozen config mismatch: {name}")
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1 for name in ("max_dataset_workers", "max_mlx_heavy_processes")):
            raise ValueError("Exp421 resource bounds must be positive integers")


FROZEN_CONFIG = Exp421Config()
FROZEN_CONFIG.validate()
