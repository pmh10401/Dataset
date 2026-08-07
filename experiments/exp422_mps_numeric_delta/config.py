"""Exact immutable configuration preregistered for Exp422."""

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from math import isfinite
from numbers import Real

from experiments.exp417_causal_multihorizon_shadow.config import (
    Exp417FrozenConfig,
    FROZEN_CONFIG as EXP417_FROZEN_CONFIG,
)


@dataclass(frozen=True)
class Exp422Config:
    """The fixed scientific and resource boundaries for the Exp422 audit."""

    experiment_id: str = "Exp422"
    exp417_config: Exp417FrozenConfig = EXP417_FROZEN_CONFIG
    candidate_batch_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)
    # These values deliberately bind to the canonical Exp417 gate fields at
    # class definition time; they are not duplicated Exp422 literals.
    max_normalized_error: float = EXP417_FROZEN_CONFIG.mps_max_error
    min_spearman: float = EXP417_FROZEN_CONFIG.mps_min_spearman
    min_speedup: float = EXP417_FROZEN_CONFIG.mps_min_speedup
    max_series: int = 64
    max_batch_bytes: int = 256 * 1024 * 1024
    model_seeds: tuple[int, ...] = (41701, 41702, 41703)
    perturbation_seeds: tuple[int, ...] = (41711, 41712, 41713)

    def validate(self) -> None:
        """Reject every deviation from the preregistered Exp422 boundary."""
        if type(self.exp417_config) is not Exp417FrozenConfig:
            raise ValueError("Exp422 frozen config mismatch: exp417_config")
        self.exp417_config.validate()
        expected = type(self)()
        for field in fields(self):
            actual = getattr(self, field.name)
            wanted = getattr(expected, field.name)
            if type(actual) is not type(wanted) or actual != wanted:
                raise ValueError(f"Exp422 frozen config mismatch: {field.name}")
        if self.model_seeds != self.exp417_config.model_seeds:
            raise ValueError("Exp422 model seeds must bind to Exp417")
        if self.perturbation_seeds != self.exp417_config.perturbation_seeds:
            raise ValueError("Exp422 perturbation seeds must bind to Exp417")
        canonical_gates = {
            "max_normalized_error": self.exp417_config.mps_max_error,
            "min_spearman": self.exp417_config.mps_min_spearman,
            "min_speedup": self.exp417_config.mps_min_speedup,
        }
        for name, wanted in canonical_gates.items():
            if getattr(self, name) != wanted:
                raise ValueError(f"Exp422 {name} must bind to Exp417")
        if (
            self.candidate_batch_sizes != tuple(sorted(self.candidate_batch_sizes))
            or len(self.candidate_batch_sizes) != len(set(self.candidate_batch_sizes))
            or any(type(size) is not int or size < 1 for size in self.candidate_batch_sizes)
        ):
            raise ValueError("Exp422 batch sizes must be sorted, unique positive integers")
        for name in ("max_normalized_error", "min_spearman", "min_speedup"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"Exp422 frozen config mismatch: {name}")
        for name in ("max_series", "max_batch_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Exp422 resource bound must be a positive integer: {name}")


FROZEN_CONFIG = Exp422Config()
FROZEN_CONFIG.validate()


def frozen_config_sha256(config: Exp422Config = FROZEN_CONFIG) -> str:
    """Return the canonical digest for the exact immutable Exp422 config."""
    if type(config) is not Exp422Config:
        raise ValueError("config must be an Exp422Config")
    config.validate()
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(payload).hexdigest()
