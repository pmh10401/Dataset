from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_file
from experiments.exp403_408.contracts import FrozenConfig
from experiments.exp403_408.synthetic_anomaly import (
    SYNTHETIC_PERTURBATION_TYPES,
    SyntheticAnomalies,
    SyntheticAnomalyConfig,
    generate_synthetic_anomalies,
)
import experiments.exp403_408.synthetic_anomaly as exp407_module


def perturbation_types() -> tuple[str, ...]:
    return tuple(SYNTHETIC_PERTURBATION_TYPES)


def generate_train_only_perturbations(
    rows: object,
    *,
    target_length: int,
    seed: int,
) -> SyntheticAnomalies:
    config = SyntheticAnomalyConfig(
        frozen=FrozenConfig(seed=int(seed), target_length=int(target_length))
    )
    values = np.asarray(rows, dtype=np.float64)
    return generate_synthetic_anomalies(values, config, int(seed))


def perturbation_provenance(
    *,
    target_length: int,
    seed: int,
) -> dict[str, object]:
    config = SyntheticAnomalyConfig(
        frozen=FrozenConfig(seed=int(seed), target_length=int(target_length))
    )
    source = Path(exp407_module.__file__).resolve()
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "config_sha256": sha256(canonical_json_bytes(asdict(config))).hexdigest(),
        "perturbation_types": perturbation_types(),
    }


__all__ = [
    "generate_train_only_perturbations",
    "perturbation_provenance",
    "perturbation_types",
]
