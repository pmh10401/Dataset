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

from .config import FROZEN_CONFIG, Exp411FrozenConfig


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


def perturbation_contract_provenance(
    config_value: Exp411FrozenConfig = FROZEN_CONFIG,
) -> dict[str, object]:
    config = asdict(SyntheticAnomalyConfig())
    frozen = dict(config["frozen"])
    frozen["seed"] = "<sha256-derived-per-query>"
    frozen["target_length"] = "<same-as-query-length>"
    config["frozen"] = frozen
    source = Path(exp407_module.__file__).resolve()
    result = {
        "source_module": "experiments.exp403_408.synthetic_anomaly",
        "source_sha256": sha256_file(source),
        "config_contract_sha256": sha256(canonical_json_bytes(config)).hexdigest(),
        "dynamic_fields": {
            "seed": "sha256(Exp411,dataset,fold,query_identity)",
            "target_length": "same_as_query_length",
        },
        "perturbation_types": perturbation_types(),
    }
    if (
        result["source_sha256"] != config_value.perturbation_source_sha256
        or result["config_contract_sha256"]
        != config_value.perturbation_config_contract_sha256
    ):
        raise ValueError("Exp411 perturbation provenance SHA256 mismatch")
    return result


__all__ = [
    "generate_train_only_perturbations",
    "perturbation_contract_provenance",
    "perturbation_provenance",
    "perturbation_types",
]
