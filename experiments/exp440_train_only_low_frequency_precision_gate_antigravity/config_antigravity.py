"""Frozen scientific and runtime configuration for Exp440."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os


@dataclass(frozen=True)
class Exp440FrozenConfig:
    experiment_id: str = "Exp440"
    run_identity: str = "exp440-train-only-low-frequency-precision-gate-antigravity-20260808"
    dataset_count: int = 1117
    database_path: str = "/Users/mac/Documents/Dataset/univariate_ts.db"
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    exp439_prediction_sha256: str = "500f6956b8640ea329cd47c8cdb504debaeff6ec67c44d0c56b96c1ecd77f3ef"
    hard_prediction_sha256: str = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"
    nearest_k: int = 3
    quantile: float = 0.99
    quantile_method: str = "higher"
    comparison: str = "strict_greater"
    adjacent_offsets: tuple[int, ...] = (-1, 0, 1)
    minimum_consensus_depths: int = 2
    minimum_train_rows: int = 4
    robust_scale_floor: float = 1e-12
    stability_ratio_max: float = 2.0
    outside_hard_budget: int = 100
    bootstrap_seed: int = 44020260808
    bootstrap_samples: int = 20000
    mlx_seed: int = 44001
    batch_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)
    acceleration_tolerance: float = 1e-5
    acceleration_speedup_min: float = 1.1
    review_only: bool = True
    operational_alert_eligible: bool = False
    run_parent: str = "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808"
    readiness_path: str = "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/readiness_antigravity.json"
    output_path: str = "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/output_antigravity"
    checkpoint_path: str = "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity"
    publication_path: str = "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/publication_antigravity"
    evaluation_path: str = "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/evaluation_antigravity"

    def __post_init__(self) -> None:
        if self.experiment_id != "Exp440" or self.dataset_count != 1117:
            raise ValueError("Exp440 identity is frozen")
        if self.nearest_k != 3 or self.quantile != 0.99 or self.quantile_method != "higher":
            raise ValueError("Exp440 calibration is frozen")
        if self.adjacent_offsets != (-1, 0, 1) or self.minimum_consensus_depths != 2:
            raise ValueError("Exp440 depth consensus is frozen")
        if self.outside_hard_budget != 100 or self.stability_ratio_max != 2.0:
            raise ValueError("Exp440 precision gates are frozen")
        if self.batch_sizes != (16, 32, 64, 128, 256):
            raise ValueError("Exp440 batch sizes are frozen")
        if not self.review_only or self.operational_alert_eligible:
            raise ValueError("Exp440 is review-only")


FROZEN_CONFIG = Exp440FrozenConfig()


def cpu_worker_count_antigravity(logical_cpus: int | None = None) -> int:
    count = os.cpu_count() if logical_cpus is None else logical_cpus
    if type(count) is not int or count < 1:
        raise ValueError("logical CPU count must be a positive integer")
    return max(1, count - 1)


def frozen_config_payload_antigravity() -> dict[str, object]:
    payload = asdict(FROZEN_CONFIG)
    payload["adjacent_offsets"] = list(FROZEN_CONFIG.adjacent_offsets)
    payload["batch_sizes"] = list(FROZEN_CONFIG.batch_sizes)
    return payload


__all__ = ["Exp440FrozenConfig", "FROZEN_CONFIG", "cpu_worker_count_antigravity", "frozen_config_payload_antigravity"]
