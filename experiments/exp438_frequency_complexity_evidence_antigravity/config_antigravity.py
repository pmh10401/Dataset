from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

from experiments.exp403_408.artifact_io import canonical_json_bytes


DATABASE_PATH_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
RUN_PARENT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp438_frequency_complexity_evidence_antigravity_20260807"
)
OUTPUT_ROOT_ANTIGRAVITY = RUN_PARENT_ANTIGRAVITY / "output_antigravity"
CHECKPOINT_ROOT_ANTIGRAVITY = RUN_PARENT_ANTIGRAVITY / "checkpoint_antigravity"
EVALUATION_ROOT_ANTIGRAVITY = RUN_PARENT_ANTIGRAVITY / "evaluation_antigravity"
READINESS_PATH_ANTIGRAVITY = RUN_PARENT_ANTIGRAVITY / "readiness_antigravity.json"


def cpu_worker_count_antigravity(logical_cpus: int | None = None) -> int:
    if logical_cpus is None:
        detected = os.cpu_count() or 1
    else:
        if type(logical_cpus) is not int:
            raise TypeError("logical_cpus must be an integer or None")
        if logical_cpus < 1:
            raise ValueError("logical_cpus must be positive")
        detected = logical_cpus
    return max(1, detected - 1)


@dataclass(frozen=True)
class Exp438FrozenConfig:
    experiment_id: str = "Exp438"
    run_identity: str = "exp438-frequency-complexity-evidence-antigravity-20260807"
    dataset_count: int = 1117
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    minimum_raw_length: int = 16
    minimum_train_rows: int = 4
    maximum_depth: int = 5
    embedding_dimension: int = 3
    delays: tuple[int, ...] = (1, 2, 4)
    nearest_k: int = 3
    quantile: float = 0.99
    quantile_method: str = "higher"
    strict_greater: bool = True
    channel_candidate_budget: int = 10_000
    total_candidate_budget: int = 30_000
    smoke_seed: int = 43801
    batch_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)
    warmup_runs: int = 2
    timed_runs: int = 5
    parity_tolerance: float = 1e-5
    minimum_speedup: float = 1.1
    bootstrap_seed: int = 43820260807
    bootstrap_samples: int = 20_000
    review_only: bool = True
    operational_alert_eligible: bool = False

    def __post_init__(self) -> None:
        if self.experiment_id != "Exp438" or self.run_identity != "exp438-frequency-complexity-evidence-antigravity-20260807":
            raise ValueError("frozen Exp438 identity is invalid")
        if self.dataset_count != 1117:
            raise ValueError("frozen Exp438 population is invalid")
        if len(self.database_sha256) != 64 or any(character not in "0123456789abcdef" for character in self.database_sha256):
            raise ValueError("frozen Exp438 database digest is invalid")
        integer_fields = (
            self.minimum_raw_length,
            self.minimum_train_rows,
            self.maximum_depth,
            self.embedding_dimension,
            self.nearest_k,
            self.channel_candidate_budget,
            self.total_candidate_budget,
            self.smoke_seed,
            self.warmup_runs,
            self.timed_runs,
            self.bootstrap_seed,
            self.bootstrap_samples,
        )
        if any(type(value) is not int or value < 1 for value in integer_fields):
            raise ValueError("frozen Exp438 positive integer is invalid")
        if self.delays != (1, 2, 4) or self.batch_sizes != (16, 32, 64, 128, 256):
            raise ValueError("frozen Exp438 tuple is invalid")
        if self.quantile_method != "higher" or not self.strict_greater:
            raise ValueError("frozen Exp438 threshold rule is invalid")
        if not (math.isfinite(self.quantile) and 0.0 < self.quantile < 1.0):
            raise ValueError("frozen Exp438 quantile is invalid")
        if not (math.isfinite(self.parity_tolerance) and self.parity_tolerance > 0.0):
            raise ValueError("frozen Exp438 parity tolerance is invalid")
        if not (math.isfinite(self.minimum_speedup) and self.minimum_speedup >= 1.1):
            raise ValueError("frozen Exp438 speed gate is invalid")
        if not self.review_only or self.operational_alert_eligible:
            raise ValueError("frozen Exp438 review policy is invalid")


FROZEN_CONFIG = Exp438FrozenConfig()


def frozen_config_payload_antigravity() -> dict[str, object]:
    return json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG)))


def validate_frozen_config_antigravity(value: object) -> Exp438FrozenConfig:
    if type(value) is not Exp438FrozenConfig or value != FROZEN_CONFIG:
        raise ValueError("frozen Exp438 config does not match")
    return FROZEN_CONFIG
