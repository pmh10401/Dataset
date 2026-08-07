"""Immutable preregistered scientific and runtime values for Exp436."""

from __future__ import annotations

from dataclasses import dataclass, fields
import os


def cpu_worker_count_antigravity() -> int:
    """Reserve one logical CPU for the parent process and operating system."""

    return max(1, (os.cpu_count() or 1) - 1)


@dataclass(frozen=True)
class Exp436FrozenConfig:
    """The only configuration which may reach Exp436 input or scoring code."""

    experiment_id: str = "Exp436"
    run_identity: str = "exp436-missing-safe-strict-consensus-antigravity-20260806"
    expected_datasets: int = 1117
    neighbor_count: int = 3
    candidate_alpha: float = 0.01
    train_quantile: float = 0.99
    quantile_method: str = "higher"
    min_finite_train_rows: int = 4
    min_scored_length: int = 2
    min_acf_length: int = 16
    max_acf_lag: int = 32
    numeric_floor: float = 1e-12
    variance_floor: float = 1e-12
    robust_scale_multiplier: float = 1.4826
    score_tolerance: float = 1e-5
    global_candidate_budget: int = 100
    cpu_workers: int = cpu_worker_count_antigravity()
    backend_smoke_seed: int = 43601
    backend_batch_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)
    backend_warmup_runs: int = 2
    backend_timed_runs: int = 5
    backend_protocol: str = "paired-synchronized-v1"
    backend_min_speedup: float = 1.1
    bootstrap_repetitions: int = 20_000
    bootstrap_seed: int = 43_520_260_806
    rescued_dataset_minimum: int = 10
    added_false_positive_maximum: int = 30

    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    exp291_prediction_sha256: str = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"
    exp291_manifest_sha256: str = "761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e"
    exp292_prediction_sha256: str = "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b"
    exp292_manifest_sha256: str = "0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0"
    hard_prediction_sha256: str = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"

    def validate(self) -> None:
        """Fail closed, including on a value with a merely compatible type."""

        expected = type(self)()
        for item in fields(self):
            actual = getattr(self, item.name)
            wanted = getattr(expected, item.name)
            if type(actual) is not type(wanted) or actual != wanted:
                raise ValueError(f"Exp436 frozen config mismatch: {item.name}")


FROZEN_CONFIG = Exp436FrozenConfig()


__all__ = ["Exp436FrozenConfig", "FROZEN_CONFIG", "cpu_worker_count_antigravity"]
