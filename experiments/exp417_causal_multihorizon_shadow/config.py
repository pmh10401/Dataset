"""Exact immutable configuration preregistered for Exp417."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Exp417FrozenConfig:
    experiment_id: str = "Exp417"
    expected_datasets: int = 1117
    expected_hard: int = 2218
    expected_standard: int = 806
    expected_priority: int = 31
    expected_automatic_normal: int = 35
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    exp291_prediction_sha256: str = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"
    exp291_manifest_sha256: str = "761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e"
    exp292_prediction_sha256: str = "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b"
    exp292_manifest_sha256: str = "0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0"
    model_seeds: tuple[int, ...] = (41701, 41702, 41703)
    perturbation_seeds: tuple[int, ...] = (41711, 41712, 41713)
    horizon_fractions: tuple[float, float] = (0.02, 0.10)
    candidate_alpha: float = 0.01
    persistence_fraction: float = 0.05
    min_train_model: int = 50
    min_train_candidate: int = 100
    oof_candidate_rate: float = 0.01
    oof_wilson_upper: float = 0.05
    pooled_synthetic_sensitivity: float = 0.70
    per_type_synthetic_sensitivity: float = 0.50
    dataset_workload_rate: float = 0.05
    global_workload_rate: float = 0.01
    mps_max_error: float = 1e-4
    mps_min_spearman: float = 0.999
    mps_min_speedup: float = 1.1

    def validate(self) -> None:
        expected = type(self)()
        for field in fields(self):
            actual = getattr(self, field.name)
            wanted = getattr(expected, field.name)
            if type(actual) is not type(wanted) or actual != wanted:
                raise ValueError(f"Exp417 frozen config mismatch: {field.name}")


FROZEN_CONFIG = Exp417FrozenConfig()
