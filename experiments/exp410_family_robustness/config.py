from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Exp410FrozenConfig:
    experiment_id: str = "Exp410"
    expected_datasets: int = 1117
    expected_families: int = 128
    expected_family_size_buckets: tuple[int, int, int, int] = (1, 74, 39, 14)
    expected_hard: int = 2218
    expected_tp: int = 1915
    expected_fp: int = 303
    expected_fn: int = 2003
    expected_dataset_macro_f1: float = 0.6089825086931719
    display_dataset_macro_f1: str = "0.608983"
    prediction_sha256: str = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"
    frozen_labels_sha256: str = "b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4"
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    exp291_sha256: str = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"
    exp292_sha256: str = "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b"
    family_weight_caps: tuple[int | None, ...] = (1, 3, 5, 10, None)
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 410
    float_tolerance: float = 1e-12
    robustness_margin: float = -0.05
    severe_margin: float = -0.10
    operational_policy_changed: bool = False
    deployment_eligible: bool = False
    unseen_family_validated: bool = False
    prospective_validated: bool = False

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


FROZEN_CONFIG = Exp410FrozenConfig()
