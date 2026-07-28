from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Exp411FrozenConfig:
    experiment_id: str = "Exp411"
    expected_datasets: int = 1117
    expected_families: int = 128
    database_sha256: str = (
        "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    )
    expected_database_snapshot_keys: tuple[str, ...] = ("db",)
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.10)
    quantile_method: str = "lower"
    seed: int = 411
    bootstrap_iterations: int = 2000
    minimum_identity_groups: int = 5
    five_fold_minimum_groups: int = 20
    minimum_same_length_fit_groups: int = 5
    mad_floor: float = 1e-12
    distance_tolerance: float = 1e-6
    speedup_gate: float = 1.1
    minimum_normal_acceptance: float = 0.01
    pooled_synthetic_false_normal_max: float = 0.001
    per_perturbation_false_normal_max: float = 0.005
    bootstrap_upper_max: float = 0.005

    def validate(self) -> None:
        observed = (
            self.experiment_id,
            self.expected_datasets,
            self.expected_families,
            self.database_sha256,
            self.expected_database_snapshot_keys,
            self.quantiles,
            self.quantile_method,
            self.seed,
            self.bootstrap_iterations,
            self.minimum_identity_groups,
            self.five_fold_minimum_groups,
            self.minimum_same_length_fit_groups,
            self.mad_floor,
            self.distance_tolerance,
            self.speedup_gate,
            self.minimum_normal_acceptance,
            self.pooled_synthetic_false_normal_max,
            self.per_perturbation_false_normal_max,
            self.bootstrap_upper_max,
        )
        expected = (
            "Exp411",
            1117,
            128,
            "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
            ("db",),
            (0.01, 0.05, 0.10),
            "lower",
            411,
            2000,
            5,
            20,
            5,
            1e-12,
            1e-6,
            1.1,
            0.01,
            0.001,
            0.005,
            0.005,
        )
        if observed != expected:
            raise ValueError("Exp411 frozen research parameters changed")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


FROZEN_CONFIG = Exp411FrozenConfig()
