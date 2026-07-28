from dataclasses import asdict, dataclass

from experiments.exp411_near_repeat_qualification.config import (
    FROZEN_CONFIG as EXP411_FROZEN_CONFIG,
)


@dataclass(frozen=True)
class Exp414FrozenConfig:
    experiment_id: str = "Exp414"
    exp411_source_commit: str = "5c4268fb89ac806dac6bdccd31f5724ee520c63a"
    database_sha256: str = (
        "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    )
    population_manifest_sha256: str = (
        "41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8"
    )
    expected_datasets: int = 1117
    expected_families: int = 128
    expected_missing_datasets: int = 19
    expected_missing_runs: int = 164
    expected_nonfinite_values: int = 983
    expected_all_missing_runs: int = 0
    affected_datasets_sha256: str = (
        "ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9"
    )
    repair_internal_method: str = "linear_interpolation"
    repair_edge_method: str = "nearest_finite_extension"
    repair_output_dtype: str = "<f4"
    preserve_run_length: bool = True
    all_missing_status: str = "missing_support_abstain"
    prohibit_tail_replacement: bool = True
    prohibit_length_interpolation: bool = True
    prohibit_padding: bool = True
    prohibit_truncation: bool = True
    prohibit_all_missing_zero_replacement: bool = True
    prohibit_all_missing_run_removal: bool = True
    train_only: bool = True
    prohibit_test_split_access: bool = True
    prohibit_labels_blob_access: bool = True
    prohibit_actual_anomaly_position_access: bool = True
    prohibit_historical_test_performance_access: bool = True
    perturbation_source_sha256: str = (
        "03f93e4e38fbce11e5a7b9a62354c1b8b43df5dec6444141e92d3944bada25a3"
    )
    perturbation_config_contract_sha256: str = (
        "aadbbe42ae66479135e4c740def20b7a0d289090b2e298a5ae978552a9c65fc3"
    )
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.10)
    quantile_method: str = "lower"
    seed: int = 411
    bootstrap_iterations: int = 2000
    minimum_identity_groups: int = 5
    five_fold_minimum_groups: int = 20
    minimum_same_length_fit_groups: int = 5
    distance_views: tuple[str, ...] = ("raw_rms", "diff_rms", "spectrum_rms")
    mad_floor: float = 1e-12
    distance_tolerance: float = 1e-6
    speedup_gate: float = 1.1
    minimum_normal_acceptance: float = 0.01
    pooled_synthetic_false_normal_max: float = 0.001
    per_perturbation_false_normal_max: float = 0.005
    bootstrap_upper_max: float = 0.005
    numpy_is_numerical_oracle: bool = True
    numba_threads: int = 12
    nested_numerical_threads: int = 1
    require_numpy_numba_tag_parity: bool = True
    require_numpy_numba_selected_quantile_parity: bool = True
    mlx_enabled: bool = False
    pytorch_enabled: bool = False
    mps_enabled: bool = False
    exp411_checkpoint_read_only: bool = True
    separate_exp414_checkpoint_and_output: bool = True
    exp412_started: bool = False
    exp413_started: bool = False
    post_qualification_prediction_experiment: str = "Exp415"
    post_prediction_audit_experiment: str = "Exp416"
    full_run_requires_implementation_review_benchmark: bool = True
    full_run_requires_explicit_user_approval: bool = True

    def validate(self) -> None:
        EXP411_FROZEN_CONFIG.validate()
        if self != Exp414FrozenConfig():
            raise ValueError("Exp414 frozen config changed")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


FROZEN_CONFIG = Exp414FrozenConfig()
