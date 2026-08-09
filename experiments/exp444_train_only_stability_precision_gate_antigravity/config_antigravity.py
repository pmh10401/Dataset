"""Frozen scientific and runtime configuration for Exp444.

This module is intentionally dependency-free.  It is imported before NumPy by
the environment and runtime entry points, so all experiment identity and
thread-policy literals live here and cannot be supplied by a CLI override.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


_DATASET_ROOT = Path("/Users/mac/Documents/Dataset")
_RUN_ROOT = _DATASET_ROOT / "exp444_train_only_stability_precision_gate_antigravity_20260810"
_ENV_ROOT = _DATASET_ROOT / "venvs/exp444_stability_antigravity_py314"
_PACKAGE_REQUIREMENTS = ("mlx==0.32.0", "numba==0.66.0", "numpy==2.4.6")

FROZEN_INPUT_SHA256_ANTIGRAVITY: dict[str, str] = {
    "database": "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
    "exp404_hard_prediction": "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
    "exp442_prediction": "b79bad6012ad6bc07050ade9968e815796a2cc871b5c85d63f0bd7dbdf6f81f4",
    "exp442_candidate_ranking": "d8d858cd74553aa64f6f0a5424064872b59f682fed1c8b2ace63fffba6ad20cc",
    "exp442_selection": "06004f18dbd9b0df3fa857bb8b1a9fd7eafd4d927f3736fb8cfb25db56b72408",
    "exp442_selected_coordinate_identity": "db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048",
}

_SHA256_FIELDS = {
    "reviewed_design_sha256",
    "database_sha256",
    "exp404_hard_prediction_sha256",
    "exp442_prediction_sha256",
    "exp442_candidate_ranking_sha256",
    "exp442_selection_sha256",
    "exp442_selected_coordinate_identity_sha256",
    "environment_lock_sha256",
}

_INNER_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


@dataclass(frozen=True)
class Exp444FrozenConfig:
    """Every literal that contributes to Exp444 scientific or run identity."""

    experiment_id: str = "Exp444"
    run_identity: str = "exp444-train-only-stability-precision-gate-antigravity-20260810"
    reviewed_design_sha256: str = "1bb76f19d07ef312d5ac5498414d0c914ce94b99c5912e0bbcf3f0197c1400e8"
    dataset_count: int = 1117
    source_candidate_count: int = 30
    candidate_per_dataset: int = 1
    maximum_stability_views: int = 16
    stability_fraction: float = 0.875
    mask_seed: int = 44420260810
    bootstrap_seed: int = 44420260811
    bootstrap_samples: int = 20000
    acceleration_tolerance: float = 1e-5
    acceleration_speedup_min: float = 1.1
    logical_cpu_count: int = 12
    expected_cpu_workers_on_current_mac: int = 11
    cpu_workers: int = 11
    inner_numeric_threads: int = 1
    inner_threads: int = 1
    python_version: str = "3.14.6"
    mlx_version: str = "0.32.0"
    numba_version: str = "0.66.0"
    numpy_version: str = "2.4.6"
    database_path: str = str(_DATASET_ROOT / "univariate_ts.db")
    database_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["database"]
    exp404_hard_prediction_path: str = str(_DATASET_ROOT / "exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl")
    exp404_hard_prediction_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp404_hard_prediction"]
    exp442_prediction_path: str = str(_DATASET_ROOT / "exp442_terminal_eta_publication_repair_antigravity_20260809/prediction_antigravity.jsonl")
    exp442_prediction_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp442_prediction"]
    exp442_candidate_ranking_path: str = str(_DATASET_ROOT / "exp442_terminal_eta_publication_repair_antigravity_20260809/candidate_ranking_antigravity.jsonl")
    exp442_candidate_ranking_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp442_candidate_ranking"]
    exp442_selection_path: str = str(_DATASET_ROOT / "exp442_terminal_eta_publication_repair_antigravity_20260809/selection_antigravity.json")
    exp442_selection_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp442_selection"]
    exp442_selected_coordinate_identity_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp442_selected_coordinate_identity"]
    run_root: str = str(_RUN_ROOT)
    isolated_environment_root: str = str(_ENV_ROOT)
    environment_lock_path: str = str(Path(__file__).with_name("requirements_lock_antigravity.txt"))
    # Bound to the exact sorted ``pip freeze --all`` bytes produced by the
    # isolated bootstrap; any lock or environment drift fails closed.
    environment_lock_sha256: str = "8835702b9793a5cc08b5e95621b49c0a4a4136325354da7a655761f6b91f3a94"
    requirements: tuple[str, ...] = _PACKAGE_REQUIREMENTS
    omit_fraction: float = 0.10
    minimum_train_rows: int = 4
    nearest_k: int = 3
    quantile: float = 0.99
    quantile_method: str = "higher"
    comparison: str = "strict_greater"
    adjacent_offsets: tuple[int, ...] = (-1, 0, 1)
    minimum_consensus_depths: int = 2
    robust_scale_floor: float = 1e-12
    official_run_purpose: str = "train_only_stability_review"
    scientific_success_criterion: None = None
    evaluation_status: str = "descriptive_complete_no_promotion"
    adaptive_followup: bool = True
    independent_validation: bool = False
    external_generalization_supported: bool = False
    confirmatory_statistical_claim: bool = False
    scientific_winner_selected: bool = False
    review_only: bool = True
    operational_alert_eligible: bool = False
    same_experiment_reselection_allowed: bool = False
    test_informed_reselection_allowed: bool = False
    acceleration_decision: str = "cpu_until_parity_speed_gate"

    def __post_init__(self) -> None:
        if type(self.experiment_id) is not str or self.experiment_id != "Exp444":
            raise ValueError("Exp444 experiment identity is frozen")
        if type(self.run_identity) is not str or self.run_identity != "exp444-train-only-stability-precision-gate-antigravity-20260810":
            raise ValueError("Exp444 run identity is frozen")
        if self.source_candidate_count != 30 or self.candidate_per_dataset != 1 or self.maximum_stability_views != 16:
            raise ValueError("Exp444 candidate/view budget is frozen")
        if self.dataset_count != 1117 or self.stability_fraction != 0.875 or self.mask_seed != 44420260810:
            raise ValueError("Exp444 population/mask contract is frozen")
        if self.bootstrap_seed != 44420260811 or self.bootstrap_samples != 20000:
            raise ValueError("Exp444 bootstrap contract is frozen")
        if self.acceleration_tolerance != 1e-5 or self.acceleration_speedup_min != 1.1:
            raise ValueError("Exp444 acceleration gate is frozen")
        if self.logical_cpu_count != 12 or self.expected_cpu_workers_on_current_mac != 11 or self.cpu_workers != 11:
            raise ValueError("Exp444 CPU topology is frozen")
        if self.inner_numeric_threads != 1 or self.inner_threads != 1:
            raise ValueError("Exp444 inner thread count is frozen")
        if (self.python_version, self.mlx_version, self.numba_version, self.numpy_version) != ("3.14.6", "0.32.0", "0.66.0", "2.4.6"):
            raise ValueError("Exp444 environment versions are frozen")
        if self.requirements != _PACKAGE_REQUIREMENTS:
            raise ValueError("Exp444 direct requirements are frozen")
        if self.omit_fraction != 0.10 or self.minimum_train_rows != 4:
            raise ValueError("Exp444 omission contract is frozen")
        if (self.nearest_k, self.quantile, self.quantile_method, self.comparison, self.adjacent_offsets, self.minimum_consensus_depths) != (3, 0.99, "higher", "strict_greater", (-1, 0, 1), 2):
            raise ValueError("Exp440 score contract is frozen")
        if not self.review_only or self.operational_alert_eligible or self.scientific_winner_selected:
            raise ValueError("Exp444 is review-only")
        if self.scientific_success_criterion is not None or self.evaluation_status != "descriptive_complete_no_promotion":
            raise ValueError("Exp444 descriptive governance is frozen")
        if not self.adaptive_followup or self.independent_validation or self.same_experiment_reselection_allowed or self.test_informed_reselection_allowed:
            raise ValueError("Exp444 governance is invalid")
        for name in _SHA256_FIELDS:
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA256")
        for name in ("run_root", "isolated_environment_root", "environment_lock_path"):
            if type(getattr(self, name)) is not str or "antigravity" not in getattr(self, name):
                raise ValueError(f"{name} must remain in the antigravity namespace")

    def validate(self) -> None:
        validate_antigravity(self)


FROZEN_CONFIG = Exp444FrozenConfig()


def cpu_worker_count_antigravity(logical_cpus: int | None = None) -> int:
    count = os.cpu_count() if logical_cpus is None else logical_cpus
    if type(count) is not int or count < 1:
        raise ValueError("logical CPU count must be a positive exact integer")
    return max(1, count - 1)


def _json_safe_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_antigravity(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_antigravity(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def frozen_config_payload_antigravity() -> dict[str, object]:
    """Return a detached canonical-JSON-safe copy of the frozen config."""
    return dict(_json_safe_antigravity(asdict(FROZEN_CONFIG)))


def scientific_config_sha256_antigravity() -> str:
    encoded = json.dumps(frozen_config_payload_antigravity(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_CANONICAL_PAYLOAD_ANTIGRAVITY = frozen_config_payload_antigravity()
_CANONICAL_SHA256_ANTIGRAVITY = scientific_config_sha256_antigravity()


def validate_antigravity(config: Exp444FrozenConfig | None = None) -> bool:
    candidate = FROZEN_CONFIG if config is None else config
    if not isinstance(candidate, Exp444FrozenConfig):
        raise ValueError("Exp444 configuration has the wrong type")
    candidate_payload = _json_safe_antigravity(asdict(candidate))
    if candidate_payload != _CANONICAL_PAYLOAD_ANTIGRAVITY:
        raise ValueError("Exp444 configuration is mutated")
    encoded = json.dumps(candidate_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != _CANONICAL_SHA256_ANTIGRAVITY:
        raise ValueError("Exp444 configuration identity changed")
    return True


def inner_thread_environment_names_antigravity() -> tuple[str, ...]:
    return _INNER_THREAD_ENV_NAMES


__all__ = [
    "Exp444FrozenConfig", "FROZEN_CONFIG", "FROZEN_INPUT_SHA256_ANTIGRAVITY",
    "cpu_worker_count_antigravity", "frozen_config_payload_antigravity",
    "scientific_config_sha256_antigravity", "validate_antigravity",
    "inner_thread_environment_names_antigravity",
]
