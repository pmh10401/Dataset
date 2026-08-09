"""Frozen scientific and runtime configuration for Exp443.

The values in this module are deliberately boring: later tasks may consume the
configuration but may not derive new policy, threshold, or routing choices from
runtime input.  ``FROZEN_CONFIG`` is the single reviewed source of truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


_DATASET_ROOT = Path("/Users/mac/Documents/Dataset")
_RUN_ROOT = _DATASET_ROOT / "exp443_missing_normal_train_policy_comparison_antigravity_20260809"
_ENV_ROOT = _DATASET_ROOT / "venvs/exp443_minirocket_antigravity_py314"

FROZEN_INPUT_SHA256_ANTIGRAVITY: dict[str, str] = {
    "database": "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
    "exp249_prediction": "fc72d2dfbaf6a1653ec804dbe349397214992387083b22bdab0e8c26a66876c2",
    "exp249_prediction_manifest": "25ade7c3f8362559a29c15e43fead6070a7fac6cab648f03f62bcb20382b8c8f",
    "exp249_artifact_manifest": "42fee6f9a26b2c6fd19747f83887a431bf5edad144e2af3464033256a0108da9",
    "exp249_cache_manifest": "44c0ecc80ff7cbae383d49d2b8b0cefed442c49b8a12a7bd0cf8569ce121c971",
    "exp249_audit": "144ae3f65fea3b4a8cf306fd54b097788c64666dc2d21b673fbf65500c12e0ee",
    "exp291_routing": "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
    "exp291_manifest": "761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e",
    "exp414_repair_audit": "548f493a1608521bc81e4b742cc061fab7298f0d03c69f23845e37528b26ab91",
    "exp414_manifest": "99556922dc8f5178feaf03bc61eca4bfae628ae8f6b6092c2f6159da8c5dd3cc",
    "exp404_hard": "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
    "exp404_manifest": "41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8",
}

_POLICIES = (
    "linear_edge_nearest_control",
    "exclude_nonfinite_row",
    "leave_one_out_peer_median",
)
_SHA256_FIELDS = {
    "reviewed_design_sha256",
    "database_sha256",
    "exp249_prediction_sha256",
    "exp249_prediction_manifest_sha256",
    "exp249_artifact_manifest_sha256",
    "exp249_cache_manifest_sha256",
    "exp249_audit_sha256",
    "exp291_routing_sha256",
    "exp291_manifest_sha256",
    "exp414_repair_audit_sha256",
    "exp414_manifest_sha256",
    "exp404_hard_sha256",
    "exp404_manifest_sha256",
    "environment_lock_sha256",
}


@dataclass(frozen=True)
class Exp443FrozenConfig:
    """Every literal that changes the Exp443 scientific or runtime identity."""

    experiment_id: str = "Exp443"
    run_identity: str = "exp443-missing-normal-train-policy-comparison-antigravity-20260809"
    reviewed_design_sha256: str = "b52142a0a48f9c557a56180ee96dc1a7a8e46760a15a547aa92f8b5999808f9b"
    dataset_count: int = 1117
    affected_dataset_count: int = 19
    affected_family_count: int = 4
    affected_normal_train_rows: int = 164
    nonfinite_scalar_count: int = 983
    all_missing_row_count: int = 0
    recompute_units: int = 57
    policies: tuple[str, ...] = _POLICIES
    database_path: str = str(_DATASET_ROOT / "univariate_ts.db")
    database_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["database"]
    exp249_prediction_path: str = str(_DATASET_ROOT / "exp249_250_official_mini_multirocket/experiment_249_official_minirocket/frozen_predictions.jsonl")
    exp249_prediction_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp249_prediction"]
    exp249_prediction_manifest_path: str = str(_DATASET_ROOT / "exp249_250_official_mini_multirocket/experiment_249_official_minirocket/prediction_manifest.json")
    exp249_prediction_manifest_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp249_prediction_manifest"]
    exp249_artifact_manifest_path: str = str(_DATASET_ROOT / "exp249_250_official_mini_multirocket/experiment_249_official_minirocket/MANIFEST.json")
    exp249_artifact_manifest_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp249_artifact_manifest"]
    exp249_cache_manifest_path: str = str(_DATASET_ROOT / "exp249_250_official_mini_multirocket/experiment_249_official_minirocket/cache_manifest.csv")
    exp249_cache_manifest_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp249_cache_manifest"]
    exp249_audit_path: str = str(_DATASET_ROOT / "exp249_250_official_mini_multirocket/experiment_249_official_minirocket/audit.json")
    exp249_audit_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp249_audit"]
    exp291_routing_path: str = str(_DATASET_ROOT / "exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl")
    exp291_routing_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp291_routing"]
    exp291_manifest_path: str = str(_DATASET_ROOT / "exp291_final_operational_routing_freeze/exp291_prediction_manifest.json")
    exp291_manifest_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp291_manifest"]
    exp414_repair_audit_path: str = str(_DATASET_ROOT / "exp414_missing_aware_near_repeat/exp414_missing_repair_audit.csv")
    exp414_repair_audit_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp414_repair_audit"]
    exp414_manifest_path: str = str(_DATASET_ROOT / "exp414_missing_aware_near_repeat/MANIFEST.json")
    exp414_manifest_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp414_manifest"]
    exp404_hard_path: str = str(_DATASET_ROOT / "exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl")
    exp404_hard_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp404_hard"]
    exp404_manifest_path: str = str(_DATASET_ROOT / "exp403_408_feature_ml_validation/exp404_prediction_manifest.json")
    exp404_manifest_sha256: str = FROZEN_INPUT_SHA256_ANTIGRAVITY["exp404_manifest"]
    run_root: str = str(_RUN_ROOT)
    isolated_environment_root: str = str(_ENV_ROOT)
    environment_lock_path: str = str(Path(__file__).with_name("requirements_lock_antigravity.txt"))
    # Bound to the reviewed lock bytes after the environment bootstrap step.
    environment_lock_sha256: str = "bb2de6e2696a3f0d92480f0215b15bdaf6d89375210dd4be48257cfa6b44d548"
    kernel_requests: tuple[int, ...] = (1024, 4096)
    kernel_dimensions: tuple[int, ...] = (1008, 4032)
    max_dilations: int = 32
    seeds: tuple[int, ...] = (20260721, 20261721)
    alpha: float = 0.005
    folds: int = 5
    knn: int = 3
    max_reference: int = 1000
    query_chunk: int = 128
    robust_quantile_range: tuple[int, int] = (10, 90)
    target_length_clip: tuple[int, int] = (8, 2048)
    outer_workers: int = 11
    inner_threads: int = 1
    logical_cpu_count: int = 12
    python_version: str = "3.14.6"
    aeon_version: str = "1.5.0"
    numpy_version: str = "2.3.5"
    scikit_learn_version: str = "1.8.0"
    bootstrap_seed: int = 44320260809
    bootstrap_samples: int = 20000
    bootstrap_percentiles: tuple[float, float] = (2.5, 97.5)
    bootstrap_endpoint: bool = False
    baseline_tp: int = 1915
    baseline_fp: int = 303
    baseline_fn: int = 2003
    baseline_tn: int = 191679
    official_run_purpose: str = "controlled_missing_normal_train_policy_comparison"
    adaptive_followup: bool = True
    independent_validation: bool = False
    external_generalization_supported: bool = False
    confirmatory_statistical_claim: bool = False
    scientific_winner_selected: bool = False
    review_only: bool = True
    operational_alert_eligible: bool = False
    test_informed_reselection_allowed: bool = False
    acceleration_decision: str = "inapplicable_official_cpu_transformer"

    def __post_init__(self) -> None:
        # ``frozen=True`` prevents normal mutation; these checks also reject
        # dataclasses.replace() with a changed scientific literal.
        if type(self.experiment_id) is not str or self.experiment_id != "Exp443":
            raise ValueError("Exp443 experiment identity is frozen")
        if type(self.run_identity) is not str or not self.run_identity.endswith("antigravity-20260809"):
            raise ValueError("Exp443 run identity is frozen")
        if type(self.policies) is not tuple or self.policies != _POLICIES:
            raise ValueError("Exp443 policy order is frozen")
        if self.kernel_requests != (1024, 4096) or self.kernel_dimensions != (1008, 4032):
            raise ValueError("Exp443 kernels are frozen")
        if self.seeds != (20260721, 20261721) or self.max_dilations != 32:
            raise ValueError("Exp443 transformer identity is frozen")
        if self.alpha != 0.005 or self.folds != 5 or self.knn != 3 or self.max_reference != 1000 or self.query_chunk != 128:
            raise ValueError("Exp443 score contract is frozen")
        if self.robust_quantile_range != (10, 90) or self.target_length_clip != (8, 2048):
            raise ValueError("Exp443 scaling and alignment are frozen")
        if self.outer_workers != 11 or self.inner_threads != 1 or self.logical_cpu_count != 12:
            raise ValueError("Exp443 CPU topology is frozen")
        if (self.python_version, self.aeon_version, self.numpy_version, self.scikit_learn_version) != ("3.14.6", "1.5.0", "2.3.5", "1.8.0"):
            raise ValueError("Exp443 environment versions are frozen")
        if self.bootstrap_seed != 44320260809 or self.bootstrap_samples != 20000 or self.bootstrap_percentiles != (2.5, 97.5) or self.bootstrap_endpoint is not False:
            raise ValueError("Exp443 bootstrap contract is frozen")
        if not self.review_only or self.operational_alert_eligible or self.scientific_winner_selected:
            raise ValueError("Exp443 governance is review-only")
        if self.acceleration_decision != "inapplicable_official_cpu_transformer":
            raise ValueError("Exp443 acceleration decision is frozen")
        for name in _SHA256_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or (name != "environment_lock_sha256" and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value))):
                raise ValueError(f"{name} must be a lowercase SHA256")
        if len(self.environment_lock_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.environment_lock_sha256):
            raise ValueError("Exp443 environment lock SHA is not bound")

    def validate(self) -> None:
        """Compatibility method used by neighbouring antigravity configs."""
        validate_antigravity(self)


# The lock digest is patched immediately after the isolated environment is
# resolved.  Keeping the pending value guarded above prevents accidental use.
FROZEN_CONFIG = Exp443FrozenConfig()


def cpu_worker_count_antigravity(logical_cpus: int | None = None) -> int:
    """Return the exact outer-worker rule, with no caller override in production."""
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
    """Return a detached JSON-safe payload with tuples represented as lists."""
    return dict(_json_safe_antigravity(asdict(FROZEN_CONFIG)))


def governance_payload_antigravity() -> dict[str, object]:
    """Return the exact flat governance block used by all later artifacts."""
    names = (
        "official_run_purpose", "adaptive_followup", "independent_validation",
        "external_generalization_supported", "confirmatory_statistical_claim",
        "scientific_winner_selected", "review_only", "operational_alert_eligible",
        "test_informed_reselection_allowed",
    )
    return {name: getattr(FROZEN_CONFIG, name) for name in names}


def validate_antigravity(config: Exp443FrozenConfig | None = None) -> bool:
    """Fail closed when any frozen field differs from the import-time contract."""
    candidate = FROZEN_CONFIG if config is None else config
    if not isinstance(candidate, Exp443FrozenConfig):
        raise ValueError("Exp443 configuration has the wrong type")
    payload = _json_safe_antigravity(asdict(candidate))
    if payload != _CANONICAL_PAYLOAD_ANTIGRAVITY:
        raise ValueError("Exp443 configuration is mutated")
    if scientific_config_sha256_antigravity() != _CANONICAL_SHA256_ANTIGRAVITY:
        raise ValueError("Exp443 configuration identity changed")
    return True


def scientific_config_sha256_antigravity() -> str:
    """Hash the canonical JSON bytes of the frozen config (without a newline)."""
    encoded = json.dumps(frozen_config_payload_antigravity(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_CANONICAL_PAYLOAD_ANTIGRAVITY = _json_safe_antigravity(asdict(FROZEN_CONFIG))
_CANONICAL_SHA256_ANTIGRAVITY = scientific_config_sha256_antigravity()


__all__ = [
    "Exp443FrozenConfig", "FROZEN_CONFIG", "FROZEN_INPUT_SHA256_ANTIGRAVITY",
    "cpu_worker_count_antigravity", "frozen_config_payload_antigravity",
    "governance_payload_antigravity", "scientific_config_sha256_antigravity",
    "validate_antigravity",
]
