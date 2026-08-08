"""Frozen, label-free configuration for Exp441."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path


_DATASET_ROOT = Path("/Users/mac/Documents/Dataset")
_EXP440_ROOT = _DATASET_ROOT / "exp440_train_only_low_frequency_precision_gate_antigravity_20260808"
_EXP440_CHECKPOINT = _EXP440_ROOT / "checkpoint_antigravity"
_EXP440_PUBLICATION = _EXP440_ROOT / "publication_antigravity"
_HARD_ROOT = _DATASET_ROOT / "exp403_408_feature_ml_validation"
_PUBLICATION_FILENAMES = (
    "abstention_ledger_antigravity.jsonl",
    "artifact_sha256_antigravity.json",
    "backend_decision_antigravity.json",
    "dataset_summary_antigravity.jsonl",
    "depth_evidence_antigravity.jsonl",
    "manifest_antigravity.json",
    "prediction_antigravity.jsonl",
    "retained_candidate_ranking_antigravity.jsonl",
    "status_antigravity.json",
    "train_exclusion_audit_antigravity.jsonl",
)


@dataclass(frozen=True)
class Exp441FrozenConfig:
    """All literals that define the reviewed Exp441 replay contract."""

    experiment_id: str = "Exp441"
    run_identity: str = "exp441-hard-empty-single-shot-recovery-antigravity-20260808"
    reviewed_design_sha256: str = "c4d692297290869be90815d115c1fda7f71120326b4be504ebf685619b6545da"
    dataset_count: int = 1117
    candidate_budget: int = 30
    candidate_per_dataset: int = 1
    bootstrap_seed: int = 44120260808
    bootstrap_samples: int = 20000
    numpy_version: str = "2.4.6"
    tensor_eligible: bool = False
    official_run_purpose: str = "deterministic_retrospective_reproduction"
    bootstrap_interpretation: str = "descriptive_post_selection"
    adaptive_followup: bool = True
    independent_validation: bool = False
    external_generalization_supported: bool = False
    confirmatory_statistical_claim: bool = False
    review_only: bool = True
    operational_alert_eligible: bool = False
    scientific_winner_selected: bool = False
    database_path: str = "/Users/mac/Documents/Dataset/univariate_ts.db"
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    checkpoint_context_path: str = "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity/context_antigravity.json"
    checkpoint_context_sha256: str = "d598a159d798f0f1dbdcbf684ef7994c89c982b60bce62b877a82c7e04ba7dfc"
    checkpoint_state_path: str = "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity/state_antigravity.json"
    checkpoint_state_sha256: str = "486235b70856247912a02c115748e22477ea11224d3a50de088187903da56268"
    checkpoint_records_path: str = "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity/records_antigravity"
    checkpoint_retained_inventory_path: str = "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity/retained_candidates_antigravity.json"
    checkpoint_retained_inventory_sha256: str = "c31c0414dced3f463d187e8e17adf84a50eb0c7cf86a6e8399f9074fda94ccf2"
    publication_path: str = "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/publication_antigravity"
    publication_prediction_sha256: str = "06d905d9d6f96fa67704111b984ed87b74229e13fd0690401db67a4a0fdd79af"
    publication_manifest_sha256: str = "3778a0e8770e06f4be87d1b87281c858518169c377ee6c53b78f32ce01d39e85"
    publication_artifact_map_sha256: str = "7a083fa8f1287205d3469e36f2f88b95f72adecd90b3b79167aad673b3439a1a"
    hard_prediction_path: str = "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl"
    hard_prediction_sha256: str = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"
    hard_manifest_path: str = "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json"
    hard_manifest_sha256: str = "41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8"
    hard_coordinate_identity_sha256: str = "858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322"
    selected_coordinate_identity_sha256: str = "db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048"

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if type(getattr(self, name)) is not type(field.default) or getattr(self, name) != field.default:
                raise ValueError("Exp441 configuration is frozen")


FROZEN_CONFIG = Exp441FrozenConfig()


def _allowlisted_prediction_paths_antigravity() -> frozenset[Path]:
    """Return the closed set of resolved label-free source paths."""
    checkpoint = {
        _EXP440_CHECKPOINT / "context_antigravity.json",
        _EXP440_CHECKPOINT / "state_antigravity.json",
        _EXP440_CHECKPOINT / "retained_candidates_antigravity.json",
    }
    records = {
        _EXP440_CHECKPOINT / "records_antigravity" / f"{index:04d}_record_antigravity.json"
        for index in range(1117)
    }
    publication = {_EXP440_PUBLICATION / name for name in _PUBLICATION_FILENAMES}
    hard = {
        _HARD_ROOT / "exp404_hard_confidence.jsonl",
        _HARD_ROOT / "exp404_prediction_manifest.json",
    }
    return frozenset(path.resolve(strict=False) for path in checkpoint | records | publication | hard)


def _require_allowlisted_prediction_path_antigravity(path: Path) -> Path:
    """Reject every resolved prediction source outside the immutable allowlist."""
    resolved = Path(path).resolve(strict=False)
    if resolved in _allowlisted_prediction_paths_antigravity():
        return resolved
    raise ValueError("prediction source is outside the positive allowlist")


def cpu_worker_count_antigravity(logical_cpus: int | None = None) -> int:
    """Apply Exp441's fixed max-minus-one CPU policy."""
    count = os.cpu_count() if logical_cpus is None else logical_cpus
    if type(count) is not int or count < 1:
        raise ValueError("logical CPU count must be a positive integer")
    return max(1, count - 1)


def frozen_config_payload_antigravity() -> dict[str, object]:
    """Return a JSON-safe copy of the frozen configuration, with no overrides."""
    return asdict(FROZEN_CONFIG)


def governance_payload_antigravity() -> dict[str, object]:
    """Return precisely the nine flat retrospective-governance fields."""
    names = (
        "official_run_purpose", "bootstrap_interpretation", "adaptive_followup",
        "independent_validation", "external_generalization_supported",
        "confirmatory_statistical_claim", "scientific_winner_selected", "review_only",
        "operational_alert_eligible",
    )
    return {name: getattr(FROZEN_CONFIG, name) for name in names}


__all__ = [
    "Exp441FrozenConfig", "FROZEN_CONFIG", "cpu_worker_count_antigravity",
    "frozen_config_payload_antigravity", "governance_payload_antigravity",
]
