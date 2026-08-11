"""Exact, evaluation-only configuration frozen for Exp447."""
from __future__ import annotations

from dataclasses import dataclass


_FROZEN_VALUES_ANTIGRAVITY = {
    "experiment_id": "Exp447",
    "artifact_schema": "exp447",
    "run_identity": "exp447-exp446-postfreeze-performance-evaluation-antigravity-20260811",
    "run_root": "/Users/mac/Documents/Dataset/exp447_exp446_postfreeze_performance_evaluation_antigravity_20260811",
    "dataset_count": 1117,
    "label_rows": 195900,
    "source_count": 30,
    "stable_count": 26,
    "unstable_count": 4,
    "bootstrap_seed": 44720260811,
    "bootstrap_samples": 20000,
    "bootstrap_draw_sha256": "f960dec37c48599a09d95f7a766d160d133eb9b085536974962bb5412dea993e",
    "baseline_confusion": (1915, 303, 2003, 191679),
    "exp404_hard_prediction_sha256": "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
    "exp446_prediction_sha256": "602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100",
    "exp446_manifest_file_sha256": "bbaf9506e17045b90eb9f72c41fad19d0f8d5459084ead329c6441d8af303c11",
    "exp446_artifact_map_file_sha256": "d28129dc6d6ca5097c31c26e44feadcabc422493fd9e2be7abbdaacc403f6d0d",
    "scientific_success_criterion": None,
    "promotion_allowed": False,
    "review_only": True,
    "operational_alert_eligible": False,
    "scientific_winner_selected": False,
    "same_experiment_reselection_allowed": False,
    "test_informed_reselection_allowed": False,
}


@dataclass(frozen=True)
class Exp447FrozenConfigAntigravity:
    """Constants that must not vary across post-freeze evaluation replays."""

    experiment_id: str = "Exp447"
    artifact_schema: str = "exp447"
    run_identity: str = "exp447-exp446-postfreeze-performance-evaluation-antigravity-20260811"
    run_root: str = "/Users/mac/Documents/Dataset/exp447_exp446_postfreeze_performance_evaluation_antigravity_20260811"
    dataset_count: int = 1117
    label_rows: int = 195900
    source_count: int = 30
    stable_count: int = 26
    unstable_count: int = 4
    bootstrap_seed: int = 44720260811
    bootstrap_samples: int = 20000
    bootstrap_draw_sha256: str = "f960dec37c48599a09d95f7a766d160d133eb9b085536974962bb5412dea993e"
    baseline_confusion: tuple[int, int, int, int] = (1915, 303, 2003, 191679)
    exp404_hard_prediction_sha256: str = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"
    exp446_prediction_sha256: str = "602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100"
    exp446_manifest_file_sha256: str = "bbaf9506e17045b90eb9f72c41fad19d0f8d5459084ead329c6441d8af303c11"
    exp446_artifact_map_file_sha256: str = "d28129dc6d6ca5097c31c26e44feadcabc422493fd9e2be7abbdaacc403f6d0d"
    scientific_success_criterion: None = None
    promotion_allowed: bool = False
    review_only: bool = True
    operational_alert_eligible: bool = False
    scientific_winner_selected: bool = False
    same_experiment_reselection_allowed: bool = False
    test_informed_reselection_allowed: bool = False

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            actual = getattr(self, name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f"{name} is frozen")
        if (self.experiment_id, self.artifact_schema, self.run_identity) != (
            "Exp447", "exp447", "exp447-exp446-postfreeze-performance-evaluation-antigravity-20260811",
        ):
            raise ValueError("Exp447 identity is frozen")
        if (self.dataset_count, self.label_rows, self.source_count, self.stable_count, self.unstable_count) != (1117, 195900, 30, 26, 4):
            raise ValueError("Exp447 population is frozen")
        if self.source_count != self.stable_count + self.unstable_count:
            raise ValueError("Exp447 lane population is frozen")
        if (self.bootstrap_seed, self.bootstrap_samples, self.baseline_confusion) != (44720260811, 20000, (1915, 303, 2003, 191679)):
            raise ValueError("Exp447 evaluation statistics are frozen")
        if any(type(value) is not int or value < 0 for value in (*self.baseline_confusion, self.dataset_count, self.label_rows, self.source_count, self.stable_count, self.unstable_count, self.bootstrap_seed, self.bootstrap_samples)):
            raise ValueError("Exp447 counts must be exact non-negative integers")
        hashes = (self.bootstrap_draw_sha256, self.exp404_hard_prediction_sha256, self.exp446_prediction_sha256, self.exp446_manifest_file_sha256, self.exp446_artifact_map_file_sha256)
        if any(type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in hashes):
            raise ValueError("Exp447 source hashes are invalid")
        if self.scientific_success_criterion is not None or self.promotion_allowed or not self.review_only:
            raise ValueError("Exp447 remains descriptive and review-only")
        if self.operational_alert_eligible or self.scientific_winner_selected or self.same_experiment_reselection_allowed or self.test_informed_reselection_allowed:
            raise ValueError("Exp447 promotion is prohibited")


FROZEN_CONFIG_ANTIGRAVITY = Exp447FrozenConfigAntigravity()


__all__ = ["Exp447FrozenConfigAntigravity", "FROZEN_CONFIG_ANTIGRAVITY"]
