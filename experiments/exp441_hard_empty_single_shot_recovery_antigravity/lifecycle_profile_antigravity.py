"""Frozen lifecycle identities shared by Exp441 and its Exp442 engineering repair."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from .artifacts_antigravity import canonical_json_bytes_antigravity
from .config_antigravity import FROZEN_CONFIG


_SCIENTIFIC_CONFIG_SHA256_ANTIGRAVITY = "d6086725b68aaf3de13c006aa817faa9daa80257a36965729935f1590b389ffb"


def shared_scientific_config_sha256_antigravity() -> str:
    """Hash every frozen scientific field while excluding lifecycle identity only."""
    payload = asdict(FROZEN_CONFIG)
    for name in ("experiment_id", "run_identity", "reviewed_design_sha256"):
        payload.pop(name)
    digest = sha256(canonical_json_bytes_antigravity(payload)).hexdigest()
    if digest != _SCIENTIFIC_CONFIG_SHA256_ANTIGRAVITY:
        raise ValueError("shared scientific configuration changed")
    return digest


@dataclass(frozen=True)
class RecoveryLifecycleProfileAntigravity:
    """Closed lifecycle identity; it cannot override any scientific setting."""

    experiment_id: str
    schema_namespace: str
    run_identity: str
    run_root: str
    reviewed_preregistration_sha256: str
    reviewed_design_sha256: str
    cli_program: str
    scientific_config_sha256: str = _SCIENTIFIC_CONFIG_SHA256_ANTIGRAVITY

    def __post_init__(self) -> None:
        expected = {
            "Exp441": (
                "exp441", "exp441-hard-empty-single-shot-recovery-antigravity-20260808",
                "/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808",
                "a4d440e56d057fd8605cffed623307d9e40bf6640ebb65f12a495e41d877cc61",
                "c4d692297290869be90815d115c1fda7f71120326b4be504ebf685619b6545da",
                "run_exp441_hard_empty_single_shot_recovery_antigravity",
            ),
            "Exp442": (
                "exp442", "exp442-terminal-eta-publication-repair-antigravity-20260808",
                "/Users/mac/Documents/Dataset/exp442_terminal_eta_publication_repair_antigravity_20260808",
                "ab909c35d0f4d0252d1a29fa80ea27d51d0ba56175c8cc0d8d3428092ec2746c",
                "cf0b6e916a85cdf21c8cc28c735894f927b63e4c33aa4d62e95753540bd06a7b",
                "run_exp442_terminal_eta_publication_repair_antigravity",
            ),
        }
        actual = (
            self.schema_namespace, self.run_identity, self.run_root,
            self.reviewed_preregistration_sha256, self.reviewed_design_sha256,
            self.cli_program,
        )
        if self.experiment_id not in expected or actual != expected[self.experiment_id]:
            raise ValueError("recovery lifecycle profile identity is invalid")
        if self.scientific_config_sha256 != shared_scientific_config_sha256_antigravity():
            raise ValueError("recovery lifecycle scientific binding is invalid")
        root = Path(self.run_root)
        if not root.is_absolute() or "antigravity" not in root.name:
            raise ValueError("recovery lifecycle run root is invalid")

    @property
    def root_antigravity(self) -> Path:
        return Path(self.run_root)

    def schema_antigravity(self, artifact: str) -> str:
        if not isinstance(artifact, str) or not artifact or any(character not in "abcdefghijklmnopqrstuvwxyz_-" for character in artifact):
            raise ValueError("lifecycle schema artifact is invalid")
        return f"{self.schema_namespace}.{artifact}.v1"

    def identity_payload_antigravity(self) -> dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "schema_namespace": self.schema_namespace,
            "run_identity": self.run_identity,
            "run_root": self.run_root,
            "reviewed_preregistration_sha256": self.reviewed_preregistration_sha256,
            "reviewed_design_sha256": self.reviewed_design_sha256,
            "scientific_config_sha256": self.scientific_config_sha256,
            "cli_program": self.cli_program,
        }

    @property
    def identity_sha256_antigravity(self) -> str:
        return sha256(canonical_json_bytes_antigravity(self.identity_payload_antigravity())).hexdigest()


EXP441_PROFILE_ANTIGRAVITY = RecoveryLifecycleProfileAntigravity(
    experiment_id="Exp441",
    schema_namespace="exp441",
    run_identity="exp441-hard-empty-single-shot-recovery-antigravity-20260808",
    run_root="/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808",
    reviewed_preregistration_sha256="a4d440e56d057fd8605cffed623307d9e40bf6640ebb65f12a495e41d877cc61",
    reviewed_design_sha256="c4d692297290869be90815d115c1fda7f71120326b4be504ebf685619b6545da",
    cli_program="run_exp441_hard_empty_single_shot_recovery_antigravity",
)

EXP442_PROFILE_ANTIGRAVITY = RecoveryLifecycleProfileAntigravity(
    experiment_id="Exp442",
    schema_namespace="exp442",
    run_identity="exp442-terminal-eta-publication-repair-antigravity-20260808",
    run_root="/Users/mac/Documents/Dataset/exp442_terminal_eta_publication_repair_antigravity_20260808",
    reviewed_preregistration_sha256="ab909c35d0f4d0252d1a29fa80ea27d51d0ba56175c8cc0d8d3428092ec2746c",
    reviewed_design_sha256="cf0b6e916a85cdf21c8cc28c735894f927b63e4c33aa4d62e95753540bd06a7b",
    cli_program="run_exp442_terminal_eta_publication_repair_antigravity",
)


__all__ = [
    "RecoveryLifecycleProfileAntigravity", "EXP441_PROFILE_ANTIGRAVITY",
    "EXP442_PROFILE_ANTIGRAVITY", "shared_scientific_config_sha256_antigravity",
]
