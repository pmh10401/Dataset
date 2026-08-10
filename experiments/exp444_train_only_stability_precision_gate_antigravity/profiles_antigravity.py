"""Closed Exp444/Exp445 lifecycle identities with no caller-supplied routing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LifecycleProfileAntigravity:
    """One of the two reviewed lifecycle identities, never a mutable selector."""

    experiment_id: str
    schema: str
    run_identity: str
    run_root: Path
    reviewed_design_path: Path
    preregistration_path: Path
    predecessor_required: bool

    def __post_init__(self) -> None:
        exp444 = (
            "Exp444",
            "exp444",
            "exp444-train-only-stability-precision-gate-antigravity-20260810",
            Path("/Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810"),
            _REPOSITORY_ROOT_ANTIGRAVITY / "docs/superpowers/specs/2026-08-10-exp444-train-only-stability-precision-gate-antigravity-design.md",
            _REPOSITORY_ROOT_ANTIGRAVITY / "docs/EXP444_PREREGISTRATION_antigravity_KO.md",
            False,
        )
        exp445 = (
            "Exp445",
            "exp445",
            "exp445-exp444-publication-source-authority-repair-antigravity-20260810",
            Path("/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810"),
            _REPOSITORY_ROOT_ANTIGRAVITY / "docs/superpowers/specs/2026-08-10-exp445-exp444-publication-source-authority-repair-antigravity-design.md",
            _REPOSITORY_ROOT_ANTIGRAVITY / "docs/EXP445_PREREGISTRATION_antigravity_KO.md",
            True,
        )
        actual = (
            self.experiment_id,
            self.schema,
            self.run_identity,
            self.run_root,
            self.reviewed_design_path,
            self.preregistration_path,
            self.predecessor_required,
        )
        if actual not in (exp444, exp445):
            raise ValueError("lifecycle profile is not one of the two reviewed identities")


EXP444_PROFILE_ANTIGRAVITY = LifecycleProfileAntigravity(
    experiment_id="Exp444",
    schema="exp444",
    run_identity="exp444-train-only-stability-precision-gate-antigravity-20260810",
    run_root=Path("/Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810"),
    reviewed_design_path=_REPOSITORY_ROOT_ANTIGRAVITY / "docs/superpowers/specs/2026-08-10-exp444-train-only-stability-precision-gate-antigravity-design.md",
    preregistration_path=_REPOSITORY_ROOT_ANTIGRAVITY / "docs/EXP444_PREREGISTRATION_antigravity_KO.md",
    predecessor_required=False,
)

EXP445_PROFILE_ANTIGRAVITY = LifecycleProfileAntigravity(
    experiment_id="Exp445",
    schema="exp445",
    run_identity="exp445-exp444-publication-source-authority-repair-antigravity-20260810",
    run_root=Path("/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810"),
    reviewed_design_path=_REPOSITORY_ROOT_ANTIGRAVITY / "docs/superpowers/specs/2026-08-10-exp445-exp444-publication-source-authority-repair-antigravity-design.md",
    preregistration_path=_REPOSITORY_ROOT_ANTIGRAVITY / "docs/EXP445_PREREGISTRATION_antigravity_KO.md",
    predecessor_required=True,
)


def _require_exp444_profile_antigravity(profile: object) -> LifecycleProfileAntigravity:
    """Reject equal-looking caller-minted values; Exp444 may use its singleton only."""
    if profile is not EXP444_PROFILE_ANTIGRAVITY:
        raise ValueError("Exp444 lifecycle profile is invalid")
    return EXP444_PROFILE_ANTIGRAVITY


def _require_exp445_profile_antigravity(profile: object) -> LifecycleProfileAntigravity:
    """Reserve Exp445's singleton for its later wrapper without exposing selection."""
    if profile is not EXP445_PROFILE_ANTIGRAVITY:
        raise ValueError("Exp445 lifecycle profile is invalid")
    return EXP445_PROFILE_ANTIGRAVITY


__all__ = [
    "LifecycleProfileAntigravity",
    "EXP444_PROFILE_ANTIGRAVITY",
    "EXP445_PROFILE_ANTIGRAVITY",
]
