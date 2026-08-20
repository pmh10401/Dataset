"""Frozen Exp450 identity, population, policy, and create-only root authority."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.exp449_reproducible_rebaseline_antigravity.config_antigravity import (
    FROZEN_EXP449_CONFIG,
)
from paper.equipment_anomaly_detection.contracts_paper import FrozenPaperConfig

from .contracts_antigravity import Exp450Condition


SMOKE_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp450_resampling_image_ablation_smoke_antigravity_20260820"
)
OFFICIAL_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp450_resampling_image_ablation_official_antigravity_20260820"
)
EVALUATION_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp450_resampling_image_ablation_evaluation_antigravity_20260820"
)
REPORT_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp450_resampling_image_ablation_report_antigravity_20260820"
)
APPROVED_DESIGN_PATH_ANTIGRAVITY = Path(
    "docs/superpowers/specs/2026-08-20-exp450-resampling-image-ablation-antigravity-design.md"
)
_APPROVED_CONDITIONS = (
    Exp450Condition("A", "linear", "linear"),
    Exp450Condition("B", "polyphase_pchip", "linear"),
    Exp450Condition("C", "linear", "area"),
    Exp450Condition("D", "polyphase_pchip", "area"),
)
_APPROVED_ROOTS = (SMOKE_ROOT, OFFICIAL_ROOT, EVALUATION_ROOT, REPORT_ROOT)


@dataclass(frozen=True, slots=True)
class FrozenExp450Config:
    """Complete, non-overridable identity for the four-condition ablation."""

    experiment_id: str
    database_path: Path
    database_sha256: str
    dataset_count: int
    test_row_count: int
    smoke_datasets: tuple[str, ...]
    policy_id: str
    conditions: tuple[Exp450Condition, ...]
    roots: tuple[Path, ...]
    approved_design_path: Path

    def __post_init__(self) -> None:
        paper = FrozenPaperConfig()
        if self.experiment_id != "Exp450":
            raise ValueError("Exp450 frozen identity is invalid")
        if (
            Path(self.database_path) != Path(paper.database_path)
            or self.database_sha256 != paper.database_sha256
        ):
            raise ValueError("Exp450 database identity differs from the frozen paper config")
        if (self.dataset_count, self.test_row_count) != (1117, 195900):
            raise ValueError("Exp450 frozen population differs from the approved design")
        if (self.dataset_count, self.test_row_count) != (
            paper.dataset_count,
            paper.test_row_count,
        ):
            raise ValueError("Exp450 population differs from the frozen paper config")
        smoke_datasets = tuple(self.smoke_datasets)
        if smoke_datasets != FROZEN_EXP449_CONFIG.smoke_datasets:
            raise ValueError("Exp450 smoke inventory differs from frozen Exp449")
        if self.policy_id != "hard_plus_stable":
            raise ValueError("Exp450 policy differs from the approved design")
        conditions = tuple(self.conditions)
        if conditions != _APPROVED_CONDITIONS:
            raise ValueError("Exp450 conditions differ from the approved factorial")
        roots = tuple(Path(root) for root in self.roots)
        if roots != _APPROVED_ROOTS or any(
            not root.is_absolute() or "antigravity" not in root.name for root in roots
        ):
            raise ValueError("Exp450 roots differ from the approved isolated namespaces")
        design_path = Path(self.approved_design_path)
        if design_path != APPROVED_DESIGN_PATH_ANTIGRAVITY:
            raise ValueError("Exp450 design path differs from the approved design")
        object.__setattr__(self, "database_path", Path(self.database_path))
        object.__setattr__(self, "smoke_datasets", smoke_datasets)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "approved_design_path", design_path)


_PAPER_CONFIG = FrozenPaperConfig()
FROZEN_EXP450_CONFIG = FrozenExp450Config(
    experiment_id="Exp450",
    database_path=Path(_PAPER_CONFIG.database_path),
    database_sha256=_PAPER_CONFIG.database_sha256,
    dataset_count=1117,
    test_row_count=195900,
    smoke_datasets=FROZEN_EXP449_CONFIG.smoke_datasets,
    policy_id="hard_plus_stable",
    conditions=_APPROVED_CONDITIONS,
    roots=_APPROVED_ROOTS,
    approved_design_path=APPROVED_DESIGN_PATH_ANTIGRAVITY,
)


__all__ = [
    "APPROVED_DESIGN_PATH_ANTIGRAVITY",
    "EVALUATION_ROOT",
    "FROZEN_EXP450_CONFIG",
    "FrozenExp450Config",
    "OFFICIAL_ROOT",
    "REPORT_ROOT",
    "SMOKE_ROOT",
]
