"""Frozen Exp467 identity, population, policy, source authority, and create-only roots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.exp449_reproducible_rebaseline_antigravity.config_antigravity import (
    FROZEN_EXP449_CONFIG,
)
from paper.equipment_anomaly_detection.contracts_paper import FrozenPaperConfig

from .contracts_antigravity import Exp467Condition


SMOKE_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp467_numba_calibration_resume_smoke_antigravity_20260823"
)
OFFICIAL_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp467_numba_calibration_resume_official_antigravity_20260823"
)
EVALUATION_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp467_numba_calibration_resume_evaluation_antigravity_20260823"
)
REPORT_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp467_numba_calibration_resume_report_antigravity_20260823"
)

EXP450_OFFICIAL_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp450_resampling_image_ablation_official_antigravity_20260820"
)
EXP450_SOURCE_ROOT = EXP450_OFFICIAL_ROOT
EXP450_SOURCE_CONTEXT_SHA256 = (
    "5d5b44aec0e738ae61d9dafd85d9eaea1448baba49705c4d042c5ed1d6886f83"
)
EXP450_SOURCE_TRANSFORMS_SHA256 = (
    "28b5fcf731fc7ed31546939bbbaa932e38770f3f6e77864fc353328ddaf059c7"
)
EXP450_SOURCE_CATALOG_SHA256 = (
    "99aeceb59e98cac160ddc87b252a9be7aca087ba78627847bfbe4005f4b39c4b"
)
EXP450_SOURCE_COMMIT = "eac0d3b4"

APPROVED_DESIGN_PATH_ANTIGRAVITY = Path(
    "docs/superpowers/specs/2026-08-23-exp467-numba-calibration-resume-antigravity-design.md"
)
APPROVED_PLAN_PATH_ANTIGRAVITY = Path(
    "docs/superpowers/plans/2026-08-23-exp467-numba-calibration-resume-antigravity-plan.md"
)

EXP467_TOTAL_COORDINATES = 4468
EXP467_MIGRATED_COORDINATES = 4234
EXP467_REMAINING_COORDINATES = 234
EXP467_MIGRATED_COUNTS_BY_CONDITION = {"A": 1117, "B": 1117, "C": 1000, "D": 1000}
EXP467_FINAL_COUNTS_BY_CONDITION = {"A": 1117, "B": 1117, "C": 1117, "D": 1117}
SMOKE_DATASET_ANTIGRAVITY = "MelbournePedestrian_normal_1"
SMOKE_CONDITION_ANTIGRAVITY = "C"

_APPROVED_CONDITIONS = (
    Exp467Condition("A", "linear", "linear"),
    Exp467Condition("B", "polyphase_pchip", "linear"),
    Exp467Condition("C", "linear", "area"),
    Exp467Condition("D", "polyphase_pchip", "area"),
)
_APPROVED_ROOTS = (SMOKE_ROOT, OFFICIAL_ROOT, EVALUATION_ROOT, REPORT_ROOT)


@dataclass(frozen=True, slots=True)
class FrozenExp467Config:
    """Complete, non-overridable identity for Exp467 verified accelerated resume."""

    experiment_id: str
    database_path: Path
    database_sha256: str
    dataset_count: int
    test_row_count: int
    smoke_datasets: tuple[str, ...]
    policy_id: str
    conditions: tuple[Exp467Condition, ...]
    roots: tuple[Path, ...]
    approved_design_path: Path
    source_root: Path
    source_context_sha256: str
    source_transforms_sha256: str
    source_catalog_sha256: str
    source_commit: str

    def __post_init__(self) -> None:
        paper = FrozenPaperConfig()
        if self.experiment_id != "Exp467":
            raise ValueError("Exp467 frozen identity is invalid")
        if (
            Path(self.database_path) != Path(paper.database_path)
            or self.database_sha256 != paper.database_sha256
        ):
            raise ValueError("Exp467 database identity differs from the frozen paper config")
        if (self.dataset_count, self.test_row_count) != (1117, 195900):
            raise ValueError("Exp467 frozen population differs from the approved design")
        if (self.dataset_count, self.test_row_count) != (
            paper.dataset_count,
            paper.test_row_count,
        ):
            raise ValueError("Exp467 population differs from the frozen paper config")
        smoke_datasets = tuple(self.smoke_datasets)
        if smoke_datasets != FROZEN_EXP449_CONFIG.smoke_datasets:
            raise ValueError("Exp467 smoke inventory differs from frozen Exp449")
        if self.policy_id != "hard_plus_stable":
            raise ValueError("Exp467 policy differs from the approved design")
        conditions = tuple(self.conditions)
        if conditions != _APPROVED_CONDITIONS:
            raise ValueError("Exp467 conditions differ from the approved factorial")
        roots = tuple(Path(root) for root in self.roots)
        if roots != _APPROVED_ROOTS or any(
            not root.is_absolute()
            or "antigravity" not in root.name
            or "exp467" not in str(root).lower()
            for root in roots
        ):
            raise ValueError("Exp467 roots differ from the approved isolated namespaces")
        design_path = Path(self.approved_design_path)
        if design_path != APPROVED_DESIGN_PATH_ANTIGRAVITY:
            raise ValueError("Exp467 design path differs from the approved design")
        if (
            Path(self.source_root) != EXP450_OFFICIAL_ROOT
            or self.source_context_sha256 != EXP450_SOURCE_CONTEXT_SHA256
            or self.source_transforms_sha256 != EXP450_SOURCE_TRANSFORMS_SHA256
            or self.source_catalog_sha256 != EXP450_SOURCE_CATALOG_SHA256
            or self.source_commit != EXP450_SOURCE_COMMIT
        ):
            raise ValueError("Exp467 source authority constants differ from approved Exp450 authority")
        object.__setattr__(self, "database_path", Path(self.database_path))
        object.__setattr__(self, "smoke_datasets", smoke_datasets)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "approved_design_path", design_path)
        object.__setattr__(self, "source_root", Path(self.source_root))


_PAPER_CONFIG = FrozenPaperConfig()
FROZEN_EXP467_CONFIG = FrozenExp467Config(
    experiment_id="Exp467",
    database_path=Path(_PAPER_CONFIG.database_path),
    database_sha256=_PAPER_CONFIG.database_sha256,
    dataset_count=1117,
    test_row_count=195900,
    smoke_datasets=FROZEN_EXP449_CONFIG.smoke_datasets,
    policy_id="hard_plus_stable",
    conditions=_APPROVED_CONDITIONS,
    roots=_APPROVED_ROOTS,
    approved_design_path=APPROVED_DESIGN_PATH_ANTIGRAVITY,
    source_root=EXP450_OFFICIAL_ROOT,
    source_context_sha256=EXP450_SOURCE_CONTEXT_SHA256,
    source_transforms_sha256=EXP450_SOURCE_TRANSFORMS_SHA256,
    source_catalog_sha256=EXP450_SOURCE_CATALOG_SHA256,
    source_commit=EXP450_SOURCE_COMMIT,
)


__all__ = [
    "APPROVED_DESIGN_PATH_ANTIGRAVITY",
    "APPROVED_PLAN_PATH_ANTIGRAVITY",
    "EVALUATION_ROOT",
    "EXP450_OFFICIAL_ROOT",
    "EXP450_SOURCE_CATALOG_SHA256",
    "EXP450_SOURCE_COMMIT",
    "EXP450_SOURCE_CONTEXT_SHA256",
    "EXP450_SOURCE_ROOT",
    "EXP450_SOURCE_TRANSFORMS_SHA256",
    "EXP467_FINAL_COUNTS_BY_CONDITION",
    "EXP467_MIGRATED_COORDINATES",
    "EXP467_MIGRATED_COUNTS_BY_CONDITION",
    "EXP467_REMAINING_COORDINATES",
    "EXP467_TOTAL_COORDINATES",
    "FROZEN_EXP467_CONFIG",
    "FrozenExp467Config",
    "OFFICIAL_ROOT",
    "REPORT_ROOT",
    "SMOKE_CONDITION_ANTIGRAVITY",
    "SMOKE_DATASET_ANTIGRAVITY",
    "SMOKE_ROOT",
]
