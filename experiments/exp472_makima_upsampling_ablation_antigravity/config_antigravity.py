"""Frozen Exp472 population, roots, scoring identity, and backend policy."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
import os
from pathlib import Path

from experiments.exp450_resampling_image_ablation_antigravity.config_antigravity import (
    FROZEN_EXP450_CONFIG,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import (
    EXP471_ROOT_ANTIGRAVITY,
)
from paper.equipment_anomaly_detection.contracts_paper import FrozenPaperConfig

from .contracts_antigravity import (
    Exp472BackendDecisionAntigravity,
    Exp472ConditionAntigravity,
    MLX_FALLBACK_REASON_ANTIGRAVITY,
)


EXP472_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp472_makima_upsampling_ablation_recovery_antigravity_20260831"
)
EXP472_SMOKE_ROOT_ANTIGRAVITY = EXP472_ROOT_ANTIGRAVITY / "smoke_antigravity"
EXP472_OFFICIAL_ROOT_ANTIGRAVITY = EXP472_ROOT_ANTIGRAVITY / "official_antigravity"
EXP472_EVALUATION_ROOT_ANTIGRAVITY = (
    EXP472_ROOT_ANTIGRAVITY / "evaluation_antigravity"
)
EXP472_REPORT_ROOT_ANTIGRAVITY = EXP472_ROOT_ANTIGRAVITY / "report_antigravity"
EXP471_REUSE_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp471_exp467_evaluation_recovery_antigravity_20260824"
)
EXP467_VERIFICATION_REPOSITORY_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Time Series/.worktrees/"
    "exp471-exp467-evaluation-recovery-antigravity"
)

APPROVED_DESIGN_PATH_ANTIGRAVITY = Path(
    "docs/superpowers/specs/"
    "2026-08-28-exp472-makima-upsampling-ablation-antigravity-design.md"
)
APPROVED_PLAN_PATH_ANTIGRAVITY = Path(
    "docs/superpowers/plans/"
    "2026-08-28-exp472-makima-upsampling-ablation-antigravity.md"
)
BOOTSTRAP_SEED_ANTIGRAVITY = 47220260828
BOOTSTRAP_DRAWS_ANTIGRAVITY = 20_000
BOOTSTRAP_INTERVAL_METHOD_ANTIGRAVITY = "percentile"
BOOTSTRAP_CONFIDENCE_LEVEL_ANTIGRAVITY = 0.95
SCORE_THRESHOLD_ANTIGRAVITY = 0.5
TIE_TOLERANCE_ANTIGRAVITY = 1e-12
POLICY_ID_ANTIGRAVITY = "hard_plus_stable"
CONDITION_E_ANTIGRAVITY = Exp472ConditionAntigravity()
METHOD_ORDER_ANTIGRAVITY = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "knn_distance",
    "lof_novelty",
    "ocsvm_rbf",
    "isolation_forest",
    "deep_svdd_1d",
    "drocc_1d",
)

LOGICAL_CPU_COUNT_ANTIGRAVITY = os.cpu_count()
if (
    type(LOGICAL_CPU_COUNT_ANTIGRAVITY) is not int
    or LOGICAL_CPU_COUNT_ANTIGRAVITY < 1
):
    raise RuntimeError("Exp472 cannot detect a positive logical CPU count")
CPU_WORKERS_ANTIGRAVITY = max(1, LOGICAL_CPU_COUNT_ANTIGRAVITY - 1)
INNER_NUMERICAL_THREADS_ANTIGRAVITY = 1

_ROOTS_ANTIGRAVITY = (
    EXP472_SMOKE_ROOT_ANTIGRAVITY,
    EXP472_OFFICIAL_ROOT_ANTIGRAVITY,
    EXP472_EVALUATION_ROOT_ANTIGRAVITY,
    EXP472_REPORT_ROOT_ANTIGRAVITY,
)


@dataclass(frozen=True, slots=True)
class FrozenExp472ConfigAntigravity:
    """Complete non-overridable identity for the isolated Makima ablation."""

    experiment_id: str
    database_path: Path
    database_sha256: str
    dataset_count: int
    test_row_count: int
    smoke_datasets: tuple[str, ...]
    condition: Exp472ConditionAntigravity
    policy_id: str
    score_threshold: float
    roots: tuple[Path, ...]
    exp471_reuse_root: Path
    exp467_verification_repository: Path
    bootstrap_seed: int
    bootstrap_draws: int
    bootstrap_interval_method: str
    bootstrap_confidence_level: float
    tie_tolerance: float
    approved_design_path: Path
    approved_plan_path: Path

    def __post_init__(self) -> None:
        paper = FrozenPaperConfig()
        if self.experiment_id != "Exp472":
            raise ValueError("Exp472 frozen experiment identity is invalid")
        database_path = Path(self.database_path)
        if (
            database_path != Path(paper.database_path)
            or self.database_sha256 != paper.database_sha256
        ):
            raise ValueError("Exp472 database differs from the frozen paper identity")
        if (self.dataset_count, self.test_row_count) != (1117, 195900) or (
            self.dataset_count,
            self.test_row_count,
        ) != (paper.dataset_count, paper.test_row_count):
            raise ValueError("Exp472 population differs from the approved design")
        smoke_datasets = tuple(self.smoke_datasets)
        if smoke_datasets != FROZEN_EXP450_CONFIG.smoke_datasets:
            raise ValueError("Exp472 smoke inventory differs from frozen Exp450")
        if self.condition != CONDITION_E_ANTIGRAVITY:
            raise ValueError("Exp472 condition differs from the approved design")
        if self.policy_id != POLICY_ID_ANTIGRAVITY or self.score_threshold != 0.5:
            raise ValueError("Exp472 score policy differs from the approved design")
        roots = tuple(Path(root) for root in self.roots)
        if roots != _ROOTS_ANTIGRAVITY or any(
            not root.is_absolute()
            or root.parent != EXP472_ROOT_ANTIGRAVITY
            or "antigravity" not in root.name
            for root in roots
        ):
            raise ValueError("Exp472 roots differ from the isolated namespaces")
        reuse_root = Path(self.exp471_reuse_root)
        if (
            reuse_root != EXP471_REUSE_ROOT_ANTIGRAVITY
            or reuse_root != EXP471_ROOT_ANTIGRAVITY
        ):
            raise ValueError("Exp472 Exp471 reuse root differs from frozen authority")
        verification_repository = Path(self.exp467_verification_repository)
        if verification_repository != EXP467_VERIFICATION_REPOSITORY_ANTIGRAVITY:
            raise ValueError("Exp472 Exp467 verification repository differs")
        if (
            self.bootstrap_seed != BOOTSTRAP_SEED_ANTIGRAVITY
            or self.bootstrap_draws != BOOTSTRAP_DRAWS_ANTIGRAVITY
            or self.bootstrap_interval_method
            != BOOTSTRAP_INTERVAL_METHOD_ANTIGRAVITY
            or self.bootstrap_confidence_level
            != BOOTSTRAP_CONFIDENCE_LEVEL_ANTIGRAVITY
            or self.tie_tolerance != TIE_TOLERANCE_ANTIGRAVITY
        ):
            raise ValueError("Exp472 paired-analysis policy differs")
        design_path = Path(self.approved_design_path)
        plan_path = Path(self.approved_plan_path)
        if (
            design_path != APPROVED_DESIGN_PATH_ANTIGRAVITY
            or plan_path != APPROVED_PLAN_PATH_ANTIGRAVITY
        ):
            raise ValueError("Exp472 governance document identity differs")
        object.__setattr__(self, "database_path", database_path)
        object.__setattr__(self, "smoke_datasets", smoke_datasets)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "exp471_reuse_root", reuse_root)
        object.__setattr__(
            self, "exp467_verification_repository", verification_repository
        )
        object.__setattr__(self, "approved_design_path", design_path)
        object.__setattr__(self, "approved_plan_path", plan_path)


def _package_version_antigravity(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def assess_exp472_backend_antigravity() -> Exp472BackendDecisionAntigravity:
    """Observe readiness and freeze MPS-first area-image backend provenance."""
    pytorch_version = _package_version_antigravity("torch")
    mps_available = False
    if pytorch_version != "unavailable":
        try:
            torch = importlib.import_module("torch")
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            mps_available = bool(
                mps is not None
                and callable(getattr(mps, "is_available", None))
                and mps.is_available()
            )
        except Exception:
            mps_available = False
    scipy_version = _package_version_antigravity("scipy")
    mlx_version = _package_version_antigravity("mlx")
    mlx_metal_version = _package_version_antigravity("mlx-metal")
    return Exp472BackendDecisionAntigravity(
        pytorch_available=pytorch_version != "unavailable",
        pytorch_version=pytorch_version,
        scipy_available=scipy_version != "unavailable",
        scipy_version=scipy_version,
        mlx_available=mlx_version != "unavailable",
        mlx_version=mlx_version,
        mlx_metal_available=mlx_metal_version != "unavailable",
        mlx_metal_version=mlx_metal_version,
        mps_available=mps_available,
        tensor_backend="mps" if mps_available else "cpu",
        logical_cpu_count=LOGICAL_CPU_COUNT_ANTIGRAVITY,
        cpu_workers=CPU_WORKERS_ANTIGRAVITY,
        inner_numerical_threads=INNER_NUMERICAL_THREADS_ANTIGRAVITY,
        mlx_fallback_reason=MLX_FALLBACK_REASON_ANTIGRAVITY,
    )


_PAPER_CONFIG_ANTIGRAVITY = FrozenPaperConfig()
FROZEN_EXP472_CONFIG = FrozenExp472ConfigAntigravity(
    experiment_id="Exp472",
    database_path=Path(_PAPER_CONFIG_ANTIGRAVITY.database_path),
    database_sha256=_PAPER_CONFIG_ANTIGRAVITY.database_sha256,
    dataset_count=1117,
    test_row_count=195900,
    smoke_datasets=FROZEN_EXP450_CONFIG.smoke_datasets,
    condition=CONDITION_E_ANTIGRAVITY,
    policy_id=POLICY_ID_ANTIGRAVITY,
    score_threshold=SCORE_THRESHOLD_ANTIGRAVITY,
    roots=_ROOTS_ANTIGRAVITY,
    exp471_reuse_root=EXP471_REUSE_ROOT_ANTIGRAVITY,
    exp467_verification_repository=EXP467_VERIFICATION_REPOSITORY_ANTIGRAVITY,
    bootstrap_seed=BOOTSTRAP_SEED_ANTIGRAVITY,
    bootstrap_draws=BOOTSTRAP_DRAWS_ANTIGRAVITY,
    bootstrap_interval_method=BOOTSTRAP_INTERVAL_METHOD_ANTIGRAVITY,
    bootstrap_confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL_ANTIGRAVITY,
    tie_tolerance=TIE_TOLERANCE_ANTIGRAVITY,
    approved_design_path=APPROVED_DESIGN_PATH_ANTIGRAVITY,
    approved_plan_path=APPROVED_PLAN_PATH_ANTIGRAVITY,
)


__all__ = [
    "APPROVED_DESIGN_PATH_ANTIGRAVITY",
    "APPROVED_PLAN_PATH_ANTIGRAVITY",
    "BOOTSTRAP_CONFIDENCE_LEVEL_ANTIGRAVITY",
    "BOOTSTRAP_DRAWS_ANTIGRAVITY",
    "BOOTSTRAP_INTERVAL_METHOD_ANTIGRAVITY",
    "BOOTSTRAP_SEED_ANTIGRAVITY",
    "CONDITION_E_ANTIGRAVITY",
    "CPU_WORKERS_ANTIGRAVITY",
    "EXP467_VERIFICATION_REPOSITORY_ANTIGRAVITY",
    "EXP471_REUSE_ROOT_ANTIGRAVITY",
    "EXP472_EVALUATION_ROOT_ANTIGRAVITY",
    "EXP472_OFFICIAL_ROOT_ANTIGRAVITY",
    "EXP472_REPORT_ROOT_ANTIGRAVITY",
    "EXP472_ROOT_ANTIGRAVITY",
    "EXP472_SMOKE_ROOT_ANTIGRAVITY",
    "FROZEN_EXP472_CONFIG",
    "FrozenExp472ConfigAntigravity",
    "INNER_NUMERICAL_THREADS_ANTIGRAVITY",
    "LOGICAL_CPU_COUNT_ANTIGRAVITY",
    "METHOD_ORDER_ANTIGRAVITY",
    "MLX_FALLBACK_REASON_ANTIGRAVITY",
    "POLICY_ID_ANTIGRAVITY",
    "SCORE_THRESHOLD_ANTIGRAVITY",
    "TIE_TOLERANCE_ANTIGRAVITY",
    "assess_exp472_backend_antigravity",
]
