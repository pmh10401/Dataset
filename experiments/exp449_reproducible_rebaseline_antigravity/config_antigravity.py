"""Frozen Exp449 identity, reproducibility limits, and create-only root authority."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


_DATASET_ROOT = Path("/Users/mac/Documents/Dataset")
_APPROVED_ROOTS = (
    _DATASET_ROOT / "paper_proposal_exp449_smoke_a_antigravity_20260814",
    _DATASET_ROOT / "paper_proposal_exp449_smoke_a_antigravity_20260814_ops_antigravity",
    _DATASET_ROOT / "paper_proposal_exp449_smoke_b_antigravity_20260814",
    _DATASET_ROOT / "paper_proposal_exp449_smoke_b_antigravity_20260814_ops_antigravity",
    _DATASET_ROOT / "paper_proposal_exp449_official_antigravity_20260814",
    _DATASET_ROOT / "paper_proposal_exp449_official_antigravity_20260814_ops_antigravity",
)


@dataclass(frozen=True, slots=True)
class FrozenExp449Config:
    """The complete, non-overridable Exp449 configuration surface."""

    experiment_id: str
    dataset_count: int
    test_row_count: int
    policy_id: str
    cpu_repeat_tolerance: float
    accelerated_tolerance: float
    minimum_accelerated_speedup: float
    maximum_waves: int
    preprocessing_contracts: tuple[str, str]
    smoke_datasets: tuple[str, ...]
    smoke_a_scientific_root: Path
    smoke_a_operations_root: Path
    smoke_b_scientific_root: Path
    smoke_b_operations_root: Path
    official_scientific_root: Path
    official_operations_root: Path

    def __post_init__(self) -> None:
        if self.experiment_id != "Exp449" or self.policy_id != "hard_plus_stable":
            raise ValueError("Exp449 frozen identity is invalid")
        if any(type(value) is not int or value < 1 for value in (self.dataset_count, self.test_row_count, self.maximum_waves)):
            raise ValueError("Exp449 frozen counts are invalid")
        if (self.dataset_count, self.test_row_count, self.maximum_waves) != (1117, 195900, 2):
            raise ValueError("Exp449 frozen counts differ from the approved design")
        tolerances = (
            self.cpu_repeat_tolerance,
            self.accelerated_tolerance,
            self.minimum_accelerated_speedup,
        )
        if any(type(value) is not float or not isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("Exp449 frozen numerical limits are invalid")
        if tolerances != (1e-12, 1e-5, 1.1):
            raise ValueError("Exp449 frozen numerical limits differ from the approved design")
        contracts = tuple(self.preprocessing_contracts)
        if contracts != (
            "task4_train_only",
            "task5_label_free_train_test_value_length_transductive",
        ):
            raise ValueError("Exp449 preprocessing contracts differ from the approved design")
        object.__setattr__(self, "preprocessing_contracts", contracts)
        datasets = tuple(self.smoke_datasets)
        if len(datasets) != 23 or datasets != tuple(sorted(datasets)) or len(set(datasets)) != len(datasets):
            raise ValueError("Exp449 smoke datasets must be sorted and unique")
        if any(type(dataset) is not str or not dataset for dataset in datasets):
            raise ValueError("Exp449 smoke datasets are invalid")
        object.__setattr__(self, "smoke_datasets", datasets)
        roots = tuple(Path(root) for root in (
            self.smoke_a_scientific_root,
            self.smoke_a_operations_root,
            self.smoke_b_scientific_root,
            self.smoke_b_operations_root,
            self.official_scientific_root,
            self.official_operations_root,
        ))
        if any(not root.is_absolute() or "antigravity" not in root.name for root in roots):
            raise ValueError("Exp449 roots must be absolute antigravity namespaces")
        if len(set(roots)) != len(roots):
            raise ValueError("Exp449 roots must be independent")
        if roots != _APPROVED_ROOTS:
            raise ValueError("Exp449 roots differ from the approved design")
        for field_name, root in zip(
            (
                "smoke_a_scientific_root",
                "smoke_a_operations_root",
                "smoke_b_scientific_root",
                "smoke_b_operations_root",
                "official_scientific_root",
                "official_operations_root",
            ),
            roots,
            strict=True,
        ):
            object.__setattr__(self, field_name, root)


FROZEN_EXP449_CONFIG = FrozenExp449Config(
    experiment_id="Exp449",
    dataset_count=1117,
    test_row_count=195900,
    policy_id="hard_plus_stable",
    cpu_repeat_tolerance=1e-12,
    accelerated_tolerance=1e-5,
    minimum_accelerated_speedup=1.1,
    maximum_waves=2,
    preprocessing_contracts=(
        "task4_train_only",
        "task5_label_free_train_test_value_length_transductive",
    ),
    smoke_datasets=(
        "ChlorineConcentration_normal_3",
        "DodgerLoopDay_normal_1",
        "DodgerLoopDay_normal_2",
        "DodgerLoopDay_normal_4",
        "DodgerLoopDay_normal_5",
        "DodgerLoopDay_normal_6",
        "DodgerLoopDay_normal_7",
        "DodgerLoopGame_normal_1",
        "DodgerLoopGame_normal_2",
        "DodgerLoopWeekend_normal_1",
        "DodgerLoopWeekend_normal_2",
        "MelbournePedestrian_normal_1",
        "MelbournePedestrian_normal_10",
        "MelbournePedestrian_normal_2",
        "MelbournePedestrian_normal_3",
        "MelbournePedestrian_normal_4",
        "MelbournePedestrian_normal_5",
        "MelbournePedestrian_normal_6",
        "MelbournePedestrian_normal_7",
        "MelbournePedestrian_normal_8",
        "MelbournePedestrian_normal_9",
        "Phoneme_normal_39",
        "PigAirwayPressure_normal_1",
    ),
    smoke_a_scientific_root=_APPROVED_ROOTS[0],
    smoke_a_operations_root=_APPROVED_ROOTS[1],
    smoke_b_scientific_root=_APPROVED_ROOTS[2],
    smoke_b_operations_root=_APPROVED_ROOTS[3],
    official_scientific_root=_APPROVED_ROOTS[4],
    official_operations_root=_APPROVED_ROOTS[5],
)

FROZEN_EXP449_POLICY_KEYS = (
    "hard_only",
    "hard_plus_source",
    "hard_plus_stable",
    "hard_plus_unstable",
)
