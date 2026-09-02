"""Frozen Exp473 identity for the contemporary same-environment PCHIP control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
    Exp450Condition,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.config_antigravity import (
    FROZEN_EXP472_CONFIG,
)


EXP473_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp473_contemporary_pchip_control_antigravity_20260901"
)
EXP473_SMOKE_ROOT_ANTIGRAVITY = EXP473_ROOT_ANTIGRAVITY / "smoke_antigravity"
EXP473_OFFICIAL_ROOT_ANTIGRAVITY = EXP473_ROOT_ANTIGRAVITY / "official_antigravity"
APPROVED_DESIGN_PATH_ANTIGRAVITY = Path(
    "docs/EXP472_D_PRIME_FOLLOWUP_DESIGN_SOL_HIGH_antigravity_KO.md"
)
P0_SMOKE_DATASETS_EXP473_ANTIGRAVITY = (
    "AllGestureWiimoteX_normal_1",
    "AllGestureWiimoteY_normal_1",
    "AllGestureWiimoteZ_normal_1",
    "CricketY_normal_3",
    "FaceFour_normal_4",
    "GestureMidAirD1_normal_1",
    "GestureMidAirD2_normal_1",
    "GestureMidAirD3_normal_1",
    "GesturePebbleZ1_normal_1",
    "GesturePebbleZ2_normal_1",
    "MelbournePedestrian_normal_2",
    "PLAID_normal_0",
    "PickupGestureWiimoteZ_normal_1",
    "ShakeGestureWiimoteZ_normal_1",
    "Yoga_normal_2",
)


@dataclass(frozen=True, slots=True)
class Exp473ConditionAntigravity:
    """Separate execution identity for the unchanged scientific condition D."""

    control_id: str = "D_prime"
    condition_id: str = "D"
    length_mode: str = "polyphase_pchip"
    image_mode: str = "area"

    def __post_init__(self) -> None:
        if (
            self.control_id,
            self.condition_id,
            self.length_mode,
            self.image_mode,
        ) != ("D_prime", "D", "polyphase_pchip", "area"):
            raise ValueError("Exp473 condition differs from the approved design")

    def as_exp450_condition_antigravity(self) -> Exp450Condition:
        return Exp450Condition(
            self.condition_id, self.length_mode, self.image_mode
        )


CONDITION_D_PRIME_EXP473_ANTIGRAVITY = Exp473ConditionAntigravity()


@dataclass(frozen=True, slots=True)
class FrozenExp473ConfigAntigravity:
    experiment_id: str
    database_path: Path
    database_sha256: str
    dataset_count: int
    test_row_count: int
    smoke_datasets: tuple[str, ...]
    condition: Exp473ConditionAntigravity
    policy_id: str
    score_threshold: float
    roots: tuple[Path, Path]
    approved_design_path: Path

    def __post_init__(self) -> None:
        if self.experiment_id != "Exp473":
            raise ValueError("Exp473 frozen experiment identity is invalid")
        if (
            Path(self.database_path) != FROZEN_EXP472_CONFIG.database_path
            or self.database_sha256 != FROZEN_EXP472_CONFIG.database_sha256
            or (self.dataset_count, self.test_row_count) != (1117, 195900)
            or tuple(self.smoke_datasets) != P0_SMOKE_DATASETS_EXP473_ANTIGRAVITY
        ):
            raise ValueError("Exp473 population differs from frozen Exp472")
        if self.condition != CONDITION_D_PRIME_EXP473_ANTIGRAVITY:
            raise ValueError("Exp473 condition differs from the approved design")
        if self.policy_id != "hard_plus_stable" or self.score_threshold != 0.5:
            raise ValueError("Exp473 scoring policy differs from the approved design")
        roots = tuple(Path(root) for root in self.roots)
        if roots != (
            EXP473_SMOKE_ROOT_ANTIGRAVITY,
            EXP473_OFFICIAL_ROOT_ANTIGRAVITY,
        ) or any(
            not root.is_absolute()
            or root.parent != EXP473_ROOT_ANTIGRAVITY
            or "antigravity" not in root.name
            for root in roots
        ):
            raise ValueError("Exp473 roots differ from isolated namespaces")
        if Path(self.approved_design_path) != APPROVED_DESIGN_PATH_ANTIGRAVITY:
            raise ValueError("Exp473 design document identity differs")
        object.__setattr__(self, "database_path", Path(self.database_path))
        object.__setattr__(self, "smoke_datasets", tuple(self.smoke_datasets))
        object.__setattr__(self, "roots", roots)
        object.__setattr__(
            self, "approved_design_path", Path(self.approved_design_path)
        )


FROZEN_EXP473_CONFIG_ANTIGRAVITY = FrozenExp473ConfigAntigravity(
    experiment_id="Exp473",
    database_path=FROZEN_EXP472_CONFIG.database_path,
    database_sha256=FROZEN_EXP472_CONFIG.database_sha256,
    dataset_count=1117,
    test_row_count=195900,
    smoke_datasets=P0_SMOKE_DATASETS_EXP473_ANTIGRAVITY,
    condition=CONDITION_D_PRIME_EXP473_ANTIGRAVITY,
    policy_id="hard_plus_stable",
    score_threshold=0.5,
    roots=(EXP473_SMOKE_ROOT_ANTIGRAVITY, EXP473_OFFICIAL_ROOT_ANTIGRAVITY),
    approved_design_path=APPROVED_DESIGN_PATH_ANTIGRAVITY,
)


__all__ = [
    "APPROVED_DESIGN_PATH_ANTIGRAVITY",
    "CONDITION_D_PRIME_EXP473_ANTIGRAVITY",
    "EXP473_OFFICIAL_ROOT_ANTIGRAVITY",
    "EXP473_ROOT_ANTIGRAVITY",
    "EXP473_SMOKE_ROOT_ANTIGRAVITY",
    "Exp473ConditionAntigravity",
    "FROZEN_EXP473_CONFIG_ANTIGRAVITY",
    "FrozenExp473ConfigAntigravity",
    "P0_SMOKE_DATASETS_EXP473_ANTIGRAVITY",
]
