"""Frozen Task 2 scientific inputs for Exp460."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType


def _exact_antigravity(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is tuple:
        return len(value) == len(expected) and all(  # type: ignore[arg-type]
            _exact_antigravity(item, wanted)
            for item, wanted in zip(value, expected, strict=True)  # type: ignore[arg-type]
        )
    return value == expected


@dataclass(frozen=True, slots=True)
class Exp460ConfigAntigravity:
    experiment: str
    kind: str
    dataset: str
    rule_order: tuple[str, str, str, str]
    official_decision_rule: str
    strict_failure_route: str
    hydra_seeds: tuple[int, int]
    folds: int
    period_change_auc_min: float
    baseline_confusion: tuple[int, int, int, int]
    candidate_confusion: tuple[int, int, int, int]
    exp459_frozen_authority_digest: str
    exp459_official_artifact_map_sha256: str
    exp459_decision_sha256: str
    label_query_count: int
    runtime_executable: str

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(f"{name} differs from the frozen Exp460 configuration")
        if not isfinite(self.period_change_auc_min):
            raise ValueError("period_change_auc_min must be finite")


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "experiment": "Exp460",
        "kind": "ecg_strict_fallback",
        "dataset": "ECG5000_normal_2",
        "rule_order": (
            "always_baseline",
            "always_candidate",
            "majority_8_of_10",
            "strict_10_of_10",
        ),
        "official_decision_rule": "strict_10_of_10_only",
        "strict_failure_route": "baseline",
        "hydra_seeds": (20260721, 20261721),
        "folds": 5,
        "period_change_auc_min": 0.85,
        "baseline_confusion": (609, 66, 837, 70788),
        "candidate_confusion": (617, 66, 829, 70788),
        "exp459_frozen_authority_digest": "f661c4ce6c7fd6625ac5cf767db688ce7ede7a0d0557721c9e9fa7e7ea6508ee",
        "exp459_official_artifact_map_sha256": "a30fbe7ed34f88c1ab9b7fd40963edebbf1a4127d8d64c99141224cb2dbb106d",
        "exp459_decision_sha256": "527318cacdbc63905ab58237d793803eb286192456eb007362e6bed11c8f8716",
        "label_query_count": 0,
        "runtime_executable": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
    }
)


FROZEN_CONFIG_ANTIGRAVITY = Exp460ConfigAntigravity(**_FROZEN_VALUES_ANTIGRAVITY)  # type: ignore[arg-type]


__all__ = ("Exp460ConfigAntigravity", "FROZEN_CONFIG_ANTIGRAVITY")
