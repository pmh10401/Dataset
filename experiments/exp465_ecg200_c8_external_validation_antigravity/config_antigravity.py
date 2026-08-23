"""Frozen Exp465 ECG200 C8 external-validation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping


def _exact_antigravity(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, MappingProxyType):
        if tuple(value) != tuple(expected):  # type: ignore[arg-type]
            return False
        return all(  # type: ignore[arg-type]
            _exact_antigravity(value[key], expected[key])  # type: ignore[index]
            for key in expected
        )
    if isinstance(expected, tuple):
        return len(value) == len(expected) and all(  # type: ignore[arg-type]
            _exact_antigravity(item, wanted)
            for item, wanted in zip(value, expected, strict=True)  # type: ignore[arg-type]
        )
    return value == expected


def _unit_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class Exp465ConfigAntigravity:
    experiment: str
    kind: str
    run_identity: str
    dataset: str
    data_split: str
    train_rows: int
    width: int
    normal_rows: int
    infarct_rows: int
    train_md5: str
    interpolated_width: int
    seeds: tuple[int, int]
    folds: int
    onset: int
    suffix_length: int
    cosine_min: float
    method_order: tuple[str, str, str]
    method_definitions: Mapping[str, str]
    primary_candidate: str
    per_type_auc_min: float
    mean_auc_min: float
    required_stability_records: int
    normal_trials_per_seed: int
    normal_confirmation_count_must_remain: int
    label_query_count_must_remain: int
    official_evaluate_count_must_remain: int
    official_inspect_count_must_remain: int
    approved_design_sha256: str
    predecessor_experiment: str
    predecessor_terminal_closeout_commit: str

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(
                    f"{name} differs from the frozen Exp465 configuration",
                )

    @property
    def train_shape(self) -> tuple[int, int]:
        return (self.train_rows, self.width)

    @property
    def stability_record_count(self) -> int:
        return len(self.seeds) * self.folds


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "experiment": "Exp465",
        "kind": "ecg200_c8_external_train_qualification",
        "run_identity": (
            "exp465-ecg200-c8-external-validation-antigravity-20260822"
        ),
        "dataset": "ECG200",
        "data_split": "TRAIN_only",
        "train_rows": 100,
        "width": 96,
        "normal_rows": 69,
        "infarct_rows": 31,
        "train_md5": "808875aeac1c358a90c630fe369327c4",
        "interpolated_width": 140,
        "seeds": (20260721, 20261721),
        "folds": 5,
        "onset": 56,
        "suffix_length": 84,
        "cosine_min": 0.95,
        "method_order": ("C0", "C6", "C8"),
        "method_definitions": MappingProxyType(
            {
                "C0": "Exp464 C0 exact replay control",
                "C6": "Exp464 C6 exact diagnostic contrast only",
                "C8": "Exp464 C8 exact sole primary candidate",
            }
        ),
        "primary_candidate": "C8",
        "per_type_auc_min": 0.85,
        "mean_auc_min": 0.90,
        "required_stability_records": 10,
        "normal_trials_per_seed": 69,
        "normal_confirmation_count_must_remain": 0,
        "label_query_count_must_remain": 0,
        "official_evaluate_count_must_remain": 0,
        "official_inspect_count_must_remain": 0,
        "approved_design_sha256": (
            "a5f5f43e1787ad2942c883761ee95478e8cf5fcdcc958722a22fe0e1ad98d904"
        ),
        "predecessor_experiment": "Exp464",
        "predecessor_terminal_closeout_commit": (
            "1fdc8422996fe7e85325987f7543e00db79e22d9"
        ),
    }
)

FROZEN_CONFIG_ANTIGRAVITY = Exp465ConfigAntigravity(
    **dict(_FROZEN_VALUES_ANTIGRAVITY),
)

__all__ = (
    "Exp465ConfigAntigravity",
    "FROZEN_CONFIG_ANTIGRAVITY",
)
