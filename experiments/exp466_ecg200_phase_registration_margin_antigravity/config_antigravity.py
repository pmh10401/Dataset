"""Frozen Exp466 ECG200 phase-registration-margin configuration."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


def _exact_antigravity(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(value) == len(expected) and all(  # type: ignore[arg-type]
            _exact_antigravity(item, wanted)
            for item, wanted in zip(value, expected, strict=True)  # type: ignore[arg-type]
        )
    return value == expected


@dataclass(frozen=True, slots=True)
class Exp466ConfigAntigravity:
    experiment: str
    dataset: str
    data_split: str
    train_rows: int
    width: int
    normal_rows: int
    train_md5: str
    interpolated_width: int
    seeds: tuple[int, int]
    folds: int
    onset: int
    suffix_length: int
    warp_factor: float
    shared_length: int
    method_order: tuple[str, str, str]
    primary_candidate: str
    per_type_auc_min: float
    mean_auc_min: float
    required_fold_results: int
    normal_trials_per_seed: int
    label_query_count_must_remain: int
    official_evaluate_count_must_remain: int
    official_inspect_count_must_remain: int
    approved_design_sha256: str

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(
                    f"{name} differs from the frozen Exp466 configuration",
                )

    @property
    def train_shape(self) -> tuple[int, int]:
        return (self.train_rows, self.width)


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "experiment": "Exp466",
        "dataset": "ECG200",
        "data_split": "TRAIN_only",
        "train_rows": 100,
        "width": 96,
        "normal_rows": 69,
        "train_md5": "808875aeac1c358a90c630fe369327c4",
        "interpolated_width": 140,
        "seeds": (20260721, 20261721),
        "folds": 5,
        "onset": 56,
        "suffix_length": 84,
        "warp_factor": 1.25,
        "shared_length": 67,
        "method_order": ("C0", "P9", "C9"),
        "primary_candidate": "C9",
        "per_type_auc_min": 0.85,
        "mean_auc_min": 0.90,
        "required_fold_results": 10,
        "normal_trials_per_seed": 69,
        "label_query_count_must_remain": 0,
        "official_evaluate_count_must_remain": 0,
        "official_inspect_count_must_remain": 0,
        "approved_design_sha256": (
            "eddb057724e0ef21402c07b9f362bc6b260fc917a9f6bfdcb130f118b30a5c66"
        ),
    }
)

FROZEN_CONFIG_ANTIGRAVITY = Exp466ConfigAntigravity(
    **dict(_FROZEN_VALUES_ANTIGRAVITY),
)

__all__ = (
    "Exp466ConfigAntigravity",
    "FROZEN_CONFIG_ANTIGRAVITY",
)
