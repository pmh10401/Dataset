"""Bind Exp466 C0 to the reviewed Exp465 scoring authority."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.contracts_antigravity import (
    Exp463AuthorityAntigravity,
    validate_exp463_authority_antigravity,
)
from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
    capture_exp463_authority_antigravity,
)
from experiments.exp465_ecg200_c8_external_validation_antigravity.scoring_antigravity import (
    score_fold_methods_antigravity as score_exp465_fold_methods_antigravity,
)


_EXP465_CLOSEOUT_COMMIT_ANTIGRAVITY = "23528af929da9dbf57cbcdf87c06abd58cb2363b"
_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


@dataclass(frozen=True, slots=True)
class Exp465AuthorityAntigravity:
    closeout_commit: str
    score_source: Callable[..., object]
    predecessor_authority: Exp463AuthorityAntigravity

    def __post_init__(self) -> None:
        if self.closeout_commit != _EXP465_CLOSEOUT_COMMIT_ANTIGRAVITY:
            raise ValueError("closeout_commit differs from reviewed Exp465 authority")
        if self.score_source is not score_exp465_fold_methods_antigravity:
            raise ValueError("score_source must be the exact Exp465 fold scorer")
        validate_exp463_authority_antigravity(self.predecessor_authority)


def _readonly_vector_antigravity(values: object, name: str) -> np.ndarray:
    if type(values) is not np.ndarray:
        raise ValueError(f"{name} must be an exact ndarray")
    if values.ndim != 1 or values.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be a one-dimensional float64 array")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class C0FoldScoresAntigravity:
    normal_scores: np.ndarray
    primary_scores: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        normal = _readonly_vector_antigravity(self.normal_scores, "normal_scores")
        if not isinstance(self.primary_scores, Mapping):
            raise ValueError("primary_scores must be a mapping")
        if tuple(self.primary_scores) != _CHANGE_TYPES_ANTIGRAVITY:
            raise ValueError("primary_scores must preserve frozen change order")
        primary = {
            name: _readonly_vector_antigravity(
                self.primary_scores[name],
                f"primary_scores[{name}]",
            )
            for name in _CHANGE_TYPES_ANTIGRAVITY
        }
        if any(len(values) != len(normal) for values in primary.values()):
            raise ValueError("C0 score lengths must match")
        object.__setattr__(self, "normal_scores", normal)
        object.__setattr__(self, "primary_scores", MappingProxyType(primary))


def capture_exp465_authority_antigravity() -> Exp465AuthorityAntigravity:
    """Return the frozen Exp465 closeout and exact C0 score source."""
    return Exp465AuthorityAntigravity(
        closeout_commit=_EXP465_CLOSEOUT_COMMIT_ANTIGRAVITY,
        score_source=score_exp465_fold_methods_antigravity,
        predecessor_authority=capture_exp463_authority_antigravity(),
    )


def score_c0_fold_antigravity(
    train140: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
    authority: Exp465AuthorityAntigravity,
) -> C0FoldScoresAntigravity:
    """Project the exact Exp465 C0 normal and primary score vectors."""
    if type(authority) is not Exp465AuthorityAntigravity:
        raise ValueError("authority must be exact Exp465AuthorityAntigravity")
    if authority.closeout_commit != _EXP465_CLOSEOUT_COMMIT_ANTIGRAVITY:
        raise ValueError("authority closeout commit drifted")
    if authority.score_source is not score_exp465_fold_methods_antigravity:
        raise ValueError("authority score source drifted")

    source = authority.score_source(
        train140,
        fit_indices,
        query_indices,
        seed=seed,
        fold_index=fold_index,
        authority=authority.predecessor_authority,
    )
    if getattr(source, "methods", None) != ("C0", "C6", "C8"):
        raise ValueError("Exp465 scorer returned an invalid method order")
    original = getattr(source, "original", None)
    primary = getattr(source, "primary", None)
    if not isinstance(original, Mapping) or not isinstance(primary, Mapping):
        raise ValueError("Exp465 scorer returned invalid C0 mappings")
    if "C0" not in original or "C0" not in primary:
        raise ValueError("Exp465 scorer omitted C0")
    return C0FoldScoresAntigravity(
        normal_scores=original["C0"],
        primary_scores=primary["C0"],
    )


__all__ = (
    "C0FoldScoresAntigravity",
    "Exp465AuthorityAntigravity",
    "capture_exp465_authority_antigravity",
    "score_c0_fold_antigravity",
)
