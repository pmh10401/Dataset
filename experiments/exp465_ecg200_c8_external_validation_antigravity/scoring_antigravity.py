"""Exp464-bound C0/C6/C8 projection for Exp465."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.contracts_antigravity import (
    Exp463AuthorityAntigravity,
)
from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity import (
    score_fold_methods_antigravity as score_exp464_fold_methods_antigravity,
)


_PROJECTED_METHOD_FIELDS_ANTIGRAVITY = (
    "original",
    "primary",
    "confirmation_counts",
    "fit_amplitudes",
    "query_amplitudes",
    "query_cosines",
    "raw_evidence_kind",
    "raw_evidence_source_method",
    "raw_fit_vector_blobs",
    "raw_query_vector_blobs",
    "feature_dimensions",
    "provenance",
)


@dataclass(frozen=True, slots=True)
class FoldMethodScoresAntigravity:
    methods: tuple[str, ...]
    original: Mapping[str, np.ndarray]
    primary: Mapping[str, Mapping[str, np.ndarray]]
    confirmation_counts: Mapping[str, Mapping[str, int]]
    fit_amplitudes: Mapping[str, np.ndarray]
    query_amplitudes: Mapping[str, Mapping[str, np.ndarray]]
    query_cosines: Mapping[str, Mapping[str, np.ndarray]]
    raw_evidence_kind: Mapping[str, str]
    raw_evidence_source_method: Mapping[str, str]
    raw_fit_vector_blobs: Mapping[str, str]
    raw_query_vector_blobs: Mapping[str, Mapping[str, str]]
    feature_dimensions: Mapping[str, int]
    provenance: Mapping[str, str]


def project_methods_antigravity(source: object, methods: Sequence[str]) -> FoldMethodScoresAntigravity:
    """Project exactly the requested methods from one Exp464 result without copying mathematics."""
    method_order = tuple(methods)
    if method_order != ("C0", "C6", "C8"):
        raise ValueError("Exp465 must project frozen C0,C6,C8 order")
    if source.methods != ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):  # type: ignore[attr-defined]
        raise ValueError("source must be a complete Exp464 fold-method result")
    if tuple(source.methods) != tuple(getattr(source, "methods")):
        raise ValueError("source methods must be immutable")

    projected: dict[str, Any] = {"methods": method_order}
    for field_name in _PROJECTED_METHOD_FIELDS_ANTIGRAVITY:
        source_mapping = getattr(source, field_name)
        if not isinstance(source_mapping, Mapping) or not all(method in source_mapping for method in method_order):
            raise ValueError(f"source.{field_name} is incomplete")
        projected[field_name] = MappingProxyType({method: source_mapping[method] for method in method_order})
    return FoldMethodScoresAntigravity(**projected)


def score_fold_methods_antigravity(
    train140: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
    authority: Exp463AuthorityAntigravity,
) -> FoldMethodScoresAntigravity:
    """Call Exp464 once and return only its exact C0/C6/C8 fields."""
    return project_methods_antigravity(
        score_exp464_fold_methods_antigravity(
            train140,
            fit_indices,
            query_indices,
            seed=seed,
            fold_index=fold_index,
            authority=authority,
        ),
        ("C0", "C6", "C8"),
    )


__all__ = (
    "FoldMethodScoresAntigravity",
    "project_methods_antigravity",
    "score_fold_methods_antigravity",
)
