"""Exp451's exact immutable view of the descriptor-pinned Exp448 authority."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import (
    capture_exp448_authority_antigravity,
)


_TARGET_FAMILIES_ANTIGRAVITY = ("Crop", "ElectricDevices", "FordA", "FordB", "Yoga", "ECG5000", "ChlorineConcentration", "StarLightCurves")
_TARGET_COUNTS_ANTIGRAVITY = (24, 7, 2, 2, 2, 5, 3, 3)
_TARGET_FN_ANTIGRAVITY = (246, 200, 88, 75, 63, 56, 56, 53)
_TARGET_TEST_LENGTHS_ANTIGRAVITY = (1000, 1000, 2350, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 2950, 1800, 50, 150, 50, 1400, 4250, 1600, 2650, 4350, 1250, 1300, 2550, 2400, 2300, 2200, 1350, 2600, 5400, 1550, 1800)


@dataclass(frozen=True, slots=True)
class Exp448PredecessorAuthorityAntigravity:
    root_identity: tuple[int, int, int, int, int, int, int]
    artifact_map_sha256: str
    evaluation_sha256: str
    decision_sha256: str
    population: int
    target_dataset_count: int
    target_dataset_order: tuple[int, ...]
    target_dataset_names: tuple[str, ...]
    target_baseline_rows: tuple[object, ...]
    baseline_confusion: tuple[int, int, int, int]
    target_family_fn: tuple[int, ...]
    target_test_rows: int
    target_test_lengths: tuple[int, ...]
    label_access_count: int


def verify_exp448_predecessor_antigravity() -> Exp448PredecessorAuthorityAntigravity:
    """Revalidate Exp448 and expose no label-capable or mutable predecessor state."""
    source = capture_exp448_authority_antigravity()
    confusion = source.baseline_confusion
    names = tuple(source.target_dataset_names)
    rows = tuple(source.target_baseline_rows)
    families = tuple(row.family for row in rows)
    family_counts = tuple(families.count(family) for family in _TARGET_FAMILIES_ANTIGRAVITY)
    if (
        source.population != 1117
        or source.target_dataset_count != 48
        or len(names) != 48
        or len(set(names)) != 48
        or tuple(source.target_dataset_order) != tuple(row.dataset_order for row in rows)
        or family_counts != _TARGET_COUNTS_ANTIGRAVITY
        or any(not name.startswith(family + "_") for name, family in zip(names, families, strict=True))
        or tuple(source.target_family_fn) != _TARGET_FN_ANTIGRAVITY
        or (confusion.tp, confusion.fp, confusion.fn, confusion.tn) != (1930, 314, 1988, 191668)
    ):
        raise ValueError("Exp448 predecessor semantics differ")
    return Exp448PredecessorAuthorityAntigravity(
        tuple(source.root_identity), source.artifact_map_sha256, source.evaluation_sha256,
        source.decision_sha256, 1117, 48, tuple(source.target_dataset_order), names, rows,
        (1930, 314, 1988, 191668), _TARGET_FN_ANTIGRAVITY, 72_300, _TARGET_TEST_LENGTHS_ANTIGRAVITY, 0,
    )


__all__ = ["Exp448PredecessorAuthorityAntigravity", "verify_exp448_predecessor_antigravity"]
