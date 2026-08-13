"""Exp451's exact immutable view of the descriptor-pinned Exp448 authority."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib, json

from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import (
    capture_exp448_authority_antigravity,
)
from experiments.exp448_exp447_text_label_boundary_repair_antigravity.sources_antigravity import capture_exp446_publication_antigravity, capture_hard_predictions_antigravity


_TARGET_FAMILIES_ANTIGRAVITY = ("Crop", "ElectricDevices", "FordA", "FordB", "Yoga", "ECG5000", "ChlorineConcentration", "StarLightCurves")
_TARGET_COUNTS_ANTIGRAVITY = (24, 7, 2, 2, 2, 5, 3, 3)
_TARGET_FN_ANTIGRAVITY = (246, 200, 88, 75, 63, 56, 56, 53)
_TARGET_TEST_LENGTHS_ANTIGRAVITY = (1000, 1000, 2350, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 2950, 1800, 50, 150, 50, 1400, 4250, 1600, 2650, 4350, 1250, 1300, 2550, 2400, 2300, 2200, 1350, 2600, 5400, 1550, 1800)
_FULL_BASELINE_AUTHORITY_SHA256_ANTIGRAVITY="ff01bc863b1b61850646fabfb96c906d5874efd240c1fae8fe715d59e23cfefe"


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
    target_full_baseline_candidates: tuple[frozenset[int], ...]
    full_baseline_authority_sha256: str


def verify_exp448_predecessor_antigravity() -> Exp448PredecessorAuthorityAntigravity:
    """Revalidate Exp448 and expose no label-capable or mutable predecessor state."""
    source = capture_exp448_authority_antigravity()
    confusion = source.baseline_confusion
    names = tuple(source.target_dataset_names)
    rows = tuple(source.target_baseline_rows)
    families = tuple(row.family for row in rows)
    publication=capture_exp446_publication_antigravity()
    if publication.label_access_count!=0: raise ValueError("Exp446 publication accessed labels")
    hard=capture_hard_predictions_antigravity(publication)
    hard_by_name={name:set() for name in names}
    for name,index in hard.coordinates:
        if name in hard_by_name: hard_by_name[name].add(index)
    full=tuple(frozenset(hard_by_name[name]|set(row.stable_candidates)) for name,row in zip(names,rows,strict=True))
    if sum(map(len,full))!=675 or any(any(index>=length for index in values) for values,length in zip(full,_TARGET_TEST_LENGTHS_ANTIGRAVITY,strict=True)): raise ValueError("full Exp448 target baseline coordinates differ")
    payload={"rows":[{"identity":[row.dataset_order,row.family,name],"target_length":length,"indices":sorted(values)} for row,name,length,values in zip(rows,names,_TARGET_TEST_LENGTHS_ANTIGRAVITY,full,strict=True)]}
    full_sha=hashlib.sha256((json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()
    if full_sha!=_FULL_BASELINE_AUTHORITY_SHA256_ANTIGRAVITY: raise ValueError("full baseline authority hash differs")
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
        (1930, 314, 1988, 191668), _TARGET_FN_ANTIGRAVITY, 72_300, _TARGET_TEST_LENGTHS_ANTIGRAVITY, 0, full, full_sha,
    )


__all__ = ["Exp448PredecessorAuthorityAntigravity", "verify_exp448_predecessor_antigravity"]
