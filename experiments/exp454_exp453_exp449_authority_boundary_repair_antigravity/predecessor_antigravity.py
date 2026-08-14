"""Exp453 terminal hashes plus the single Exp449 target-authority conversion."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import (
    capture_exp448_authority_antigravity,
)

from .config_antigravity import EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY
from .contracts_antigravity import ConfusionAntigravity, DatasetBaselineAntigravity


EXP453_STOPPED_RECEIPT_SHA256_ANTIGRAVITY = "1b6b2ea01c0c4dde9128051e57c764d32cbdf8bc1af7fa0115f19c7a10181d97"
EXP453_READINESS_RECEIPT_SHA256_ANTIGRAVITY = "38607b6c8cad785ca8cd0830dacd42775d43f9175a67ed01d487657a3c44764e"
EXP453_EVALUATE_RECEIPT_SHA256_ANTIGRAVITY = "caa705be90ddd56a00c53d5eb94a90604768dac0a80274846adce2061349b3b6"
EXP453_FAILURE_AUDIT_SHA256_ANTIGRAVITY = "97e7bea79bd1a830b925e7ffa2234e4e34725ac6734e080216f409688cb58276"


@dataclass(frozen=True, slots=True)
class NormalizedTargetAuthorityAntigravity:
    target_dataset_names: tuple[str, ...]
    target_baseline_rows: tuple[DatasetBaselineAntigravity, ...]
    baseline_confusion: ConfusionAntigravity


def _target_snapshot_antigravity(authority: object) -> tuple[tuple[str, ...], tuple[tuple[object, ...], ...], tuple[int, int, int, int]]:
    datasets = getattr(authority, "target_dataset_names", None)
    raw_rows = getattr(authority, "target_baseline_rows", None)
    raw_global = getattr(authority, "baseline_confusion", None)
    if type(datasets) is not tuple or type(raw_rows) is not tuple or len(datasets) != 48 or len(raw_rows) != 48:
        raise ValueError("Exp449 target inventory differs")
    rows: list[tuple[object, ...]] = []
    for dataset, raw in zip(datasets, raw_rows, strict=True):
        order = getattr(raw, "dataset_order", None)
        family = getattr(raw, "family", None)
        candidates = getattr(raw, "stable_candidates", None)
        confusion = getattr(raw, "confusion", None)
        if type(dataset) is not str or type(order) is not int or type(family) is not str or type(candidates) is not frozenset:
            raise ValueError("Exp449 target order differs")
        rows.append(
            (
                dataset,
                order,
                family,
                frozenset(candidates),
                (
                    getattr(confusion, "tp", None),
                    getattr(confusion, "fp", None),
                    getattr(confusion, "fn", None),
                    getattr(confusion, "tn", None),
                ),
            )
        )
    global_confusion = (
        getattr(raw_global, "tp", None),
        getattr(raw_global, "fp", None),
        getattr(raw_global, "fn", None),
        getattr(raw_global, "tn", None),
    )
    if any(type(value) is not int for value in global_confusion) or any(
        type(value) is not int for _dataset, _order, _family, _candidates, counts in rows for value in counts
    ):
        raise ValueError("Exp449 target inventory differs")
    return datasets, tuple(rows), global_confusion  # type: ignore[return-value]


def normalize_exp449_target_authority_antigravity(authority: object) -> NormalizedTargetAuthorityAntigravity:
    """Copy exact Exp449 target fields into Exp454 immutable types once."""
    snapshot = _target_snapshot_antigravity(authority)
    expected = _target_snapshot_antigravity(capture_exp448_authority_antigravity())
    if snapshot != expected:
        if snapshot[0] != expected[0] or tuple(row[1] for row in snapshot[1]) != tuple(row[1] for row in expected[1]):
            raise ValueError("Exp449 target order differs")
        raise ValueError("Exp449 target authority differs")
    datasets, rows, global_confusion = snapshot
    copied = []
    for _dataset, order, family, candidates, counts in rows:
        copied.append(
            DatasetBaselineAntigravity(
                order,  # type: ignore[arg-type]
                family,  # type: ignore[arg-type]
                frozenset(candidates),  # type: ignore[arg-type]
                ConfusionAntigravity(*counts),  # type: ignore[arg-type]
            )
        )
    return NormalizedTargetAuthorityAntigravity(datasets, tuple(copied), ConfusionAntigravity(*global_confusion))


def capture_normalized_target_authority_antigravity() -> NormalizedTargetAuthorityAntigravity:
    """Call the real Exp449 capture and apply the single conversion."""
    return normalize_exp449_target_authority_antigravity(capture_exp448_authority_antigravity())


__all__ = [
    "EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY",
    "EXP453_EVALUATE_RECEIPT_SHA256_ANTIGRAVITY",
    "EXP453_FAILURE_AUDIT_SHA256_ANTIGRAVITY",
    "EXP453_READINESS_RECEIPT_SHA256_ANTIGRAVITY",
    "EXP453_STOPPED_RECEIPT_SHA256_ANTIGRAVITY",
    "NormalizedTargetAuthorityAntigravity",
    "capture_normalized_target_authority_antigravity",
    "normalize_exp449_target_authority_antigravity",
]
