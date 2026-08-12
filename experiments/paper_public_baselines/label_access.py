"""Evaluation-only, single-pass access to authenticated TEXT TEST labels."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from experiments.exp448_exp447_text_label_boundary_repair_antigravity.publication_identity_antigravity import (
    capture_exp446_publication_antigravity,
)
from experiments.exp448_exp447_text_label_boundary_repair_antigravity.sources_antigravity import (
    capture_hard_predictions_antigravity,
    open_test_labels_antigravity,
)


def load_test_labels(
    dataset_order: tuple[str, ...],
) -> Mapping[str, tuple[tuple[int, int], ...]]:
    """Read TEST labels once, in frozen order, then close and attest the DB."""
    order = tuple(dataset_order)
    first = capture_exp446_publication_antigravity()
    second = capture_exp446_publication_antigravity()
    if first is second or first != second:
        raise ValueError("two independent Exp446 captures are required")
    if tuple(first.dataset_order[: len(order)]) != order:
        raise ValueError("requested label order differs from frozen population")
    hard = capture_hard_predictions_antigravity(first)
    session = open_test_labels_antigravity(first, second, hard)
    labels: dict[str, tuple[tuple[int, int], ...]] = {}
    try:
        for dataset in order:
            rows = session.labels_for_dataset_antigravity(dataset)
            labels[dataset] = tuple(rows)
    finally:
        session.close_antigravity()
    if tuple(labels) != order:
        raise ValueError("label session did not return the requested population")
    return MappingProxyType(labels)


__all__ = ["load_test_labels"]
