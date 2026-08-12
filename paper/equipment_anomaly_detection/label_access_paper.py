"""Lazy, post-parity TEST label capability for paper evaluation only."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from pathlib import Path

from .config_paper import FROZEN_CONFIG
from .data_paper import _open_verified_read_only


TEST_LABELS_SQL = """SELECT i.instance_index, i.label
FROM instances i JOIN datasets d ON d.id=i.dataset_id
WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index"""


def _convert(raw: object) -> int:
    if type(raw) is not str or raw not in {"0", "1"}:
        raise ValueError("TEST label must be exact TEXT '0' or '1'")
    return 0 if raw == "0" else 1


def load_test_labels(db_path: Path, datasets: tuple[str, ...]) -> Mapping[str, tuple[int, ...]]:
    """Read exact TEXT TEST labels only after the label-free score stages freeze."""
    if (
        type(datasets) is not tuple
        or not datasets
        or len(set(datasets)) != len(datasets)
        or any(type(dataset) is not str or not dataset for dataset in datasets)
    ):
        raise ValueError("label datasets must be a non-empty unique tuple")
    labels: dict[str, tuple[int, ...]] = {}
    with _open_verified_read_only(db_path, config=FROZEN_CONFIG) as connection:
        for dataset in datasets:
            rows = connection.execute(TEST_LABELS_SQL, (dataset,)).fetchall()
            if not rows:
                raise ValueError(f"no TEST labels for dataset={dataset}")
            indices: list[int] = []
            converted: list[int] = []
            for raw_index, raw_label in rows:
                if type(raw_index) is not int or raw_index < 0:
                    raise ValueError("TEST label index is invalid")
                indices.append(raw_index)
                converted.append(_convert(raw_label))
            if tuple(indices) != tuple(range(len(indices))):
                raise ValueError("TEST label indices must be contiguous from zero")
            labels[dataset] = tuple(converted)
    return MappingProxyType(labels)


__all__ = ["load_test_labels"]
