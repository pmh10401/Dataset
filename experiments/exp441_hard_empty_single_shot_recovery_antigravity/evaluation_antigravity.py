"""Fail-closed post-freeze loader for Exp441 TEST-label facts only."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .artifacts_antigravity import canonical_json_bytes_antigravity
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    FrozenPublicationAntigravity,
    FrozenSourceSnapshotAntigravity,
    SelectionResultAntigravity,
    validate_production_selection_antigravity,
)
from .reporting_antigravity import inspect_exp441_publication_antigravity
from .runtime_antigravity import _snapshot_payload_antigravity
from .selector_antigravity import select_hard_empty_single_shot_antigravity
from .sources_antigravity import verify_frozen_sources_antigravity


_LABEL_SQL_ANTIGRAVITY = """\
SELECT i.instance_index, i.label
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index
"""


@dataclass(frozen=True)
class FrozenContextLabelsAntigravity:
    """Detached TEST labels bound to one unchanged Exp441 publication snapshot."""

    context_datasets: tuple[str, ...]
    labels_by_dataset: Mapping[str, tuple[tuple[int, int], ...]]
    publication_sha256: str
    source_audit_sha256: str
    selection: SelectionResultAntigravity
    descriptor_bound_connection_count: int
    label_query_calls: int
    out_of_scope_label_rows: int

    def __post_init__(self) -> None:
        datasets = tuple(self.context_datasets)
        if (not datasets or any(type(dataset) is not str or not dataset for dataset in datasets)
                or len(datasets) != len(set(datasets))):
            raise ValueError("context datasets are invalid")
        if not isinstance(self.labels_by_dataset, Mapping) or set(self.labels_by_dataset) != set(datasets):
            raise ValueError("label datasets are invalid")
        copied: dict[str, tuple[tuple[int, int], ...]] = {}
        for dataset in datasets:
            try:
                rows = tuple(self.labels_by_dataset[dataset])
            except (KeyError, TypeError) as error:
                raise ValueError("label rows are invalid") from error
            normalized: list[tuple[int, int]] = []
            previous = -1
            for row in rows:
                if not isinstance(row, tuple) or len(row) != 2:
                    raise ValueError("label row schema is invalid")
                index, label = row
                if type(index) is not int or index < 0 or index <= previous:
                    raise ValueError("label index is invalid")
                if type(label) is not int or label not in (0, 1):
                    raise ValueError("label value is invalid")
                normalized.append((index, label))
                previous = index
            copied[dataset] = tuple(normalized)
        if (any(type(value) is not str or len(value) != 64 for value in
                (self.publication_sha256, self.source_audit_sha256))
                or not isinstance(self.selection, SelectionResultAntigravity)):
            raise ValueError("frozen label identity is invalid")
        if (type(self.descriptor_bound_connection_count) is not int
                or self.descriptor_bound_connection_count != 1
                or type(self.label_query_calls) is not int
                or self.label_query_calls != len(datasets)
                or type(self.out_of_scope_label_rows) is not int
                or self.out_of_scope_label_rows != 0):
            raise ValueError("frozen label audit is invalid")
        object.__setattr__(self, "context_datasets", datasets)
        object.__setattr__(self, "labels_by_dataset", MappingProxyType(copied))


def _source_audit_sha256_antigravity(snapshot: FrozenSourceSnapshotAntigravity) -> str:
    return sha256(canonical_json_bytes_antigravity(_snapshot_payload_antigravity(snapshot))).hexdigest()


def _preflight_antigravity() -> tuple[FrozenPublicationAntigravity, FrozenSourceSnapshotAntigravity, SelectionResultAntigravity, tuple[str, ...], Mapping[str, int]]:
    publication = inspect_exp441_publication_antigravity()
    if not isinstance(publication, FrozenPublicationAntigravity):
        raise ValueError("publication is invalid")
    snapshot = verify_frozen_sources_antigravity()
    if not isinstance(snapshot, FrozenSourceSnapshotAntigravity) or snapshot.label_access_count != 0:
        raise ValueError("frozen sources are invalid")
    selection = select_hard_empty_single_shot_antigravity(snapshot)
    validate_production_selection_antigravity(selection)
    if selection != publication.selection:
        raise ValueError("publication selection changed")
    datasets = tuple(snapshot.context_datasets)
    row_counts = MappingProxyType({dataset: snapshot.test_row_count_by_dataset[dataset] for dataset in datasets})
    coordinates = set(snapshot.hard_coordinates)
    coordinates.update(item.candidate.coordinate for item in selection.selected)
    for dataset, index in coordinates:
        if dataset not in row_counts or type(index) is not int or index < 0 or index >= row_counts[dataset]:
            raise ValueError("frozen coordinate is outside row-count identity")
    if _source_audit_sha256_antigravity(snapshot) != publication.source_audit_sha256:
        raise ValueError("publication source-audit binding changed")
    final_publication = inspect_exp441_publication_antigravity()
    if final_publication != publication:
        raise ValueError("publication changed before label import")
    if (not isinstance(final_publication, FrozenPublicationAntigravity)
            or final_publication.selection != selection
            or final_publication.source_audit_sha256 != _source_audit_sha256_antigravity(snapshot)):
        raise ValueError("publication final binding is invalid")
    return publication, snapshot, selection, datasets, row_counts


def _normalize_label_antigravity(value: object) -> int:
    if type(value) is int and value in (0, 1):
        return value
    if type(value) is str and value in ("0", "1"):
        return int(value)
    raise ValueError("label value is invalid")


def _load_labels_antigravity(connection: object, datasets: tuple[str, ...], row_counts: Mapping[str, int]) -> Mapping[str, tuple[tuple[int, int], ...]]:
    quick_check = connection.execute("PRAGMA quick_check").fetchall()  # type: ignore[attr-defined]
    if quick_check != [("ok",)]:
        raise ValueError("quick_check failed")
    labels: dict[str, tuple[tuple[int, int], ...]] = {}
    for dataset in datasets:
        rows = connection.execute(_LABEL_SQL_ANTIGRAVITY, (dataset,)).fetchall()  # type: ignore[attr-defined]
        normalized: list[tuple[int, int]] = []
        previous = -1
        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("label row schema is invalid")
            index, value = row
            if type(index) is not int or index < 0 or index <= previous:
                raise ValueError("label index is invalid")
            normalized.append((index, _normalize_label_antigravity(value)))
            previous = index
        if len(normalized) != row_counts[dataset]:
            raise ValueError("label row count is invalid")
        labels[dataset] = tuple(normalized)
    return MappingProxyType(labels)


def _require_loaded_coordinates_antigravity(labels: Mapping[str, tuple[tuple[int, int], ...]], snapshot: FrozenSourceSnapshotAntigravity, selection: SelectionResultAntigravity) -> None:
    identities = {dataset: {index for index, _ in rows} for dataset, rows in labels.items()}
    coordinates = set(snapshot.hard_coordinates)
    coordinates.update(item.candidate.coordinate for item in selection.selected)
    for dataset, index in coordinates:
        if dataset not in identities or index not in identities[dataset]:
            raise ValueError("frozen coordinate is absent from loaded TEST identity")


def load_frozen_context_labels_antigravity() -> FrozenContextLabelsAntigravity:
    """Load only frozen-context TEST labels after binding publication and source twice."""
    publication, snapshot, selection, datasets, row_counts = _preflight_antigravity()
    from experiments.exp417_causal_multihorizon_shadow.inputs import verified_read_only_connection
    with verified_read_only_connection(Path(FROZEN_CONFIG.database_path), FROZEN_CONFIG.database_sha256) as connection:
        labels = _load_labels_antigravity(connection, datasets, row_counts)
        _require_loaded_coordinates_antigravity(labels, snapshot, selection)
    post_publication = inspect_exp441_publication_antigravity()
    post_snapshot = verify_frozen_sources_antigravity()
    if (post_publication != publication
            or not isinstance(post_snapshot, FrozenSourceSnapshotAntigravity)
            or post_snapshot != snapshot
            or post_snapshot.label_access_count != 0):
        raise ValueError("post-connection frozen authority changed")
    post_selection = select_hard_empty_single_shot_antigravity(post_snapshot)
    validate_production_selection_antigravity(post_selection)
    if (post_selection != selection
            or _source_audit_sha256_antigravity(post_snapshot) != publication.source_audit_sha256):
        raise ValueError("post-connection source binding changed")
    return FrozenContextLabelsAntigravity(
        datasets, labels, publication.publication_sha256, publication.source_audit_sha256,
        selection, 1, len(datasets), 0,
    )


__all__ = ["FrozenContextLabelsAntigravity", "load_frozen_context_labels_antigravity"]


def __getattr__(name: str) -> object:
    """Keep the pre-Task-10 CLI evaluation route unavailable without exporting it."""
    if name == "evaluate_exp441_antigravity":
        raise ModuleNotFoundError("Exp441 evaluation is not implemented at the Task 9 label boundary")
    raise AttributeError(name)
