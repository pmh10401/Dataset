"""Descriptor-bound, exact-length, label-free Exp439 prediction inputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.config import (
    FROZEN_CONFIG as EXP417_FROZEN_CONFIG,
)
from experiments.exp417_causal_multihorizon_shadow.inputs import freeze_lane_inputs
from experiments.exp425_selective_recovery.database import (
    Exp425DatabaseSnapshot,
    open_database_snapshot,
)

from .config_antigravity import DATABASE_PATH_ANTIGRAVITY, FROZEN_CONFIG
from .contracts_antigravity import (
    ExcludedTrainRowAudit,
    PreparedTrainGroup,
    RawValueSeries,
)


DEFAULT_DATABASE = DATABASE_PATH_ANTIGRAVITY
DEFAULT_EXP291_PREDICTION = Path(
    "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/"
    "exp291_frozen_operational_routing.jsonl"
)
DEFAULT_EXP291_MANIFEST = Path(
    "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/"
    "exp291_prediction_manifest.json"
)
DEFAULT_EXP292_PREDICTION = Path(
    "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/"
    "exp292_frozen_review_tiers.jsonl"
)
DEFAULT_EXP292_MANIFEST = Path(
    "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/"
    "exp292_prediction_manifest.json"
)


@dataclass(frozen=True)
class Exp439CanonicalSourcePaths:
    database: Path
    exp291_prediction: Path
    exp291_manifest: Path
    exp292_prediction: Path
    exp292_manifest: Path

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{name} must be an absolute canonical source path")

    def to_payload(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}


def canonical_source_paths_antigravity(
    database: Path = DEFAULT_DATABASE,
    exp291_prediction: Path = DEFAULT_EXP291_PREDICTION,
    exp291_manifest: Path = DEFAULT_EXP291_MANIFEST,
    exp292_prediction: Path = DEFAULT_EXP292_PREDICTION,
    exp292_manifest: Path = DEFAULT_EXP292_MANIFEST,
) -> Exp439CanonicalSourcePaths:
    return Exp439CanonicalSourcePaths(
        *(Path(value).resolve(strict=False) for value in (
            database,
            exp291_prediction,
            exp291_manifest,
            exp292_prediction,
            exp292_manifest,
        ))
    )


def decode_raw_value_row_antigravity(
    dataset: str,
    split: str,
    instance_index: int,
    blob: bytes,
) -> RawValueSeries:
    if type(dataset) is not str or not dataset:
        raise ValueError("raw value dataset is invalid")
    if split not in {"TRAIN", "TEST"}:
        raise ValueError("raw value split is invalid")
    if type(instance_index) is not int or instance_index < 0:
        raise ValueError("raw value index must be an exact nonnegative integer")
    if type(blob) is not bytes:
        raise ValueError("raw values blob must be exact bytes")
    if not blob or len(blob) % 4:
        raise ValueError("raw values blob length is invalid")
    values = np.frombuffer(blob, dtype="<f4")
    return RawValueSeries(
        dataset=dataset,
        instance_index=instance_index,
        split=split,
        values=values,
        values_blob_sha256=sha256(blob).hexdigest(),
    )


@dataclass(frozen=True)
class Exp439RawDataset:
    dataset: str
    split: str
    series: tuple[RawValueSeries, ...]

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("raw dataset name is invalid")
        if self.split not in {"TRAIN", "TEST"}:
            raise ValueError("raw dataset split is invalid")
        rows = tuple(self.series)
        if not rows or any(
            type(row) is not RawValueSeries
            or row.dataset != self.dataset
            or row.split != self.split
            for row in rows
        ):
            raise ValueError("raw dataset rows are invalid")
        identities = tuple(row.instance_index for row in rows)
        if tuple(sorted(identities)) != identities or len(set(identities)) != len(identities):
            raise ValueError("raw dataset identities are invalid")
        object.__setattr__(self, "series", rows)

    @property
    def identity_lengths(self) -> tuple[tuple[int, int], ...]:
        return tuple((row.instance_index, row.raw_length) for row in self.series)


@dataclass(frozen=True)
class TrainPreparationAntigravity:
    dataset: str
    groups: tuple[PreparedTrainGroup, ...]
    exclusions: tuple[ExcludedTrainRowAudit, ...]
    insufficient_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("TRAIN preparation dataset is invalid")
        groups = tuple(self.groups)
        exclusions = tuple(self.exclusions)
        lengths = tuple(self.insufficient_lengths)
        if any(group.dataset != self.dataset for group in groups) or any(item.dataset != self.dataset for item in exclusions):
            raise ValueError("TRAIN preparation contains a foreign dataset")
        if tuple(group.raw_length for group in groups) != tuple(sorted(group.raw_length for group in groups)):
            raise ValueError("TRAIN groups are not length ordered")
        if lengths != tuple(sorted(set(lengths))) or any(type(length) is not int or length < 1 for length in lengths):
            raise ValueError("insufficient TRAIN lengths are invalid")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "exclusions", exclusions)
        object.__setattr__(self, "insufficient_lengths", lengths)


def partition_train_rows_antigravity(train: Exp439RawDataset) -> TrainPreparationAntigravity:
    if type(train) is not Exp439RawDataset or train.split != "TRAIN":
        raise ValueError("TRAIN partition requires an Exp439 TRAIN dataset")
    finite_by_length: dict[int, list[RawValueSeries]] = defaultdict(list)
    exclusions: list[ExcludedTrainRowAudit] = []
    all_lengths: set[int] = set()
    for row in train.series:
        all_lengths.add(row.raw_length)
        nonfinite = tuple(int(index) for index in np.flatnonzero(~np.isfinite(row.values)))
        if nonfinite:
            exclusions.append(
                ExcludedTrainRowAudit(
                    dataset=row.dataset,
                    instance_index=row.instance_index,
                    raw_length=row.raw_length,
                    nonfinite_positions=nonfinite,
                    values_blob_sha256=row.values_blob_sha256,
                )
            )
        else:
            finite_by_length[row.raw_length].append(row)
    groups: list[PreparedTrainGroup] = []
    insufficient: list[int] = []
    for raw_length in sorted(all_lengths):
        rows = tuple(finite_by_length.get(raw_length, ()))
        if raw_length < FROZEN_CONFIG.minimum_raw_length or len(rows) < FROZEN_CONFIG.minimum_train_rows:
            insufficient.append(raw_length)
            continue
        groups.append(PreparedTrainGroup(train.dataset, raw_length, rows))
    return TrainPreparationAntigravity(
        dataset=train.dataset,
        groups=tuple(groups),
        exclusions=tuple(sorted(exclusions, key=lambda item: item.instance_index)),
        insufficient_lengths=tuple(insufficient),
    )


def test_row_abstention_reason_antigravity(row: RawValueSeries) -> str | None:
    if type(row) is not RawValueSeries or row.split != "TEST":
        raise ValueError("TEST abstention requires an Exp439 TEST row")
    if not np.isfinite(row.values).all():
        return "nonfinite_test"
    if row.raw_length < FROZEN_CONFIG.minimum_raw_length:
        return "unsupported_length"
    return None


@dataclass(slots=True)
class Exp439InputSession:
    _database: Exp425DatabaseSnapshot
    context: object
    source_paths: Exp439CanonicalSourcePaths
    _active: bool = field(default=True, init=False, repr=False)
    _train_authority_antigravity: dict[
        str, tuple[Exp439RawDataset, tuple[tuple[int, bytes], ...]]
    ] = field(default_factory=dict, init=False, repr=False)

    @property
    def label_access_count(self) -> int:
        return 0

    def _require_active_dataset(self, dataset: str) -> None:
        if not self._active:
            raise ValueError("Exp439 input session is closed")
        if type(dataset) is not str or dataset not in self.context.datasets:
            raise ValueError("dataset is not frozen in this Exp439 input session")

    def _close_antigravity(self) -> None:
        self._train_authority_antigravity.clear()
        self._active = False

    def load_train(self, dataset: str) -> Exp439RawDataset:
        self._require_active_dataset(dataset)
        cached = self._train_authority_antigravity.get(dataset)
        if cached is not None:
            return cached[0]
        blobs = self._database.load_normal_train_value_blobs(dataset, self.context)
        public = Exp439RawDataset(
            dataset,
            "TRAIN",
            tuple(
                decode_raw_value_row_antigravity(dataset, "TRAIN", index, blob)
                for index, blob in blobs
            ),
        )
        self._train_authority_antigravity[dataset] = (public, blobs)
        return public

    def load_test_values(
        self,
        dataset: str,
        train: Exp439RawDataset,
    ) -> Exp439RawDataset:
        self._require_active_dataset(dataset)
        cached = self._train_authority_antigravity.get(dataset)
        if cached is None or train is not cached[0]:
            raise ValueError("TEST values require this session's bound TRAIN object")
        blobs = self._database.load_test_value_blobs_after_train(
            dataset,
            cached[1],
            self.context,
        )
        result = Exp439RawDataset(
            dataset,
            "TEST",
            tuple(
                decode_raw_value_row_antigravity(dataset, "TEST", index, blob)
                for index, blob in blobs
            ),
        )
        if result.identity_lengths != tuple(self.context.test_identity_manifest[dataset]):
            raise ValueError("TEST values do not match the frozen identity manifest")
        return result


def _assert_shared_frozen_identity() -> None:
    expected = getattr(EXP417_FROZEN_CONFIG, "expected_datasets", None)
    if FROZEN_CONFIG.dataset_count != expected:
        raise ValueError("Exp439/Exp417 frozen dataset count mismatch")
    if FROZEN_CONFIG.database_sha256 != getattr(EXP417_FROZEN_CONFIG, "database_sha256", None):
        raise ValueError("Exp439/Exp417 frozen database mismatch")


def _validate_population(population: object, *, source: str) -> tuple[str, ...]:
    if type(population) is not tuple or len(population) != FROZEN_CONFIG.dataset_count:
        raise ValueError(f"{source} population must contain exactly {FROZEN_CONFIG.dataset_count} names")
    if any(type(name) is not str or not name for name in population):
        raise ValueError(f"{source} population contains an invalid dataset")
    if population != tuple(sorted(population)) or len(set(population)) != len(population):
        raise ValueError(f"{source} population must be sorted and unique")
    return population


@contextmanager
def open_exp439_input_session(
    database: Path = DEFAULT_DATABASE,
    exp291_prediction: Path = DEFAULT_EXP291_PREDICTION,
    exp291_manifest: Path = DEFAULT_EXP291_MANIFEST,
    exp292_prediction: Path = DEFAULT_EXP292_PREDICTION,
    exp292_manifest: Path = DEFAULT_EXP292_MANIFEST,
    *,
    verified_code_commit: str,
) -> Iterator[Exp439InputSession]:
    _assert_shared_frozen_identity()
    if type(verified_code_commit) is not str or re.fullmatch(r"[0-9a-f]{40}", verified_code_commit) is None:
        raise ValueError("Exp439 reviewed code commit is invalid")
    sources = canonical_source_paths_antigravity(
        database,
        exp291_prediction,
        exp291_manifest,
        exp292_prediction,
        exp292_manifest,
    )
    lanes = freeze_lane_inputs(
        sources.exp291_prediction,
        sources.exp291_manifest,
        sources.exp292_prediction,
        sources.exp292_manifest,
        config=EXP417_FROZEN_CONFIG,
    )
    lane_population = _validate_population(tuple(lanes.population), source="frozen lane")
    with open_database_snapshot(
        sources.database,
        expected_sha256=FROZEN_CONFIG.database_sha256,
    ) as snapshot:
        context = snapshot.freeze_context(lanes, verified_code_commit=verified_code_commit)
        context_population = _validate_population(tuple(context.datasets), source="frozen context")
        if context_population != lane_population:
            raise ValueError("frozen lane and context populations do not match")
        session = Exp439InputSession(snapshot, context, sources)
        try:
            yield session
        finally:
            session._close_antigravity()


__all__ = [
    "Exp439CanonicalSourcePaths",
    "Exp439InputSession",
    "Exp439RawDataset",
    "TrainPreparationAntigravity",
    "canonical_source_paths_antigravity",
    "decode_raw_value_row_antigravity",
    "open_exp439_input_session",
    "partition_train_rows_antigravity",
    "test_row_abstention_reason_antigravity",
]
