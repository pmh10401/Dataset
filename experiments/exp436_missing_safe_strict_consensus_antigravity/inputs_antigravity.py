"""Descriptor-bound, missing-safe, label-free Exp436 prediction inputs.

This module owns the only Exp436 prediction-time route from frozen lanes to
raw values.  It retains the exact values-only TRAIN blob tuple returned by the
single Exp425 snapshot as private authorization for TEST values; it never
constructs a label-bearing object.
"""

from __future__ import annotations

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
from experiments.exp417_causal_multihorizon_shadow.contracts import FrozenRunContext
from experiments.exp417_causal_multihorizon_shadow.inputs import freeze_lane_inputs
from experiments.exp425_selective_recovery.database import (
    Exp425DatabaseSnapshot,
    open_database_snapshot,
)

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import MissingSafeRawDataset, RawValueSeries


DEFAULT_DATABASE = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
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

_LANE_HASH_FIELDS = (
    "exp291_prediction_sha256",
    "exp291_manifest_sha256",
    "exp292_prediction_sha256",
    "exp292_manifest_sha256",
)


@dataclass(frozen=True)
class Exp436CanonicalSourcePaths:
    """Immutable source bindings persisted by the runtime context."""

    database: Path
    exp291_prediction: Path
    exp291_manifest: Path
    exp292_prediction: Path
    exp292_manifest: Path

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{field_name} must be an absolute canonical source path")

    def to_payload(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}


def canonical_source_paths_antigravity(
    database: Path = DEFAULT_DATABASE,
    exp291_prediction: Path = DEFAULT_EXP291_PREDICTION,
    exp291_manifest: Path = DEFAULT_EXP291_MANIFEST,
    exp292_prediction: Path = DEFAULT_EXP292_PREDICTION,
    exp292_manifest: Path = DEFAULT_EXP292_MANIFEST,
) -> Exp436CanonicalSourcePaths:
    """Freeze exactly the caller-selected absolute bindings before opening DB."""

    return Exp436CanonicalSourcePaths(*(Path(value).resolve(strict=False) for value in (
        database, exp291_prediction, exp291_manifest, exp292_prediction, exp292_manifest
    )))


def decode_raw_value_row_antigravity(index: int, blob: bytes) -> RawValueSeries:
    """Decode one exact little-endian float32 BLOB without normalizing values."""

    if type(index) is not int or index < 0:
        raise ValueError("raw value index must be an exact nonnegative integer")
    if type(blob) is not bytes:
        raise ValueError("raw values blob must be exact bytes")
    if not blob or len(blob) % 4:
        raise ValueError("raw values blob length is invalid")
    digest = sha256(blob).hexdigest()
    values = np.frombuffer(blob, dtype="<f4")
    return RawValueSeries(index, tuple(float(value) for value in values), digest)


@dataclass(slots=True)
class Exp436InputSession:
    """One active snapshot and its private exact TRAIN values-only authority."""

    _database: Exp425DatabaseSnapshot
    context: FrozenRunContext
    source_paths: Exp436CanonicalSourcePaths | None = None
    _active: bool = field(default=True, init=False, repr=False)
    _train_authority_antigravity: dict[
        str, tuple[MissingSafeRawDataset, tuple[tuple[int, bytes], ...]]
    ] = field(default_factory=dict, init=False, repr=False)

    @property
    def label_access_count(self) -> int:
        """Prediction has no label-capable route; this invariant is observable."""

        return 0

    def _require_active_dataset(self, dataset: str) -> None:
        if not self._active:
            raise ValueError("Exp436 input session is closed")
        if type(dataset) is not str or dataset not in self.context.datasets:
            raise ValueError("dataset is not frozen in this Exp436 input session")

    def _close_antigravity(self) -> None:
        self._train_authority_antigravity.clear()
        self._active = False

    def load_train(self, dataset: str) -> MissingSafeRawDataset:
        """Return the cached public TRAIN object or make the one permitted query."""

        self._require_active_dataset(dataset)
        cached = self._train_authority_antigravity.get(dataset)
        if cached is not None:
            return cached[0]
        raw_rows = self._database.load_normal_train_value_blobs(dataset, self.context)
        public_train = MissingSafeRawDataset(
            dataset,
            "TRAIN",
            tuple(
                decode_raw_value_row_antigravity(index, blob)
                for index, blob in raw_rows
            ),
        )
        self._train_authority_antigravity[dataset] = (public_train, raw_rows)
        return public_train

    def load_test_values(
        self, dataset: str, train: MissingSafeRawDataset
    ) -> MissingSafeRawDataset:
        """Read TEST values only after exact cached TRAIN-object authorization."""

        self._require_active_dataset(dataset)
        cached = self._train_authority_antigravity.get(dataset)
        if cached is None or train is not cached[0]:
            raise ValueError("TEST values require this session's bound TRAIN object")
        raw_rows = self._database.load_test_value_blobs_after_train(
            dataset, cached[1], self.context
        )
        result = MissingSafeRawDataset(
            dataset,
            "TEST",
            tuple(
                decode_raw_value_row_antigravity(index, blob)
                for index, blob in raw_rows
            ),
        )
        if result.identity_lengths != tuple(self.context.test_identity_manifest[dataset]):
            raise ValueError("TEST values do not match the frozen identity manifest")
        return result


def _assert_shared_frozen_identity() -> None:
    """Require the exact Exp417 source fields instead of a compatible fork."""

    FROZEN_CONFIG.validate()
    for field_name in ("expected_datasets", "database_sha256", *_LANE_HASH_FIELDS):
        if getattr(FROZEN_CONFIG, field_name) != getattr(
            EXP417_FROZEN_CONFIG, field_name
        ):
            raise ValueError(f"Exp436/Exp417 frozen input mismatch: {field_name}")


def _validate_population(population: object, *, source: str) -> tuple[str, ...]:
    """Require exact sorted unique frozen names before a snapshot may open."""

    if not isinstance(population, tuple):
        raise ValueError(f"{source} population must be a tuple")
    if len(population) != FROZEN_CONFIG.expected_datasets:
        raise ValueError(
            f"{source} population must contain exactly "
            f"{FROZEN_CONFIG.expected_datasets} sorted unique names"
        )
    if any(type(name) is not str or not name.strip() for name in population):
        raise ValueError(f"{source} population contains an invalid dataset name")
    if population != tuple(sorted(population)) or len(population) != len(set(population)):
        raise ValueError(
            f"{source} population must contain exactly "
            f"{FROZEN_CONFIG.expected_datasets} sorted unique names"
        )
    return population


@contextmanager
def open_exp436_input_session(
    database: Path = DEFAULT_DATABASE,
    exp291_prediction: Path = DEFAULT_EXP291_PREDICTION,
    exp291_manifest: Path = DEFAULT_EXP291_MANIFEST,
    exp292_prediction: Path = DEFAULT_EXP292_PREDICTION,
    exp292_manifest: Path = DEFAULT_EXP292_MANIFEST,
    *,
    verified_code_commit: str,
) -> Iterator[Exp436InputSession]:
    """Open exactly one lane-gated descriptor snapshot and freeze its context."""

    _assert_shared_frozen_identity()
    sources = canonical_source_paths_antigravity(database, exp291_prediction, exp291_manifest, exp292_prediction, exp292_manifest)
    lanes = freeze_lane_inputs(
        sources.exp291_prediction,
        sources.exp291_manifest,
        sources.exp292_prediction,
        sources.exp292_manifest,
        config=EXP417_FROZEN_CONFIG,
    )
    lane_population = _validate_population(lanes.population, source="frozen lane")
    if (
        type(verified_code_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", verified_code_commit) is None
    ):
        raise ValueError("Exp436 reviewed code commit is invalid")
    with open_database_snapshot(
        sources.database, expected_sha256=FROZEN_CONFIG.database_sha256
    ) as snapshot:
        context = snapshot.freeze_context(lanes, verified_code_commit=verified_code_commit)
        context_population = _validate_population(
            context.datasets, source="frozen context"
        )
        if context_population != lane_population:
            raise ValueError("frozen lane and context populations do not match")
        session = Exp436InputSession(snapshot, context, sources)
        try:
            yield session
        finally:
            session._close_antigravity()


__all__ = [
    "DEFAULT_DATABASE",
    "DEFAULT_EXP291_MANIFEST",
    "DEFAULT_EXP291_PREDICTION",
    "DEFAULT_EXP292_MANIFEST",
    "DEFAULT_EXP292_PREDICTION",
    "Exp436InputSession",
    "Exp436CanonicalSourcePaths",
    "canonical_source_paths_antigravity",
    "decode_raw_value_row_antigravity",
    "open_exp436_input_session",
]
