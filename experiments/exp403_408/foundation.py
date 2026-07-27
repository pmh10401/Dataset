"""Values-only data loading and frozen lane validation for Exp403-408."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from types import MappingProxyType
from typing import Any

import numpy as np

from experiments.exp226_235_hard_alert_quality.config import DB_PATH
from experiments.exp226_235_hard_alert_quality.contracts import DatasetBundle
from experiments.exp226_235_hard_alert_quality.database import load_values_bundle
from experiments.exp226_235_hard_alert_quality.preprocessing import (
    PREPROCESSING_ORDER,
    align_bundle,
    preprocessing_contract,
)

from .artifact_io import verify_sha256
from .contracts import (
    EXPECTED_DATASETS,
    EXPECTED_HARD,
    EXPECTED_PRIORITY,
    EXPECTED_STANDARD,
    SEED,
    DatasetInput,
    Fold,
)


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
EXP291_PREDICTION_PATH = (
    DATA_ROOT
    / "exp291_final_operational_routing_freeze"
    / "exp291_frozen_operational_routing.jsonl"
)
EXP291_MANIFEST_PATH = EXP291_PREDICTION_PATH.with_name("exp291_prediction_manifest.json")
EXP291_SHA256 = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"
EXP292_PREDICTION_PATH = (
    DATA_ROOT
    / "exp292_standard_review_evidence_tiering"
    / "exp292_frozen_review_tiers.jsonl"
)
EXP292_MANIFEST_PATH = EXP292_PREDICTION_PATH.with_name("exp292_prediction_manifest.json")
EXP292_SHA256 = "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b"


@dataclass(frozen=True)
class _PreprocessedDatasetInput(DatasetInput):
    """Task 2 extension that retains the upstream preprocessing provenance."""

    preprocessing_order: tuple[str, ...] = PREPROCESSING_ORDER


class _CrossfitFold(Fold):
    """A frozen Fold that also supports the brief's ``for fit, query`` form."""

    def __iter__(self):
        yield self.train_indices
        yield self.validation_indices


ValuesLoader = Callable[[str, Path], DatasetBundle]
FrozenLaneMap = dict[str, dict[str, frozenset[int]]]
_FROZEN_LANE_UNIVERSE_PROOF = object()


class FrozenLaneUniverse:
    """Immutable lanes created only after the full Exp291/Exp292 validation gate."""

    __slots__ = ("_lanes",)

    def __init__(
        self, lanes: Mapping[str, Mapping[str, Iterable[int]]], *, _proof: object
    ) -> None:
        if _proof is not _FROZEN_LANE_UNIVERSE_PROOF:
            raise TypeError("FrozenLaneUniverse instances must come from load_frozen_lanes")
        self._lanes = MappingProxyType({
            str(dataset): MappingProxyType({
                "hard": frozenset(int(index) for index in lane.get("hard", ())),
                "standard": frozenset(int(index) for index in lane.get("standard", ())),
                "priority": frozenset(int(index) for index in lane.get("priority", ())),
            })
            for dataset, lane in lanes.items()
        })

    def lane_for(self, dataset: str) -> Mapping[str, frozenset[int]]:
        return self._lanes.get(str(dataset), MappingProxyType({}))


def load_dataset_input(
    dataset: str,
    *,
    db_path: Path = DB_PATH,
    loader: ValuesLoader = load_values_bundle,
    frozen_lane_universe: FrozenLaneUniverse | None = None,
) -> DatasetInput:
    """Load only values, apply the frozen preprocessing order, and attach lanes."""
    contract = preprocessing_contract()
    if tuple(contract.get("order", ())) != PREPROCESSING_ORDER:
        raise ValueError(f"preprocessing contract mismatch: {contract}")
    raw = loader(str(dataset), Path(db_path))
    aligned = align_bundle(raw)
    train = _z_normalize_per_run(np.asarray(aligned.train_normal, dtype=np.float32))
    test = _z_normalize_per_run(np.asarray(aligned.test_values, dtype=np.float32))
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError(f"non-finite preprocessed values: dataset={dataset}")
    universe = load_frozen_lanes() if frozen_lane_universe is None else frozen_lane_universe
    if not isinstance(universe, FrozenLaneUniverse):
        raise TypeError("frozen_lane_universe must be a verified FrozenLaneUniverse")
    lane = universe.lane_for(str(dataset))
    return _PreprocessedDatasetInput(
        dataset=str(dataset),
        train_normal=train,
        test_values=test,
        target_length=int(aligned.target_length),
        hard_indices=frozenset(int(index) for index in lane.get("hard", ())),
        standard_indices=frozenset(int(index) for index in lane.get("standard", ())),
        priority_indices=frozenset(int(index) for index in lane.get("priority", ())),
    )


def list_datasets_small_to_large(db_path: Path = DB_PATH) -> list[str]:
    """Return dataset keys ordered by TEST population using a read-only connection."""
    resolved = Path(db_path).resolve(strict=True)
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT d.name, COUNT(i.id) AS n_test
            FROM datasets AS d
            JOIN instances AS i ON i.dataset_id = d.id AND i.split = 'TEST'
            GROUP BY d.id, d.name
            ORDER BY n_test ASC, d.name ASC
            """
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def crossfit_plan(
    n_train: int, seed: int = SEED, folds: int = 5
) -> tuple[Fold, ...]:
    """Create deterministic disjoint TRAIN fit/validation folds."""
    size = int(n_train)
    requested_folds = int(folds)
    if size < 2:
        raise ValueError("cross-fit requires at least two TRAIN rows")
    if requested_folds < 2:
        raise ValueError("cross-fit requires at least two folds")
    shuffled = np.arange(size, dtype=int)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    validation_parts = tuple(
        np.sort(part)
        for part in np.array_split(shuffled, min(size, requested_folds))
        if len(part)
    )
    result: list[Fold] = []
    for index, validation in enumerate(validation_parts):
        fit = np.sort(np.concatenate(
            [part for other, part in enumerate(validation_parts) if other != index]
        ))
        result.append(
            _CrossfitFold(
                train_indices=tuple(int(value) for value in fit),
                validation_indices=tuple(int(value) for value in validation),
            )
        )
    return tuple(result)


def load_frozen_lanes(
    *,
    exp291_prediction_path: Path = EXP291_PREDICTION_PATH,
    exp291_manifest_path: Path = EXP291_MANIFEST_PATH,
    exp292_prediction_path: Path = EXP292_PREDICTION_PATH,
    exp292_manifest_path: Path = EXP292_MANIFEST_PATH,
) -> FrozenLaneUniverse:
    """Load matching Exp291/292 lane snapshots only after hash validation."""
    exp291 = _load_verified_frozen_rows(
        exp291_prediction_path, exp291_manifest_path, EXP291_SHA256, "Exp291"
    )
    exp292 = _load_verified_frozen_rows(
        exp292_prediction_path, exp292_manifest_path, EXP292_SHA256, "Exp292"
    )
    validate_frozen_universe(exp291)
    validate_frozen_universe(exp292)
    lanes = _lane_map(exp291)
    if lanes != _lane_map(exp292):
        raise ValueError("Exp291/Exp292 frozen lane mismatch")
    return FrozenLaneUniverse(lanes, _proof=_FROZEN_LANE_UNIVERSE_PROOF)


def validate_frozen_universe(rows: Sequence[Mapping[str, object]]) -> None:
    """Reject lane files that cannot preserve the frozen Exp291/Exp292 universe."""
    hard_total = _lane_total(rows, "automatic_hard_indices")
    standard_total = _lane_total(rows, "standard_review_indices")
    priority_total = _lane_total(rows, "priority_review_indices")
    if hard_total != EXPECTED_HARD:
        raise ValueError(f"Hard lane total mismatch: {hard_total}/{EXPECTED_HARD}")
    if standard_total != EXPECTED_STANDARD:
        raise ValueError(
            f"Standard lane total mismatch: {standard_total}/{EXPECTED_STANDARD}"
        )
    if priority_total != EXPECTED_PRIORITY:
        raise ValueError(
            f"Priority lane total mismatch: {priority_total}/{EXPECTED_PRIORITY}"
        )
    names = [str(row.get("dataset", "")) for row in rows]
    if len(names) != EXPECTED_DATASETS or not all(names) or len(names) != len(set(names)):
        raise ValueError(f"dataset universe mismatch: {len(names)}/{EXPECTED_DATASETS}")


def _z_normalize_per_run(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    center = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return ((values - center) / np.where(scale > 1e-8, scale, 1.0)).astype(np.float32)


def _load_verified_frozen_rows(
    prediction_path: Path, manifest_path: Path, expected_sha256: str, source: str
) -> list[dict[str, object]]:
    verify_sha256(prediction_path, expected_sha256)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("prediction_sha256") != expected_sha256:
        raise ValueError(f"{source} manifest SHA256 mismatch")
    if manifest.get("contains_test_labels") is not False:
        raise ValueError(f"{source} manifest contains TEST labels")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError(f"{source} manifest contains evaluation metrics")
    rows = [
        json.loads(line)
        for line in Path(prediction_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError(f"{source} manifest row count mismatch")
    return rows


def _lane_map(rows: Sequence[Mapping[str, object]]) -> FrozenLaneMap:
    return {
        str(row["dataset"]): {
            "hard": _validated_indices(row, "automatic_hard_indices"),
            "standard": _validated_indices(row, "standard_review_indices"),
            "priority": _validated_indices(row, "priority_review_indices"),
        }
        for row in rows
    }


def _lane_total(rows: Sequence[Mapping[str, object]], field: str) -> int:
    return sum(len(_validated_indices(row, field)) for row in rows)


def _validated_indices(row: Mapping[str, object], field: str) -> frozenset[int]:
    try:
        n_test = int(row["n_test"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{field} missing n_test") from error
    if n_test < 1:
        raise ValueError(f"{field} has invalid n_test")
    indices: set[int] = set()
    values: Any = row.get(field, ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{field} contains invalid index")
        try:
            index = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} contains invalid index") from error
        if index < 0 or index >= n_test:
            raise ValueError(f"{field} index out of range")
        indices.add(index)
    if len(indices) != len(values):
        raise ValueError(f"{field} contains duplicate indices")
    return frozenset(indices)
