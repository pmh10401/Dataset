"""Descriptor-bound, label-free input boundary for Exp435 prediction.

Exp435 owns the session lifecycle but deliberately delegates SQLite decoding to
the verified Exp425 snapshot and the frozen Exp417 values-only loaders.  The
lane artifacts are hashed before the database is opened; the resulting lane
population is the authority for dataset selection, so names that exist only in
the database can never enter the prediction context.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re

from experiments.exp417_causal_multihorizon_shadow.config import (
    FROZEN_CONFIG as EXP417_FROZEN_CONFIG,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    FrozenRunContext,
    RawTestDataset,
    RawTrainDataset,
)
from experiments.exp417_causal_multihorizon_shadow.inputs import (
    freeze_lane_inputs,
)
from experiments.exp425_selective_recovery.database import (
    Exp425DatabaseSnapshot,
    open_database_snapshot,
)

from .config_antigravity import FROZEN_CONFIG


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


@dataclass
class Exp435InputSession:
    """One live Exp425 snapshot bound to one frozen Exp417 run context."""

    database: Exp425DatabaseSnapshot
    context: FrozenRunContext
    label_access_count: int = 0

    def load_train(self, dataset: str) -> RawTrainDataset:
        """Load normal TRAIN values through the held descriptor."""

        return self.database.load_train(dataset, self.context)

    def load_test_values(
        self, dataset: str, train: RawTrainDataset
    ) -> RawTestDataset:
        """Load TEST values only after actual normal TRAIN values exist."""

        loader = getattr(self.database, "load_test_values_after_train", None)
        if not callable(loader):
            raise ValueError("database snapshot lacks the TRAIN-bound TEST values loader")
        return loader(dataset, train, self.context)


def _assert_shared_frozen_identity() -> None:
    """Ensure Exp435 reuses, rather than silently forks, Exp417 identities."""

    FROZEN_CONFIG.validate()
    for field in ("expected_datasets", "database_sha256", *_LANE_HASH_FIELDS):
        if getattr(FROZEN_CONFIG, field) != getattr(EXP417_FROZEN_CONFIG, field):
            raise ValueError(f"Exp435/Exp417 frozen input mismatch: {field}")


def _validate_population(population: object, *, source: str) -> tuple[str, ...]:
    """Require the exact sorted lane population before opening a snapshot."""

    if not isinstance(population, tuple):
        raise ValueError(f"{source} population must be a tuple")
    if len(population) != FROZEN_CONFIG.expected_datasets:
        raise ValueError(
            f"{source} population must contain exactly "
            f"{FROZEN_CONFIG.expected_datasets} sorted unique names"
        )
    if any(not isinstance(name, str) or not name.strip() for name in population):
        raise ValueError(f"{source} population contains an invalid dataset name")
    if population != tuple(sorted(population)) or len(population) != len(set(population)):
        raise ValueError(
            f"{source} population must contain exactly "
            f"{FROZEN_CONFIG.expected_datasets} sorted unique names"
        )
    return population


@contextmanager
def open_exp435_input_session(
    database: Path = DEFAULT_DATABASE,
    exp291_prediction: Path = DEFAULT_EXP291_PREDICTION,
    exp291_manifest: Path = DEFAULT_EXP291_MANIFEST,
    exp292_prediction: Path = DEFAULT_EXP292_PREDICTION,
    exp292_manifest: Path = DEFAULT_EXP292_MANIFEST,
    *,
    verified_code_commit: str,
) -> Iterator[Exp435InputSession]:
    """Open one hash-gated, descriptor-bound Exp435 input session.

    Lane files are verified first.  Only the lane population is passed through
    to Exp425's context freezer; the database may contain additional names, but
    those names are never admitted to the session.
    """

    _assert_shared_frozen_identity()
    lanes = freeze_lane_inputs(
        Path(exp291_prediction),
        Path(exp291_manifest),
        Path(exp292_prediction),
        Path(exp292_manifest),
        config=EXP417_FROZEN_CONFIG,
    )
    lane_population = _validate_population(lanes.population, source="frozen lane")
    if not isinstance(verified_code_commit, str) or re.fullmatch(r"[0-9a-f]{40}", verified_code_commit) is None:
        raise ValueError("Exp435 reviewed code commit is invalid")

    # The Exp425 manager closes the retained descriptors on all paths, including
    # lane/context failures and exceptions raised by the prediction caller.
    with open_database_snapshot(
        Path(database), expected_sha256=FROZEN_CONFIG.database_sha256
    ) as snapshot:
        context = snapshot.freeze_context(
            lanes, verified_code_commit=verified_code_commit
        )
        datasets = _validate_population(context.datasets, source="frozen context")
        if datasets != lane_population:
            raise ValueError("frozen lane and context populations do not match")
        yield Exp435InputSession(snapshot, context)


__all__ = [
    "DEFAULT_DATABASE",
    "DEFAULT_EXP291_MANIFEST",
    "DEFAULT_EXP291_PREDICTION",
    "DEFAULT_EXP292_MANIFEST",
    "DEFAULT_EXP292_PREDICTION",
    "Exp435InputSession",
    "open_exp435_input_session",
]
