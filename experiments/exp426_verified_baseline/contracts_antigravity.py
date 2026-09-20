"""Immutable, label-free contracts for the Exp426 verified baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import re
from types import MappingProxyType
from typing import Callable, Mapping

from experiments.exp403_408.artifact_io import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_LANES = frozenset({"hard", "standard", "priority", "automatic_normal"})


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is invalid")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _optional_digest(value: object, label: str) -> str | None:
    return None if value is None else _digest(value, label)


@dataclass(frozen=True)
class Exp426Provenance:
    run_identity: str
    git_commit: str
    database_sha256: str
    source_root_sha256: str
    source_progress_sha256: str
    source_record_set_sha256: str
    lane_artifact_sha256: Mapping[str, str]
    fold_contract_sha256: str
    expected_datasets: tuple[str, ...]
    workers: int = 12
    context_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_identity", _name(self.run_identity, "run_identity"))
        if not isinstance(self.git_commit, str) or _COMMIT.fullmatch(self.git_commit) is None:
            raise ValueError("git_commit must be a lowercase 40-character commit")
        for field in (
            "database_sha256", "source_root_sha256", "source_progress_sha256",
            "source_record_set_sha256", "fold_contract_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if self.context_sha256 is not None:
            object.__setattr__(self, "context_sha256", _digest(self.context_sha256, "context_sha256"))
        if not isinstance(self.lane_artifact_sha256, Mapping) or set(self.lane_artifact_sha256) != _LANES:
            raise ValueError("lane_artifact_sha256 must contain exactly four lanes")
        lanes = {name: _digest(value, f"lane {name}") for name, value in self.lane_artifact_sha256.items()}
        object.__setattr__(self, "lane_artifact_sha256", MappingProxyType(dict(sorted(lanes.items()))))
        datasets = tuple(self.expected_datasets)
        if (
            not datasets or datasets != tuple(sorted(datasets))
            or len(datasets) != len(set(datasets))
            or any(not isinstance(item, str) or not item for item in datasets)
        ):
            raise ValueError("expected_datasets must be sorted, unique, and non-empty")
        object.__setattr__(self, "expected_datasets", datasets)
        if type(self.workers) is not int or self.workers != 12:
            raise ValueError("workers must equal the frozen value 12")

    def with_context_sha256(self, digest: str) -> "Exp426Provenance":
        return replace(self, context_sha256=_digest(digest, "context_sha256"))

    def to_payload(self) -> dict[str, object]:
        if self.context_sha256 is None:
            raise ValueError("context_sha256 is required")
        return {
            "schema": "exp426.checkpoint-provenance.v1",
            "run_identity": self.run_identity,
            "git_commit": self.git_commit,
            "database_sha256": self.database_sha256,
            "source_root_sha256": self.source_root_sha256,
            "source_progress_sha256": self.source_progress_sha256,
            "source_record_set_sha256": self.source_record_set_sha256,
            "lane_artifact_sha256": dict(self.lane_artifact_sha256),
            "fold_contract_sha256": self.fold_contract_sha256,
            "expected_datasets": list(self.expected_datasets),
            "workers": self.workers,
            "context_sha256": self.context_sha256,
        }


@dataclass(frozen=True)
class Exp426DatasetRecord:
    dataset: str
    origin: str
    status: str
    candidate_coordinates: tuple[tuple[str, int], ...]
    source_record_sha256: str | None
    source_result_sha256: str | None
    train_identity_sha256: str | None
    eligibility_sha256: str | None
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _name(self.dataset, "dataset"))
        if self.origin not in {"exp424_import", "verified_train_abstention"}:
            raise ValueError("origin is invalid")
        if self.status not in {"scored", "abstained"}:
            raise ValueError("status is invalid")
        coordinates = tuple(self.candidate_coordinates)
        checked: list[tuple[str, int]] = []
        for coordinate in coordinates:
            if not isinstance(coordinate, tuple) or len(coordinate) != 2:
                raise ValueError("candidate coordinate is invalid")
            dataset, index = coordinate
            if dataset != self.dataset:
                raise ValueError("candidate dataset does not match record")
            if type(index) is not int or index < 0:
                raise ValueError("candidate index is invalid")
            checked.append((dataset, index))
        if tuple(checked) != tuple(sorted(checked)) or len(checked) != len(set(checked)):
            raise ValueError("candidate coordinates must be sorted and unique")
        object.__setattr__(self, "candidate_coordinates", tuple(checked))
        for field in (
            "source_record_sha256", "source_result_sha256",
            "train_identity_sha256", "eligibility_sha256",
        ):
            object.__setattr__(self, field, _optional_digest(getattr(self, field), field))
        object.__setattr__(self, "reason", _name(self.reason, "reason"))
        if self.source_record_sha256 is None or self.source_result_sha256 is None:
            raise ValueError("source evidence is required")
        if self.origin == "exp424_import":
            if self.status != "scored":
                raise ValueError("imported record status must be scored")
            if self.train_identity_sha256 is not None or self.eligibility_sha256 is not None:
                raise ValueError("imported record cannot claim new TRAIN evidence")
            if self.reason != "verified_exp424_nonterminal":
                raise ValueError("imported record reason is invalid")
        else:
            if self.status != "abstained" or self.candidate_coordinates:
                raise ValueError("verified TRAIN abstention requires no candidates")
            if self.train_identity_sha256 is None or self.eligibility_sha256 is None:
                raise ValueError("verified TRAIN abstention requires TRAIN evidence")
            if self.reason != "oof_identity_insufficient":
                raise ValueError("verified TRAIN abstention reason is invalid")

    def scientific_payload(self) -> dict[str, object]:
        return {
            "schema": "exp426.dataset-record.v1",
            "dataset": self.dataset,
            "origin": self.origin,
            "status": self.status,
            "candidate_coordinates": [list(item) for item in self.candidate_coordinates],
            "source_record_sha256": self.source_record_sha256,
            "source_result_sha256": self.source_result_sha256,
            "train_identity_sha256": self.train_identity_sha256,
            "eligibility_sha256": self.eligibility_sha256,
            "reason": self.reason,
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json_bytes(self.scientific_payload())).hexdigest()

    def validate_against(
        self, test_sizes: Mapping[str, int], lane_contains: Callable[[str, int], bool]
    ) -> None:
        if self.dataset not in test_sizes:
            raise ValueError("record dataset is outside the frozen universe")
        size = test_sizes[self.dataset]
        for dataset, index in self.candidate_coordinates:
            if index >= size:
                raise ValueError("candidate index is outside TEST identity metadata")
            if lane_contains(dataset, index):
                raise ValueError("candidate overlaps a frozen lane")

    @classmethod
    def from_payload(cls, value: object) -> "Exp426DatasetRecord":
        if not isinstance(value, dict) or set(value) != {
            "schema", "dataset", "origin", "status", "candidate_coordinates",
            "source_record_sha256", "source_result_sha256", "train_identity_sha256",
            "eligibility_sha256", "reason",
        } or value.get("schema") != "exp426.dataset-record.v1":
            raise ValueError("dataset record payload is invalid")
        raw_coordinates = value["candidate_coordinates"]
        if not isinstance(raw_coordinates, list):
            raise ValueError("candidate coordinates are invalid")
        coordinates: list[tuple[str, int]] = []
        for item in raw_coordinates:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("candidate coordinate is invalid")
            coordinates.append((item[0], item[1]))  # type: ignore[arg-type]
        return cls(
            value["dataset"], value["origin"], value["status"], tuple(coordinates),
            value["source_record_sha256"], value["source_result_sha256"],
            value["train_identity_sha256"], value["eligibility_sha256"], value["reason"],
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class Exp426Counts:
    total: int
    completed: int
    imported: int
    verified_train_abstentions: int
    abstained: int
    failed: int

    def __post_init__(self) -> None:
        values = (
            self.total, self.completed, self.imported,
            self.verified_train_abstentions, self.abstained, self.failed,
        )
        if any(type(value) is not int or value < 0 for value in values) or self.completed > self.total:
            raise ValueError("counts are invalid")
        if (
            self.imported + self.verified_train_abstentions + self.failed != self.completed
            or self.abstained != self.verified_train_abstentions
        ):
            raise ValueError("count conservation is invalid")

    @property
    def remaining(self) -> int:
        return self.total - self.completed


@dataclass(frozen=True)
class Exp426Progress:
    state: str
    counts: Exp426Counts
    current_dataset: str | None = None
    workers: int = 12

    def __post_init__(self) -> None:
        if self.state not in {
            "initializing", "validating_source", "importing", "verifying_train",
            "finalizing", "interrupted", "complete_review_only", "failed",
        }:
            raise ValueError("progress state is invalid")
        if not isinstance(self.counts, Exp426Counts) or self.workers != 12:
            raise ValueError("progress accounting or workers are invalid")
        if self.current_dataset is not None:
            _name(self.current_dataset, "current_dataset")


@dataclass(frozen=True)
class WorkerEvidence:
    ordinal: int
    dataset: str
    record: Exp426DatasetRecord

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("worker ordinal is invalid")
        if self.dataset != self.record.dataset:
            raise ValueError("worker evidence dataset does not match record")

    @property
    def scientific_sha256(self) -> str:
        return self.record.sha256
