"""Label-blind value contracts for Exp409 readiness publication."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass

from .config import FROZEN_CONFIG


@dataclass(frozen=True)
class SourceStatus:
    source: str
    status: str
    reason: str = ""


@dataclass(frozen=True)
class ShadowCandidate:
    dataset: str
    candidate_index: int
    variant: str
    alpha: str
    supporting_components: tuple[tuple[str, ...], ...]
    supporting_source_markers: tuple[tuple[str, ...], ...]
    disposition: str = "review_only"

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetReadinessResult:
    dataset: str
    train_size: int
    test_size: int
    hard_indices: tuple[int, ...]
    standard_indices: tuple[int, ...]
    priority_indices: tuple[int, ...]
    source_statuses: tuple[SourceStatus, ...]
    candidates: tuple[ShadowCandidate, ...]
    source_output_sha256: Mapping[str, str]
    status: str
    abstention_reason: str = ""


@dataclass(frozen=True)
class ReadinessBatch:
    dataset_results: tuple[DatasetReadinessResult, ...]

    @property
    def candidates(self) -> tuple[ShadowCandidate, ...]:
        return tuple(
            candidate
            for result in self.dataset_results
            for candidate in result.candidates
        )


def validate_candidate(
    candidate: ShadowCandidate,
    *,
    n_test: int,
    hard_indices: Collection[int],
) -> None:
    if not candidate.dataset:
        raise ValueError("Exp409 candidate requires dataset identity")
    index = candidate.candidate_index
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n_test:
        raise ValueError("Exp409 candidate index is outside TEST range")
    if index in {int(value) for value in hard_indices}:
        raise ValueError("Exp409 candidate overlaps frozen Hard")
    if candidate.variant != FROZEN_CONFIG.variant:
        raise ValueError("Exp409 candidate variant is not frozen")
    if candidate.alpha != FROZEN_CONFIG.alpha:
        raise ValueError("Exp409 candidate alpha is not frozen")
    if candidate.disposition != FROZEN_CONFIG.disposition:
        raise ValueError("Exp409 candidate must remain review_only")
    if len(candidate.supporting_components) < 3:
        raise ValueError("Exp409 candidate lacks independent support")
    if len(candidate.supporting_source_markers) != len(
        candidate.supporting_components
    ):
        raise ValueError("Exp409 candidate support markers are inconsistent")


def validate_batch(batch: ReadinessBatch, *, review_budget: int) -> None:
    if review_budget != FROZEN_CONFIG.review_budget:
        raise ValueError("Exp409 review budget is frozen")
    dataset_names = [result.dataset for result in batch.dataset_results]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("duplicate Exp409 dataset result")
    for result in batch.dataset_results:
        if result.status not in {"ready", "abstained"}:
            raise ValueError("Exp409 dataset result is not publishable")
        identities = [status.source for status in result.source_statuses]
        if len(identities) != 13 or len(set(identities)) != 13:
            raise ValueError("Exp409 requires exactly thirteen source statuses")
        if set(result.source_output_sha256) != set(identities):
            raise ValueError("Exp409 source provenance coverage mismatch")
        for digest in result.source_output_sha256.values():
            if not _is_sha256(digest):
                raise ValueError("Exp409 source provenance hash is invalid")
        for candidate in result.candidates:
            if candidate.dataset != result.dataset:
                raise ValueError("Exp409 candidate dataset mismatch")
            validate_candidate(
                candidate,
                n_test=result.test_size,
                hard_indices=result.hard_indices,
            )
    keys = [
        (candidate.dataset, candidate.candidate_index)
        for candidate in batch.candidates
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate Exp409 candidate")
    if len(keys) > review_budget:
        raise ValueError("Exp409 review budget exceeded")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
