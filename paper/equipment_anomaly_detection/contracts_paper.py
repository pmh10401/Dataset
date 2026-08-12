"""Immutable values shared by the paper rerun provenance boundary."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _evidence_float_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite numeric vector")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite numeric vector")
    return result


def _evidence_bool_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be a bool vector")
    return np.array(array, dtype=bool, copy=True)


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def _deeply_immutable(value: object) -> object:
    """Detach nested manifest data so public freeze objects have no mutable leaf."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deeply_immutable(nested) for key, nested in value.items()})
    if isinstance(value, list):
        return tuple(_deeply_immutable(nested) for nested in value)
    if isinstance(value, tuple):
        return tuple(_deeply_immutable(nested) for nested in value)
    return value


@dataclass(frozen=True, slots=True)
class LineageNode:
    node_id: str
    stage: str
    input_kind: str
    module: str
    symbols: tuple[str, ...]
    source_sha256: str | None
    paper_module: str | None
    label_access: bool
    parity_oracle: str | None = None


@dataclass(frozen=True, slots=True)
class FrozenPaperConfig:
    run_identity: str = "paper-proposal-clean-rerun-20260812"
    database_path: str = "/Users/mac/Documents/Dataset/univariate_ts.db"
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    excluded_datasets: tuple[str, str] = ("CornellWhaleChallenge", "Wafer_normal_1")
    dataset_count: int = 1117
    test_row_count: int = 195900
    seed: int = 20260812
    ablation_alert_counts: tuple[int, int, int, int] = (2218, 2248, 2244, 2222)


@dataclass(frozen=True, slots=True)
class Task8FrozenManifest:
    """A standalone, pre-parity Task 8 proposal inventory."""

    schema: str
    catalog_sha256: str
    datasets: tuple[str, ...]
    test_row_count: int
    shard_hashes: Mapping[str, str]
    parity_status: str
    acceptance_path: str | None
    acceptance_sha256: str | None
    acceptance_file_sha256: str | None

    def __post_init__(self) -> None:
        hashes = dict(self.shard_hashes)
        if self.schema != "paper-task8-proposal-manifest.v1" or self.datasets != tuple(sorted(self.datasets)) or not self.datasets:
            raise ValueError("Task 8 proposal manifest inventory is invalid")
        if set(hashes) != set(self.datasets) or type(self.test_row_count) is not int or self.test_row_count < 0:
            raise ValueError("Task 8 proposal manifest coverage is invalid")
        if type(self.catalog_sha256) is not str or len(self.catalog_sha256) != 64 or any(type(item) is not str or len(item) != 64 for item in hashes.values()):
            raise ValueError("Task 8 proposal manifest hashes are invalid")
        if self.parity_status == "not_available":
            if any(value is not None for value in (self.acceptance_path, self.acceptance_sha256, self.acceptance_file_sha256)):
                raise ValueError("Task 8 unavailable parity manifest has acceptance authority")
        elif self.parity_status == "available_not_run":
            if type(self.acceptance_path) is not str or not self.acceptance_path or any(type(item) is not str or len(item) != 64 for item in (self.acceptance_sha256, self.acceptance_file_sha256)):
                raise ValueError("Task 8 available parity manifest lacks acceptance authority")
        else:
            raise ValueError("Task 8 proposal parity status is invalid")
        object.__setattr__(self, "shard_hashes", MappingProxyType(hashes))


@dataclass(frozen=True, slots=True)
class FrozenTask8BackendDecision:
    """The only Task 8 backend declaration until Task 9 owns acceleration."""

    selected_backend: str = "cpu"
    reason: str = "task9_pending"
    verdict: str = "cpu_fallback"

    def __post_init__(self) -> None:
        if (self.selected_backend, self.reason, self.verdict) != ("cpu", "task9_pending", "cpu_fallback"):
            raise ValueError("Task 8 backend decision is invalid")


@dataclass(frozen=True, slots=True)
class RawDataset:
    """Detached, index-aligned raw value rows used by the label-free score stages."""

    dataset: str
    metadata_length: int
    train_indices: tuple[int, ...]
    train_blobs: tuple[bytes, ...]
    train_rows: tuple[np.ndarray, ...]
    test_indices: tuple[int, ...]
    test_blobs: tuple[bytes, ...]
    test_rows: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class SourceDatasetMetadata:
    """TRAIN-derived source configuration metadata with no label capability."""

    dataset: str
    dataset_name: str
    family: str
    metadata_length: int
    actual_len_median: int
    actual_len_max: int
    train_target_len: int
    test_actual_len_median: int


@dataclass(frozen=True, slots=True)
class LeafEvidence:
    """Immutable continuous evidence for one score-source leaf."""

    pvalue: np.ndarray
    margin: np.ndarray
    strength: np.ndarray
    native_pass: np.ndarray
    evidence_missing: np.ndarray

    def __post_init__(self) -> None:
        """Defensively bind validated, read-only evidence arrays at the public boundary."""
        pvalue = _evidence_float_vector(self.pvalue, "pvalue")
        margin = _evidence_float_vector(self.margin, "margin")
        strength = _evidence_float_vector(self.strength, "strength")
        native_pass = _evidence_bool_vector(self.native_pass, "native_pass")
        evidence_missing = _evidence_bool_vector(self.evidence_missing, "evidence_missing")
        fields = (margin, strength, native_pass, evidence_missing)
        if any(field.shape != pvalue.shape for field in fields):
            raise ValueError("LeafEvidence fields must be aligned")
        if np.any((pvalue <= 0.0) | (pvalue > 1.0)):
            raise ValueError("pvalue must be in (0, 1]")
        if np.any((strength < 0.0) | (strength >= 1.0)):
            raise ValueError("strength must be in [0, 1)")
        object.__setattr__(self, "pvalue", _readonly(pvalue))
        object.__setattr__(self, "margin", _readonly(margin))
        object.__setattr__(self, "strength", _readonly(strength))
        object.__setattr__(self, "native_pass", _readonly(native_pass))
        object.__setattr__(self, "evidence_missing", _readonly(evidence_missing))


@dataclass(frozen=True, slots=True)
class PaperCodemap:
    required_fields: tuple[str, ...]
    modules: dict[str, dict[str, object]]


def _low_frequency_float(value: object, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class LowFrequencyDepthEvidence:
    """One immutable raw Exp440 depth score and its TRAIN-only calibration."""

    depth: int
    score: float
    threshold: float
    strict_pass: bool
    excess_ratio: float
    empirical_rank: float

    def __post_init__(self) -> None:
        if type(self.depth) is not int or self.depth < 1:
            raise ValueError("low-frequency depth is invalid")
        score = _low_frequency_float(self.score, "low-frequency score", nonnegative=True)
        threshold = _low_frequency_float(self.threshold, "low-frequency threshold", nonnegative=True)
        ratio = _low_frequency_float(self.excess_ratio, "low-frequency excess ratio", nonnegative=True)
        rank = _low_frequency_float(self.empirical_rank, "low-frequency empirical rank", positive=True)
        if rank > 1.0 or type(self.strict_pass) is not bool or self.strict_pass != (score > threshold):
            raise ValueError("low-frequency depth evidence is inconsistent")
        if not np.isclose(ratio, score / max(threshold, 1e-12), rtol=1e-12, atol=1e-12):
            raise ValueError("low-frequency excess ratio is inconsistent")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "excess_ratio", ratio)
        object.__setattr__(self, "empirical_rank", rank)


@dataclass(frozen=True, slots=True)
class LowFrequencyCandidate:
    """A strict two-depth provisional coordinate, before Hard-empty selection."""

    dataset: str
    instance_index: int
    raw_length: int
    depths: tuple[int, ...]
    second_excess_ratio: float
    second_empirical_rank: float
    depth_evidence: tuple[LowFrequencyDepthEvidence, ...] = ()

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset or type(self.instance_index) is not int or self.instance_index < 0 or type(self.raw_length) is not int or self.raw_length < 1:
            raise ValueError("low-frequency candidate identity is invalid")
        depths = tuple(self.depths)
        if len(depths) < 2 or depths != tuple(sorted(set(depths))) or any(type(depth) is not int or depth < 1 for depth in depths):
            raise ValueError("low-frequency candidate depths are invalid")
        ratio = _low_frequency_float(self.second_excess_ratio, "second low-frequency excess ratio", nonnegative=True)
        rank = _low_frequency_float(self.second_empirical_rank, "second low-frequency empirical rank", positive=True)
        if rank > 1.0:
            raise ValueError("second low-frequency empirical rank is invalid")
        evidence = tuple(self.depth_evidence)
        if evidence and (
            any(not isinstance(item, LowFrequencyDepthEvidence) for item in evidence)
            or tuple(item.depth for item in evidence) != tuple(sorted(set(item.depth for item in evidence)))
            or tuple(item.depth for item in evidence if item.strict_pass) != depths
            or not np.isclose(ratio, sorted((item.excess_ratio for item in evidence if item.strict_pass), reverse=True)[1], rtol=1e-12, atol=1e-12)
            or not np.isclose(rank, sorted(item.empirical_rank for item in evidence if item.strict_pass)[1], rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("low-frequency candidate evidence is inconsistent")
        object.__setattr__(self, "depths", depths)
        object.__setattr__(self, "second_excess_ratio", ratio)
        object.__setattr__(self, "second_empirical_rank", rank)
        object.__setattr__(self, "depth_evidence", evidence)

    @property
    def coordinate(self) -> tuple[str, int]:
        return self.dataset, self.instance_index

    @property
    def dataset_key(self) -> tuple[float, float, int]:
        return -self.second_excess_ratio, self.second_empirical_rank, self.instance_index

    @property
    def global_key(self) -> tuple[float, float, str, int]:
        return -self.second_excess_ratio, self.second_empirical_rank, self.dataset, self.instance_index


@dataclass(frozen=True, slots=True)
class LowFrequencyRow:
    """A complete TEST-row audit record; unavailable evidence is explicit."""

    dataset: str
    instance_index: int
    raw_length: int
    depth_evidence: tuple[LowFrequencyDepthEvidence, ...]
    strict_pass: bool
    second_excess_ratio: float | None
    second_empirical_rank: float | None
    provisional: bool
    evidence_missing: bool
    missing_reason: str | None

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset or type(self.instance_index) is not int or self.instance_index < 0 or type(self.raw_length) is not int or self.raw_length < 1:
            raise ValueError("low-frequency row identity is invalid")
        evidence = tuple(self.depth_evidence)
        if any(not isinstance(item, LowFrequencyDepthEvidence) for item in evidence):
            raise ValueError("low-frequency row evidence is invalid")
        if type(self.strict_pass) is not bool or type(self.provisional) is not bool or type(self.evidence_missing) is not bool:
            raise ValueError("low-frequency row flags are invalid")
        if self.evidence_missing:
            if evidence or self.strict_pass or self.provisional or self.second_excess_ratio is not None or self.second_empirical_rank is not None or type(self.missing_reason) is not str or not self.missing_reason:
                raise ValueError("missing low-frequency evidence is inconsistent")
        else:
            if type(self.missing_reason) is not type(None) or not evidence:
                raise ValueError("available low-frequency evidence is inconsistent")
            strict = sum(item.strict_pass for item in evidence) >= 2
            if self.strict_pass != strict or self.provisional != strict:
                raise ValueError("low-frequency provisional status is inconsistent")
            if strict:
                ratios = sorted((item.excess_ratio for item in evidence if item.strict_pass), reverse=True)
                ranks = sorted(item.empirical_rank for item in evidence if item.strict_pass)
                if not np.isclose(_low_frequency_float(self.second_excess_ratio, "second low-frequency excess ratio"), ratios[1], rtol=1e-12, atol=1e-12) or not np.isclose(_low_frequency_float(self.second_empirical_rank, "second low-frequency empirical rank", positive=True), ranks[1], rtol=1e-12, atol=1e-12):
                    raise ValueError("low-frequency second-depth evidence is inconsistent")
            elif self.second_excess_ratio is not None or self.second_empirical_rank is not None:
                raise ValueError("non-provisional low-frequency row has ranking evidence")
        object.__setattr__(self, "depth_evidence", evidence)


@dataclass(frozen=True, slots=True)
class LowFrequencyResult:
    """Full source replay result containing all TEST rows and global selection."""

    datasets: tuple[str, ...]
    rows: tuple[LowFrequencyRow, ...]
    provisional_candidates: tuple[LowFrequencyCandidate, ...]
    selected_candidates: tuple[LowFrequencyCandidate, ...]
    excluded_train_rows: int
    hard_lanes: Mapping[str, frozenset[int]]

    def __post_init__(self) -> None:
        datasets = tuple(self.datasets)
        if not datasets or datasets != tuple(sorted(set(datasets))) or any(type(dataset) is not str or not dataset for dataset in datasets) or type(self.excluded_train_rows) is not int or self.excluded_train_rows < 0:
            raise ValueError("low-frequency result identity is invalid")
        rows, candidates, selected = tuple(self.rows), tuple(self.provisional_candidates), tuple(self.selected_candidates)
        if any(not isinstance(row, LowFrequencyRow) or row.dataset not in datasets for row in rows) or len({(row.dataset, row.instance_index) for row in rows}) != len(rows) or any(tuple(row.instance_index for row in rows if row.dataset == dataset) != tuple(range(sum(row.dataset == dataset for row in rows))) for dataset in datasets):
            raise ValueError("low-frequency TEST row coverage is invalid")
        if any(not isinstance(candidate, LowFrequencyCandidate) or candidate.dataset not in datasets for candidate in candidates) or len({candidate.coordinate for candidate in candidates}) != len(candidates):
            raise ValueError("low-frequency provisional candidates are invalid")
        rows_by_coordinate = {(row.dataset, row.instance_index): row for row in rows}
        provisional_rows = tuple(row for row in rows if row.provisional)
        if len({(row.dataset, row.instance_index) for row in provisional_rows}) != len(provisional_rows) or {candidate.coordinate for candidate in candidates} != {(row.dataset, row.instance_index) for row in provisional_rows}:
            raise ValueError("low-frequency row and candidate coverage is not bijective")
        if any(
            candidate.coordinate not in rows_by_coordinate
            or not rows_by_coordinate[candidate.coordinate].provisional
            or candidate.dataset != rows_by_coordinate[candidate.coordinate].dataset
            or candidate.instance_index != rows_by_coordinate[candidate.coordinate].instance_index
            or candidate.raw_length != rows_by_coordinate[candidate.coordinate].raw_length
            or candidate.depth_evidence != rows_by_coordinate[candidate.coordinate].depth_evidence
            or candidate.second_excess_ratio != rows_by_coordinate[candidate.coordinate].second_excess_ratio
            or candidate.second_empirical_rank != rows_by_coordinate[candidate.coordinate].second_empirical_rank
            for candidate in candidates
        ):
            raise ValueError("low-frequency provisional candidates do not match row evidence")
        lanes = dict(self.hard_lanes)
        if set(lanes) != set(datasets) or any(not isinstance(indices, frozenset) or any(type(index) is not int or index < 0 for index in indices) for indices in lanes.values()):
            raise ValueError("low-frequency Hard lanes are invalid")
        eligible: dict[str, LowFrequencyCandidate] = {}
        for candidate in candidates:
            if not lanes[candidate.dataset]:
                previous = eligible.get(candidate.dataset)
                if previous is None or candidate.dataset_key < previous.dataset_key:
                    eligible[candidate.dataset] = candidate
        expected = tuple(sorted(eligible.values(), key=lambda candidate: candidate.global_key)[:30])
        if selected != expected:
            raise ValueError("low-frequency selected candidates are invalid")
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "provisional_candidates", candidates)
        object.__setattr__(self, "selected_candidates", selected)
        object.__setattr__(self, "hard_lanes", MappingProxyType(dict(lanes)))
