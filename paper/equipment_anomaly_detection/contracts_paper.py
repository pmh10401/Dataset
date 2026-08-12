"""Immutable values shared by the paper rerun provenance boundary."""
from __future__ import annotations

from dataclasses import dataclass

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
