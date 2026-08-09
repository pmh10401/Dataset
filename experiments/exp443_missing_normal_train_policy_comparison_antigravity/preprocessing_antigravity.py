"""Frozen, label-free preprocessing for the Exp443 policy comparison.

Only normal TRAIN handling differs between the three policies.  Every TEST row
uses the Exp249 linear/nearest-edge repair, the target length is computed once
from the original normal TRAIN lengths, and alignment followed by row-wise
float32 z-normalization is shared by all policies.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from .contracts_antigravity import RawSeriesAntigravity, RunPolicyAuditAntigravity


POLICY_A_ANTIGRAVITY = "linear_edge_nearest_control"
POLICY_B_ANTIGRAVITY = "exclude_nonfinite_row"
POLICY_C_ANTIGRAVITY = "leave_one_out_peer_median"
POLICIES_ANTIGRAVITY = (
    POLICY_A_ANTIGRAVITY,
    POLICY_B_ANTIGRAVITY,
    POLICY_C_ANTIGRAVITY,
)

_ACTIONS_ANTIGRAVITY = frozenset({
    "unchanged",
    "linear_and_edge_nearest",
    "excluded_nonfinite_row",
    "leave_one_out_peer_median",
    "excluded_unsupported_peer",
})
_SHA256_HEX_ANTIGRAVITY = frozenset("0123456789abcdef")


def _canonical_json_bytes_antigravity(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest_antigravity(value: str, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _SHA256_HEX_ANTIGRAVITY for char in value):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _freeze_array_antigravity(value: np.ndarray, *, ndim: int, finite: bool = True) -> np.ndarray:
    """Detach an exact little-endian float32 array behind immutable bytes."""
    array = np.asarray(value)
    if array.dtype != np.dtype("<f4") or array.ndim != ndim:
        raise ValueError(f"array must have rank {ndim}")
    contiguous = np.array(array, dtype="<f4", order="C", copy=True)
    if finite and not np.isfinite(contiguous).all():
        raise ValueError("prepared values must be finite")
    payload = contiguous.tobytes(order="C")
    frozen = np.frombuffer(payload, dtype="<f4").reshape(contiguous.shape)
    frozen.setflags(write=False)
    return frozen


def _row_values_antigravity(row: object, *, split: str, index: int) -> np.ndarray:
    if isinstance(row, RawSeriesAntigravity):
        if row.split != split:
            raise ValueError(f"raw row split is not {split}")
        source = row.values
    elif hasattr(row, "values") and not isinstance(row, np.ndarray):
        source = getattr(row, "values")
    else:
        source = row
    try:
        values = np.asarray(source, dtype="<f4").reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{split} row {index} values are invalid") from error
    if values.size < 1:
        raise ValueError(f"{split} row {index} must be non-empty")
    return np.array(values, dtype="<f4", copy=True)


def _row_index_antigravity(row: object, fallback: int) -> int:
    value = getattr(row, "instance_index", fallback)
    if type(value) is not int or value < 0:
        raise ValueError("raw instance index is invalid")
    return value


def _row_digest_antigravity(row: object, values: np.ndarray) -> str:
    actual = sha256(np.asarray(values, dtype="<f4").tobytes(order="C")).hexdigest()
    value = getattr(row, "raw_blob_sha256", None)
    if value is None:
        return actual
    _digest_antigravity(value, "raw_blob_sha256")
    if value != actual:
        raise ValueError("raw_blob_sha256 does not match exact float32 values")
    return actual


def _row_missing_antigravity(row: object, values: np.ndarray) -> tuple[int, ...]:
    value = getattr(row, "missing_positions", None)
    expected = tuple(int(item) for item in np.flatnonzero(~np.isfinite(values)))
    if value is None:
        return expected
    try:
        positions = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError("raw missing positions are invalid") from error
    if positions != expected or positions != tuple(sorted(set(positions))):
        raise ValueError("raw missing positions do not match values")
    return positions


@dataclass(frozen=True)
class _RowAntigravity:
    source: object
    values: np.ndarray
    instance_index: int
    raw_blob_sha256: str
    missing_positions: tuple[int, ...]


def _train_rows_antigravity(raw_or_rows: object) -> tuple[_RowAntigravity, ...]:
    if hasattr(raw_or_rows, "train"):
        source_rows = getattr(raw_or_rows, "train")
    elif hasattr(raw_or_rows, "train_rows"):
        source_rows = getattr(raw_or_rows, "train_rows")
    else:
        source_rows = raw_or_rows
    if isinstance(source_rows, (str, bytes)) or not isinstance(source_rows, Iterable):
        raise ValueError("normal TRAIN rows are invalid")
    rows: list[_RowAntigravity] = []
    for position, row in enumerate(tuple(source_rows)):
        values = _row_values_antigravity(row, split="TRAIN", index=position)
        index = _row_index_antigravity(row, position)
        rows.append(_RowAntigravity(row, values, index, _row_digest_antigravity(row, values), _row_missing_antigravity(row, values)))
    if not rows:
        raise ValueError("normal TRAIN rows are empty")
    if len({row.instance_index for row in rows}) != len(rows):
        raise ValueError("normal TRAIN instance indices are duplicated")
    return tuple(rows)


def _test_rows_antigravity(raw: object) -> tuple[_RowAntigravity, ...]:
    if hasattr(raw, "test"):
        source_rows = getattr(raw, "test")
    elif hasattr(raw, "test_rows"):
        source_rows = getattr(raw, "test_rows")
    else:
        raise ValueError("raw dataset has no TEST rows")
    if isinstance(source_rows, (str, bytes)) or not isinstance(source_rows, Iterable):
        raise ValueError("TEST rows are invalid")
    rows: list[_RowAntigravity] = []
    for position, row in enumerate(tuple(source_rows)):
        values = _row_values_antigravity(row, split="TEST", index=position)
        index = _row_index_antigravity(row, position)
        rows.append(_RowAntigravity(row, values, index, _row_digest_antigravity(row, values), _row_missing_antigravity(row, values)))
    if not rows:
        raise ValueError("TEST rows are empty")
    if len({row.instance_index for row in rows}) != len(rows):
        raise ValueError("TEST instance indices are duplicated")
    return tuple(rows)


def _dataset_name_antigravity(raw: object, rows: Sequence[_RowAntigravity]) -> str:
    value = getattr(raw, "dataset", None)
    if value is None and rows:
        value = getattr(rows[0].source, "dataset", None)
    if type(value) is not str or not value:
        value = "fixture_antigravity"
    return value


def original_target_length_antigravity(train_rows: Sequence[object] | object) -> int:
    """Compute the fixed target from original normal TRAIN lengths only."""
    rows = _train_rows_antigravity(train_rows)
    lengths = np.asarray([len(row.values) for row in rows], dtype=np.int64)
    # This is intentionally the Exp249 formula: float64 median, then clipping.
    return int(np.clip(np.median(lengths), 8, 2048))


def _require_target_antigravity(target_length: int, rows: Sequence[_RowAntigravity]) -> int:
    if type(target_length) is not int or not 8 <= target_length <= 2048:
        raise ValueError("target_length must be within [8, 2048]")
    expected = int(np.clip(np.median([len(row.values) for row in rows]), 8, 2048))
    if target_length != expected:
        raise ValueError("target_length does not equal original TRAIN derived target")
    return expected


def repair_linear_edge_nearest_antigravity(values: np.ndarray) -> np.ndarray:
    """Repair non-finite values on the original time grid, fail closed otherwise."""
    series = np.asarray(values, dtype="<f4").reshape(-1)
    if len(series) == 0:
        raise ValueError("series must not be empty")
    finite = np.isfinite(series)
    if not finite.any():
        raise ValueError("series is entirely non-finite")
    cleaned = np.array(series, dtype="<f4", copy=True)
    if not finite.all():
        positions = np.arange(len(series))
        cleaned[~finite] = np.interp(positions[~finite], positions[finite], series[finite])
    return np.asarray(cleaned, dtype="<f4")


def _align_normalize_antigravity(repaired: np.ndarray, target_length: int) -> np.ndarray:
    if type(target_length) is not int or target_length < 1:
        raise ValueError("target_length is invalid")
    row = np.asarray(repaired, dtype="<f4").reshape(-1)
    if len(row) == 0 or not np.isfinite(row).all():
        raise ValueError("repaired row must be non-empty and finite")
    # Preserve the Exp249 early-return path: no np.interp when lengths match.
    if len(row) == target_length:
        aligned = row.astype("<f4", copy=True)
    else:
        aligned = np.interp(
            np.linspace(0.0, 1.0, target_length, dtype=np.float64),
            np.linspace(0.0, 1.0, len(row), dtype=np.float64),
            row,
        ).astype("<f4")
    center = aligned.mean(dtype=np.float32)
    scale = aligned.std(dtype=np.float32)
    normalized = ((aligned - center) / (scale if scale > 1e-8 else 1.0)).astype("<f4")
    if not np.isfinite(normalized).all():
        raise ValueError("normalized row is non-finite")
    return normalized


def _missing_positions_sha256_antigravity(positions: tuple[int, ...]) -> str:
    return sha256(_canonical_json_bytes_antigravity(list(positions))).hexdigest()


@dataclass(frozen=True)
class PreprocessingAuditAntigravity:
    """Per-row identity and action audit, including the missing-mask digest."""

    dataset: str
    policy: str
    instance_index: int
    original_length: int
    missing_positions: tuple[int, ...]
    missing_positions_sha256: str
    raw_blob_sha256: str
    action: str
    reference_instance_indices: tuple[int, ...] = ()
    unsupported_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("audit dataset is invalid")
        if self.policy not in POLICIES_ANTIGRAVITY:
            raise ValueError("audit policy is not preregistered")
        if type(self.instance_index) is not int or self.instance_index < 0:
            raise ValueError("audit instance index is invalid")
        if type(self.original_length) is not int or self.original_length < 1:
            raise ValueError("audit original length is invalid")
        missing = tuple(self.missing_positions)
        if missing != tuple(sorted(set(missing))) or any(type(value) is not int or value < 0 or value >= self.original_length for value in missing):
            raise ValueError("audit missing positions are invalid")
        if self.missing_positions_sha256 != _missing_positions_sha256_antigravity(missing):
            raise ValueError("audit missing-position hash does not match positions")
        _digest_antigravity(self.raw_blob_sha256, "audit raw blob SHA256")
        if self.action not in _ACTIONS_ANTIGRAVITY:
            raise ValueError("audit action is invalid")
        references = tuple(self.reference_instance_indices)
        unsupported = tuple(self.unsupported_positions)
        if references != tuple(sorted(set(references))) or any(type(value) is not int or value < 0 for value in references):
            raise ValueError("audit references are invalid")
        if self.instance_index in references:
            raise ValueError("audit references must exclude the row itself")
        if unsupported != tuple(sorted(set(unsupported))) or any(type(value) is not int or value not in missing for value in unsupported):
            raise ValueError("audit unsupported positions are invalid")
        if self.action not in _ACTIONS_ANTIGRAVITY:
            raise ValueError("audit action is invalid")
        if not missing:
            if self.action != "unchanged" or references or unsupported:
                raise ValueError("finite audit action is inconsistent")
        elif self.policy == POLICY_A_ANTIGRAVITY:
            if self.action != "linear_and_edge_nearest" or references or unsupported:
                raise ValueError("Policy A audit action is inconsistent")
        elif self.policy == POLICY_B_ANTIGRAVITY:
            if self.action != "excluded_nonfinite_row" or references or unsupported:
                raise ValueError("Policy B audit action is inconsistent")
        elif self.action == "leave_one_out_peer_median":
            if unsupported or not references:
                raise ValueError("Policy C peer-median audit action is inconsistent")
        elif self.action == "excluded_unsupported_peer":
            if not unsupported:
                raise ValueError("Policy C unsupported audit action is inconsistent")
        else:
            raise ValueError("Policy C audit action is inconsistent")
        object.__setattr__(self, "missing_positions", missing)
        object.__setattr__(self, "reference_instance_indices", references)
        object.__setattr__(self, "unsupported_positions", unsupported)

    @property
    def length(self) -> int:
        return self.original_length

    @property
    def raw_values_sha256(self) -> str:
        return self.raw_blob_sha256

    @property
    def repair_method(self) -> str:
        return self.action

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "policy": self.policy,
            "instance_index": self.instance_index,
            "original_length": self.original_length,
            "missing_positions": list(self.missing_positions),
            "missing_positions_sha256": self.missing_positions_sha256,
            "raw_blob_sha256": self.raw_blob_sha256,
            "action": self.action,
            "reference_instance_indices": list(self.reference_instance_indices),
            "unsupported_positions": list(self.unsupported_positions),
        }

    def as_contract_audit_antigravity(self) -> RunPolicyAuditAntigravity:
        return RunPolicyAuditAntigravity(
            self.dataset,
            self.policy,
            self.instance_index,
            self.original_length,
            self.missing_positions,
            self.missing_positions_sha256,
            self.raw_blob_sha256,
            self.action,
            self.reference_instance_indices,
            self.unsupported_positions,
        )


@dataclass(frozen=True)
class PreparedTrainAntigravity:
    dataset: str
    policy: str
    target_length: int
    train_values: np.ndarray
    instance_indices: tuple[int, ...]
    train_total: int
    train_missing: int
    train_used: int
    train_excluded: int
    audits: tuple[PreprocessingAuditAntigravity, ...]
    status: str

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset or self.policy not in POLICIES_ANTIGRAVITY:
            raise ValueError("prepared TRAIN identity is invalid")
        if type(self.target_length) is not int or not 8 <= self.target_length <= 2048:
            raise ValueError("prepared target length must be within [8, 2048]")
        values = _freeze_array_antigravity(self.train_values, ndim=2)
        indices = tuple(self.instance_indices)
        if indices != tuple(dict.fromkeys(indices)) or any(type(value) is not int or value < 0 for value in indices):
            raise ValueError("prepared TRAIN instance indices are invalid")
        audits = tuple(self.audits)
        if len(audits) != self.train_total or any(not isinstance(audit, PreprocessingAuditAntigravity) for audit in audits):
            raise ValueError("prepared TRAIN audits are incomplete")
        if any(audit.dataset != self.dataset or audit.policy != self.policy for audit in audits):
            raise ValueError("prepared TRAIN audit identity disagrees")
        audit_indices = tuple(audit.instance_index for audit in audits)
        if audit_indices != tuple(dict.fromkeys(audit_indices)):
            raise ValueError("prepared TRAIN audit identities are duplicated")
        expected_target = int(np.clip(np.median([audit.original_length for audit in audits]), 8, 2048)) if audits else None
        if expected_target != self.target_length:
            raise ValueError("prepared target length is not derived from original TRAIN audits")
        retained_actions = {"unchanged", "linear_and_edge_nearest", "leave_one_out_peer_median"}
        excluded_actions = {"excluded_nonfinite_row", "excluded_unsupported_peer"}
        expected_indices = tuple(audit.instance_index for audit in audits if audit.action in retained_actions)
        expected_excluded = tuple(audit.instance_index for audit in audits if audit.action in excluded_actions)
        if any(audit.action not in retained_actions | excluded_actions for audit in audits):
            raise ValueError("prepared TRAIN audit action is invalid")
        if indices != expected_indices or len(indices) != values.shape[0] or values.shape[1] != self.target_length:
            raise ValueError("prepared TRAIN shape/order does not reconcile audits")
        counts = (self.train_total, self.train_missing, self.train_used, self.train_excluded)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("prepared TRAIN counts are inconsistent")
        expected_missing = sum(bool(audit.missing_positions) for audit in audits)
        if self.train_total != len(audits) or self.train_missing != expected_missing or self.train_used != len(expected_indices) or self.train_excluded != len(expected_excluded) or self.train_used + self.train_excluded != self.train_total:
            raise ValueError("prepared TRAIN counts do not reconcile audits")
        expected_status = "scored" if self.train_used >= 5 else "abstained"
        if self.status != expected_status:
            raise ValueError("prepared TRAIN status does not reconcile used rows")
        object.__setattr__(self, "train_values", values)
        object.__setattr__(self, "instance_indices", indices)
        object.__setattr__(self, "audits", audits)

    @property
    def values(self) -> np.ndarray:
        return self.train_values

    @property
    def train_normal(self) -> np.ndarray:
        return self.train_values

    @property
    def train_rows(self) -> np.ndarray:
        return self.train_values

    @property
    def contract_audits(self) -> tuple[RunPolicyAuditAntigravity, ...]:
        return tuple(audit.as_contract_audit_antigravity() for audit in self.audits)


@dataclass(frozen=True)
class PreparedDatasetAntigravity:
    dataset: str
    policy: str
    target_length: int
    train_values: np.ndarray
    test_values: np.ndarray
    train_instance_indices: tuple[int, ...]
    test_instance_indices: tuple[int, ...]
    train_total: int
    train_missing: int
    train_used: int
    train_excluded: int
    audits: tuple[PreprocessingAuditAntigravity, ...]
    status: str

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset or self.policy not in POLICIES_ANTIGRAVITY:
            raise ValueError("prepared dataset identity is invalid")
        if type(self.target_length) is not int or not 8 <= self.target_length <= 2048:
            raise ValueError("prepared dataset target length must be within [8, 2048]")
        train = _freeze_array_antigravity(self.train_values, ndim=2)
        test = _freeze_array_antigravity(self.test_values, ndim=2)
        test_indices = tuple(self.test_instance_indices)
        train_indices = tuple(self.train_instance_indices)
        if test_indices != tuple(dict.fromkeys(test_indices)) or train_indices != tuple(dict.fromkeys(train_indices)) or any(type(value) is not int or value < 0 for value in test_indices + train_indices):
            raise ValueError("prepared dataset instance identity is duplicated")
        if test_indices != tuple(range(len(test_indices))):
            raise ValueError("prepared dataset TEST identity is not contiguous from zero")
        audits = tuple(self.audits)
        if len(audits) != self.train_total or any(not isinstance(audit, PreprocessingAuditAntigravity) for audit in audits):
            raise ValueError("prepared dataset audits are incomplete")
        train_result = PreparedTrainAntigravity(
            self.dataset, self.policy, self.target_length, train, train_indices,
            self.train_total, self.train_missing, self.train_used, self.train_excluded,
            audits, self.status,
        )
        if train.shape[1] != self.target_length or test.shape[1] != self.target_length:
            raise ValueError("prepared dataset matrices do not use the derived target")
        if test.shape[0] < 1 or test.shape[0] != len(test_indices):
            raise ValueError("prepared dataset TEST identity is incomplete")
        if train_result.dataset != self.dataset or train_result.policy != self.policy or train_result.target_length != self.target_length or train_result.instance_indices != train_indices or train_result.audits != audits or train_result.status != self.status:
            raise ValueError("prepared dataset TRAIN boundary disagrees")
        object.__setattr__(self, "train_values", train)
        object.__setattr__(self, "test_values", test)
        object.__setattr__(self, "train_instance_indices", train_indices)
        object.__setattr__(self, "test_instance_indices", test_indices)
        object.__setattr__(self, "audits", audits)

    @property
    def train_normal(self) -> np.ndarray:
        return self.train_values

    @property
    def test(self) -> np.ndarray:
        return self.test_values

    @property
    def train(self) -> np.ndarray:
        return self.train_values

    @property
    def train_result(self) -> PreparedTrainAntigravity:
        return PreparedTrainAntigravity(
            self.dataset,
            self.policy,
            self.target_length,
            self.train_values,
            self.train_instance_indices,
            self.train_total,
            self.train_missing,
            self.train_used,
            self.train_excluded,
            self.audits,
            self.status,
        )


def _audit_antigravity(
    dataset: str,
    policy: str,
    row: _RowAntigravity,
    action: str,
    *,
    references: tuple[int, ...] = (),
    unsupported: tuple[int, ...] = (),
) -> PreprocessingAuditAntigravity:
    return PreprocessingAuditAntigravity(
        dataset,
        policy,
        row.instance_index,
        len(row.values),
        row.missing_positions,
        _missing_positions_sha256_antigravity(row.missing_positions),
        row.raw_blob_sha256,
        action,
        references,
        unsupported,
    )


def _require_no_all_missing_antigravity(rows: Sequence[_RowAntigravity]) -> None:
    for row in rows:
        if len(row.missing_positions) == len(row.values):
            raise ValueError(
                f"unexpected_all_missing_train: instance_index={row.instance_index}"
            )


def _policy_c_repair_antigravity(row: _RowAntigravity, peers: Sequence[_RowAntigravity]) -> tuple[np.ndarray | None, tuple[int, ...]]:
    repaired = np.array(row.values, dtype="<f4", copy=True)
    unsupported: list[int] = []
    for position in row.missing_positions:
        finite_peers = np.asarray(
            [peer.values[position] for peer in peers if np.isfinite(peer.values[position])],
            dtype="<f4",
        )
        if len(finite_peers) == 0:
            unsupported.append(position)
            continue
        # Ordinary NumPy median is deliberate; for even peer counts it averages
        # the two middle values rather than choosing an endpoint.
        repaired[position] = np.asarray(np.median(finite_peers), dtype="<f4")
    if unsupported:
        return None, tuple(unsupported)
    return repaired, ()


def apply_train_policy_antigravity(raw: object, policy: str, target_length: int) -> PreparedTrainAntigravity:
    """Apply exactly one frozen policy to raw normal TRAIN rows."""
    if policy not in POLICIES_ANTIGRAVITY:
        raise ValueError("policy is not preregistered")
    rows = _train_rows_antigravity(raw)
    _require_target_antigravity(target_length, rows)
    dataset = _dataset_name_antigravity(raw, rows)
    _require_no_all_missing_antigravity(rows)
    groups: dict[int, tuple[_RowAntigravity, ...]] = {}
    if policy == POLICY_C_ANTIGRAVITY:
        grouped: dict[int, list[_RowAntigravity]] = {}
        for row in rows:
            grouped.setdefault(len(row.values), []).append(row)
        groups = {length: tuple(group) for length, group in grouped.items()}

    prepared_rows: list[np.ndarray] = []
    prepared_indices: list[int] = []
    audits: list[PreprocessingAuditAntigravity] = []
    missing_count = sum(bool(row.missing_positions) for row in rows)
    for row in rows:
        missing = bool(row.missing_positions)
        if policy == POLICY_A_ANTIGRAVITY:
            repaired = repair_linear_edge_nearest_antigravity(row.values)
            action = "linear_and_edge_nearest" if missing else "unchanged"
            audit = _audit_antigravity(dataset, policy, row, action)
        elif policy == POLICY_B_ANTIGRAVITY:
            if missing:
                audits.append(_audit_antigravity(dataset, policy, row, "excluded_nonfinite_row"))
                continue
            repaired = row.values
            audit = _audit_antigravity(dataset, policy, row, "unchanged")
        else:
            peer_group = groups[len(row.values)]
            peers = tuple(peer for peer in peer_group if peer.instance_index != row.instance_index)
            reference_indices = tuple(sorted(peer.instance_index for peer in peers))
            if missing:
                repaired, unsupported = _policy_c_repair_antigravity(row, peers)
                if repaired is None:
                    audits.append(_audit_antigravity(dataset, policy, row, "excluded_unsupported_peer", references=reference_indices, unsupported=unsupported))
                    continue
                audit = _audit_antigravity(dataset, policy, row, "leave_one_out_peer_median", references=reference_indices)
            else:
                repaired = row.values
                audit = _audit_antigravity(dataset, policy, row, "unchanged")
        normalized = _align_normalize_antigravity(repaired, target_length)
        prepared_rows.append(normalized)
        prepared_indices.append(row.instance_index)
        audits.append(audit)

    # Policy B/C still return a deterministic abstention record when fewer than
    # five rows remain.  No synthetic zero row is ever inserted.
    used = len(prepared_rows)
    matrix = np.vstack(prepared_rows).astype("<f4") if prepared_rows else np.empty((0, target_length), dtype="<f4")
    status = "scored" if used >= 5 else "abstained"
    return PreparedTrainAntigravity(
        dataset,
        policy,
        target_length,
        matrix,
        tuple(prepared_indices),
        len(rows),
        missing_count,
        used,
        len(rows) - used,
        tuple(audits),
        status,
    )


def prepare_dataset_antigravity(raw: object, policy: str) -> PreparedDatasetAntigravity:
    """Prepare normal TRAIN under ``policy`` and fixed, shared TEST values."""
    train_rows = _train_rows_antigravity(raw)
    target_length = original_target_length_antigravity(train_rows)
    prepared_train = apply_train_policy_antigravity(raw, policy, target_length)
    test_rows = _test_rows_antigravity(raw)
    dataset = prepared_train.dataset
    test_prepared = []
    for row in test_rows:
        _require_no_all_missing_antigravity((row,))
        test_prepared.append(_align_normalize_antigravity(repair_linear_edge_nearest_antigravity(row.values), target_length))
    test_matrix = np.vstack(test_prepared).astype("<f4")
    return PreparedDatasetAntigravity(
        dataset,
        policy,
        target_length,
        prepared_train.train_values,
        test_matrix,
        prepared_train.instance_indices,
        tuple(row.instance_index for row in test_rows),
        prepared_train.train_total,
        prepared_train.train_missing,
        prepared_train.train_used,
        prepared_train.train_excluded,
        prepared_train.audits,
        prepared_train.status,
    )


__all__ = [
    "POLICY_A_ANTIGRAVITY",
    "POLICY_B_ANTIGRAVITY",
    "POLICY_C_ANTIGRAVITY",
    "POLICIES_ANTIGRAVITY",
    "PreprocessingAuditAntigravity",
    "PreparedTrainAntigravity",
    "PreparedDatasetAntigravity",
    "original_target_length_antigravity",
    "repair_linear_edge_nearest_antigravity",
    "apply_train_policy_antigravity",
    "prepare_dataset_antigravity",
]
