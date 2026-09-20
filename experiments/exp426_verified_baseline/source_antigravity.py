"""Read-only Exp424 record intake and TRAIN-only verification for Exp426."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterator

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    _record_from_bytes,
    canonical_result_sha256,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    DatasetResult,
    DatasetTerminalEvidence,
    RawSeries,
    RawTrainDataset,
)
from experiments.exp417_causal_multihorizon_shadow.preprocessing import (
    preprocess_length_groups,
)
from experiments.exp425_selective_recovery.eligibility import apply_group_eligibility
from experiments.exp425_selective_recovery.predecessor import (
    EXP424_PROGRESS_SHA256,
    EXP424_RECORD_SET_SHA256,
    EXP424_ROOT_PROVENANCE_SHA256,
    EXP424_SOURCE_ROOT,
    Exp424CompatibilityContract,
    ValidatedExp424Source,
    _payload_digest_from_canonical_bytes,
    _read_at,
    _validate_source,
    open_validated_exp424_source,
)

from .contracts_antigravity import Exp426DatasetRecord


_TERMINAL_REASON_ALIASES = {
    "oof_identity_insufficient": "oof_identity_insufficient",
    "early stopping requires at least five identities on both sides":
        "oof_identity_insufficient",
}


@dataclass(frozen=True)
class Exp424SourceTask:
    dataset: str
    raw: bytes
    expected_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("source task dataset is invalid")
        if not isinstance(self.raw, bytes) or not self.raw:
            raise ValueError("source task bytes are invalid")
        _lower_sha256(self.expected_digest, "source task digest")


@dataclass(frozen=True)
class Exp424SourceClassification:
    dataset: str
    source_record_sha256: str
    source_result_sha256: str
    imported_record: Exp426DatasetRecord | None
    terminal_reason: str | None
    train_size: int
    train_lengths: tuple[int, ...]
    test_size: int
    test_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("source classification dataset is invalid")
        _lower_sha256(self.source_record_sha256, "source record digest")
        _lower_sha256(self.source_result_sha256, "source result digest")
        imported = self.imported_record is not None
        terminal = self.terminal_reason is not None
        if imported == terminal:
            raise ValueError("source classification must be imported or terminal")
        if imported and self.imported_record.dataset != self.dataset:
            raise ValueError("source classification record dataset does not match")
        if terminal and (
            not isinstance(self.terminal_reason, str) or not self.terminal_reason
        ):
            raise ValueError("source terminal reason is invalid")
        if (
            type(self.train_size) is not int or self.train_size < 0
            or type(self.test_size) is not int or self.test_size <= 0
        ):
            raise ValueError("source classification sizes are invalid")
        for label, values in (
            ("TRAIN", self.train_lengths), ("TEST", self.test_lengths),
        ):
            if (
                not isinstance(values, tuple)
                or any(type(value) is not int or value <= 0 for value in values)
                or values != tuple(sorted(set(values)))
            ):
                raise ValueError(f"source classification {label} lengths are invalid")

    @property
    def scientific_sha256(self) -> str:
        payload = {
            "schema": "exp426.source-classification.v1",
            "dataset": self.dataset,
            "source_record_sha256": self.source_record_sha256,
            "source_result_sha256": self.source_result_sha256,
            "imported_record_sha256": (
                None if self.imported_record is None else self.imported_record.sha256
            ),
            "terminal_reason": self.terminal_reason,
            "train_size": self.train_size,
            "train_lengths": list(self.train_lengths),
            "test_size": self.test_size,
            "test_lengths": list(self.test_lengths),
        }
        return sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class Exp424ProductionSourceReport:
    records: int
    importable: int
    terminal: int
    terminal_reasons: tuple[str, ...]
    source_root_sha256: str
    source_progress_sha256: str
    source_record_set_sha256: str


def inspect_production_source_authority() -> Exp424ProductionSourceReport:
    """Perform the explicit, read-only 1,117-record production source audit."""
    contract = Exp424CompatibilityContract.production()
    with open_validated_exp424_source(EXP424_SOURCE_ROOT, contract) as source:
        values = _audit_verified_source(source)
    expected = {
        "records": 1117,
        "importable": 983,
        "terminal": 134,
        "terminal_reasons": ("oof_identity_insufficient",),
        "source_mutated": False,
    }
    if dict(values) != expected:
        raise ValueError(
            f"Exp424 production source accounting is incompatible: {values!r}"
        )
    return Exp424ProductionSourceReport(
        records=1117,
        importable=983,
        terminal=134,
        terminal_reasons=("oof_identity_insufficient",),
        source_root_sha256=EXP424_ROOT_PROVENANCE_SHA256,
        source_progress_sha256=EXP424_PROGRESS_SHA256,
        source_record_set_sha256=EXP424_RECORD_SET_SHA256,
    )


def revalidate_verified_source(source: ValidatedExp424Source) -> None:
    """Recheck exact source bytes and the descriptor-bound inventory in one pass."""
    _audit_verified_source(source)


def _audit_verified_source(source: ValidatedExp424Source) -> dict[str, object]:
    if type(source) is not ValidatedExp424Source:
        raise ValueError("Exp424 source authority is invalid")
    source._require_live()
    importable = 0
    terminal = 0
    reasons: set[str] = set()
    for descriptor in source._records:
        raw, identity = _read_at(
            source._datasets_fd, descriptor.filename, "Exp424 dataset record"
        )
        if (
            identity != descriptor.identity
            or _payload_digest_from_canonical_bytes(raw) != descriptor.digest
        ):
            raise ValueError("Exp424 source record changed after validation")
        if b'"raw_train_identity_sha256":null' in raw:
            payload = _record_from_bytes(raw)
            result = payload.get("qualification")
            if (
                not isinstance(result, DatasetResult)
                or result.dataset != descriptor.dataset
                or not isinstance(result.decision_evidence, DatasetTerminalEvidence)
            ):
                raise ValueError("Exp424 terminal record is structurally invalid")
            terminal += 1
            reasons.add(_canonical_terminal_reason(result.decision_evidence.reason))
        else:
            importable += 1
    _validate_source(
        source._root,
        source._contract,
        source._root_fd,
        source._datasets_fd,
        source._root_identity,
        source._datasets_identity,
        source._records,
        metadata_identities=(
            source._provenance_identity, source._progress_identity,
        ),
        full_records=False,
    )
    return {
        "records": importable + terminal,
        "importable": importable,
        "terminal": terminal,
        "terminal_reasons": tuple(sorted(reasons)),
        "source_mutated": False,
    }


def _lower_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _canonical_terminal_reason(value: object) -> str:
    if not isinstance(value, str) or value not in _TERMINAL_REASON_ALIASES:
        raise ValueError("Exp424 terminal reason is not a verified identity-insufficient reason")
    return _TERMINAL_REASON_ALIASES[value]


def read_verified_source_tasks(
    source: ValidatedExp424Source,
) -> Iterator[Exp424SourceTask]:
    """Read exact record bytes through the live source directory descriptor."""
    if type(source) is not ValidatedExp424Source:
        raise ValueError("Exp424 source authority is invalid")
    source._require_live()
    for descriptor in source._records:
        raw, identity = _read_at(
            source._datasets_fd, descriptor.filename, "Exp424 dataset record"
        )
        if (
            identity != descriptor.identity
            or _payload_digest_from_canonical_bytes(raw) != descriptor.digest
        ):
            raise ValueError("Exp424 source record changed after validation")
        yield Exp424SourceTask(descriptor.dataset, raw, descriptor.digest)


def classify_exp424_record_bytes(
    dataset: str, raw: bytes, expected_digest: str,
) -> Exp424SourceClassification:
    """Classify one already descriptor-verified record without opening a path."""
    _lower_sha256(expected_digest, "expected source record digest")
    if not isinstance(raw, bytes):
        raise ValueError("Exp424 source record must be bytes")
    digest = _payload_digest_from_canonical_bytes(raw)
    if digest != expected_digest:
        raise ValueError("Exp424 source record digest does not match")
    payload = _record_from_bytes(raw)
    result = payload.get("qualification")
    if not isinstance(result, DatasetResult) or result.dataset != dataset:
        raise ValueError("Exp424 source result dataset does not match")
    result_digest = canonical_result_sha256(result)
    test_manifest = result.context.test_identity_manifest[dataset]
    test_lengths = tuple(sorted({length for _index, length in test_manifest}))
    if isinstance(result.decision_evidence, DatasetTerminalEvidence):
        return Exp424SourceClassification(
            dataset, digest, result_digest, None,
            _canonical_terminal_reason(result.decision_evidence.reason),
            0, (), result.test_size, test_lengths,
        )
    coordinates = tuple(
        (candidate.dataset, candidate.instance_index)
        for candidate in result.candidates
    )
    record = Exp426DatasetRecord(
        dataset=dataset,
        origin="exp424_import",
        status="scored",
        candidate_coordinates=coordinates,
        source_record_sha256=digest,
        source_result_sha256=result_digest,
        train_identity_sha256=None,
        eligibility_sha256=None,
        reason="verified_exp424_nonterminal",
    )
    return Exp424SourceClassification(
        dataset, digest, result_digest, record, None,
        sum(group.series_count for group in result.groups),
        tuple(group.length for group in result.groups),
        result.test_size,
        test_lengths,
    )


def verify_train_abstention(
    dataset: str,
    train: RawTrainDataset,
    fold_contract_sha256: str,
    *,
    source_record_sha256: str,
    source_result_sha256: str,
) -> Exp426DatasetRecord:
    """Reconfirm a source terminal outcome using normal TRAIN values only."""
    if not isinstance(train, RawTrainDataset) or train.dataset != dataset:
        raise ValueError("TRAIN dataset does not match verification task")
    _lower_sha256(fold_contract_sha256, "fold contract digest")
    _lower_sha256(source_record_sha256, "source record digest")
    _lower_sha256(source_result_sha256, "source result digest")
    processed = preprocess_length_groups(train)
    assessment = apply_group_eligibility(
        dataset, processed, fold_contract_sha256=fold_contract_sha256
    )
    statuses = tuple(item.status for item in assessment.evidence.values())
    allowed = {"oof_identity_insufficient", "preprocessing_abstained"}
    if (
        not statuses
        or any(status not in allowed for status in statuses)
        or "oof_identity_insufficient" not in statuses
    ):
        raise ValueError("TRAIN verification found an eligible or unproven group")
    eligibility = {
        "schema": "exp426.train-eligibility.v1",
        "dataset": dataset,
        "fold_contract_sha256": fold_contract_sha256,
        "groups": [
            {
                "length": item.length,
                "processed_status": item.processed_status,
                "train_identity_sha256": item.train_identity_sha256,
                "unique_identity_count": item.unique_identity_count,
                "folds": [
                    {
                        "oof_fit": fold.oof_fit,
                        "oof_validation": fold.oof_validation,
                        "early_stop_fit": fold.early_stop_fit,
                        "early_stop_validation": fold.early_stop_validation,
                    }
                    for fold in item.folds
                ],
                "final_fit": item.final_fit,
                "final_validation": item.final_validation,
                "status": item.status,
                "reason": item.reason,
            }
            for _length, item in sorted(assessment.evidence.items())
        ],
    }
    return Exp426DatasetRecord(
        dataset=dataset,
        origin="verified_train_abstention",
        status="abstained",
        candidate_coordinates=(),
        source_record_sha256=source_record_sha256,
        source_result_sha256=source_result_sha256,
        train_identity_sha256=_raw_train_sha256(train),
        eligibility_sha256=sha256(canonical_json_bytes(eligibility)).hexdigest(),
        reason="oof_identity_insufficient",
    )


def _raw_train_sha256(train: RawTrainDataset) -> str:
    digest = sha256()
    _digest_text(digest, "exp426.raw-train.v1")
    _digest_text(digest, train.dataset)
    digest.update(len(train.series).to_bytes(8, "little", signed=False))
    for item in train.series:
        if not isinstance(item, RawSeries):
            raise ValueError("TRAIN series is invalid")
        digest.update(item.instance_index.to_bytes(8, "little", signed=False))
        values = np.asarray(item.values, dtype="<f4")
        if values.ndim != 1 or len(values) != len(item.values):
            raise ValueError("TRAIN values are invalid")
        digest.update(len(values).to_bytes(8, "little", signed=False))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _digest_text(digest: object, value: str) -> None:
    payload = value.encode("utf-8")
    digest.update(len(payload).to_bytes(8, "little", signed=False))  # type: ignore[attr-defined]
    digest.update(payload)  # type: ignore[attr-defined]
