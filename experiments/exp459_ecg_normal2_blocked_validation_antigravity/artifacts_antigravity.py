"""Create the exact replayable seven-leaf Exp459 Task 4 result artifact."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from .blocked_folds_antigravity import blocked_plan_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import ValidationDecisionAntigravity
from .dashboard_export_antigravity import export_dashboard_antigravity
from .evaluation_antigravity import (
    EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY,
    _qualification_for_decision_antigravity,
    validate_validation_decision_semantics_antigravity,
)


_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "fold_plan_antigravity.jsonl",
    "fold_evidence_antigravity.jsonl",
    "seed_summary_antigravity.jsonl",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)


@dataclass(frozen=True, slots=True)
class PublishedResultAntigravity:
    root: Path
    leaf_sha256: tuple[tuple[str, str], ...]
    artifact_map_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise ValueError("published root must be a Path")
        if type(self.leaf_sha256) is not tuple or tuple(name for name, _digest in self.leaf_sha256) != _LEAVES_ANTIGRAVITY:
            raise ValueError("published leaf order differs")
        for _name, digest in self.leaf_sha256:
            if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("published leaf digest is invalid")
        if type(self.artifact_map_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", self.artifact_map_sha256) is None:
            raise ValueError("published artifact-map digest is invalid")


def _json_bytes_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _jsonl_bytes_antigravity(rows: tuple[object, ...]) -> bytes:
    return b"".join(_json_bytes_antigravity(row) for row in rows)


def _runtime_payload_antigravity() -> dict[str, str]:
    return {
        "executable": FROZEN_CONFIG_ANTIGRAVITY.runtime_executable,
        "prefix": FROZEN_CONFIG_ANTIGRAVITY.runtime_prefix,
        "python_version": FROZEN_CONFIG_ANTIGRAVITY.runtime_python_version,
        "numpy_version": FROZEN_CONFIG_ANTIGRAVITY.runtime_numpy_version,
        "aeon_version": FROZEN_CONFIG_ANTIGRAVITY.runtime_aeon_version,
    }


def _context_payload_antigravity() -> dict[str, object]:
    return {
        "experiment_id": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "approved_design_sha256": FROZEN_CONFIG_ANTIGRAVITY.approved_design_sha256,
        "run_identity": FROZEN_CONFIG_ANTIGRAVITY.run_identity,
        "predecessor_reviewed_commit": FROZEN_CONFIG_ANTIGRAVITY.predecessor_reviewed_commit,
        "predecessor_artifact_map_sha256": FROZEN_CONFIG_ANTIGRAVITY.predecessor_artifact_map_sha256,
        "exp458_predecessor": dict(EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY),
        "runtime": _runtime_payload_antigravity(),
        "label_query_count": 0,
    }


def _fold_plan_rows_antigravity() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for fold in blocked_plan_antigravity(FROZEN_CONFIG_ANTIGRAVITY.normal2_train_rows):
        rows.append(
            {
                "n_rows": fold.n_rows,
                "fold_index": fold.fold_index,
                "fit_count": len(fold.fit_indices),
                "query_count": len(fold.query_indices),
                "query_start": int(fold.query_indices[0]),
                "query_stop": int(fold.query_indices[-1]),
                "fit_indices": fold.fit_indices.tolist(),
                "query_indices": fold.query_indices.tolist(),
            }
        )
    return tuple(rows)


def _fold_evidence_row_antigravity(record: object) -> dict[str, object]:
    return {
        "dataset": record.dataset,
        "seed": record.seed,
        "fold_index": record.fold_index,
        "normal_trials": record.normal_trials,
        "normal_alerts": record.normal_alerts,
        "normal_wilson_upper": record.normal_wilson_upper,
        "per_type_auc": {
            name: record.per_type_auc[name]
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
        },
        "pooled_auc": record.pooled_auc,
        "passed": record.passed,
    }


def _seed_summary_row_antigravity(summary: object) -> dict[str, object]:
    return {
        "dataset": summary.dataset,
        "seed": summary.seed,
        "fold_count": len(summary.fold_records),
        "passed": summary.passed,
    }


def _decision_payload_antigravity(decision: ValidationDecisionAntigravity) -> dict[str, object]:
    return {
        "dataset": decision.dataset,
        "status": decision.status,
        "passed": decision.passed,
    }


def _payloads_antigravity(decision: ValidationDecisionAntigravity) -> dict[str, bytes]:
    value = validate_validation_decision_semantics_antigravity(decision)
    qualification = _qualification_for_decision_antigravity(value)
    payloads = {
        "context_antigravity.json": _json_bytes_antigravity(_context_payload_antigravity()),
        "fold_plan_antigravity.jsonl": _jsonl_bytes_antigravity(_fold_plan_rows_antigravity()),
        "fold_evidence_antigravity.jsonl": _jsonl_bytes_antigravity(
            tuple(_fold_evidence_row_antigravity(record) for record in qualification.fold_evidence)
        ),
        "seed_summary_antigravity.jsonl": _jsonl_bytes_antigravity(
            tuple(_seed_summary_row_antigravity(summary) for summary in qualification.seed_summaries)
        ),
        "decision_antigravity.json": _json_bytes_antigravity(_decision_payload_antigravity(value)),
        "dashboard_snapshot_antigravity.json": _json_bytes_antigravity(export_dashboard_antigravity(value)),
    }
    if tuple(payloads) != _LEAVES_ANTIGRAVITY[:-1]:
        raise ValueError("publication payload inventory differs")
    return payloads


def publish_result_antigravity(
    root: Path, decision: ValidationDecisionAntigravity
) -> PublishedResultAntigravity:
    if not isinstance(root, Path) or root.exists() or root.is_symlink():
        raise ValueError("publication destination must not exist")
    payloads = _payloads_antigravity(decision)
    artifact_map = {name: sha256(payloads[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
    map_bytes = _json_bytes_antigravity(artifact_map)
    try:
        root.mkdir(mode=0o700)
        for name in _LEAVES_ANTIGRAVITY[:-1]:
            descriptor = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payloads[name])
        descriptor = os.open(root / _LEAVES_ANTIGRAVITY[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(map_bytes)
    except OSError as error:
        raise ValueError("publication destination could not be created") from error
    leaf_sha256 = tuple(
        (name, sha256((root / name).read_bytes()).hexdigest())
        for name in _LEAVES_ANTIGRAVITY
    )
    return PublishedResultAntigravity(root, leaf_sha256, sha256(map_bytes).hexdigest())


__all__ = (
    "PublishedResultAntigravity",
    "_LEAVES_ANTIGRAVITY",
    "publish_result_antigravity",
)
