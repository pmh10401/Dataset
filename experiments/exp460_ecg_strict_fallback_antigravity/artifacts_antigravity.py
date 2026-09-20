"""Create the exact replayable six-leaf Exp460 Task 4 result artifact."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp459FallbackAuthorityAntigravity,
    FallbackEvaluationAntigravity,
    validate_exp459_fallback_authority_antigravity,
    validate_fallback_evaluation_antigravity,
)
from .dashboard_export_antigravity import export_dashboard_antigravity


_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "rule_comparisons_antigravity.jsonl",
    "failed_folds_antigravity.jsonl",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)


@dataclass(frozen=True, slots=True)
class FrozenFallbackPublicationAntigravity:
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


def _comparison_rows_antigravity(result: FallbackEvaluationAntigravity) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "rule": row.rule,
            "route": row.route,
            "tp": row.tp,
            "fp": row.fp,
            "fn": row.fn,
            "tn": row.tn,
            "f1": row.f1,
        }
        for row in result.rule_results
    )


def _failed_folds_antigravity(
    authority: Exp459FallbackAuthorityAntigravity,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "seed": record.seed,
            "fold_index": record.fold_index,
            "period_change_auc": record.period_change_auc,
        }
        for record in authority.fold_records
        if not record.passed
    )


def _decision_payload_antigravity(result: FallbackEvaluationAntigravity) -> dict[str, object]:
    return {
        "all_records_passed": result.all_records_passed,
        "joint_success": result.joint_success,
        "passed_record_count": result.passed_record_count,
        "strict_route": result.strict_route,
        "strict_status": result.strict_status,
        "worst_period_change_auc": result.worst_period_change_auc,
    }


def _context_payload_antigravity(
    authority: Exp459FallbackAuthorityAntigravity,
) -> dict[str, object]:
    return {
        "dataset": FROZEN_CONFIG_ANTIGRAVITY.dataset,
        "experiment_id": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "exp459_decision_sha256": authority.decision_sha256,
        "exp459_frozen_authority_digest": authority.frozen_authority_digest,
        "exp459_official_artifact_map_sha256": authority.official_artifact_map_sha256,
        "label_query_count": authority.label_query_count,
        "official_decision_rule": FROZEN_CONFIG_ANTIGRAVITY.official_decision_rule,
        "predecessor_experiment_id": "Exp459",
        "predecessor_kind": "ecg_normal2_blocked_validation",
        "strict_failure_route": FROZEN_CONFIG_ANTIGRAVITY.strict_failure_route,
    }


def _payloads_antigravity(
    authority: Exp459FallbackAuthorityAntigravity,
    result: FallbackEvaluationAntigravity,
) -> dict[str, bytes]:
    trusted_authority = validate_exp459_fallback_authority_antigravity(authority)
    trusted_result = validate_fallback_evaluation_antigravity(result, trusted_authority)
    payloads = {
        "context_antigravity.json": _json_bytes_antigravity(_context_payload_antigravity(trusted_authority)),
        "rule_comparisons_antigravity.jsonl": _jsonl_bytes_antigravity(
            _comparison_rows_antigravity(trusted_result)
        ),
        "failed_folds_antigravity.jsonl": _jsonl_bytes_antigravity(
            _failed_folds_antigravity(trusted_authority)
        ),
        "decision_antigravity.json": _json_bytes_antigravity(
            _decision_payload_antigravity(trusted_result)
        ),
        "dashboard_snapshot_antigravity.json": _json_bytes_antigravity(
            export_dashboard_antigravity(trusted_authority, trusted_result)
        ),
    }
    if tuple(payloads) != _LEAVES_ANTIGRAVITY[:-1]:
        raise ValueError("publication payload inventory differs")
    return payloads


def publish_result_antigravity(
    root: Path,
    authority: Exp459FallbackAuthorityAntigravity,
    result: FallbackEvaluationAntigravity,
) -> FrozenFallbackPublicationAntigravity:
    if not isinstance(root, Path) or root.exists() or root.is_symlink():
        raise ValueError("publication destination must not exist")
    payloads = _payloads_antigravity(authority, result)
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
    return FrozenFallbackPublicationAntigravity(root, leaf_sha256, sha256(map_bytes).hexdigest())


__all__ = (
    "FrozenFallbackPublicationAntigravity",
    "_LEAVES_ANTIGRAVITY",
    "_json_bytes_antigravity",
    "_payloads_antigravity",
    "publish_result_antigravity",
)
