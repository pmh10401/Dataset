"""Create the bounded, replayable Exp455 evaluation artifact in an injected root."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path

from .dashboard_export_antigravity import export_aggregate_dashboard_antigravity
from .contracts_antigravity import FrozenResultAntigravity, RoutingEvaluationAntigravity
from .evaluation_antigravity import evidence_for_evaluation_antigravity


_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "features_antigravity.jsonl",
    "fold_decisions_antigravity.jsonl",
    "comparison_metrics_antigravity.json",
    "family_metrics_antigravity.jsonl",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)


def _json_bytes_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _jsonl_bytes_antigravity(rows: tuple[object, ...]) -> bytes:
    return b"".join(_json_bytes_antigravity(row) for row in rows)


def _context_antigravity(evaluation: RoutingEvaluationAntigravity) -> dict[str, object]:
    """Store only the minimal label-free row identity and confusion evidence for replay."""
    # Evaluation deliberately has no authority field.  Task 4 publication receives its
    # immutable evidence through the private attachment below, set only by evaluator use.
    evidence, _features, inventory = evidence_for_evaluation_antigravity(evaluation)
    rows = []
    for row in evidence:
        rows.append({
            "dataset": row.dataset,
            "family": row.family,
            "target_length": row.target_length,
            "method": row.method,
            "baseline_confusion": asdict(row.baseline_confusion),
            "candidate_confusion": None if row.candidate_confusion is None else asdict(row.candidate_confusion),
        })
    return {"experiment_id": "Exp455", "exp454_leaf_sha256": inventory, "rows": rows}


def _features_antigravity(evaluation: RoutingEvaluationAntigravity) -> tuple[dict[str, object], ...]:
    _evidence, features, _inventory = evidence_for_evaluation_antigravity(evaluation)
    return tuple(asdict(row) for row in features)


def _decision_row_antigravity(decision: object) -> dict[str, object]:
    result = asdict(decision)
    return result


def _payloads_antigravity(evaluation: RoutingEvaluationAntigravity) -> dict[str, bytes]:
    if type(evaluation) is not RoutingEvaluationAntigravity:
        raise ValueError("evaluation is invalid")
    return {
        "context_antigravity.json": _json_bytes_antigravity(_context_antigravity(evaluation)),
        "features_antigravity.jsonl": _jsonl_bytes_antigravity(_features_antigravity(evaluation)),
        "fold_decisions_antigravity.jsonl": _jsonl_bytes_antigravity(tuple(_decision_row_antigravity(row) for row in evaluation.decisions)),
        "comparison_metrics_antigravity.json": _json_bytes_antigravity({"comparisons": [asdict(row) for row in evaluation.comparisons]}),
        "family_metrics_antigravity.jsonl": _jsonl_bytes_antigravity(tuple({"family": family, "fn_delta": delta} for family, delta in evaluation.family_fn_delta)),
        "decision_antigravity.json": _json_bytes_antigravity({"passed": evaluation.passed, "status": evaluation.status}),
        "dashboard_snapshot_antigravity.json": _json_bytes_antigravity(export_aggregate_dashboard_antigravity(evaluation)),
    }


def publish_evaluation_antigravity(root: str | Path, evaluation: RoutingEvaluationAntigravity) -> FrozenResultAntigravity:
    """Publish one exact create-only artifact inventory under an injected empty path."""
    destination = Path(root)
    if destination.exists() or destination.is_symlink():
        raise ValueError("publication destination must not exist")
    payloads = _payloads_antigravity(evaluation)
    if set(payloads) != set(_LEAVES_ANTIGRAVITY[:-1]):
        raise ValueError("publication inventory is invalid")
    try:
        destination.mkdir(mode=0o700)
        for name in _LEAVES_ANTIGRAVITY[:-1]:
            descriptor = os.open(destination / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payloads[name])
        artifact_map = {name: sha256(payloads[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
        map_bytes = _json_bytes_antigravity(artifact_map)
        descriptor = os.open(destination / _LEAVES_ANTIGRAVITY[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(map_bytes)
    except OSError as error:
        raise ValueError("publication destination could not be created") from error
    leaf_sha256 = tuple((name, sha256((destination / name).read_bytes()).hexdigest()) for name in _LEAVES_ANTIGRAVITY)
    return FrozenResultAntigravity(str(destination), leaf_sha256, sha256(map_bytes).hexdigest())


__all__ = ["publish_evaluation_antigravity"]
