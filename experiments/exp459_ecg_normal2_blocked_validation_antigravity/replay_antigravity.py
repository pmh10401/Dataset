"""Source-free semantic replay for Exp459 seven-leaf artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .artifacts_antigravity import (
    _LEAVES_ANTIGRAVITY,
    _context_payload_antigravity,
    _decision_payload_antigravity,
    _fold_plan_rows_antigravity,
    _json_bytes_antigravity,
    _payloads_antigravity,
    _seed_summary_row_antigravity,
)
from .blocked_folds_antigravity import blocked_plan_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    BlockedQualificationAntigravity,
    FoldEvidenceAntigravity,
    SeedSummaryAntigravity,
    ValidationDecisionAntigravity,
)
from .dashboard_export_antigravity import export_dashboard_antigravity
from .evaluation_antigravity import evaluate_blocked_qualification_antigravity


def _strict_json_antigravity(payload: bytes) -> object:
    def no_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate artifact JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicates_antigravity)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact JSON is invalid") from error
    if payload != _json_bytes_antigravity(value):
        raise ValueError("artifact JSON is not canonical")
    return value


def _strict_jsonl_antigravity(payload: bytes) -> tuple[object, ...]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("artifact JSONL is invalid")
    return tuple(_strict_json_antigravity(line + b"\n") for line in payload.splitlines())


def _load_antigravity(root: Path) -> dict[str, bytes]:
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise ValueError("artifact root is invalid")
    if {entry.name for entry in root.iterdir()} != set(_LEAVES_ANTIGRAVITY):
        raise ValueError("artifact inventory differs")
    leaves: dict[str, bytes] = {}
    for name in _LEAVES_ANTIGRAVITY:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("artifact leaf is invalid")
        leaves[name] = path.read_bytes()
    artifact_map = _strict_json_antigravity(leaves["artifact_sha256_antigravity.json"])
    expected = {name: sha256(leaves[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
    if artifact_map != expected:
        raise ValueError("artifact hash map differs")
    return leaves


def _exact_str_antigravity(value: object, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _exact_int_antigravity(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    return value


def _exact_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{name} must be an exact float")
    return value


def _exact_bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


def _dict_keys_antigravity(value: object, keys: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} keys differ")
    return value


def _context_antigravity(payload: bytes) -> None:
    value = _strict_json_antigravity(payload)
    if value != _context_payload_antigravity():
        raise ValueError("context artifact differs")


def _fold_plan_antigravity(payload: bytes) -> tuple[object, ...]:
    rows = _strict_jsonl_antigravity(payload)
    if rows != _fold_plan_rows_antigravity():
        raise ValueError("fold plan differs from blocked-plan recomputation")
    return tuple(blocked_plan_antigravity(FROZEN_CONFIG_ANTIGRAVITY.normal2_train_rows))


def _fold_evidence_antigravity(
    payload: bytes,
    folds: tuple[object, ...],
) -> tuple[FoldEvidenceAntigravity, ...]:
    rows = _strict_jsonl_antigravity(payload)
    if len(rows) != len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds:
        raise ValueError("fold evidence row count differs")
    keys = {
        "dataset",
        "seed",
        "fold_index",
        "normal_trials",
        "normal_alerts",
        "normal_wilson_upper",
        "per_type_auc",
        "pooled_auc",
        "passed",
    }
    expected_order = tuple(
        (seed, fold_index)
        for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
        for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    records: list[FoldEvidenceAntigravity] = []
    for row, (seed, fold_index) in zip(rows, expected_order, strict=True):
        value = _dict_keys_antigravity(row, keys, "fold evidence")
        if (
            _exact_str_antigravity(value["dataset"], "dataset") != FROZEN_CONFIG_ANTIGRAVITY.dataset
            or _exact_int_antigravity(value["seed"], "seed") != seed
            or _exact_int_antigravity(value["fold_index"], "fold_index") != fold_index
        ):
            raise ValueError("fold evidence order or identity differs")
        per_type = _dict_keys_antigravity(
            value["per_type_auc"],
            set(FROZEN_CONFIG_ANTIGRAVITY.primary_types),
            "per_type_auc",
        )
        fold = folds[fold_index]
        records.append(
            FoldEvidenceAntigravity(
                dataset=value["dataset"],
                seed=value["seed"],
                fold_index=value["fold_index"],
                fit_indices=fold.fit_indices,
                query_indices=fold.query_indices,
                normal_trials=_exact_int_antigravity(value["normal_trials"], "normal_trials"),
                normal_alerts=_exact_int_antigravity(value["normal_alerts"], "normal_alerts"),
                normal_wilson_upper=_exact_float_antigravity(value["normal_wilson_upper"], "normal_wilson_upper"),
                per_type_auc={
                    name: _exact_float_antigravity(per_type[name], f"per_type_auc[{name}]")
                    for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
                },
                pooled_auc=_exact_float_antigravity(value["pooled_auc"], "pooled_auc"),
                passed=_exact_bool_antigravity(value["passed"], "passed"),
            )
        )
    return tuple(records)


def _seed_summaries_antigravity(
    payload: bytes,
    fold_records: tuple[FoldEvidenceAntigravity, ...],
) -> tuple[SeedSummaryAntigravity, SeedSummaryAntigravity]:
    rows = _strict_jsonl_antigravity(payload)
    if len(rows) != len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds):
        raise ValueError("seed summary row count differs")
    keys = {"dataset", "seed", "fold_count", "passed"}
    summaries: list[SeedSummaryAntigravity] = []
    for summary_index, row in enumerate(rows):
        value = _dict_keys_antigravity(row, keys, "seed summary")
        seed = FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds[summary_index]
        if (
            _exact_str_antigravity(value["dataset"], "dataset") != FROZEN_CONFIG_ANTIGRAVITY.dataset
            or _exact_int_antigravity(value["seed"], "seed") != seed
            or _exact_int_antigravity(value["fold_count"], "fold_count") != FROZEN_CONFIG_ANTIGRAVITY.folds
        ):
            raise ValueError("seed summary identity differs")
        start = summary_index * FROZEN_CONFIG_ANTIGRAVITY.folds
        stop = start + FROZEN_CONFIG_ANTIGRAVITY.folds
        summary = SeedSummaryAntigravity(
            dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
            seed=seed,
            fold_records=fold_records[start:stop],
            passed=all(record.passed for record in fold_records[start:stop]),
        )
        if _seed_summary_row_antigravity(summary) != {
            "dataset": value["dataset"],
            "seed": value["seed"],
            "fold_count": value["fold_count"],
            "passed": _exact_bool_antigravity(value["passed"], "passed"),
        }:
            raise ValueError("seed summary differs from fold recomputation")
        summaries.append(summary)
    return tuple(summaries)  # type: ignore[return-value]


def _decision_antigravity(
    payload: bytes,
    fold_records: tuple[FoldEvidenceAntigravity, ...],
    seed_summaries: tuple[SeedSummaryAntigravity, SeedSummaryAntigravity],
) -> ValidationDecisionAntigravity:
    value = _dict_keys_antigravity(
        _strict_json_antigravity(payload),
        {"dataset", "status", "passed"},
        "decision",
    )
    qualification = BlockedQualificationAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        fold_evidence=fold_records,
        seed_summaries=seed_summaries,
        passed=all(summary.passed for summary in seed_summaries),
    )
    decision = evaluate_blocked_qualification_antigravity(qualification)
    expected_payload = _decision_payload_antigravity(decision)
    observed_payload = {
        "dataset": _exact_str_antigravity(value["dataset"], "dataset"),
        "status": _exact_str_antigravity(value["status"], "status"),
        "passed": _exact_bool_antigravity(value["passed"], "passed"),
    }
    if observed_payload != expected_payload:
        raise ValueError("decision differs from qualification recomputation")
    return decision


def inspect_result_at_antigravity(root: Path) -> ValidationDecisionAntigravity:
    leaves = _load_antigravity(root)
    _context_antigravity(leaves["context_antigravity.json"])
    folds = _fold_plan_antigravity(leaves["fold_plan_antigravity.jsonl"])
    fold_records = _fold_evidence_antigravity(leaves["fold_evidence_antigravity.jsonl"], folds)
    seed_summaries = _seed_summaries_antigravity(
        leaves["seed_summary_antigravity.jsonl"],
        fold_records,
    )
    decision = _decision_antigravity(
        leaves["decision_antigravity.json"],
        fold_records,
        seed_summaries,
    )
    dashboard = _strict_json_antigravity(leaves["dashboard_snapshot_antigravity.json"])
    if dashboard != export_dashboard_antigravity(decision):
        raise ValueError("dashboard differs from semantic recomputation")
    expected_payloads = _payloads_antigravity(decision)
    for name in _LEAVES_ANTIGRAVITY[:-1]:
        if leaves[name] != expected_payloads[name]:
            raise ValueError("artifact payload differs from semantic replay")
    return decision


__all__ = ("inspect_result_at_antigravity",)
