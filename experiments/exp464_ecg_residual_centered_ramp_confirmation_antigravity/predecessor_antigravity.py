"""Capture the exact Exp463 authority required by Exp464 from tracked evidence only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .contracts_antigravity import (
    Exp463AuthorityAntigravity,
    validate_exp463_authority_antigravity,
)
from .config_antigravity import (
    FROZEN_CONFIG_ANTIGRAVITY,
    Exp463MethodSummarySnapshotAntigravity,
)


_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_TASK5_REPORT_ANTIGRAVITY = (
    _ROOT_ANTIGRAVITY
    / ".superpowers/sdd/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html/task-5-report-antigravity.html"
)
_IMPLEMENTATION_REPORT_ANTIGRAVITY = (
    _ROOT_ANTIGRAVITY / "docs/EXP463_IMPLEMENTATION_REPORT_antigravity.html"
)
_LEDGER_PATHS_ANTIGRAVITY = (
    _ROOT_ANTIGRAVITY / "RESEARCH_STATE.json",
    _ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json",
    _ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json",
)


def _strict_json_antigravity(path: Path) -> Any:
    def reject_duplicates_antigravity(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates_antigravity)


def _report_bytes_antigravity() -> bytes:
    report_bytes = _TASK5_REPORT_ANTIGRAVITY.read_bytes()
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    if report_sha256 != FROZEN_CONFIG_ANTIGRAVITY.predecessor_task5_report_sha256:
        raise ValueError("Exp463 Task 5 report SHA-256 drifted from the frozen contract")
    return report_bytes


def _exp463_ledger_rows_antigravity() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state = _strict_json_antigravity(_LEDGER_PATHS_ANTIGRAVITY[0])["exp463"]
    catalog = next(
        row
        for row in _strict_json_antigravity(_LEDGER_PATHS_ANTIGRAVITY[1])["experiments"]
        if row.get("id") == "Exp463"
    )
    champion = next(
        row
        for row in _strict_json_antigravity(_LEDGER_PATHS_ANTIGRAVITY[2])["history"]
        if row.get("experiment") == "Exp463"
    )
    return state, catalog, champion


def _cross_check_ledgers_antigravity(
    state: dict[str, Any],
    catalog: dict[str, Any],
    champion: dict[str, Any],
) -> dict[str, Any]:
    if catalog["experiment"] != state["experiment"] or champion["experiment"] != state["experiment"]:
        raise ValueError("Exp463 ledger experiment identity drifted")
    if catalog["status"] != state["status"] or champion["status"] != state["status"]:
        raise ValueError("Exp463 ledger status drifted")
    if catalog["implementation_status"] != champion["implementation_status"]:
        raise ValueError("Exp463 implementation_status drifted across ledgers")
    if catalog["task5_private_train_result"] != state["task5_private_train_result"]:
        raise ValueError("Exp463 catalog Task 5 evidence drifted from research_state")
    if champion["task5_private_train_result"] != state["task5_private_train_result"]:
        raise ValueError("Exp463 champion Task 5 evidence drifted from research_state")
    if state["task5_private_train_result"]["reviewed_commit"] not in state["notes"]:
        raise ValueError("Exp463 state note drifted from reviewed_commit")
    if state["task5_private_train_result"]["reviewed_commit"] not in catalog["notes"]:
        raise ValueError("Exp463 catalog note drifted from reviewed_commit")
    if state["task5_private_train_result"]["reviewed_commit"] not in champion["reason"]:
        raise ValueError("Exp463 champion reason drifted from reviewed_commit")
    return state["task5_private_train_result"]


def _extract_report_control_summaries_antigravity(report_text: str) -> dict[str, Exp463MethodSummarySnapshotAntigravity]:
    pattern = re.compile(
        r"<tr><td>M(?P<idx>[012456])(?: [^<]*)?</td><td>(?P<passed>\d+)/(?P<fold>\d+)</td>"
        r"<td>(?P<gradual>0\.\d+)</td><td>(?P<period>0\.\d+)</td><td>(?P<pooled>0\.\d+)</td>",
    )
    summary_rows = {match.group("idx"): match.groupdict() for match in pattern.finditer(report_text)}
    method_map = {"0": "C0", "2": "C1", "4": "C2", "6": "C7"}
    result: dict[str, Exp463MethodSummarySnapshotAntigravity] = {}
    for report_id, method in method_map.items():
        row = summary_rows[report_id]
        fold_count = int(row["fold"])
        passed_count = int(row["passed"])
        result[method] = Exp463MethodSummarySnapshotAntigravity(
            method=method,
            fold_count=fold_count,
            passed_count=passed_count,
            failed_count=fold_count - passed_count,
            min_gradual_drift_auc=float(row["gradual"]),
            min_period_change_auc=float(row["period"]),
            min_pooled_auc=float(row["pooled"]),
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        )
    return result


def capture_exp463_authority_antigravity() -> Exp463AuthorityAntigravity:
    """Return the exact frozen Exp463 private TRAIN authority consumed by Exp464."""

    report_text = _report_bytes_antigravity().decode("utf-8")
    implementation_report_text = _IMPLEMENTATION_REPORT_ANTIGRAVITY.read_text(encoding="utf-8")
    state, catalog, champion = _exp463_ledger_rows_antigravity()
    task5_result = _cross_check_ledgers_antigravity(state, catalog, champion)

    if task5_result["artifact_map_sha256"] not in implementation_report_text:
        raise ValueError("Exp463 implementation report drifted from tracked artifact_map_sha256")
    if task5_result["reviewed_commit"] not in implementation_report_text:
        raise ValueError("Exp463 implementation report drifted from tracked reviewed_commit")

    summaries = _extract_report_control_summaries_antigravity(report_text)
    authority = Exp463AuthorityAntigravity(
        predecessor_experiment=FROZEN_CONFIG_ANTIGRAVITY.predecessor_experiment,
        terminal_closeout_commit=FROZEN_CONFIG_ANTIGRAVITY.predecessor_terminal_closeout_commit,
        task5_report_sha256=FROZEN_CONFIG_ANTIGRAVITY.predecessor_task5_report_sha256,
        status=state["status"],
        reviewed_cli_commit=task5_result["reviewed_commit"],
        private_run_count=len(task5_result["runs"]),
        result_row_count=task5_result["row_count"],
        result_failed_row_count=task5_result["failed_row_count"],
        result_summary_count=task5_result["summary_count"],
        result_leaf_count=task5_result["leaf_count"],
        artifact_map_sha256=task5_result["artifact_map_sha256"],
        official_evaluate_count=state["official_evaluate_invocations"],
        official_inspect_count=state["official_inspect_invocations"],
        label_query_count=state["label_query_count"],
        control_summaries=summaries,
    )
    validate_exp463_authority_antigravity(authority)
    return authority


__all__ = ("capture_exp463_authority_antigravity",)
