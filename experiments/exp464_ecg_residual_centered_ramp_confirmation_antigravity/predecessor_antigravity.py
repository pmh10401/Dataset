"""Capture the exact Exp463 authority required by Exp464 from tracked evidence only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from .contracts_antigravity import (
    Exp463AuthorityAntigravity,
    Exp463MethodSummaryAntigravity,
    validate_exp463_authority_antigravity,
)
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


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
_EXP463_CODEMAP_PATHS_ANTIGRAVITY = (
    _ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp463_antigravity.html",
    _ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp463_antigravity.json",
    _ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp463_antigravity.lock",
)
_TOP_LEVEL_TERMINAL_FIELDS_ANTIGRAVITY = (
    ("experiment", "Exp463"),
    ("status", "stopped_before_official_evaluation_qualification_failed"),
    ("implementation_status", "task_5_review_ship_qualification_failed_task_6_forbidden"),
    ("review_only", False),
    ("official_evaluate_started", False),
    ("official_inspect_started", False),
    ("official_evaluate_invocations", 0),
    ("official_inspect_invocations", 0),
    ("result_root_created", False),
    ("official_result", None),
    ("official_root_present", False),
    ("prediction_sha256", None),
    ("evaluation_sha256", None),
    ("decision_sha256", None),
    ("artifact_map_sha256", None),
    ("dashboard_snapshot_sha256", None),
    ("test_labels_opened", False),
    ("label_query_count", 0),
)
_PRIVATE_RESULT_FIELDS_ANTIGRAVITY = (
    ("reviewed_commit", "923cc4ad6dd6737f84f4ea2c4f621eb652ac54b4"),
    ("row_count", 70),
    ("failed_row_count", 30),
    ("summary_count", 7),
    ("leaf_count", 7),
    ("artifact_map_sha256", "25fb4ab237ae8f2601c2bc03b05f4e6a2d0cb9889876a83fa0f2c4ee7c4f0ea5"),
    ("label_query_count", 0),
    ("official_root_exists_after", False),
    ("status", "qualification_failed_blocked_before_routing_research"),
    ("temporary_roots_removed", True),
)
_PRIVATE_EQUALITY_FIELDS_ANTIGRAVITY = (
    "rows",
    "failed_rows",
    "summaries",
    "decision",
    "dashboard",
    "leaf_bytes",
    "leaf_hashes",
    "artifact_map_hash",
)
_CONTROL_METHOD_MAP_ANTIGRAVITY = {"0": "C0", "2": "C1", "4": "C2", "6": "C7"}
def _canonical_json_bytes_antigravity(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


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


def _exact_value_antigravity(actual: object, expected: object, name: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{name} drifted from the frozen Exp463 authority")


def _git_commit_antigravity(paths: tuple[Path, ...]) -> str:
    result = subprocess.run(
        [
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            *(str(path.relative_to(_ROOT_ANTIGRAVITY)) for path in paths),
        ],
        cwd=_ROOT_ANTIGRAVITY,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Exp463 terminal closeout commit could not be resolved from tracked evidence")
    return commit


def _tracked_terminal_closeout_commit_antigravity() -> str:
    for path in _EXP463_CODEMAP_PATHS_ANTIGRAVITY:
        if not path.is_file():
            raise ValueError(f"missing tracked Exp463 terminal evidence: {path.name}")
    return _git_commit_antigravity(_EXP463_CODEMAP_PATHS_ANTIGRAVITY)


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


def _validate_top_level_row_antigravity(row: dict[str, Any], row_name: str) -> None:
    if type(row) is not dict:
        raise ValueError(f"{row_name} terminal row must be a dict")
    for field_name, expected_value in _TOP_LEVEL_TERMINAL_FIELDS_ANTIGRAVITY:
        _exact_value_antigravity(
            row.get(field_name),
            expected_value,
            f"{row_name}.{field_name}",
        )


def _validate_private_runs_antigravity(runs: object, row_name: str) -> None:
    if type(runs) is not list or len(runs) != 2:
        raise ValueError(f"{row_name}.task5_private_train_result.runs must contain exactly two runs")
    for expected_run, run in enumerate(runs, start=1):
        if type(run) is not dict:
            raise ValueError(f"{row_name}.task5_private_train_result.runs[{expected_run - 1}] must be a dict")
        _exact_value_antigravity(run.get("run"), expected_run, f"{row_name}.runs[{expected_run - 1}].run")
        _exact_value_antigravity(
            run.get("evaluate_exit"),
            0,
            f"{row_name}.runs[{expected_run - 1}].evaluate_exit",
        )
        _exact_value_antigravity(
            run.get("inspect_exit"),
            0,
            f"{row_name}.runs[{expected_run - 1}].inspect_exit",
        )
        for duration_name in ("evaluate_seconds", "inspect_seconds"):
            duration = run.get(duration_name)
            if type(duration) is not float or duration <= 0.0:
                raise ValueError(
                    f"{row_name}.runs[{expected_run - 1}].{duration_name} must be a positive float"
                )


def _validate_private_result_antigravity(task5_result: dict[str, Any], row_name: str) -> None:
    if type(task5_result) is not dict:
        raise ValueError(f"{row_name}.task5_private_train_result must be a dict")
    for field_name, expected_value in _PRIVATE_RESULT_FIELDS_ANTIGRAVITY:
        _exact_value_antigravity(
            task5_result.get(field_name),
            expected_value,
            f"{row_name}.task5_private_train_result.{field_name}",
        )
    exact_equality = task5_result.get("exact_equality")
    if type(exact_equality) is not dict:
        raise ValueError(f"{row_name}.task5_private_train_result.exact_equality must be a dict")
    if tuple(exact_equality) != _PRIVATE_EQUALITY_FIELDS_ANTIGRAVITY:
        raise ValueError(f"{row_name}.task5_private_train_result.exact_equality order drifted")
    for field_name in _PRIVATE_EQUALITY_FIELDS_ANTIGRAVITY:
        _exact_value_antigravity(
            exact_equality.get(field_name),
            True,
            f"{row_name}.task5_private_train_result.exact_equality.{field_name}",
        )
    _validate_private_runs_antigravity(task5_result.get("runs"), row_name)
    overlap_nonresult = task5_result.get("overlap_nonresult")
    if type(overlap_nonresult) is not dict:
        raise ValueError(f"{row_name}.task5_private_train_result.overlap_nonresult must be a dict")
    _exact_value_antigravity(
        overlap_nonresult.get("publisher_error"),
        "publication destination must be a new private path",
        f"{row_name}.task5_private_train_result.overlap_nonresult.publisher_error",
    )
    _exact_value_antigravity(
        overlap_nonresult.get("writes"),
        False,
        f"{row_name}.task5_private_train_result.overlap_nonresult.writes",
    )


def _cross_check_ledgers_antigravity(
    state: dict[str, Any],
    catalog: dict[str, Any],
    champion: dict[str, Any],
) -> dict[str, Any]:
    for row_name, row in (("state", state), ("catalog", catalog), ("champion", champion)):
        _validate_top_level_row_antigravity(row, row_name)
        _validate_private_result_antigravity(row["task5_private_train_result"], row_name)
    if catalog["experiment"] != state["experiment"] or champion["experiment"] != state["experiment"]:
        raise ValueError("Exp463 ledger experiment identity drifted")
    if catalog["status"] != state["status"] or champion["status"] != state["status"]:
        raise ValueError("Exp463 ledger status drifted")
    if catalog["implementation_status"] != state["implementation_status"] or champion["implementation_status"] != state["implementation_status"]:
        raise ValueError("Exp463 implementation_status drifted across ledgers")
    if catalog["task5_private_train_result"] != state["task5_private_train_result"]:
        raise ValueError("Exp463 catalog Task 5 evidence drifted from research_state")
    if champion["task5_private_train_result"] != state["task5_private_train_result"]:
        raise ValueError("Exp463 champion Task 5 evidence drifted from research_state")
    note = state.get("notes")
    catalog_note = catalog.get("notes")
    champion_reason = champion.get("reason")
    if type(note) is not str or type(catalog_note) is not str or type(champion_reason) is not str:
        raise ValueError("Exp463 terminal narrative drifted from tracked strings")
    if note != catalog_note or note != champion_reason:
        raise ValueError("Exp463 terminal narrative drifted across ledgers")
    reviewed_commit = state["task5_private_train_result"]["reviewed_commit"]
    for required in (
        reviewed_commit,
        "exactly two immutable evaluate-to-inspect reproductions",
        "Official evaluate/inspect counts remain 0",
        "official result hashes remain null",
        "Task 6 is permanently forbidden",
    ):
        if required not in note:
            raise ValueError("Exp463 terminal narrative drifted from frozen Task 5 truth")
    return state["task5_private_train_result"]


def _extract_wilson_upper_antigravity(text: str) -> float:
    match = re.search(
        r"(?:Wilson maximum|max normal Wilson upper was) <code>(?P<wilson>0\.\d+)</code>",
        text,
    )
    if match is None:
        raise ValueError("Exp463 Wilson upper evidence is missing from tracked reports")
    return float(match.group("wilson"))


def _extract_report_status_antigravity(report_text: str) -> str:
    match = re.search(r"<strong>Status:</strong> <code>(?P<status>[^<]+)</code>", report_text)
    if match is None:
        raise ValueError("Exp463 Task 5 report status is missing")
    return match.group("status")


def _build_summary_antigravity(
    *,
    method: str,
    fold_count: int,
    passed_count: int,
    min_gradual_drift_auc: float,
    min_period_change_auc: float,
    min_pooled_auc: float,
    max_normal_wilson_upper: float,
    all_scores_finite: bool,
) -> Exp463MethodSummaryAntigravity:
    return Exp463MethodSummaryAntigravity(
        method=method,
        fold_count=fold_count,
        passed_count=passed_count,
        failed_count=fold_count - passed_count,
        min_gradual_drift_auc=min_gradual_drift_auc,
        min_period_change_auc=min_period_change_auc,
        min_pooled_auc=min_pooled_auc,
        max_normal_wilson_upper=max_normal_wilson_upper,
        all_scores_finite=all_scores_finite,
    )


def _extract_report_control_summaries_antigravity(report_text: str) -> dict[str, Exp463MethodSummaryAntigravity]:
    pattern = re.compile(
        r"<tr><td>M(?P<idx>[012456])(?: [^<]*)?</td><td>(?P<passed>\d+)/(?P<fold>\d+)</td>"
        r"<td>(?P<gradual>0\.\d+)</td><td>(?P<period>0\.\d+)</td><td>(?P<pooled>0\.\d+)</td>",
    )
    all_rows = {match.group("idx"): match.groupdict() for match in pattern.finditer(report_text)}
    summary_rows = {
        report_id: all_rows[report_id]
        for report_id in _CONTROL_METHOD_MAP_ANTIGRAVITY
        if report_id in all_rows
    }
    if tuple(summary_rows) != tuple(_CONTROL_METHOD_MAP_ANTIGRAVITY):
        raise ValueError("Exp463 Task 5 report control rows drifted")
    wilson_upper = _extract_wilson_upper_antigravity(report_text)
    result: dict[str, Exp463MethodSummaryAntigravity] = {}
    for report_id, method in _CONTROL_METHOD_MAP_ANTIGRAVITY.items():
        row = summary_rows[report_id]
        result[method] = _build_summary_antigravity(
            method=method,
            fold_count=int(row["fold"]),
            passed_count=int(row["passed"]),
            min_gradual_drift_auc=float(row["gradual"]),
            min_period_change_auc=float(row["period"]),
            min_pooled_auc=float(row["pooled"]),
            max_normal_wilson_upper=wilson_upper,
            all_scores_finite=True,
        )
    return result


def _extract_implementation_control_summaries_antigravity(
    implementation_report_text: str,
) -> dict[str, Exp463MethodSummaryAntigravity]:
    pattern = re.compile(
        r"<tr><td>M(?P<idx>[012456])(?: [^<]*)?</td><td>(?P<passed>\d+)/(?P<fold>\d+)</td>"
        r"<td>(?P<gradual>0\.\d+)</td><td>(?P<period>0\.\d+)</td><td>(?P<pooled>0\.\d+)</td>"
        r"<td>(?P<finite>true|false)</td></tr>",
    )
    all_rows = {
        match.group("idx"): match.groupdict()
        for match in pattern.finditer(implementation_report_text)
    }
    summary_rows = {
        report_id: all_rows[report_id]
        for report_id in _CONTROL_METHOD_MAP_ANTIGRAVITY
        if report_id in all_rows
    }
    if tuple(summary_rows) != tuple(_CONTROL_METHOD_MAP_ANTIGRAVITY):
        raise ValueError("Exp463 implementation report control rows drifted")
    wilson_upper = _extract_wilson_upper_antigravity(implementation_report_text)
    result: dict[str, Exp463MethodSummaryAntigravity] = {}
    for report_id, method in _CONTROL_METHOD_MAP_ANTIGRAVITY.items():
        row = summary_rows[report_id]
        result[method] = _build_summary_antigravity(
            method=method,
            fold_count=int(row["fold"]),
            passed_count=int(row["passed"]),
            min_gradual_drift_auc=float(row["gradual"]),
            min_period_change_auc=float(row["period"]),
            min_pooled_auc=float(row["pooled"]),
            max_normal_wilson_upper=wilson_upper,
            all_scores_finite=row["finite"] == "true",
        )
    return result


def _validate_report_evidence_antigravity(
    report_text: str,
    implementation_report_text: str,
    task5_result: dict[str, Any],
) -> dict[str, Exp463MethodSummaryAntigravity]:
    _exact_value_antigravity(
        _extract_report_status_antigravity(report_text),
        "stopped_before_official_evaluation_qualification_failed",
        "task5_report.status",
    )
    for required in (
        task5_result["reviewed_commit"],
        task5_result["artifact_map_sha256"],
        "evaluate 0 / inspect 0; official root absent;",
        "label query count 0",
        "passed 7/10",
        "Task 6 remains explicitly forbidden",
    ):
        if required not in report_text:
            raise ValueError("Exp463 Task 5 report drifted from frozen terminal truth")
    for required in (
        task5_result["reviewed_commit"],
        task5_result["artifact_map_sha256"],
        "official evaluate 0 / inspect 0",
        "official root absent",
        "label query count 0",
        "all official result hashes remain <code>null</code>",
        "There is no next official action.",
    ):
        if required not in implementation_report_text:
            raise ValueError("Exp463 implementation report drifted from frozen terminal truth")
    report_summaries = _extract_report_control_summaries_antigravity(report_text)
    implementation_summaries = _extract_implementation_control_summaries_antigravity(
        implementation_report_text
    )
    if implementation_summaries != report_summaries:
        raise ValueError("Exp463 implementation/report control summaries drifted")
    return report_summaries


def _control_row_digests_antigravity() -> dict[str, str]:
    result = dict(FROZEN_CONFIG_ANTIGRAVITY.predecessor_control_row_digests)
    expected = tuple(FROZEN_CONFIG_ANTIGRAVITY.predecessor_control_row_digests)
    if tuple(result) != expected:
        raise ValueError("Exp463 control row digests drifted from frozen order")
    return result


def capture_exp463_authority_antigravity() -> Exp463AuthorityAntigravity:
    """Return the exact frozen Exp463 private TRAIN authority consumed by Exp464."""

    report_text = _report_bytes_antigravity().decode("utf-8")
    implementation_report_text = _IMPLEMENTATION_REPORT_ANTIGRAVITY.read_text(encoding="utf-8")
    state, catalog, champion = _exp463_ledger_rows_antigravity()
    task5_result = _cross_check_ledgers_antigravity(state, catalog, champion)
    terminal_closeout_commit = _tracked_terminal_closeout_commit_antigravity()

    if task5_result["artifact_map_sha256"] not in implementation_report_text:
        raise ValueError("Exp463 implementation report drifted from tracked artifact_map_sha256")
    if task5_result["reviewed_commit"] not in implementation_report_text:
        raise ValueError("Exp463 implementation report drifted from tracked reviewed_commit")

    summaries = _validate_report_evidence_antigravity(
        report_text,
        implementation_report_text,
        task5_result,
    )
    authority = Exp463AuthorityAntigravity(
        predecessor_experiment=FROZEN_CONFIG_ANTIGRAVITY.predecessor_experiment,
        terminal_closeout_commit=terminal_closeout_commit,
        task5_report_sha256=FROZEN_CONFIG_ANTIGRAVITY.predecessor_task5_report_sha256,
        status=state["status"],
        reviewed_cli_commit=task5_result["reviewed_commit"],
        private_run_count=len(task5_result["runs"]),
        result_row_count=task5_result["row_count"],
        result_failed_row_count=task5_result["failed_row_count"],
        result_summary_count=task5_result["summary_count"],
        result_leaf_count=task5_result["leaf_count"],
        method_fold_metrics_sha256=task5_result["leaf_sha256"]["method_fold_metrics_antigravity.jsonl"],
        artifact_map_sha256=task5_result["artifact_map_sha256"],
        official_evaluate_count=state["official_evaluate_invocations"],
        official_inspect_count=state["official_inspect_invocations"],
        label_query_count=state["label_query_count"],
        control_summaries=summaries,
        control_row_digests=_control_row_digests_antigravity(),
    )
    validate_exp463_authority_antigravity(authority)
    return authority


__all__ = ("capture_exp463_authority_antigravity",)
