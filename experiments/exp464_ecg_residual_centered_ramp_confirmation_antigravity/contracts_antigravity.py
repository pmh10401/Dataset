"""Immutable Task 2 contracts for Exp464 and frozen Exp463 authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
import re
from types import MappingProxyType

from .config_antigravity import (
    FROZEN_CONFIG_ANTIGRAVITY,
    Exp463MethodSummarySnapshotAntigravity as Exp463MethodSummaryAntigravity,
)


_CONTROL_ORDER_ANTIGRAVITY = tuple(FROZEN_CONFIG_ANTIGRAVITY.predecessor_control_summaries)


def _int_antigravity(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _sha64_antigravity(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex digest")
    return value


def _summary_field_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _summary_antigravity(value: object) -> Exp463MethodSummaryAntigravity:
    if type(value) is not Exp463MethodSummaryAntigravity:
        raise ValueError("control_summaries must contain exact Exp463MethodSummaryAntigravity values")
    return value


def _frozen_control_summaries_antigravity(
    value: Mapping[str, Exp463MethodSummaryAntigravity] | object,
) -> MappingProxyType[str, Exp463MethodSummaryAntigravity]:
    if not isinstance(value, Mapping):
        raise ValueError("control_summaries must be a mapping")
    if tuple(value) != _CONTROL_ORDER_ANTIGRAVITY:
        raise ValueError("control_summaries must preserve exact C0/C1/C2/C7 order")
    frozen = {
        method: _summary_antigravity(value[method])  # type: ignore[index]
        for method in _CONTROL_ORDER_ANTIGRAVITY
    }
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class Exp463AuthorityAntigravity:
    predecessor_experiment: str
    terminal_closeout_commit: str
    task5_report_sha256: str
    status: str
    reviewed_cli_commit: str
    private_run_count: int
    result_row_count: int
    result_failed_row_count: int
    result_summary_count: int
    result_leaf_count: int
    artifact_map_sha256: str
    official_evaluate_count: int
    official_inspect_count: int
    label_query_count: int
    control_summaries: Mapping[str, Exp463MethodSummaryAntigravity]

    def __post_init__(self) -> None:
        if type(self.predecessor_experiment) is not str or self.predecessor_experiment != "Exp463":
            raise ValueError("predecessor_experiment must be Exp463")
        if type(self.terminal_closeout_commit) is not str or re.fullmatch(r"[0-9a-f]{40}", self.terminal_closeout_commit) is None:
            raise ValueError("terminal_closeout_commit must be a lowercase 40-hex commit")
        _sha64_antigravity(self.task5_report_sha256, "task5_report_sha256")
        if type(self.status) is not str or not self.status:
            raise ValueError("status must be a non-empty string")
        if type(self.reviewed_cli_commit) is not str or re.fullmatch(r"[0-9a-f]{40}", self.reviewed_cli_commit) is None:
            raise ValueError("reviewed_cli_commit must be a lowercase 40-hex commit")
        _int_antigravity(self.private_run_count, "private_run_count", minimum=1)
        _int_antigravity(self.result_row_count, "result_row_count", minimum=1)
        _int_antigravity(self.result_failed_row_count, "result_failed_row_count")
        _int_antigravity(self.result_summary_count, "result_summary_count", minimum=1)
        _int_antigravity(self.result_leaf_count, "result_leaf_count", minimum=1)
        _sha64_antigravity(self.artifact_map_sha256, "artifact_map_sha256")
        _int_antigravity(self.official_evaluate_count, "official_evaluate_count")
        _int_antigravity(self.official_inspect_count, "official_inspect_count")
        _int_antigravity(self.label_query_count, "label_query_count")
        object.__setattr__(
            self,
            "control_summaries",
            _frozen_control_summaries_antigravity(self.control_summaries),
        )


def validate_exp463_authority_antigravity(authority: Exp463AuthorityAntigravity) -> None:
    if type(authority) is not Exp463AuthorityAntigravity:
        raise ValueError("authority must be Exp463AuthorityAntigravity")
    expected = FROZEN_CONFIG_ANTIGRAVITY
    expected_scalars = {
        "predecessor_experiment": expected.predecessor_experiment,
        "terminal_closeout_commit": expected.predecessor_terminal_closeout_commit,
        "task5_report_sha256": expected.predecessor_task5_report_sha256,
        "status": "stopped_before_official_evaluation_qualification_failed",
        "reviewed_cli_commit": "923cc4ad6dd6737f84f4ea2c4f621eb652ac54b4",
        "private_run_count": expected.predecessor_private_run_count,
        "result_row_count": expected.predecessor_result_row_count,
        "result_failed_row_count": expected.predecessor_result_failed_row_count,
        "result_summary_count": expected.predecessor_result_summary_count,
        "result_leaf_count": expected.predecessor_result_leaf_count,
        "artifact_map_sha256": expected.predecessor_artifact_map_sha256,
        "official_evaluate_count": expected.official_evaluate_count_must_remain,
        "official_inspect_count": expected.official_inspect_count_must_remain,
        "label_query_count": expected.label_query_count_must_remain,
    }
    for field_name, expected_value in expected_scalars.items():
        actual_value = getattr(authority, field_name)
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ValueError(f"{field_name} differs from the frozen Exp463 authority")
    if tuple(authority.control_summaries) != _CONTROL_ORDER_ANTIGRAVITY:
        raise ValueError("control_summaries order differs from the frozen Exp463 authority")
    for method, expected_summary in expected.predecessor_control_summaries.items():
        actual_summary = authority.control_summaries[method]
        if type(actual_summary) is not Exp463MethodSummaryAntigravity:
            raise ValueError(f"{method} summary type differs from the frozen Exp463 authority")
        for field_name in (
            "method",
            "fold_count",
            "passed_count",
            "failed_count",
            "min_gradual_drift_auc",
            "min_period_change_auc",
            "min_pooled_auc",
            "max_normal_wilson_upper",
            "all_scores_finite",
        ):
            actual_value = getattr(actual_summary, field_name)
            expected_value = getattr(expected_summary, field_name)
            if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                raise ValueError(
                    f"{method}.{field_name} differs from the frozen Exp463 authority"
                )


__all__ = (
    "Exp463AuthorityAntigravity",
    "Exp463MethodSummaryAntigravity",
    "validate_exp463_authority_antigravity",
)
