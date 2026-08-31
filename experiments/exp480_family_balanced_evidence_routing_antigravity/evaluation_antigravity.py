"""Identical out-of-family qualification for Exp480 and its controls."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from experiments.exp476_minimum_zf_fallback_routing_antigravity.policy_antigravity import (
    MinimumZFGuardKNNPolicyAntigravity,
)

from .config_antigravity import (
    FAMILY_POLICY_CONFIG_ANTIGRAVITY,
    FEATURE_KEYS_ANTIGRAVITY,
    METHODS_ANTIGRAVITY,
    ROW_POLICY_CONFIG_ANTIGRAVITY,
)
from .contracts_antigravity import CandidateMetricAntigravity, EvidenceRowAntigravity
from .policy_antigravity import FamilyBalancedCandidatePolicyAntigravity


BASELINE_METHOD_ANTIGRAVITY = METHODS_ANTIGRAVITY[0]
EVALUATION_BACKEND_ANTIGRAVITY = "cpu_numpy"


def _route_input_audit_antigravity() -> dict[str, object]:
    route_input_keys = tuple(FEATURE_KEYS_ANTIGRAVITY)
    name_keys = tuple(
        key
        for key in route_input_keys
        if any(token in key for token in ("dataset", "family", "name"))
    )
    score_keys = tuple(
        key
        for key in route_input_keys
        if any(token in key for token in ("auc", "zf", "score", "label"))
    )
    return {
        "route_input_keys": list(route_input_keys),
        "route_input_key_count": len(route_input_keys),
        "dataset_name_route_input_keys": list(name_keys),
        "held_out_score_route_input_keys": list(score_keys),
        "dataset_name_feature_count": len(name_keys),
        "held_out_score_feature_count": len(score_keys),
    }


def _feature_mapping_antigravity(row: EvidenceRowAntigravity) -> dict[str, float]:
    return dict(zip(FEATURE_KEYS_ANTIGRAVITY, row.features, strict=True))


def _score_mapping_antigravity(
    row: EvidenceRowAntigravity,
) -> dict[str, CandidateMetricAntigravity]:
    return {metric.method: metric for metric in row.scores}


def iter_family_splits_antigravity(
    rows: Sequence[EvidenceRowAntigravity],
) -> Iterator[
    tuple[str, tuple[EvidenceRowAntigravity, ...], tuple[EvidenceRowAntigravity, ...]]
]:
    """Yield deterministic full-family holdouts with no dataset overlap."""
    ordered = tuple(sorted(rows, key=lambda row: row.dataset))
    if not ordered or any(type(row) is not EvidenceRowAntigravity for row in ordered):
        raise ValueError("rows must contain strict EvidenceRowAntigravity values")
    if len({row.dataset for row in ordered}) != len(ordered):
        raise ValueError("dataset names must be unique")
    for held_family in sorted({row.source_family for row in ordered}):
        train_rows = tuple(row for row in ordered if row.source_family != held_family)
        test_rows = tuple(row for row in ordered if row.source_family == held_family)
        if not train_rows or not test_rows:
            raise ValueError("each family split requires train and test rows")
        yield held_family, train_rows, test_rows


def _fit_row_for_control_antigravity(row: EvidenceRowAntigravity) -> dict[str, object]:
    scores = _score_mapping_antigravity(row)
    baseline = scores[BASELINE_METHOD_ANTIGRAVITY]
    candidate = scores[row.proposal]
    return {
        "features": _feature_mapping_antigravity(row),
        "proposal": row.proposal,
        "auc_gain": candidate.auc - baseline.auc,
        "zf_gain": candidate.zf - baseline.zf,
    }


def summarise_method_rows_antigravity(
    rows: Iterable[Mapping[str, object]],
    *,
    accepted_non_baseline: int,
) -> dict[str, object]:
    """Report dataset-weighted and family-macro metrics separately."""
    measured = tuple(rows)
    if not measured:
        raise ValueError("summary rows must not be empty")
    if type(accepted_non_baseline) is not int or accepted_non_baseline < 0:
        raise ValueError("accepted_non_baseline must be a nonnegative integer")
    aucs = np.asarray([float(row["auc"]) for row in measured], dtype=np.float64)
    zfs = np.asarray([int(row["zf"]) for row in measured], dtype=np.int64)
    baseline_aucs = np.asarray(
        [float(row["baseline_auc"]) for row in measured], dtype=np.float64
    )
    if not np.isfinite(aucs).all() or not np.isfinite(baseline_aucs).all():
        raise ValueError("summary AUC values must be finite")
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(measured):
        family = row.get("held_family")
        if type(family) is not str or not family:
            raise ValueError("each summary row requires held_family")
        by_family[family].append(index)
    family_auc_means = [float(np.mean(aucs[indices])) for indices in by_family.values()]
    family_zf_means = [float(np.mean(zfs[indices])) for indices in by_family.values()]
    losses = baseline_aucs - aucs
    return {
        "dataset_count": len(measured),
        "dataset_mean_auc": float(np.mean(aucs)),
        "zf_total": int(np.sum(zfs)),
        "family_mean_auc": float(np.mean(family_auc_means)),
        "family_mean_zf": float(np.mean(family_zf_means)),
        "loss_over_0_05": int(np.sum(losses > 0.05)),
        "loss_over_0_10": int(np.sum(losses > 0.10)),
        "accepted_non_baseline": accepted_non_baseline,
    }


def apply_qualification_gates_antigravity(
    measured: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Apply the frozen strict-ZF, non-regressing-AUC qualification gates."""
    try:
        mini = measured["mini"]
        row_knn = measured["row_knn"]
        exp480 = measured["exp480"]
    except KeyError as error:
        raise ValueError("mini, row_knn and exp480 summaries are required") from error
    gates: dict[str, object] = {
        "zf_total": {
            "actual": int(exp480["zf_total"]),
            "required": f"> {max(int(mini['zf_total']), int(row_knn['zf_total']))}",
            "pass": int(exp480["zf_total"])
            > max(int(mini["zf_total"]), int(row_knn["zf_total"])),
        },
        "family_mean_zf": {
            "actual": float(exp480["family_mean_zf"]),
            "required": f"> {max(float(mini['family_mean_zf']), float(row_knn['family_mean_zf']))}",
            "pass": float(exp480["family_mean_zf"])
            > max(float(mini["family_mean_zf"]), float(row_knn["family_mean_zf"])),
        },
        "dataset_mean_auc": {
            "actual": float(exp480["dataset_mean_auc"]),
            "required": f">= {max(float(mini['dataset_mean_auc']), float(row_knn['dataset_mean_auc']))}",
            "pass": float(exp480["dataset_mean_auc"])
            >= max(float(mini["dataset_mean_auc"]), float(row_knn["dataset_mean_auc"])),
        },
        "family_mean_auc": {
            "actual": float(exp480["family_mean_auc"]),
            "required": f">= {max(float(mini['family_mean_auc']), float(row_knn['family_mean_auc']))}",
            "pass": float(exp480["family_mean_auc"])
            >= max(float(mini["family_mean_auc"]), float(row_knn["family_mean_auc"])),
        },
        "loss_over_0_05": {
            "actual": int(exp480["loss_over_0_05"]),
            "required": f"<= {int(row_knn['loss_over_0_05'])}",
            "pass": int(exp480["loss_over_0_05"])
            <= int(row_knn["loss_over_0_05"]),
        },
        "loss_over_0_10": {
            "actual": int(exp480["loss_over_0_10"]),
            "required": f"<= {int(row_knn['loss_over_0_10'])}",
            "pass": int(exp480["loss_over_0_10"])
            <= int(row_knn["loss_over_0_10"]),
        },
        "accepted_non_baseline": {
            "actual": int(exp480["accepted_non_baseline"]),
            "required": ">= 1",
            "pass": int(exp480["accepted_non_baseline"]) >= 1,
        },
    }
    gates["overall_pass"] = all(
        bool(value["pass"]) for value in gates.values() if isinstance(value, dict)
    )
    return gates


def evaluate_out_of_family_antigravity(
    rows: Sequence[EvidenceRowAntigravity],
    *,
    family_policy_kwargs: Mapping[str, object] | None = None,
    row_policy_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate all three policies using the same held-family training rows."""
    family_kwargs = dict(FAMILY_POLICY_CONFIG_ANTIGRAVITY)
    family_kwargs.update(family_policy_kwargs or {})
    row_kwargs = dict(ROW_POLICY_CONFIG_ANTIGRAVITY)
    row_kwargs.update(row_policy_kwargs or {})
    decisions: list[dict[str, object]] = []
    family_overlap_count = 0
    for held_family, train_rows, test_rows in iter_family_splits_antigravity(rows):
        train_families = {row.source_family for row in train_rows}
        test_families = {row.source_family for row in test_rows}
        family_overlap_count += len(train_families & test_families)
        row_policy = MinimumZFGuardKNNPolicyAntigravity(**row_kwargs).fit(
            _fit_row_for_control_antigravity(row) for row in train_rows
        )
        family_policy = FamilyBalancedCandidatePolicyAntigravity(
            **family_kwargs
        ).fit(train_rows)
        for row in test_rows:
            features = _feature_mapping_antigravity(row)
            row_decision = row_policy.route(features, row.proposal)
            family_decision = family_policy.route(features)
            candidate_assessments = tuple(
                {
                    "method": assessment.method,
                    "passed": assessment.passed,
                    "support": assessment.support,
                    "positive_auc_rate": assessment.positive_auc_rate,
                    "positive_zf_rate": assessment.positive_zf_rate,
                    "lower_auc_gain": assessment.lower_auc_gain,
                    "mean_zf_gain": assessment.mean_zf_gain,
                    "neighbour_families": assessment.neighbour_families,
                }
                for assessment in family_decision.candidates
            )
            scores = _score_mapping_antigravity(row)
            mini = scores[BASELINE_METHOD_ANTIGRAVITY]
            row_score = scores[row_decision.method]
            family_score = scores[family_decision.method]
            decisions.append(
                {
                    "dataset": row.dataset,
                    "held_family": held_family,
                    "proposal": row.proposal,
                    "mini_method": BASELINE_METHOD_ANTIGRAVITY,
                    "row_knn_method": row_decision.method,
                    "exp480_method": family_decision.method,
                    "exp480_candidate": family_decision.candidate,
                    "exp480_accepted": family_decision.accepted,
                    "exp480_reason": family_decision.reason,
                    "support": family_decision.support,
                    "neighbour_families": family_decision.neighbour_families,
                    "selected_positive_auc_rate": family_decision.positive_auc_rate,
                    "selected_positive_zf_rate": family_decision.positive_zf_rate,
                    "selected_lower_auc_gain": family_decision.lower_auc_gain,
                    "selected_mean_zf_gain": family_decision.mean_zf_gain,
                    "candidate_assessments": candidate_assessments,
                    "mini_auc": mini.auc,
                    "mini_zf": mini.zf,
                    "row_knn_auc": row_score.auc,
                    "row_knn_zf": row_score.zf,
                    "exp480_auc": family_score.auc,
                    "exp480_zf": family_score.zf,
                }
            )
    decisions.sort(key=lambda row: str(row["dataset"]))

    def summary_rows(prefix: str) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "held_family": row["held_family"],
                "auc": row[f"{prefix}_auc"],
                "zf": row[f"{prefix}_zf"],
                "baseline_auc": row["mini_auc"],
            }
            for row in decisions
        )

    summaries = {
        "mini": summarise_method_rows_antigravity(
            summary_rows("mini"), accepted_non_baseline=0
        ),
        "row_knn": summarise_method_rows_antigravity(
            summary_rows("row_knn"),
            accepted_non_baseline=sum(
                row["row_knn_method"] != BASELINE_METHOD_ANTIGRAVITY
                for row in decisions
            ),
        ),
        "exp480": summarise_method_rows_antigravity(
            summary_rows("exp480"),
            accepted_non_baseline=sum(
                bool(row["exp480_accepted"]) for row in decisions
            ),
        ),
    }
    route_input_audit = _route_input_audit_antigravity()
    return {
        "backend": EVALUATION_BACKEND_ANTIGRAVITY,
        "dataset_count": len(decisions),
        "family_count": len({str(row["held_family"]) for row in decisions}),
        "summaries": summaries,
        "gates": apply_qualification_gates_antigravity(summaries),
        "leakage": {
            "family_overlap_count": family_overlap_count,
            **route_input_audit,
            "held_out_scores_read_after_routing": True,
        },
        "decisions": decisions,
    }


__all__ = [
    "EVALUATION_BACKEND_ANTIGRAVITY",
    "apply_qualification_gates_antigravity",
    "evaluate_out_of_family_antigravity",
    "iter_family_splits_antigravity",
    "summarise_method_rows_antigravity",
]
