"""Nested out-of-family evaluation for Exp481 strict residual correction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from experiments.exp476_minimum_zf_fallback_routing_antigravity.policy_antigravity import (
    MinimumZFGuardKNNPolicyAntigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.config_antigravity import (
    ROW_POLICY_CONFIG_ANTIGRAVITY,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.evaluation_antigravity import (
    iter_family_splits_antigravity,
)

from .config_antigravity import (
    ALTERNATIVE_METHODS_ANTIGRAVITY,
    BASELINE_METHOD_ANTIGRAVITY,
    EXP481_THRESHOLDS_ANTIGRAVITY,
    FEATURE_KEYS_ANTIGRAVITY,
    INNER_MIN_CHANGED_FAMILIES_ANTIGRAVITY,
    K_ANTIGRAVITY,
    METHODS_ANTIGRAVITY,
)
from .contracts_antigravity import InnerCandidateGateAntigravity
from .policy_antigravity import StrictResidualPolicyAntigravity


EVALUATION_BACKEND_ANTIGRAVITY = "numpy_reference"


def _exact_method_antigravity(value: object, label: str) -> str:
    if type(value) is not str or value not in METHODS_ANTIGRAVITY:
        raise ValueError(f"{label} must be one of the four frozen methods")
    return value


@dataclass(frozen=True)
class MeasuredDecisionAntigravity:
    dataset: str
    held_family: str
    row_knn_method: str
    method: str
    changed: bool
    auc: float
    zf: int
    row_knn_auc: float
    row_knn_zf: int
    auc_gain: float
    zf_gain: int
    reason: str
    inner_enabled_methods: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or "_normal_" not in self.dataset:
            raise ValueError("dataset must contain _normal_")
        expected_family = self.dataset.split("_normal_", 1)[0]
        if type(self.held_family) is not str or self.held_family != expected_family:
            raise ValueError("held_family must match the dataset family")
        row_method = _exact_method_antigravity(self.row_knn_method, "row_knn_method")
        method = _exact_method_antigravity(self.method, "method")
        if type(self.changed) is not bool or self.changed != (method != row_method):
            raise ValueError("changed must exactly describe a method change")
        for label, value in (
            ("auc", self.auc),
            ("row_knn_auc", self.row_knn_auc),
            ("auc_gain", self.auc_gain),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{label} must be an exact finite float")
        if not 0.0 <= self.auc <= 1.0 or not 0.0 <= self.row_knn_auc <= 1.0:
            raise ValueError("AUC values must be in [0, 1]")
        if type(self.zf) is not int or self.zf < 0:
            raise ValueError("zf must be an exact nonnegative integer")
        if type(self.row_knn_zf) is not int or self.row_knn_zf < 0:
            raise ValueError("row_knn_zf must be an exact nonnegative integer")
        if type(self.zf_gain) is not int:
            raise ValueError("zf_gain must be an exact integer")
        if not math.isclose(self.auc - self.row_knn_auc, self.auc_gain, abs_tol=1e-12):
            raise ValueError("auc_gain must match selected minus row-KNN AUC")
        if self.zf - self.row_knn_zf != self.zf_gain:
            raise ValueError("zf_gain must match selected minus row-KNN ZF")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if type(self.inner_enabled_methods) is not tuple:
            raise ValueError("inner_enabled_methods must be a tuple")
        if any(method not in ALTERNATIVE_METHODS_ANTIGRAVITY for method in self.inner_enabled_methods):
            raise ValueError("inner_enabled_methods must contain frozen alternatives")
        if len(set(self.inner_enabled_methods)) != len(self.inner_enabled_methods):
            raise ValueError("inner_enabled_methods must be unique")


def _feature_mapping_antigravity(row: EvidenceRowAntigravity) -> dict[str, float]:
    return dict(zip(FEATURE_KEYS_ANTIGRAVITY, row.features, strict=True))


def _score_mapping_after_decision_antigravity(
    row: EvidenceRowAntigravity,
) -> dict[str, CandidateMetricAntigravity]:
    """Held-row score boundary: callers invoke this only after routing."""
    return {metric.method: metric for metric in row.scores}


def _fit_row_for_control_antigravity(row: EvidenceRowAntigravity) -> dict[str, object]:
    scores = {metric.method: metric for metric in row.scores}
    baseline = scores[BASELINE_METHOD_ANTIGRAVITY]
    candidate = scores[row.proposal]
    return {
        "features": _feature_mapping_antigravity(row),
        "proposal": row.proposal,
        "auc_gain": candidate.auc - baseline.auc,
        "zf_gain": candidate.zf - baseline.zf,
    }


def aggregate_inner_gate_antigravity(
    method: str,
    selected_records: Iterable[MeasuredDecisionAntigravity],
    *,
    min_changed_families: int = INNER_MIN_CHANGED_FAMILIES_ANTIGRAVITY,
) -> InnerCandidateGateAntigravity:
    if method not in ALTERNATIVE_METHODS_ANTIGRAVITY:
        raise ValueError("method must be a frozen alternative")
    if type(min_changed_families) is not int or min_changed_families < 1:
        raise ValueError("min_changed_families must be a positive integer")
    by_family: dict[str, list[MeasuredDecisionAntigravity]] = defaultdict(list)
    for record in selected_records:
        if type(record) is not MeasuredDecisionAntigravity:
            raise ValueError("selected_records must contain strict measured decisions")
        if record.changed and record.method == method:
            by_family[record.held_family].append(record)
    if not by_family:
        return InnerCandidateGateAntigravity(
            method=method,
            enabled=False,
            changed_families=0,
            winning_families=0,
            losing_families=0,
            tied_families=0,
            min_auc_gain=0.0,
            q20_auc_gain=0.0,
        )
    family_zf = {
        family: sum(item.zf_gain for item in items) for family, items in by_family.items()
    }
    auc_gains = np.asarray(
        [item.auc_gain for items in by_family.values() for item in items], dtype=np.float64
    )
    wins = sum(value > 0 for value in family_zf.values())
    losses = sum(value < 0 for value in family_zf.values())
    ties = len(family_zf) - wins - losses
    minimum = float(np.min(auc_gains))
    q20 = float(np.quantile(auc_gains, 0.20))
    enabled = bool(
        len(by_family) >= min_changed_families
        and wins > losses
        and minimum >= 0.0
        and q20 > 0.0
    )
    return InnerCandidateGateAntigravity(
        method=method,
        enabled=enabled,
        changed_families=len(by_family),
        winning_families=wins,
        losing_families=losses,
        tied_families=ties,
        min_auc_gain=minimum,
        q20_auc_gain=q20,
    )


def _row_policy_antigravity(
    rows: Sequence[EvidenceRowAntigravity], kwargs: Mapping[str, object] | None
) -> MinimumZFGuardKNNPolicyAntigravity:
    config = dict(ROW_POLICY_CONFIG_ANTIGRAVITY)
    config.update(kwargs or {})
    return MinimumZFGuardKNNPolicyAntigravity(**config).fit(
        _fit_row_for_control_antigravity(row) for row in rows
    )


def evaluate_outer_row_antigravity(
    row: EvidenceRowAntigravity,
    row_policy: object,
    residual_policy: StrictResidualPolicyAntigravity,
    enabled_methods: Iterable[str],
) -> MeasuredDecisionAntigravity:
    enabled = tuple(sorted(enabled_methods))
    features = _feature_mapping_antigravity(row)
    row_decision = row_policy.route(features, row.proposal)
    descriptors = getattr(residual_policy, "descriptor_values", None)
    residual_features = features if descriptors is None else dict(zip(
        residual_policy.feature_keys, descriptors[row.dataset], strict=True
    ))
    residual_decision = residual_policy.route(
        residual_features,
        row_knn_method=row_decision.method,
        enabled_methods=None,
    )
    final_method = residual_decision.method
    final_changed = residual_decision.changed
    final_reason = residual_decision.reason
    if (
        residual_decision.changed
        and residual_decision.selected_candidate not in enabled
    ):
        # The approved order is local rank first, then the inner safety gate.
        # Never rerank a runner-up after the local winner fails the inner gate.
        final_method = row_decision.method
        final_changed = False
        final_reason = "inner_candidate_gate_failed"
    scores = _score_mapping_after_decision_antigravity(row)
    selected = scores[final_method]
    baseline = scores[row_decision.method]
    return MeasuredDecisionAntigravity(
        dataset=row.dataset,
        held_family=row.source_family,
        row_knn_method=row_decision.method,
        method=final_method,
        changed=final_changed,
        auc=float(selected.auc),
        zf=int(selected.zf),
        row_knn_auc=float(baseline.auc),
        row_knn_zf=int(baseline.zf),
        auc_gain=float(selected.auc - baseline.auc),
        zf_gain=int(selected.zf - baseline.zf),
        reason=final_reason,
        inner_enabled_methods=enabled,
    )


def qualify_inner_methods_antigravity(
    outer_train_rows: Sequence[EvidenceRowAntigravity],
    *,
    row_policy_kwargs: Mapping[str, object] | None = None,
    residual_policy_kwargs: Mapping[str, object] | None = None,
    min_changed_families: int = INNER_MIN_CHANGED_FAMILIES_ANTIGRAVITY,
) -> tuple[InnerCandidateGateAntigravity, ...]:
    residual_kwargs = {"k": K_ANTIGRAVITY}
    residual_kwargs.update(residual_policy_kwargs or {})
    selected: list[MeasuredDecisionAntigravity] = []
    for _inner_family, inner_train, inner_test in iter_family_splits_antigravity(
        outer_train_rows
    ):
        row_policy = _row_policy_antigravity(inner_train, row_policy_kwargs)
        residual_policy = StrictResidualPolicyAntigravity(**residual_kwargs).fit(inner_train)
        for row in inner_test:
            measured = evaluate_outer_row_antigravity(
                row,
                row_policy,
                residual_policy,
                ALTERNATIVE_METHODS_ANTIGRAVITY,
            )
            if measured.changed:
                selected.append(measured)
    return tuple(
        aggregate_inner_gate_antigravity(
            method, selected, min_changed_families=min_changed_families
        )
        for method in ALTERNATIVE_METHODS_ANTIGRAVITY
    )


def _summary_antigravity(
    decisions: Sequence[MeasuredDecisionAntigravity], *, selected: bool
) -> dict[str, object]:
    if not decisions:
        raise ValueError("decisions must not be empty")
    aucs = np.asarray(
        [item.auc if selected else item.row_knn_auc for item in decisions], dtype=np.float64
    )
    zfs = np.asarray(
        [item.zf if selected else item.row_knn_zf for item in decisions], dtype=np.int64
    )
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(decisions):
        by_family[item.held_family].append(index)
    result: dict[str, object] = {
        "dataset_count": len(decisions),
        "dataset_mean_auc": float(np.mean(aucs)),
        "zf_total": int(np.sum(zfs)),
        "family_mean_auc": float(
            np.mean([np.mean(aucs[indices]) for indices in by_family.values()])
        ),
        "family_mean_zf": float(
            np.mean([np.mean(zfs[indices]) for indices in by_family.values()])
        ),
    }
    if selected:
        gains = np.asarray([item.auc_gain for item in decisions], dtype=np.float64)
        result.update(
            added_loss_0_05=int(np.sum(gains <= -0.05)),
            added_loss_0_10=int(np.sum(gains <= -0.10)),
        )
    return result


def _changed_summary_antigravity(
    decisions: Sequence[MeasuredDecisionAntigravity],
) -> dict[str, object]:
    changed = tuple(item for item in decisions if item.changed)
    wins = sum(item.zf_gain > 0 for item in changed)
    losses = sum(item.zf_gain < 0 for item in changed)
    ties = len(changed) - wins - losses
    counts = Counter(item.method for item in changed)
    dominant_method = None
    dominance_rate = 0.0
    if changed:
        dominant_method, dominant_count = sorted(
            counts.items(), key=lambda pair: (-pair[1], pair[0])
        )[0]
        dominance_rate = dominant_count / len(changed)
    return {
        "count": len(changed),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "tie_rate": ties / len(changed) if changed else 0.0,
        "method_counts": dict(sorted(counts.items())),
        "dominance_rate": dominance_rate,
        "dominant_method": dominant_method,
        "dominance_control": None,
    }


def _dominance_control_antigravity(
    rows: Sequence[EvidenceRowAntigravity],
    decisions: Sequence[MeasuredDecisionAntigravity],
    changed: dict[str, object],
) -> dict[str, object] | None:
    threshold = float(EXP481_THRESHOLDS_ANTIGRAVITY["dominance_control_rate"])
    method = changed["dominant_method"]
    if method is None or float(changed["dominance_rate"]) < threshold:
        return None
    rows_by_dataset = {row.dataset: row for row in rows}
    aucs: list[float] = []
    zfs: list[int] = []
    for decision in decisions:
        scores = {metric.method: metric for metric in rows_by_dataset[decision.dataset].scores}
        selected_method = method if decision.row_knn_method == BASELINE_METHOD_ANTIGRAVITY else decision.row_knn_method
        aucs.append(float(scores[selected_method].auc))
        zfs.append(int(scores[selected_method].zf))
    return {
        "method": method,
        "change_share": float(changed["dominance_rate"]),
        "dataset_mean_auc": float(np.mean(np.asarray(aucs, dtype=np.float64))),
        "zf_total": int(np.sum(np.asarray(zfs, dtype=np.int64))),
    }


def apply_exp481_gates_antigravity(summary: Mapping[str, object]) -> dict[str, object]:
    try:
        exp481 = summary["exp481"]
        changed = summary["changed"]
    except KeyError as error:
        raise ValueError("exp481 and changed summaries are required") from error
    if not isinstance(exp481, Mapping) or not isinstance(changed, Mapping):
        raise ValueError("summary sections must be mappings")
    thresholds = EXP481_THRESHOLDS_ANTIGRAVITY
    gates: dict[str, object] = {
        "zf_total": {
            "actual": int(exp481["zf_total"]),
            "required": f"> {int(thresholds['zf_total'])}",
            "pass": int(exp481["zf_total"]) > int(thresholds["zf_total"]),
        },
        "family_mean_zf": {
            "actual": float(exp481["family_mean_zf"]),
            "required": f"> {float(thresholds['family_mean_zf'])}",
            "pass": float(exp481["family_mean_zf"]) > float(thresholds["family_mean_zf"]),
        },
        "dataset_mean_auc": {
            "actual": float(exp481["dataset_mean_auc"]),
            "required": f">= {float(thresholds['dataset_mean_auc'])}",
            "pass": float(exp481["dataset_mean_auc"]) >= float(thresholds["dataset_mean_auc"]),
        },
        "family_mean_auc": {
            "actual": float(exp481["family_mean_auc"]),
            "required": f">= {float(thresholds['family_mean_auc'])}",
            "pass": float(exp481["family_mean_auc"]) >= float(thresholds["family_mean_auc"]),
        },
        "added_loss_0_05": {
            "actual": int(exp481["added_loss_0_05"]),
            "required": "= 0 for AUC gain <= -0.05",
            "pass": int(exp481["added_loss_0_05"]) == 0,
        },
        "added_loss_0_10": {
            "actual": int(exp481["added_loss_0_10"]),
            "required": "= 0 for AUC gain <= -0.10",
            "pass": int(exp481["added_loss_0_10"]) == 0,
        },
        "changed_count": {
            "actual": int(changed["count"]),
            "required": f">= {int(thresholds['min_changed'])}",
            "pass": int(changed["count"]) >= int(thresholds["min_changed"]),
        },
        "changed_zf_wins": {
            "actual": {"wins": int(changed["wins"]), "losses": int(changed["losses"])},
            "required": "wins > losses",
            "pass": int(changed["wins"]) > int(changed["losses"]),
        },
        "changed_zf_tie_rate": {
            "actual": float(changed["tie_rate"]),
            "required": f"<= {float(thresholds['max_changed_tie_rate'])}",
            "pass": float(changed["tie_rate"]) <= float(thresholds["max_changed_tie_rate"]),
        },
    }
    dominance_triggered = bool(
        int(changed["count"]) > 0
        and float(changed["dominance_rate"])
        >= float(thresholds["dominance_control_rate"])
    )
    gates["dominance_control"] = {
        "triggered": dominance_triggered,
        "available": changed.get("dominance_control") is not None,
        "result": changed.get("dominance_control"),
        "note": "control only; dominance is not an automatic failure",
    }
    gates["retire_seven_feature_space"] = int(changed["count"]) == 0
    gates["overall_pass"] = all(
        bool(value["pass"])
        for value in gates.values()
        if isinstance(value, dict) and "pass" in value
    )
    return gates


def _route_input_audit_antigravity(feature_keys=FEATURE_KEYS_ANTIGRAVITY) -> dict[str, object]:
    keys = tuple(feature_keys)
    name_keys = tuple(
        key for key in keys if any(token in key for token in ("dataset", "family", "name"))
    )
    score_keys = tuple(
        key for key in keys if any(token in key for token in ("auc", "zf", "score", "label"))
    )
    return {
        "route_input_keys": list(keys),
        "route_input_key_count": len(keys),
        "dataset_name_route_input_keys": list(name_keys),
        "held_out_score_route_input_keys": list(score_keys),
        "dataset_name_feature_count": len(name_keys),
        "held_out_score_feature_count": len(score_keys),
    }


def evaluate_nested_out_of_family_antigravity(
    rows: Sequence[EvidenceRowAntigravity],
    *,
    row_policy_kwargs: Mapping[str, object] | None = None,
    residual_policy_kwargs: Mapping[str, object] | None = None,
    inner_min_changed_families: int = INNER_MIN_CHANGED_FAMILIES_ANTIGRAVITY,
) -> dict[str, object]:
    ordered = tuple(sorted(rows, key=lambda row: row.dataset))
    residual_kwargs = {"k": K_ANTIGRAVITY}
    residual_kwargs.update(residual_policy_kwargs or {})
    decisions: list[MeasuredDecisionAntigravity] = []
    family_overlap_count = 0
    outer_family_seen_by_inner_count = 0
    inner_gates_by_outer: dict[str, list[dict[str, object]]] = {}
    for outer_family, outer_train, outer_test in iter_family_splits_antigravity(ordered):
        train_families = {row.source_family for row in outer_train}
        test_families = {row.source_family for row in outer_test}
        family_overlap_count += len(train_families & test_families)
        if outer_family in train_families:
            outer_family_seen_by_inner_count += 1
        gates = qualify_inner_methods_antigravity(
            outer_train,
            row_policy_kwargs=row_policy_kwargs,
            residual_policy_kwargs=residual_policy_kwargs,
            min_changed_families=inner_min_changed_families,
        )
        enabled = tuple(gate.method for gate in gates if gate.enabled)
        inner_gates_by_outer[outer_family] = [asdict(gate) for gate in gates]
        row_policy = _row_policy_antigravity(outer_train, row_policy_kwargs)
        residual_policy = StrictResidualPolicyAntigravity(**residual_kwargs).fit(outer_train)
        for row in outer_test:
            decisions.append(
                evaluate_outer_row_antigravity(
                    row, row_policy, residual_policy, enabled
                )
            )
    decisions.sort(key=lambda item: item.dataset)
    summaries: dict[str, object] = {
        "row_knn": _summary_antigravity(decisions, selected=False),
        "exp481": _summary_antigravity(decisions, selected=True),
        "changed": _changed_summary_antigravity(decisions),
    }
    changed = summaries["changed"]
    assert isinstance(changed, dict)
    changed["dominance_control"] = _dominance_control_antigravity(ordered, decisions, changed)
    leakage = {
        "family_overlap_count": family_overlap_count,
        "outer_family_seen_by_inner_count": outer_family_seen_by_inner_count,
        **_route_input_audit_antigravity(residual_kwargs.get("feature_keys", FEATURE_KEYS_ANTIGRAVITY)),
        "held_out_scores_read_after_routing": True,
    }
    return {
        "backend": EVALUATION_BACKEND_ANTIGRAVITY,
        "dataset_count": len(decisions),
        "family_count": len({item.held_family for item in decisions}),
        "summaries": summaries,
        "gates": apply_exp481_gates_antigravity(summaries),
        "leakage": leakage,
        "inner_gates_by_outer_family": inner_gates_by_outer,
        "decisions": [asdict(item) for item in decisions],
    }


__all__ = [
    "EVALUATION_BACKEND_ANTIGRAVITY",
    "MeasuredDecisionAntigravity",
    "aggregate_inner_gate_antigravity",
    "apply_exp481_gates_antigravity",
    "evaluate_nested_out_of_family_antigravity",
    "evaluate_outer_row_antigravity",
    "qualify_inner_methods_antigravity",
]
