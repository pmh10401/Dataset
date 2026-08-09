"""Fail-closed Policy A reproduction gate for the Exp443 control."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import PolicyDatasetResultAntigravity
from .routing_antigravity import route_policy_antigravity


POLICY_A_ANTIGRAVITY = "linear_edge_nearest_control"
CONTROL_TOLERANCE_ANTIGRAVITY = 1e-6
CONTROL_FAILURE_ANTIGRAVITY = "control_reproduction_failed"
_EXPECTED_SCORE_PAIRS_ANTIGRAVITY = frozenset(
    (kernel, seed)
    for kernel in FROZEN_CONFIG.kernel_requests
    for seed in FROZEN_CONFIG.seeds
)


def _environment_identity_antigravity() -> Mapping[str, object]:
    return MappingProxyType({
        "python_version": FROZEN_CONFIG.python_version,
        "aeon_version": FROZEN_CONFIG.aeon_version,
        "numpy_version": FROZEN_CONFIG.numpy_version,
        "scikit_learn_version": FROZEN_CONFIG.scikit_learn_version,
        "environment_lock_sha256": FROZEN_CONFIG.environment_lock_sha256,
        "outer_workers": FROZEN_CONFIG.outer_workers,
        "inner_threads": FROZEN_CONFIG.inner_threads,
        "acceleration_decision": FROZEN_CONFIG.acceleration_decision,
        "acceleration_reason": "official_aeon_minirocket_cpu_transformer_no_mlx_rewrite",
    })


@dataclass(frozen=True)
class ControlGateResultAntigravity:
    """Frozen, label-free evidence for the Policy A control decision."""

    status: str
    passed: bool
    reason: str | None
    score_evidence: tuple[Mapping[str, object], ...]
    routing_identity: Mapping[str, object]
    environment_identity: Mapping[str, object]
    label_access_count: int
    max_abs_difference: float = 0.0
    candidate_mismatch_count: int = 0

    def __post_init__(self) -> None:
        if self.status not in {"control_reproduction_passed", CONTROL_FAILURE_ANTIGRAVITY}:
            raise ValueError("control gate status is invalid")
        if type(self.passed) is not bool or self.passed != (self.status == "control_reproduction_passed"):
            raise ValueError("control gate pass/status identity is invalid")
        if self.reason is not None and type(self.reason) is not str:
            raise ValueError("control gate reason is invalid")
        if type(self.label_access_count) is not int or self.label_access_count < 0:
            raise ValueError("control gate label-access count is invalid")
        if type(self.max_abs_difference) not in (int, float) or not np.isfinite(self.max_abs_difference) or self.max_abs_difference < 0:
            raise ValueError("control gate score difference is invalid")
        if type(self.candidate_mismatch_count) is not int or self.candidate_mismatch_count < 0:
            raise ValueError("control gate candidate mismatch count is invalid")
        evidence = tuple(self.score_evidence)
        if any(not isinstance(item, Mapping) for item in evidence):
            raise ValueError("control gate score evidence is invalid")
        object.__setattr__(self, "score_evidence", tuple(MappingProxyType(dict(item)) for item in evidence))
        if not isinstance(self.routing_identity, Mapping) or not isinstance(self.environment_identity, Mapping):
            raise ValueError("control gate identity evidence is invalid")
        object.__setattr__(self, "routing_identity", MappingProxyType(dict(self.routing_identity)))
        object.__setattr__(self, "environment_identity", MappingProxyType(dict(self.environment_identity)))


def _failed_result_antigravity(
    reasons: list[str],
    *,
    score_evidence: tuple[Mapping[str, object], ...] = (),
    routing_identity: Mapping[str, object] | None = None,
    label_access_count: int = 0,
    max_abs_difference: float = 0.0,
    candidate_mismatch_count: int = 0,
) -> ControlGateResultAntigravity:
    return ControlGateResultAntigravity(
        CONTROL_FAILURE_ANTIGRAVITY,
        False,
        "; ".join(reasons) if reasons else CONTROL_FAILURE_ANTIGRAVITY,
        score_evidence,
        {} if routing_identity is None else routing_identity,
        _environment_identity_antigravity(),
        label_access_count,
        max_abs_difference,
        candidate_mismatch_count,
    )


def _valid_score_array_antigravity(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and value.dtype == np.dtype("<f8")
        and value.ndim == 1
        and value.flags.c_contiguous
        and not value.flags.writeable
        and bool(np.isfinite(value).all())
    )


def _archive_candidates_antigravity(oof: np.ndarray, test: np.ndarray) -> tuple[int, ...]:
    from experiments.exp239_248_diverse_score_sources.calibration import empirical_candidates

    values = empirical_candidates(oof, test, FROZEN_CONFIG.alpha).indices
    return tuple(int(index) for index in values)


def _expected_routing_coordinates_antigravity(snapshot: object) -> tuple[
    frozenset[tuple[str, int]],
    frozenset[tuple[str, int]],
    frozenset[tuple[str, int]],
    frozenset[tuple[str, int]],
]:
    lanes = getattr(snapshot, "exp236_lanes")
    hard = frozenset(
        (dataset, index)
        for dataset, values in lanes.items()
        for index in values["hard"]
    )
    promotions = frozenset(
        (dataset, index)
        for dataset, values in lanes.items()
        for index in values["promotion"]
    )
    pre_exact = frozenset(hard | promotions)
    final_hard = frozenset(getattr(snapshot, "hard_coordinates_exp291"))
    removals = frozenset(pre_exact.difference(final_hard))
    return promotions, pre_exact, removals, final_hard


def compare_policy_a_to_archive_antigravity(
    snapshot: object,
    results: Mapping[str, PolicyDatasetResultAntigravity],
) -> ControlGateResultAntigravity:
    """Compare Policy A's score arrays, candidates, and complete routing output."""

    reasons: list[str] = []
    evidence: list[Mapping[str, object]] = []
    max_abs_difference = 0.0
    candidate_mismatch_count = 0
    try:
        affected = tuple(getattr(snapshot, "affected_datasets"))
        datasets = tuple(getattr(snapshot, "datasets"))
        if not affected or affected != tuple(sorted(set(affected))) or not set(affected).issubset(datasets):
            reasons.append("affected dataset authority is invalid")
        if not isinstance(results, Mapping) or set(results) != set(affected):
            reasons.append("Policy A result population does not equal affected authority")
        label_access_count = getattr(snapshot, "label_access_count", 0)
        if type(label_access_count) is not int or label_access_count != 0:
            reasons.append("label access was observed")
        else:
            label_access_count = 0
    except (AttributeError, TypeError, ValueError) as error:
        return _failed_result_antigravity([f"control input is invalid: {error}"])

    for dataset in affected:
        try:
            result = results[dataset]
            if type(result) is not PolicyDatasetResultAntigravity or result.policy != POLICY_A_ANTIGRAVITY or result.dataset != dataset or result.status != "scored":
                raise ValueError("Policy A result identity/status is invalid")
            if result.test_rows != getattr(snapshot, "test_row_count_by_dataset")[dataset]:
                raise ValueError("Policy A TEST row count disagrees with authority")
            observed_scores = {(score.kernel, score.seed): score for score in result.kernel_scores}
            if set(observed_scores) != _EXPECTED_SCORE_PAIRS_ANTIGRAVITY or len(observed_scores) != len(result.kernel_scores):
                raise ValueError("Policy A kernel/seed score identity is incomplete")
            arrays = getattr(snapshot, "cache_inventory")[dataset]["arrays"]
            for kernel in FROZEN_CONFIG.kernel_requests:
                archive_intersection: set[int] | None = None
                policy_intersection: set[int] | None = None
                for seed_index, seed in enumerate(FROZEN_CONFIG.seeds):
                    score = observed_scores[(kernel, seed)]
                    archive_oof = np.asarray(arrays[f"kernels_{kernel}__train_by_seed"][seed_index])
                    archive_test = np.asarray(arrays[f"kernels_{kernel}__test_by_seed"][seed_index])
                    policy_oof = score.oof_scores
                    policy_test = score.test_scores
                    train_shape = tuple(policy_oof.shape) if isinstance(policy_oof, np.ndarray) else None
                    test_shape = tuple(policy_test.shape) if isinstance(policy_test, np.ndarray) else None
                    archive_train_shape = tuple(archive_oof.shape)
                    archive_test_shape = tuple(archive_test.shape)
                    shape_equal = train_shape == archive_train_shape and test_shape == archive_test_shape
                    finite = (
                        _valid_score_array_antigravity(policy_oof)
                        and _valid_score_array_antigravity(policy_test)
                        and _valid_score_array_antigravity(archive_oof)
                        and _valid_score_array_antigravity(archive_test)
                    )
                    train_difference = float("nan")
                    test_difference = float("nan")
                    difference = float("nan")
                    if shape_equal and finite:
                        train_difference = float(np.max(np.abs(policy_oof.astype("<f8", copy=False) - archive_oof.astype("<f8", copy=False)))) if policy_oof.size else 0.0
                        test_difference = float(np.max(np.abs(policy_test.astype("<f8", copy=False) - archive_test.astype("<f8", copy=False)))) if policy_test.size else 0.0
                        difference = max(train_difference, test_difference)
                        max_abs_difference = max(max_abs_difference, difference)
                    archive_candidates = _archive_candidates_antigravity(archive_oof, archive_test) if finite else ()
                    candidate_equal = tuple(score.candidate_indices) == archive_candidates
                    if not shape_equal:
                        reasons.append(f"{dataset}/kernel{kernel}/seed{seed} score shape mismatch")
                    if not finite:
                        reasons.append(f"{dataset}/kernel{kernel}/seed{seed} score arrays are not finite")
                    if shape_equal and finite and difference > CONTROL_TOLERANCE_ANTIGRAVITY:
                        reasons.append(f"{dataset}/kernel{kernel}/seed{seed} max absolute difference exceeds 1e-6")
                    if not candidate_equal:
                        candidate_mismatch_count += 1
                        reasons.append(f"{dataset}/kernel{kernel}/seed{seed} candidate mismatch")
                    evidence.append({
                        "dataset": dataset,
                        "kernel": kernel,
                        "seed": seed,
                        "train_shape": train_shape,
                        "test_shape": test_shape,
                        "archive_train_shape": archive_train_shape,
                        "archive_test_shape": archive_test_shape,
                        "finite": finite,
                        "train_max_abs_difference": train_difference,
                        "test_max_abs_difference": test_difference,
                        "max_abs_difference": difference,
                        "candidate_equal": candidate_equal,
                        "archive_candidates": archive_candidates,
                        "policy_candidates": tuple(score.candidate_indices),
                    })
                    if archive_intersection is None:
                        archive_intersection = set(archive_candidates)
                        policy_intersection = set(score.candidate_indices)
                    else:
                        archive_intersection.intersection_update(archive_candidates)
                        policy_intersection.intersection_update(score.candidate_indices)
                expected_candidates = tuple(sorted(archive_intersection or set()))
                observed_candidates = tuple(sorted(policy_intersection or set()))
                frozen_candidates = tuple(getattr(snapshot, "exp249_candidate_maps")[dataset][kernel])
                if observed_candidates != expected_candidates or expected_candidates != frozen_candidates:
                    candidate_mismatch_count += 1
                    reasons.append(f"{dataset}/kernel{kernel} seed-intersection candidate mismatch")
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
            reasons.append(f"{dataset} control comparison failed: {error}")

    routing_identity: dict[str, object] = {
        "rows": 0,
        "promotions": 0,
        "pre_exact": 0,
        "exact_normal_removals": 0,
        "final_hard": 0,
        "promotions_coordinate_equal": False,
        "pre_exact_coordinate_equal": False,
        "exact_normal_removals_coordinate_equal": False,
        "exp291_coordinate_equal": False,
        "exp404_coordinate_equal": False,
        "recomputed_rows": 0,
        "imported_rows": 0,
    }
    try:
        prediction = route_policy_antigravity(snapshot, results, POLICY_A_ANTIGRAVITY)
        expected_promotions, expected_pre, expected_removals, expected_final = _expected_routing_coordinates_antigravity(snapshot)
        source_counts = {
            row["provenance"]["candidate_source"]
            for row in prediction.rows
        }
        recomputed_rows = sum(row["provenance"]["candidate_source"] == "recomputed_policy" for row in prediction.rows)
        imported_rows = sum(row["provenance"]["candidate_source"] == "exp249_frozen_seed_intersection" for row in prediction.rows)
        routing_identity = {
            "rows": len(prediction.rows),
            "promotions": len(prediction.promotions),
            "pre_exact": len(prediction.pre_exact_coordinates),
            "exact_normal_removals": len(prediction.exact_normal_removals),
            "final_hard": len(prediction.final_hard_coordinates),
            "promotions_coordinate_equal": prediction.promotions == expected_promotions,
            "pre_exact_coordinate_equal": prediction.pre_exact_coordinates == expected_pre,
            "exact_normal_removals_coordinate_equal": prediction.exact_normal_removals == expected_removals,
            "exp291_coordinate_equal": prediction.final_hard_coordinates == getattr(snapshot, "hard_coordinates_exp291"),
            "exp404_coordinate_equal": prediction.final_hard_coordinates == getattr(snapshot, "hard_coordinates_exp404"),
            "recomputed_rows": recomputed_rows,
            "imported_rows": imported_rows,
            "candidate_sources": tuple(sorted(source_counts)),
        }
        if tuple(row["dataset"] for row in prediction.rows) != datasets:
            reasons.append("routing dataset identity is not the complete sorted population")
        if not all(routing_identity[name] for name in ("promotions_coordinate_equal", "pre_exact_coordinate_equal", "exact_normal_removals_coordinate_equal", "exp291_coordinate_equal", "exp404_coordinate_equal")):
            reasons.append("routing coordinate identity disagrees with frozen authorities")
        if routing_identity["rows"] != len(datasets) or routing_identity["promotions"] != len(expected_promotions) or routing_identity["pre_exact"] != len(expected_pre) or routing_identity["exact_normal_removals"] != len(expected_removals) or routing_identity["final_hard"] != len(expected_final):
            reasons.append("routing population counts disagree with frozen authorities")
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        reasons.append(f"routing control failed: {error}")

    if reasons:
        return _failed_result_antigravity(
            reasons,
            score_evidence=tuple(evidence),
            routing_identity=routing_identity,
            label_access_count=label_access_count,
            max_abs_difference=max_abs_difference,
            candidate_mismatch_count=candidate_mismatch_count,
        )
    return ControlGateResultAntigravity(
        "control_reproduction_passed",
        True,
        None,
        tuple(evidence),
        routing_identity,
        _environment_identity_antigravity(),
        label_access_count,
        max_abs_difference,
        candidate_mismatch_count,
    )


def require_control_gate_antigravity(result: ControlGateResultAntigravity) -> None:
    """Block all downstream policy work unless the complete control passed."""

    if type(result) is not ControlGateResultAntigravity or not result.passed or result.status != "control_reproduction_passed" or result.label_access_count != 0:
        reason = getattr(result, "reason", None)
        raise ValueError(f"{CONTROL_FAILURE_ANTIGRAVITY}: {reason or 'control gate did not pass'}")


__all__ = [
    "CONTROL_FAILURE_ANTIGRAVITY",
    "CONTROL_TOLERANCE_ANTIGRAVITY",
    "ControlGateResultAntigravity",
    "compare_policy_a_to_archive_antigravity",
    "require_control_gate_antigravity",
]
