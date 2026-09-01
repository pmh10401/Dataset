"""Reference CPU NumPy path for Exp481 strict residual policy.

Task 2 freezes the exact selection rules on this simpler NumPy path.
Task 4 decides NumPy versus Numba after parity and full-stage speed measurement.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    EvidenceRowAntigravity,
)

from .config_antigravity import (
    ALTERNATIVE_METHODS_ANTIGRAVITY,
    AUC_QUANTILE_ANTIGRAVITY,
    BASELINE_METHOD_ANTIGRAVITY,
    FEATURE_KEYS_ANTIGRAVITY,
    K_ANTIGRAVITY,
    METHODS_ANTIGRAVITY,
)
from .contracts_antigravity import (
    ReferenceStatisticsAntigravity,
    ResidualDecisionAntigravity,
    StrictCandidateAssessmentAntigravity,
)


def nearest_distinct_family_indices_reference_antigravity(
    distances: np.ndarray, families: Sequence[str], k: int
) -> np.ndarray:
    """Keep the first row of each family under a stable ascending distance order."""
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(families):
        raise ValueError("distances and families must be aligned one-dimensional values")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    chosen: list[int] = []
    seen: set[str] = set()
    for raw_index in np.argsort(values, kind="stable"):
        index = int(raw_index)
        family = str(families[index])
        if family not in seen:
            seen.add(family)
            chosen.append(index)
            if len(chosen) == k:
                break
    return np.asarray(chosen, dtype=np.int64)


def strict_statistics_reference_antigravity(
    zf_gains: Sequence[int],
    auc_gains: Sequence[float],
    q: float = AUC_QUANTILE_ANTIGRAVITY,
) -> ReferenceStatisticsAntigravity:
    """Reference linear-NumPy strict local statistics. ZF ties are not wins."""
    zf = tuple(int(value) for value in zf_gains)
    auc = tuple(float(value) for value in auc_gains)
    if len(zf) != len(auc) or not zf:
        raise ValueError("zf_gains and auc_gains must be aligned and non-empty")
    q_value = float(q)
    if not np.isfinite(q_value) or not 0.0 <= q_value <= 1.0:
        raise ValueError("q must be a finite value in [0, 1]")
    wins = int(sum(value > 0 for value in zf))
    losses = int(sum(value < 0 for value in zf))
    ties = int(len(zf) - wins - losses)
    minimum = float(min(auc))
    q20 = float(np.quantile(np.asarray(auc, dtype=np.float64), q_value))
    passed = bool(wins > losses and minimum >= 0.0 and q20 > 0.0)
    return ReferenceStatisticsAntigravity(
        passed=passed,
        zf_wins=wins,
        zf_losses=losses,
        zf_ties=ties,
        min_auc_gain=minimum,
        q20_auc_gain=q20,
    )


def _transformed_tuple_antigravity(features: Sequence[float]) -> np.ndarray:
    raw = np.asarray(tuple(float(value) for value in features), dtype=np.float64)
    if raw.shape != (len(FEATURE_KEYS_ANTIGRAVITY),):
        raise ValueError("features must contain the seven frozen values")
    transformed = raw.copy()
    if np.isfinite(transformed[0]):
        transformed[0] = np.log1p(max(0.0, float(transformed[0])))
    if np.isfinite(transformed[1]):
        transformed[1] = np.log1p(max(0.0, float(transformed[1])))
    return transformed


def _require_exact_feature_keys_antigravity(features: Mapping[str, float]) -> None:
    if not isinstance(features, Mapping):
        raise ValueError("features must be a mapping")
    if set(features) != set(FEATURE_KEYS_ANTIGRAVITY):
        raise ValueError("query must contain exactly the seven frozen feature keys")


def _enabled_set_antigravity(
    enabled_methods: Iterable[str] | None,
) -> frozenset[str] | None:
    if enabled_methods is None:
        return None
    if isinstance(enabled_methods, (str, bytes)):
        raise ValueError("enabled_methods must be an iterable of method names")
    enabled = tuple(enabled_methods)
    if any(
        type(method) is not str or method not in ALTERNATIVE_METHODS_ANTIGRAVITY
        for method in enabled
    ):
        raise ValueError("enabled_methods must contain frozen alternatives")
    return frozenset(enabled)


def _unique_best_antigravity(
    assessments: tuple[StrictCandidateAssessmentAntigravity, ...],
) -> tuple[StrictCandidateAssessmentAntigravity | None, str]:
    if not assessments:
        return None, "strict_local_gate_failed"
    ranks = tuple(
        (item.zf_wins - item.zf_losses, item.min_auc_gain, item.q20_auc_gain)
        for item in assessments
    )
    best_rank = max(ranks)
    winners = tuple(
        item
        for item, rank in zip(assessments, ranks, strict=True)
        if rank == best_rank
    )
    if len(winners) != 1:
        return None, "candidate_rank_tie"
    return winners[0], "strict_candidate_accept"


class StrictResidualPolicyAntigravity:
    """Preserve row-KNN unless MiniRocket residual evidence uniquely passes."""

    def __init__(
        self, *, k: int = K_ANTIGRAVITY, backend: str = "numpy_reference"
    ) -> None:
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        if backend not in ("numpy_reference", "numba_cpu"):
            raise ValueError("backend must be numpy_reference or numba_cpu")
        self.k = k
        self.backend = backend
        self._fitted = False

    def fit(
        self, rows: Iterable[EvidenceRowAntigravity]
    ) -> "StrictResidualPolicyAntigravity":
        fitted = tuple(rows)
        if not fitted or any(type(row) is not EvidenceRowAntigravity for row in fitted):
            raise ValueError("fit requires strict EvidenceRowAntigravity rows")
        matrix = np.stack(
            [_transformed_tuple_antigravity(row.features) for row in fitted]
        )
        finite = np.isfinite(matrix)
        medians = np.asarray(
            [
                np.median(matrix[finite[:, index], index])
                if finite[:, index].any()
                else 0.0
                for index in range(matrix.shape[1])
            ],
            dtype=np.float64,
        )
        matrix = np.where(finite, matrix, medians[None, :])
        scales = matrix.std(axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        self._medians = medians
        self._mean = matrix.mean(axis=0)
        self._scale = scales
        self._matrix = (matrix - self._mean[None, :]) / self._scale[None, :]
        self._families = tuple(row.source_family for row in fitted)
        family_codes = {
            family: code for code, family in enumerate(dict.fromkeys(self._families))
        }
        self._family_codes = np.asarray(
            [family_codes[family] for family in self._families], dtype=np.int64
        )
        alternatives = []
        baseline_index = 0
        for method_index, method in enumerate(ALTERNATIVE_METHODS_ANTIGRAVITY, start=1):
            auc_gains = np.asarray(
                [
                    row.scores[method_index].auc - row.scores[baseline_index].auc
                    for row in fitted
                ],
                dtype=np.float64,
            )
            zf_gains = np.asarray(
                [
                    row.scores[method_index].zf - row.scores[baseline_index].zf
                    for row in fitted
                ],
                dtype=np.int64,
            )
            alternatives.append((method, auc_gains, zf_gains))
        self._alternatives = tuple(alternatives)
        self._fitted = True
        return self

    def _assess_all_antigravity(
        self, features: Mapping[str, float]
    ) -> tuple[StrictCandidateAssessmentAntigravity, ...]:
        raw = _transformed_tuple_antigravity(
            tuple(float(features[key]) for key in FEATURE_KEYS_ANTIGRAVITY)
        )
        raw = np.where(np.isfinite(raw), raw, self._medians)
        query = (raw - self._mean) / self._scale
        distances = np.linalg.norm(self._matrix - query[None, :], axis=1)
        if self.backend == "numba_cpu":
            from .kernels_numba_antigravity import (
                nearest_distinct_family_indices_numba_antigravity,
            )

            neighbours = nearest_distinct_family_indices_numba_antigravity(
                distances, self._family_codes, self.k
            )
        else:
            neighbours = nearest_distinct_family_indices_reference_antigravity(
                distances, self._families, self.k
            )
        neighbour_families = tuple(self._families[int(index)] for index in neighbours)
        assessments = []
        for method, auc_gains, zf_gains in self._alternatives:
            if self.backend == "numba_cpu":
                from .kernels_numba_antigravity import strict_statistics_numba_antigravity

                raw_stats = strict_statistics_numba_antigravity(
                    zf_gains[neighbours], auc_gains[neighbours], AUC_QUANTILE_ANTIGRAVITY
                )
                stats = ReferenceStatisticsAntigravity(
                    passed=raw_stats[0], zf_wins=raw_stats[1],
                    zf_losses=raw_stats[2], zf_ties=raw_stats[3],
                    min_auc_gain=raw_stats[4], q20_auc_gain=raw_stats[5],
                )
            else:
                stats = strict_statistics_reference_antigravity(
                    zf_gains[neighbours], auc_gains[neighbours]
                )
            assessments.append(
                StrictCandidateAssessmentAntigravity(
                    method=method,
                    passed=bool(stats.passed),
                    support=int(len(neighbours)),
                    zf_wins=int(stats.zf_wins),
                    zf_losses=int(stats.zf_losses),
                    zf_ties=int(stats.zf_ties),
                    min_auc_gain=float(stats.min_auc_gain),
                    q20_auc_gain=float(stats.q20_auc_gain),
                    neighbour_families=neighbour_families,
                )
            )
        return tuple(assessments)

    def route(
        self,
        features: Mapping[str, float],
        *,
        row_knn_method: str,
        enabled_methods: Iterable[str] | None = None,
    ) -> ResidualDecisionAntigravity:
        if not self._fitted:
            raise RuntimeError("fit must be called before route")
        if type(row_knn_method) is not str or row_knn_method not in METHODS_ANTIGRAVITY:
            raise ValueError("row_knn_method must be one of the four frozen methods")
        _require_exact_feature_keys_antigravity(features)
        enabled = _enabled_set_antigravity(enabled_methods)
        if row_knn_method != BASELINE_METHOD_ANTIGRAVITY:
            return ResidualDecisionAntigravity(
                method=row_knn_method,
                changed=False,
                reason="row_knn_alternative_preserved",
                selected_candidate=None,
                assessments=(),
            )
        assessments = self._assess_all_antigravity(features)
        if enabled is not None and not enabled:
            return ResidualDecisionAntigravity(
                method=BASELINE_METHOD_ANTIGRAVITY,
                changed=False,
                reason="inner_gate_disabled",
                selected_candidate=None,
                assessments=assessments,
            )
        eligible = tuple(
            item
            for item in assessments
            if item.passed and (enabled is None or item.method in enabled)
        )
        winner, reason = _unique_best_antigravity(eligible)
        if winner is None:
            return ResidualDecisionAntigravity(
                method=BASELINE_METHOD_ANTIGRAVITY,
                changed=False,
                reason=reason,
                selected_candidate=None,
                assessments=assessments,
            )
        return ResidualDecisionAntigravity(
            method=winner.method,
            changed=True,
            reason="strict_candidate_accept",
            selected_candidate=winner.method,
            assessments=assessments,
        )


__all__ = [
    "StrictResidualPolicyAntigravity",
    "nearest_distinct_family_indices_reference_antigravity",
    "strict_statistics_reference_antigravity",
]
