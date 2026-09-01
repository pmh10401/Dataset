"""Task 4 parity tests for the optional Numba kernels."""

from __future__ import annotations

import numpy as np

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.kernels_numba_antigravity import (
    nearest_distinct_family_indices_numba_antigravity,
    strict_statistics_numba_antigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.policy_antigravity import (
    StrictResidualPolicyAntigravity,
    nearest_distinct_family_indices_reference_antigravity,
    strict_statistics_reference_antigravity,
)


def test_numba_indices_and_statistics_match_reference_antigravity() -> None:
    distances = np.asarray((0.4, 0.1, 0.1, 0.2, 0.3), dtype=np.float64)
    families = ("B", "A", "A", "C", "D")
    family_codes = np.asarray((1, 0, 0, 2, 3), dtype=np.int64)
    expected = nearest_distinct_family_indices_reference_antigravity(
        distances, families, 3
    )
    actual = nearest_distinct_family_indices_numba_antigravity(
        distances, family_codes, 3
    )
    assert actual.tolist() == expected.tolist()

    zf = np.asarray((1, 0, -1, 2), dtype=np.int64)
    auc = np.asarray((0.1, 0.2, 0.3, 0.4), dtype=np.float64)
    reference = strict_statistics_reference_antigravity(zf, auc, 0.20)
    candidate = strict_statistics_numba_antigravity(zf, auc, 0.20)
    assert candidate[:4] == (
        reference.passed,
        reference.zf_wins,
        reference.zf_losses,
        reference.zf_ties,
    )
    assert candidate[4:] == (reference.min_auc_gain, reference.q20_auc_gain)


def test_numba_ties_are_not_wins_and_empty_inputs_fail_antigravity() -> None:
    result = strict_statistics_numba_antigravity(
        np.asarray((0, 0, -1), dtype=np.int64),
        np.asarray((0.2, 0.2, 0.2), dtype=np.float64),
        0.20,
    )
    assert result[:4] == (False, 0, 1, 2)

    for distances, codes in (
        (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64)),
        (np.asarray((0.1,), dtype=np.float64), np.asarray((-1,), dtype=np.int64)),
    ):
        try:
            nearest_distinct_family_indices_numba_antigravity(distances, codes, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Numba input must fail")


def test_numba_policy_decision_matches_reference_antigravity() -> None:
    rows = tuple(
        EvidenceRowAntigravity(
            dataset=f"F{index}_normal_1",
            source_family=f"F{index}",
            proposal="minirocket_local_gap",
            features=(float(index + 1), 20.0, 0.4, 0.1, 3.0, 1.0, 0.2),
            scores=(
                CandidateMetricAntigravity("minirocket_local_gap", 0.70, 2),
                CandidateMetricAntigravity("multirocket_local_gap", 0.71, 3),
                CandidateMetricAntigravity("hydra_local_gap", 0.69, 1),
                CandidateMetricAntigravity("hydra_multirocket_local_gap", 0.70, 2),
            ),
        )
        for index in range(4)
    )
    query = {
        "width": 1.5, "n_train": 20.0, "spectral_entropy": 0.4,
        "skewness": 0.1, "kurtosis": 3.0, "diff_std_ratio": 1.0,
        "zero_crossing_rate": 0.2,
    }
    reference = StrictResidualPolicyAntigravity(k=3).fit(rows).route(
        query, row_knn_method="minirocket_local_gap"
    )
    candidate = StrictResidualPolicyAntigravity(k=3, backend="numba_cpu").fit(rows).route(
        query, row_knn_method="minirocket_local_gap"
    )
    assert candidate == reference
