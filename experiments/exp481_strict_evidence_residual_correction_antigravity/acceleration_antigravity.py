"""Deterministic Numba/MLX eligibility assessment for Exp481."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numba
import numpy as np

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)

from .config_antigravity import ACCELERATION_ASSESSMENT_PATH_ANTIGRAVITY
from .kernels_numba_antigravity import (
    nearest_distinct_family_indices_numba_antigravity,
    strict_statistics_numba_antigravity,
)
from .policy_antigravity import (
    StrictResidualPolicyAntigravity,
    nearest_distinct_family_indices_reference_antigravity,
    strict_statistics_reference_antigravity,
)


def _fixed_inputs_antigravity() -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    distances = np.linspace(0.0, 1.0, 512, dtype=np.float64)[::-1].copy()
    family_codes = np.arange(512, dtype=np.int64) % 128
    families = tuple(f"F{int(code)}" for code in family_codes)
    zf = (np.arange(31, dtype=np.int64) % 3) - 1
    auc = np.linspace(0.001, 0.031, 31, dtype=np.float64)
    return distances, families, family_codes, zf, auc


def _assess_parity_antigravity() -> dict[str, object]:
    distances, families, codes, zf, auc = _fixed_inputs_antigravity()
    reference_indices = nearest_distinct_family_indices_reference_antigravity(
        distances, families, 31
    )
    numba_indices = nearest_distinct_family_indices_numba_antigravity(
        distances, codes, 31
    )
    reference = strict_statistics_reference_antigravity(zf, auc, 0.20)
    candidate = strict_statistics_numba_antigravity(zf, auc, 0.20)
    reference_tuple = (
        reference.passed, reference.zf_wins, reference.zf_losses,
        reference.zf_ties, reference.min_auc_gain, reference.q20_auc_gain,
    )
    numeric_error = max(
        abs(float(candidate[4]) - float(reference_tuple[4])),
        abs(float(candidate[5]) - float(reference_tuple[5])),
    )
    policy_rows = tuple(
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
    reference_decision = StrictResidualPolicyAntigravity(
        k=3, backend="numpy_reference"
    ).fit(policy_rows).route(query, row_knn_method="minirocket_local_gap")
    numba_decision = StrictResidualPolicyAntigravity(
        k=3, backend="numba_cpu"
    ).fit(policy_rows).route(query, row_knn_method="minirocket_local_gap")
    decision_parity = numba_decision == reference_decision
    finite = bool(np.isfinite(np.asarray(candidate[4:], dtype=np.float64)).all())
    return {
        "candidate_index_parity": bool(np.array_equal(reference_indices, numba_indices)),
        "statistics_parity": bool(candidate == reference_tuple),
        "decision_parity": bool(decision_parity),
        "numerical_agreement": bool(numeric_error <= 1e-12),
        "finite_outputs": finite,
        "max_abs_error": float(numeric_error),
    }


def _benchmark_complete_stage_antigravity(repeats: int = 50) -> dict[str, float]:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    distances, families, codes, zf, auc = _fixed_inputs_antigravity()
    start = perf_counter()
    for _ in range(repeats):
        nearest_distinct_family_indices_reference_antigravity(distances, families, 31)
        strict_statistics_reference_antigravity(zf, auc, 0.20)
    numpy_seconds = perf_counter() - start

    warmup_start = perf_counter()
    # Array conversion and first JIT compilation are deliberately included.
    nearest_distinct_family_indices_numba_antigravity(
        np.asarray(distances), np.asarray(codes), 31
    )
    strict_statistics_numba_antigravity(np.asarray(zf), np.asarray(auc), 0.20)
    warmup_seconds = perf_counter() - warmup_start
    execution_start = perf_counter()
    for _ in range(repeats):
        nearest_distinct_family_indices_numba_antigravity(
            np.asarray(distances), np.asarray(codes), 31
        )
        strict_statistics_numba_antigravity(np.asarray(zf), np.asarray(auc), 0.20)
    execution_seconds = perf_counter() - execution_start
    total = warmup_seconds + execution_seconds
    return {
        "numpy_seconds": float(numpy_seconds),
        "numba_warmup_seconds": float(warmup_seconds),
        "numba_execution_seconds": float(execution_seconds),
        "numba_total_seconds": float(total),
        "numba_speedup_including_warmup": float(numpy_seconds / total if total else 0.0),
    }


def assess_acceleration_antigravity(
    *, output_path: Path | None = None, repeats: int = 50
) -> dict[str, object]:
    parity = _assess_parity_antigravity()
    timing = _benchmark_complete_stage_antigravity(repeats=repeats)
    parity_pass = all(
        bool(parity[key])
        for key in (
            "candidate_index_parity", "statistics_parity", "decision_parity",
            "numerical_agreement", "finite_outputs",
        )
    )
    speed_pass = float(timing["numba_speedup_including_warmup"]) > 1.0
    if not parity_pass:
        selected_backend = "numpy_reference"
        fallback_reason = "numba_parity_failed"
    elif not speed_pass:
        selected_backend = "numpy_reference"
        fallback_reason = "numba_no_full_stage_speed_benefit"
    else:
        selected_backend = "numba_cpu"
        fallback_reason = "mlx_not_exact_for_sequential_family_selection"
    result: dict[str, object] = {
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        **parity,
        **timing,
        "numba_speed_gate_pass": speed_pass,
        "mlx_eligible": False,
        "mlx_candidate_index_parity": False,
        "mlx_speedup_including_transfer": 0.0,
        "selected_backend": selected_backend,
        "fallback_reason": fallback_reason,
    }
    target = output_path
    if target is not None:
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_default_assessment_antigravity() -> dict[str, object]:
    return assess_acceleration_antigravity(
        output_path=ACCELERATION_ASSESSMENT_PATH_ANTIGRAVITY
    )


__all__ = ["assess_acceleration_antigravity", "write_default_assessment_antigravity"]
