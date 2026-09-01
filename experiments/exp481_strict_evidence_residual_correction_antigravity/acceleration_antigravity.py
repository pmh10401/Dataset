"""Deterministic Numba/MLX eligibility assessment for Exp481."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
import platform
from time import perf_counter

import numba
import numpy as np

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)

from .config_antigravity import ACCELERATION_ASSESSMENT_PATH_ANTIGRAVITY
from .evaluation_antigravity import evaluate_nested_out_of_family_antigravity
from .kernels_numba_antigravity import (
    nearest_distinct_family_indices_numba_antigravity,
    strict_statistics_numba_antigravity,
)
from .policy_antigravity import (
    nearest_distinct_family_indices_reference_antigravity,
    strict_statistics_reference_antigravity,
)


def _fixed_inputs_antigravity(
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    distances = np.linspace(0.0, 1.0, 512, dtype=np.float64)[::-1].copy()
    family_codes = np.arange(512, dtype=np.int64) % 128
    families = tuple(f"F{int(code)}" for code in family_codes)
    zf = (np.arange(31, dtype=np.int64) % 3) - 1
    auc = np.linspace(0.001, 0.031, 31, dtype=np.float64)
    return distances, families, family_codes, zf, auc


def _fixed_evidence_rows_antigravity() -> tuple[EvidenceRowAntigravity, ...]:
    return tuple(
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
        for index in range(5)
    )


def _nested_stage_antigravity(backend: str) -> dict[str, object]:
    return evaluate_nested_out_of_family_antigravity(
        _fixed_evidence_rows_antigravity(),
        row_policy_kwargs={"k": 1, "min_support": 1},
        residual_policy_kwargs={"k": 3, "backend": backend},
        inner_min_changed_families=1,
    )


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
        reference.passed,
        reference.zf_wins,
        reference.zf_losses,
        reference.zf_ties,
        reference.min_auc_gain,
        reference.q20_auc_gain,
    )
    numeric_error = max(
        abs(float(candidate[4]) - float(reference_tuple[4])),
        abs(float(candidate[5]) - float(reference_tuple[5])),
    )
    reference_nested = _nested_stage_antigravity("numpy_reference")
    numba_nested = _nested_stage_antigravity("numba_cpu")
    decision_parity = bool(
        numba_nested["decisions"] == reference_nested["decisions"]
        and numba_nested["summaries"] == reference_nested["summaries"]
        and numba_nested["gates"] == reference_nested["gates"]
        and numba_nested["leakage"] == reference_nested["leakage"]
    )
    return {
        "candidate_index_parity": bool(
            np.array_equal(reference_indices, numba_indices)
        ),
        "statistics_parity": bool(candidate == reference_tuple),
        "decision_parity": decision_parity,
        "numerical_agreement": bool(numeric_error <= 1e-12),
        "finite_outputs": bool(
            np.isfinite(np.asarray(candidate[4:], dtype=np.float64)).all()
        ),
        "max_abs_error": float(numeric_error),
    }


def _benchmark_complete_stage_antigravity(repeats: int = 1) -> dict[str, float]:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    start = perf_counter()
    for _ in range(repeats):
        _nested_stage_antigravity("numpy_reference")
    numpy_seconds = perf_counter() - start

    cold_start = perf_counter()
    for _ in range(repeats):
        _nested_stage_antigravity("numba_cpu")
    cold_total = perf_counter() - cold_start
    hot_start = perf_counter()
    _nested_stage_antigravity("numba_cpu")
    hot_seconds = perf_counter() - hot_start
    return {
        "numpy_seconds": float(numpy_seconds),
        "numba_warmup_seconds": float(cold_total),
        "numba_execution_seconds": float(hot_seconds),
        "numba_total_seconds": float(cold_total),
        "numba_speedup_including_warmup": float(
            numpy_seconds / cold_total if cold_total else 0.0
        ),
        "benchmark_dataset_count": 5,
        "benchmark_family_count": 5,
        "benchmark_residual_k": 3,
        "benchmark_inner_min_changed_families": 1,
        "exact_frozen_stage_benchmarked": False,
        "numba_cold_jit_verified": False,
    }


def _assess_mlx_antigravity() -> dict[str, object]:
    try:
        available = find_spec("mlx.core") is not None
    except ModuleNotFoundError:
        available = False
    try:
        mlx_version: str | None = version("mlx")
    except PackageNotFoundError:
        mlx_version = None
    try:
        mlx_metal_version: str | None = version("mlx-metal")
    except PackageNotFoundError:
        mlx_metal_version = None
    return {
        "mlx_available": available,
        "mlx_eligible": False,
        "mlx_version": mlx_version,
        "mlx_metal_version": mlx_metal_version,
        "mlx_device": platform.machine(),
        "mlx_candidate_index_parity": None,
        "mlx_finite_outputs": None,
        "mlx_speedup_including_transfer": None,
        "mlx_fallback_reason": (
            "mlx_unavailable_in_frozen_runtime"
            if not available
            else "mlx_exact_stage_not_implemented"
        ),
    }


def assess_acceleration_antigravity(
    *, output_path: Path | None = None, repeats: int = 1
) -> dict[str, object]:
    timing = _benchmark_complete_stage_antigravity(repeats=repeats)
    parity = _assess_parity_antigravity()
    mlx = _assess_mlx_antigravity()
    parity_pass = all(
        bool(parity[key])
        for key in (
            "candidate_index_parity",
            "statistics_parity",
            "decision_parity",
            "numerical_agreement",
            "finite_outputs",
        )
    )
    exact_stage = bool(timing.get("exact_frozen_stage_benchmarked", False))
    cold_jit = bool(timing.get("numba_cold_jit_verified", False))
    speed_pass = float(timing["numba_speedup_including_warmup"]) > 1.0
    if not parity_pass:
        selected_backend = "numpy_reference"
        fallback_reason = "numba_parity_failed"
    elif not exact_stage:
        selected_backend = "numpy_reference"
        fallback_reason = "numba_exact_frozen_stage_not_benchmarked"
    elif not cold_jit:
        selected_backend = "numpy_reference"
        fallback_reason = "numba_cold_jit_not_verified"
    elif not speed_pass:
        selected_backend = "numpy_reference"
        fallback_reason = "numba_no_full_stage_speed_benefit"
    else:
        selected_backend = "numba_cpu"
        fallback_reason = str(mlx["mlx_fallback_reason"])
    result: dict[str, object] = {
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        **parity,
        **timing,
        "benchmark_scope": "diagnostic_nested_smoke_not_official_1117",
        "numba_speed_gate_pass": speed_pass,
        **mlx,
        "selected_backend": selected_backend,
        "fallback_reason": fallback_reason,
    }
    if output_path is not None:
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_default_assessment_antigravity() -> dict[str, object]:
    return assess_acceleration_antigravity(
        output_path=ACCELERATION_ASSESSMENT_PATH_ANTIGRAVITY
    )


__all__ = ["assess_acceleration_antigravity", "write_default_assessment_antigravity"]
