"""Acceleration gates for the two Exp484 distance variants."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from math import isfinite
import subprocess
import sys
from time import perf_counter
from typing import Sequence

import numba
import numpy as np
import sklearn

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    EvidenceRowAntigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.config_antigravity import (
    FEATURE_KEYS_ANTIGRAVITY,
    K_ANTIGRAVITY,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.kernels_numba_antigravity import (
    distance_values_numba_antigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.policy_antigravity import (
    StrictResidualPolicyAntigravity,
    nearest_distinct_family_indices_reference_antigravity,
)


VARIANTS_ANTIGRAVITY = (
    "robust_manhattan",
    "robust_shrinkage_mahalanobis",
)


def choose_backend_antigravity(
    *,
    numpy_seconds: float,
    candidate_seconds: float,
    finite_outputs: bool,
    numerical_parity: bool,
    candidate_index_parity: bool,
    decision_parity: bool,
) -> dict[str, object]:
    reference = float(numpy_seconds)
    candidate = float(candidate_seconds)
    if not isfinite(reference) or not isfinite(candidate) or min(reference, candidate) <= 0:
        raise ValueError("benchmark times must be finite and positive")
    speedup = reference / candidate
    if not all(
        (finite_outputs, numerical_parity, candidate_index_parity, decision_parity)
    ):
        selected, reason = "numpy_reference", "parity_gate_failed"
    elif speedup <= 1.0:
        selected, reason = "numpy_reference", "numba_not_faster"
    else:
        selected, reason = "numba_cpu", None
    return {
        "selected_backend": selected,
        "speedup": speedup,
        "fallback_reason": reason,
    }


def choose_mlx_backend_antigravity(
    *,
    cpu_seconds: float,
    mlx_seconds: float,
    finite_outputs: bool,
    numerical_parity: bool,
    candidate_index_parity: bool,
    decision_parity: bool,
) -> dict[str, object]:
    cpu = float(cpu_seconds)
    mlx = float(mlx_seconds)
    if not isfinite(cpu) or not isfinite(mlx) or min(cpu, mlx) <= 0:
        raise ValueError("benchmark times must be finite and positive")
    speedup = cpu / mlx
    parity = all(
        (finite_outputs, numerical_parity, candidate_index_parity, decision_parity)
    )
    selected = parity and speedup >= 1.1
    reason = None if selected else (
        "mlx_speed_gate_failed" if parity else "mlx_parity_gate_failed"
    )
    return {
        "selected": selected,
        "speedup": speedup,
        "speed_gate_pass": speedup >= 1.1,
        "fallback_reason": reason,
    }


def _installed_version_antigravity(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _numpy_distances_antigravity(
    matrix: np.ndarray,
    query: np.ndarray,
    *,
    variant: str,
    precision: np.ndarray | None,
) -> np.ndarray:
    delta = matrix - query[None, :]
    if variant == "robust_manhattan":
        return np.sum(np.abs(delta), axis=1)
    squared = np.einsum("nd,df,nf->n", delta, precision, delta)
    return np.sqrt(np.maximum(0.0, squared))


def _feature_mapping_antigravity(row: EvidenceRowAntigravity) -> dict[str, float]:
    return dict(zip(FEATURE_KEYS_ANTIGRAVITY, row.features, strict=True))


def _assess_mlx_antigravity(
    policies: dict[str, StrictResidualPolicyAntigravity],
    queries: np.ndarray,
    numpy_outputs: dict[str, tuple[np.ndarray, ...]],
    numpy_seconds: dict[str, float],
) -> dict[str, object]:
    result: dict[str, object] = {
        "assessed": True,
        "available": False,
        "mlx_version": _installed_version_antigravity("mlx"),
        "mlx_metal_version": _installed_version_antigravity("mlx-metal"),
        "device": "unavailable",
        "probe_seconds": None,
        "benchmark_seconds": None,
        "finite_outputs": "not_run",
        "numerical_parity": "not_run",
        "candidate_index_parity": "not_run",
        "decision_parity": "not_run",
        "speedup": None,
        "speed_gate_pass": False,
        "selected": False,
        "variants": {},
    }
    probe_started = perf_counter()
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlx.core as mx; value=mx.array([1.0]); mx.eval(value); print(mx.default_device())",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    result["probe_seconds"] = perf_counter() - probe_started
    if probe.returncode != 0 or "No Metal device available" in probe.stderr:
        reason = (
            "mlx_metal_device_unavailable"
            if "No Metal device available" in probe.stderr
            else "mlx_probe_failed"
        )
        result["fallback_reason"] = reason
        result["variants"] = {
            variant: {
                "cpu_seconds": numpy_seconds[variant],
                "mlx_seconds_including_transfer_and_sync": None,
                "finite_outputs": "not_run",
                "numerical_parity": "not_run",
                "candidate_index_parity": "not_run",
                "decision_parity": "not_run",
                "speedup": None,
                "speed_gate_pass": False,
                "selected": False,
                "fallback_reason": reason,
            }
            for variant in policies
        }
        return result
    try:
        import mlx.core as mx

        variant_results = {}
        for variant, policy in policies.items():
            started = perf_counter()
            matrix = mx.array(policy._matrix)
            precision = (
                mx.array(policy._precision)
                if variant == "robust_shrinkage_mahalanobis"
                else None
            )
            outputs = []
            for query in queries:
                delta = matrix - mx.array(query)
                if variant == "robust_manhattan":
                    distance = mx.sum(mx.abs(delta), axis=1)
                else:
                    distance = mx.sqrt(
                        mx.maximum(0.0, mx.sum((delta @ precision) * delta, axis=1))
                    )
                mx.eval(distance)
                outputs.append(np.asarray(distance, dtype=np.float64))
            elapsed = perf_counter() - started
            parity = all(
                np.allclose(left, right, rtol=1e-12, atol=1e-12)
                for left, right in zip(outputs, numpy_outputs[variant], strict=True)
            )
            finite = all(
                np.isfinite(item).all()
                for item in (*outputs, *numpy_outputs[variant])
            )
            candidate_indices = all(
                np.array_equal(
                    nearest_distinct_family_indices_reference_antigravity(
                        left, policy._families, K_ANTIGRAVITY
                    ),
                    nearest_distinct_family_indices_reference_antigravity(
                        right, policy._families, K_ANTIGRAVITY
                    ),
                )
                for left, right in zip(
                    numpy_outputs[variant], outputs, strict=True
                )
            )
            gate = choose_mlx_backend_antigravity(
                cpu_seconds=numpy_seconds[variant],
                mlx_seconds=elapsed,
                finite_outputs=finite,
                numerical_parity=parity,
                candidate_index_parity=candidate_indices,
                decision_parity=candidate_indices,
            )
            variant_results[variant] = {
                "cpu_seconds": numpy_seconds[variant],
                "mlx_seconds_including_transfer_and_sync": elapsed,
                "finite_outputs": finite,
                "numerical_parity": parity,
                "candidate_index_parity": candidate_indices,
                "decision_parity": candidate_indices,
                "decision_parity_basis": "exact_candidate_indices_plus_frozen_gate",
                **gate,
            }
        finite = all(bool(item["finite_outputs"]) for item in variant_results.values())
        numerical = all(bool(item["numerical_parity"]) for item in variant_results.values())
        candidate_indices = all(
            bool(item["candidate_index_parity"]) for item in variant_results.values()
        )
        decisions = all(bool(item["decision_parity"]) for item in variant_results.values())
        total_cpu = sum(numpy_seconds.values())
        total_mlx = sum(
            float(item["mlx_seconds_including_transfer_and_sync"])
            for item in variant_results.values()
        )
        result.update(
            {
                "available": True,
                "device": probe.stdout.strip() or str(mx.default_device()),
                "benchmark_seconds": total_mlx,
                "finite_outputs": finite,
                "numerical_parity": numerical,
                "candidate_index_parity": candidate_indices,
                "decision_parity": decisions,
                "speedup": total_cpu / total_mlx,
                "speed_gate_pass": all(
                    bool(item["speed_gate_pass"]) for item in variant_results.values()
                ),
                "selected": any(
                    bool(item["selected"]) for item in variant_results.values()
                ),
                "variants": variant_results,
            }
        )
    except Exception as error:
        result["fallback_reason"] = f"mlx_unavailable: {type(error).__name__}: {error}"
    return result


def assess_acceleration_antigravity(
    rows: Sequence[EvidenceRowAntigravity], *, query_limit: int = 256
) -> dict[str, object]:
    fitted = tuple(rows)
    if not fitted or any(type(row) is not EvidenceRowAntigravity for row in fitted):
        raise ValueError("rows must contain strict evidence rows")
    if type(query_limit) is not int or query_limit < 1:
        raise ValueError("query_limit must be a positive integer")
    limit = min(query_limit, len(fitted))
    policies = {
        variant: StrictResidualPolicyAntigravity(
            k=K_ANTIGRAVITY,
            backend="numpy_reference",
            distance_variant=variant,
        ).fit(fitted)
        for variant in VARIANTS_ANTIGRAVITY
    }
    queries = policies[VARIANTS_ANTIGRAVITY[0]]._matrix[:limit]
    numpy_outputs: dict[str, tuple[np.ndarray, ...]] = {}
    assessments = {}
    for variant, policy in policies.items():
        precision = getattr(policy, "_precision", None)
        started = perf_counter()
        reference = tuple(
            _numpy_distances_antigravity(
                policy._matrix,
                query,
                variant=variant,
                precision=precision,
            )
            for query in queries
        )
        numpy_seconds = perf_counter() - started
        numpy_outputs[variant] = reference

        started = perf_counter()
        first = distance_values_numba_antigravity(
            policy._matrix,
            queries[0],
            variant=variant,
            precision=precision,
        )
        first_seconds = perf_counter() - started
        started = perf_counter()
        remaining = tuple(
            distance_values_numba_antigravity(
                policy._matrix,
                query,
                variant=variant,
                precision=precision,
            )
            for query in queries[1:]
        )
        warm_seconds = perf_counter() - started
        candidate = (first, *remaining)
        numba_seconds = first_seconds + warm_seconds
        finite = all(np.isfinite(item).all() for item in (*reference, *candidate))
        numerical = all(
            np.allclose(left, right, rtol=1e-12, atol=1e-12)
            for left, right in zip(reference, candidate, strict=True)
        )
        candidate_indices = all(
            np.array_equal(
                nearest_distinct_family_indices_reference_antigravity(
                    left, policy._families, K_ANTIGRAVITY
                ),
                nearest_distinct_family_indices_reference_antigravity(
                    right, policy._families, K_ANTIGRAVITY
                ),
            )
            for left, right in zip(reference, candidate, strict=True)
        )
        numba_policy = StrictResidualPolicyAntigravity(
            k=K_ANTIGRAVITY,
            backend="numpy_reference",
            distance_backend="numba_cpu",
            distance_variant=variant,
        ).fit(fitted)
        decisions = all(
            policy.route(
                _feature_mapping_antigravity(row),
                row_knn_method="minirocket_local_gap",
            )
            == numba_policy.route(
                _feature_mapping_antigravity(row),
                row_knn_method="minirocket_local_gap",
            )
            for row in fitted[:limit]
        )
        gate = choose_backend_antigravity(
            numpy_seconds=numpy_seconds,
            candidate_seconds=numba_seconds,
            finite_outputs=finite,
            numerical_parity=numerical,
            candidate_index_parity=candidate_indices,
            decision_parity=decisions,
        )
        assessments[variant] = {
            "query_count": limit,
            "fit_row_count": len(fitted),
            "numpy_seconds": numpy_seconds,
            "numba_first_call_seconds": first_seconds,
            "numba_warm_seconds": warm_seconds,
            "numba_seconds_including_first_call": numba_seconds,
            "finite_outputs": finite,
            "numerical_parity": numerical,
            "candidate_index_parity": candidate_indices,
            "decision_parity": decisions,
            **gate,
        }
    return {
        "schema": "exp484_acceleration_v1",
        "versions": {
            "numpy": np.__version__,
            "numba": numba.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "variants": assessments,
        "mlx": _assess_mlx_antigravity(
            policies,
            queries,
            numpy_outputs,
            {
                variant: float(assessment["numpy_seconds"])
                for variant, assessment in assessments.items()
            },
        ),
    }


__all__ = [
    "assess_acceleration_antigravity",
    "choose_backend_antigravity",
    "choose_mlx_backend_antigravity",
]
