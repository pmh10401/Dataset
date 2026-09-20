"""Focused behavior tests for Exp484 distance variants."""

from __future__ import annotations

import numpy as np
import subprocess
import sys

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.config_antigravity import (
    FEATURE_KEYS_ANTIGRAVITY,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.policy_antigravity import (
    StrictResidualPolicyAntigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity import (
    kernels_numba_antigravity,
)


def _row_antigravity(
    family: str, *, skewness: float, kurtosis: float
) -> EvidenceRowAntigravity:
    return EvidenceRowAntigravity(
        dataset=f"{family}_normal_1",
        source_family=family,
        proposal="minirocket_local_gap",
        features=(10.0, 20.0, 0.5, skewness, kurtosis, 1.0, 0.2),
        scores=(
            CandidateMetricAntigravity("minirocket_local_gap", 0.70, 2),
            CandidateMetricAntigravity("multirocket_local_gap", 0.80, 3),
            CandidateMetricAntigravity("hydra_local_gap", 0.70, 2),
            CandidateMetricAntigravity("hydra_multirocket_local_gap", 0.70, 2),
        ),
    )


def _query_antigravity() -> dict[str, float]:
    return dict(
        zip(
            FEATURE_KEYS_ANTIGRAVITY,
            (10.0, 20.0, 0.5, -2.0, 0.0, 1.0, 0.2),
            strict=True,
        )
    )


def test_robust_manhattan_changes_the_selected_family_set_antigravity() -> None:
    """Break caught: silently falling back to the old standard Euclidean order."""
    rows = (
        _row_antigravity("A", skewness=-3.0, kurtosis=-3.0),
        _row_antigravity("B", skewness=-3.0, kurtosis=-2.0),
        _row_antigravity("C", skewness=-3.0, kurtosis=-1.0),
        _row_antigravity("D", skewness=-2.0, kurtosis=-3.0),
    )
    query = _query_antigravity()

    standard = StrictResidualPolicyAntigravity(k=2).fit(rows).route(
        query, row_knn_method="minirocket_local_gap"
    )
    robust = StrictResidualPolicyAntigravity(
        k=2, distance_variant="robust_manhattan"
    ).fit(rows).route(query, row_knn_method="minirocket_local_gap")

    assert standard.assessments[0].neighbour_families == ("C", "B")
    assert robust.assessments[0].neighbour_families == ("D", "C")


def test_shrinkage_mahalanobis_uses_correlated_geometry_antigravity() -> None:
    """Break caught: replacing shrinkage precision with diagonal distance."""
    points = (
        (-4.0, 2.0),
        (-3.0, 3.0),
        (-5.0, 5.0),
        (3.0, -4.0),
        (4.0, 4.0),
        (-4.0, -2.0),
        (-4.0, 4.0),
        (-4.0, -3.0),
    )
    rows = tuple(
        _row_antigravity(chr(ord("A") + index), skewness=x, kurtosis=y)
        for index, (x, y) in enumerate(points)
    )
    decision = StrictResidualPolicyAntigravity(
        k=3, distance_variant="robust_shrinkage_mahalanobis"
    ).fit(rows).route(
        dict(
            zip(
                FEATURE_KEYS_ANTIGRAVITY,
                (10.0, 20.0, 0.5, -5.0, 0.0, 1.0, 0.2),
                strict=True,
            )
        ),
        row_knn_method="minirocket_local_gap",
    )

    assert decision.assessments[0].neighbour_families == ("F", "A", "H")


def test_numba_distance_kernel_matches_hand_checked_values_antigravity() -> None:
    """Break caught: wrong L1 sum or Mahalanobis cross term in the CPU kernel."""
    matrix = np.asarray(((0.0, 1.0), (3.0, 5.0)), dtype=np.float64)
    query = np.asarray((1.0, 2.0), dtype=np.float64)
    precision = np.asarray(((2.0, 0.5), (0.5, 1.0)), dtype=np.float64)

    manhattan = kernels_numba_antigravity.distance_values_numba_antigravity(
        matrix, query, variant="robust_manhattan"
    )
    mahalanobis = kernels_numba_antigravity.distance_values_numba_antigravity(
        matrix,
        query,
        variant="robust_shrinkage_mahalanobis",
        precision=precision,
    )

    np.testing.assert_allclose(manhattan, (2.0, 5.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        mahalanobis, (2.0, np.sqrt(23.0)), rtol=1e-14, atol=1e-14
    )


def test_numba_policy_keeps_numpy_neighbour_indices_antigravity() -> None:
    """Break caught: selecting Numba while still using NumPy or changing neighbours."""
    points = ((-4.0, 2.0), (-3.0, 3.0), (-5.0, 5.0), (3.0, -4.0),
              (4.0, 4.0), (-4.0, -2.0), (-4.0, 4.0), (-4.0, -3.0))
    rows = tuple(
        _row_antigravity(chr(ord("A") + index), skewness=x, kurtosis=y)
        for index, (x, y) in enumerate(points)
    )
    query = dict(
        zip(
            FEATURE_KEYS_ANTIGRAVITY,
            (10.0, 20.0, 0.5, -5.0, 0.0, 1.0, 0.2),
            strict=True,
        )
    )
    for variant in ("robust_manhattan", "robust_shrinkage_mahalanobis"):
        numpy_policy = StrictResidualPolicyAntigravity(
            k=3, distance_variant=variant, backend="numpy_reference"
        ).fit(rows)
        numba_policy = StrictResidualPolicyAntigravity(
            k=3, distance_variant=variant, backend="numba_cpu"
        ).fit(rows)

        numpy_result = numpy_policy.route(
            query, row_knn_method="minirocket_local_gap"
        )
        numba_result = numba_policy.route(
            query, row_knn_method="minirocket_local_gap"
        )

        assert numba_policy.distance_backend_used == "numba_cpu"
        assert numba_result == numpy_result


def test_numba_can_accelerate_distance_without_changing_other_policy_kernels_antigravity() -> None:
    """Break caught: coupling the new distance backend to frozen gate kernels."""
    rows = tuple(
        _row_antigravity(f"F{index}", skewness=float(index), kurtosis=3.0)
        for index in range(4)
    )
    policy = StrictResidualPolicyAntigravity(
        k=2,
        backend="numpy_reference",
        distance_backend="numba_cpu",
        distance_variant="robust_manhattan",
    ).fit(rows)

    decision = policy.route(
        _query_antigravity(), row_knn_method="minirocket_local_gap"
    )

    assert policy.backend == "numpy_reference"
    assert policy.distance_backend_used == "numba_cpu"
    assert len(decision.assessments) == 3


def test_backend_gate_requires_parity_finite_output_and_real_speedup_antigravity() -> None:
    """Break caught: choosing an accelerated backend from speed alone."""
    from experiments.exp484_distance_qualification_antigravity.acceleration_antigravity import (
        choose_backend_antigravity,
    )

    passing = choose_backend_antigravity(
        numpy_seconds=2.0,
        candidate_seconds=1.0,
        finite_outputs=True,
        numerical_parity=True,
        candidate_index_parity=True,
        decision_parity=True,
    )
    unsafe = choose_backend_antigravity(
        numpy_seconds=2.0,
        candidate_seconds=1.0,
        finite_outputs=True,
        numerical_parity=False,
        candidate_index_parity=True,
        decision_parity=True,
    )
    slower = choose_backend_antigravity(
        numpy_seconds=1.0,
        candidate_seconds=1.1,
        finite_outputs=True,
        numerical_parity=True,
        candidate_index_parity=True,
        decision_parity=True,
    )

    assert passing == {
        "selected_backend": "numba_cpu",
        "speedup": 2.0,
        "fallback_reason": None,
    }
    assert unsafe["selected_backend"] == "numpy_reference"
    assert unsafe["fallback_reason"] == "parity_gate_failed"
    assert slower["selected_backend"] == "numpy_reference"
    assert slower["fallback_reason"] == "numba_not_faster"


def test_mlx_gate_requires_all_parity_and_at_least_1_1x_speedup_antigravity() -> None:
    """Break caught: selecting MLX below the frozen 1.1x speed threshold."""
    from experiments.exp484_distance_qualification_antigravity.acceleration_antigravity import (
        choose_mlx_backend_antigravity,
    )

    passing = choose_mlx_backend_antigravity(
        cpu_seconds=1.1,
        mlx_seconds=1.0,
        finite_outputs=True,
        numerical_parity=True,
        candidate_index_parity=True,
        decision_parity=True,
    )
    too_slow = choose_mlx_backend_antigravity(
        cpu_seconds=1.09,
        mlx_seconds=1.0,
        finite_outputs=True,
        numerical_parity=True,
        candidate_index_parity=True,
        decision_parity=True,
    )
    unsafe = choose_mlx_backend_antigravity(
        cpu_seconds=2.0,
        mlx_seconds=1.0,
        finite_outputs=True,
        numerical_parity=True,
        candidate_index_parity=False,
        decision_parity=True,
    )

    assert passing == {
        "selected": True,
        "speedup": 1.1,
        "speed_gate_pass": True,
        "fallback_reason": None,
    }
    assert too_slow["selected"] is False
    assert too_slow["speed_gate_pass"] is False
    assert too_slow["fallback_reason"] == "mlx_speed_gate_failed"
    assert unsafe["selected"] is False
    assert unsafe["fallback_reason"] == "mlx_parity_gate_failed"


def test_acceleration_assessment_checks_both_distance_variants_antigravity() -> None:
    """Break caught: benchmarking only one candidate or omitting parity evidence."""
    from experiments.exp484_distance_qualification_antigravity.acceleration_antigravity import (
        assess_acceleration_antigravity,
    )

    points = ((-4.0, 2.0), (-3.0, 3.0), (-5.0, 5.0), (3.0, -4.0),
              (4.0, 4.0), (-4.0, -2.0), (-4.0, 4.0), (-4.0, -3.0))
    rows = tuple(
        _row_antigravity(chr(ord("A") + index), skewness=x, kurtosis=y)
        for index, (x, y) in enumerate(points)
    )

    result = assess_acceleration_antigravity(rows, query_limit=4)

    assert result["schema"] == "exp484_acceleration_v1"
    assert set(result["variants"]) == {
        "robust_manhattan",
        "robust_shrinkage_mahalanobis",
    }
    for assessment in result["variants"].values():
        assert assessment["finite_outputs"] is True
        assert assessment["numerical_parity"] is True
        assert assessment["candidate_index_parity"] is True
        assert assessment["decision_parity"] is True
        assert assessment["numpy_seconds"] > 0.0
        assert assessment["numba_seconds_including_first_call"] > 0.0
        assert assessment["selected_backend"] in {
            "numpy_reference",
            "numba_cpu",
        }
    assert result["mlx"]["assessed"] is True


def test_unavailable_mlx_records_versions_device_and_skipped_gates_antigravity(
    monkeypatch,
) -> None:
    """Break caught: an unavailable MLX path publishing an unauditable blank record."""
    from experiments.exp484_distance_qualification_antigravity import (
        acceleration_antigravity,
    )

    monkeypatch.setattr(
        acceleration_antigravity.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="No Metal device available"
        ),
    )
    rows = tuple(
        _row_antigravity(chr(ord("A") + index), skewness=float(index), kurtosis=0.0)
        for index in range(4)
    )

    mlx = acceleration_antigravity.assess_acceleration_antigravity(
        rows, query_limit=1
    )["mlx"]

    assert mlx["available"] is False
    assert mlx["mlx_version"]
    assert mlx["mlx_metal_version"]
    assert mlx["device"] == "unavailable"
    assert mlx["probe_seconds"] >= 0.0
    assert mlx["benchmark_seconds"] is None
    assert mlx["finite_outputs"] == "not_run"
    assert mlx["numerical_parity"] == "not_run"
    assert mlx["candidate_index_parity"] == "not_run"
    assert mlx["decision_parity"] == "not_run"
    assert mlx["speedup"] is None
    assert mlx["speed_gate_pass"] is False
    assert mlx["selected"] is False
    assert set(mlx["variants"]) == {
        "robust_manhattan",
        "robust_shrinkage_mahalanobis",
    }
    assert all(
        variant["fallback_reason"] == "mlx_metal_device_unavailable"
        for variant in mlx["variants"].values()
    )


def test_mlx_device_failure_is_isolated_from_the_parent_process_antigravity() -> None:
    """Break caught: MLX device probing registers a noisy parent atexit callback."""
    code = """
from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import CandidateMetricAntigravity, EvidenceRowAntigravity
from experiments.exp484_distance_qualification_antigravity.acceleration_antigravity import assess_acceleration_antigravity
rows = tuple(EvidenceRowAntigravity(dataset=f'F{i}_normal_1', source_family=f'F{i}', proposal='minirocket_local_gap', features=(10.0,20.0,0.5,float(i),float(i*i),1.0,0.2), scores=(CandidateMetricAntigravity('minirocket_local_gap',0.7,2), CandidateMetricAntigravity('multirocket_local_gap',0.8,3), CandidateMetricAntigravity('hydra_local_gap',0.7,2), CandidateMetricAntigravity('hydra_multirocket_local_gap',0.7,2))) for i in range(4))
assess_acceleration_antigravity(rows, query_limit=1)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[3]),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Exception ignored in atexit callback" not in completed.stderr
