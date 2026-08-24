"""Frozen Exp471 source identity, roots, ordering, and backend policy."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path
from types import MappingProxyType


EXP467_SOURCE_COMMIT_ANTIGRAVITY = "1e616731a88da9bc0268823e1d2cef2a53c1054f"
EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY = (
    "31341d8e14cf58a486cf7418276c432ac2dbda1af139699ff2afbd85969442af"
)
EXP467_SOURCE_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp467_numba_calibration_resume_official_antigravity_20260823"
)
EXP471_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp471_exp467_evaluation_recovery_antigravity_20260824"
)
EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY = (
    EXP471_ROOT_ANTIGRAVITY / "score_cache_antigravity"
)
EXP471_EVALUATION_ROOT_ANTIGRAVITY = (
    EXP471_ROOT_ANTIGRAVITY / "evaluation_antigravity"
)
EXP471_REPORT_ROOT_ANTIGRAVITY = EXP471_ROOT_ANTIGRAVITY / "report_antigravity"
OFFICIAL_PYTHON_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/venvs/"
    "detection-coverage-antigravity-py314/bin/python"
)

METHOD_ORDER_ANTIGRAVITY = (
    "A",
    "B",
    "C",
    "D",
    "knn_distance",
    "lof_novelty",
    "ocsvm_rbf",
    "isolation_forest",
    "deep_svdd_1d",
    "drocc_1d",
)
CONDITION_ORDER_ANTIGRAVITY = METHOD_ORDER_ANTIGRAVITY[:4]

EXP471_BASELINE_DATABASE_SHA256_ANTIGRAVITY = (
    "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
)
EXP471_TEST_COORDINATE_INVENTORY_SHA256_ANTIGRAVITY = (
    "f7999eb329668cfa2bcee6db69e65a0d542d1960dced27f28bc39751756acee6"
)
EXP471_BASELINE_DATASET_COUNT_ANTIGRAVITY = 1117
EXP471_BASELINE_ROOT_BY_METHOD_ANTIGRAVITY = MappingProxyType(
    {
        "knn_distance": "classical",
        "lof_novelty": "classical",
        "ocsvm_rbf": "classical",
        "isolation_forest": "classical",
        "deep_svdd_1d": "deep_svdd",
        "drocc_1d": "drocc",
    }
)
EXP471_BASELINE_ROOT_AUTHORITY_ANTIGRAVITY = MappingProxyType(
    {
        "classical": MappingProxyType(
            {
                "manifest_sha256": (
                    "c546c841c86c8f488e22d0b0b844a81bc17cdd36b1bfcec3b41869d04a774355"
                ),
                "run_config_sha256": (
                    "315f39cc078f4e9db40c4745a8964a5b5e8037df38c6f12627d301f2b815f69d"
                ),
            }
        ),
        "deep_svdd": MappingProxyType(
            {
                "manifest_sha256": (
                    "b1c86dfab4287adb68dae4af05b17279897229138ad3137df8ef418d1ab2f06d"
                ),
                "run_config_sha256": (
                    "b5c60586db2057a9c05d27332fe47a384ad1b3a95361b039870a3e2329be6aa5"
                ),
            }
        ),
        "drocc": MappingProxyType(
            {
                "manifest_sha256": (
                    "999932ee7e41781fd8398bf8c0d386e05a95b29d9b51d10838046f507e256481"
                ),
                "run_config_sha256": (
                    "8ee0d805667abf7889f9149dfdb52ae0977be9194a5583b170f78e2dffd97367"
                ),
            }
        ),
    }
)

EXP467_TOTAL_SUCCESSES_ANTIGRAVITY = 4468
EXP467_IMPORTED_TOTAL_ANTIGRAVITY = 4234
EXP467_NATIVE_TOTAL_ANTIGRAVITY = 234

LOGICAL_CPU_COUNT_ANTIGRAVITY = os.cpu_count()
if (
    type(LOGICAL_CPU_COUNT_ANTIGRAVITY) is not int
    or LOGICAL_CPU_COUNT_ANTIGRAVITY < 1
):
    raise RuntimeError("Exp471 cannot detect a positive logical CPU count")
CPU_WORKERS_ANTIGRAVITY = max(1, LOGICAL_CPU_COUNT_ANTIGRAVITY - 1)
INNER_NUMERICAL_THREADS_ANTIGRAVITY = 1


def _package_version_antigravity(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _acceleration_assessment_antigravity() -> MappingProxyType[str, object]:
    """Observe package/backend readiness without allocating accelerator tensors."""

    pytorch_version = _package_version_antigravity("torch")
    mps_built = False
    mps_available = False
    if pytorch_version != "unavailable":
        try:
            torch = importlib.import_module("torch")
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            mps_built = bool(
                mps is not None
                and callable(getattr(mps, "is_built", None))
                and mps.is_built()
            )
            mps_available = bool(
                mps is not None
                and callable(getattr(mps, "is_available", None))
                and mps.is_available()
            )
        except Exception:
            mps_built = False
            mps_available = False

    mlx_version = _package_version_antigravity("mlx")
    mlx_metal_version = _package_version_antigravity("mlx-metal")
    return MappingProxyType(
        {
            "schema": "exp471-acceleration-assessment-antigravity.v1",
            "tensor_eligible": False,
            "pytorch_available": pytorch_version != "unavailable",
            "pytorch_version": pytorch_version,
            "mlx_available": mlx_version != "unavailable",
            "mlx_version": mlx_version,
            "mlx_metal_available": mlx_metal_version != "unavailable",
            "mlx_metal_version": mlx_metal_version,
            "mps_built": mps_built,
            "mps_available": mps_available,
            "exact_mlx_expression_available": False,
            "transfer_compilation_overhead": (
                "not_benchmarked_without_an_exact_frozen_expression"
            ),
            "deterministic_numerical_parity_requirement": (
                "identical_shard_bytes_and_candidate_indices_with_finite_scores"
            ),
            "canonical_cpu_fallback": "numba_float64_cpu",
            "benchmark_requirement": (
                "accelerator_must_be_at_least_1.1x_faster_including_transfer_"
                "compilation_and_synchronization"
            ),
            "selected_backend": "numba_cpu",
            "fallback_reason": (
                "no_exact_validated_mlx_or_mps_expression_for_frozen_"
                "order_statistic_calibration"
            ),
            "mps_heavy_process_launched": False,
        }
    )


ACCELERATION_ASSESSMENT_ANTIGRAVITY = _acceleration_assessment_antigravity()


__all__ = [
    "ACCELERATION_ASSESSMENT_ANTIGRAVITY",
    "CONDITION_ORDER_ANTIGRAVITY",
    "CPU_WORKERS_ANTIGRAVITY",
    "EXP467_IMPORTED_TOTAL_ANTIGRAVITY",
    "EXP467_NATIVE_TOTAL_ANTIGRAVITY",
    "EXP467_SOURCE_COMMIT_ANTIGRAVITY",
    "EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY",
    "EXP467_SOURCE_ROOT_ANTIGRAVITY",
    "EXP467_TOTAL_SUCCESSES_ANTIGRAVITY",
    "EXP471_BASELINE_DATABASE_SHA256_ANTIGRAVITY",
    "EXP471_BASELINE_DATASET_COUNT_ANTIGRAVITY",
    "EXP471_BASELINE_ROOT_AUTHORITY_ANTIGRAVITY",
    "EXP471_BASELINE_ROOT_BY_METHOD_ANTIGRAVITY",
    "EXP471_EVALUATION_ROOT_ANTIGRAVITY",
    "EXP471_REPORT_ROOT_ANTIGRAVITY",
    "EXP471_ROOT_ANTIGRAVITY",
    "EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY",
    "EXP471_TEST_COORDINATE_INVENTORY_SHA256_ANTIGRAVITY",
    "INNER_NUMERICAL_THREADS_ANTIGRAVITY",
    "LOGICAL_CPU_COUNT_ANTIGRAVITY",
    "METHOD_ORDER_ANTIGRAVITY",
    "OFFICIAL_PYTHON_ANTIGRAVITY",
]
