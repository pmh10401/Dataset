#!/usr/bin/env python3
"""Benchmark the complete TRAIN-only Exp468 S10 workload."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(ROOT_ANTIGRAVITY))

from experiments.exp466_ecg200_phase_registration_margin_antigravity.cli_antigravity import (  # noqa: E402
    _blocked_fold_indices_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.acceleration_antigravity import (  # noqa: E402
    AccelerationAssessmentAntigravity,
    AccelerationFoldAntigravity,
    AccelerationWorkloadAntigravity,
    assess_acceleration_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.config_antigravity import (  # noqa: E402
    FROZEN_CONFIG_ANTIGRAVITY,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.interpolation_antigravity import (  # noqa: E402
    interpolate_to_140_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.sources_antigravity import (  # noqa: E402
    load_ecg200_normal_train_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.synthetic_antigravity import (  # noqa: E402
    primary_rows_antigravity,
)


TRAIN_PATH_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp465_ecg200_train_antigravity/"
    "ECG200_TRAIN_antigravity.ts"
)


def _package_version_antigravity(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _apple_device_antigravity() -> dict[str, str | None]:
    chip = None
    try:
        chip = subprocess.check_output(
            ("sysctl", "-n", "machdep.cpu.brand_string"),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "chip": chip,
    }


def build_workload_antigravity() -> AccelerationWorkloadAntigravity:
    """Build data/synthetic batches outside every measured backend execution."""

    normal96 = load_ecg200_normal_train_antigravity(TRAIN_PATH_ANTIGRAVITY)
    train140 = interpolate_to_140_antigravity(normal96)
    folds: list[AccelerationFoldAntigravity] = []
    frozen_folds = _blocked_fold_indices_antigravity()
    for _seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        for fit_indices, query_indices in frozen_folds:
            normal_query = np.asarray(train140[query_indices], dtype=np.float64)
            generated = tuple(
                primary_rows_antigravity(row)
                for row in normal_query
            )
            primary_scopes = tuple(
                np.stack(
                    tuple(row_changes[name] for row_changes in generated),
                ).astype(np.float64, copy=False)
                for name in FROZEN_CONFIG_ANTIGRAVITY.change_types
            )
            folds.append(
                AccelerationFoldAntigravity(
                    fit_rows=np.asarray(train140[fit_indices], dtype=np.float64),
                    query_scopes=(normal_query, *primary_scopes),
                ),
            )
    workload = AccelerationWorkloadAntigravity(folds=tuple(folds))
    fit_raw = sum(fold.fit_rows.shape[0] for fold in workload.folds)
    query_raw = sum(
        scope.shape[0] for fold in workload.folds for scope in fold.query_scopes
    )
    if (len(workload.folds), fit_raw, query_raw) != (10, 552, 690):
        raise ValueError("assembled workload differs from frozen 10-fold counts")
    return workload


def _backend_payload_antigravity(backend: object) -> dict[str, Any]:
    fields = (
        "backend", "available", "parity_passed", "finite",
        "max_raw_abs_error", "rank_exact", "confirmation_exact",
        "deterministic", "cold_total_seconds", "cold_compile_seconds",
        "warm_median_seconds", "warm_iqr_seconds", "gate_seconds",
        "speedup", "eligible", "fallback_reason",
    )
    return {name: getattr(backend, name) for name in fields}


def assessment_payload_antigravity(
    assessment: AccelerationAssessmentAntigravity,
) -> dict[str, Any]:
    return {
        "schema": "exp468_acceleration_benchmark_v1",
        "status": (
            "ACCELERATION_GATE_PASSED"
            if assessment.gate_passed
            else "BLOCKED_AT_ACCELERATION_GATE"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": _package_version_antigravity("numba"),
            "llvmlite": _package_version_antigravity("llvmlite"),
            "mlx": _package_version_antigravity("mlx"),
            "mlx_metal": _package_version_antigravity("mlx-metal"),
            "apple_device": _apple_device_antigravity(),
        },
        "workload": {
            "seeds": list(FROZEN_CONFIG_ANTIGRAVITY.seeds),
            "folds_per_seed": FROZEN_CONFIG_ANTIGRAVITY.folds,
            "query_sizes_per_seed": [14, 14, 14, 14, 13],
            "fit_sizes_per_seed": [55, 55, 55, 55, 56],
            "scope_order": ["normal", *FROZEN_CONFIG_ANTIGRAVITY.change_types],
            **dict(assessment.workload_counts),
            "length_28_frame_transforms": assessment.workload_counts["frames"],
            "fit_calibration_reused_across_five_scopes": True,
            "dataset_load_interpolation_synthetic_assembly_timed": False,
            "backend_preparation_timed": True,
        },
        "timing_protocol": {
            "numpy_full_workload_repetitions": 21,
            "numba_clean_process_cold_entries": 1,
            "numba_warm_full_workload_repetitions": 21,
            "gate_definition": "cold_compile_plus_one_warm_10_fold_execution",
            "required_speedup": assessment.required_speedup,
            "simultaneous_heavy_backends": False,
        },
        "backends": {
            name: _backend_payload_antigravity(backend)
            for name, backend in assessment.backends.items()
        },
        "selected_backend": assessment.selected_backend,
        "gate_passed": assessment.gate_passed,
        "fallback_reason": (
            None
            if assessment.gate_passed
            else "No eligible backend passed exact parity, determinism, and the cold 1.1x gate; retain NumPy CPU."
        ),
        "profile_and_batching_review": {
            "attempted": True,
            "initial_measurement": {
                "numpy_median_seconds": 0.03215087531134486,
                "numba_cold_gate_seconds": 1.2641031672246754,
                "numba_warm_median_seconds": 0.029086542315781116,
                "numba_speedup": 0.02543374318247438,
                "numba_max_raw_abs_error": 5.967448757360216e-16,
            },
            "profile_finding": "A clean profile attributed 2.194 seconds to Numba compilation of the monolithic transform/reference/JS scorer.",
            "improvement": "Keep the combined five-scope batch and sequential direct DFT, but JIT only the DFT transform; reuse the NumPy oracle's vectorized reference/JS reduction to shrink compilation without changing formulas or outputs.",
            "scientific_formula_changed": False,
            "workload_changed": False,
            "remeasured_after_improvement": True,
        },
        "official_runs": 0,
        "private_runs": 0,
        "result": None,
        "result_hash": None,
    }


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    workload = build_workload_antigravity()
    assessment = assess_acceleration_antigravity(workload)
    payload = assessment_payload_antigravity(assessment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main_antigravity()
