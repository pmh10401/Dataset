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
import tempfile
from time import perf_counter
from typing import Any

import numpy as np


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(ROOT_ANTIGRAVITY))

from experiments.exp466_ecg200_phase_registration_margin_antigravity.cli_antigravity import (  # noqa: E402
    _blocked_fold_indices_antigravity,
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


RawFoldAntigravity = tuple[np.ndarray, tuple[np.ndarray, ...]]
RawWorkloadAntigravity = tuple[RawFoldAntigravity, ...]


def build_workload_antigravity() -> RawWorkloadAntigravity:
    """Build data/synthetic batches outside every measured backend execution."""

    normal96 = load_ecg200_normal_train_antigravity(TRAIN_PATH_ANTIGRAVITY)
    train140 = interpolate_to_140_antigravity(normal96)
    folds: list[RawFoldAntigravity] = []
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
                (
                    np.asarray(train140[fit_indices], dtype=np.float64),
                    (normal_query, *primary_scopes),
                ),
            )
    workload = tuple(folds)
    fit_raw = sum(fit_rows.shape[0] for fit_rows, _query_scopes in workload)
    query_raw = sum(
        scope.shape[0] for _fit_rows, query_scopes in workload for scope in query_scopes
    )
    if (len(workload), fit_raw, query_raw) != (10, 552, 690):
        raise ValueError("assembled workload differs from frozen 10-fold counts")
    return workload


def _save_raw_workload_antigravity(
    path: Path,
    workload: RawWorkloadAntigravity,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for fold_index, (fit_rows, query_scopes) in enumerate(workload):
        arrays[f"fit_{fold_index}_antigravity"] = fit_rows
        for scope_index, scope in enumerate(query_scopes):
            arrays[f"query_{fold_index}_{scope_index}_antigravity"] = scope
    arrays["fold_count_antigravity"] = np.array([len(workload)], dtype=np.int64)
    arrays["scope_count_antigravity"] = np.array(
        [len(query_scopes) for _fit_rows, query_scopes in workload],
        dtype=np.int64,
    )
    np.savez(path, **arrays)


def _load_raw_workload_antigravity(path: Path) -> RawWorkloadAntigravity:
    with np.load(path, allow_pickle=False) as archive:
        fold_count = int(archive["fold_count_antigravity"][0])
        scope_counts = archive["scope_count_antigravity"]
        folds = []
        for fold_index in range(fold_count):
            fit_rows = np.array(
                archive[f"fit_{fold_index}_antigravity"],
                dtype=np.float64,
                copy=True,
            )
            query_scopes = tuple(
                np.array(
                    archive[f"query_{fold_index}_{scope_index}_antigravity"],
                    dtype=np.float64,
                    copy=True,
                )
                for scope_index in range(int(scope_counts[fold_index]))
            )
            folds.append((fit_rows, query_scopes))
    return tuple(folds)


def _materialize_after_timer_antigravity(
    raw_workload: RawWorkloadAntigravity,
    *,
    numpy_repetitions: int,
    warm_repetitions: int,
) -> tuple[object, float]:
    acceleration_module = (
        "experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity."
        "acceleration_antigravity"
    )
    if acceleration_module in sys.modules:
        raise RuntimeError("acceleration backend was imported before the clean timer")
    preparation_started = perf_counter()
    from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.acceleration_antigravity import (
        AccelerationFoldAntigravity,
        AccelerationWorkloadAntigravity,
        assess_acceleration_antigravity,
    )

    workload = AccelerationWorkloadAntigravity(
        folds=tuple(
            AccelerationFoldAntigravity(
                fit_rows=fit_rows,
                query_scopes=query_scopes,
            )
            for fit_rows, query_scopes in raw_workload
        ),
        numpy_repetitions=numpy_repetitions,
        warm_repetitions=warm_repetitions,
    )
    preparation_seconds = perf_counter() - preparation_started
    return assess_acceleration_antigravity(workload), preparation_seconds


def _include_clean_preparation_antigravity(
    payload: dict[str, Any],
    *,
    preparation_seconds: float,
) -> dict[str, Any]:
    backends = payload["backends"]
    for backend in backends.values():
        backend["backend_preparation_seconds"] = 0.0
    numba = backends["numba"]
    numba["backend_preparation_seconds"] = preparation_seconds
    for field in ("cold_total_seconds", "gate_seconds"):
        if numba[field] is not None:
            numba[field] += preparation_seconds
    if numba["gate_seconds"] is not None:
        numpy_seconds = float(backends["numpy"]["warm_median_seconds"])
        numba["speedup"] = numpy_seconds / float(numba["gate_seconds"])
        numba["eligible"] = bool(
            numba["parity_passed"]
            and numba["deterministic"]
            and numba["speedup"] >= float(payload["timing_protocol"]["required_speedup"])
        )
        if not numba["eligible"] and numba["parity_passed"] and numba["deterministic"]:
            numba["fallback_reason"] = (
                "Numba clean-process backend preparation, cold compile, and one warm "
                "execution missed the 1.1x gate"
            )
    eligible = [
        backend
        for name, backend in backends.items()
        if name != "numpy" and backend["eligible"]
    ]
    if eligible:
        selected = min(eligible, key=lambda item: float(item["gate_seconds"]))["backend"]
    else:
        selected = "numpy"
    payload["selected_backend"] = selected
    payload["gate_passed"] = selected != "numpy"
    payload["status"] = (
        "ACCELERATION_GATE_PASSED"
        if payload["gate_passed"]
        else "BLOCKED_AT_ACCELERATION_GATE"
    )
    payload["fallback_reason"] = (
        None
        if payload["gate_passed"]
        else "No eligible backend passed exact parity, determinism, and the clean-process cold 1.1x gate; retain NumPy CPU."
    )
    payload["timing_protocol"]["clean_subprocess"] = True
    payload["timing_protocol"]["backend_preparation_included"] = True
    return payload


def _run_clean_worker_antigravity(
    raw_path: Path,
    output_path: Path,
    *,
    numpy_repetitions: int,
    warm_repetitions: int,
) -> None:
    raw_workload = _load_raw_workload_antigravity(raw_path)
    assessment, preparation_seconds = _materialize_after_timer_antigravity(
        raw_workload,
        numpy_repetitions=numpy_repetitions,
        warm_repetitions=warm_repetitions,
    )
    payload = _include_clean_preparation_antigravity(
        assessment_payload_antigravity(assessment),
        preparation_seconds=preparation_seconds,
    )
    payload["timing_protocol"]["numpy_full_workload_repetitions"] = numpy_repetitions
    payload["timing_protocol"]["numba_warm_full_workload_repetitions"] = warm_repetitions
    payload["timing_protocol"]["acceleration_preloaded_before_timer"] = False
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def measure_clean_subprocess_antigravity(
    raw_workload: RawWorkloadAntigravity,
    *,
    numpy_repetitions: int = 21,
    warm_repetitions: int = 21,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="exp468_acceleration_antigravity_") as directory:
        root = Path(directory)
        raw_path = root / "raw_workload_antigravity.npz"
        output_path = root / "cold_assessment_antigravity.json"
        _save_raw_workload_antigravity(raw_path, raw_workload)
        subprocess.run(
            (
                sys.executable,
                str(Path(__file__).resolve()),
                "--cold-worker-antigravity",
                str(raw_path),
                "--worker-output-antigravity",
                str(output_path),
                "--numpy-repetitions-antigravity",
                str(numpy_repetitions),
                "--warm-repetitions-antigravity",
                str(warm_repetitions),
            ),
            check=True,
            cwd=ROOT_ANTIGRAVITY,
        )
        return json.loads(output_path.read_text(encoding="utf-8"))


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
    assessment: object,
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
            "gate_definition": "clean_process_backend_preparation_plus_cold_compile_plus_one_warm_10_fold_execution",
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-worker-antigravity", type=Path)
    parser.add_argument("--worker-output-antigravity", type=Path)
    parser.add_argument("--numpy-repetitions-antigravity", type=int, default=21)
    parser.add_argument("--warm-repetitions-antigravity", type=int, default=21)
    args = parser.parse_args()
    if args.cold_worker_antigravity is not None:
        if args.worker_output_antigravity is None:
            parser.error("--worker-output-antigravity is required for the cold worker")
        _run_clean_worker_antigravity(
            args.cold_worker_antigravity,
            args.worker_output_antigravity,
            numpy_repetitions=args.numpy_repetitions_antigravity,
            warm_repetitions=args.warm_repetitions_antigravity,
        )
        return
    if args.output is None:
        parser.error("--output is required")
    raw_workload = build_workload_antigravity()
    payload = measure_clean_subprocess_antigravity(raw_workload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main_antigravity()
