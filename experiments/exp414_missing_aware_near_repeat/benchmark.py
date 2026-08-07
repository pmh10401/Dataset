"""Fixed repaired-TRAIN benchmark and fail-closed acceleration adoption."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import resource
from time import perf_counter

from experiments.exp411_near_repeat_qualification.acceleration import (
    ACCELERATION_POLICY,
    configure_runtime,
    reset_acceleration_benchmark_state,
)
from experiments.exp411_near_repeat_qualification.inputs import (
    database_snapshot,
)
from experiments.exp411_near_repeat_qualification.qualification import (
    aggregate_qualification,
    qualify_dataset,
)

from .config import FROZEN_CONFIG
from .inputs import iter_prepared_train_datasets
from .runner import _contract_values, _family_from_dataset, _require_clean_git


BENCHMARK_DATASETS = {
    "small": "ACSF1_normal_0",
    "medium": "HandOutlines_normal_0",
    "large": "StarLightCurves_normal_3",
    "missing": "DodgerLoopDay_normal_1",
}
_SCHEMA_VERSION = 1


def require_acceleration_adoption(
    report: Mapping[str, object],
    expected_commit: str,
) -> None:
    """Require the complete commit-bound Exp414 benchmark contract."""
    if not isinstance(report, Mapping) or report.get("schema_version") != 1:
        raise ValueError("Exp414 benchmark report schema is invalid")
    datasets = report.get("benchmark_datasets")
    if (
        not isinstance(datasets, Mapping)
        or set(datasets) != set(BENCHMARK_DATASETS)
    ):
        raise ValueError("Exp414 benchmark dataset coverage is invalid")
    for size, expected_dataset in BENCHMARK_DATASETS.items():
        row = datasets[size]
        if not isinstance(row, Mapping) or row.get("dataset") != expected_dataset:
            raise ValueError("Exp414 benchmark dataset identity is invalid")
        try:
            numpy_seconds = float(row["numpy_seconds"])
            cold_seconds = float(row["numba_cold_seconds"])
            warm_seconds = float(row["numba_warm_seconds"])
            max_abs_error = float(row["max_abs_error"])
            warm_speedup = float(row["warm_speedup"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Exp414 benchmark values are invalid") from error
        timings = (numpy_seconds, cold_seconds, warm_seconds, warm_speedup)
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in timings)
            or not math.isclose(
                warm_speedup,
                numpy_seconds / warm_seconds,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        ):
            raise ValueError("Exp414 benchmark timing evidence is invalid")
        if (
            not math.isfinite(max_abs_error)
            or max_abs_error < 0.0
            or max_abs_error > FROZEN_CONFIG.distance_tolerance
            or row.get("qualification_identity") is not True
            or row.get("tag_identity") is not True
            or row.get("selected_quantile_identity") is not True
        ):
            raise ValueError("Exp414 benchmark numerical parity failed")
        cold_backend = row.get("numba_cold_backend")
        warm_backend = row.get("numba_warm_backend")
        if (
            type(row.get("numba_cold_backend_audit_count")) is not int
            or row.get("numba_cold_backend_audit_count") != 1
            or type(row.get("numba_warm_backend_audit_count")) is not int
            or row.get("numba_warm_backend_audit_count") != 1
            or cold_backend not in {"numpy_cpu", "numba_cpu"}
            or warm_backend not in {"numpy_cpu", "numba_cpu"}
            or row.get("numba_cold_fallback_reason") is not None
            or row.get("numba_warm_fallback_reason") is not None
        ):
            raise ValueError("Exp414 benchmark backend audit evidence failed")
        if size in {"medium", "large"} and (
            cold_backend != "numba_cpu" or warm_backend != "numba_cpu"
        ):
            raise ValueError("Exp414 benchmark Numba backend adoption failed")
        if size in {"medium", "large"} and (
            warm_speedup < ACCELERATION_POLICY.adoption_speedup
        ):
            raise ValueError("Exp414 benchmark speedup gate failed")
    missing = datasets["missing"]
    if (
        missing.get("repaired_runs") != 1
        or missing.get("repaired_values") != 71
    ):
        raise ValueError("Exp414 benchmark missing repair coverage failed")
    runtime = report.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("numba_threads") != FROZEN_CONFIG.numba_threads
        or runtime.get("nested_threads")
        != FROZEN_CONFIG.nested_numerical_threads
    ):
        raise ValueError("Exp414 benchmark thread contract failed")
    if report.get("git_commit") != expected_commit:
        raise ValueError("Exp414 benchmark Git commit mismatch")
    if dict(report.get("database_snapshot", {})) != {
        "db": FROZEN_CONFIG.database_sha256
    }:
        raise ValueError("Exp414 benchmark database snapshot mismatch")
    for name, expected in _contract_values().items():
        if report.get(name) != expected:
            raise ValueError(f"Exp414 benchmark contract mismatch: {name}")


def run_fixed_train_benchmark(db_path: Path) -> Mapping[str, object]:
    """Run NumPy, cold Numba, and warm Numba over four repaired TRAIN sets."""
    FROZEN_CONFIG.validate()
    git_commit = _require_clean_git()
    runtime = dict(configure_runtime())
    before = database_snapshot(db_path)
    if dict(before) != {"db": FROZEN_CONFIG.database_sha256}:
        raise ValueError("Exp414 benchmark database snapshot mismatch")
    prepared = {
        row.dataset: row
        for row in iter_prepared_train_datasets(
            db_path,
            tuple(BENCHMARK_DATASETS.values()),
        )
    }
    if set(prepared) != set(BENCHMARK_DATASETS.values()):
        raise ValueError("Exp414 benchmark prepared dataset coverage mismatch")
    if dict(database_snapshot(db_path)) != dict(before):
        raise ValueError("Exp414 benchmark database changed during read")

    rows: dict[str, object] = {}
    for size, dataset in BENCHMARK_DATASETS.items():
        source = prepared[dataset]
        if source.repair_audit.status != "ready" or not source.runs:
            raise ValueError("Exp414 benchmark prepared dataset is not ready")
        baseline_trace: list[dict[str, object]] = []
        started = perf_counter()
        baseline = qualify_dataset(
            dataset=dataset,
            family=_family_from_dataset(dataset),
            runs=source.runs,
            accelerated=False,
            parity_trace=baseline_trace,
        )
        numpy_seconds = perf_counter() - started

        cold_trace: list[dict[str, object]] = []
        cold_audits = []
        reset_acceleration_benchmark_state()
        started = perf_counter()
        cold = qualify_dataset(
            dataset=dataset,
            family=_family_from_dataset(dataset),
            runs=source.runs,
            accelerated=True,
            backend_audit=cold_audits,
            parity_trace=cold_trace,
        )
        cold_seconds = perf_counter() - started

        warm_trace: list[dict[str, object]] = []
        warm_audits = []
        started = perf_counter()
        warm = qualify_dataset(
            dataset=dataset,
            family=_family_from_dataset(dataset),
            runs=source.runs,
            accelerated=True,
            backend_audit=warm_audits,
            parity_trace=warm_trace,
        )
        warm_seconds = perf_counter() - started
        if len(cold_audits) != 1 or len(warm_audits) != 1:
            raise ValueError("Exp414 benchmark backend audit coverage failed")
        cold_audit = cold_audits[0]
        warm_audit = warm_audits[0]
        selected = tuple(
            aggregate_qualification(
                (result,),
                bootstrap_iterations=20,
            )["selected_quantile"]
            for result in (baseline, cold, warm)
        )
        repair_rows = source.repair_audit.runs
        rows[size] = {
            "dataset": dataset,
            "normal_train_runs": len(source.runs),
            "identity_groups": len(
                {run.values_sha256 for run in source.runs}
            ),
            "numpy_seconds": numpy_seconds,
            "numba_cold_seconds": cold_seconds,
            "numba_warm_seconds": warm_seconds,
            "warm_speedup": numpy_seconds / warm_seconds,
            "max_abs_error": max(
                cold_audit.max_abs_error,
                warm_audit.max_abs_error,
            ),
            "numba_cold_backend_audit_count": len(cold_audits),
            "numba_warm_backend_audit_count": len(warm_audits),
            "numba_cold_backend": cold_audit.backend,
            "numba_warm_backend": warm_audit.backend,
            "numba_cold_fallback_reason": cold_audit.fallback_reason,
            "numba_warm_fallback_reason": warm_audit.fallback_reason,
            "qualification_identity": baseline == cold == warm,
            "tag_identity": (
                baseline_trace == cold_trace == warm_trace
            ),
            "selected_quantile_identity": selected[0] == selected[1] == selected[2],
            "repaired_runs": sum(
                row.repair_method == "linear_and_edge_nearest"
                for row in repair_rows
            ),
            "repaired_values": sum(
                row.nan_count + row.posinf_count + row.neginf_count
                for row in repair_rows
            ),
            "raw_train_identity_sha256": (
                source.repair_audit.raw_train_identity_sha256
            ),
            "repaired_train_identity_sha256": (
                source.repair_audit.repaired_train_identity_sha256
            ),
            "process_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "peak_rss_scope": "cumulative_process_high_water",
        }
    report = {
        "schema_version": _SCHEMA_VERSION,
        "benchmark_datasets": rows,
        "runtime": runtime,
        "database_snapshot": dict(before),
        "git_commit": git_commit,
        **_contract_values(),
    }
    require_acceleration_adoption(report, git_commit)
    return report


__all__ = [
    "BENCHMARK_DATASETS",
    "require_acceleration_adoption",
    "run_fixed_train_benchmark",
]
