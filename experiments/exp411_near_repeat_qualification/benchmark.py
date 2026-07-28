"""Fixed normal-TRAIN benchmark and fail-closed acceleration adoption."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import resource
import sqlite3
from time import perf_counter
from urllib.parse import quote

from .acceleration import (
    ACCELERATION_POLICY,
    acceleration_contract_sha256,
    configure_runtime,
)
from .config import FROZEN_CONFIG
from .inputs import (
    _load_one_dataset,
    database_snapshot,
    trace_sql_contract,
)
from .qualification import aggregate_qualification, qualify_dataset


BENCHMARK_DATASETS = {
    "small": "ACSF1_normal_0",
    "medium": "HandOutlines_normal_0",
    "large": "StarLightCurves_normal_3",
}
_SCHEMA_VERSION = 1


def require_acceleration_adoption(report: Mapping[str, object]) -> None:
    if not isinstance(report, Mapping) or report.get("schema_version") != 1:
        raise ValueError("Exp411 benchmark report schema is invalid")
    datasets = report.get("benchmark_datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(
        BENCHMARK_DATASETS
    ):
        raise ValueError("Exp411 benchmark dataset coverage is invalid")
    for size, expected_dataset in BENCHMARK_DATASETS.items():
        row = datasets[size]
        if (
            not isinstance(row, Mapping)
            or row.get("dataset") != expected_dataset
        ):
            raise ValueError("Exp411 benchmark dataset identity is invalid")
        try:
            max_abs_error = float(row["max_abs_error"])
            warm_speedup = float(row["warm_speedup"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Exp411 benchmark values are invalid") from error
        if (
            not math.isfinite(max_abs_error)
            or max_abs_error < 0.0
            or max_abs_error > ACCELERATION_POLICY.distance_tolerance
            or row.get("tag_identity") is not True
            or row.get("selected_quantile_identity") is not True
        ):
            raise ValueError("Exp411 benchmark numerical parity failed")
        if size in {"medium", "large"} and (
            not math.isfinite(warm_speedup)
            or warm_speedup < ACCELERATION_POLICY.adoption_speedup
        ):
            raise ValueError("Exp411 benchmark speedup gate failed")
    runtime = report.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("numba_threads")
        != ACCELERATION_POLICY.numba_threads
        or runtime.get("nested_threads") != 1
    ):
        raise ValueError("Exp411 benchmark thread contract failed")


def run_fixed_train_benchmark(db_path: Path) -> Mapping[str, object]:
    FROZEN_CONFIG.validate()
    runtime = dict(configure_runtime())
    before = database_snapshot(db_path)
    if (
        tuple(before) != FROZEN_CONFIG.expected_database_snapshot_keys
        or before["db"] != FROZEN_CONFIG.database_sha256
    ):
        raise ValueError("Exp411 benchmark database snapshot mismatch")
    resolved = Path(db_path).resolve()
    connection = sqlite3.connect(
        f"file:{quote(str(resolved), safe='/')}?mode=ro",
        uri=True,
    )
    trace: list[str] = []
    connection.set_trace_callback(trace.append)
    try:
        if connection.execute("PRAGMA quick_check;").fetchall() != [("ok",)]:
            raise ValueError("Exp411 benchmark SQLite quick_check failed")
        runs_by_size = {
            size: _load_one_dataset(connection, dataset)
            for size, dataset in BENCHMARK_DATASETS.items()
        }
        trace_sql_contract(trace)
    except sqlite3.Error as error:
        raise ValueError("Exp411 benchmark TRAIN query failed") from error
    finally:
        connection.close()
    if dict(database_snapshot(db_path)) != dict(before):
        raise ValueError("Exp411 benchmark database changed during read")

    rows: dict[str, object] = {}
    for size, dataset in BENCHMARK_DATASETS.items():
        runs = runs_by_size[size]
        baseline_trace: list[dict[str, object]] = []
        started = perf_counter()
        baseline = qualify_dataset(
            dataset=dataset,
            family=_family_from_dataset(dataset),
            runs=runs,
            parity_trace=baseline_trace,
        )
        numpy_seconds = perf_counter() - started

        cold_trace: list[dict[str, object]] = []
        cold_audits = []
        started = perf_counter()
        cold = qualify_dataset(
            dataset=dataset,
            family=_family_from_dataset(dataset),
            runs=runs,
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
            runs=runs,
            accelerated=True,
            backend_audit=warm_audits,
            parity_trace=warm_trace,
        )
        warm_seconds = perf_counter() - started
        baseline_selected = aggregate_qualification(
            (baseline,),
            bootstrap_iterations=20,
        )["selected_quantile"]
        warm_selected = aggregate_qualification(
            (warm,),
            bootstrap_iterations=20,
        )["selected_quantile"]
        max_abs_error = max(
            [audit.max_abs_error for audit in cold_audits + warm_audits],
            default=0.0,
        )
        rows[size] = {
            "dataset": dataset,
            "normal_train_runs": len(runs),
            "identity_groups": len({run.values_sha256 for run in runs}),
            "numpy_seconds": numpy_seconds,
            "numba_cold_seconds": cold_seconds,
            "numba_warm_seconds": warm_seconds,
            "warm_speedup": numpy_seconds / warm_seconds,
            "max_abs_error": max_abs_error,
            "tag_identity": (
                baseline == cold == warm
                and baseline_trace == cold_trace == warm_trace
            ),
            "selected_quantile_identity": baseline_selected == warm_selected,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
    report = {
        "schema_version": _SCHEMA_VERSION,
        "benchmark_datasets": rows,
        "runtime": runtime,
        "database_snapshot": dict(before),
        "acceleration_contract_sha256": acceleration_contract_sha256(),
    }
    require_acceleration_adoption(report)
    return report


def _family_from_dataset(dataset: str) -> str:
    family, marker, suffix = dataset.rpartition("_normal_")
    if marker != "_normal_" or not family or not suffix.isdecimal():
        raise ValueError("Exp411 benchmark dataset identity is invalid")
    return family


__all__ = [
    "BENCHMARK_DATASETS",
    "require_acceleration_adoption",
    "run_fixed_train_benchmark",
]
