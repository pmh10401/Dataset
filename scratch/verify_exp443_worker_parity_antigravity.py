#!/usr/bin/env python3
"""Opt-in real Exp443 Policy A sequential/11-worker parity evidence.

The script performs no TEST-label or evaluator import.  It writes one compact
JSON report under ``scratch``; prediction/control bytes are compared exactly,
while elapsed time is informational only.
"""
from __future__ import annotations

import os

for _name_antigravity in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ[_name_antigravity] = "1"

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import time

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.control_antigravity import compare_policy_a_to_archive_antigravity, require_control_gate_antigravity
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.database_antigravity import load_raw_dataset_antigravity, pin_database_snapshot_antigravity
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.minirocket_antigravity import score_prepared_dataset_antigravity
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.preprocessing_antigravity import POLICY_A_ANTIGRAVITY, prepare_dataset_antigravity
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import canonical_prediction_bytes_antigravity, route_policy_antigravity
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.runtime_antigravity import (
    PolicyTaskAntigravity,
    run_ordered_policy_schedule_antigravity,
    worker_parity_antigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.sources_antigravity import verify_frozen_sources_antigravity


def _score(task: PolicyTaskAntigravity):
    prepared = prepare_dataset_antigravity(task.raw, task.unit.policy)
    return score_prepared_dataset_antigravity(prepared)


def _control_payload(control: object) -> dict[str, object]:
    return {
        "status": getattr(control, "status"),
        "passed": getattr(control, "passed"),
        "reason": getattr(control, "reason"),
        "score_evidence": [dict(item) for item in getattr(control, "score_evidence")],
        "routing_identity": dict(getattr(control, "routing_identity")),
        "environment_identity": dict(getattr(control, "environment_identity")),
        "label_access_count": getattr(control, "label_access_count"),
        "max_abs_difference": getattr(control, "max_abs_difference"),
        "candidate_mismatch_count": getattr(control, "candidate_mismatch_count"),
    }


def main() -> int:
    if os.environ.get("EXP443_REAL_PARITY_ANTIGRAVITY") != "1":
        raise SystemExit("set EXP443_REAL_PARITY_ANTIGRAVITY=1 for the pinned real run")
    reviewed_commit = os.popen("git rev-parse HEAD").read().strip()
    snapshot = verify_frozen_sources_antigravity()
    all_units = tuple(
        __import__("experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity", fromlist=["ordered_work_units_antigravity"]).ordered_work_units_antigravity(snapshot.affected_datasets)
    )
    units = tuple(unit for unit in all_units if unit.policy == POLICY_A_ANTIGRAVITY)
    if len(units) != FROZEN_CONFIG.affected_dataset_count:
        raise RuntimeError(f"Policy A parity requires exactly 19 units, got {len(units)}")
    with tempfile.TemporaryDirectory(prefix="exp443_task8_parity_pin_antigravity_") as temporary:
        pinned = pin_database_snapshot_antigravity(FROZEN_CONFIG.database_path, Path(temporary) / "database_antigravity", expected_sha256=FROZEN_CONFIG.database_sha256)
        raw_cache: dict[str, object] = {}

        def load_task(unit):
            if unit.dataset not in raw_cache:
                raw_cache[unit.dataset] = load_raw_dataset_antigravity(pinned, unit.dataset)
            return PolicyTaskAntigravity(unit, raw_cache[unit.dataset])

        def payload(results: dict[int, object]) -> dict[str, object]:
            by_dataset = {units[sequence].dataset: result for sequence, result in results.items()}
            control = compare_policy_a_to_archive_antigravity(snapshot, by_dataset)
            require_control_gate_antigravity(control)
            prediction = route_policy_antigravity(snapshot, by_dataset, POLICY_A_ANTIGRAVITY)
            prediction_bytes = canonical_prediction_bytes_antigravity(prediction)
            return {
                "results": {str(sequence): results[sequence].to_payload() for sequence in sorted(results)},
                "control": _control_payload(control),
                "prediction_bytes_sha256": sha256(prediction_bytes).hexdigest(),
                "prediction_bytes": prediction_bytes.decode("utf-8"),
                "label_access_count": pinned.label_access_count,
                "reviewed_commit": reviewed_commit,
            }

        sequential_results: dict[int, object] = {}
        start = time.perf_counter()
        for completed, unit in enumerate(units, 1):
            sequential_results[unit.sequence] = _score(load_task(unit))
            elapsed = time.perf_counter() - start
            throughput = completed / elapsed if elapsed else 0.0
            eta = (len(units) - completed) / throughput if throughput else None
            print(json.dumps({"stage": "sequential", "dataset": unit.dataset, "completed": completed, "total": len(units), "elapsed_seconds": elapsed, "eta_seconds": eta}, ensure_ascii=False), flush=True)
        sequential_elapsed = time.perf_counter() - start
        sequential_payload = payload(sequential_results)

        parallel_results: dict[int, object] = {}
        start = time.perf_counter()
        parallel_completed = 0

        def commit_parallel(unit, result):
            nonlocal parallel_completed
            parallel_results[unit.sequence] = result
            parallel_completed += 1
            elapsed = time.perf_counter() - start
            throughput = parallel_completed / elapsed if elapsed else 0.0
            eta = (len(units) - parallel_completed) / throughput if throughput else None
            print(json.dumps({"stage": "parallel", "dataset": unit.dataset, "completed": parallel_completed, "total": len(units), "elapsed_seconds": elapsed, "eta_seconds": eta}, ensure_ascii=False), flush=True)

        run_ordered_policy_schedule_antigravity(
            units=units,
            task_loader=load_task,
            score_task=_score,
            commit=commit_parallel,
            workers=FROZEN_CONFIG.outer_workers,
        )
        parallel_elapsed = time.perf_counter() - start
        parallel_payload = payload(parallel_results)
        evidence = worker_parity_antigravity(
            units,
            sequential_runner=lambda: sequential_payload,
            parallel_runner=lambda: parallel_payload,
            sequential_elapsed=max(sequential_elapsed, 1e-12),
            parallel_elapsed=max(parallel_elapsed, 1e-12),
        )
        if pinned.label_access_count != 0 or snapshot.label_access_count != 0:
            raise RuntimeError("real parity accessed TEST labels")
        report = {
            "schema": "exp443.worker_parity.v1",
            "experiment_id": FROZEN_CONFIG.experiment_id,
            "reviewed_commit": reviewed_commit,
            "policy": POLICY_A_ANTIGRAVITY,
            "unit_count": len(units),
            "workers": FROZEN_CONFIG.outer_workers,
            "logical_cpu_count": FROZEN_CONFIG.logical_cpu_count,
            "inner_threads": FROZEN_CONFIG.inner_threads,
            "label_access_count": 0,
            "semantic_equal": evidence.semantic_equal,
            "sequential_payload_sha256": evidence.sequential_payload_sha256,
            "parallel_payload_sha256": evidence.parallel_payload_sha256,
            "sequential_elapsed": evidence.sequential_elapsed,
            "parallel_elapsed": evidence.parallel_elapsed,
            "speed_ratio": evidence.speed_ratio,
            "control_status": sequential_payload["control"]["status"],
            "control_max_abs_difference": sequential_payload["control"]["max_abs_difference"],
            "prediction_bytes_sha256": sequential_payload["prediction_bytes_sha256"],
        }
        output = Path("scratch/exp443_worker_parity_report_antigravity.json")
        output.write_bytes(canonical_json_bytes_antigravity(report))
        print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    pinned.close(terminal_evidence_durable=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
