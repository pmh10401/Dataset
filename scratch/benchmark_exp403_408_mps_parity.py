#!/usr/bin/env python3
"""Record the four-run CPU/MPS parity decision for the frozen Exp406 path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp403_408.artifact_io import canonical_json_bytes, write_json_atomic
from experiments.exp403_408.contracts import FrozenConfig, SEED
from experiments.exp403_408.foundation import load_dataset_input
from experiments.exp403_408.tri_view_ssl import _produce_raw_scores, score_tri_view


def _score_hash(scores: object) -> str:
    return hashlib.sha256(canonical_json_bytes(asdict(scores))).hexdigest()


def _raw_parity(train: object, test: object, config: FrozenConfig) -> tuple[bool, float]:
    cpu = np.asarray(_produce_raw_scores(train, test, config, "cpu").test_scores, dtype=float)
    mps = np.asarray(_produce_raw_scores(train, test, config, "mps").test_scores, dtype=float)
    rank_match = bool(np.array_equal(np.argsort(-cpu, kind="stable"), np.argsort(-mps, kind="stable")))
    return rank_match, float(np.max(np.abs(cpu - mps), initial=0.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CricketX_normal_1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/minho/Documents/Dataset/exp403_408_feature_ml_validation"),
    )
    args = parser.parse_args()
    bundle = load_dataset_input(args.dataset)
    config = FrozenConfig(target_length=int(bundle.target_length))
    command = (
        "python3 scratch/benchmark_exp403_408_mps_parity.py "
        f"--dataset {args.dataset} --output-dir {args.output_dir}"
    )
    runs: list[dict[str, object]] = []
    for device in ("cpu", "cpu", "mps", "mps"):
        started = time.perf_counter()
        scores = score_tri_view(bundle.train_normal, bundle.test_values, config, device=device)
        runtime = time.perf_counter() - started
        if device == "mps":
            rank_match, max_delta = _raw_parity(bundle.train_normal, bundle.test_values, config)
        else:
            rank_match, max_delta = True, 0.0
        runs.append({
            "run": f"{device}-{sum(run['device'] == device for run in runs) + 1}",
            "device": device,
            "runtime_seconds": runtime,
            "prediction_sha256": _score_hash(scores),
            "rank_match": rank_match,
            "max_raw_score_delta": max_delta,
        })
    cpu_runtime = sum(float(run["runtime_seconds"]) for run in runs if run["device"] == "cpu") / 2.0
    mps_runtime = sum(float(run["runtime_seconds"]) for run in runs if run["device"] == "mps") / 2.0
    if not all(bool(run["rank_match"]) and float(run["max_raw_score_delta"]) <= 1e-5 for run in runs):
        raise ValueError("CPU/MPS parity benchmark did not satisfy the frozen gate")
    if mps_runtime <= cpu_runtime:
        raise ValueError("CPU retention benchmark expected MPS to be slower")
    payload = {
        "schema": "exp403_408_task13_mps_benchmark/v1",
        "benchmark_command": command,
        "full_command": "python3 scratch/run_exp403_408_feature_ml_validation.py --stage all --workers 7 --device auto --reset-checkpoint --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation",
        "report_command": "python3 scratch/run_exp403_408_feature_ml_validation.py --stage report --workers 7 --device auto --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation",
        "dataset": str(bundle.dataset),
        "n_train": int(len(bundle.train_normal)),
        "n_test": int(len(bundle.test_values)),
        "seed": SEED,
        "device": "mps",
        "score_tolerance": 1e-5,
        "runs": runs,
        "decision": {
            "selected_device": "cpu",
            "reason": "parity passed but MPS slower",
            "cpu_mean_runtime_seconds": cpu_runtime,
            "mps_mean_runtime_seconds": mps_runtime,
        },
    }
    output = args.output_dir / "task13_exp406_mps_benchmark.json"
    digest = write_json_atomic(output, payload)
    print(f"{output} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
