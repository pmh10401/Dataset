"""Importable macOS-spawn smoke helper for the Task 7 CPU cache worker."""

from __future__ import annotations

import json
import multiprocessing
import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.population import load_frozen_population
from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import (
    _run_contract,
    _source_code_hash,
    _worker_build_cache_unit,
    tcn_exclusive_lock,
)
from experiments.exp239_248_diverse_score_sources.foundation import database_hash


def run_spawn_smoke() -> dict[str, object]:
    """Run two real scattering cache units through pickle-safe spawn workers."""
    datasets = tuple(load_frozen_population()[:2])
    db_sha = database_hash()
    contract = _run_contract("scattering", device="cpu", benchmark_record=None, db_sha256=db_sha)
    code_sha = _source_code_hash("scattering")
    with tempfile.TemporaryDirectory(prefix="exp267-task7-spawn-") as directory:
        arguments = [
            (directory, "scattering", dataset, "cpu", True, None, db_sha, code_sha, contract)
            for dataset in datasets
        ]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
            rows = list(executor.map(_worker_build_cache_unit, arguments))
    return {
        "datasets": list(datasets),
        "cached": [str(row["dataset"]) for row in rows],
        "eligible": [bool(row["eligible"]) for row in rows],
        "contains_test_labels": [bool(row["contains_test_labels"]) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--try-tcn-lock", type=Path)
    args = parser.parse_args()
    if args.try_tcn_lock is not None:
        try:
            with tcn_exclusive_lock(args.try_tcn_lock):
                print(json.dumps({"entered": True}, sort_keys=True))
        except RuntimeError:
            print(json.dumps({"entered": False}, sort_keys=True))
        return
    print(json.dumps(run_spawn_smoke(), sort_keys=True))


if __name__ == "__main__":
    main()
