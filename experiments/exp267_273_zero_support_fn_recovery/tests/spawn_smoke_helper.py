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
    _stage_invocation_contexts,
    _worker_build_cache_unit,
    tcn_exclusive_lock,
)


def run_spawn_smoke() -> dict[str, object]:
    """Run two real scattering cache units through pickle-safe spawn workers."""
    datasets = tuple(load_frozen_population()[:2])
    context = _stage_invocation_contexts(("scattering",))["scattering"]
    with tempfile.TemporaryDirectory(prefix="exp267-task7-spawn-") as directory:
        arguments = [
            (directory, "scattering", dataset, True, context)
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
            with tcn_exclusive_lock(lock_path=args.try_tcn_lock):
                print(json.dumps({"entered": True}, sort_keys=True))
        except RuntimeError:
            print(json.dumps({"entered": False}, sort_keys=True))
        return
    print(json.dumps(run_spawn_smoke(), sort_keys=True))


if __name__ == "__main__":
    main()
