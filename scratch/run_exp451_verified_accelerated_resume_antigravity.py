"""Closed Exp451 command surface for verified migration and accelerated resume."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def _emit_antigravity(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


def _roots_antigravity() -> dict[str, Path]:
    from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import (
        EVALUATION_ROOT, OFFICIAL_ROOT, REPORT_ROOT, SMOKE_ROOT,
    )
    return {
        "smoke": SMOKE_ROOT,
        "official": OFFICIAL_ROOT,
        "evaluation": EVALUATION_ROOT,
        "report": REPORT_ROOT,
    }


def _code_closure_antigravity() -> dict[str, str]:
    """Bind the migration/run authority to the present Exp451 execution closure."""
    directory = REPOSITORY / "experiments" / "exp451_verified_accelerated_resume_antigravity"
    paths = sorted(directory.glob("*_antigravity.py"))
    paths.append(REPOSITORY / "paper" / "equipment_anomaly_detection" / "data_paper.py")
    return {
        str(path.relative_to(REPOSITORY)): sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _full_inputs_antigravity():
    """Build the read-only full catalog and current Exp451 run context."""
    from experiments.exp451_verified_accelerated_resume_antigravity.backend_antigravity import (
        assess_exp451_backend_antigravity,
    )
    from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import (
        FROZEN_EXP451_CONFIG, OFFICIAL_ROOT,
    )
    from experiments.exp451_verified_accelerated_resume_antigravity.contracts_antigravity import (
        Exp451RunContext, canonical_json_bytes,
    )
    from paper.equipment_anomaly_detection.catalog_paper import build_catalog

    catalog = build_catalog(FROZEN_EXP451_CONFIG.database_path, mode="full")
    backend = assess_exp451_backend_antigravity()
    config_sha256 = sha256(canonical_json_bytes({
        "experiment": "Exp451", "mode": "full", "frozen_config": str(FROZEN_EXP451_CONFIG),
        "cpu_workers": backend.cpu_workers, "inner_numerical_threads": backend.inner_numerical_threads,
    })).hexdigest()
    context = Exp451RunContext(
        "full", OFFICIAL_ROOT, FROZEN_EXP451_CONFIG.database_sha256,
        catalog["catalog_sha256"], _code_closure_antigravity(), backend, config_sha256,
    )
    return context, catalog


def _load_raw_antigravity(catalog: dict[str, object]):
    from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import FROZEN_EXP451_CONFIG
    from paper.equipment_anomaly_detection.data_paper import (
        bind_preverified_database_read_only_paper, load_dataset_values,
    )

    binding = bind_preverified_database_read_only_paper(
        FROZEN_EXP451_CONFIG.database_path,
        verified_catalog_database=catalog["database"],
    )
    return {
        entry["dataset"]: load_dataset_values(
            FROZEN_EXP451_CONFIG.database_path, entry["dataset"],
            preverified_database=binding,
        )
        for entry in catalog["datasets"]
    }


def _migrate_antigravity() -> dict[str, object]:
    from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import EXP450_SOURCE_ROOT
    from experiments.exp451_verified_accelerated_resume_antigravity.migration_antigravity import (
        migrate_verified_exp450_results_antigravity,
    )

    context, catalog = _full_inputs_antigravity()
    manifest = migrate_verified_exp450_results_antigravity(
        repository=REPOSITORY, source_root=EXP450_SOURCE_ROOT, target_root=context.run_root,
        catalog=catalog, target_context_sha256=context.sha256,
    )
    return {
        "experiment": "Exp451", "command": "migrate", "entries": manifest.total_count,
        "counts_by_condition": manifest.counts_by_condition, "manifest_sha256": manifest.sha256,
        "run_root": str(context.run_root),
    }


def _verify_migration_antigravity():
    from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import EXP450_SOURCE_ROOT
    from experiments.exp451_verified_accelerated_resume_antigravity.migration_antigravity import (
        verify_migration_manifest_antigravity,
    )

    context, catalog = _full_inputs_antigravity()
    manifest = verify_migration_manifest_antigravity(
        repository=REPOSITORY, source_root=EXP450_SOURCE_ROOT, target_root=context.run_root,
        catalog=catalog, target_context_sha256=context.sha256,
    )
    return context, catalog, manifest


def _run_antigravity() -> dict[str, object]:
    from experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity import (
        execute_exp451_resume_antigravity,
    )

    context, catalog, manifest = _verify_migration_antigravity()
    raw = _load_raw_antigravity(catalog)
    inventory = execute_exp451_resume_antigravity(
        context.run_root, context, catalog, manifest, raw, progress=_emit_antigravity,
    )
    return {
        "experiment": "Exp451", "command": "run", "completed": inventory.completed_count,
        "remaining": inventory.remaining_count, "by_condition": inventory.by_condition,
        "cpu_workers": context.backend.cpu_workers,
        "maximum_concurrent_mps_jobs": 1 if context.backend.tensor_backend == "mps" else 0,
    }


def _status_antigravity() -> dict[str, object]:
    from experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity import (
        status_snapshot_antigravity,
    )

    roots = _roots_antigravity()
    official = roots["official"]
    manifest = official / "migration_manifest_antigravity.json"
    migrated = 4234 if manifest.is_file() and not manifest.is_symlink() else 0
    return {
        "experiment": "Exp451", "command": "status",
        "roots": {name: {"path": str(root), "exists": root.exists()} for name, root in roots.items()},
        "migration_manifest_exists": migrated == 4234,
        "progress_log": str(official / "progress_antigravity.jsonl"),
        **status_snapshot_antigravity(official, imported_completed=migrated),
    }


def _delegate_later_stage_antigravity(command: str) -> dict[str, object]:
    """Reserve Task 6's verified evaluation/report implementation without duplicating it."""
    raise ValueError(f"Exp451 {command} is unavailable until Task 6 verification/evaluation is implemented")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("migrate", "verify-migration", "run", "status", "verify", "evaluate", "report")
    )
    parser.add_argument("--mode", choices=("full",), default="full")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "migrate":
            value = _migrate_antigravity()
        elif arguments.command == "verify-migration":
            context, _, manifest = _verify_migration_antigravity()
            value = {
                "experiment": "Exp451", "command": "verify-migration",
                "entries": manifest.total_count, "counts_by_condition": manifest.counts_by_condition,
                "manifest_sha256": manifest.sha256, "run_root": str(context.run_root),
            }
        elif arguments.command == "run":
            value = _run_antigravity()
        elif arguments.command == "status":
            value = _status_antigravity()
        else:
            value = _delegate_later_stage_antigravity(arguments.command)
        _emit_antigravity(value)
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
