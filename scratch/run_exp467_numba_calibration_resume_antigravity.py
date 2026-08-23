"""Closed Exp467 command surface for verified migration and accelerated resume."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

_HISTORICAL_SERIALIZER_PATH_ANTIGRAVITY = (
    "experiments/exp450_resampling_image_ablation_antigravity/artifacts_antigravity.py"
)
_RUNTIME_RESOURCE_PATHS_ANTIGRAVITY = (
    "paper/equipment_anomaly_detection/lineage_lock_paper.json",
)


def _module_name_antigravity(path: Path) -> str:
    relative = path.relative_to(REPOSITORY).with_suffix("")
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    return ".".join(parts)


def _repository_modules_antigravity() -> dict[str, Path]:
    """Map every local Python module deterministically without importing it."""
    return {
        _module_name_antigravity(path): path
        for path in sorted(REPOSITORY.rglob("*.py"))
    }


def _package_initializers_antigravity(path: Path) -> tuple[Path, ...]:
    """Include every local package initializer executed before a closed module import."""
    relative = path.relative_to(REPOSITORY)
    initializers: list[Path] = []
    for index in range(1, len(relative.parts)):
        initializer = REPOSITORY.joinpath(*relative.parts[:index], "__init__.py")
        if initializer.is_file():
            initializers.append(initializer)
    return tuple(initializers)


def _local_import_modules_antigravity(path: Path) -> tuple[str, ...]:
    """Return only syntactically imported local-module candidates for one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = _module_name_antigravity(path)
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            if node.module:
                base.extend(node.module.split("."))
            if base:
                imported.add(".".join(base))
                imported.update(
                    ".".join((*base, alias.name))
                    for alias in node.names
                    if alias.name != "*"
                )
    return tuple(sorted(imported))


def runtime_dependency_relative_paths_antigravity() -> tuple[str, ...]:
    """Return the deterministic transitive local closure used by the official Exp467 CLI."""
    namespace = REPOSITORY / "experiments" / "exp467_numba_calibration_resume_antigravity"
    roots = {
        Path(__file__).resolve(),
        REPOSITORY / _HISTORICAL_SERIALIZER_PATH_ANTIGRAVITY,
        *namespace.glob("*_antigravity.py"),
    }
    modules = _repository_modules_antigravity()
    pending = sorted(roots)
    closure: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in closure:
            continue
        if not path.is_file():
            raise ValueError("Exp467 runtime code closure is incomplete")
        closure.add(path)
        pending.extend(
            initializer for initializer in _package_initializers_antigravity(path)
            if initializer not in closure
        )
        for imported in _local_import_modules_antigravity(path):
            dependency = modules.get(imported)
            if dependency is not None and dependency not in closure:
                pending.append(dependency)
    resources = tuple(REPOSITORY / relative for relative in _RUNTIME_RESOURCE_PATHS_ANTIGRAVITY)
    if any(not path.is_file() for path in resources):
        raise ValueError("Exp467 runtime resource closure is incomplete")
    return tuple(path.relative_to(REPOSITORY).as_posix() for path in sorted((*closure, *resources)))


def _emit_antigravity(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


def _roots_antigravity() -> dict[str, Path]:
    from experiments.exp467_numba_calibration_resume_antigravity.config_antigravity import (
        EVALUATION_ROOT, OFFICIAL_ROOT, REPORT_ROOT, SMOKE_ROOT,
    )
    return {
        "smoke": SMOKE_ROOT,
        "official": OFFICIAL_ROOT,
        "evaluation": EVALUATION_ROOT,
        "report": REPORT_ROOT,
    }


def _code_closure_antigravity() -> dict[str, str]:
    """Bind the migration/run authority to the present Exp467 execution closure."""
    return {
        relative: sha256((REPOSITORY / relative).read_bytes()).hexdigest()
        for relative in runtime_dependency_relative_paths_antigravity()
    }


def _full_inputs_antigravity():
    """Build the read-only full catalog and current Exp467 run context."""
    from experiments.exp467_numba_calibration_resume_antigravity.backend_antigravity import (
        assess_exp467_backend_antigravity,
    )
    from experiments.exp467_numba_calibration_resume_antigravity.config_antigravity import (
        FROZEN_EXP467_CONFIG, OFFICIAL_ROOT,
    )
    from experiments.exp467_numba_calibration_resume_antigravity.contracts_antigravity import (
        Exp467RunContext, canonical_json_bytes,
    )
    from paper.equipment_anomaly_detection.catalog_paper import build_catalog

    catalog = build_catalog(FROZEN_EXP467_CONFIG.database_path, mode="full")
    backend = assess_exp467_backend_antigravity()
    config_sha256 = sha256(canonical_json_bytes({
        "experiment": "Exp467", "mode": "full", "frozen_config": str(FROZEN_EXP467_CONFIG),
        "cpu_workers": backend.cpu_workers, "inner_numerical_threads": backend.inner_numerical_threads,
    })).hexdigest()
    context = Exp467RunContext(
        "full", OFFICIAL_ROOT, FROZEN_EXP467_CONFIG.database_sha256,
        catalog["catalog_sha256"], _code_closure_antigravity(), backend, config_sha256,
    )
    return context, catalog


def _smoke_inputs_antigravity():
    """Build one isolated, values-only frozen-smoke context without reading TEST labels."""
    from experiments.exp467_numba_calibration_resume_antigravity.backend_antigravity import assess_exp467_backend_antigravity
    from experiments.exp467_numba_calibration_resume_antigravity.config_antigravity import (
        FROZEN_EXP467_CONFIG, SMOKE_CONDITION_ANTIGRAVITY, SMOKE_DATASET_ANTIGRAVITY, SMOKE_ROOT,
    )
    from experiments.exp467_numba_calibration_resume_antigravity.contracts_antigravity import Exp467RunContext, canonical_json_bytes
    from paper.equipment_anomaly_detection.catalog_paper import build_catalog

    catalog = build_catalog(
        FROZEN_EXP467_CONFIG.database_path, mode="smoke", smoke_datasets=FROZEN_EXP467_CONFIG.smoke_datasets,
    )
    backend = assess_exp467_backend_antigravity()
    config_sha256 = sha256(canonical_json_bytes({
        "experiment": "Exp467", "mode": "smoke", "condition": SMOKE_CONDITION_ANTIGRAVITY,
        "smoke_dataset": SMOKE_DATASET_ANTIGRAVITY,
    })).hexdigest()
    context = Exp467RunContext(
        "smoke", SMOKE_ROOT, FROZEN_EXP467_CONFIG.database_sha256,
        catalog["catalog_sha256"], _code_closure_antigravity(), backend, config_sha256,
    )
    from paper.equipment_anomaly_detection.data_paper import bind_preverified_database_read_only_paper, load_dataset_values
    binding = bind_preverified_database_read_only_paper(
        FROZEN_EXP467_CONFIG.database_path, verified_catalog_database=catalog["database"],
    )
    if tuple(entry["dataset"] for entry in catalog["datasets"]) != FROZEN_EXP467_CONFIG.smoke_datasets:
        raise ValueError("Exp467 smoke catalog inventory differs")
    dataset = SMOKE_DATASET_ANTIGRAVITY
    raw = {dataset: load_dataset_values(FROZEN_EXP467_CONFIG.database_path, dataset, preverified_database=binding)}
    return context, catalog, raw


def _smoke_antigravity() -> dict[str, object]:
    """Run and durably reread one label-free native C checkpoint under SMOKE_ROOT only."""
    from experiments.exp467_numba_calibration_resume_antigravity.runner_antigravity import (
        validate_native_success_antigravity, write_native_success_antigravity,
    )
    from experiments.exp467_numba_calibration_resume_antigravity.scoring_antigravity import score_dataset_condition_exp467_antigravity
    from experiments.exp467_numba_calibration_resume_antigravity.config_antigravity import SMOKE_CONDITION_ANTIGRAVITY
    from experiments.exp467_numba_calibration_resume_antigravity.contracts_antigravity import Exp467Condition

    context, catalog, raw = _smoke_inputs_antigravity()
    dataset = next(iter(raw))
    condition = Exp467Condition(SMOKE_CONDITION_ANTIGRAVITY, "linear", "area")
    result = score_dataset_condition_exp467_antigravity(raw[dataset], condition, context.backend)
    path = write_native_success_antigravity(context.run_root, context, catalog, raw[dataset], result)
    validate_native_success_antigravity(context.run_root, context, catalog, raw[dataset], result)
    event = {
        "experiment": "Exp467", "command": "smoke", "mode": "smoke", "dataset": dataset,
        "condition": SMOKE_CONDITION_ANTIGRAVITY, "run_root": str(context.run_root), "checkpoint": str(path),
        "context_sha256": context.sha256, "backend": context.backend.tensor_backend,
    }
    _emit_antigravity({"experiment": "Exp467", "command": "smoke-progress", "dataset": dataset, "condition": SMOKE_CONDITION_ANTIGRAVITY, "status": "success"})
    return event


def _load_raw_antigravity(catalog: dict[str, object]):
    from experiments.exp467_numba_calibration_resume_antigravity.config_antigravity import FROZEN_EXP467_CONFIG
    from paper.equipment_anomaly_detection.data_paper import (
        bind_preverified_database_read_only_paper, load_dataset_values,
    )

    binding = bind_preverified_database_read_only_paper(
        FROZEN_EXP467_CONFIG.database_path,
        verified_catalog_database=catalog["database"],
    )
    return {
        entry["dataset"]: load_dataset_values(
            FROZEN_EXP467_CONFIG.database_path, entry["dataset"],
            preverified_database=binding,
        )
        for entry in catalog["datasets"]
    }


def _migrate_antigravity() -> dict[str, object]:
    from experiments.exp467_numba_calibration_resume_antigravity.config_antigravity import EXP450_SOURCE_ROOT
    from experiments.exp467_numba_calibration_resume_antigravity.migration_antigravity import (
        migrate_verified_exp450_results_antigravity,
    )

    context, catalog = _full_inputs_antigravity()
    manifest = migrate_verified_exp450_results_antigravity(
        repository=REPOSITORY, source_root=EXP450_SOURCE_ROOT, target_root=context.run_root,
        catalog=catalog, target_context_sha256=context.sha256,
    )
    return {
        "experiment": "Exp467", "command": "migrate", "entries": manifest.total_count,
        "counts_by_condition": manifest.counts_by_condition, "manifest_sha256": manifest.sha256,
        "run_root": str(context.run_root),
    }


def _verify_migration_antigravity():
    from experiments.exp467_numba_calibration_resume_antigravity.config_antigravity import EXP450_SOURCE_ROOT
    from experiments.exp467_numba_calibration_resume_antigravity.migration_antigravity import (
        verify_migration_manifest_antigravity,
    )

    context, catalog = _full_inputs_antigravity()
    manifest = verify_migration_manifest_antigravity(
        repository=REPOSITORY, source_root=EXP450_SOURCE_ROOT, target_root=context.run_root,
        catalog=catalog, target_context_sha256=context.sha256,
    )
    return context, catalog, manifest


def _run_antigravity() -> dict[str, object]:
    from experiments.exp467_numba_calibration_resume_antigravity.runner_antigravity import (
        execute_exp467_resume_antigravity,
    )

    context, catalog, manifest = _verify_migration_antigravity()
    raw = _load_raw_antigravity(catalog)
    inventory = execute_exp467_resume_antigravity(
        context.run_root, context, catalog, manifest, raw, progress=_emit_antigravity,
    )
    return {
        "experiment": "Exp467", "command": "run", "completed": inventory.completed_count,
        "remaining": inventory.remaining_count, "by_condition": inventory.by_condition,
        "cpu_workers": context.backend.cpu_workers,
        "maximum_concurrent_mps_jobs": 1 if context.backend.tensor_backend == "mps" else 0,
    }


def _status_antigravity() -> dict[str, object]:
    from experiments.exp467_numba_calibration_resume_antigravity.runner_antigravity import (
        read_resume_inventory_antigravity, status_snapshot_antigravity,
    )

    roots = _roots_antigravity()
    official = roots["official"]
    # A status claim is authoritative only after the full source-to-target
    # verifier accepts the manifest; file existence is never sufficient.
    context, catalog, manifest = _verify_migration_antigravity()
    inventory = read_resume_inventory_antigravity(official, context, catalog, manifest)
    return {
        "experiment": "Exp467", "command": "status",
        "roots": {name: {"path": str(root), "exists": root.exists()} for name, root in roots.items()},
        "migration_manifest_exists": True,
        "progress_log": str(official / "progress_antigravity.jsonl"),
        **status_snapshot_antigravity(
            official, imported_completed=len(inventory.imported),
            durable_completed=inventory.completed_count, durable_failed=len(inventory.failures),
        ),
    }


def _verify_antigravity() -> dict[str, object]:
    """Freeze four condition manifests only after complete checkpoint verification."""
    from experiments.exp467_numba_calibration_resume_antigravity.verifier_antigravity import (
        provenance_summary_antigravity,
        verify_exp467_completion_antigravity,
        write_exp467_condition_manifests_antigravity,
    )

    context, catalog, manifest = _verify_migration_antigravity()
    verification = verify_exp467_completion_antigravity(
        context.run_root, context=context, catalog=catalog, migration_manifest=manifest,
    )
    paths = write_exp467_condition_manifests_antigravity(context.run_root, verification)
    return {
        "experiment": "Exp467", "command": "verify",
        **provenance_summary_antigravity(verification),
        "condition_manifests": {condition: str(path) for condition, path in paths.items()},
    }


def _evaluate_result_antigravity():
    """Build label-free score manifests before entering the one frozen label boundary."""
    from experiments.exp467_numba_calibration_resume_antigravity.config_antigravity import EVALUATION_ROOT
    from experiments.exp467_numba_calibration_resume_antigravity.evaluation_antigravity import (
        evaluate_exp467_antigravity,
    )
    from experiments.exp467_numba_calibration_resume_antigravity.verifier_antigravity import (
        build_exp467_score_manifests_antigravity,
        read_persisted_exp467_condition_manifests_antigravity,
        verify_exp467_completion_antigravity,
        write_exp467_condition_manifests_antigravity,
    )

    context, catalog, manifest = _verify_migration_antigravity()
    # Make one authority, then bind the persisted four-manifest evidence and
    # label-free scores to that exact object.  A second verification here would
    # create a distinct capability and weaken the hand-off audit trail.
    verification = verify_exp467_completion_antigravity(
        context.run_root, context=context, catalog=catalog, migration_manifest=manifest,
    )
    # Persisted manifests are required label-boundary evidence. They may be
    # published only from the complete in-memory authority and are reopened
    # before the score/label stages proceed.
    write_exp467_condition_manifests_antigravity(context.run_root, verification)
    read_persisted_exp467_condition_manifests_antigravity(context.run_root, verification)
    raw = _load_raw_antigravity(catalog)
    score_manifests = build_exp467_score_manifests_antigravity(
        context.run_root, verification=verification, context=context, catalog=catalog,
        migration_manifest=manifest, raw_datasets=raw,
    )
    return verification, evaluate_exp467_antigravity(
        EVALUATION_ROOT, verification=verification, score_manifests=score_manifests,
    )


def _evaluate_antigravity() -> dict[str, object]:
    verification, evaluation = _evaluate_result_antigravity()
    return {
        "experiment": "Exp467", "command": "evaluate",
        "method_order": list(evaluation.frozen_evaluation.metric_tables.method_order),
        "inventory_sha256": verification.inventory_sha256,
    }


def _report_antigravity() -> dict[str, object]:
    from experiments.exp467_numba_calibration_resume_antigravity.config_antigravity import REPORT_ROOT
    from experiments.exp467_numba_calibration_resume_antigravity.reporting_antigravity import (
        write_exp467_report_antigravity,
    )

    verification, evaluation = _evaluate_result_antigravity()
    artifact = write_exp467_report_antigravity(evaluation, REPORT_ROOT, verification=verification)
    return {
        "experiment": "Exp467", "command": "report",
        "report_root": str(artifact.frozen_report.root),
        "provenance": str(artifact.provenance_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("migrate", "verify-migration", "run", "status", "verify", "evaluate", "report", "smoke"))
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    arguments = parser.parse_args(argv)
    if (arguments.command == "smoke") != (arguments.mode == "smoke"):
        parser.error("smoke command requires --mode smoke; all other commands require --mode full")
    try:
        if arguments.command == "smoke":
            value = _smoke_antigravity()
        elif arguments.command == "migrate":
            value = _migrate_antigravity()
        elif arguments.command == "verify-migration":
            context, _, manifest = _verify_migration_antigravity()
            value = {
                "experiment": "Exp467", "command": "verify-migration",
                "entries": manifest.total_count, "counts_by_condition": manifest.counts_by_condition,
                "manifest_sha256": manifest.sha256, "run_root": str(context.run_root),
            }
        elif arguments.command == "run":
            value = _run_antigravity()
        elif arguments.command == "status":
            value = _status_antigravity()
        elif arguments.command == "verify":
            value = _verify_antigravity()
        elif arguments.command == "evaluate":
            value = _evaluate_antigravity()
        else:
            value = _report_antigravity()
        _emit_antigravity(value)
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
