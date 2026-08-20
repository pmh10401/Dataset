"""Closed Exp450 command surface with fixed roots and explicit science gates."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
_EXPERIMENT = "Exp450"
_CONDITIONS = ("A", "B", "C", "D")
_EVALUATION_LEAF = "exp450_evaluation_antigravity.json"


def progress_record_antigravity(*, dataset: str, condition: str, status: str, elapsed_seconds: float, completed: int, total: int) -> dict[str, object]:
    """Return the one-line progress schema used by run and smoke."""
    if status not in {"success", "failure"} or condition not in _CONDITIONS or completed < 0 or total < completed or elapsed_seconds < 0:
        raise ValueError("Exp450 progress record is invalid")
    eta = 0.0 if completed == 0 else max(0.0, elapsed_seconds / completed * (total - completed))
    return {"experiment": _EXPERIMENT, "dataset": dataset, "condition": condition, "status": status, "elapsed_seconds": float(elapsed_seconds), "eta_seconds": float(eta), "completed": completed, "total": total}


def _emit(value: object, *, stream=sys.stdout) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False), file=stream, flush=True)


def _fixed_roots():
    from experiments.exp450_resampling_image_ablation_antigravity.config_antigravity import EVALUATION_ROOT, OFFICIAL_ROOT, REPORT_ROOT, SMOKE_ROOT
    return {"smoke": SMOKE_ROOT, "run": OFFICIAL_ROOT, "evaluation": EVALUATION_ROOT, "report": REPORT_ROOT}


def _science_modules_loaded() -> list[str]:
    blocked = ("scoring_antigravity", "label_access", "runner_antigravity")
    return sorted(name for name in sys.modules if any(token in name for token in blocked))


def _status() -> dict[str, object]:
    roots = _fixed_roots()
    return {"experiment": _EXPERIMENT, "command": "status", "roots": {name: {"path": str(root), "exists": root.exists()} for name, root in roots.items()}, "science_modules_loaded": _science_modules_loaded()}


def _code_closure() -> dict[str, str]:
    directory = REPOSITORY / "experiments" / "exp450_resampling_image_ablation_antigravity"
    return {str(path.relative_to(REPOSITORY)): sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("*_antigravity.py"))}


def _verification_inputs(mode: str):
    """Construct exact mode authority without scorer or label imports."""
    from experiments.exp450_resampling_image_ablation_antigravity.backend_antigravity import assess_exp450_backend_antigravity
    from experiments.exp450_resampling_image_ablation_antigravity.config_antigravity import FROZEN_EXP450_CONFIG, OFFICIAL_ROOT, SMOKE_ROOT
    from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import Exp450RunContext, canonical_json_bytes
    from paper.equipment_anomaly_detection.catalog_paper import build_catalog
    if mode not in {"smoke", "full"}:
        raise ValueError("Exp450 verification mode is invalid")
    root = SMOKE_ROOT if mode == "smoke" else OFFICIAL_ROOT
    catalog = build_catalog(FROZEN_EXP450_CONFIG.database_path, mode=mode, smoke_datasets=FROZEN_EXP450_CONFIG.smoke_datasets if mode == "smoke" else ())
    backend = assess_exp450_backend_antigravity()
    config_sha256 = sha256(canonical_json_bytes({"experiment": _EXPERIMENT, "mode": mode, "config": str(FROZEN_EXP450_CONFIG)})).hexdigest()
    context = Exp450RunContext(mode, root, FROZEN_EXP450_CONFIG.database_sha256, catalog["catalog_sha256"], _code_closure(), backend, config_sha256)
    return root, context, catalog


def _verify(mode: str, *, verification_inputs=_verification_inputs) -> dict[str, object]:
    """Run the unchanged v2 scorer-free verifier for every selected-mode leaf."""
    from experiments.exp450_resampling_image_ablation_antigravity.verifier_antigravity import verify_exp450_score_root_antigravity
    root, context, catalog = verification_inputs(mode)
    manifests = {condition: verify_exp450_score_root_antigravity(root, context=context, catalog=catalog, condition_id=condition) for condition in _CONDITIONS}
    return {"experiment": _EXPERIMENT, "command": "verify", "mode": mode, "verified_manifests": {condition: {"manifest_sha256": manifest["manifest_sha256"], "datasets": len(manifest["datasets"]), "test_rows": sum(entry["test"]["count"] for entry in manifest["datasets"])} for condition, manifest in manifests.items()}, "science_modules_loaded": _science_modules_loaded()}


def _science_inputs(mode: str):
    """Load values-only frozen inputs only after a science command was selected."""
    from experiments.exp450_resampling_image_ablation_antigravity.config_antigravity import FROZEN_EXP450_CONFIG
    from paper.equipment_anomaly_detection.data_paper import load_dataset_values

    root, context, catalog = _verification_inputs(mode)
    backend = context.backend
    raw = {entry["dataset"]: load_dataset_values(FROZEN_EXP450_CONFIG.database_path, entry["dataset"]) for entry in catalog["datasets"]}
    return root, context, catalog, raw, backend


def _verified_progress_state(mode: str, root, context, catalog, raw: dict[str, object]) -> tuple[int, set[tuple[str, str]], int]:
    """Recover live-progress counters from the exact durable mode inventory."""
    from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import verify_dataset_condition_inventory_antigravity

    if mode not in {"smoke", "full"} or context.mode != mode:
        raise ValueError("Exp450 progress mode differs from run context")
    inventory = verify_dataset_condition_inventory_antigravity(root, context, catalog, raw)
    total = len(catalog["datasets"]) * len(_CONDITIONS)
    return total, set(inventory.completed), len(inventory.failures)


def _run(mode: str) -> int:
    from experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity import compose_all_conditions_antigravity, execute_local_schedule_antigravity, freeze_all_conditions_antigravity

    started = monotonic()
    root, context, catalog, raw, backend = _science_inputs(mode)
    total, succeeded, failed = _verified_progress_state(mode, root, context, catalog, raw)
    def progress(event: dict[str, object]) -> None:
        nonlocal failed
        coordinate = (str(event["dataset"]), str(event["condition"]))
        if event["status"] == "success":
            if coordinate in succeeded:
                raise ValueError("Exp450 progress cannot re-emit an existing completed coordinate")
            succeeded.add(coordinate)
        else:
            failed += 1
        elapsed = monotonic() - started
        completed = len(succeeded)
        remaining = total - completed
        samples = completed + failed
        eta = 0.0 if samples == 0 else max(0.0, elapsed / samples * remaining)
        _emit({"experiment": _EXPERIMENT, **event, "completed": completed, "failed": failed, "remaining": remaining, "total": total, "elapsed_seconds": elapsed, "eta_seconds": eta})
    results = execute_local_schedule_antigravity(root, context, catalog, raw, backend, progress=progress)
    scores = compose_all_conditions_antigravity(results, raw, backend)
    freeze_all_conditions_antigravity(scores, root, context, catalog)
    return 0


def _require_verified_full_manifests():
    """Verify all A–D full manifests before importing evaluation or label code."""
    root, context, catalog = _verification_inputs("full")
    from experiments.exp450_resampling_image_ablation_antigravity.verifier_antigravity import verify_exp450_score_root_antigravity
    try:
        manifests = {condition: verify_exp450_score_root_antigravity(root, context=context, catalog=catalog, condition_id=condition) for condition in _CONDITIONS}
    except ValueError as error:
        raise ValueError("all four full score manifests must verify before evaluation") from error
    return root, context, catalog, manifests


def evaluation_payload_antigravity(evaluation) -> dict[str, object]:
    from experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity import canonicalize_exp450_json_antigravity
    return canonicalize_exp450_json_antigravity({"method_order": evaluation.metric_tables.method_order, "per_dataset": evaluation.metric_tables.per_dataset, "macro": evaluation.metric_tables.macro, "micro": evaluation.metric_tables.micro, "paired_statistics": evaluation.metric_tables.paired_statistics, "curve_tables": evaluation.curve_tables})


def _write_create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError("Exp450 evaluation artifact already exists with different bytes") from None
        return
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Exp450 evaluation artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_evaluation_artifact_antigravity(evaluation, root: Path) -> Path:
    from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import canonical_json_bytes
    target = Path(root) / _EVALUATION_LEAF
    _write_create_only(target, canonical_json_bytes(evaluation_payload_antigravity(evaluation)))
    return target


def load_evaluation_artifact_antigravity(path: Path):
    from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import Exp450EvaluationAntigravity, MetricTablesExp450Antigravity
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return Exp450EvaluationAntigravity({}, MetricTablesExp450Antigravity(tuple(value["method_order"]), tuple(value["per_dataset"]), value["macro"], value["micro"], tuple(value["paired_statistics"])), value["curve_tables"])


def _evaluate() -> int:
    root, context, catalog, manifests = _require_verified_full_manifests()
    from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import evaluate_exp450_antigravity

    evaluation = evaluate_exp450_antigravity(root, verifier=lambda condition: manifests[condition])
    target = write_evaluation_artifact_antigravity(evaluation, _fixed_roots()["evaluation"])
    _emit({"experiment": _EXPERIMENT, "command": "evaluate", "evaluation": str(target)})
    return 0


def _report() -> int:
    from experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity import write_exp450_report_antigravity
    target = _fixed_roots()["evaluation"] / _EVALUATION_LEAF
    if not target.is_file():
        raise ValueError("verified Exp450 evaluation artifact is required before report")
    evaluation = load_evaluation_artifact_antigravity(target)
    artifact = write_exp450_report_antigravity(evaluation, _fixed_roots()["report"])
    _emit({"experiment": _EXPERIMENT, "command": "report", "html": str(artifact.html_path), "manifest": str(artifact.manifest_path)})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("readiness", "smoke", "run", "status", "verify", "evaluate", "report"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "readiness":
            roots = _fixed_roots()
            selected = roots["smoke" if arguments.mode == "smoke" else "run"]
            _emit({**_status(), "command": "readiness", "mode": arguments.mode, "run_root": str(selected)})
            return 0
        if arguments.command == "status":
            _emit(_status())
            return 0
        if arguments.command == "verify":
            _emit(_verify(arguments.mode))
            return 0
        if arguments.command == "smoke":
            return _run("smoke")
        if arguments.command == "run":
            return _run("full")
        if arguments.command == "evaluate":
            return _evaluate()
        return _report()
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
