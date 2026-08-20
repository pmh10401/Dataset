"""Closed Exp450 command surface with fixed roots and explicit science gates."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
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


def _verify() -> dict[str, object]:
    """Inspect fixed inventory without loading scorer or label capability."""
    roots = _fixed_roots()
    root = roots["run"]
    manifests = {condition: root / "score_roots_antigravity" / f"{condition}_antigravity" / "condition_score_manifest_antigravity.json" for condition in _CONDITIONS}
    return {"experiment": _EXPERIMENT, "command": "verify", "full_manifest_paths": {condition: str(path) for condition, path in manifests.items()}, "all_full_manifests_present": all(path.is_file() for path in manifests.values()), "science_modules_loaded": _science_modules_loaded()}


def _code_closure() -> dict[str, str]:
    directory = REPOSITORY / "experiments" / "exp450_resampling_image_ablation_antigravity"
    return {str(path.relative_to(REPOSITORY)): sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("*_antigravity.py"))}


def _science_inputs(mode: str):
    """Load values-only frozen inputs only after a science command was selected."""
    from experiments.exp450_resampling_image_ablation_antigravity.backend_antigravity import assess_exp450_backend_antigravity
    from experiments.exp450_resampling_image_ablation_antigravity.config_antigravity import FROZEN_EXP450_CONFIG, OFFICIAL_ROOT, SMOKE_ROOT
    from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import Exp450RunContext, canonical_json_bytes
    from paper.equipment_anomaly_detection.catalog_paper import build_catalog
    from paper.equipment_anomaly_detection.data_paper import load_dataset_values

    if mode not in {"smoke", "full"}:
        raise ValueError("Exp450 science mode is invalid")
    root = SMOKE_ROOT if mode == "smoke" else OFFICIAL_ROOT
    catalog = build_catalog(FROZEN_EXP450_CONFIG.database_path, mode=mode, smoke_datasets=FROZEN_EXP450_CONFIG.smoke_datasets if mode == "smoke" else ())
    backend = assess_exp450_backend_antigravity()
    config_sha256 = sha256(canonical_json_bytes({"experiment": _EXPERIMENT, "mode": mode, "config": FROZEN_EXP450_CONFIG.__dict__ if hasattr(FROZEN_EXP450_CONFIG, "__dict__") else str(FROZEN_EXP450_CONFIG)})).hexdigest()
    context = Exp450RunContext(mode, root, FROZEN_EXP450_CONFIG.database_sha256, catalog["catalog_sha256"], _code_closure(), backend, config_sha256)
    raw = {entry["dataset"]: load_dataset_values(FROZEN_EXP450_CONFIG.database_path, entry["dataset"]) for entry in catalog["datasets"]}
    return root, context, catalog, raw, backend


def _run(mode: str) -> int:
    from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import verify_dataset_condition_inventory_antigravity
    from experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity import compose_all_conditions_antigravity, execute_local_schedule_antigravity, freeze_all_conditions_antigravity

    started = monotonic()
    root, context, catalog, raw, backend = _science_inputs(mode)
    results = execute_local_schedule_antigravity(root, context, catalog, raw, backend)
    scores = compose_all_conditions_antigravity(results, raw, backend)
    freeze_all_conditions_antigravity(scores, root, context, catalog)
    inventory = verify_dataset_condition_inventory_antigravity(root, context, catalog, raw)
    total = len(catalog["datasets"]) * len(_CONDITIONS)
    for completed, (dataset, condition) in enumerate(inventory.completed, start=1):
        _emit(progress_record_antigravity(dataset=dataset, condition=condition, status="success", elapsed_seconds=monotonic() - started, completed=completed, total=total))
    for failure in inventory.failures:
        _emit(progress_record_antigravity(dataset=failure.dataset, condition=failure.condition_id, status="failure", elapsed_seconds=monotonic() - started, completed=total, total=total))
    return 0


def _require_verified_full_manifests():
    """Verify all A–D full manifests before importing evaluation or label code."""
    roots = _fixed_roots()
    expected = [roots["run"] / "score_roots_antigravity" / f"{condition}_antigravity" / "condition_score_manifest_antigravity.json" for condition in _CONDITIONS]
    if not all(path.is_file() for path in expected):
        raise ValueError("all four full score manifests must verify before evaluation")
    root, context, catalog, _raw, _backend = _science_inputs("full")
    from experiments.exp450_resampling_image_ablation_antigravity.verifier_antigravity import verify_exp450_score_root_antigravity
    manifests = {condition: verify_exp450_score_root_antigravity(root, context=context, catalog=catalog, condition_id=condition) for condition in _CONDITIONS}
    return root, context, catalog, manifests


def _evaluate() -> int:
    root, context, catalog, manifests = _require_verified_full_manifests()
    from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import canonical_json_bytes
    from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import evaluate_exp450_antigravity

    evaluation = evaluate_exp450_antigravity(root, verifier=lambda condition: manifests[condition])
    payload = {"method_order": list(evaluation.metric_tables.method_order), "per_dataset": [dict(row) for row in evaluation.metric_tables.per_dataset], "macro": {key: dict(value) for key, value in evaluation.metric_tables.macro.items()}, "micro": {key: dict(value) for key, value in evaluation.metric_tables.micro.items()}, "paired_statistics": [dict(row) for row in evaluation.metric_tables.paired_statistics], "curve_tables": {key: dict(value) for key, value in evaluation.curve_tables.items()}}
    target = _fixed_roots()["evaluation"] / _EVALUATION_LEAF
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload)
    if target.exists() and target.read_bytes() != data:
        raise ValueError("Exp450 evaluation artifact already exists with different bytes")
    if not target.exists():
        target.write_bytes(data)
    _emit({"experiment": _EXPERIMENT, "command": "evaluate", "evaluation": str(target)})
    return 0


def _report() -> int:
    from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import Exp450EvaluationAntigravity, MetricTablesExp450Antigravity
    from experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity import write_exp450_report_antigravity
    target = _fixed_roots()["evaluation"] / _EVALUATION_LEAF
    if not target.is_file():
        raise ValueError("verified Exp450 evaluation artifact is required before report")
    payload = json.loads(target.read_text(encoding="utf-8"))
    evaluation = Exp450EvaluationAntigravity({}, MetricTablesExp450Antigravity(tuple(payload["method_order"]), tuple(payload["per_dataset"]), payload["macro"], payload["micro"], tuple(payload["paired_statistics"])), payload["curve_tables"])
    artifact = write_exp450_report_antigravity(evaluation, _fixed_roots()["report"])
    _emit({"experiment": _EXPERIMENT, "command": "report", "html": str(artifact.html_path), "manifest": str(artifact.manifest_path)})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("readiness", "smoke", "run", "status", "verify", "evaluate", "report"))
    arguments = parser.parse_args(argv)
    try:
        if arguments.command in {"readiness", "status"}:
            _emit({**_status(), "command": arguments.command})
            return 0
        if arguments.command == "verify":
            _emit(_verify())
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
