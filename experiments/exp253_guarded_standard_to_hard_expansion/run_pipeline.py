from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from .config import EXTERNAL_STATUS_PATH, EXPERIMENT_ID, HEARTBEAT_PATH, OUTPUT_ROOT, SOURCE
from .evaluation import evaluate
from .independent_reaggregate import reaggregate
from .inputs import audit_inputs, load_frozen_inputs
from .prediction import predict_all
from .reporting import write_manifest, write_report


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_status(output_root: Path, status: str, phase: str, done: int, total: int,
                 current: str | None = None, errors: int = 0) -> None:
    now = time.time()
    payload = {
        "id": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "label": "Exp253 - guarded Standard-to-Hard expansion",
        "source": SOURCE,
        "status": status,
        "phase": phase,
        "pid": os.getpid(),
        "child_pid": os.getpid(),
        "done": int(done),
        "datasets_done": int(done),
        "expected": int(total),
        "expected_datasets": int(total),
        "progress_percent": 100.0 * done / max(1, total),
        "current_dataset": current,
        "errors": int(errors),
        "updated_unix": now,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "retrospective_guarded_alert_expansion",
        "comparison_eligible": False,
        "contains_test_labels": phase in {"evaluate", "completed"},
        "selected_from_prior_test_review": True,
        "deployment_eligible": False,
        "detail_csv": str(output_root / "results.csv"),
        "stdout_log": str(output_root / "full.log"),
    }
    _atomic_json(output_root / "status.json", payload)
    _atomic_json(EXTERNAL_STATUS_PATH, payload)
    _atomic_json(HEARTBEAT_PATH, payload)


def _names(limit: int | None) -> tuple[str, ...]:
    names = tuple(sorted(load_frozen_inputs()))
    return names[:limit] if limit else names


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    names = _names(args.dataset_limit)
    total = len(names)
    write_status(output_root, "running", args.stage, 0, total)
    if args.stage in {"audit", "all"}:
        audit_inputs(output_root)
        write_status(output_root, "running", "audit", total, total, names[-1] if names else None)
    if args.stage in {"predict", "all"}:
        write_status(output_root, "running", "predict", 0, total)
        predict_all(output_root, names=names)
        write_status(output_root, "running", "predict", total, total, names[-1] if names else None)
    if args.stage in {"evaluate", "all"}:
        write_status(output_root, "running", "evaluate", 0, total)
        evaluate(output_root)
        result = reaggregate(output_root)
        if not result["passed"]:
            raise RuntimeError("independent reaggregation failed")
        write_report(output_root)
        write_manifest(output_root, " ".join(map(str, sys.argv)))
        write_status(output_root, "completed", "completed", total, total)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--stage", choices=("audit", "predict", "evaluate", "all"), default="all")
    value.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    value.add_argument("--dataset-limit", type=int)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
