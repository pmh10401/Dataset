from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.exp393_396.artifact_io import load_test_labels

from experiments.exp400_402 import BATCH_ID
from experiments.exp400_402.artifact_io import load_inputs, write_prediction
from experiments.exp400_402.evaluation import evaluate_prediction
from experiments.exp400_402.independent_reaggregate import (
    run_independent_reaggregate,
)
from experiments.exp400_402.prediction import build_shadow_predictions
from experiments.exp400_402.reporting import write_reports


DEFAULT_OUTPUT = Path(
    "/Users/minho/Documents/Dataset/exp400_402_exp397_shadow_validation"
)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _status(
    output_root: Path,
    *,
    phase: str,
    status: str,
    completed: int,
    total: int,
) -> None:
    _atomic_json(
        output_root / "status.json",
        {
            "experiment_id": BATCH_ID,
            "phase": phase,
            "status": status,
            "completed": completed,
            "total": total,
            "progress": completed / total if total else 0.0,
            "updated_at_epoch": time.time(),
        },
    )


def _select_datasets(
    exp292_rows: Sequence[Mapping[str, object]],
    exp397_rows: Sequence[Mapping[str, object]],
    dataset_limit: int | None,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    names = sorted(str(row["dataset"]) for row in exp292_rows)
    if dataset_limit is not None:
        names = names[: max(0, int(dataset_limit))]
    selected = set(names)
    return (
        [row for row in exp292_rows if str(row["dataset"]) in selected],
        [row for row in exp397_rows if str(row["dataset"]) in selected],
    )


def run_prediction(
    output_root: Path,
    *,
    dataset_limit: int | None = None,
    input_loader: Callable[
        [],
        tuple[
            list[dict[str, Any]],
            list[dict[str, Any]],
            dict[str, Any],
        ],
    ] = load_inputs,
    full_run: bool | None = None,
) -> dict[str, object]:
    exp292_rows, exp397_rows, input_hashes = input_loader()
    exp292_selected, exp397_selected = _select_datasets(
        exp292_rows,
        exp397_rows,
        dataset_limit,
    )
    total = len(exp292_selected)
    _status(
        output_root,
        phase="predict",
        status="running",
        completed=0,
        total=total,
    )
    rows = build_shadow_predictions(exp292_selected, exp397_selected)
    manifest = write_prediction(
        output_root,
        rows,
        full_run=(dataset_limit is None if full_run is None else full_run),
        input_hashes=input_hashes,
    )
    _status(
        output_root,
        phase="predict",
        status="completed",
        completed=total,
        total=total,
    )
    return manifest


def run_pipeline(
    stage: str,
    output_root: Path = DEFAULT_OUTPUT,
    *,
    dataset_limit: int | None = None,
    input_loader: Callable[
        [],
        tuple[
            list[dict[str, Any]],
            list[dict[str, Any]],
            dict[str, Any],
        ],
    ] = load_inputs,
    label_loader: Callable[
        [set[str]],
        Mapping[str, Sequence[object]],
    ] = load_test_labels,
    full_run: bool | None = None,
) -> dict[str, object]:
    if stage not in {"predict", "evaluate", "audit", "all"}:
        raise ValueError(f"unsupported stage: {stage}")
    output_root = Path(output_root)
    result: dict[str, object] = {}
    if stage in {"predict", "all"}:
        result["predict"] = run_prediction(
            output_root,
            dataset_limit=dataset_limit,
            input_loader=input_loader,
            full_run=full_run,
        )
    if stage in {"evaluate", "all"}:
        _status(
            output_root,
            phase="evaluate",
            status="running",
            completed=0,
            total=1,
        )
        result["evaluate"] = evaluate_prediction(
            output_root,
            label_loader=label_loader,
        )
        _status(
            output_root,
            phase="evaluate",
            status="completed",
            completed=1,
            total=1,
        )
    if stage in {"audit", "all"}:
        _status(
            output_root,
            phase="audit",
            status="running",
            completed=0,
            total=2,
        )
        result["independent"] = run_independent_reaggregate(output_root)
        _status(
            output_root,
            phase="audit",
            status="running",
            completed=1,
            total=2,
        )
        result["reports"] = write_reports(output_root)
        _status(
            output_root,
            phase="audit",
            status="completed",
            completed=2,
            total=2,
        )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run Exp400-402 Exp397 Shadow validation."
    )
    result.add_argument(
        "--stage",
        choices=("predict", "evaluate", "audit", "all"),
        default="all",
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--dataset-limit", type=int)
    return result


def main() -> int:
    args = parser().parse_args()
    result = run_pipeline(
        args.stage,
        args.output_dir,
        dataset_limit=args.dataset_limit,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
