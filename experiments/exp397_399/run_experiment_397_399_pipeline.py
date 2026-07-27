from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp397_399.artifact_io import (  # noqa: E402
    EXP292_MANIFEST,
    EXP292_PREDICTION,
    EXP393_MANIFEST,
    EXP393_PREDICTION,
    load_exp292_rows,
    load_exp393_rows,
    sha256_file,
    write_ranking_artifact,
)
from experiments.exp397_399.contracts import (  # noqa: E402
    ALPHAS,
    EXPECTED_DATASETS,
    EXPECTED_STANDARD_CANDIDATES,
    VARIANTS,
    WORKLOADS,
)
from experiments.exp397_399.evaluation import run_evaluation  # noqa: E402
from experiments.exp397_399.independent_reaggregate_exp397_399 import (  # noqa: E402
    run_independent_reaggregate,
)
from experiments.exp397_399.prediction import build_prediction  # noqa: E402
from experiments.exp397_399.reporting import write_reports  # noqa: E402


DEFAULT_OUTPUT = Path(
    "/Users/minho/Documents/Dataset/exp397_399_review_ranking"
)


def _write_status(
    output_root: Path,
    *,
    phase: str,
    completed: int,
    total: int,
    status: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": "experiment_397_399_review_ranking",
        "phase": phase,
        "completed": int(completed),
        "total": int(total),
        "progress": completed / total if total else 0.0,
        "status": status,
        "updated_at_epoch": time.time(),
    }
    (output_root / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_prediction(
    output_root: Path,
    *,
    dataset_limit: int | None = None,
) -> dict[str, Any]:
    exp292_rows, exp292_manifest = load_exp292_rows()
    exp393_rows, exp393_manifest = load_exp393_rows()
    names = sorted({str(row["dataset"]) for row in exp292_rows})
    if dataset_limit is not None:
        names = names[: int(dataset_limit)]
    selected = set(names)
    exp292_rows = [
        row for row in exp292_rows if str(row["dataset"]) in selected
    ]
    exp393_rows = [
        row for row in exp393_rows if str(row["dataset"]) in selected
    ]
    _write_status(
        output_root,
        phase="predict",
        completed=0,
        total=len(names),
        status="running",
    )
    ranking_rows = build_prediction(exp292_rows, exp393_rows)
    standard_count = sum(
        len(set(map(int, row.get("standard_review_indices", []))))
        for row in exp292_rows
    )
    hard_count = sum(
        len(set(map(int, row.get("automatic_hard_indices", []))))
        for row in exp292_rows
    )
    priority_count = sum(
        len(set(map(int, row.get("priority_review_indices", []))))
        for row in exp292_rows
    )
    if dataset_limit is None:
        if len(names) != EXPECTED_DATASETS:
            raise ValueError("Exp397 full dataset coverage mismatch")
        if standard_count != EXPECTED_STANDARD_CANDIDATES:
            raise ValueError("Exp397 standard review universe mismatch")
        if hard_count != 2218 or priority_count != 31:
            raise ValueError("Exp397 frozen lane count mismatch")
    prediction_path, manifest_path = write_ranking_artifact(
        output_root,
        ranking_rows,
        {
            "datasets": len(names),
            "dataset_names": names,
            "alphas": list(ALPHAS),
            "variants": list(VARIANTS),
            "workloads": list(WORKLOADS),
            "standard_review_candidates": standard_count,
            "automatic_hard_count": hard_count,
            "priority_review_count": priority_count,
            "exp292_prediction_sha256": sha256_file(EXP292_PREDICTION),
            "exp292_manifest_sha256": sha256_file(EXP292_MANIFEST),
            "exp393_prediction_sha256": sha256_file(EXP393_PREDICTION),
            "exp393_manifest_sha256": sha256_file(EXP393_MANIFEST),
            "exp292_manifest_prediction_sha256": exp292_manifest[
                "prediction_sha256"
            ],
            "exp393_manifest_prediction_sha256": exp393_manifest[
                "prediction_sha256"
            ],
        },
    )
    _write_status(
        output_root,
        phase="predict",
        completed=len(names),
        total=len(names),
        status="completed",
    )
    return {
        "prediction_path": str(prediction_path),
        "manifest_path": str(manifest_path),
        "datasets": len(names),
        "standard_review_candidates": standard_count,
    }


def run_pipeline(
    stage: str,
    output_root: Path,
    *,
    dataset_limit: int | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    result: dict[str, Any] = {}
    if stage in {"predict", "all"}:
        result["predict"] = run_prediction(
            output_root,
            dataset_limit=dataset_limit,
        )
    if stage in {"evaluate", "all"}:
        _write_status(
            output_root,
            phase="evaluate",
            completed=0,
            total=1,
            status="running",
        )
        result["evaluate"] = run_evaluation(output_root)
        _write_status(
            output_root,
            phase="evaluate",
            completed=1,
            total=1,
            status="completed",
        )
    if stage in {"audit", "all"}:
        _write_status(
            output_root,
            phase="audit",
            completed=0,
            total=1,
            status="running",
        )
        result["audit"] = write_reports(output_root)
        result["independent_reaggregate"] = run_independent_reaggregate(
            output_root
        )
        if not result["independent_reaggregate"]["consistent"]:
            raise ValueError("independent reaggregation mismatch")
        _write_status(
            output_root,
            phase="completed",
            completed=1,
            total=1,
            status="completed",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("predict", "evaluate", "audit", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-limit", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            run_pipeline(
                args.stage,
                args.output_dir,
                dataset_limit=args.dataset_limit,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
