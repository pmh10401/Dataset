"""Orchestrate the modular Exp153--157 TRAIN-only improvement pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import exp153_157_core as core
import build_exp153_157_score_cache as cache_builder
import run_experiment_153_train_normal_calibration_audit as exp153
import run_experiment_154_pooled_conformal as exp154
import run_experiment_155_source_fusion as exp155
import run_experiment_156_confirmation_calibration as exp156
import run_experiment_157_risk_controlled_alpha_policy as exp157
import independent_reaggregate_exp153_157 as independent

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/exp153_157_train_only_improvements"


def write_output_manifest(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or "cache" in path.parts:
            continue
        if path.name in {"MANIFEST.csv", "MANIFEST.md"}:
            continue
        rows.append(
            {
                "file": str(path.relative_to(output_dir)),
                "sha256": core.sha256_file(path),
                "retrospective_or_prospective": "retrospective_counterfactual",
                "autonomous_or_human_assisted": "hard_autonomous_review_human_assisted",
            }
        )
    with (output_dir / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["file", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "MANIFEST.md").write_text(
        "# Exp153--157 output manifest\n\n"
        "No TEST metric selected alpha, pooling, fusion, confirmation, or risk target.\n\n"
        + "\n".join(f"- `{row['file']}`: `{row['sha256']}`" for row in rows)
        + "\n",
        encoding="utf-8",
    )


def run_stages(
    stages,
    *,
    output_dir: Path,
    dataset_limit: int | None,
    workers: int,
    reset_cache: bool,
    skip_db_hash: bool,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        print(f"\n=== stage {stage} ===", flush=True)
        if stage == "cache":
            cache_builder.build_cache(
                output_dir,
                dataset_limit=dataset_limit,
                workers=workers,
                reset=reset_cache,
                skip_db_hash=skip_db_hash,
            )
        elif stage == "153":
            exp153.run(output_dir)
        elif stage == "154-predict":
            exp154.run_prediction(output_dir)
        elif stage == "154-evaluate":
            exp154.run_evaluation(output_dir)
        elif stage == "155-predict":
            exp155.run_prediction(output_dir)
        elif stage == "155-evaluate":
            exp155.run_evaluation(output_dir)
        elif stage == "156-predict":
            exp156.run_prediction(output_dir)
        elif stage == "156-evaluate":
            exp156.run_evaluation(output_dir)
        elif stage == "157-predict":
            exp157.run_prediction(output_dir)
        elif stage == "157-evaluate":
            exp157.run_evaluation(output_dir)
        elif stage == "independent":
            independent.run(output_dir)
        elif stage == "manifest":
            write_output_manifest(output_dir)
        else:
            raise ValueError(f"unknown stage: {stage}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        action="append",
        choices=(
            "cache",
            "153",
            "154-predict",
            "154-evaluate",
            "155-predict",
            "155-evaluate",
            "156-predict",
            "156-evaluate",
            "157-predict",
            "157-evaluate",
            "independent",
            "manifest",
            "all",
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7")))
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--skip-db-hash", action="store_true")
    args = parser.parse_args()
    full = [
        "cache",
        "153",
        "154-predict",
        "154-evaluate",
        "155-predict",
        "155-evaluate",
        "156-predict",
        "156-evaluate",
        "157-predict",
        "157-evaluate",
        "independent",
        "manifest",
    ]
    stages = args.stage or ["all"]
    if "all" in stages:
        stages = full
    run_stages(
        stages,
        output_dir=Path(args.output_dir),
        dataset_limit=args.dataset_limit,
        workers=args.workers,
        reset_cache=args.reset_cache,
        skip_db_hash=args.skip_db_hash,
    )


if __name__ == "__main__":
    main()
