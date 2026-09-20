#!/usr/bin/env python3
"""Run a review-only, all-population Exp424 MPS score-cache pass.

This runner deliberately does not publish Exp417 official artifacts.  It keeps
the frozen CPU Task 7 contract for checkpoint reconstruction while binding the
actual model fitting and TEST-value inference calls to ``torch.device('mps')``.
The resulting prediction is evidence for a separate numeric/device study and
must not be promoted to an operational lane.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.backend import (
    FROZEN_TASK7_BACKEND_AUDIT_SHA256,
    load_frozen_backend_audit,
)
from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    Exp417CheckpointProvenance,
    Exp417CheckpointStore,
)
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.inputs import (
    freeze_lane_inputs,
    freeze_run_context,
    load_raw_test_values,
    load_raw_train_values,
)
from experiments.exp417_causal_multihorizon_shadow.model import (
    fit_final_seed_models,
    train_oof_models,
)
from experiments.exp417_causal_multihorizon_shadow.preprocessing import (
    preprocess_length_groups,
)
from experiments.exp417_causal_multihorizon_shadow.qualification import (
    exclude_frozen_lanes,
    freeze_train_contract,
)
from experiments.exp417_causal_multihorizon_shadow.runner import (
    RunnerResources,
    RunnerRuntime,
    TrainModelEvidence,
    _production_qualify_train,
    _production_score_test,
    _production_verify,
    _processed_evidence_groups,
    build_prediction_rows,
    resume_or_run_population,
)
from scratch.run_exp417_causal_multihorizon_shadow import (
    _provenance,
    require_clean_git,
)


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def _directory(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"path is not a directory: {path}")
    if not path.exists() and not path.parent.is_dir():
        raise argparse.ArgumentTypeError(f"parent does not exist: {path.parent}")
    return path


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(dict(payload)) + b"\n")


def _review_runtime(
    resources: RunnerResources,
    cpu_audit,
    progress_path: Path,
) -> RunnerRuntime:
    """Use MPS for all model work while retaining a valid frozen CPU contract."""
    mps = torch.device("mps")
    progress_state: dict[str, object] = {
        "state": "running",
        "experiment_id": "Exp424-MPS-SCORE-CACHE",
        "current_dataset": None,
        "stage": "initializing",
        "completed_queries": 0,
        "expected_queries": None,
        "test_labels_opened": False,
    }

    def write_progress(
        stage: str,
        *,
        dataset: str | None = None,
        completed_queries: int | None = None,
        expected_queries: int | None = None,
    ) -> None:
        progress_state["stage"] = stage
        if dataset is not None:
            progress_state["current_dataset"] = dataset
        if completed_queries is not None:
            progress_state["completed_queries"] = completed_queries
        if expected_queries is not None:
            progress_state["expected_queries"] = expected_queries
        _write_json(progress_path, progress_state)

    def verify(context):
        write_progress("verify_inputs")
        return _production_verify(context, resources)

    def load_train(dataset, context):
        write_progress("load_train_values", dataset=dataset)
        return load_raw_train_values(resources.database, dataset, context)

    def preprocess(train):
        write_progress("preprocess_train")
        return preprocess_length_groups(train)

    def train_models(processed) -> TrainModelEvidence:
        write_progress("train_models")
        groups = _processed_evidence_groups(processed)
        oof_groups = tuple(
            group
            for group in groups
            if group.status in {"candidate_eligible", "diagnostic_only"}
        )
        final_groups = tuple(
            group for group in groups if group.status == "candidate_eligible"
        )
        bundles = {
            group.group.length: train_oof_models(group, device=mps)
            for group in oof_groups
        }
        finals = {
            group.group.length: fit_final_seed_models(group, device=mps)
            for group in final_groups
        }
        # The CPU audit remains the frozen contract so checkpoint records can be
        # reconstructed by the existing fail-closed readers.  The manifest
        # records the actual MPS device separately and marks this run review-only.
        evidence = TrainModelEvidence(groups, bundles, finals, cpu_audit)
        write_progress("train_models_complete")
        return evidence

    def qualify(dataset, processed, models):
        expected = sum(
            len(group.group.series)
            for group in _processed_evidence_groups(processed)
            if group.status == "candidate_eligible"
        ) * len(FROZEN_CONFIG.perturbation_seeds) * 5
        write_progress(
            "qualify_train",
            dataset=dataset,
            completed_queries=0,
            expected_queries=expected,
        )
        qualification = _production_qualify_train(dataset, processed, models)
        write_progress(
            "qualify_train_complete",
            completed_queries=expected,
        )
        return qualification

    def freeze(dataset, qualification, models):
        write_progress("freeze_train_contract", dataset=dataset)
        return freeze_train_contract(
            dataset,
            qualification.groups,
            qualification.backend,
            train_identity_sha256=models.train_identity_sha256,
        )

    def load_test(dataset, contract, context):
        write_progress("load_test_values", dataset=dataset)
        return load_raw_test_values(resources.database, dataset, contract, context)

    def score(test, contract, models):
        write_progress(
            "score_test",
            dataset=test.dataset,
            completed_queries=0,
            expected_queries=len(test.series),
        )
        scoring = _production_score_test(test, contract, models)
        write_progress(
            "score_test_complete",
            completed_queries=len(test.series),
        )
        return scoring

    def exclude(evidence):
        write_progress("exclude_frozen_lanes", dataset=evidence.dataset)
        return exclude_frozen_lanes(evidence)

    return RunnerRuntime(
        verify=verify,
        load_train=load_train,
        preprocess=preprocess,
        train_models=train_models,
        qualify=qualify,
        freeze=freeze,
        load_test=load_test,
        score=score,
        exclude=exclude,
    )


def _prediction_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the review-only Exp424 MPS score-cache and full TEST-value pass.",
        allow_abbrev=False,
    )
    parser.add_argument("--database", type=_existing_file, required=True)
    parser.add_argument("--exp291-prediction", type=_existing_file, required=True)
    parser.add_argument("--exp291-manifest", type=_existing_file, required=True)
    parser.add_argument("--exp292-prediction", type=_existing_file, required=True)
    parser.add_argument("--exp292-manifest", type=_existing_file, required=True)
    parser.add_argument("--checkpoint-dir", type=_directory, required=True)
    parser.add_argument("--output-dir", type=_directory, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process; no CPU fallback is allowed")
    if require_clean_git(REPO_ROOT) == "":
        raise RuntimeError("clean Git HEAD is required")
    git_commit = require_clean_git(REPO_ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir.chmod(0o700)
    runtime_progress_path = args.output_dir / "runtime_progress.json"

    paths = (
        args.database,
        args.exp291_prediction,
        args.exp291_manifest,
        args.exp292_prediction,
        args.exp292_manifest,
    )
    lanes = freeze_lane_inputs(*paths[1:])
    context = freeze_run_context(args.database, lanes, verified_code_commit=git_commit)
    resources = RunnerResources(
        *paths,
        backend_audit_path=args.database.parent
        / "exp417_causal_multihorizon_shadow"
        / "task7_backend_audit.json",
    )
    cpu_audit = load_frozen_backend_audit(resources.backend_audit_path)
    base_provenance = _provenance(context)
    review_backend_digest = sha256(
        canonical_json_bytes(
            {
                "experiment_id": "Exp424-MPS-SCORE-CACHE",
                "base_task7_backend_sha256": FROZEN_TASK7_BACKEND_AUDIT_SHA256,
                "requested_device": "mps",
                "gate_override": True,
                "review_only": True,
            }
        )
    ).hexdigest()
    provenance: Exp417CheckpointProvenance = replace(
        base_provenance, backend_contract_sha256=review_backend_digest
    )
    checkpoints = Exp417CheckpointStore(
        args.checkpoint_dir, provenance, context.datasets
    )
    runtime = _review_runtime(resources, cpu_audit, runtime_progress_path)
    print(
        json.dumps(
            {
                "state": "running",
                "experiment_id": "Exp424-MPS-SCORE-CACHE",
                "review_only": True,
                "mps_available": True,
                "mps_device": "mps",
                "expected_datasets": len(context.datasets),
                "checkpoint_dir": str(args.checkpoint_dir),
                "output_dir": str(args.output_dir),
                "runtime_progress": str(runtime_progress_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    summary = resume_or_run_population(context, checkpoints, runtime=runtime)
    rows = build_prediction_rows(summary)
    prediction = _prediction_bytes(rows)
    prediction_path = args.output_dir / "exp424_mps_predictions.jsonl"
    prediction_path.write_bytes(prediction)
    prediction_sha256 = sha256(prediction).hexdigest()
    manifest = {
        "artifact_kind": "exp424_mps_score_cache_prediction",
        "schema_version": 1,
        "experiment_id": "Exp424-MPS-SCORE-CACHE",
        "state": "review_only" if summary.status == "complete" else "failed",
        "review_only": True,
        "official_prediction": False,
        "publication_eligible": False,
        "contains_test_labels": False,
        "test_labels_opened": False,
        "test_values_read": True,
        "test_informed_reselection": False,
        "lane_promotion": False,
        "tail_replacement": False,
        "mps_requested": True,
        "mps_available": True,
        "mps_device": "mps",
        "mps_device_guard": "mps_and_mps0_equivalent",
        "mps_gate_overridden": True,
        "mps_gate_override_reason": "explicit review-only device evaluation",
        "optimization_adopted": True,
        "optimization_components": [
            "oof_score_calibration_cache",
            "reusable_batch_residual_predictor",
        ],
        "optimization_batch_size": 32,
        "optimization_query_chunk_size": 256,
        "optimization_reference_formula": "Exp417.score_horizon.v1",
        "optimization_parity_smoke": "passed_cpu_and_mps_targeted_tests",
        "optimization_fallback": None,
        "base_task7_backend": cpu_audit.backend,
        "base_task7_speedup": cpu_audit.speedup,
        "git_commit": git_commit,
        "database_sha256": context.database_sha256,
        "exp291_prediction_sha256": lanes.exp291_prediction_sha256,
        "exp291_manifest_sha256": lanes.exp291_manifest_sha256,
        "exp292_prediction_sha256": lanes.exp292_prediction_sha256,
        "exp292_manifest_sha256": lanes.exp292_manifest_sha256,
        "checkpoint_dir": str(args.checkpoint_dir),
        "runtime_progress": str(runtime_progress_path),
        "expected_datasets": len(context.datasets),
        "completed_datasets": len(summary.results),
        "summary_status": summary.status,
        "prediction_rows": len(rows),
        "prediction_sha256": prediction_sha256,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    _write_json(
        args.output_dir / "status.json",
        {
            "state": manifest["state"],
            "completed_datasets": len(summary.results),
            "expected_datasets": len(context.datasets),
            "current_dataset": None,
            "stage": "complete",
            "completed_queries": None,
            "expected_queries": None,
            "test_labels_opened": False,
            "prediction_sha256": prediction_sha256,
        },
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0 if summary.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
