#!/usr/bin/env python3
"""Run exactly one preregistered, TRAIN-only Exp425 native MPS parity smoke."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.inputs import freeze_lane_inputs
from experiments.exp417_causal_multihorizon_shadow.runner import _groups_digest, _oof_calibration_digest, _series_digest
from experiments.exp425_selective_recovery import capabilities
from experiments.exp425_selective_recovery.checkpoints import Exp425CheckpointStore
from experiments.exp425_selective_recovery.contracts import Exp425RunProvenance
from experiments.exp425_selective_recovery.database import Exp425DatabaseSnapshot, open_database_snapshot
from experiments.exp425_selective_recovery.mps_parity import (
    DATABASE_SHA256, EXPECTED_LEGACY_OOF, EXPECTED_MODEL_HASHES, FIXTURE,
    GOLDEN_SOURCE_COMMIT, SmokeEvidenceWriter, compare_shared_science,
    authenticated_cpu_audit, review_only_mps_assessment, validate_golden_reference,
)
from experiments.exp425_selective_recovery.runtime import AtomicProgressWriter, _compute_native, build_exp425_runtime


DATABASE = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
SOURCE_ROOT = Path("/Users/mac/Documents/Dataset/exp424_mps_score_cache_20260802-153831.checkpoint")
GOLDEN = SOURCE_ROOT / "datasets/0a19fece1acc72bd1fd0e2845ee962e54988940a96da37ba5c231fa52e551ce8.json"
LANE_PATHS = (
    Path("/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl"),
    Path("/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json"),
    Path("/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl"),
    Path("/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json"),
)


def _sha(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _assert_clean_heavy_process() -> None:
    listing = subprocess.run(["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True).stdout
    foreign = [line.strip() for line in listing.splitlines() if "exp42" in line.lower() and "mps" in line.lower() and str(os.getpid()) not in line]
    if foreign:
        raise ValueError("competing Exp42x MPS-heavy process is active")


class _CapturingModels(dict[str, object]):
    """Observe (but do not alter) evidence when the native seam removes it."""
    captured: object | None = None

    def pop(self, key: str, default: object = None) -> object:  # type: ignore[override]
        value = super().pop(key, default)
        if value is not default and value is not None:
            self.captured = value
        return value


def _golden_shared(record: object) -> dict[str, object]:
    assert isinstance(record, dict)
    qualification = record["qualification"]
    groups = [
        {"length": group.length, "series_count": group.series_count, "status": group.status,
         "oof_candidate_rate": group.oof_candidate_rate, "wilson_upper": group.wilson_upper,
         "pooled_synthetic_sensitivity": group.pooled_synthetic_sensitivity,
         "per_type_synthetic_sensitivity": dict(group.per_type_synthetic_sensitivity),
         "seed_agreement": group.seed_agreement, "finite": group.finite,
         "backend_fallback": group.backend_fallback}
        for group in qualification.groups
    ]
    return {
        "dataset": record["dataset"], "database_sha256": DATABASE_SHA256,
        "raw_train_sha256": record["raw_train_identity_sha256"],
        "repaired_train_sha256": record["repaired_train_identity_sha256"],
        "length_identity_sha256": record["length_group_identity_sha256"],
        "config": {"alpha": 0.01, "model_seeds": [41701, 41702, 41703], "perturbation_seeds": [41711, 41712, 41713], "batch_size": 32, "query_chunk": 256},
        "series_count": 106, "lengths": [96], "final_model_hashes": record["canonical_model_hashes"],
        "legacy_oof_digest": record["score_calibration_hashes"], "qualification": groups,
        "status": qualification.status, "candidate_coordinates": [[item.dataset, item.instance_index] for item in qualification.candidates],
        "finiteness": all(group["finite"] for group in groups), "fallback": False,
    }


def _actual_shared(native: object, models: object, train: object) -> dict[str, object]:
    contract = native.model_hashes["frozen_train_contract"]  # type: ignore[attr-defined]
    model_evidence = native.model_hashes["model_evidence"]  # type: ignore[attr-defined]
    finals: dict[str, dict[str, str]] = {}
    for item in model_evidence["final_models"]:
        finals.setdefault(str(item["length"]), {})[str(item["seed"])] = item["model_sha256"]
    processed = models.processed  # type: ignore[union-attr]
    return {
        "dataset": train.dataset, "database_sha256": DATABASE_SHA256,
        "raw_train_sha256": _series_digest("exp417.raw-train.v1", train.dataset, train.series),
        "repaired_train_sha256": _groups_digest("exp417.repaired-train.v1", train.dataset, processed),
        "length_identity_sha256": {str(group.group.length): _series_digest("exp417.repaired-group-members.v1", train.dataset, group.group.series) for group in processed},
        "config": {"alpha": FROZEN_CONFIG.candidate_alpha, "model_seeds": list(FROZEN_CONFIG.model_seeds), "perturbation_seeds": list(FROZEN_CONFIG.perturbation_seeds), "batch_size": 32, "query_chunk": 256},
        "series_count": len(train.series), "lengths": [group.group.length for group in processed],
        "final_model_hashes": finals,
        "legacy_oof_digest": {str(length): _oof_calibration_digest(bundle) for length, bundle in models.oof_bundles.items()},  # type: ignore[union-attr]
        "qualification": contract["groups"],
        "status": native.result_payload["effective_result"]["status"],  # type: ignore[attr-defined]
        "candidate_coordinates": native.result_payload["effective_result"]["candidate_coordinates"],  # type: ignore[attr-defined]
        "finiteness": all(group["finite"] for group in contract["groups"]), "fallback": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_dir.absolute()
    if not output.name.startswith("exp425_task5_mps_parity_") or output.parent != DATABASE.parent:
        raise ValueError("output must be a new exp425_task5_mps_parity_* directory beneath Dataset")
    writer = SmokeEvidenceWriter.create(output, "exp425-task5-" + str(os.getpid()), FIXTURE)
    runtime_writer: AtomicProgressWriter | None = None
    session = None
    try:
        import torch
        if torch.__version__.split("+", 1)[0] != "2.13.0" or not torch.backends.mps.is_built() or not torch.backends.mps.is_available() or torch.device("mps").type != "mps":
            raise ValueError("actual MPS built/available/device gate failed")
        _assert_clean_heavy_process()
        current_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        if subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout or len(current_head) != 40:
            raise ValueError("clean current Exp425 commit is required")
        golden = validate_golden_reference(ROOT, SOURCE_ROOT, GOLDEN)
        writer.progress(stage="preflight", current_dataset=None, success=None, backend="mps", torch=torch.__version__, mps_built=True, mps_available=True, completed=0, total=1)
        lanes = freeze_lane_inputs(*LANE_PATHS)
        with open_database_snapshot(DATABASE, expected_sha256=DATABASE_SHA256) as snapshot:
            context = snapshot.freeze_context(lanes, verified_code_commit=current_head)
            provenance = Exp425RunProvenance(
                run_identity=writer.run_identity, git_commit=current_head, database_sha256=DATABASE_SHA256,
                lane_artifact_sha256={"hard": lanes.exp291_prediction_sha256, "standard": lanes.exp292_prediction_sha256, "priority": lanes.exp292_prediction_sha256, "automatic_normal": lanes.exp291_prediction_sha256},
                frozen_config_sha256=_sha(asdict(FROZEN_CONFIG)), fold_contract_sha256=_sha({"runner": golden.source_hashes["experiments/exp417_causal_multihorizon_shadow/runner.py"]}),
                model_contract_sha256=_sha(golden.source_hashes), backend_contract_sha256=_sha({"review_only": True, "cpu_audit": asdict(authenticated_cpu_audit()), "adoption": review_only_mps_assessment().payload()}),
                model_seeds=FROZEN_CONFIG.model_seeds, perturbation_seeds=FROZEN_CONFIG.perturbation_seeds,
                runtime_versions=context.runtime_versions, expected_datasets=context.datasets,
                predecessor_compatibility_sha256=_sha({"golden": golden.record_sha256, "payload": golden.record["payload_sha256"]}),
            )
            checkpoint_root = output / "checkpoint"
            store = Exp425CheckpointStore.create(checkpoint_root, provenance, context)
            runtime_writer = AtomicProgressWriter.create(output_root=output / "runtime", run_identity=writer.run_identity, command="new", checkpoint_root=checkpoint_root, total_datasets=len(context.datasets), backend="mps")
            audit, adoption = authenticated_cpu_audit(), review_only_mps_assessment()
            session = build_exp425_runtime(snapshot, audit, adoption, runtime_writer, current_head)
            session._fold_contract_sha256 = provenance.fold_contract_sha256
            counters = {"train": 0, "test_values": 0, "test_labels": 0}
            original_train = Exp425DatabaseSnapshot.load_train
            def counted_train(instance, dataset, frozen):
                counters["train"] += 1
                if dataset != FIXTURE or counters["train"] != 1: raise ValueError("TRAIN loader gate failed")
                return original_train(instance, dataset, frozen)
            def forbidden_test(*_args, **_kwargs):
                counters["test_values"] += 1
                raise ValueError("TEST value loader is forbidden")
            captured = _CapturingModels(); session._models = captured
            with patch.object(Exp425DatabaseSnapshot, "load_train", counted_train), patch.object(Exp425DatabaseSnapshot, "load_test", forbidden_test):
                writer.progress(stage="native_train_qualification", current_dataset=FIXTURE, success=None, completed=0, total=1, loader_counts=counters)
                train = snapshot.load_train(FIXTURE, context)
                started = perf_counter(); native, trace = _compute_native(context, store, session, FIXTURE, train); wall = perf_counter() - started
            models = captured.captured
            if models is None or counters != {"train": 1, "test_values": 0, "test_labels": 0}:
                raise ValueError("loader or model-capture gate failed")
            actual = _actual_shared(native, models, train)
            if actual["final_model_hashes"] != EXPECTED_MODEL_HASHES or actual["legacy_oof_digest"] != EXPECTED_LEGACY_OOF:
                raise ValueError("exact model or legacy OOF gate failed")
            parity = compare_shared_science(_golden_shared(dict(golden.record)), actual, tolerance=0.0001)
            if not parity["passed"] or actual["status"] != "abstained" or actual["candidate_coordinates"] != []:
                raise ValueError("shared parity/status/candidate gate failed")
            capability = capabilities._mint(session, kind="native", run_identity=store.provenance.run_identity, dataset=FIXTURE, context_sha256=store.run_context_sha256, trace_sha256=native.trace_sha256, record_sha256=native.record_payload_sha256, backend_sha256=native.backend_audit_sha256, process_nonce=session.process_nonce)
            checkpoint_digest = store._commit_native(capability, native)
            result = {"schema": "exp425.task5.mps-parity-result.v1", "success": True, "fixture": FIXTURE, "stage": "complete", "backend": "mps", "fallback": False, "loader_counts": counters, "wall_seconds": wall, "timer_scope": "_compute_native return after production implicit CPU/NumPy/serialization/calibration synchronization", "explicit_mps_synchronize_calls": 0, "torch": torch.__version__, "device": str(torch.device("mps")), "golden": {"record_sha256": golden.record_sha256, "payload_sha256": golden.record["payload_sha256"], "source_hashes": dict(golden.source_hashes)}, "cpu_scientific_audit": asdict(audit), "mps_adoption": adoption.payload(), "actual": actual, "parity": parity, "native_trace": asdict(trace), "native_record_sha256": native.record_payload_sha256, "checkpoint_record_sha256": checkpoint_digest, "checkpoint_path": str(checkpoint_root), "limitation": "Exp424 retained no per-synthetic-query scores; direct historical per-query deltas are unavailable."}
            writer.result(result); writer.progress(stage="complete", current_dataset=None, success=True, completed=1, total=1, loader_counts=counters, wall_seconds=wall, checkpoint=str(checkpoint_root)); writer._append("parent.log", b"complete\n")
        return 0
    except BaseException as error:
        writer.failure(error)
        raise
    finally:
        if session is not None: session.close()
        if runtime_writer is not None: runtime_writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
