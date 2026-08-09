"""Freeze and replay the three Exp443 label-free policy predictions."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    canonical_jsonl_bytes_antigravity,
    exclusive_rename_antigravity,
    fsync_directory_antigravity,
    parse_canonical_json_bytes_antigravity,
    read_pinned_regular_bytes_antigravity,
    write_create_only_antigravity,
)
from .checkpoint_antigravity import (
    CompletedCheckpointInspectionAntigravity,
    Exp443CheckpointStoreAntigravity,
    ordered_work_units_antigravity,
)
from .config_antigravity import FROZEN_CONFIG, scientific_config_sha256_antigravity
from .readiness_antigravity import _source_evidence_antigravity
from .routing_antigravity import (
    PolicyPredictionAntigravity,
    canonical_prediction_bytes_antigravity,
    route_policy_antigravity,
)
from .sources_antigravity import verify_frozen_sources_antigravity


PUBLICATION_NAMES_ANTIGRAVITY = (
    "prediction_linear_edge_nearest_control_antigravity.jsonl",
    "prediction_exclude_nonfinite_row_antigravity.jsonl",
    "prediction_leave_one_out_peer_median_antigravity.jsonl",
    "policy_score_audit_antigravity.jsonl",
    "train_policy_audit_antigravity.jsonl",
    "control_gate_antigravity.json",
    "worker_parity_antigravity.json",
    "source_audit_antigravity.json",
    "backend_decision_antigravity.json",
    "manifest_antigravity.json",
    "status_antigravity.json",
    "artifact_sha256_antigravity.json",
)
_LEAF_NAMES_ANTIGRAVITY = PUBLICATION_NAMES_ANTIGRAVITY[:9]
_POLICIES_ANTIGRAVITY = tuple(FROZEN_CONFIG.policies)
_FORBIDDEN_KEYS_ANTIGRAVITY = frozenset({
    "label", "labels", "test_label", "test_labels", "ground_truth",
    "actual_anomaly", "anomaly_location", "evaluation", "evaluation_metrics",
    "confusion", "precision", "recall", "f1", "fpr", "specificity",
})


def _digest_antigravity(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _hex_antigravity(value: object, *, length: int = 64, label: str = "SHA256") -> str:
    if type(value) is not str or len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} is invalid")
    return value


def _reject_forbidden_antigravity(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("publication mapping key is invalid")
            if key.lower() in _FORBIDDEN_KEYS_ANTIGRAVITY:
                raise ValueError("publication contains TEST-label or evaluation data")
            _reject_forbidden_antigravity(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_forbidden_antigravity(item)


def _attempt_antigravity(value: Path | str) -> Path:
    attempt = Path(value)
    if "antigravity" not in attempt.name or attempt.is_symlink() or not attempt.is_dir():
        raise ValueError("Exp443 publication attempt is invalid")
    return attempt


def _canonical_file_antigravity(path: Path, label: str) -> tuple[dict[str, object], bytes, str]:
    payload, digest = read_pinned_regular_bytes_antigravity(path)
    parsed = parse_canonical_json_bytes_antigravity(payload)
    if not isinstance(parsed, dict) or canonical_json_bytes_antigravity(parsed) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    _reject_forbidden_antigravity(parsed)
    return parsed, payload, digest


def _prediction_filename_antigravity(policy: str) -> str:
    if policy not in _POLICIES_ANTIGRAVITY:
        raise ValueError("publication policy is invalid")
    return f"prediction_{policy}_antigravity.jsonl"


@dataclass(frozen=True)
class FrozenPublicationAntigravity:
    """Compact identity returned by both publication and independent replay."""

    attempt_path: str
    publication_path: str
    manifest_sha256: str
    artifact_map_sha256: str
    prediction_sha256: tuple[tuple[str, str], ...]
    runtime_prediction_sha256: tuple[tuple[str, str], ...]
    label_access_count: int = 0

    def __post_init__(self) -> None:
        if not self.attempt_path or not self.publication_path:
            raise ValueError("frozen publication path is invalid")
        _hex_antigravity(self.manifest_sha256, label="manifest SHA256")
        _hex_antigravity(self.artifact_map_sha256, label="artifact map SHA256")
        for pairs, label in ((self.prediction_sha256, "prediction"), (self.runtime_prediction_sha256, "runtime prediction")):
            if tuple(policy for policy, _ in pairs) != _POLICIES_ANTIGRAVITY:
                raise ValueError(f"{label} policy identity is invalid")
            for _policy, digest in pairs:
                _hex_antigravity(digest, label=f"{label} SHA256")
        if type(self.label_access_count) is not int or self.label_access_count != 0:
            raise ValueError("frozen publication label access is nonzero")


def _validate_progress_antigravity(attempt: Path) -> tuple[dict[str, object], dict[str, object]]:
    progress, _progress_bytes, _ = _canonical_file_antigravity(attempt / "progress_antigravity.json", "progress")
    pid, _pid_bytes, _ = _canonical_file_antigravity(attempt / "pid_antigravity.json", "PID evidence")
    if (
        progress.get("stage") != "complete"
        or progress.get("active") is not False
        or progress.get("completed") != FROZEN_CONFIG.recompute_units
        or progress.get("total") != FROZEN_CONFIG.recompute_units
        or progress.get("recomputed") != FROZEN_CONFIG.recompute_units
        or progress.get("error") != 0
    ):
        raise ValueError("publication progress is not complete")
    if type(progress.get("success")) is not int or type(progress.get("abstention")) is not int or progress["success"] + progress["abstention"] != FROZEN_CONFIG.recompute_units:
        raise ValueError("publication progress outcomes are incomplete")
    if pid.get("active") is not False or pid.get("pid") != progress.get("pid"):
        raise ValueError("publication PID evidence is not inactive")
    return progress, pid


def _validate_control_antigravity(attempt: Path) -> tuple[dict[str, object], bytes]:
    control, payload, _ = _canonical_file_antigravity(attempt / "control_gate_antigravity.json", "control gate")
    required = {
        "status", "passed", "reason", "score_evidence", "routing_identity",
        "environment_identity", "label_access_count", "max_abs_difference",
        "candidate_mismatch_count",
    }
    if set(control) != required or control.get("status") != "control_reproduction_passed" or control.get("passed") is not True:
        raise ValueError("publication control gate did not pass")
    if control.get("label_access_count") != 0 or control.get("candidate_mismatch_count") != 0:
        raise ValueError("publication control gate is not label-free or candidate-identical")
    evidence = control.get("score_evidence")
    if not isinstance(evidence, list) or len(evidence) != FROZEN_CONFIG.affected_dataset_count * len(FROZEN_CONFIG.kernel_requests) * len(FROZEN_CONFIG.seeds):
        raise ValueError("publication control score evidence is incomplete")
    observed_evidence: set[tuple[object, object, object]] = set()
    evidence_datasets: set[str] = set()
    for row in evidence:
        if not isinstance(row, Mapping):
            raise ValueError("publication control score evidence is invalid")
        if type(row.get("dataset")) is not str:
            raise ValueError("publication control score evidence dataset is invalid")
        evidence_datasets.add(row["dataset"])
        observed_evidence.add((row.get("dataset"), row.get("kernel"), row.get("seed")))
        row_difference = row.get("max_abs_difference")
        if row.get("finite") is not True or row.get("candidate_equal") is not True or type(row_difference) not in (int, float) or not math.isfinite(float(row_difference)) or float(row_difference) > 1e-6:
            raise ValueError("publication control score evidence did not pass")
    expected_evidence = {
        (dataset, kernel, seed)
        for dataset in evidence_datasets
        for kernel in FROZEN_CONFIG.kernel_requests
        for seed in FROZEN_CONFIG.seeds
    }
    if len(evidence_datasets) != FROZEN_CONFIG.affected_dataset_count or observed_evidence != expected_evidence:
        raise ValueError("publication control score evidence identity differs")
    difference = control.get("max_abs_difference")
    if type(difference) not in (int, float) or not math.isfinite(float(difference)) or float(difference) > 1e-6:
        raise ValueError("publication control score difference is invalid")
    routing = control.get("routing_identity")
    if not isinstance(routing, Mapping):
        raise ValueError("publication control routing identity is absent")
    expected = {
        "rows": 1117, "promotions": 137, "pre_exact": 2253,
        "exact_normal_removals": 35, "final_hard": 2218,
        "recomputed_rows": 19, "imported_rows": 1098,
    }
    if any(routing.get(name) != value for name, value in expected.items()):
        raise ValueError("publication control routing counts differ")
    for name in (
        "promotions_coordinate_equal", "pre_exact_coordinate_equal",
        "exact_normal_removals_coordinate_equal", "exp291_coordinate_equal",
        "exp404_coordinate_equal",
    ):
        if routing.get(name) is not True:
            raise ValueError("publication control routing coordinates differ")
    return control, payload


def _validate_parity_antigravity(attempt: Path) -> tuple[dict[str, object], bytes]:
    local = attempt / "worker_parity_antigravity.json"
    path = local if local.is_file() and not local.is_symlink() else Path(__file__).resolve().parents[2] / "scratch" / "exp443_worker_parity_report_antigravity.json"
    parity, payload, _ = _canonical_file_antigravity(path, "worker parity")
    required = {
        "schema", "experiment_id", "reviewed_commit", "policy", "unit_count",
        "workers", "logical_cpu_count", "inner_threads", "label_access_count",
        "semantic_equal", "sequential_payload_sha256", "parallel_payload_sha256",
        "control_status", "control_max_abs_difference", "prediction_bytes_sha256",
    }
    if not required.issubset(parity):
        raise ValueError("publication worker parity schema is incomplete")
    if (
        parity["schema"] != "exp443.worker_parity.v1"
        or parity["experiment_id"] != FROZEN_CONFIG.experiment_id
        or parity["policy"] != _POLICIES_ANTIGRAVITY[0]
        or parity["unit_count"] != FROZEN_CONFIG.affected_dataset_count
        or parity["workers"] != FROZEN_CONFIG.outer_workers
        or parity["logical_cpu_count"] != FROZEN_CONFIG.logical_cpu_count
        or parity["inner_threads"] != FROZEN_CONFIG.inner_threads
        or parity["label_access_count"] != 0
        or parity["semantic_equal"] is not True
        or parity["control_status"] != "control_reproduction_passed"
    ):
        raise ValueError("publication worker parity did not pass")
    sequential = _hex_antigravity(parity["sequential_payload_sha256"], label="sequential payload SHA256")
    parallel = _hex_antigravity(parity["parallel_payload_sha256"], label="parallel payload SHA256")
    if sequential != parallel:
        raise ValueError("publication worker parity semantic hashes differ")
    _hex_antigravity(parity["prediction_bytes_sha256"], label="parity prediction SHA256")
    _hex_antigravity(parity["reviewed_commit"], length=40, label="parity reviewed commit")
    return parity, payload


def _inspect_checkpoint_and_sources_antigravity(attempt: Path) -> tuple[CompletedCheckpointInspectionAntigravity, object, dict[str, object]]:
    inspection = Exp443CheckpointStoreAntigravity.inspect_complete_antigravity(attempt / "checkpoint_antigravity")
    snapshot = verify_frozen_sources_antigravity()
    source = _source_evidence_antigravity(snapshot)
    source_sha = _digest_antigravity(canonical_json_bytes_antigravity(source))
    binding = inspection.binding
    if (
        binding.run_identity != FROZEN_CONFIG.run_identity
        or binding.scientific_config_sha256 != scientific_config_sha256_antigravity()
        or binding.environment_lock_sha256 != FROZEN_CONFIG.environment_lock_sha256
        or binding.database_sha256 != FROZEN_CONFIG.database_sha256
        or binding.source_snapshot_sha256 != source_sha
    ):
        raise ValueError("publication checkpoint binding differs from frozen sources")
    _hex_antigravity(binding.reviewed_commit, length=40, label="checkpoint reviewed commit")
    affected = tuple(getattr(snapshot, "affected_datasets", ()))
    expected_units = ordered_work_units_antigravity(affected)
    if inspection.units != expected_units or len(inspection.results) != FROZEN_CONFIG.recompute_units:
        raise ValueError("publication checkpoint 57-unit identity differs")
    return inspection, snapshot, source


def _policy_results_antigravity(inspection: CompletedCheckpointInspectionAntigravity, policy: str) -> dict[str, object]:
    results = {
        unit.dataset: inspection.results[unit.sequence]
        for unit in inspection.units if unit.policy == policy
    }
    if tuple(sorted(results)) != tuple(unit.dataset for unit in inspection.units if unit.policy == policy) or len(results) != FROZEN_CONFIG.affected_dataset_count:
        raise ValueError("publication policy result identity is incomplete")
    return results


def _reroute_antigravity(inspection: CompletedCheckpointInspectionAntigravity, snapshot: object, control: Mapping[str, object], attempt: Path) -> tuple[dict[str, PolicyPredictionAntigravity], dict[str, bytes], dict[str, bytes]]:
    predictions: dict[str, PolicyPredictionAntigravity] = {}
    runtime_bytes: dict[str, bytes] = {}
    jsonl_bytes: dict[str, bytes] = {}
    for policy in _POLICIES_ANTIGRAVITY:
        prediction = route_policy_antigravity(snapshot, _policy_results_antigravity(inspection, policy), policy)
        if len(prediction.rows) != FROZEN_CONFIG.dataset_count:
            raise ValueError("publication prediction row count differs")
        canonical_runtime = canonical_prediction_bytes_antigravity(prediction)
        stored_runtime, _stored_digest = read_pinned_regular_bytes_antigravity(attempt / f"prediction_{policy}_antigravity.json")
        if stored_runtime != canonical_runtime:
            raise ValueError("publication runtime prediction differs from checkpoint replay")
        payload = prediction.to_payload()
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != FROZEN_CONFIG.dataset_count:
            raise ValueError("publication prediction rows are invalid")
        published = canonical_jsonl_bytes_antigravity(rows)
        if len(published.splitlines()) != FROZEN_CONFIG.dataset_count:
            raise ValueError("publication prediction JSONL row count differs")
        predictions[policy] = prediction
        runtime_bytes[policy] = canonical_runtime
        jsonl_bytes[policy] = published
    control_routing = control["routing_identity"]
    policy_a = predictions[_POLICIES_ANTIGRAVITY[0]]
    observed = {
        "rows": len(policy_a.rows),
        "promotions": len(policy_a.promotions),
        "pre_exact": len(policy_a.pre_exact_coordinates),
        "exact_normal_removals": len(policy_a.exact_normal_removals),
        "final_hard": len(policy_a.final_hard_coordinates),
    }
    if not isinstance(control_routing, Mapping) or any(control_routing.get(name) != value for name, value in observed.items()):
        raise ValueError("publication Policy A prediction disagrees with control routing identity")
    return predictions, runtime_bytes, jsonl_bytes


def _audit_files_antigravity(inspection: CompletedCheckpointInspectionAntigravity) -> tuple[bytes, bytes]:
    score_rows: list[dict[str, object]] = []
    train_rows: list[dict[str, object]] = []
    for unit in inspection.units:
        result = inspection.results[unit.sequence]
        result_payload = result.to_payload()
        kernel_payloads = result_payload["kernel_scores"]
        audit_payloads = result_payload["audits"]
        action_counts = Counter(item["action"] for item in audit_payloads)
        score_rows.append({
            "sequence": unit.sequence,
            "policy": unit.policy,
            "dataset": unit.dataset,
            "status": result.status,
            "target_length": result.target_length,
            "train_total": result.train_total,
            "train_missing": result.train_missing,
            "train_used": result.train_used,
            "train_excluded": result.train_excluded,
            "test_rows": result.test_rows,
            "candidates_by_kernel": result_payload["candidates_by_kernel"],
            "kernel_scores_sha256": _digest_antigravity(canonical_json_bytes_antigravity(kernel_payloads)),
            "result_sha256": _digest_antigravity(canonical_json_bytes_antigravity(result_payload)),
            "record_sha256": inspection.record_sha256_prefix[unit.sequence],
        })
        train_rows.append({
            "sequence": unit.sequence,
            "policy": unit.policy,
            "dataset": unit.dataset,
            "train_total": result.train_total,
            "train_missing": result.train_missing,
            "train_used": result.train_used,
            "train_excluded": result.train_excluded,
            "audit_count": len(audit_payloads),
            "action_counts": dict(sorted(action_counts.items())),
            "detailed_audit_sha256": _digest_antigravity(canonical_json_bytes_antigravity(audit_payloads)),
            "record_sha256": inspection.record_sha256_prefix[unit.sequence],
        })
    return canonical_jsonl_bytes_antigravity(score_rows), canonical_jsonl_bytes_antigravity(train_rows)


def _build_publication_files_antigravity(attempt: Path) -> tuple[dict[str, bytes], FrozenPublicationAntigravity]:
    _validate_progress_antigravity(attempt)
    inspection, snapshot, source = _inspect_checkpoint_and_sources_antigravity(attempt)
    control, control_bytes = _validate_control_antigravity(attempt)
    if tuple(sorted({row["dataset"] for row in control["score_evidence"]})) != tuple(getattr(snapshot, "affected_datasets", ())):
        raise ValueError("publication control datasets differ from the frozen affected inventory")
    parity, parity_bytes = _validate_parity_antigravity(attempt)
    predictions, runtime_bytes, published_bytes = _reroute_antigravity(inspection, snapshot, control, attempt)
    if parity["prediction_bytes_sha256"] != _digest_antigravity(runtime_bytes[_POLICIES_ANTIGRAVITY[0]]):
        raise ValueError("publication worker parity prediction differs from Policy A replay")
    score_bytes, train_bytes = _audit_files_antigravity(inspection)

    source_audit = {
        "schema": "exp443.source-audit.v1",
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "reviewed_commit": inspection.binding.reviewed_commit,
        "scientific_config_sha256": inspection.binding.scientific_config_sha256,
        "environment_lock_sha256": inspection.binding.environment_lock_sha256,
        "database_sha256": inspection.binding.database_sha256,
        "source_snapshot_sha256": inspection.binding.source_snapshot_sha256,
        "source": source,
        "label_access_count": 0,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    environment = control["environment_identity"]
    backend = {
        "schema": "exp443.backend-decision.v1",
        "backend": "cpu",
        "outer_workers": FROZEN_CONFIG.outer_workers,
        "inner_threads": FROZEN_CONFIG.inner_threads,
        "logical_cpu_count": FROZEN_CONFIG.logical_cpu_count,
        "acceleration_decision": FROZEN_CONFIG.acceleration_decision,
        "environment": environment,
        "worker_parity_semantic_equal": parity["semantic_equal"],
        "label_access_count": 0,
    }
    leaf_files: dict[str, bytes] = {
        _prediction_filename_antigravity(policy): published_bytes[policy]
        for policy in _POLICIES_ANTIGRAVITY
    }
    leaf_files.update({
        "policy_score_audit_antigravity.jsonl": score_bytes,
        "train_policy_audit_antigravity.jsonl": train_bytes,
        "control_gate_antigravity.json": control_bytes,
        "worker_parity_antigravity.json": parity_bytes,
        "source_audit_antigravity.json": canonical_json_bytes_antigravity(source_audit),
        "backend_decision_antigravity.json": canonical_json_bytes_antigravity(backend),
    })
    if tuple(leaf_files) != _LEAF_NAMES_ANTIGRAVITY:
        raise ValueError("publication leaf inventory is invalid")
    leaf_hashes = {name: _digest_antigravity(payload) for name, payload in leaf_files.items()}
    runtime_hashes = {policy: _digest_antigravity(runtime_bytes[policy]) for policy in _POLICIES_ANTIGRAVITY}
    published_hashes = {policy: leaf_hashes[_prediction_filename_antigravity(policy)] for policy in _POLICIES_ANTIGRAVITY}
    route_counts = {
        policy: {
            "rows": len(predictions[policy].rows),
            "promotions": len(predictions[policy].promotions),
            "pre_exact": len(predictions[policy].pre_exact_coordinates),
            "exact_normal_removals": len(predictions[policy].exact_normal_removals),
            "final_hard": len(predictions[policy].final_hard_coordinates),
        }
        for policy in _POLICIES_ANTIGRAVITY
    }
    manifest = {
        "schema": "exp443.publication-manifest.v1",
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "run_identity": FROZEN_CONFIG.run_identity,
        "reviewed_commit": inspection.binding.reviewed_commit,
        "policies": list(_POLICIES_ANTIGRAVITY),
        "dataset_count": FROZEN_CONFIG.dataset_count,
        "affected_dataset_count": FROZEN_CONFIG.affected_dataset_count,
        "result_count": FROZEN_CONFIG.recompute_units,
        "checkpoint_record_prefix_sha256": _digest_antigravity(canonical_json_bytes_antigravity(list(inspection.record_sha256_prefix))),
        "leaf_sha256": leaf_hashes,
        "runtime_prediction_sha256": runtime_hashes,
        "prediction_jsonl_sha256": published_hashes,
        "prediction_row_count": {policy: FROZEN_CONFIG.dataset_count for policy in _POLICIES_ANTIGRAVITY},
        "routing_counts": route_counts,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "label_access_count": 0,
        "review_only": True,
        "operational_alert_eligible": False,
    }
    manifest_bytes = canonical_json_bytes_antigravity(manifest)
    status = {
        "schema": "exp443.publication-status.v1",
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "stage": "complete_review_only",
        "manifest_sha256": _digest_antigravity(manifest_bytes),
        "prediction_sha256": published_hashes,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "label_access_count": 0,
        "review_only": True,
        "operational_alert_eligible": False,
    }
    status_bytes = canonical_json_bytes_antigravity(status)
    files = dict(leaf_files)
    files["manifest_antigravity.json"] = manifest_bytes
    files["status_antigravity.json"] = status_bytes
    artifact_map = {
        "schema": "exp443.artifact-map.v1",
        "files": {name: _digest_antigravity(files[name]) for name in PUBLICATION_NAMES_ANTIGRAVITY[:-1]},
        "self_excluded": "artifact_sha256_antigravity.json",
    }
    artifact_bytes = canonical_json_bytes_antigravity(artifact_map)
    files["artifact_sha256_antigravity.json"] = artifact_bytes
    if tuple(files) != PUBLICATION_NAMES_ANTIGRAVITY:
        raise ValueError("publication inventory construction is invalid")
    frozen = FrozenPublicationAntigravity(
        str(attempt),
        str(attempt / "publication_antigravity"),
        _digest_antigravity(manifest_bytes),
        _digest_antigravity(artifact_bytes),
        tuple((policy, published_hashes[policy]) for policy in _POLICIES_ANTIGRAVITY),
        tuple((policy, runtime_hashes[policy]) for policy in _POLICIES_ANTIGRAVITY),
        0,
    )
    return files, frozen


def _read_publication_inventory_antigravity(root: Path) -> dict[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("publication root is invalid")
    entries = {entry.name: entry for entry in root.iterdir()}
    if set(entries) != set(PUBLICATION_NAMES_ANTIGRAVITY):
        raise ValueError("publication artifact inventory is invalid")
    payloads: dict[str, bytes] = {}
    for name in PUBLICATION_NAMES_ANTIGRAVITY:
        payloads[name], _ = read_pinned_regular_bytes_antigravity(entries[name])
    return payloads


def inspect_publication_antigravity(attempt: Path | str) -> FrozenPublicationAntigravity:
    """Recompute every label-free byte and compare it with the frozen files."""
    attempt_path = _attempt_antigravity(attempt)
    expected, frozen = _build_publication_files_antigravity(attempt_path)
    actual = _read_publication_inventory_antigravity(attempt_path / "publication_antigravity")
    for name in PUBLICATION_NAMES_ANTIGRAVITY:
        if actual[name] != expected[name]:
            raise ValueError(f"publication artifact bytes changed: {name}")
    return frozen


def publish_predictions_antigravity(attempt: Path | str) -> FrozenPublicationAntigravity:
    """Create one no-clobber publication, or replay an existing publication."""
    attempt_path = _attempt_antigravity(attempt)
    destination = attempt_path / "publication_antigravity"
    if destination.exists() or destination.is_symlink():
        return inspect_publication_antigravity(attempt_path)
    files, _frozen = _build_publication_files_antigravity(attempt_path)
    staging = attempt_path / f".publication_{uuid4().hex}_antigravity"
    staging.mkdir(mode=0o700)
    try:
        for name in PUBLICATION_NAMES_ANTIGRAVITY:
            write_create_only_antigravity(staging / name, files[name])
        fsync_directory_antigravity(staging)
        staged = _read_publication_inventory_antigravity(staging)
        if any(staged[name] != files[name] for name in PUBLICATION_NAMES_ANTIGRAVITY):
            raise ValueError("publication staging bytes differ")
        exclusive_rename_antigravity(staging, destination)
        fsync_directory_antigravity(attempt_path)
    finally:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
    return inspect_publication_antigravity(attempt_path)


__all__ = [
    "FrozenPublicationAntigravity",
    "PUBLICATION_NAMES_ANTIGRAVITY",
    "inspect_publication_antigravity",
    "publish_predictions_antigravity",
]
