"""Create-only, label-free publication and replay inspection for Exp441."""
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import stat
from uuid import uuid4
from typing import Mapping

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity, canonical_jsonl_bytes_antigravity,
    exclusive_rename_antigravity, fsync_directory_antigravity,
    parse_canonical_json_bytes_antigravity, read_pinned_regular_bytes_antigravity,
    write_create_only_antigravity,
)
from .config_antigravity import FROZEN_CONFIG, governance_payload_antigravity
from .contracts_antigravity import CompletedAttemptAntigravity, FrozenPublicationAntigravity, SelectionResultAntigravity
from .runtime_antigravity import _selection_from_payload_antigravity, _snapshot_payload_antigravity
from .selector_antigravity import select_hard_empty_single_shot_antigravity, selected_coordinate_identity_sha256_antigravity
from .sources_antigravity import verify_frozen_sources_antigravity


_RUN_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808")
_PUBLICATION_NAMES_ANTIGRAVITY = (
    "prediction_antigravity.jsonl", "candidate_ranking_antigravity.jsonl",
    "source_audit_antigravity.json", "manifest_antigravity.json",
    "status_antigravity.json", "artifact_sha256_antigravity.json",
)
_GOVERNANCE_NAMES_ANTIGRAVITY = frozenset(governance_payload_antigravity())
_FORBIDDEN_ROW_KEYS_ANTIGRAVITY = frozenset({"label", "labels", "actual_anomaly", "anomaly_location", "confusion", "f1", "evaluation", "metrics"})


def _sha256_antigravity(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _require_lower_hex_antigravity(value: object, length: int, message: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(item not in "0123456789abcdef" for item in value):
        raise ValueError(message)
    return value


def _require_regular_inventory_antigravity(root: Path) -> dict[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("publication root is invalid")
    entries = {entry.name: entry for entry in root.iterdir()}
    if set(entries) != set(_PUBLICATION_NAMES_ANTIGRAVITY):
        raise ValueError("publication inventory is invalid")
    result: dict[str, bytes] = {}
    for name, path in entries.items():
        if path.is_symlink() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
            raise ValueError("publication entry is not a regular file")
        payload, _digest = read_pinned_regular_bytes_antigravity(path)
        if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) != 0o600:
            raise ValueError("publication file mode is invalid")
        result[name] = payload
    return result


def _parse_jsonl_antigravity(payload: bytes, name: str) -> list[dict[str, object]]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{name} JSONL is invalid")
    rows: list[dict[str, object]] = []
    for line in payload.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError(f"{name} JSONL is invalid")
        row = parse_canonical_json_bytes_antigravity(line)
        if not isinstance(row, dict):
            raise ValueError(f"{name} row is invalid")
        rows.append(row)
    return rows


def _forbid_governance_or_evaluation_antigravity(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or key in _GOVERNANCE_NAMES_ANTIGRAVITY or key in _FORBIDDEN_ROW_KEYS_ANTIGRAVITY:
                raise ValueError("label-free artifact contains a forbidden key")
            _forbid_governance_or_evaluation_antigravity(item)
    elif isinstance(value, list):
        for item in value:
            _forbid_governance_or_evaluation_antigravity(item)


def _exact_governance_antigravity(payload: Mapping[str, object]) -> None:
    expected = governance_payload_antigravity()
    if {name: payload.get(name) for name in expected} != expected:
        raise ValueError("publication governance is invalid")
    if any(type(payload[name]) is not type(value) for name, value in expected.items()):
        raise ValueError("publication governance types are invalid")
    if any(name in payload for name in ("governance", "governance_payload", "review_status")):
        raise ValueError("publication governance must be flat")


def _attempt_path_antigravity(attempt: CompletedAttemptAntigravity) -> Path:
    if not isinstance(attempt, CompletedAttemptAntigravity):
        raise ValueError("completed attempt is invalid")
    path = _RUN_ROOT_ANTIGRAVITY / "attempts_antigravity" / f"attempt_{attempt.attempt_sequence:04d}_antigravity"
    if path.is_symlink() or not path.is_dir():
        raise ValueError("attempt root is invalid")
    return path


def _parse_attempt_antigravity(attempt: CompletedAttemptAntigravity) -> tuple[Path, dict[str, object], dict[str, object]]:
    path = _attempt_path_antigravity(attempt)
    required = {"source_snapshot_antigravity.json", "selection_antigravity.json", "progress_antigravity.json", "pid_antigravity"}
    entries = {entry.name: entry for entry in path.iterdir()}
    if not required <= set(entries):
        raise ValueError("attempt evidence is incomplete")
    values: dict[str, object] = {}
    for name in required:
        item = entries[name]
        if item.is_symlink() or not stat.S_ISREG(item.stat(follow_symlinks=False).st_mode):
            raise ValueError("attempt evidence is invalid")
        values[name] = parse_canonical_json_bytes_antigravity(read_pinned_regular_bytes_antigravity(item)[0])
    snapshot = values["source_snapshot_antigravity.json"]
    selection = values["selection_antigravity.json"]
    progress = values["progress_antigravity.json"]
    pid = values["pid_antigravity"]
    if not all(isinstance(item, dict) for item in (snapshot, selection, progress, pid)):
        raise ValueError("attempt JSON is invalid")
    if _sha256_antigravity(canonical_json_bytes_antigravity(snapshot)) != attempt.source_snapshot_sha256:
        raise ValueError("attempt snapshot hash is invalid")
    if _sha256_antigravity(canonical_json_bytes_antigravity(selection)) != attempt.selection_sha256:
        raise ValueError("attempt selection hash is invalid")
    if (progress.get("stage"), progress.get("active"), progress.get("completed"), progress.get("total"), progress.get("error")) != ("complete", False, 1117, 1117, 0):
        raise ValueError("attempt progress is not terminal")
    if (progress.get("provisional_candidates"), progress.get("outside_hard_candidates"), progress.get("eligible_candidates"), progress.get("selected_candidates")) != (840, 394, 73, 30):
        raise ValueError("attempt selection counts are invalid")
    if pid.get("active") is not False or type(pid.get("pid")) is not int:
        raise ValueError("attempt PID remains active")
    if selection.get("reviewed_commit") != attempt.reviewed_commit or selection.get("source_snapshot_sha256") != attempt.source_snapshot_sha256:
        raise ValueError("attempt identity is inconsistent")
    required_selection = set(SelectionResultAntigravity.__dataclass_fields__) | {"reviewed_commit", "source_snapshot_sha256"} | _GOVERNANCE_NAMES_ANTIGRAVITY
    if set(selection) != required_selection:
        raise ValueError("attempt selection schema is invalid")
    _exact_governance_antigravity(selection)
    return path, snapshot, selection


def _reverified_selection_antigravity(attempt: CompletedAttemptAntigravity, snapshot_payload: Mapping[str, object], selection_payload: Mapping[str, object]) -> SelectionResultAntigravity:
    snapshot = verify_frozen_sources_antigravity()
    if snapshot.label_access_count != 0 or _snapshot_payload_antigravity(snapshot) != snapshot_payload:
        raise ValueError("frozen source authority changed")
    expected = select_hard_empty_single_shot_antigravity(snapshot)
    if expected.to_payload() != {name: selection_payload[name] for name in SelectionResultAntigravity.__dataclass_fields__}:
        raise ValueError("attempt selection is not the frozen selector result")
    reopened = _selection_from_payload_antigravity(selection_payload)
    if reopened != expected or reopened != attempt.selection:
        raise ValueError("attempt selection is inconsistent")
    return expected


def _publication_payloads_antigravity(attempt: CompletedAttemptAntigravity, snapshot_payload: Mapping[str, object], selection: SelectionResultAntigravity) -> dict[str, bytes]:
    prediction_rows = [item.to_payload() for item in sorted(selection.selected, key=lambda item: item.coordinate)]
    ranking_rows = [item.to_payload() for item in selection.winners]
    source_audit = dict(snapshot_payload)
    _forbid_governance_or_evaluation_antigravity(prediction_rows)
    _forbid_governance_or_evaluation_antigravity(ranking_rows)
    _forbid_governance_or_evaluation_antigravity(source_audit)
    prediction = canonical_jsonl_bytes_antigravity(prediction_rows)
    ranking = canonical_jsonl_bytes_antigravity(ranking_rows)
    audit = canonical_json_bytes_antigravity(source_audit)
    manifest: dict[str, object] = {
        "schema": "exp441.publication.v1", "experiment_id": FROZEN_CONFIG.experiment_id,
        "run_identity": FROZEN_CONFIG.run_identity, "reviewed_commit": attempt.reviewed_commit,
        "source_snapshot_sha256": attempt.source_snapshot_sha256, "selection_sha256": attempt.selection_sha256,
        "dataset_count": 1117, "provisional_candidate_count": 840, "outside_hard_candidate_count": 394,
        "eligible_candidate_count": 73, "winner_count": 49, "candidate_count": 30,
        "candidate_budget": 30, "candidate_per_dataset": 1, "label_access_count": 0,
        "contains_test_labels": False, "uses_frozen_test_candidate_empty_gate": True,
        "uses_test_labels_for_prediction": False,
        "prediction_sha256": _sha256_antigravity(prediction),
        "candidate_ranking_sha256": _sha256_antigravity(ranking), "source_audit_sha256": _sha256_antigravity(audit),
        "selected_coordinate_identity_sha256": FROZEN_CONFIG.selected_coordinate_identity_sha256,
        "hard_coordinate_identity_sha256": FROZEN_CONFIG.hard_coordinate_identity_sha256,
    }
    manifest.update(governance_payload_antigravity())
    manifest_bytes = canonical_json_bytes_antigravity(manifest)
    status: dict[str, object] = {
        "schema": "exp441.status.v1", "stage": "publication_complete", "reviewed_commit": attempt.reviewed_commit,
        "manifest_sha256": _sha256_antigravity(manifest_bytes), "prediction_sha256": _sha256_antigravity(prediction),
        "completed_retrospective_only": True, "label_access_count": 0, "contains_test_labels": False,
    }
    status.update(governance_payload_antigravity())
    status_bytes = canonical_json_bytes_antigravity(status)
    files = {
        "prediction_antigravity.jsonl": prediction,
        "candidate_ranking_antigravity.jsonl": ranking,
        "source_audit_antigravity.json": audit,
        "manifest_antigravity.json": manifest_bytes,
        "status_antigravity.json": status_bytes,
    }
    artifact_map = {name: _sha256_antigravity(payload) for name, payload in files.items()}
    files["artifact_sha256_antigravity.json"] = canonical_json_bytes_antigravity({
        "schema": "exp441.artifact-map.v1", "files": artifact_map,
        "self_excluded": "artifact_sha256_antigravity.json",
    })
    return files


def _failure_payload_antigravity(attempt: CompletedAttemptAntigravity, error: BaseException) -> dict[str, object]:
    payload: dict[str, object] = {
        "attempt_sequence": attempt.attempt_sequence, "reviewed_commit": attempt.reviewed_commit,
        "status": "failed", "error": str(error) or type(error).__name__,
    }
    payload.update(governance_payload_antigravity())
    return payload


def _validate_failure_payload_antigravity(payload: object) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {"attempt_sequence", "reviewed_commit", "status", "error"} | _GOVERNANCE_NAMES_ANTIGRAVITY:
        raise ValueError("failed staging evidence schema is invalid")
    if type(payload["attempt_sequence"]) is not int or payload["attempt_sequence"] < 0 or payload["status"] != "failed" or not isinstance(payload["error"], str) or not payload["error"]:
        raise ValueError("failed staging evidence is invalid")
    _require_lower_hex_antigravity(payload["reviewed_commit"], 40, "failed staging reviewed commit is invalid")
    _exact_governance_antigravity(payload)


def _preserve_failed_staging_antigravity(staging: Path, attempt: CompletedAttemptAntigravity, error: BaseException) -> None:
    try:
        if not staging.exists():
            return
        failure = _failure_payload_antigravity(attempt, error)
        _validate_failure_payload_antigravity(failure)
        write_create_only_antigravity(staging / "failure_antigravity.json", canonical_json_bytes_antigravity(failure))
        failed = staging.with_name(f"failed_staging_{uuid4().hex}_antigravity")
        os.rename(staging, failed)
        fsync_directory_antigravity(failed.parent)
    except BaseException:
        pass


def _inspect_payloads_antigravity(payloads: Mapping[str, bytes], attempt: CompletedAttemptAntigravity | None = None) -> FrozenPublicationAntigravity:
    prediction_rows = _parse_jsonl_antigravity(payloads["prediction_antigravity.jsonl"], "prediction")
    ranking_rows = _parse_jsonl_antigravity(payloads["candidate_ranking_antigravity.jsonl"], "ranking")
    audit = parse_canonical_json_bytes_antigravity(payloads["source_audit_antigravity.json"])
    manifest = parse_canonical_json_bytes_antigravity(payloads["manifest_antigravity.json"])
    status = parse_canonical_json_bytes_antigravity(payloads["status_antigravity.json"])
    artifact = parse_canonical_json_bytes_antigravity(payloads["artifact_sha256_antigravity.json"])
    if not all(isinstance(item, dict) for item in (audit, manifest, status, artifact)):
        raise ValueError("publication JSON schema is invalid")
    _forbid_governance_or_evaluation_antigravity(prediction_rows)
    _forbid_governance_or_evaluation_antigravity(ranking_rows)
    _forbid_governance_or_evaluation_antigravity(audit)
    _exact_governance_antigravity(manifest)
    _exact_governance_antigravity(status)
    manifest_required = {
        "schema", "experiment_id", "run_identity", "reviewed_commit", "source_snapshot_sha256", "selection_sha256",
        "dataset_count", "provisional_candidate_count", "outside_hard_candidate_count", "eligible_candidate_count",
        "winner_count", "candidate_count", "candidate_budget", "candidate_per_dataset", "label_access_count",
        "contains_test_labels", "uses_frozen_test_candidate_empty_gate", "uses_test_labels_for_prediction",
        "prediction_sha256", "candidate_ranking_sha256", "source_audit_sha256", "selected_coordinate_identity_sha256",
        "hard_coordinate_identity_sha256",
    } | _GOVERNANCE_NAMES_ANTIGRAVITY
    status_required = {
        "schema", "stage", "reviewed_commit", "manifest_sha256", "prediction_sha256",
        "completed_retrospective_only", "label_access_count", "contains_test_labels",
    } | _GOVERNANCE_NAMES_ANTIGRAVITY
    if set(manifest) != manifest_required or set(status) != status_required:
        raise ValueError("publication governing schema is invalid")
    if (manifest.get("schema"), manifest.get("experiment_id"), manifest.get("run_identity"),
            manifest.get("dataset_count"), manifest.get("provisional_candidate_count"),
            manifest.get("outside_hard_candidate_count"), manifest.get("eligible_candidate_count"),
            manifest.get("winner_count"), manifest.get("candidate_count"), manifest.get("candidate_budget"),
            manifest.get("candidate_per_dataset"), manifest.get("uses_frozen_test_candidate_empty_gate")) != (
                "exp441.publication.v1", FROZEN_CONFIG.experiment_id, FROZEN_CONFIG.run_identity,
                1117, 840, 394, 73, 49, 30, 30, 1, True):
        raise ValueError("publication manifest counts are invalid")
    if (status.get("schema"), status.get("stage"), status.get("completed_retrospective_only")) != ("exp441.status.v1", "publication_complete", True):
        raise ValueError("publication status is invalid")
    if set(artifact) != {"schema", "files", "self_excluded"} or artifact.get("schema") != "exp441.artifact-map.v1" or artifact.get("self_excluded") != "artifact_sha256_antigravity.json" or not isinstance(artifact.get("files"), dict):
        raise ValueError("publication artifact map is invalid")
    _forbid_governance_or_evaluation_antigravity(artifact)
    if set(artifact["files"]) != set(_PUBLICATION_NAMES_ANTIGRAVITY) - {"artifact_sha256_antigravity.json"}:
        raise ValueError("publication artifact map inventory is invalid")
    for name, digest in artifact["files"].items():
        if _require_lower_hex_antigravity(digest, 64, "publication artifact map hash is invalid") != _sha256_antigravity(payloads[name]):
            raise ValueError("publication artifact hash is invalid")
    if (manifest.get("prediction_sha256"), manifest.get("candidate_ranking_sha256"), manifest.get("source_audit_sha256")) != (_sha256_antigravity(payloads["prediction_antigravity.jsonl"]), _sha256_antigravity(payloads["candidate_ranking_antigravity.jsonl"]), _sha256_antigravity(payloads["source_audit_antigravity.json"])):
        raise ValueError("publication manifest hash graph is invalid")
    if (status.get("manifest_sha256"), status.get("prediction_sha256"), status.get("label_access_count"), status.get("contains_test_labels")) != (_sha256_antigravity(payloads["manifest_antigravity.json"]), _sha256_antigravity(payloads["prediction_antigravity.jsonl"]), 0, False):
        raise ValueError("publication status hash graph is invalid")
    if manifest.get("label_access_count") != 0 or manifest.get("contains_test_labels") is not False or manifest.get("uses_test_labels_for_prediction") is not False:
        raise ValueError("publication label firewall is invalid")
    snapshot = verify_frozen_sources_antigravity()
    expected = select_hard_empty_single_shot_antigravity(snapshot)
    if snapshot.label_access_count != 0 or _snapshot_payload_antigravity(snapshot) != audit:
        raise ValueError("publication source authority is invalid")
    source_sha256 = _sha256_antigravity(canonical_json_bytes_antigravity(audit))
    selection_sha256 = _sha256_antigravity(canonical_json_bytes_antigravity({
        **expected.to_payload(), "reviewed_commit": manifest.get("reviewed_commit"),
        "source_snapshot_sha256": source_sha256, **governance_payload_antigravity(),
    }))
    if (manifest.get("source_snapshot_sha256"), manifest.get("selection_sha256"),
            manifest.get("reviewed_commit"), manifest.get("selected_coordinate_identity_sha256"),
            manifest.get("hard_coordinate_identity_sha256")) != (
                source_sha256, selection_sha256, _require_lower_hex_antigravity(manifest.get("reviewed_commit"), 40, "publication reviewed commit is invalid"),
                FROZEN_CONFIG.selected_coordinate_identity_sha256, FROZEN_CONFIG.hard_coordinate_identity_sha256):
        raise ValueError("publication immutable identity is invalid")
    if attempt is not None and expected != attempt.selection:
        raise ValueError("publication attempt selection is invalid")
    if selected_coordinate_identity_sha256_antigravity(expected.selected) != FROZEN_CONFIG.selected_coordinate_identity_sha256:
        raise ValueError("publication selected coordinate identity is invalid")
    expected_prediction = [item.to_payload() for item in sorted(expected.selected, key=lambda item: item.coordinate)]
    expected_ranking = [item.to_payload() for item in expected.winners]
    if prediction_rows != expected_prediction or ranking_rows != expected_ranking:
        raise ValueError("publication selection payload is invalid")
    if (len(prediction_rows), len(ranking_rows), len({(row["dataset"], row["instance_index"]) for row in prediction_rows})) != (30, 49, 30):
        raise ValueError("publication selection invariants are invalid")
    return FrozenPublicationAntigravity(_sha256_antigravity(payloads["artifact_sha256_antigravity.json"]), _sha256_antigravity(payloads["prediction_antigravity.jsonl"]), _sha256_antigravity(payloads["candidate_ranking_antigravity.jsonl"]), _sha256_antigravity(payloads["source_audit_antigravity.json"]), expected)


def inspect_exp441_publication_antigravity() -> FrozenPublicationAntigravity:
    """Read and verify the one fixed publication root without labels or SQLite."""
    return _inspect_payloads_antigravity(_require_regular_inventory_antigravity(_RUN_ROOT_ANTIGRAVITY / "publication_antigravity"))


def publish_exp441_antigravity(attempt: CompletedAttemptAntigravity) -> FrozenPublicationAntigravity:
    """Freeze one terminal attempt through private staging and Darwin no-clobber rename."""
    destination = _RUN_ROOT_ANTIGRAVITY / "publication_antigravity"
    if destination.exists() or destination.is_symlink():
        return inspect_exp441_publication_antigravity()
    _attempt_path, snapshot_payload, selection_payload = _parse_attempt_antigravity(attempt)
    selection = _reverified_selection_antigravity(attempt, snapshot_payload, selection_payload)
    staging = destination.with_name(f".publication_{uuid4().hex}_staging_antigravity")
    staging.mkdir(mode=0o700)
    os.chmod(staging, 0o700)
    try:
        payloads = _publication_payloads_antigravity(attempt, snapshot_payload, selection)
        for name in _PUBLICATION_NAMES_ANTIGRAVITY:
            write_create_only_antigravity(staging / name, payloads[name])
        staged = _require_regular_inventory_antigravity(staging)
        if staged != payloads:
            raise ValueError("staged publication bytes are invalid")
        _inspect_payloads_antigravity(staged, attempt)
        # Last source replay closes the staging-to-rename mutation window.
        if _reverified_selection_antigravity(attempt, snapshot_payload, selection_payload) != selection:
            raise ValueError("frozen sources changed before publication rename")
        if destination.exists() or destination.is_symlink():
            raise ValueError("publication destination raced")
        exclusive_rename_antigravity(staging, destination)
        fsync_directory_antigravity(destination.parent)
        return inspect_exp441_publication_antigravity()
    except BaseException as error:
        _preserve_failed_staging_antigravity(staging, attempt, error)
        raise


__all__ = ["publish_exp441_antigravity", "inspect_exp441_publication_antigravity"]
