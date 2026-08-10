"""Create-only, label-free Exp444 stability-lane publication and inspection.

The checkpoint is the prediction authority.  Publishing may re-open only the
already approved values-only source boundary to reproduce the original-control
replay; inspection deliberately never imports that boundary (or any runtime or
evaluation entrypoint), opens no database, and mutates no attempt authority.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    canonical_jsonl_bytes_antigravity,
    exclusive_rename_antigravity,
    parse_canonical_json_bytes_antigravity,
)
from .config_antigravity import FROZEN_CONFIG, scientific_config_sha256_antigravity
from .contracts_antigravity import (
    BackendDecisionAntigravity,
    CandidateInputAntigravity,
    CandidateStabilityResultAntigravity,
    CheckpointContextAntigravity,
    SourceCandidateAntigravity,
    StabilitySelectionAntigravity,
    validate_prediction_payload_antigravity,
)


PUBLICATION_DIRECTORY_ANTIGRAVITY = "publication_antigravity"
DATASET_SUMMARY_FILE_ANTIGRAVITY = "dataset_summary_antigravity.jsonl"
SOURCE_CANDIDATES_FILE_ANTIGRAVITY = "source_candidates_antigravity.jsonl"
CANDIDATE_STABILITY_FILE_ANTIGRAVITY = "candidate_stability_antigravity.jsonl"
VIEW_EVIDENCE_FILE_ANTIGRAVITY = "view_evidence_antigravity.jsonl"
STABLE_REVIEW_FILE_ANTIGRAVITY = "stable_review_prediction_antigravity.jsonl"
UNSTABLE_REVIEW_FILE_ANTIGRAVITY = "unstable_review_prediction_antigravity.jsonl"
CONTROL_REPLAY_FILE_ANTIGRAVITY = "control_replay_antigravity.json"
SOURCE_AUDIT_FILE_ANTIGRAVITY = "source_audit_antigravity.json"
BACKEND_DECISION_FILE_ANTIGRAVITY = "backend_decision_antigravity.json"
RUNTIME_FILE_ANTIGRAVITY = "runtime_antigravity.json"
MANIFEST_FILE_ANTIGRAVITY = "manifest_antigravity.json"
STATUS_FILE_ANTIGRAVITY = "status_antigravity.json"
ARTIFACT_SHA256_FILE_ANTIGRAVITY = "artifact_sha256_antigravity.json"

_LEAF_FILES_ANTIGRAVITY = (
    DATASET_SUMMARY_FILE_ANTIGRAVITY,
    SOURCE_CANDIDATES_FILE_ANTIGRAVITY,
    CANDIDATE_STABILITY_FILE_ANTIGRAVITY,
    VIEW_EVIDENCE_FILE_ANTIGRAVITY,
    STABLE_REVIEW_FILE_ANTIGRAVITY,
    UNSTABLE_REVIEW_FILE_ANTIGRAVITY,
    CONTROL_REPLAY_FILE_ANTIGRAVITY,
    SOURCE_AUDIT_FILE_ANTIGRAVITY,
    BACKEND_DECISION_FILE_ANTIGRAVITY,
    RUNTIME_FILE_ANTIGRAVITY,
)
_PUBLICATION_FILES_ANTIGRAVITY = frozenset((*_LEAF_FILES_ANTIGRAVITY, MANIFEST_FILE_ANTIGRAVITY, STATUS_FILE_ANTIGRAVITY, ARTIFACT_SHA256_FILE_ANTIGRAVITY))
_RECORD_NAMES_ANTIGRAVITY = tuple(f"{index:04d}_record_antigravity.json" for index in range(FROZEN_CONFIG.source_candidate_count))


def _sha256_antigravity(value: bytes) -> str:
    return sha256(value).hexdigest()


def _read_canonical_json_antigravity(path: Path, name: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Exp444 {name} is absent or invalid")
    encoded = path.read_bytes()
    try:
        payload = parse_canonical_json_bytes_antigravity(encoded)
    except ValueError as error:
        raise ValueError(f"Exp444 {name} is invalid") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes_antigravity(payload) != encoded:
        raise ValueError(f"Exp444 {name} is not canonical")
    return dict(payload)


def _read_canonical_jsonl_antigravity(path: Path, name: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Exp444 {name} is absent or invalid")
    encoded = path.read_bytes()
    if not encoded:
        return ()
    if not encoded.endswith(b"\n"):
        raise ValueError(f"Exp444 {name} is not canonical")
    rows: list[dict[str, object]] = []
    for line in encoded.splitlines():
        try:
            value = parse_canonical_json_bytes_antigravity(line + b"\n")
        except ValueError as error:
            raise ValueError(f"Exp444 {name} is invalid") from error
        if not isinstance(value, Mapping) or canonical_json_bytes_antigravity(value) != line + b"\n":
            raise ValueError(f"Exp444 {name} is not canonical")
        rows.append(dict(value))
    if canonical_jsonl_bytes_antigravity(rows) != encoded:
        raise ValueError(f"Exp444 {name} is not canonical")
    return tuple(rows)


def _ensure_label_free_antigravity(value: object, name: str) -> None:
    """Apply the prediction contract to scientific rows, never governance."""
    try:
        validate_prediction_payload_antigravity(value)
    except ValueError as error:
        raise ValueError(f"Exp444 {name} contains a forbidden prediction field") from error


@dataclass(frozen=True)
class FrozenPublicationAntigravity:
    """Immutable publication handoff for the later, separately gated evaluator."""

    publication_root: Path
    manifest: Mapping[str, object]
    selection: StabilitySelectionAntigravity

    def __post_init__(self) -> None:
        root = Path(self.publication_root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Exp444 publication root is absent or invalid")
        manifest = dict(self.manifest)
        if type(manifest.get("label_access_count")) is not int or manifest["label_access_count"] != 0:
            raise ValueError("Exp444 publication label governance is invalid")
        if not isinstance(self.selection, StabilitySelectionAntigravity):
            raise ValueError("Exp444 publication selection is invalid")
        object.__setattr__(self, "publication_root", root)
        object.__setattr__(self, "manifest", MappingProxyType(manifest))

    @property
    def stable_count(self) -> int:
        return self.selection.stable_count

    @property
    def unstable_count(self) -> int:
        return self.selection.unstable_count

    @property
    def source_coordinates(self) -> frozenset[tuple[str, int]]:
        return frozenset(item.coordinate for item in self.selection.source)

    @property
    def stable_coordinates(self) -> frozenset[tuple[str, int]]:
        return frozenset(item.coordinate for item in self.selection.stable)

    @property
    def unstable_coordinates(self) -> frozenset[tuple[str, int]]:
        return frozenset(item.coordinate for item in self.selection.unstable)

    @property
    def label_access_count(self) -> int:
        return 0

    @property
    def prediction_sha256(self) -> str:
        value = self.manifest.get("prediction_sha256")
        if type(value) is not str:
            raise ValueError("Exp444 publication prediction SHA256 is invalid")
        return value


@dataclass(frozen=True)
class _PublishSourceAuthorityAntigravity:
    candidates: tuple[SourceCandidateAntigravity, ...]
    dataset_summaries: tuple[dict[str, object], ...]
    source_hashes: Mapping[str, str]
    control_rows: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        if len(self.candidates) != FROZEN_CONFIG.source_candidate_count:
            raise ValueError("Exp444 source authority candidate count is invalid")
        if len(self.dataset_summaries) != FROZEN_CONFIG.dataset_count:
            raise ValueError("Exp444 dataset summary count is invalid")
        if any(not isinstance(item, SourceCandidateAntigravity) for item in self.candidates):
            raise ValueError("Exp444 source authority candidates are invalid")
        names = tuple(row.get("dataset") for row in self.dataset_summaries)
        if any(type(name) is not str or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Exp444 dataset summaries are invalid")
        if tuple(sorted(names)) != names:
            raise ValueError("Exp444 dataset summaries are not in canonical order")
        for value, name in ((self.dataset_summaries, "dataset summary"), (self.control_rows, "control replay")):
            _ensure_label_free_antigravity(value, name)
        if len(self.control_rows) != FROZEN_CONFIG.source_candidate_count:
            raise ValueError("Exp444 original-control replay count is invalid")


def _checkpoint_context_antigravity() -> CheckpointContextAntigravity:
    return CheckpointContextAntigravity(
        FROZEN_CONFIG.experiment_id,
        FROZEN_CONFIG.run_identity,
        scientific_config_sha256_antigravity(),
        FROZEN_CONFIG.environment_lock_sha256,
        FROZEN_CONFIG.source_candidate_count,
        FROZEN_CONFIG.cpu_workers,
        FROZEN_CONFIG.inner_threads,
    )


def _complete_checkpoint_results_antigravity(attempt_root: Path) -> tuple[CandidateStabilityResultAntigravity, ...]:
    attempt = Path(attempt_root)
    if attempt.is_symlink() or not attempt.is_dir():
        raise ValueError("Exp444 attempt root is absent or invalid")
    checkpoint = attempt / "checkpoint_antigravity"
    context = _read_canonical_json_antigravity(checkpoint / "context_antigravity.json", "checkpoint context")
    if context != _checkpoint_context_antigravity().to_payload():
        raise ValueError("Exp444 checkpoint context changed")
    state = _read_canonical_json_antigravity(checkpoint / "state_antigravity.json", "checkpoint state")
    expected_state = {"context_sha256", "status", "completed_units", "total_units", "error_count", "current_unit", "current_view", "payload_sha256", "record_sha256_prefix", "process", "error", "updated_at"}
    if set(state) != expected_state or state["status"] != "complete" or state["completed_units"] != 30 or state["total_units"] != 30 or state["error_count"] != 0 or state["current_unit"] is not None or state["current_view"] is not None or state["error"] is not None:
        raise ValueError("Exp444 checkpoint is not a complete inactive 30-record authority")
    progress = _read_canonical_json_antigravity(attempt / "progress_antigravity.json", "progress")
    pid = _read_canonical_json_antigravity(attempt / "pid_antigravity.json", "PID")
    if progress.get("stage") != "complete" or progress.get("active") is not False or progress.get("completed") != 30 or progress.get("total") != 30 or progress.get("backend") != "cpu" or pid.get("active") is not False:
        raise ValueError("Exp444 attempt is not complete and inactive")
    prefix = state["record_sha256_prefix"]
    if not isinstance(prefix, list) or len(prefix) != 30 or any(type(item) is not str or len(item) != 64 for item in prefix):
        raise ValueError("Exp444 checkpoint record prefix is invalid")
    if state["payload_sha256"] != _sha256_antigravity(canonical_json_bytes_antigravity(prefix)):
        raise ValueError("Exp444 checkpoint prefix hash changed")
    records = checkpoint / "records_antigravity"
    if records.is_symlink() or not records.is_dir() or tuple(sorted(item.name for item in records.iterdir())) != _RECORD_NAMES_ANTIGRAVITY:
        raise ValueError("Exp444 checkpoint record inventory is invalid")
    results: list[CandidateStabilityResultAntigravity] = []
    for sequence, name in enumerate(_RECORD_NAMES_ANTIGRAVITY):
        path = records / name
        payload = _read_canonical_json_antigravity(path, f"checkpoint record {sequence}")
        if set(payload) != {"sequence", "result"} or payload["sequence"] != sequence or not isinstance(payload["result"], Mapping):
            raise ValueError("Exp444 checkpoint record is invalid")
        encoded = path.read_bytes()
        if _sha256_antigravity(encoded) != prefix[sequence]:
            raise ValueError("Exp444 checkpoint record hash changed")
        try:
            result = CandidateStabilityResultAntigravity.from_payload(payload["result"])
        except ValueError as error:
            raise ValueError("Exp444 checkpoint stability result is invalid") from error
        if result.candidate.original_rank != sequence + 1:
            raise ValueError("Exp444 checkpoint rank order changed")
        results.append(result)
    return tuple(results)


def _exp442_dataset_summaries_antigravity() -> tuple[dict[str, object], ...]:
    """Read the already frozen Exp442 summary, never an unbounded database query."""
    source = Path(FROZEN_CONFIG.exp442_prediction_path).with_name(DATASET_SUMMARY_FILE_ANTIGRAVITY)
    rows = _read_canonical_jsonl_antigravity(source, "frozen source dataset summary")
    return rows


def _publish_source_authority_antigravity() -> _PublishSourceAuthorityAntigravity:
    """Lazy values-only source replay used only before first publication creation."""
    from .sources_antigravity import load_candidate_values_antigravity, verify_frozen_sources_antigravity
    from .stability_antigravity import replay_original_candidates_antigravity

    snapshot = verify_frozen_sources_antigravity()
    session = snapshot.session
    try:
        inputs = load_candidate_values_antigravity(snapshot)
        controls = replay_original_candidates_antigravity(inputs)
        if snapshot.label_access_count != 0 or getattr(session, "label_access_count", None) != 0:
            raise ValueError("Exp444 publication source replay accessed labels")
        if tuple(item.candidate for item in inputs) != snapshot.candidates or tuple(item.candidate for item in controls) != snapshot.candidates:
            raise ValueError("Exp444 original-control source coordinates changed")
        control_rows = tuple(
            {
                "candidate": control.candidate.to_payload(),
                "original_depths": list(control.original_depths),
                "original_rank": control.original_rank,
            }
            for control in controls
        )
        return _PublishSourceAuthorityAntigravity(snapshot.candidates, _exp442_dataset_summaries_antigravity(), dict(snapshot.source_hashes), control_rows)
    finally:
        if session is not None:
            session.close()


def _result_rows_antigravity(results: Sequence[CandidateStabilityResultAntigravity]) -> tuple[dict[str, object], ...]:
    rows = tuple(item.to_payload() for item in results)
    _ensure_label_free_antigravity(rows, "candidate stability")
    return rows


def _view_rows_antigravity(results: Sequence[CandidateStabilityResultAntigravity]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for result in results:
        for view in result.views:
            rows.append({"dataset": result.candidate.dataset, "instance_index": result.candidate.instance_index, "original_rank": result.candidate.original_rank, "lane": result.lane, "view": view.to_payload()})
    _ensure_label_free_antigravity(rows, "view evidence")
    return tuple(rows)


def _source_audit_payload_antigravity(authority: _PublishSourceAuthorityAntigravity) -> dict[str, object]:
    return {"schema": "exp444.source-audit.v1", "source_hashes": dict(sorted(authority.source_hashes.items())), "source_candidate_count": 30, "dataset_count": 1117, "label_access_count": 0}


def _runtime_payload_antigravity(attempt: Path, results: Sequence[CandidateStabilityResultAntigravity]) -> dict[str, object]:
    state = _read_canonical_json_antigravity(attempt / "checkpoint_antigravity/state_antigravity.json", "checkpoint state")
    return {"schema": "exp444.runtime.v1", "attempt_name": attempt.name, "backend": "cpu", "workers": 11, "completed": len(results), "record_sha256_prefix": list(state["record_sha256_prefix"]), "label_access_count": 0}


def _backend_payload_antigravity() -> dict[str, object]:
    return BackendDecisionAntigravity("cpu", 11, (), "cpu_fallback_until_parity_speed_gate").to_payload()


def _payloads_antigravity(attempt: Path, authority: _PublishSourceAuthorityAntigravity) -> tuple[dict[str, bytes], StabilitySelectionAntigravity]:
    results = _complete_checkpoint_results_antigravity(attempt)
    source = tuple(authority.candidates)
    if tuple(item.candidate for item in results) != source:
        raise ValueError("Exp444 checkpoint coordinates differ from frozen source")
    stable = tuple(item for item in results if item.lane == "stable_review")
    unstable = tuple(item for item in results if item.lane == "unstable_review")
    selection = StabilitySelectionAntigravity(source, stable, unstable)
    control_candidates = tuple(SourceCandidateAntigravity.from_payload(row["candidate"]) for row in authority.control_rows if isinstance(row.get("candidate"), Mapping))
    if control_candidates != source or tuple(row.get("original_rank") for row in authority.control_rows) != tuple(range(1, 31)):
        raise ValueError("Exp444 original-control parity failed")
    for row, candidate in zip(authority.control_rows, source, strict=True):
        if row.get("original_depths") != list(candidate.original_depths):
            raise ValueError("Exp444 original-control depths changed")
    result_rows = _result_rows_antigravity(results)
    stable_rows = _result_rows_antigravity(stable)
    unstable_rows = _result_rows_antigravity(unstable)
    payloads: dict[str, bytes] = {
        DATASET_SUMMARY_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(authority.dataset_summaries),
        SOURCE_CANDIDATES_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(item.to_payload() for item in source),
        CANDIDATE_STABILITY_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(result_rows),
        VIEW_EVIDENCE_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(_view_rows_antigravity(results)),
        STABLE_REVIEW_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(stable_rows),
        UNSTABLE_REVIEW_FILE_ANTIGRAVITY: canonical_jsonl_bytes_antigravity(unstable_rows),
        CONTROL_REPLAY_FILE_ANTIGRAVITY: canonical_json_bytes_antigravity({"schema": "exp444.control-replay.v1", "rows": list(authority.control_rows)}),
        SOURCE_AUDIT_FILE_ANTIGRAVITY: canonical_json_bytes_antigravity(_source_audit_payload_antigravity(authority)),
        BACKEND_DECISION_FILE_ANTIGRAVITY: canonical_json_bytes_antigravity(_backend_payload_antigravity()),
        RUNTIME_FILE_ANTIGRAVITY: canonical_json_bytes_antigravity(_runtime_payload_antigravity(attempt, results)),
    }
    return payloads, selection


def _final_payloads_antigravity(leaf_payloads: Mapping[str, bytes], selection: StabilitySelectionAntigravity) -> dict[str, bytes]:
    if set(leaf_payloads) != set(_LEAF_FILES_ANTIGRAVITY):
        raise ValueError("Exp444 publication leaf inventory is invalid")
    leaf_hashes = {name: _sha256_antigravity(leaf_payloads[name]) for name in _LEAF_FILES_ANTIGRAVITY}
    prediction_sha = _sha256_antigravity(leaf_payloads[STABLE_REVIEW_FILE_ANTIGRAVITY] + leaf_payloads[UNSTABLE_REVIEW_FILE_ANTIGRAVITY])
    manifest = {"schema": "exp444.publication.v1", "experiment_id": "Exp444", "run_identity": FROZEN_CONFIG.run_identity, "source_candidate_count": 30, "dataset_count": 1117, "stable_count": selection.stable_count, "unstable_count": selection.unstable_count, "prediction_sha256": prediction_sha, "label_access_count": 0, "leaf_sha256": leaf_hashes}
    manifest_bytes = canonical_json_bytes_antigravity(manifest)
    status = {"schema": "exp444.status.v1", "status": "descriptive_complete_no_promotion", "review_only": True, "operational_alert_eligible": False, "scientific_winner_selected": False, "source_candidate_count": 30, "stable_count": selection.stable_count, "unstable_count": selection.unstable_count, "label_access_count": 0}
    status_bytes = canonical_json_bytes_antigravity(status)
    artifact_map = {"schema": "exp444.artifact-sha256.v1", "files": {**leaf_hashes, MANIFEST_FILE_ANTIGRAVITY: _sha256_antigravity(manifest_bytes), STATUS_FILE_ANTIGRAVITY: _sha256_antigravity(status_bytes)}}
    return {**leaf_payloads, MANIFEST_FILE_ANTIGRAVITY: manifest_bytes, STATUS_FILE_ANTIGRAVITY: status_bytes, ARTIFACT_SHA256_FILE_ANTIGRAVITY: canonical_json_bytes_antigravity(artifact_map)}


def _write_staging_antigravity(destination: Path, payloads: Mapping[str, bytes]) -> None:
    staging = destination.with_name(f".{destination.name}.{uuid4().hex}.staging_antigravity")
    if staging.exists() or staging.is_symlink():
        raise ValueError("Exp444 publication staging path already exists")
    staging.mkdir(mode=0o700)
    try:
        for name in (*_LEAF_FILES_ANTIGRAVITY, MANIFEST_FILE_ANTIGRAVITY, STATUS_FILE_ANTIGRAVITY, ARTIFACT_SHA256_FILE_ANTIGRAVITY):
            path = staging / name
            with path.open("xb") as output:
                output.write(payloads[name])
                output.flush()
                os.fsync(output.fileno())
            if path.read_bytes() != payloads[name]:
                raise ValueError("Exp444 publication staging bytes changed")
        exclusive_rename_antigravity(staging, destination)
    except BaseException:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            for item in staging.iterdir():
                if item.is_file() and not item.is_symlink():
                    item.unlink()
            staging.rmdir()
        raise


def _parse_selection_antigravity(root: Path) -> StabilitySelectionAntigravity:
    source = tuple(SourceCandidateAntigravity.from_payload(row) for row in _read_canonical_jsonl_antigravity(root / SOURCE_CANDIDATES_FILE_ANTIGRAVITY, "source candidates"))
    results = tuple(CandidateStabilityResultAntigravity.from_payload(row) for row in _read_canonical_jsonl_antigravity(root / CANDIDATE_STABILITY_FILE_ANTIGRAVITY, "candidate stability"))
    stable = tuple(CandidateStabilityResultAntigravity.from_payload(row) for row in _read_canonical_jsonl_antigravity(root / STABLE_REVIEW_FILE_ANTIGRAVITY, "stable review"))
    unstable = tuple(CandidateStabilityResultAntigravity.from_payload(row) for row in _read_canonical_jsonl_antigravity(root / UNSTABLE_REVIEW_FILE_ANTIGRAVITY, "unstable review"))
    if tuple(item.candidate for item in results) != source:
        raise ValueError("Exp444 publication lane ordering changed")
    expected_stable = tuple(item for item in results if item.lane == "stable_review")
    expected_unstable = tuple(item for item in results if item.lane == "unstable_review")
    if stable != expected_stable or unstable != expected_unstable:
        raise ValueError("Exp444 publication lane membership changed")
    return StabilitySelectionAntigravity(source, stable, unstable)


def inspect_publication_antigravity(publication_root: Path) -> FrozenPublicationAntigravity:
    """Fully replay a frozen publication without database, worker, or evaluator access."""
    root = Path(publication_root)
    if root.is_symlink() or not root.is_dir() or tuple(sorted(item.name for item in root.iterdir())) != tuple(sorted(_PUBLICATION_FILES_ANTIGRAVITY)):
        raise ValueError("Exp444 publication inventory is invalid")
    selection = _parse_selection_antigravity(root)
    summaries = _read_canonical_jsonl_antigravity(root / DATASET_SUMMARY_FILE_ANTIGRAVITY, "dataset summary")
    if len(summaries) != 1117 or tuple(row.get("dataset") for row in summaries) != tuple(sorted(row.get("dataset") for row in summaries)) or len({row.get("dataset") for row in summaries}) != 1117:
        raise ValueError("Exp444 publication dataset summaries are invalid")
    _ensure_label_free_antigravity(summaries, "dataset summary")
    for name in (VIEW_EVIDENCE_FILE_ANTIGRAVITY,):
        _ensure_label_free_antigravity(_read_canonical_jsonl_antigravity(root / name, name), name)
    ordered_results = tuple(sorted((*selection.stable, *selection.unstable), key=lambda item: item.candidate.original_rank))
    if (root / VIEW_EVIDENCE_FILE_ANTIGRAVITY).read_bytes() != canonical_jsonl_bytes_antigravity(_view_rows_antigravity(ordered_results)):
        raise ValueError("Exp444 publication view evidence changed")
    control = _read_canonical_json_antigravity(root / CONTROL_REPLAY_FILE_ANTIGRAVITY, "control replay")
    if set(control) != {"schema", "rows"} or control["schema"] != "exp444.control-replay.v1" or not isinstance(control["rows"], list):
        raise ValueError("Exp444 publication control replay is invalid")
    control_candidates = tuple(SourceCandidateAntigravity.from_payload(row["candidate"]) for row in control["rows"] if isinstance(row, Mapping) and isinstance(row.get("candidate"), Mapping))
    if control_candidates != selection.source or len(control_candidates) != 30:
        raise ValueError("Exp444 publication original-control parity is invalid")
    expected_control_rows = [
        {"candidate": candidate.to_payload(), "original_depths": list(candidate.original_depths), "original_rank": candidate.original_rank}
        for candidate in selection.source
    ]
    if control["rows"] != expected_control_rows:
        raise ValueError("Exp444 publication original-control replay changed")
    audit = _read_canonical_json_antigravity(root / SOURCE_AUDIT_FILE_ANTIGRAVITY, "source audit")
    if set(audit) != {"schema", "source_hashes", "source_candidate_count", "dataset_count", "label_access_count"} or audit["schema"] != "exp444.source-audit.v1" or audit["source_candidate_count"] != 30 or audit["dataset_count"] != 1117 or audit["label_access_count"] != 0 or not isinstance(audit["source_hashes"], Mapping):
        raise ValueError("Exp444 publication source audit is invalid")
    if not audit["source_hashes"] or any(type(name) is not str or type(digest) is not str or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest) for name, digest in audit["source_hashes"].items()):
        raise ValueError("Exp444 publication source audit hashes are invalid")
    backend = BackendDecisionAntigravity.from_payload(_read_canonical_json_antigravity(root / BACKEND_DECISION_FILE_ANTIGRAVITY, "backend decision"))
    if backend.backend != "cpu" or backend.workers != 11:
        raise ValueError("Exp444 publication backend decision is invalid")
    runtime = _read_canonical_json_antigravity(root / RUNTIME_FILE_ANTIGRAVITY, "runtime")
    if set(runtime) != {"schema", "attempt_name", "backend", "workers", "completed", "record_sha256_prefix", "label_access_count"} or runtime["schema"] != "exp444.runtime.v1" or runtime["backend"] != "cpu" or runtime["workers"] != 11 or runtime["completed"] != 30 or runtime["label_access_count"] != 0 or not isinstance(runtime["record_sha256_prefix"], list) or len(runtime["record_sha256_prefix"]) != 30:
        raise ValueError("Exp444 publication runtime evidence is invalid")
    manifest = _read_canonical_json_antigravity(root / MANIFEST_FILE_ANTIGRAVITY, "manifest")
    status = _read_canonical_json_antigravity(root / STATUS_FILE_ANTIGRAVITY, "status")
    leaf_payloads = {name: (root / name).read_bytes() for name in _LEAF_FILES_ANTIGRAVITY}
    expected = _final_payloads_antigravity(leaf_payloads, selection)
    for name, payload in expected.items():
        if (root / name).read_bytes() != payload:
            raise ValueError(f"Exp444 publication {name} hash or contract changed")
    if manifest["label_access_count"] != 0 or status.get("label_access_count") != 0:
        raise ValueError("Exp444 publication label governance is invalid")
    return FrozenPublicationAntigravity(root, manifest, selection)


def publish_prediction_antigravity(attempt_root: Path) -> FrozenPublicationAntigravity:
    """Freeze a complete checkpoint into immutable stable/unstable review lanes."""
    attempt = Path(attempt_root)
    authority = _publish_source_authority_antigravity()
    payloads, selection = _payloads_antigravity(attempt, authority)
    expected = _final_payloads_antigravity(payloads, selection)
    destination = attempt / PUBLICATION_DIRECTORY_ANTIGRAVITY
    if destination.exists() or destination.is_symlink():
        inspected = inspect_publication_antigravity(destination)
        for name, payload in expected.items():
            if (destination / name).read_bytes() != payload:
                raise ValueError("Exp444 publication re-entry bytes differ from the canonical freeze")
        return inspected
    _write_staging_antigravity(destination, expected)
    return inspect_publication_antigravity(destination)


__all__ = [
    "FrozenPublicationAntigravity", "publish_prediction_antigravity", "inspect_publication_antigravity",
    "PUBLICATION_DIRECTORY_ANTIGRAVITY", "DATASET_SUMMARY_FILE_ANTIGRAVITY",
    "SOURCE_CANDIDATES_FILE_ANTIGRAVITY", "CANDIDATE_STABILITY_FILE_ANTIGRAVITY",
    "VIEW_EVIDENCE_FILE_ANTIGRAVITY", "STABLE_REVIEW_FILE_ANTIGRAVITY",
    "UNSTABLE_REVIEW_FILE_ANTIGRAVITY", "CONTROL_REPLAY_FILE_ANTIGRAVITY",
    "SOURCE_AUDIT_FILE_ANTIGRAVITY", "BACKEND_DECISION_FILE_ANTIGRAVITY",
    "RUNTIME_FILE_ANTIGRAVITY", "MANIFEST_FILE_ANTIGRAVITY", "STATUS_FILE_ANTIGRAVITY",
    "ARTIFACT_SHA256_FILE_ANTIGRAVITY",
]
