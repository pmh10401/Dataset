"""Pinned, label-free Exp440 and Exp404 source replay for Exp441."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Callable, Mapping

from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.contracts_antigravity import (
    Exp440DatasetResultAntigravity,
)

from .artifacts_antigravity import (
    parse_canonical_json_bytes_antigravity,
    read_pinned_regular_bytes_antigravity,
)
from .config_antigravity import FROZEN_CONFIG, cpu_worker_count_antigravity
from .contracts_antigravity import (
    CandidateEvidenceAntigravity,
    FrozenSourceSnapshotAntigravity,
    ParsedRecordAntigravity,
)


_FIXED_WORKERS_ANTIGRAVITY = 11
_PUBLICATION_FILENAMES_ANTIGRAVITY = (
    "abstention_ledger_antigravity.jsonl", "artifact_sha256_antigravity.json",
    "backend_decision_antigravity.json", "dataset_summary_antigravity.jsonl",
    "depth_evidence_antigravity.jsonl", "manifest_antigravity.json",
    "prediction_antigravity.jsonl", "retained_candidate_ranking_antigravity.jsonl",
    "status_antigravity.json", "train_exclusion_audit_antigravity.jsonl",
)
_NUMERICAL_THREAD_ENV_ANTIGRAVITY = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def hard_coordinate_identity_sha256_antigravity(coordinates: object) -> str:
    """Hash the Exp404 identity as a sorted, compact, newline-free JSON array."""
    try:
        values = tuple(coordinates)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("Hard coordinates are invalid") from error
    if any(not isinstance(dataset, str) or not dataset or type(index) is not int or index < 0
           for dataset, index in values):
        raise ValueError("Hard coordinates are invalid")
    if len(values) != len(set(values)):
        raise ValueError("Hard coordinates are duplicated")
    payload = json.dumps(
        [[dataset, index] for dataset, index in sorted(values)],
        ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _allowed_paths_antigravity() -> frozenset[Path]:
    checkpoint = Path(FROZEN_CONFIG.checkpoint_context_path).parent
    publication = Path(FROZEN_CONFIG.publication_path)
    return frozenset(path.resolve(strict=False) for path in (
        Path(FROZEN_CONFIG.checkpoint_context_path), Path(FROZEN_CONFIG.checkpoint_state_path),
        Path(FROZEN_CONFIG.checkpoint_retained_inventory_path), Path(FROZEN_CONFIG.hard_prediction_path),
        Path(FROZEN_CONFIG.hard_manifest_path),
        *(checkpoint / "records_antigravity" / f"{index:04d}_record_antigravity.json" for index in range(1117)),
        *(publication / filename for filename in _PUBLICATION_FILENAMES_ANTIGRAVITY),
    ))


def _validate_allowed_prediction_path_antigravity(path: Path) -> Path:
    """Resolve only a member of Exp441's closed label-free input set."""
    value = Path(path)
    forbidden = "evaluation" + "_antigravity"
    if forbidden in value.parts or forbidden in str(value):
        raise ValueError("evaluation source is forbidden")
    resolved = value.resolve(strict=False)
    if resolved not in _allowed_paths_antigravity():
        raise ValueError("prediction source is outside the positive allowlist")
    return resolved


def _read_pinned_regular_antigravity(path: Path) -> tuple[bytes, str]:
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("source is not a regular file")
    except OSError as error:
        raise ValueError("source is not a regular file") from error
    try:
        return read_pinned_regular_bytes_antigravity(path)
    except OSError as error:
        raise ValueError("source is not a regular file") from error


def _read_pinned_allowed_antigravity(path: Path, expected_sha256: str | None) -> tuple[bytes, Path]:
    resolved = _validate_allowed_prediction_path_antigravity(path)
    payload, observed = _read_pinned_regular_antigravity(resolved)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError("source SHA256 mismatch")
    return payload, resolved


def _parse_json_object_antigravity(payload: bytes, name: str) -> Mapping[str, object]:
    parsed = parse_canonical_json_bytes_antigravity(payload)
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{name} payload is invalid")
    _reject_nonfalse_label_or_metric_fields_antigravity(parsed)
    return parsed


def _reject_nonfalse_label_or_metric_fields_antigravity(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in ("label", "groundtruth", "anomaly_location", "evaluation")) and item is not False:
                raise ValueError("source contains a non-false label or evaluation field")
            _reject_nonfalse_label_or_metric_fields_antigravity(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nonfalse_label_or_metric_fields_antigravity(item)


def _context_datasets_antigravity(payload: Mapping[str, object]) -> tuple[str, ...]:
    if set(payload) != {"datasets", "reviewed_commit", "run_identity", "schema"}:
        raise ValueError("context payload is invalid")
    datasets = payload["datasets"]
    if (not isinstance(datasets, list) or len(datasets) != 1117 or
            any(not isinstance(dataset, str) or not dataset for dataset in datasets)):
        raise ValueError("context datasets are invalid")
    result = tuple(datasets)
    if result != tuple(sorted(set(result))):
        raise ValueError("context datasets must be sorted and unique")
    return result


def _state_record_digests_antigravity(payload: Mapping[str, object], datasets: tuple[str, ...]) -> Mapping[str, str]:
    if set(payload) != {"completed", "records"} or type(payload["completed"]) is not int or payload["completed"] != 1117:
        raise ValueError("state completed count is invalid")
    records = payload["records"]
    if not isinstance(records, Mapping) or set(records) != set(datasets):
        raise ValueError("state record map is invalid")
    copied: dict[str, str] = {}
    for dataset in datasets:
        digest = records[dataset]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("state record digest is invalid")
        copied[dataset] = digest
    return MappingProxyType(copied)


def _validate_publication_antigravity() -> tuple[dict[Path, str], bytes]:
    root = Path(FROZEN_CONFIG.publication_path)
    names = tuple(sorted(item.name for item in root.iterdir()))
    if names != tuple(sorted(_PUBLICATION_FILENAMES_ANTIGRAVITY)):
        raise ValueError("publication inventory is invalid")
    payloads: dict[Path, bytes] = {}
    digests: dict[Path, str] = {}
    for filename in _PUBLICATION_FILENAMES_ANTIGRAVITY:
        path = root / filename
        payload, resolved = _read_pinned_allowed_antigravity(path, None)
        payloads[resolved] = payload
        digests[resolved] = sha256(payload).hexdigest()
    artifact_path = _validate_allowed_prediction_path_antigravity(root / "artifact_sha256_antigravity.json")
    if digests[artifact_path] != FROZEN_CONFIG.publication_artifact_map_sha256:
        raise ValueError("publication artifact map SHA256 mismatch")
    artifact_map = _parse_json_object_antigravity(payloads[artifact_path], "publication artifact map")
    files = artifact_map.get("files")
    expected_names = set(_PUBLICATION_FILENAMES_ANTIGRAVITY) - {"artifact_sha256_antigravity.json"}
    if set(artifact_map) != {"files"} or not isinstance(files, Mapping) or set(files) != expected_names:
        raise ValueError("publication artifact map is invalid")
    for filename, digest in files.items():
        if not isinstance(digest, str) or len(digest) != 64 or digests[root / str(filename)] != digest:
            raise ValueError("publication artifact hash is invalid")
    if digests[_validate_allowed_prediction_path_antigravity(root / "prediction_antigravity.jsonl")] != FROZEN_CONFIG.publication_prediction_sha256:
        raise ValueError("publication prediction SHA256 mismatch")
    manifest_path = _validate_allowed_prediction_path_antigravity(root / "manifest_antigravity.json")
    if digests[manifest_path] != FROZEN_CONFIG.publication_manifest_sha256:
        raise ValueError("publication manifest SHA256 mismatch")
    manifest = _parse_json_object_antigravity(payloads[manifest_path], "publication manifest")
    manifest_hashes = {name: digest for name, digest in files.items() if name != "manifest_antigravity.json"}
    if manifest.get("artifact_sha256") != manifest_hashes or manifest.get("prediction_sha256") != FROZEN_CONFIG.publication_prediction_sha256:
        raise ValueError("publication manifest is invalid")
    return digests, payloads[artifact_path]


def _hard_coordinates_antigravity(payload: bytes, manifest_payload: Mapping[str, object]) -> frozenset[tuple[str, int]]:
    if set(manifest_payload) < {"contains_test_labels", "contains_evaluation_metrics", "datasets", "frozen_hard_lanes"}:
        raise ValueError("Hard manifest is invalid")
    if manifest_payload["contains_test_labels"] is not False or manifest_payload["contains_evaluation_metrics"] is not False:
        raise ValueError("Hard manifest is not label-free")
    coordinates: set[tuple[str, int]] = set()
    for line in payload.splitlines():
        if not line:
            raise ValueError("Hard source contains a blank line")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Hard source is invalid JSON") from error
        if not isinstance(value, Mapping):
            raise ValueError("Hard source row is invalid")
        _reject_nonfalse_label_or_metric_fields_antigravity(value)
        dataset, index = value.get("dataset"), value.get("candidate_index")
        if not isinstance(dataset, str) or not dataset or type(index) is not int or index < 0:
            raise ValueError("Hard coordinate is invalid")
        coordinate = (dataset, index)
        if coordinate in coordinates:
            raise ValueError("Hard coordinate is duplicated")
        coordinates.add(coordinate)
    if len(coordinates) != 2218 or len({dataset for dataset, _ in coordinates}) != 888:
        raise ValueError("Hard population is invalid")
    if hard_coordinate_identity_sha256_antigravity(coordinates) != FROZEN_CONFIG.hard_coordinate_identity_sha256:
        raise ValueError("Hard coordinate identity mismatch")
    return frozenset(coordinates)


def _parse_record_task_antigravity(task: tuple[str, str, str, int]) -> ParsedRecordAntigravity:
    """Parse precisely one descriptor-pinned record; task data is process-safe and label-free."""
    if (not isinstance(task, tuple) or len(task) != 4 or not isinstance(task[0], str) or
            not isinstance(task[1], str) or not isinstance(task[2], str) or type(task[3]) is not int):
        raise ValueError("record task is invalid")
    path_text, expected_dataset, expected_digest, _sequence = task
    payload, observed = _read_pinned_regular_antigravity(Path(path_text))
    if observed != expected_digest:
        raise ValueError("record digest is invalid")
    value = _parse_json_object_antigravity(payload, "record")
    if set(value) != {"dataset", "status", "provisional_candidates", "abstention_indices", "excluded_train_rows", "test_row_count"}:
        raise ValueError("record payload is invalid")
    try:
        result = Exp440DatasetResultAntigravity.from_payload(value)
    except ValueError as error:
        raise ValueError("record payload is invalid") from error
    if result.dataset != expected_dataset:
        raise ValueError("record dataset is invalid")
    candidates = tuple(CandidateEvidenceAntigravity(
        dataset=item.dataset, instance_index=item.instance_index, raw_length=item.raw_length,
        depths=tuple(item.depths), agreement_count=item.agreement_count,
        second_excess_ratio=item.second_excess_ratio, second_empirical_rank=item.second_empirical_rank,
    ) for item in result.provisional_candidates)
    try:
        return ParsedRecordAntigravity(result.dataset, result.test_row_count, candidates)
    except ValueError as error:
        raise ValueError("record candidate is outside TEST identity") from error


def _fixed_worker_count_antigravity() -> int:
    """Bind the reviewed fixed replay to the project worker policy's 12-core provenance."""
    if cpu_worker_count_antigravity(logical_cpus=12) != _FIXED_WORKERS_ANTIGRAVITY:
        raise ValueError("fixed worker policy is invalid")
    return _FIXED_WORKERS_ANTIGRAVITY


def _parse_records_fixed_workers_antigravity(
    tasks: tuple[tuple[str, str, str, int], ...], observer: Callable[[Mapping[str, object]], None],
) -> tuple[ParsedRecordAntigravity, ...]:
    """Parse with eleven processes and emit only parent-committed context-order events."""
    if tuple(task[3] for task in tasks) != tuple(range(len(tasks))):
        raise ValueError("record task sequence is invalid")
    for name in _NUMERICAL_THREAD_ENV_ANTIGRAVITY:
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=_fixed_worker_count_antigravity()) as executor:
        futures = tuple(executor.submit(_parse_record_task_antigravity, task) for task in tasks)
        records = tuple(future.result() for future in futures)
    for task, record in zip(tasks, records, strict=True):
        event = MappingProxyType({"sequence": task[3], "dataset": record.dataset, "record_digest": task[2]})
        observer(event)
    return records


def _record_tasks_antigravity(datasets: tuple[str, ...], digests: Mapping[str, str]) -> tuple[tuple[str, str, str, int], ...]:
    root = Path(FROZEN_CONFIG.checkpoint_records_path)
    names = tuple(sorted(item.name for item in root.iterdir()))
    expected = tuple(f"{index:04d}_record_antigravity.json" for index in range(1117))
    if names != expected:
        raise ValueError("record inventory is invalid")
    return tuple((str(root / expected[sequence]), dataset, digests[dataset], sequence)
                 for sequence, dataset in enumerate(datasets))


def _snapshot_from_records_antigravity(
    datasets: tuple[str, ...], digests: Mapping[str, str], records: tuple[ParsedRecordAntigravity, ...],
    hard: frozenset[tuple[str, int]], hashes: Mapping[str, str],
) -> FrozenSourceSnapshotAntigravity:
    if tuple(record.dataset for record in records) != datasets:
        raise ValueError("records are not in context order")
    rows = MappingProxyType({record.dataset: record.test_row_count for record in records})
    if len(rows) != 1117:
        raise ValueError("record dataset population is invalid")
    for dataset, index in hard:
        if dataset not in rows or type(index) is not int or not 0 <= index < rows[dataset]:
            raise ValueError("Hard coordinate is outside TEST identity")
    provisional = tuple(candidate for record in records for candidate in record.candidates)
    if (len(provisional) != 840 or len({candidate.coordinate for candidate in provisional}) != 840 or
            sum(candidate.coordinate not in hard for candidate in provisional) != 394):
        raise ValueError("provisional candidate population is invalid")
    return FrozenSourceSnapshotAntigravity(
        context_datasets=datasets, test_row_count_by_dataset=rows, record_digest_by_dataset=digests,
        hard_coordinates=hard, provisional_candidates=provisional,
        context_sha256=hashes["context"], state_sha256=hashes["state"],
        retained_inventory_sha256=hashes["retained"],
        publication_prediction_sha256=hashes["publication_prediction"],
        publication_manifest_sha256=hashes["publication_manifest"],
        publication_artifact_map_sha256=hashes["publication_artifact_map"],
        hard_prediction_sha256=hashes["hard_prediction"], hard_manifest_sha256=hashes["hard_manifest"],
        hard_coordinate_identity_sha256=FROZEN_CONFIG.hard_coordinate_identity_sha256, label_access_count=0,
    )


def _reverify_paths_antigravity(hashes: Mapping[Path, str]) -> None:
    for path, expected in hashes.items():
        _read_pinned_allowed_antigravity(path, expected)


def _load_prefix_antigravity() -> tuple[tuple[str, ...], Mapping[str, str], frozenset[tuple[str, int]], dict[str, str], dict[Path, str]]:
    watched: dict[Path, str] = {}
    context_bytes, context_path = _read_pinned_allowed_antigravity(Path(FROZEN_CONFIG.checkpoint_context_path), FROZEN_CONFIG.checkpoint_context_sha256)
    state_bytes, state_path = _read_pinned_allowed_antigravity(Path(FROZEN_CONFIG.checkpoint_state_path), FROZEN_CONFIG.checkpoint_state_sha256)
    retained_bytes, retained_path = _read_pinned_allowed_antigravity(Path(FROZEN_CONFIG.checkpoint_retained_inventory_path), FROZEN_CONFIG.checkpoint_retained_inventory_sha256)
    del retained_bytes
    watched.update({context_path: FROZEN_CONFIG.checkpoint_context_sha256, state_path: FROZEN_CONFIG.checkpoint_state_sha256, retained_path: FROZEN_CONFIG.checkpoint_retained_inventory_sha256})
    datasets = _context_datasets_antigravity(_parse_json_object_antigravity(context_bytes, "context"))
    digests = _state_record_digests_antigravity(_parse_json_object_antigravity(state_bytes, "state"), datasets)
    publication_digests, _artifact_bytes = _validate_publication_antigravity()
    watched.update(publication_digests)
    hard_manifest_bytes, hard_manifest_path = _read_pinned_allowed_antigravity(Path(FROZEN_CONFIG.hard_manifest_path), FROZEN_CONFIG.hard_manifest_sha256)
    hard_prediction_bytes, hard_prediction_path = _read_pinned_allowed_antigravity(Path(FROZEN_CONFIG.hard_prediction_path), FROZEN_CONFIG.hard_prediction_sha256)
    watched.update({hard_manifest_path: FROZEN_CONFIG.hard_manifest_sha256, hard_prediction_path: FROZEN_CONFIG.hard_prediction_sha256})
    hard = _hard_coordinates_antigravity(hard_prediction_bytes, _parse_json_object_antigravity(hard_manifest_bytes, "Hard manifest"))
    if len(set(datasets) - {dataset for dataset, _ in hard}) != 229:
        raise ValueError("Hard-empty dataset population is invalid")
    hashes = {
        "context": FROZEN_CONFIG.checkpoint_context_sha256, "state": FROZEN_CONFIG.checkpoint_state_sha256,
        "retained": FROZEN_CONFIG.checkpoint_retained_inventory_sha256,
        "publication_prediction": FROZEN_CONFIG.publication_prediction_sha256,
        "publication_manifest": FROZEN_CONFIG.publication_manifest_sha256,
        "publication_artifact_map": FROZEN_CONFIG.publication_artifact_map_sha256,
        "hard_prediction": FROZEN_CONFIG.hard_prediction_sha256, "hard_manifest": FROZEN_CONFIG.hard_manifest_sha256,
    }
    return datasets, digests, hard, hashes, watched


def _verify_frozen_sources_with_observer_antigravity(observer: Callable[[Mapping[str, object]], None]) -> FrozenSourceSnapshotAntigravity:
    """Production route: fixed eleven-worker, label-free source replay."""
    if not callable(observer):
        raise ValueError("source observer is invalid")
    datasets, digests, hard, hashes, watched = _load_prefix_antigravity()
    tasks = _record_tasks_antigravity(datasets, digests)
    records = _parse_records_fixed_workers_antigravity(tasks, observer)
    watched.update({Path(task[0]): task[2] for task in tasks})
    snapshot = _snapshot_from_records_antigravity(datasets, digests, records, hard, hashes)
    _reverify_paths_antigravity(watched)
    return snapshot


def verify_frozen_sources_antigravity() -> FrozenSourceSnapshotAntigravity:
    """Public, override-free Exp441 source verifier."""
    return _verify_frozen_sources_with_observer_antigravity(lambda _event: None)


def _verify_frozen_sources_single_worker_reference_antigravity() -> FrozenSourceSnapshotAntigravity:
    """Test/audit-only sequential parity reference; never use from production routes."""
    datasets, digests, hard, hashes, watched = _load_prefix_antigravity()
    tasks = _record_tasks_antigravity(datasets, digests)
    records = tuple(_parse_record_task_antigravity(task) for task in tasks)
    watched.update({Path(task[0]): task[2] for task in tasks})
    snapshot = _snapshot_from_records_antigravity(datasets, digests, records, hard, hashes)
    _reverify_paths_antigravity(watched)
    return snapshot


__all__ = ["verify_frozen_sources_antigravity", "hard_coordinate_identity_sha256_antigravity"]
