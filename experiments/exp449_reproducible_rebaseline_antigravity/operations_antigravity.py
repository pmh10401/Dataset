"""Exp449 parent-only two-wave recovery and result-neutral operations records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
import os
from pathlib import Path
import stat
from time import monotonic, time

from paper.equipment_anomaly_detection.artifacts_paper import (
    Task4Task5SuccessAuthority,
    canonical_json_bytes,
    read_canonical,
    verify_task4_task5_attempt_namespace,
    verify_task4_task5_success_namespace,
    write_create_only,
)
from paper.equipment_anomaly_detection.proposal_score_paper import ProposalScoreIncomplete

from .config_antigravity import FROZEN_EXP449_CONFIG
from .contracts_antigravity import Exp449Context, Exp449TerminalStatus
from .runner_antigravity import compose_exp449_proposal_score_antigravity


_TERMINAL_NAME = "proposal_score_incomplete_antigravity.json"
_PROGRESS_NAME = "operations_progress_antigravity.jsonl"
_TERMINAL_SCHEMA = "exp449-proposal-score-incomplete-antigravity.v1"
_PROGRESS_KEYS = frozenset(
    {
        "timestamp",
        "experiment",
        "mode",
        "wave",
        "completed",
        "failed",
        "pending",
        "elapsed_seconds",
        "eta_seconds",
        "dataset",
    }
)
_REQUIRED_PROGRESS_KEYS = _PROGRESS_KEYS - {"dataset"}
_FORBIDDEN_PROGRESS_KEYS = frozenset({"label", "labels", "score", "scores", "p_value", "p_values", "rank", "ranks", "coordinate", "coordinates"})


class _Task45ContextPayloadAntigravity(dict[str, object]):
    """Paper-compatible dict retaining the frozen mode for Task3's gate."""

    __slots__ = ("mode",)

    def __init__(self, payload: Mapping[str, object], mode: str) -> None:
        super().__init__(payload)
        self.mode = mode


@dataclass(frozen=True, slots=True)
class Exp449WaveConfig:
    """Verified inputs for one fixed-root Exp449 recovery lifecycle.

    The configuration deliberately carries data and authorities only.  Scoring,
    record verification, and success publication remain the existing paper
    boundaries; callers cannot replace them with executable adapters.
    """

    context: Exp449Context
    catalog: dict[str, object]
    raw_datasets: Mapping[str, object]
    acceleration_manifest: object

    def __post_init__(self) -> None:
        if not isinstance(self.context, Exp449Context) or not isinstance(self.catalog, dict):
            raise ValueError("Exp449 recovery context is invalid")
        if self.catalog.get("catalog_sha256") != self.context.catalog_sha256:
            raise ValueError("Exp449 recovery catalog identity differs")
        datasets = _catalog_datasets_antigravity(self.catalog)
        raw = dict(self.raw_datasets)
        if tuple(raw) != datasets:
            raise ValueError("Exp449 recovery raw identity differs")
        object.__setattr__(self, "raw_datasets", raw)


@dataclass(frozen=True, slots=True)
class _RecoverySnapshot:
    successes: tuple[str, ...]
    attempts: tuple[dict[str, object], ...]
    failed: tuple[str, ...]
    never_attempted: tuple[str, ...]
    wave_count: int


def _json_plain_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_plain_antigravity(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_plain_antigravity(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def task45_context_payload_antigravity(context: Exp449Context) -> dict[str, object]:
    """Make the sole deterministic bridge to the paper checkpoint context.

    It serializes the already-frozen Exp449 context, then hashes exactly the
    same canonical JSON bytes required by the paper Task4/5 authority.  This is
    a bridge, not a second provenance contract.
    """
    if not isinstance(context, Exp449Context):
        raise TypeError("Exp449 context is required")
    payload: dict[str, object] = {
        "experiment_id": context.experiment_id,
        "mode": context.mode,
        "run_root": str(context.run_root),
        "database_sha256": context.database_sha256,
        "catalog_sha256": context.catalog_sha256,
        "code_closure": _json_plain_antigravity(context.code_closure),
        "lineage_sha256": context.lineage_sha256,
        "codemap_sha256": context.codemap_sha256,
        "interpreter": context.interpreter,
        "environment": _json_plain_antigravity(context.environment),
        "backend_record": _json_plain_antigravity(context.backend_record),
        "seeds_and_constants": _json_plain_antigravity(context.seeds_and_constants),
        "historical_parity_status": context.historical_parity_status,
    }
    payload["context_sha256"] = sha256(canonical_json_bytes(payload)).hexdigest()
    return _Task45ContextPayloadAntigravity(payload, context.mode)


def _catalog_datasets_antigravity(catalog: dict[str, object]) -> tuple[str, ...]:
    entries = catalog.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Exp449 recovery catalog is invalid")
    names = tuple(entry.get("dataset") if isinstance(entry, dict) else None for entry in entries)
    if not names or any(type(name) is not str or not name for name in names) or names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError("Exp449 recovery catalog is invalid")
    expected = FROZEN_EXP449_CONFIG.smoke_datasets if len(names) == len(FROZEN_EXP449_CONFIG.smoke_datasets) else None
    if expected is not None and names != expected:
        raise ValueError("Exp449 smoke recovery catalog differs")
    if len(names) == FROZEN_EXP449_CONFIG.dataset_count:
        return names
    if expected is None:
        raise ValueError("Exp449 recovery catalog count differs")
    return names


def _operations_root_for_context_antigravity(context: Exp449Context) -> Path:
    roots = {
        FROZEN_EXP449_CONFIG.smoke_a_scientific_root: FROZEN_EXP449_CONFIG.smoke_a_operations_root,
        FROZEN_EXP449_CONFIG.smoke_b_scientific_root: FROZEN_EXP449_CONFIG.smoke_b_operations_root,
        FROZEN_EXP449_CONFIG.official_scientific_root: FROZEN_EXP449_CONFIG.official_operations_root,
    }
    try:
        root = roots[Path(context.run_root)]
    except KeyError as error:
        raise ValueError("Exp449 scientific root is not an operations authority") from error
    if root == context.run_root or context.run_root in root.parents or root in context.run_root.parents:
        raise ValueError("Exp449 operations root must be separate from scientific root")
    return root


def _authority_antigravity(config: Exp449WaveConfig) -> tuple[dict[str, object], Task4Task5SuccessAuthority]:
    context_payload = task45_context_payload_antigravity(config.context)
    authority = Task4Task5SuccessAuthority.from_verified_context(
        context_payload, config.catalog, config.acceleration_manifest
    )
    return context_payload, authority


def _attempt_tuple_antigravity(record: object) -> tuple[str, int, int, str, str]:
    dataset = getattr(record, "dataset", None)
    wave = getattr(record, "wave", None)
    attempt = getattr(record, "attempt", None)
    exception_type = getattr(record, "exception_type", None)
    exception_message = getattr(record, "exception_message", None)
    if type(dataset) is not str or not dataset or type(wave) is not int or wave not in {1, 2} or type(attempt) is not int or attempt < 1 or type(exception_type) is not str or not exception_type or type(exception_message) is not str:
        raise ValueError("Exp449 verified attempt record differs")
    return dataset, wave, attempt, exception_type, exception_message


def _read_verified_attempt_payloads_antigravity(config: Exp449WaveConfig, attempts: tuple[tuple[str, int, int, str, str], ...]) -> tuple[dict[str, object], ...]:
    """Bind the terminal ledger to every already-verified on-disk record byte."""
    if not attempts:
        return ()
    root = config.context.run_root / "stage_artifacts_paper" / "proposal_work" / "task4_task5_attempts"
    try:
        paths = sorted(root.iterdir())
    except FileNotFoundError as error:
        raise ValueError("Exp449 verified attempt ledger differs") from error
    expected_names = [f"attempt_{index:06d}_antigravity.json" for index in range(1, len(attempts) + 1)]
    if [path.name for path in paths] != expected_names:
        raise ValueError("Exp449 verified attempt ledger differs")
    payloads: list[dict[str, object]] = []
    for expected, path in zip(attempts, paths, strict=True):
        value = read_canonical(path)
        if not isinstance(value, dict):
            raise ValueError("Exp449 verified attempt ledger differs")
        dataset, wave, attempt, exception_type, exception_message = expected
        if (value.get("dataset"), value.get("wave"), value.get("attempt"), value.get("exception_type"), value.get("exception_message")) != expected or type(value.get("record_sha256")) is not str:
            raise ValueError("Exp449 verified attempt ledger differs")
        payloads.append(dict(value))
    return tuple(payloads)


def _snapshot_antigravity(config: Exp449WaveConfig) -> _RecoverySnapshot:
    """Reopen the exact paper namespaces before every status or scheduling step."""
    _context_payload, authority = _authority_antigravity(config)
    names = _catalog_datasets_antigravity(config.catalog)
    verified = tuple(verify_task4_task5_success_namespace(
        config.context.run_root, config.catalog, authority, dict(config.raw_datasets)
    ))
    if verified != tuple(sorted(set(verified))) or any(dataset not in names for dataset in verified):
        raise ValueError("Exp449 verified success namespace differs")
    attempt_tuples = tuple(_attempt_tuple_antigravity(record) for record in verify_task4_task5_attempt_namespace(
        config.context.run_root, config.catalog, authority, dict(config.raw_datasets)
    ))
    if tuple(attempt for _dataset, _wave, attempt, _type, _message in attempt_tuples) != tuple(range(1, len(attempt_tuples) + 1)):
        raise ValueError("Exp449 verified attempt identities differ")
    if len({(dataset, wave) for dataset, wave, _attempt, _type, _message in attempt_tuples}) != len(attempt_tuples):
        raise ValueError("Exp449 verified attempt identities differ")
    global_waves = [wave for _dataset, wave, _attempt, _type, _message in attempt_tuples]
    if global_waves and (global_waves[0] != 1 or global_waves != sorted(global_waves)):
        raise ValueError("Exp449 verified attempt identities differ")
    per_dataset_waves: dict[str, list[int]] = {}
    for dataset, wave, _attempt, _type, _message in attempt_tuples:
        per_dataset_waves.setdefault(dataset, []).append(wave)
    if any(waves != sorted(waves) for waves in per_dataset_waves.values()):
        raise ValueError("Exp449 verified attempt identities differ")
    if any(dataset not in names for dataset, _wave, _attempt, _type, _message in attempt_tuples):
        raise ValueError("Exp449 verified attempt namespace differs")
    attempts = _read_verified_attempt_payloads_antigravity(config, attempt_tuples)
    maximum_wave = max((wave for _dataset, wave, _attempt, _type, _message in attempt_tuples), default=0)
    pending = tuple(dataset for dataset in names if dataset not in set(verified))
    attempted_pending = {dataset for dataset, _wave, _attempt, _type, _message in attempt_tuples if dataset in pending}
    failed = tuple(dataset for dataset in pending if dataset in attempted_pending)
    never_attempted = tuple(dataset for dataset in pending if dataset not in attempted_pending)
    return _RecoverySnapshot(verified, attempts, failed, never_attempted, maximum_wave)


def _terminal_status_antigravity(config: Exp449WaveConfig, snapshot: _RecoverySnapshot) -> Exp449TerminalStatus:
    names = _catalog_datasets_antigravity(config.catalog)
    if len(snapshot.successes) == len(names):
        # A success-only first wave has no attempt ledger entry.  After a retry,
        # the highest failed wave plus one is the completed terminal wave.
        wave_count = min(FROZEN_EXP449_CONFIG.maximum_waves, max(1, snapshot.wave_count + 1))
        return Exp449TerminalStatus(config.context.mode, "dataset_local_complete", len(names), 0, 0, wave_count)
    return Exp449TerminalStatus(
        config.context.mode,
        "proposal_score_incomplete",
        len(snapshot.successes),
        len(snapshot.failed),
        len(snapshot.never_attempted),
        min(snapshot.wave_count, FROZEN_EXP449_CONFIG.maximum_waves),
    )


def _terminal_record_antigravity(config: Exp449WaveConfig, snapshot: _RecoverySnapshot) -> dict[str, object]:
    context_payload = task45_context_payload_antigravity(config.context)
    ledger = [
        _json_plain_antigravity(attempt)
        for attempt in snapshot.attempts
    ]
    payload: dict[str, object] = {
        "schema": _TERMINAL_SCHEMA,
        "context_sha256": context_payload["context_sha256"],
        "catalog_sha256": config.context.catalog_sha256,
        "verified_successes": list(snapshot.successes),
        "failed": list(snapshot.failed),
        "never_attempted": list(snapshot.never_attempted),
        "attempt_ledger_sha256": sha256(canonical_json_bytes(ledger)).hexdigest(),
    }
    payload["record_sha256"] = sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def verify_terminal_incomplete_antigravity(config: Exp449WaveConfig) -> Exp449TerminalStatus:
    """Create or byte-verify the only allowed terminal incomplete record."""
    snapshot = _snapshot_antigravity(config)
    status = _terminal_status_antigravity(config, snapshot)
    if status.status != "proposal_score_incomplete" or snapshot.wave_count < FROZEN_EXP449_CONFIG.maximum_waves:
        raise ValueError("Exp449 terminal incomplete state is not established")
    root = _operations_root_for_context_antigravity(config.context)
    write_create_only(root / _TERMINAL_NAME, canonical_json_bytes(_terminal_record_antigravity(config, snapshot)), reuse_identical=True)
    return status


def _existing_terminal_status_antigravity(config: Exp449WaveConfig) -> Exp449TerminalStatus | None:
    """Treat any present terminal artifact as permanent, byte-verified authority."""
    path = _operations_root_for_context_antigravity(config.context) / _TERMINAL_NAME
    try:
        details = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("Exp449 terminal authority is unsafe")
    snapshot = _snapshot_antigravity(config)
    status = _terminal_status_antigravity(config, snapshot)
    if status.status != "proposal_score_incomplete" or snapshot.wave_count != FROZEN_EXP449_CONFIG.maximum_waves:
        raise ValueError("Exp449 terminal authority differs from recovery state")
    expected = canonical_json_bytes(_terminal_record_antigravity(config, snapshot))
    if path.read_bytes() != expected:
        raise ValueError("Exp449 terminal authority differs")
    return status


def _mkdirs_without_symlinks_antigravity(directory: Path) -> None:
    directory = Path(directory)
    current = Path(directory.anchor) if directory.is_absolute() else Path.cwd()
    parts = directory.parts[1:] if directory.is_absolute() else directory.parts
    for part in parts:
        current /= part
        try:
            details = current.lstat()
        except FileNotFoundError:
            current.mkdir()
            details = current.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ValueError("Exp449 operations root is unsafe")


def _validate_progress_event_antigravity(event: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(event, Mapping):
        raise ValueError("Exp449 progress event keys differ")
    if set(event) & _FORBIDDEN_PROGRESS_KEYS:
        raise ValueError("Exp449 progress event contains forbidden scientific fields")
    if not (_REQUIRED_PROGRESS_KEYS <= set(event) <= _PROGRESS_KEYS):
        raise ValueError("Exp449 progress event keys differ")
    normalized = dict(event)
    if type(normalized["timestamp"]) is not str or not normalized["timestamp"] or normalized["experiment"] != "Exp449" or normalized["mode"] not in {"smoke", "official"}:
        raise ValueError("Exp449 progress event identity differs")
    for key in ("wave", "completed", "failed", "pending"):
        if type(normalized[key]) is not int or normalized[key] < 0:
            raise ValueError("Exp449 progress event count differs")
    for key in ("elapsed_seconds", "eta_seconds"):
        if type(normalized[key]) not in (int, float) or not isfinite(float(normalized[key])) or float(normalized[key]) < 0.0:
            raise ValueError("Exp449 progress event timing differs")
        normalized[key] = float(normalized[key])
    if "dataset" in normalized and (type(normalized["dataset"]) is not str or not normalized["dataset"]):
        raise ValueError("Exp449 progress event dataset differs")
    return normalized


def _write_exp449_progress_event_antigravity(operations_root: Path, event: Mapping[str, object]) -> None:
    """Append one validated result-neutral event outside the scientific root."""
    normalized = _validate_progress_event_antigravity(event)
    root = Path(operations_root)
    _mkdirs_without_symlinks_antigravity(root)
    path = root / _PROGRESS_NAME
    try:
        details = path.lstat()
    except FileNotFoundError:
        details = None
    if details is not None and (stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode)):
        raise ValueError("Exp449 progress log is unsafe")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        line = memoryview(canonical_json_bytes(normalized))
        offset = 0
        while offset < len(line):
            written = os.write(descriptor, line[offset:])
            if written <= 0:
                raise OSError("Exp449 progress log short write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _emit_progress_antigravity(config: Exp449WaveConfig, *, wave: int, snapshot: _RecoverySnapshot, started: float, completed: int | None = None, total: int | None = None, failed: int | None = None, dataset: str | None = None) -> None:
    """The logging boundary is the sole place where logging errors are swallowed."""
    completed_count = len(snapshot.successes) if completed is None else completed
    failed_count = len(snapshot.failed) if failed is None else failed
    total_count = len(_catalog_datasets_antigravity(config.catalog)) if total is None else total
    elapsed = float(monotonic() - started)
    rate = completed_count / elapsed if elapsed > 0.0 else 0.0
    event: dict[str, object] = {
        "timestamp": f"{time():.6f}",
        "experiment": "Exp449",
        "mode": config.context.mode,
        "wave": wave,
        "completed": completed_count,
        "failed": failed_count,
        "pending": max(0, total_count - completed_count),
        "elapsed_seconds": elapsed,
        "eta_seconds": (max(0, total_count - completed_count) / rate) if rate > 0.0 else 0.0,
    }
    if dataset is not None:
        event["dataset"] = dataset
    try:
        _write_exp449_progress_event_antigravity(_operations_root_for_context_antigravity(config.context), event)
    except Exception:
        return


def status_exp449_antigravity(config: Exp449WaveConfig) -> Exp449TerminalStatus:
    """Return status from freshly verified records; attempts never imply success."""
    terminal = _existing_terminal_status_antigravity(config)
    if terminal is not None:
        return terminal
    return _terminal_status_antigravity(config, _snapshot_antigravity(config))


def run_exp449_wave_antigravity(config: Exp449WaveConfig) -> Exp449TerminalStatus:
    """Execute at most one of the two preregistered dataset-local recovery waves."""
    started = monotonic()
    terminal = _existing_terminal_status_antigravity(config)
    if terminal is not None:
        return terminal
    before = _snapshot_antigravity(config)
    before_status = _terminal_status_antigravity(config, before)
    if before_status.status == "dataset_local_complete":
        return before_status
    next_wave = before.wave_count + 1
    if next_wave > FROZEN_EXP449_CONFIG.maximum_waves:
        return verify_terminal_incomplete_antigravity(config)
    _emit_progress_antigravity(config, wave=next_wave, snapshot=before, started=started)
    context_payload, _authority = _authority_antigravity(config)
    progress_failures = 0
    def fixed_progress(event: Mapping[str, object]) -> None:
        nonlocal progress_failures
        if event.get("stage") != "task4_task5" or event.get("phase") not in {"start", "error", "end"}:
            return
        if event.get("phase") == "error":
            progress_failures += 1
        _emit_progress_antigravity(
            config, wave=next_wave, snapshot=before, started=started,
            completed=len(before.successes), total=len(_catalog_datasets_antigravity(config.catalog)),
            failed=progress_failures, dataset=event.get("dataset") if event.get("phase") == "error" and type(event.get("dataset")) is str else None,
        )
    try:
        compose_exp449_proposal_score_antigravity(
            config.raw_datasets,
            config.acceleration_manifest,
            root=config.context.run_root,
            context=context_payload,
            catalog=config.catalog,
            _progress_callback=fixed_progress,
        )
    except ProposalScoreIncomplete:
        after = _snapshot_antigravity(config)
        if after.wave_count < next_wave:
            raise ValueError("Exp449 incomplete collection lacks an append-only attempt record")
        for index, dataset in enumerate((name for name in after.successes if name not in before.successes), len(before.successes) + 1):
            _emit_progress_antigravity(config, wave=next_wave, snapshot=after, started=started, completed=index, total=len(_catalog_datasets_antigravity(config.catalog)), failed=len(after.failed), dataset=dataset)
        _emit_progress_antigravity(config, wave=next_wave, snapshot=after, started=started)
        if next_wave == FROZEN_EXP449_CONFIG.maximum_waves:
            return verify_terminal_incomplete_antigravity(config)
        return _terminal_status_antigravity(config, after)
    after = _snapshot_antigravity(config)
    status = _terminal_status_antigravity(config, after)
    if status.status != "dataset_local_complete":
        raise ValueError("Exp449 composition returned before verified dataset-local closure")
    for index, dataset in enumerate((name for name in after.successes if name not in before.successes), len(before.successes) + 1):
        _emit_progress_antigravity(config, wave=next_wave, snapshot=after, started=started, completed=index, total=len(_catalog_datasets_antigravity(config.catalog)), failed=0, dataset=dataset)
    _emit_progress_antigravity(config, wave=next_wave, snapshot=after, started=started)
    return status


__all__ = [
    "Exp449WaveConfig",
    "run_exp449_wave_antigravity",
    "status_exp449_antigravity",
    "task45_context_payload_antigravity",
    "verify_terminal_incomplete_antigravity",
]
