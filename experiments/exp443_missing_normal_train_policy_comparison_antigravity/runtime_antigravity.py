"""Parent-owned, ordered CPU runtime for the Exp443 policy comparison."""
from __future__ import annotations

# These controls must be installed before importing NumPy, aeon, or the
# official MiniROCKET scorer in this module or a spawned worker.
import os

_THREAD_VARIABLES_ANTIGRAVITY = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
)
for _name_antigravity in _THREAD_VARIABLES_ANTIGRAVITY:
    os.environ[_name_antigravity] = "1"

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from multiprocessing import get_context
from pathlib import Path
import signal
import time
from typing import Any

from .artifacts_antigravity import canonical_json_bytes_antigravity, write_create_only_antigravity
from .checkpoint_antigravity import (
    CheckpointBindingAntigravity,
    Exp443CheckpointStoreAntigravity,
    ProcessIdentityAntigravity,
    WorkUnitAntigravity,
    ordered_work_units_antigravity,
)
from .config_antigravity import FROZEN_CONFIG, cpu_worker_count_antigravity, scientific_config_sha256_antigravity
from .contracts_antigravity import PolicyDatasetResultAntigravity
from .progress_antigravity import ProgressWriterAntigravity


def _worker_initializer_antigravity() -> None:
    """Pin every numerical thread pool to one thread in each child."""
    for name in _THREAD_VARIABLES_ANTIGRAVITY:
        os.environ[name] = "1"


@dataclass(frozen=True)
class PolicyTaskAntigravity:
    """One picklable raw-data policy task; no TEST labels are present."""

    unit: WorkUnitAntigravity
    raw: object

    def __post_init__(self) -> None:
        if type(self.unit) is not WorkUnitAntigravity:
            raise ValueError("policy task unit is invalid")
        if getattr(self.raw, "dataset", None) != self.unit.dataset:
            raise ValueError("policy task dataset identity differs")


@dataclass(frozen=True)
class WorkerParityEvidenceAntigravity:
    """Semantic parity evidence with timing kept informational."""

    unit_count: int
    semantic_equal: bool
    sequential_payload_sha256: str
    parallel_payload_sha256: str
    sequential_elapsed: float
    parallel_elapsed: float
    speed_ratio: float

    def __post_init__(self) -> None:
        if type(self.unit_count) is not int or self.unit_count < 1:
            raise ValueError("parity unit count is invalid")
        if type(self.semantic_equal) is not bool:
            raise ValueError("parity semantic flag is invalid")
        for value in (self.sequential_elapsed, self.parallel_elapsed, self.speed_ratio):
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise ValueError("parity timing is invalid")

    def to_payload_antigravity(self) -> dict[str, object]:
        return {
            "unit_count": self.unit_count,
            "semantic_equal": self.semantic_equal,
            "sequential_payload_sha256": self.sequential_payload_sha256,
            "parallel_payload_sha256": self.parallel_payload_sha256,
            "sequential_elapsed": self.sequential_elapsed,
            "parallel_elapsed": self.parallel_elapsed,
            "speed_ratio": self.speed_ratio,
        }


class WorkerFailureAntigravity(RuntimeError):
    """A worker failed before its result could extend the ordered prefix."""

    def __init__(self, unit: WorkUnitAntigravity, cause: BaseException) -> None:
        self.unit = unit
        self.cause = cause
        super().__init__(str(cause) or type(cause).__name__)


class _InterruptedRuntimeAntigravity(KeyboardInterrupt):
    """Internal SIGTERM exception that preserves a resumable prefix."""


def score_policy_task_antigravity(task: PolicyTaskAntigravity) -> PolicyDatasetResultAntigravity:
    """Prepare and score one raw dataset using exactly the frozen scorer."""
    if type(task) is not PolicyTaskAntigravity:
        raise ValueError("policy worker task is invalid")
    from .minirocket_antigravity import score_prepared_dataset_antigravity
    from .preprocessing_antigravity import prepare_dataset_antigravity

    prepared = prepare_dataset_antigravity(task.raw, task.unit.policy)
    result = score_prepared_dataset_antigravity(prepared)
    if type(result) is not PolicyDatasetResultAntigravity or result.policy != task.unit.policy or result.dataset != task.unit.dataset:
        raise ValueError("policy worker returned the wrong result identity")
    return result


def _validate_schedule_units_antigravity(units: Sequence[WorkUnitAntigravity]) -> tuple[WorkUnitAntigravity, ...]:
    if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
        raise ValueError("policy schedule units are invalid")
    normalized = tuple(units)
    sequences = tuple(unit.sequence for unit in normalized)
    if not normalized or any(left >= right for left, right in zip(sequences, sequences[1:])):
        raise ValueError("policy schedule sequence is invalid")
    for index, unit in enumerate(normalized):
        if type(unit) is not WorkUnitAntigravity:
            raise ValueError("policy schedule unit identity is invalid")
        if index and unit.policy == normalized[index - 1].policy and unit.dataset <= normalized[index - 1].dataset:
            raise ValueError("policy schedule dataset order is not case-sensitive sorted")
        if index and FROZEN_CONFIG.policies.index(unit.policy) < FROZEN_CONFIG.policies.index(normalized[index - 1].policy):
            raise ValueError("policy schedule policy-major order is invalid")
    return normalized


def _executor_kwargs_antigravity(workers: int) -> dict[str, object]:
    if workers != FROZEN_CONFIG.outer_workers or cpu_worker_count_antigravity(FROZEN_CONFIG.logical_cpu_count) != workers:
        raise ValueError("Exp443 workers are frozen at exactly 11")
    return {
        "max_workers": workers,
        "initializer": _worker_initializer_antigravity,
        "mp_context": get_context("spawn"),
    }


def run_ordered_policy_schedule_antigravity(
    *,
    units: Sequence[WorkUnitAntigravity],
    task_loader: Callable[[WorkUnitAntigravity], object],
    score_task: Callable[[object], PolicyDatasetResultAntigravity] = score_policy_task_antigravity,
    commit: Callable[[WorkUnitAntigravity, PolicyDatasetResultAntigravity], None],
    progress: Callable[..., None] | None = None,
    workers: int = FROZEN_CONFIG.outer_workers,
    executor_factory: Callable[..., object] = ProcessPoolExecutor,
) -> tuple[PolicyDatasetResultAntigravity, ...]:
    """Submit concurrently, buffer out-of-order results, commit one prefix."""
    frozen_units = _validate_schedule_units_antigravity(units)
    if not all(callable(item) for item in (task_loader, score_task, commit, executor_factory)):
        raise ValueError("Exp443 scheduler callable is invalid")
    if type(workers) is not int or workers != FROZEN_CONFIG.outer_workers:
        raise ValueError("Exp443 scheduler requires exactly 11 workers")
    capacity = workers
    kwargs = _executor_kwargs_antigravity(workers)
    # A small fake executor used by fixture tests may record the literal spawn
    # request rather than constructing a multiprocessing context.  Production
    # ProcessPoolExecutor receives a real SpawnContext.
    try:
        executor = executor_factory(**kwargs)
    except TypeError:
        kwargs["mp_context"] = "spawn"
        executor = executor_factory(**kwargs)
    pending: dict[Future[Any], tuple[int, WorkUnitAntigravity]] = {}
    buffered: dict[int, PolicyDatasetResultAntigravity] = {}
    committed: list[PolicyDatasetResultAntigravity] = []
    submit_index = 0
    commit_index = 0

    def notify(stage: str, unit: WorkUnitAntigravity | None = None) -> None:
        if progress is None:
            return
        progress(stage=stage, unit=unit, pending=tuple(item[1].sequence for item in pending.values()), buffered=tuple(sorted(buffered)))

    def collect_done(*, wait_for_one: bool) -> None:
        nonlocal commit_index
        if not pending:
            return
        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED) if wait_for_one else ({future for future in pending if future.done()}, set())
        for future in done:
            local_index, unit = pending.pop(future)
            notify("worker_result", unit)
            try:
                result = future.result()
            except BaseException as error:
                raise WorkerFailureAntigravity(unit, error) from error
            if type(result) is not PolicyDatasetResultAntigravity or result.policy != unit.policy or result.dataset != unit.dataset:
                raise WorkerFailureAntigravity(unit, ValueError("worker result identity mismatch"))
            buffered[local_index] = result
        while commit_index < len(frozen_units) and commit_index in buffered:
            unit = frozen_units[commit_index]
            result = buffered.pop(commit_index)
            notify("commit", unit)
            commit(unit, result)
            committed.append(result)
            commit_index += 1

    try:
        with executor:
            while submit_index < len(frozen_units) or pending:
                # Submit one at a time until the exact worker capacity is
                # reached.  Immediately completed fixture futures are drained
                # before submitting a later unit, making terminal worker errors
                # stop further submissions deterministically.
                while submit_index < len(frozen_units) and len(pending) < capacity:
                    unit = frozen_units[submit_index]
                    notify("submit", unit)
                    task = task_loader(unit)
                    future = executor.submit(score_task, task)
                    pending[future] = (submit_index, unit)
                    submit_index += 1
                    collect_done(wait_for_one=False)
                    if not pending and commit_index < submit_index:
                        # All immediately completed work was consumed; continue
                        # filling the worker pool for real executors.
                        continue
                if pending:
                    collect_done(wait_for_one=True)
                elif submit_index < len(frozen_units):
                    continue
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    if commit_index != len(frozen_units) or buffered or pending:
        raise RuntimeError("Exp443 ordered scheduler did not complete the prefix")
    return tuple(committed)


def _semantic_payload_antigravity(value: object) -> object:
    """Drop only timing/PID fields; preserve scores, candidates, controls/routes."""
    ignored = {
        "pid", "process", "timing", "heartbeat", "elapsed_seconds", "throughput",
        "speed_ratio", "sequential_elapsed", "parallel_elapsed",
    }
    if isinstance(value, Mapping):
        return {str(key): _semantic_payload_antigravity(item) for key, item in sorted(value.items(), key=lambda item: str(item[0])) if key not in ignored}
    if isinstance(value, (tuple, list)):
        return [_semantic_payload_antigravity(item) for item in value]
    return value


def _payload_digest_antigravity(value: object) -> tuple[str, bytes]:
    encoded = canonical_json_bytes_antigravity(_semantic_payload_antigravity(value))
    return sha256(encoded).hexdigest(), encoded


def worker_parity_antigravity(
    units: Sequence[WorkUnitAntigravity],
    *,
    sequential_runner: Callable[[], object],
    parallel_runner: Callable[[], object],
    sequential_elapsed: float,
    parallel_elapsed: float,
) -> WorkerParityEvidenceAntigravity:
    """Require byte-identical semantic payloads for sequential/parallel runs."""
    frozen_units = _validate_schedule_units_antigravity(units)
    if not callable(sequential_runner) or not callable(parallel_runner):
        raise ValueError("worker parity runners are invalid")
    for value in (sequential_elapsed, parallel_elapsed):
        if type(value) not in (int, float) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError("worker parity elapsed time is invalid")
    sequential = sequential_runner()
    parallel = parallel_runner()
    sequential_digest, sequential_bytes = _payload_digest_antigravity(sequential)
    parallel_digest, parallel_bytes = _payload_digest_antigravity(parallel)
    equal = sequential_bytes == parallel_bytes
    if not equal:
        raise ValueError("worker parity semantic payload mismatch")
    return WorkerParityEvidenceAntigravity(
        len(frozen_units), True, sequential_digest, parallel_digest,
        float(sequential_elapsed), float(parallel_elapsed), float(sequential_elapsed) / float(parallel_elapsed),
    )


def run_policy_a_worker_parity_antigravity(
    units: Sequence[WorkUnitAntigravity],
    *,
    task_loader: Callable[[WorkUnitAntigravity], object],
    score_task: Callable[[object], PolicyDatasetResultAntigravity] = score_policy_task_antigravity,
    workers: int = FROZEN_CONFIG.outer_workers,
    executor_factory: Callable[..., object] = ProcessPoolExecutor,
    payload_factory: Callable[[Mapping[int, PolicyDatasetResultAntigravity]], object] | None = None,
) -> WorkerParityEvidenceAntigravity:
    """Run one exact Policy A unit set serially and with 11 spawn workers."""
    frozen_units = _validate_schedule_units_antigravity(units)
    if any(unit.policy != FROZEN_CONFIG.policies[0] for unit in frozen_units):
        raise ValueError("Policy A parity requires only the control policy")

    def serial() -> object:
        results: dict[int, PolicyDatasetResultAntigravity] = {}
        for unit in frozen_units:
            results[unit.sequence] = score_task(task_loader(unit))
        return payload_factory(results) if payload_factory else {sequence: result.to_payload() for sequence, result in results.items()}

    serial_start = time.perf_counter()
    serial_payload = serial()
    serial_elapsed = time.perf_counter() - serial_start
    parallel_results: dict[int, PolicyDatasetResultAntigravity] = {}
    parallel_start = time.perf_counter()
    run_ordered_policy_schedule_antigravity(
        units=frozen_units,
        task_loader=task_loader,
        score_task=score_task,
        commit=lambda unit, result: parallel_results.__setitem__(unit.sequence, result),
        workers=workers,
        executor_factory=executor_factory,
    )
    parallel_elapsed = time.perf_counter() - parallel_start
    parallel_payload = payload_factory(parallel_results) if payload_factory else {sequence: result.to_payload() for sequence, result in parallel_results.items()}
    return worker_parity_antigravity(
        frozen_units,
        sequential_runner=lambda: serial_payload,
        parallel_runner=lambda: parallel_payload,
        sequential_elapsed=max(serial_elapsed, 1e-12),
        parallel_elapsed=max(parallel_elapsed, 1e-12),
    )


def _process_identity_antigravity() -> ProcessIdentityAntigravity:
    """Create a resumable parent identity; Linux start ticks prevent PID reuse."""
    pid = os.getpid()
    start_identity = f"pid-{pid}"
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        fields = proc_stat.read_text().split()
        if len(fields) > 21:
            start_identity = f"pid-{pid}-start-{fields[21]}"
    except (OSError, ValueError):
        pass
    return ProcessIdentityAntigravity(pid, start_identity)


def _pid_probe_antigravity(process: ProcessIdentityAntigravity) -> bool:
    """Best-effort active PID/start identity probe for checkpoint resume."""
    try:
        os.kill(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    proc_stat = Path(f"/proc/{process.pid}/stat")
    if proc_stat.is_file() and "-start-" in process.start_identity:
        try:
            fields = proc_stat.read_text().split()
            return len(fields) > 21 and process.start_identity.endswith(f"-start-{fields[21]}")
        except OSError:
            return True
    return True


def _load_readiness_antigravity(root: Path) -> dict[str, object]:
    from .artifacts_antigravity import parse_canonical_json_bytes_antigravity
    from .readiness_antigravity import validate_readiness_payload_antigravity

    path = root / "readiness_antigravity.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Exp443 readiness evidence is absent")
    payload = parse_canonical_json_bytes_antigravity(path.read_bytes())
    if not isinstance(payload, Mapping):
        raise ValueError("Exp443 readiness evidence is invalid")
    return validate_readiness_payload_antigravity(payload)


def _source_snapshot_and_environment_antigravity(reviewed_commit: str, readiness: Mapping[str, object]) -> object:
    """Revalidate source/config/environment at every scheduling boundary."""
    from .environment_antigravity import verify_environment_antigravity
    from .readiness_antigravity import _source_evidence_antigravity
    from .sources_antigravity import verify_frozen_sources_antigravity

    if reviewed_commit != readiness.get("reviewed_commit"):
        raise ValueError("runtime reviewed commit differs from readiness")
    FROZEN_CONFIG.validate()
    environment = verify_environment_antigravity()
    if environment.outer_workers != FROZEN_CONFIG.outer_workers or environment.inner_threads != FROZEN_CONFIG.inner_threads:
        raise ValueError("runtime environment topology differs")
    snapshot = verify_frozen_sources_antigravity()
    evidence = _source_evidence_antigravity(snapshot)
    source_digest = sha256(canonical_json_bytes_antigravity(evidence)).hexdigest()
    if source_digest != readiness.get("source_snapshot_sha256") or snapshot.label_access_count != 0:
        raise ValueError("runtime source snapshot differs from readiness")
    return snapshot


def _attempt_root_antigravity(root: Path) -> Path:
    attempts = root / "attempts_antigravity"
    if attempts.exists() and (attempts.is_symlink() or not attempts.is_dir()):
        raise ValueError("Exp443 attempt root is invalid")
    attempts.mkdir(mode=0o700, exist_ok=True)
    present: set[int] = set()
    for entry in attempts.iterdir():
        if entry.is_symlink() or not entry.is_dir() or not entry.name.startswith("attempt_") or not entry.name.endswith("_antigravity"):
            raise ValueError("Exp443 attempt inventory is invalid")
        middle = entry.name[len("attempt_"):-len("_antigravity")]
        if len(middle) != 4 or not middle.isdigit():
            raise ValueError("Exp443 attempt inventory is invalid")
        present.add(int(middle))
    sequence = 0
    while sequence in present:
        sequence += 1
    attempt = attempts / f"attempt_{sequence:04d}_antigravity"
    attempt.mkdir(mode=0o700)
    return attempt


def _binding_antigravity(reviewed_commit: str, readiness: Mapping[str, object]) -> CheckpointBindingAntigravity:
    return CheckpointBindingAntigravity(
        run_identity=FROZEN_CONFIG.run_identity,
        scientific_config_sha256=scientific_config_sha256_antigravity(),
        environment_lock_sha256=FROZEN_CONFIG.environment_lock_sha256,
        source_snapshot_sha256=str(readiness["source_snapshot_sha256"]),
        database_sha256=FROZEN_CONFIG.database_sha256,
        reviewed_commit=reviewed_commit,
    )


def _control_payload_antigravity(control: object) -> dict[str, object]:
    serializer = getattr(control, "to_payload_antigravity", None)
    if not callable(serializer):
        raise ValueError("control result has no canonical serializer")
    payload = serializer()
    if not isinstance(payload, dict):
        raise ValueError("control result canonical payload is invalid")
    return payload


def _write_prediction_idempotent_antigravity(path: Path, payload: bytes) -> None:
    """Create a prediction artifact or reuse exactly matching finalization bytes."""
    if type(payload) is not bytes or not payload:
        raise ValueError("prediction publication bytes are invalid")
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("prediction publication path is invalid")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("prediction publication bytes changed")
        return
    write_create_only_antigravity(path, payload)


def _mark_interrupted_antigravity(
    store: Exp443CheckpointStoreAntigravity,
    writer: ProgressWriterAntigravity,
    message: str,
) -> None:
    """Close both Task 7 authorities as one resumable interruption."""
    if store.status_antigravity not in {"error", "complete"}:
        store.mark_interrupted_resumable_antigravity()
    writer.terminal_antigravity("interrupted_resumable", message or "interrupted")


def _record_terminal_error_antigravity(
    store: Exp443CheckpointStoreAntigravity,
    writer: ProgressWriterAntigravity,
    units: tuple[WorkUnitAntigravity, ...],
    error: BaseException,
) -> None:
    """Write one ordered terminal error and close the parent progress log."""
    if store.status_antigravity not in {"error", "complete"} and store.completed_count_antigravity < len(units):
        next_sequence = store.completed_count_antigravity
        store.commit_error_antigravity(next_sequence, units[next_sequence], error)
    writer.terminal_antigravity("error", str(error) or type(error).__name__)


def _execute_policy_phases_antigravity(
    *,
    units: tuple[WorkUnitAntigravity, ...],
    task_loader: Callable[[WorkUnitAntigravity], object],
    commit: Callable[[WorkUnitAntigravity, PolicyDatasetResultAntigravity], None],
    progress: Callable[..., None] | None,
    control_runner: Callable[[Mapping[str, PolicyDatasetResultAntigravity]], object],
    revalidate: Callable[[], object],
    initial_results: Mapping[int, PolicyDatasetResultAntigravity] | None = None,
    score_task: Callable[[object], PolicyDatasetResultAntigravity] = score_policy_task_antigravity,
    executor_factory: Callable[..., object] = ProcessPoolExecutor,
) -> tuple[dict[int, PolicyDatasetResultAntigravity], object]:
    """Run A, require Task 6 control, then admit B/C in frozen order."""
    results = dict(initial_results or {})
    control: object | None = None
    for policy_index, policy in enumerate(FROZEN_CONFIG.policies):
        phase = tuple(unit for unit in units if unit.policy == policy and unit.sequence not in results)
        if phase:
            run_ordered_policy_schedule_antigravity(
                units=phase,
                task_loader=task_loader,
                score_task=score_task,
                commit=lambda unit, result: (results.__setitem__(unit.sequence, result), commit(unit, result)),
                progress=progress,
                workers=FROZEN_CONFIG.outer_workers,
                executor_factory=executor_factory,
            )
        if policy_index == 0:
            revalidate()
            a_results = {
                units[sequence].dataset: result
                for sequence, result in results.items()
                if units[sequence].policy == FROZEN_CONFIG.policies[0]
            }
            control = control_runner(a_results)
            from .control_antigravity import require_control_gate_antigravity
            require_control_gate_antigravity(control)
            if not getattr(control, "passed", False):
                raise ValueError("control_reproduction_failed")
            continue
        if policy_index == 1:
            revalidate()
    if control is None:
        raise ValueError("Policy A control was not executed")
    return results, control


def _run_exp443_antigravity(
    *,
    reviewed_commit: str,
    mode: str,
    run_root: Path | str | None = None,
    executor_factory: Callable[..., object] = ProcessPoolExecutor,
) -> dict[str, object]:
    """Execute new or resumable production attempt with parent-owned evidence."""
    from .database_antigravity import (
        load_raw_dataset_antigravity,
        pin_database_snapshot_antigravity,
        reopen_pinned_database_snapshot_antigravity,
    )
    from .readiness_antigravity import current_clean_head_antigravity
    from .sources_antigravity import verify_frozen_sources_antigravity

    root = Path(FROZEN_CONFIG.run_root if run_root is None else run_root)
    readiness = _load_readiness_antigravity(root)
    if current_clean_head_antigravity() != reviewed_commit:
        raise ValueError("runtime requires the exact clean reviewed commit")
    snapshot = _source_snapshot_and_environment_antigravity(reviewed_commit, readiness)
    units = ordered_work_units_antigravity(tuple(snapshot.affected_datasets))
    binding = _binding_antigravity(reviewed_commit, readiness)
    identity = _process_identity_antigravity()
    if mode == "new":
        attempt = _attempt_root_antigravity(root)
        checkpoint_root = attempt / "checkpoint_antigravity"
        store = Exp443CheckpointStoreAntigravity.create_antigravity(checkpoint_root, units, binding, identity)
        writer = ProgressWriterAntigravity(attempt, total=len(units), checkpoint_path=checkpoint_root / "state_antigravity.json", process=identity)
    elif mode == "resume":
        attempts = root / "attempts_antigravity"
        candidates = sorted(path for path in attempts.glob("attempt_*_antigravity") if path.is_dir())
        if not candidates:
            raise ValueError("Exp443 resume attempt is absent")
        attempt = candidates[-1]
        checkpoint_root = attempt / "checkpoint_antigravity"
        store = Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(
            checkpoint_root,
            binding,
            identity,
            _pid_probe_antigravity,
            allow_complete_finalization=True,
        )
        writer = ProgressWriterAntigravity.open_for_resume_antigravity(attempt, total=len(units), checkpoint_path=checkpoint_root / "state_antigravity.json", checkpoint_store=store, process=identity, pid_probe=_pid_probe_antigravity)
    else:
        raise ValueError("Exp443 runtime mode is invalid")

    raw_root = attempt / "database_antigravity"
    if mode == "resume":
        pinned = reopen_pinned_database_snapshot_antigravity(
            FROZEN_CONFIG.database_path,
            raw_root,
            expected_sha256=FROZEN_CONFIG.database_sha256,
        )
    else:
        pinned = pin_database_snapshot_antigravity(FROZEN_CONFIG.database_path, raw_root, expected_sha256=FROZEN_CONFIG.database_sha256)
    raw_cache: dict[str, object] = {}

    def raw_dimensions_antigravity(raw: object) -> dict[str, int | None]:
        train_rows = tuple(getattr(raw, "train", getattr(raw, "train_rows", ())))
        test_rows = tuple(getattr(raw, "test", getattr(raw, "test_rows", ())))
        return {
            "train_total": len(train_rows),
            "train_missing": sum(bool(getattr(row, "missing_positions", ())) for row in train_rows),
            "train_used": None,
            "train_excluded": None,
            "test_rows": len(test_rows),
            "target_length": int(getattr(raw, "target_length")),
        }

    def load_raw(unit: WorkUnitAntigravity) -> object:
        if unit.dataset not in raw_cache:
            raw_cache[unit.dataset] = load_raw_dataset_antigravity(pinned, unit.dataset)
        return raw_cache[unit.dataset]

    def load_task(unit: WorkUnitAntigravity) -> PolicyTaskAntigravity:
        return PolicyTaskAntigravity(unit, load_raw(unit))

    def commit(unit: WorkUnitAntigravity, result: PolicyDatasetResultAntigravity) -> None:
        next_dimensions: dict[str, int | None] | None = None
        next_sequence = store.completed_count_antigravity + 1
        if next_sequence < len(units):
            # Read the next dataset before advancing either authority.  The
            # progress seam below then publishes its dimensions together with
            # the accepted result in one atomic JSON replacement.
            next_dimensions = raw_dimensions_antigravity(load_raw(units[next_sequence]))
        store.commit_result_antigravity(unit.sequence, result)
        writer.accept_unit_and_align_antigravity(result, next_dimensions=next_dimensions)

    def progress(**fields: object) -> None:
        active = tuple(unit for unit in units if unit.sequence >= store.completed_count_antigravity)
        if fields.get("stage") in {"submit", "worker_result"}:
            writer.heartbeat_antigravity(active)

    def control_runner(results: Mapping[str, PolicyDatasetResultAntigravity]) -> object:
        from .control_antigravity import compare_policy_a_to_archive_antigravity
        return compare_policy_a_to_archive_antigravity(snapshot, results)

    prior_sigterm = signal.getsignal(signal.SIGTERM)

    def _on_sigterm_antigravity(_signum: int, _frame: object) -> None:
        raise _InterruptedRuntimeAntigravity("SIGTERM received")

    signal.signal(signal.SIGTERM, _on_sigterm_antigravity)
    finalized = False
    try:
        active_units = tuple(unit for unit in units if unit.sequence >= store.completed_count_antigravity)
        writer.begin_units_antigravity(active_units)
        if active_units:
            writer.align_current_dimensions_antigravity(**raw_dimensions_antigravity(load_raw(active_units[0])))
        initial_results = store.results_antigravity()
        all_results, control = _execute_policy_phases_antigravity(
            units=units, task_loader=load_task, commit=commit, progress=progress,
            control_runner=control_runner, revalidate=lambda: _source_snapshot_and_environment_antigravity(reviewed_commit, readiness),
            initial_results=initial_results, executor_factory=executor_factory,
        )
        control_path = attempt / "control_gate_antigravity.json"
        if not control_path.exists():
            write_create_only_antigravity(control_path, canonical_json_bytes_antigravity(_control_payload_antigravity(control)))
        _source_snapshot_and_environment_antigravity(reviewed_commit, readiness)
        from .routing_antigravity import canonical_prediction_bytes_antigravity, route_policy_antigravity
        predictions: dict[str, str] = {}
        for policy in FROZEN_CONFIG.policies:
            policy_results = {units[sequence].dataset: result for sequence, result in all_results.items() if units[sequence].policy == policy}
            prediction = route_policy_antigravity(snapshot, policy_results, policy)
            path = attempt / f"prediction_{policy}_antigravity.json"
            prediction_bytes = canonical_prediction_bytes_antigravity(prediction)
            _write_prediction_idempotent_antigravity(path, prediction_bytes)
            predictions[policy] = sha256(prediction_bytes).hexdigest()
        _source_snapshot_and_environment_antigravity(reviewed_commit, readiness)
        writer.terminal_antigravity("complete", None)
        finalized = True
        return {
            "attempt": str(attempt), "checkpoint_completed": store.completed_count_antigravity,
            "control": _control_payload_antigravity(control), "prediction_sha256": predictions,
            "label_access_count": 0,
        }
    except WorkerFailureAntigravity as failure:
        _record_terminal_error_antigravity(store, writer, units, failure)
        raise
    except _InterruptedRuntimeAntigravity as interruption:
        _mark_interrupted_antigravity(store, writer, str(interruption))
        raise
    except KeyboardInterrupt as interruption:
        _mark_interrupted_antigravity(store, writer, str(interruption) or "KeyboardInterrupt")
        raise
    except BaseException as error:
        _record_terminal_error_antigravity(store, writer, units, error)
        raise
    finally:
        signal.signal(signal.SIGTERM, prior_sigterm)
        # A complete checkpoint may still be in finalization (control/prediction
        # publication).  Keep its input pin across a resumable interruption;
        # delete it only after the progress terminal-complete write succeeds.
        pinned.close(terminal_evidence_durable=finalized)


def run_exp443_new_antigravity(*, reviewed_commit: str, run_root: Path | str | None = None, executor_factory: Callable[..., object] = ProcessPoolExecutor) -> dict[str, object]:
    return _run_exp443_antigravity(reviewed_commit=reviewed_commit, mode="new", run_root=run_root, executor_factory=executor_factory)


def resume_exp443_antigravity(*, reviewed_commit: str, run_root: Path | str | None = None, executor_factory: Callable[..., object] = ProcessPoolExecutor) -> dict[str, object]:
    return _run_exp443_antigravity(reviewed_commit=reviewed_commit, mode="resume", run_root=run_root, executor_factory=executor_factory)


__all__ = [
    "PolicyTaskAntigravity", "WorkerFailureAntigravity", "WorkerParityEvidenceAntigravity",
    "score_policy_task_antigravity", "run_ordered_policy_schedule_antigravity",
    "worker_parity_antigravity", "run_policy_a_worker_parity_antigravity",
    "_worker_initializer_antigravity", "_THREAD_VARIABLES_ANTIGRAVITY",
    "run_exp443_new_antigravity", "resume_exp443_antigravity",
    "_mark_interrupted_antigravity", "_record_terminal_error_antigravity",
    "_write_prediction_idempotent_antigravity",
]
