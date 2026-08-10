"""Parent-owned deterministic runtime for Exp444.

Only copied values-only candidate inputs cross the process boundary.  Workers
return immutable stability contracts; the parent alone owns checkpoint and
progress writes and commits results in the frozen rank order.  Evaluation is a
post-publication concern and is intentionally not imported here.
"""
from __future__ import annotations

# These controls must be installed before NumPy (and the Exp440 scorer) is
# imported, both in the parent and in every spawn child.
import os

_THREAD_VARIABLES_ANTIGRAVITY = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
for _name_antigravity in _THREAD_VARIABLES_ANTIGRAVITY:
    os.environ[_name_antigravity] = "1"

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
import re
from typing import Any

from .artifacts_antigravity import canonical_json_bytes_antigravity, parse_canonical_json_bytes_antigravity
from .checkpoint_antigravity import (
    Exp444CheckpointStoreAntigravity,
    ProcessIdentityAntigravity,
    current_process_identity_antigravity,
)
from .config_antigravity import FROZEN_CONFIG, scientific_config_sha256_antigravity
from .contracts_antigravity import (
    CandidateInputAntigravity,
    CandidateStabilityResultAntigravity,
    CheckpointContextAntigravity,
)
from .progress_antigravity import ProgressWriterAntigravity


_ATTEMPT_NAME_ANTIGRAVITY = re.compile(r"attempt_[0-9]{4}_antigravity$")


def _worker_initializer_antigravity() -> None:
    """Pin every numerical thread pool to one thread in a spawn child."""
    for name in _THREAD_VARIABLES_ANTIGRAVITY:
        os.environ[name] = "1"


def production_worker_count_antigravity() -> int:
    """Return the frozen production topology (twelve logical CPUs -> eleven)."""
    if FROZEN_CONFIG.cpu_workers != 11:
        raise ValueError("Exp444 production worker contract is not eleven")
    return FROZEN_CONFIG.cpu_workers


class WorkerFailureAntigravity(RuntimeError):
    """A worker or sequential reference failed for one frozen candidate."""

    def __init__(
        self,
        sequence: int,
        candidate: CandidateInputAntigravity,
        cause: BaseException,
        *,
        view_index: int | None = None,
    ) -> None:
        self.sequence = sequence
        self.candidate = candidate.candidate
        self.candidate_index = sequence
        self.dataset = candidate.dataset
        self.instance_index = candidate.instance_index
        self.view_index = view_index
        self.cause = cause
        suffix = f" view={view_index}" if view_index is not None else ""
        message = str(cause) or type(cause).__name__
        super().__init__(
            f"worker failed dataset={candidate.dataset} "
            f"candidate_index={sequence} instance_index={candidate.instance_index}{suffix}: {message}"
        )


def _validate_inputs_antigravity(inputs: Sequence[CandidateInputAntigravity]) -> tuple[CandidateInputAntigravity, ...]:
    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise ValueError("Exp444 candidate inputs are invalid")
    frozen = tuple(inputs)
    if not frozen or any(type(item) is not CandidateInputAntigravity for item in frozen):
        raise ValueError("Exp444 candidate inputs are invalid")
    ranks = tuple(item.candidate.original_rank for item in frozen)
    if ranks != tuple(range(ranks[0], ranks[0] + len(frozen))):
        raise ValueError("Exp444 candidate order must be the exact case-sensitive frozen rank order")
    coordinates = tuple(item.candidate.coordinate for item in frozen)
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("Exp444 candidate coordinates are duplicated")
    return frozen


def _validate_result_antigravity(
    sequence: int,
    candidate: CandidateInputAntigravity,
    result: object,
) -> CandidateStabilityResultAntigravity:
    if type(result) is not CandidateStabilityResultAntigravity:
        raise WorkerFailureAntigravity(sequence, candidate, ValueError("worker returned the wrong result type"))
    if result.candidate != candidate.candidate:
        raise WorkerFailureAntigravity(sequence, candidate, ValueError("worker result candidate identity changed"))
    return result


def _score_candidate_worker_antigravity(
    candidate: CandidateInputAntigravity,
    distance_backend: str = "cpu",
) -> CandidateStabilityResultAntigravity:
    """Spawn-safe worker entrypoint; it receives no SQLite handle or writer."""
    if type(candidate) is not CandidateInputAntigravity or distance_backend != "cpu":
        raise ValueError("Exp444 outer workers support the CPU backend only")
    return score_stability_candidate_antigravity(candidate)


def score_stability_candidate_antigravity(
    candidate: CandidateInputAntigravity,
) -> CandidateStabilityResultAntigravity:
    """Lazy public scorer seam used by workers and deterministic test doubles."""
    # Import after the seven thread controls above; this function itself is
    # picklable and therefore safe to pass through a spawn pool.
    from .stability_antigravity import score_stability_candidate_antigravity as _score

    return _score(candidate)


def _executor_antigravity(factory: Callable[..., object], workers: int) -> object:
    kwargs = {
        "max_workers": workers,
        "initializer": _worker_initializer_antigravity,
        "mp_context": get_context("spawn"),
    }
    try:
        return factory(**kwargs)
    except TypeError:
        # Small fixture executors often record a literal ``spawn`` marker
        # instead of accepting multiprocessing.context.SpawnContext.
        kwargs["mp_context"] = "spawn"
        return factory(**kwargs)


def _done_futures_antigravity(pending: Mapping[Future[Any], object]) -> set[Future[Any]]:
    """Support both real futures and deterministic fixture futures."""
    done = {future for future in pending if callable(getattr(future, "done", None)) and future.done()}
    if done:
        return done
    finished, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
    return set(finished)


def ordered_parallel_map_antigravity(
    inputs: Sequence[CandidateInputAntigravity],
    workers: int = FROZEN_CONFIG.cpu_workers,
    distance_backend: str = "cpu",
    *,
    executor_factory: Callable[..., object] | None = None,
    score_task: Callable[[CandidateInputAntigravity], CandidateStabilityResultAntigravity] | None = None,
    commit: Callable[[int, CandidateInputAntigravity, CandidateStabilityResultAntigravity], None] | None = None,
    progress: Callable[..., None] | None = None,
) -> tuple[CandidateStabilityResultAntigravity, ...]:
    """Compute concurrently with at most ``workers**2`` pending/buffered work.

    Futures may finish out of order, but ``commit`` is called only for the
    contiguous prefix.  The scheduler never receives a database connection,
    source session, label, or progress writer in a child process.
    """
    frozen = _validate_inputs_antigravity(inputs)
    if type(workers) is not int or workers < 1:
        raise ValueError("Exp444 worker count is invalid")
    if distance_backend != "cpu":
        raise ValueError("Exp444 outer ProcessPoolExecutor requires CPU backend")
    if progress is not None and not callable(progress):
        raise ValueError("Exp444 progress callback is invalid")
    if commit is not None and not callable(commit):
        raise ValueError("Exp444 commit callback is invalid")
    factory = ProcessPoolExecutor if executor_factory is None else executor_factory
    scorer = _score_candidate_worker_antigravity if score_task is None else score_task
    if not callable(factory) or not callable(scorer):
        raise ValueError("Exp444 scheduler dependency is invalid")

    pending: dict[Future[Any], tuple[int, CandidateInputAntigravity]] = {}
    buffered: dict[int, CandidateStabilityResultAntigravity] = {}
    committed: list[CandidateStabilityResultAntigravity] = []
    submit_index = 0
    commit_index = 0
    reorder_capacity = workers * workers
    executor = _executor_antigravity(factory, workers)

    def notify(stage: str, sequence: int | None = None) -> None:
        if progress is None:
            return
        progress(
            stage=stage,
            sequence=sequence,
            pending=tuple(item[0] for item in pending.values()),
            buffered=tuple(sorted(buffered)),
        )

    try:
        with executor:  # type: ignore[union-attr]
            while submit_index < len(frozen) or pending or buffered:
                while (
                    submit_index < len(frozen)
                    and len(pending) < workers
                    and len(pending) + len(buffered) < reorder_capacity
                ):
                    candidate = frozen[submit_index]
                    notify("submit", submit_index)
                    future = executor.submit(scorer, candidate)  # type: ignore[union-attr]
                    pending[future] = (submit_index, candidate)
                    submit_index += 1
                if pending:
                    done = _done_futures_antigravity(pending)
                    for future in done:
                        sequence, candidate = pending.pop(future)
                        notify("worker_result", sequence)
                        try:
                            result = future.result()
                        except KeyboardInterrupt:
                            raise
                        except BaseException as error:
                            raise WorkerFailureAntigravity(sequence, candidate, error) from error
                        buffered[sequence] = _validate_result_antigravity(sequence, candidate, result)
                while commit_index in buffered:
                    candidate = frozen[commit_index]
                    result = buffered.pop(commit_index)
                    notify("commit", commit_index)
                    if commit is not None:
                        commit(commit_index, candidate, result)
                    committed.append(result)
                    commit_index += 1
                if not pending and not buffered and submit_index >= len(frozen):
                    break
                if not pending and submit_index < len(frozen):
                    # This path is only reachable with an unusual fixture
                    # executor that completes and removes futures eagerly.
                    continue
    except BaseException:
        for future in pending:
            try:
                future.cancel()
            except BaseException:
                pass
        raise
    if commit_index != len(frozen) or pending or buffered:
        raise RuntimeError("Exp444 ordered scheduler did not complete the frozen prefix")
    return tuple(committed)


def sequential_reference_antigravity(
    inputs: Sequence[CandidateInputAntigravity],
    distance_backend: str = "cpu",
    *,
    score_task: Callable[[CandidateInputAntigravity], CandidateStabilityResultAntigravity] | None = None,
) -> tuple[CandidateStabilityResultAntigravity, ...]:
    """Run the same copied inputs in rank order without creating a process pool."""
    frozen = _validate_inputs_antigravity(inputs)
    if distance_backend != "cpu":
        raise ValueError("Exp444 sequential reference supports CPU backend only")
    scorer = _score_candidate_worker_antigravity if score_task is None else score_task
    result: list[CandidateStabilityResultAntigravity] = []
    for sequence, candidate in enumerate(frozen):
        try:
            result.append(_validate_result_antigravity(sequence, candidate, scorer(candidate)))
        except WorkerFailureAntigravity:
            raise
        except BaseException as error:
            raise WorkerFailureAntigravity(sequence, candidate, error) from error
    return tuple(result)


def canonical_results_bytes_antigravity(
    results: Sequence[CandidateStabilityResultAntigravity],
) -> bytes:
    """Canonical result bytes used by the one-worker/11-worker parity smoke."""
    values = tuple(results)
    if any(type(item) is not CandidateStabilityResultAntigravity for item in values):
        raise ValueError("Exp444 result sequence is invalid")
    return canonical_json_bytes_antigravity([item.to_payload() for item in values])


@dataclass(frozen=True)
class RuntimeOutcomeAntigravity:
    """Small immutable handoff object returned before publication/evaluation."""

    attempt_path: Path
    checkpoint_path: Path
    progress_path: Path | None
    results: tuple[CandidateStabilityResultAntigravity, ...]
    status: str
    completed_count: int
    workers: int
    backend: str
    label_access_count: int
    resumed: bool = False
    finalization_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_path, Path) or not isinstance(self.checkpoint_path, Path):
            raise ValueError("runtime paths are invalid")
        if self.progress_path is not None and not isinstance(self.progress_path, Path):
            raise ValueError("runtime progress path is invalid")
        results = tuple(self.results)
        if any(type(item) is not CandidateStabilityResultAntigravity for item in results):
            raise ValueError("runtime results are invalid")
        if type(self.completed_count) is not int or self.completed_count != len(results):
            raise ValueError("runtime completed count is invalid")
        if self.status not in {"running", "interrupted", "error", "complete"}:
            raise ValueError("runtime status is invalid")
        if self.backend != "cpu" or type(self.workers) is not int or self.workers < 1:
            raise ValueError("runtime backend/topology is invalid")
        if type(self.label_access_count) is not int or self.label_access_count != 0:
            raise ValueError("runtime label access is invalid")
        if type(self.resumed) is not bool or type(self.finalization_only) is not bool:
            raise ValueError("runtime flags are invalid")
        object.__setattr__(self, "results", results)


def _context_antigravity() -> CheckpointContextAntigravity:
    return CheckpointContextAntigravity(
        FROZEN_CONFIG.experiment_id,
        FROZEN_CONFIG.run_identity,
        scientific_config_sha256_antigravity(),
        FROZEN_CONFIG.environment_lock_sha256,
        FROZEN_CONFIG.source_candidate_count,
        FROZEN_CONFIG.cpu_workers,
        FROZEN_CONFIG.inner_threads,
    )


def _attempt_root_antigravity(run_root: Path) -> Path:
    if "antigravity" not in run_root.name or run_root.exists() or run_root.is_symlink():
        raise ValueError("Exp444 new run root must be absent and antigravity-namespaced")
    run_root.mkdir(mode=0o700, parents=True)
    attempts = run_root / "attempts_antigravity"
    attempts.mkdir(mode=0o700)
    attempt = attempts / "attempt_0000_antigravity"
    attempt.mkdir(mode=0o700)
    return attempt


def _resume_attempt_antigravity(run_root: Path, attempt_name: str) -> Path:
    if type(attempt_name) is not str or _ATTEMPT_NAME_ANTIGRAVITY.fullmatch(attempt_name) is None:
        raise ValueError("Exp444 resume attempt basename is invalid")
    attempts = run_root / "attempts_antigravity"
    attempt = attempts / attempt_name
    if attempts.is_symlink() or not attempts.is_dir() or attempt.is_symlink() or not attempt.is_dir():
        raise ValueError("Exp444 resume attempt is absent or invalid")
    return attempt


def _load_source_inputs_antigravity() -> tuple[tuple[CandidateInputAntigravity, ...], int]:
    """Verify and materialize values, then close the parent DB session."""
    from .sources_antigravity import load_candidate_values_antigravity, verify_frozen_sources_antigravity

    snapshot = verify_frozen_sources_antigravity()
    session = snapshot.session
    try:
        inputs = load_candidate_values_antigravity(snapshot)
        label_access_count = snapshot.label_access_count
        if label_access_count != 0 or session is None or session.label_access_count != 0:
            raise ValueError("Exp444 source materialization accessed TEST labels")
        return _validate_inputs_antigravity(inputs), label_access_count
    finally:
        if session is not None:
            session.close()


def _dimensions_antigravity(candidate: CandidateInputAntigravity) -> dict[str, int]:
    return {
        "train_count": int(candidate.train_values.shape[0]),
        "test_count": 1,
        "test_rows": 1,
        "raw_length": candidate.raw_length,
        "target_length": candidate.raw_length,
    }


def _persist_commit_antigravity(
    store: Exp444CheckpointStoreAntigravity,
    writer: ProgressWriterAntigravity,
    sequence: int,
    candidate: CandidateInputAntigravity,
    result: CandidateStabilityResultAntigravity,
) -> None:
    dimensions = _dimensions_antigravity(candidate)
    writer.start_candidate_antigravity(
        candidate.dataset,
        sequence,
        train_count=dimensions["train_count"],
        test_count=dimensions["test_count"],
        test_rows=dimensions["test_rows"],
        raw_length=dimensions["raw_length"],
        target_length=dimensions["target_length"],
        backend="cpu",
    )
    for view in result.views:
        writer.start_view_antigravity(
            view_index=view.view_index,
            view_count=len(result.views),
            excluded_count=len(view.mask.excluded_train_indices),
        )
    # Durable result first; progress is advanced only after checkpoint bytes
    # and its state prefix have both been fsynced.
    store.commit_antigravity(sequence, result)
    writer.complete_candidate_antigravity(
        "abstained" if result.lane == "unstable_review" else "success",
        recomputed=True,
        abstention=result.lane == "unstable_review",
    )


def _state_status_antigravity(checkpoint_path: Path) -> str:
    from .artifacts_antigravity import parse_canonical_json_bytes_antigravity

    state_path = checkpoint_path / "state_antigravity.json"
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError("Exp444 checkpoint state is absent")
    payload = parse_canonical_json_bytes_antigravity(state_path.read_bytes())
    if not isinstance(payload, Mapping) or type(payload.get("status")) is not str:
        raise ValueError("Exp444 checkpoint state is invalid")
    return payload["status"]


def _validate_checkpoint_context_antigravity(
    checkpoint_path: Path,
    context: CheckpointContextAntigravity,
) -> None:
    """Reject a different scientific/runtime context before any resume write."""
    path = checkpoint_path / "context_antigravity.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Exp444 checkpoint context is absent")
    encoded = path.read_bytes()
    payload = parse_canonical_json_bytes_antigravity(encoded)
    expected = context.to_payload()
    if not isinstance(payload, Mapping) or dict(payload) != expected:
        raise ValueError("Exp444 checkpoint context changed")
    if canonical_json_bytes_antigravity(payload) != encoded:
        raise ValueError("Exp444 checkpoint context is not canonical")


def _outcome_antigravity(
    attempt: Path,
    store: Exp444CheckpointStoreAntigravity,
    writer: ProgressWriterAntigravity | None,
    *,
    status: str,
    label_access_count: int,
    resumed: bool,
    finalization_only: bool,
) -> RuntimeOutcomeAntigravity:
    results = tuple(store.results_antigravity().get(index) for index in range(store.completed_count_antigravity))
    if any(item is None for item in results):
        raise ValueError("Exp444 checkpoint contains an error record")
    return RuntimeOutcomeAntigravity(
        attempt,
        attempt / "checkpoint_antigravity",
        None if writer is None else attempt / "progress_antigravity.json",
        tuple(item for item in results if item is not None),
        status,
        store.completed_count_antigravity,
        production_worker_count_antigravity(),
        "cpu",
        label_access_count,
        resumed,
        finalization_only,
    )


def _execute_runtime_antigravity(
    *,
    reviewed_commit: str,
    mode: str,
    attempt_name: str | None,
    run_root: Path | str | None,
    executor_factory: Callable[..., object] | None,
) -> RuntimeOutcomeAntigravity:
    if type(reviewed_commit) is not str or not reviewed_commit:
        raise ValueError("Exp444 reviewed commit is invalid")
    FROZEN_CONFIG.validate()
    root = Path(FROZEN_CONFIG.run_root if run_root is None else run_root)
    context = _context_antigravity()
    process = current_process_identity_antigravity()
    if mode == "new":
        if attempt_name is not None:
            raise ValueError("Exp444 new run cannot select an attempt")
        attempt = _attempt_root_antigravity(root)
        checkpoint_root = attempt / "checkpoint_antigravity"
        store = Exp444CheckpointStoreAntigravity.create(checkpoint_root, context, process)
        writer = ProgressWriterAntigravity.create(
            attempt,
            total=FROZEN_CONFIG.source_candidate_count,
            process=process,
            checkpoint_path=checkpoint_root / "state_antigravity.json",
            backend="cpu",
        )
        resumed = False
        finalization_only = False
        label_access_count = 0
    elif mode == "resume":
        if attempt_name is None:
            raise ValueError("Exp444 resume requires an attempt")
        attempt = _resume_attempt_antigravity(root, attempt_name)
        checkpoint_root = attempt / "checkpoint_antigravity"
        _validate_checkpoint_context_antigravity(checkpoint_root, context)
        status = _state_status_antigravity(checkpoint_root)
        if status == "complete":
            store = Exp444CheckpointStoreAntigravity.open_interrupted(
                checkpoint_root,
                context,
                process,
                allow_complete_finalization=True,
            )
            return _outcome_antigravity(
                attempt, store, None, status="complete", label_access_count=0,
                resumed=True, finalization_only=True,
            )
        # Progress validation must see the inactive interrupted checkpoint;
        # checkpoint opening then binds the new owner and switches to running.
        writer = ProgressWriterAntigravity.open_interrupted(
            attempt,
            total=FROZEN_CONFIG.source_candidate_count,
            process=process,
            checkpoint_path=checkpoint_root / "state_antigravity.json",
            pid_probe=lambda _stored: False,
        )
        store = Exp444CheckpointStoreAntigravity.open_interrupted(
            checkpoint_root,
            context,
            process,
            pid_probe=lambda _stored: False,
        )
        resumed = True
        finalization_only = False
        label_access_count = 0
    else:
        raise ValueError("Exp444 runtime mode is invalid")

    try:
        inputs, label_access_count = _load_source_inputs_antigravity()
        completed = store.completed_count_antigravity
        if completed > len(inputs):
            raise ValueError("Exp444 checkpoint prefix exceeds source inputs")
        active = inputs[completed:]

        def commit(sequence: int, candidate: CandidateInputAntigravity, result: CandidateStabilityResultAntigravity) -> None:
            if writer is None:
                raise ValueError("Exp444 active progress writer is unavailable")
            _persist_commit_antigravity(store, writer, sequence, candidate, result)

        ordered_parallel_map_antigravity(
            active,
            workers=production_worker_count_antigravity(),
            distance_backend="cpu",
            executor_factory=executor_factory,
            commit=lambda local_sequence, candidate, result: commit(completed + local_sequence, candidate, result),
        )
        if writer is None:
            raise ValueError("Exp444 active progress writer is unavailable")
        writer.close_antigravity("complete")
        return _outcome_antigravity(
            attempt, store, writer, status="complete", label_access_count=label_access_count,
            resumed=resumed, finalization_only=finalization_only,
        )
    except KeyboardInterrupt as interruption:
        if writer is not None and writer.stage_antigravity not in {"complete", "error", "failed", "interrupted", "interrupted_resumable"}:
            if store.status_antigravity not in {"complete", "error"}:
                store.mark_interrupted_antigravity()
            writer.close_antigravity("interrupted", str(interruption) or "KeyboardInterrupt")
        raise
    except BaseException as error:
        if writer is not None and writer.stage_antigravity not in {"complete", "error", "failed", "interrupted", "interrupted_resumable"}:
            if store.status_antigravity not in {"complete", "error"} and store.completed_count_antigravity < FROZEN_CONFIG.source_candidate_count:
                store.commit_error_antigravity(store.completed_count_antigravity, error)
            writer.close_antigravity("error", str(error) or type(error).__name__)
        raise


def run_new_antigravity(
    reviewed_commit: str,
    *,
    run_root: Path | str | None = None,
    executor_factory: Callable[..., object] | None = None,
) -> RuntimeOutcomeAntigravity:
    return _execute_runtime_antigravity(
        reviewed_commit=reviewed_commit,
        mode="new",
        attempt_name=None,
        run_root=run_root,
        executor_factory=executor_factory,
    )


def run_resume_antigravity(
    reviewed_commit: str,
    attempt_name: str,
    *,
    run_root: Path | str | None = None,
    executor_factory: Callable[..., object] | None = None,
) -> RuntimeOutcomeAntigravity:
    return _execute_runtime_antigravity(
        reviewed_commit=reviewed_commit,
        mode="resume",
        attempt_name=attempt_name,
        run_root=run_root,
        executor_factory=executor_factory,
    )


__all__ = [
    "RuntimeOutcomeAntigravity", "WorkerFailureAntigravity",
    "ordered_parallel_map_antigravity", "sequential_reference_antigravity",
    "canonical_results_bytes_antigravity", "production_worker_count_antigravity",
    "run_new_antigravity", "run_resume_antigravity",
    "_score_candidate_worker_antigravity", "_worker_initializer_antigravity",
    "_THREAD_VARIABLES_ANTIGRAVITY",
]
