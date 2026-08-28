"""Bounded two-wave scheduling for label-free Exp472 condition-E scores."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import os
from pathlib import Path
from time import monotonic

from .artifacts_antigravity import (
    append_failure_exp472_antigravity,
    append_progress_exp472_antigravity,
    publish_manifest_exp472_antigravity,
    publish_score_shard_exp472_antigravity,
    verify_score_shard_exp472_antigravity,
)
from .contracts_antigravity import Exp472RunContextAntigravity, Exp472ScoreShardAntigravity


@dataclass(frozen=True, slots=True)
class _OutcomeAntigravity:
    dataset: str
    status: str


def _dataset_antigravity(entry: object) -> str:
    value = entry.get("dataset") if isinstance(entry, Mapping) else getattr(entry, "dataset", None)
    if type(value) is not str or not value:
        raise ValueError("Exp472 catalog entry dataset is invalid")
    return value


def _catalog_antigravity(catalog: Sequence[object]) -> tuple[object, ...]:
    entries = tuple(catalog)
    names = tuple(_dataset_antigravity(entry) for entry in entries)
    if not entries or len(names) != len(set(names)):
        raise ValueError("Exp472 catalog coverage is invalid")
    return entries


def pending_datasets_exp472_antigravity(
    catalog: Sequence[object], *, completed: set[str] | frozenset[str], failures: set[str] | frozenset[str]
) -> tuple[str, ...]:
    """Return frozen catalog order for only missing or currently failed datasets."""
    entries = _catalog_antigravity(catalog)
    return tuple(
        dataset for dataset in (_dataset_antigravity(entry) for entry in entries)
        if dataset not in completed or dataset in failures
    )


def _one_inner_numerical_thread_antigravity() -> None:
    """Keep BLAS/Numba worker internals serial inside a CPU process slot."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = "1"


def _run_cpu_antigravity(
    pending: tuple[str, ...], score_one: Callable[[str], Exp472ScoreShardAntigravity], workers: int
) -> list[tuple[str, Exp472ScoreShardAntigravity | BaseException]]:
    outcomes: list[tuple[str, Exp472ScoreShardAntigravity | BaseException]] = []
    queue = iter(pending)
    with ProcessPoolExecutor(max_workers=workers, initializer=_one_inner_numerical_thread_antigravity) as pool:
        active: dict[object, str] = {}
        for _ in range(workers):
            try:
                dataset = next(queue)
            except StopIteration:
                break
            active[pool.submit(score_one, dataset)] = dataset
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                dataset = active.pop(future)
                try:
                    outcomes.append((dataset, future.result()))
                except BaseException as error:
                    outcomes.append((dataset, error))
                try:
                    next_dataset = next(queue)
                except StopIteration:
                    continue
                active[pool.submit(score_one, next_dataset)] = next_dataset
    return outcomes


def _execute_datasets_exp472_antigravity(
    pending: tuple[str, ...], *, context: Exp472RunContextAntigravity, score_one: Callable[[str], Exp472ScoreShardAntigravity]
) -> list[tuple[str, Exp472ScoreShardAntigravity | BaseException]]:
    if context.backend.inner_numerical_threads != 1:
        raise ValueError("Exp472 inner numerical thread policy differs")
    if context.backend.tensor_backend == "mps":
        # MPS is deliberately one job at a time: no outer parallelism reaches it.
        outcomes: list[tuple[str, Exp472ScoreShardAntigravity | BaseException]] = []
        for dataset in pending:
            try:
                outcomes.append((dataset, score_one(dataset)))
            except BaseException as error:
                outcomes.append((dataset, error))
        return outcomes
    return _run_cpu_antigravity(pending, score_one, context.backend.cpu_workers)


def _verified_completed_antigravity(root: Path, context: Exp472RunContextAntigravity, entries: tuple[object, ...]) -> set[str]:
    completed: set[str] = set()
    for entry in entries:
        dataset = _dataset_antigravity(entry)
        try:
            verify_score_shard_exp472_antigravity(root, context, entry)
        except ValueError:
            continue
        completed.add(dataset)
    return completed


def run_score_wave_exp472_antigravity(
    root: Path,
    context: Exp472RunContextAntigravity,
    catalog: Sequence[object],
    *,
    score_one: Callable[[str], Exp472ScoreShardAntigravity],
) -> dict[str, object]:
    """Run precisely two fixed-parameter waves and publish only terminal authority."""
    if not isinstance(context, Exp472RunContextAntigravity) or not callable(score_one):
        raise ValueError("Exp472 score runner inputs are invalid")
    entries = _catalog_antigravity(catalog)
    by_dataset = {_dataset_antigravity(entry): entry for entry in entries}
    total = len(entries)
    started = monotonic()
    completed = _verified_completed_antigravity(Path(root), context, entries)
    failures: set[str] = set()
    for wave in (1, 2):
        # Wave one resumes every missing coordinate; wave two is the single bounded recovery.
        pending = pending_datasets_exp472_antigravity(entries, completed=completed, failures=failures if wave == 2 else frozenset())
        if wave == 2:
            pending = tuple(dataset for dataset in pending if dataset not in completed)
        for dataset, result in _execute_datasets_exp472_antigravity(pending, context=context, score_one=score_one):
            if isinstance(result, BaseException):
                append_failure_exp472_antigravity(Path(root), context, dataset=dataset, wave=wave, error=result)
                failures.add(dataset)
                status = "failed"
            else:
                try:
                    if not isinstance(result, Exp472ScoreShardAntigravity) or result.dataset != dataset:
                        raise ValueError("Exp472 scorer result identity differs")
                    publish_score_shard_exp472_antigravity(Path(root), context, by_dataset[dataset], result)
                except BaseException as error:
                    append_failure_exp472_antigravity(Path(root), context, dataset=dataset, wave=wave, error=error)
                    failures.add(dataset)
                    status = "failed"
                else:
                    completed.add(dataset)
                    failures.discard(dataset)
                    status = "completed"
            append_progress_exp472_antigravity(
                Path(root), completed=len(completed), failed=len(failures), remaining=total - len(completed),
                elapsed_seconds=monotonic() - started, dataset=dataset, wave=wave, status=status,
            )
    # Small test catalogs intentionally cannot become terminal authorities. Production
    # smoke/full catalogs are independently checked by the manifest contract.
    manifest_published = False
    if not failures and len(completed) == total and total in {23, 1117}:
        publish_manifest_exp472_antigravity(Path(root), context, entries, failures=failures)
        manifest_published = True
    return {"waves": 2, "completed": tuple(_dataset_antigravity(entry) for entry in entries if _dataset_antigravity(entry) in completed), "failures": tuple(_dataset_antigravity(entry) for entry in entries if _dataset_antigravity(entry) in failures), "manifest_published": manifest_published}


__all__ = ["pending_datasets_exp472_antigravity", "run_score_wave_exp472_antigravity"]
