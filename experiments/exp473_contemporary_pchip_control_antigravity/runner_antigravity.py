"""Bounded two-wave execution for label-free Exp473 D-prime scores."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import os
from pathlib import Path
import pickle
from time import monotonic

from .artifacts_antigravity import (
    publish_manifest_exp473_antigravity,
    publish_score_shard_exp473_antigravity,
    verify_manifest_exp473_antigravity,
    verify_score_shard_exp473_antigravity,
)
from .contracts_antigravity import (
    Exp473ScoreShardAntigravity,
    canonical_json_bytes_exp473_antigravity,
)


_SCORE_ONE_CPU_EXP473_ANTIGRAVITY: Callable[[str], Exp473ScoreShardAntigravity] | None = None


def _one_inner_thread_exp473_antigravity() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _init_cpu_exp473_antigravity(
    score_one: Callable[[str], Exp473ScoreShardAntigravity],
) -> None:
    global _SCORE_ONE_CPU_EXP473_ANTIGRAVITY
    _SCORE_ONE_CPU_EXP473_ANTIGRAVITY = score_one
    _one_inner_thread_exp473_antigravity()


def _invoke_cpu_exp473_antigravity(dataset: str) -> dict[str, object]:
    score_one = _SCORE_ONE_CPU_EXP473_ANTIGRAVITY
    if not callable(score_one):
        raise ValueError("Exp473 CPU scorer is unbound")
    return _invoke_exp473_antigravity(score_one, dataset)


def _invoke_exp473_antigravity(
    score_one: Callable[[str], Exp473ScoreShardAntigravity], dataset: str
) -> dict[str, object]:
    shard = score_one(dataset)
    if not isinstance(shard, Exp473ScoreShardAntigravity) or shard.dataset != dataset:
        raise ValueError("Exp473 scorer result identity differs")
    return {
        "dataset": shard.dataset,
        "test_identity": dict(shard.test_identity),
        "scores": shard.scores,
        "alerts": tuple(sorted(shard.alerts)),
    }


def _packet_to_shard_exp473_antigravity(
    packet: object,
) -> Exp473ScoreShardAntigravity:
    if not isinstance(packet, dict):
        raise ValueError("Exp473 worker packet differs")
    try:
        return Exp473ScoreShardAntigravity(
            dataset=packet["dataset"],
            test_identity=packet["test_identity"],
            scores=packet["scores"],
            alerts=frozenset(packet["alerts"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Exp473 worker packet differs") from error


def _append_event_exp473_antigravity(
    root: Path, name: str, record: dict[str, object]
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    with path.open("ab") as stream:
        stream.write(canonical_json_bytes_exp473_antigravity(record))
        stream.flush()


def _run_cpu_exp473_antigravity(
    pending: tuple[str, ...],
    score_one: Callable[[str], Exp473ScoreShardAntigravity],
    workers: int,
) -> Iterator[tuple[str, Exp473ScoreShardAntigravity | BaseException]]:
    try:
        pickle.dumps(score_one)
    except BaseException as error:
        failure = ValueError(
            "Exp473 CPU score callback must be pickle-safe: "
            f"{type(error).__name__}: {error}"
        )
        yield from ((dataset, failure) for dataset in pending)
        return
    queue = iter(pending)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_cpu_exp473_antigravity,
        initargs=(score_one,),
    ) as pool:
        active: dict[object, str] = {}
        exhausted = False

        def fill() -> None:
            nonlocal exhausted
            while not exhausted and len(active) < workers:
                try:
                    dataset = next(queue)
                except StopIteration:
                    exhausted = True
                    return
                active[pool.submit(_invoke_cpu_exp473_antigravity, dataset)] = dataset

        fill()
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                dataset = active.pop(future)
                try:
                    outcome = _packet_to_shard_exp473_antigravity(future.result())
                except BaseException as error:
                    outcome = error
                yield dataset, outcome
            fill()


def _run_wave_exp473_antigravity(
    pending: tuple[str, ...],
    *,
    score_one: Callable[[str], Exp473ScoreShardAntigravity],
    cpu_workers: int,
    tensor_backend: str,
) -> Iterator[tuple[str, Exp473ScoreShardAntigravity | BaseException]]:
    if tensor_backend == "mps":
        for dataset in pending:
            try:
                outcome = _packet_to_shard_exp473_antigravity(
                    _invoke_exp473_antigravity(score_one, dataset)
                )
            except BaseException as error:
                outcome = error
            yield dataset, outcome
        return
    yield from _run_cpu_exp473_antigravity(pending, score_one, cpu_workers)


def run_score_wave_exp473_antigravity(
    root: Path,
    context_sha256: str,
    catalog: tuple[str, ...],
    *,
    score_one: Callable[[str], Exp473ScoreShardAntigravity],
    cpu_workers: int,
    tensor_backend: str,
    expected_identities: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Resume verified shards, retry failures once, and publish only terminal authority."""
    names = tuple(catalog)
    if (
        not names
        or len(names) != len(set(names))
        or not callable(score_one)
        or type(cpu_workers) is not int
        or cpu_workers < 1
        or tensor_backend not in {"cpu", "mps"}
        or (
            expected_identities is not None
            and not isinstance(expected_identities, Mapping)
        )
    ):
        raise ValueError("Exp473 score runner inputs differ")
    if tensor_backend == "mps" and cpu_workers < 1:
        raise ValueError("Exp473 MPS runner policy differs")
    root = Path(root)
    completed: set[str] = set()
    for dataset in names:
        try:
            verify_score_shard_exp473_antigravity(root, context_sha256, dataset)
        except (FileNotFoundError, ValueError, OSError):
            continue
        completed.add(dataset)
    failures: set[str] = set()
    started = monotonic()
    for wave in (1, 2):
        pending = tuple(dataset for dataset in names if dataset not in completed)
        if not pending:
            break
        for dataset, outcome in _run_wave_exp473_antigravity(
            pending,
            score_one=score_one,
            cpu_workers=cpu_workers,
            tensor_backend=tensor_backend,
        ):
            if isinstance(outcome, BaseException):
                failures.add(dataset)
                _append_event_exp473_antigravity(
                    root,
                    "failures_antigravity.jsonl",
                    {
                        "dataset": dataset,
                        "wave": wave,
                        "error_type": type(outcome).__name__,
                        "error": str(outcome),
                    },
                )
                status = "failed"
            else:
                try:
                    if expected_identities is not None:
                        try:
                            expected = expected_identities[dataset]
                        except KeyError as error:
                            raise ValueError(
                                f"Exp473 loaded TEST identity differs for {dataset}"
                            ) from error
                        if dict(outcome.test_identity) != dict(expected):
                            raise ValueError(
                                f"Exp473 loaded TEST identity differs for {dataset}"
                            )
                    publish_score_shard_exp473_antigravity(
                        root, context_sha256, outcome
                    )
                except BaseException as error:
                    failures.add(dataset)
                    _append_event_exp473_antigravity(
                        root,
                        "failures_antigravity.jsonl",
                        {
                            "dataset": dataset,
                            "wave": wave,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    )
                    status = "failed"
                else:
                    completed.add(dataset)
                    failures.discard(dataset)
                    status = "completed"
            _append_event_exp473_antigravity(
                root,
                "progress_antigravity.jsonl",
                {
                    "dataset": dataset,
                    "wave": wave,
                    "status": status,
                    "completed": len(completed),
                    "failed": len(failures),
                    "remaining": len(names) - len(completed),
                    "elapsed_seconds": monotonic() - started,
                },
            )
    manifest_published = False
    if len(completed) == len(names) and not failures:
        try:
            publish_manifest_exp473_antigravity(root, context_sha256, names)
        except FileExistsError:
            verify_manifest_exp473_antigravity(root, context_sha256, names)
        manifest_published = True
    return {
        "waves": 2,
        "workers": 1 if tensor_backend == "mps" else cpu_workers,
        "completed": len(completed),
        "failed": len(failures),
        "remaining": len(names) - len(completed),
        "manifest_published": manifest_published,
    }


__all__ = ["run_score_wave_exp473_antigravity"]
