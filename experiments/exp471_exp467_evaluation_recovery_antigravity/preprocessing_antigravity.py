"""Result-preserving Exp471 preprocessing parallelism and live ETA evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import stat
from time import monotonic

from .contracts_antigravity import canonical_json_bytes_exp471_antigravity


_PROGRESS_NAME_ANTIGRAVITY = "preprocessing_progress_antigravity.jsonl"
_EVENT_KEYS_V1_ANTIGRAVITY = {
    "schema",
    "stage",
    "phase",
    "condition",
    "dataset",
    "completed",
    "total",
    "remaining",
    "percent",
    "elapsed_seconds",
    "items_per_second",
    "eta_seconds",
    "logical_cpu_count",
    "requested_workers",
    "workers",
    "worker_speedup",
    "byte_parity",
    "parallel_accepted",
    "status",
}
_EVENT_KEYS_V2_ANTIGRAVITY = _EVENT_KEYS_V1_ANTIGRAVITY | {
    "acceleration_backend"
}


def _finite_number_antigravity(value: object, *, nullable: bool = False) -> bool:
    if nullable and value is None:
        return True
    return (
        type(value) in {int, float}
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _validated_event_antigravity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Exp471 preprocessing progress schema differs")
    event = dict(value)
    schema = event.get("schema")
    expected_keys = (
        _EVENT_KEYS_V1_ANTIGRAVITY
        if schema == "exp471-preprocessing-progress-antigravity.v1"
        else _EVENT_KEYS_V2_ANTIGRAVITY
        if schema == "exp471-preprocessing-progress-antigravity.v2"
        else None
    )
    if expected_keys is None or set(event) != expected_keys:
        raise ValueError("Exp471 preprocessing progress schema differs")
    completed = event["completed"]
    total = event["total"]
    remaining = event["remaining"]
    if (
        event["schema"]
        not in {
            "exp471-preprocessing-progress-antigravity.v1",
            "exp471-preprocessing-progress-antigravity.v2",
        }
        or event["stage"] != "score-input"
        or event["phase"] != "low-frequency"
        or type(event["condition"]) is not str
        or not event["condition"]
        or (
            event["dataset"] is not None
            and (type(event["dataset"]) is not str or not event["dataset"])
        )
        or type(completed) is not int
        or type(total) is not int
        or type(remaining) is not int
        or completed < 0
        or total < 1
        or completed > total
        or remaining != total - completed
        or type(event["workers"]) is not int
        or event["workers"] < 1
        or type(event["requested_workers"]) is not int
        or event["requested_workers"] < event["workers"]
        or type(event["logical_cpu_count"]) is not int
        or event["logical_cpu_count"] < event["requested_workers"]
        or not _finite_number_antigravity(
            event["worker_speedup"], nullable=True
        )
        or (
            event["byte_parity"] is not None
            and type(event["byte_parity"]) is not bool
        )
        or (
            event["parallel_accepted"] is not None
            and type(event["parallel_accepted"]) is not bool
        )
        or (
            event["parallel_accepted"] is True
            and (
                event["workers"] != event["requested_workers"]
                or event["workers"] <= 1
                or event["byte_parity"] is not True
                or event["worker_speedup"] is None
                or float(event["worker_speedup"]) < 1.1
            )
        )
        or event["status"]
        not in {
            "start",
            "qualified",
            "running",
            "condition_complete",
            "complete",
            "failed",
        }
        or not _finite_number_antigravity(event["percent"])
        or not _finite_number_antigravity(event["elapsed_seconds"])
        or not _finite_number_antigravity(event["items_per_second"], nullable=True)
        or not _finite_number_antigravity(event["eta_seconds"], nullable=True)
        or not math.isclose(
            float(event["percent"]), completed * 100.0 / total, abs_tol=1e-9
        )
        or (completed == 0 and event["eta_seconds"] is not None)
        or (remaining == 0 and float(event["eta_seconds"] or 0.0) != 0.0)
    ):
        raise ValueError("Exp471 preprocessing progress relation differs")
    if event["schema"].endswith(".v2") and event["acceleration_backend"] not in {
        "cpu",
        "numba_serial_safe",
    }:
        raise ValueError("Exp471 preprocessing acceleration backend differs")
    return event


def _validated_record_antigravity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or "timestamp_utc" not in value:
        raise ValueError("Exp471 preprocessing progress record differs")
    event = _validated_event_antigravity(
        {key: item for key, item in value.items() if key != "timestamp_utc"}
    )
    if set(value) != set(event) | {"timestamp_utc"}:
        raise ValueError("Exp471 preprocessing progress record differs")
    timestamp = value["timestamp_utc"]
    if type(timestamp) is not str:
        raise ValueError("Exp471 preprocessing progress timestamp differs")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("Exp471 preprocessing progress timestamp differs") from error
    if parsed.tzinfo is None:
        raise ValueError("Exp471 preprocessing progress timestamp differs")
    return {**event, "timestamp_utc": timestamp}


def append_preprocessing_progress_antigravity(
    root: Path, event: Mapping[str, object]
) -> None:
    """Append one canonical preprocessing event through a locked regular file."""

    root = Path(root)
    checked = _validated_event_antigravity(event)
    record = {
        **checked,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }
    payload = canonical_json_bytes_exp471_antigravity(record)
    root.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    root_descriptor = os.open(root, directory_flags)
    try:
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(
            _PROGRESS_NAME_ANTIGRAVITY,
            flags,
            0o600,
            dir_fd=root_descriptor,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise ValueError("Exp471 preprocessing progress log is unsafe")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError(
                        "Exp471 preprocessing progress append made no progress"
                    )
                view = view[written:]
            os.fsync(descriptor)
            os.fsync(root_descriptor)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
    finally:
        os.close(root_descriptor)


def latest_preprocessing_progress_antigravity(
    root: Path,
) -> Mapping[str, object] | None:
    """Return the latest complete canonical preprocessing log record."""

    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        root_descriptor = os.open(Path(root), directory_flags)
    except FileNotFoundError:
        return None
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(
                _PROGRESS_NAME_ANTIGRAVITY, flags, dir_fd=root_descriptor
            )
        except FileNotFoundError:
            return None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("Exp471 preprocessing progress log is unsafe")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                (before.st_dev, before.st_ino, before.st_size)
                != (after.st_dev, after.st_ino, after.st_size)
            ):
                raise ValueError("Exp471 preprocessing progress log changed")
            raw = b"".join(chunks)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
    finally:
        os.close(root_descriptor)
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("Exp471 preprocessing progress log is truncated")
    latest: dict[str, object] | None = None
    for line in raw.splitlines(keepends=True):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Exp471 preprocessing progress log is malformed") from error
        record = _validated_record_antigravity(value)
        if canonical_json_bytes_exp471_antigravity(record) != line:
            raise ValueError("Exp471 preprocessing progress log is noncanonical")
        latest = record
    return latest


def select_preprocessing_workers_antigravity(
    single_result: object,
    parallel_result: object,
    *,
    single_seconds: float,
    parallel_seconds: float,
    requested_workers: int,
) -> Mapping[str, object]:
    """Accept parallel preprocessing only after equality and a 1.1x speed gate."""

    if (
        type(requested_workers) is not int
        or requested_workers < 1
        or not _finite_number_antigravity(single_seconds)
        or not _finite_number_antigravity(parallel_seconds)
    ):
        raise ValueError("Exp471 preprocessing benchmark input differs")
    byte_parity = bool(single_result == parallel_result)
    speedup = (
        1.0
        if float(parallel_seconds) <= 0.0
        else float(single_seconds) / float(parallel_seconds)
    )
    accepted = bool(
        requested_workers > 1 and byte_parity and speedup >= 1.1
    )
    return {
        "requested_workers": requested_workers,
        "selected_workers": requested_workers if accepted else 1,
        "single_seconds": float(single_seconds),
        "parallel_seconds": float(parallel_seconds),
        "speedup": speedup,
        "byte_parity": byte_parity,
        "parallel_accepted": accepted,
    }


def _parallel_mapping_score_antigravity(
    score: Callable[..., object],
    values: Mapping[str, object],
    hard: Mapping[str, frozenset[int]],
    *,
    workers: int,
    completed: Callable[[str], None] | None = None,
) -> object:
    datasets = tuple(sorted(values))
    outcomes: dict[str, object] = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="exp471_preprocessing_antigravity",
    ) as executor:
        futures = {
            executor.submit(score, values[dataset], hard): dataset
            for dataset in datasets
        }
        for future in as_completed(futures):
            dataset = futures[future]
            outcomes[dataset] = future.result()
            if completed is not None:
                completed(dataset)
    ordered = tuple(outcomes[dataset] for dataset in datasets)
    first = ordered[0]
    candidates = tuple(
        candidate
        for result in ordered
        for candidate in result.provisional_candidates
    )
    selector = getattr(score, "__globals__", {}).get("select_source_candidates")
    if not callable(selector):
        raise ValueError("Exp471 frozen low-frequency selector is unavailable")
    return type(first)(
        datasets,
        tuple(row for result in ordered for row in result.rows),
        candidates,
        selector(candidates, {dataset: hard[dataset] for dataset in datasets}),
        sum(result.excluded_train_rows for result in ordered),
        {dataset: hard[dataset] for dataset in datasets},
    )


class Exp471PreprocessingObserverAntigravity:
    """Run the frozen per-dataset scorer concurrently and merge exact order."""

    def __init__(
        self,
        score: Callable[..., object],
        *,
        condition_order: Sequence[str],
        workers: int,
        progress: Callable[[dict[str, object]], None],
        qualify_parallel: bool = True,
        logical_cpu_count: int | None = None,
        acceleration_backend: str = "cpu",
    ) -> None:
        conditions = tuple(condition_order)
        if (
            not callable(score)
            or not conditions
            or any(type(condition) is not str or not condition for condition in conditions)
            or len(set(conditions)) != len(conditions)
            or type(workers) is not int
            or workers < 1
            or not callable(progress)
            or type(qualify_parallel) is not bool
            or acceleration_backend not in {"cpu", "numba_serial_safe"}
            or (
                logical_cpu_count is not None
                and (
                    type(logical_cpu_count) is not int
                    or logical_cpu_count < workers
                )
            )
        ):
            raise ValueError("Exp471 preprocessing observer configuration differs")
        self._score_antigravity = score
        self._conditions_antigravity = conditions
        self._requested_workers_antigravity = workers
        self._logical_cpu_count_antigravity = (
            workers + 1 if logical_cpu_count is None else logical_cpu_count
        )
        self._workers_antigravity: int | None = (
            None if qualify_parallel and workers > 1 else workers
        )
        self._worker_decision_antigravity: Mapping[str, object] | None = (
            None
            if self._workers_antigravity is None
            else {
                "speedup": 1.1 if workers > 1 else 1.0,
                "byte_parity": True,
                "parallel_accepted": workers > 1,
            }
        )
        self._progress_antigravity = progress
        self._acceleration_backend_antigravity = acceleration_backend
        self._condition_index_antigravity = 0
        self._completed_antigravity = 0
        self._total_antigravity: int | None = None
        self._started_antigravity = monotonic()

    def _event_antigravity(
        self,
        *,
        condition: str,
        dataset: str | None,
        status: str,
    ) -> dict[str, object]:
        total = self._total_antigravity
        if total is None:
            raise ValueError("Exp471 preprocessing total is unavailable")
        completed = self._completed_antigravity
        remaining = total - completed
        elapsed = max(0.0, monotonic() - self._started_antigravity)
        rate = None if completed == 0 or elapsed <= 0.0 else completed / elapsed
        eta = (
            None
            if rate is None
            else (0.0 if remaining == 0 else remaining / rate)
        )
        decision = self._worker_decision_antigravity
        return {
            "schema": "exp471-preprocessing-progress-antigravity.v2",
            "stage": "score-input",
            "phase": "low-frequency",
            "condition": condition,
            "dataset": dataset,
            "completed": completed,
            "total": total,
            "remaining": remaining,
            "percent": completed * 100.0 / total,
            "elapsed_seconds": elapsed,
            "items_per_second": rate,
            "eta_seconds": eta,
            "logical_cpu_count": self._logical_cpu_count_antigravity,
            "requested_workers": self._requested_workers_antigravity,
            "workers": (
                self._requested_workers_antigravity
                if self._workers_antigravity is None
                else self._workers_antigravity
            ),
            "worker_speedup": (
                None if decision is None else float(decision["speedup"])
            ),
            "byte_parity": (
                None if decision is None else bool(decision["byte_parity"])
            ),
            "parallel_accepted": (
                None
                if decision is None
                else bool(decision["parallel_accepted"])
            ),
            "acceleration_backend": self._acceleration_backend_antigravity,
            "status": status,
        }

    def score_antigravity(
        self, values: Mapping[str, object], hard: Mapping[str, frozenset[int]]
    ) -> object:
        datasets = tuple(sorted(values))
        if (
            not datasets
            or set(datasets) != set(hard)
            or self._condition_index_antigravity >= len(self._conditions_antigravity)
        ):
            raise ValueError("Exp471 preprocessing input differs")
        expected_total = len(datasets) * len(self._conditions_antigravity)
        if self._total_antigravity is None:
            self._total_antigravity = expected_total
        elif self._total_antigravity != expected_total:
            raise ValueError("Exp471 preprocessing dataset count changed")
        condition = self._conditions_antigravity[
            self._condition_index_antigravity
        ]
        if self._condition_index_antigravity == 0:
            self._progress_antigravity(
                self._event_antigravity(
                    condition=condition, dataset=None, status="start"
                )
            )
        if self._workers_antigravity is None:
            sample_datasets = datasets[
                : min(len(datasets), max(2, self._requested_workers_antigravity))
            ]
            sample = {dataset: values[dataset] for dataset in sample_datasets}
            started = monotonic()
            single = self._score_antigravity(sample, hard)
            single_seconds = monotonic() - started
            started = monotonic()
            parallel = _parallel_mapping_score_antigravity(
                self._score_antigravity,
                sample,
                hard,
                workers=self._requested_workers_antigravity,
            )
            parallel_seconds = monotonic() - started
            decision = select_preprocessing_workers_antigravity(
                single,
                parallel,
                single_seconds=single_seconds,
                parallel_seconds=parallel_seconds,
                requested_workers=self._requested_workers_antigravity,
            )
            self._workers_antigravity = int(decision["selected_workers"])
            self._worker_decision_antigravity = decision
            self._progress_antigravity(
                self._event_antigravity(
                    condition=condition, dataset=None, status="qualified"
                )
            )

        def record_completion_antigravity(dataset: str) -> None:
            self._completed_antigravity += 1
            self._progress_antigravity(
                self._event_antigravity(
                    condition=condition,
                    dataset=dataset,
                    status="running",
                )
            )

        try:
            merged = _parallel_mapping_score_antigravity(
                self._score_antigravity,
                values,
                hard,
                workers=self._workers_antigravity,
                completed=record_completion_antigravity,
            )
        except Exception:
            self._progress_antigravity(
                self._event_antigravity(
                    condition=condition, dataset=None, status="failed"
                )
            )
            raise
        self._condition_index_antigravity += 1
        terminal = (
            "complete"
            if self._condition_index_antigravity == len(self._conditions_antigravity)
            else "condition_complete"
        )
        self._progress_antigravity(
            self._event_antigravity(
                condition=condition, dataset=None, status=terminal
            )
        )
        return merged


@contextmanager
def instrument_captured_preprocessing_antigravity(
    session: object, observer: Exp471PreprocessingObserverAntigravity
):
    """Temporarily inject observation around the fixed scorer, then restore it."""

    verifier = getattr(session, "verifier", None)
    compose = getattr(verifier, "compose_all_conditions_antigravity", None)
    compose_globals = getattr(compose, "__globals__", None)
    compose_condition = (
        compose_globals.get("compose_condition_score_antigravity")
        if isinstance(compose_globals, dict)
        else None
    )
    scoring_globals = getattr(compose_condition, "__globals__", None)
    original = (
        scoring_globals.get("score_low_frequency")
        if isinstance(scoring_globals, dict)
        else None
    )
    if (
        not isinstance(observer, Exp471PreprocessingObserverAntigravity)
        or not callable(compose)
        or not callable(compose_condition)
        or not callable(original)
        or original is not observer._score_antigravity
    ):
        raise ValueError("Exp471 captured preprocessing boundary differs")
    replacement = observer.score_antigravity
    scoring_globals["score_low_frequency"] = replacement
    try:
        yield
    finally:
        observed = scoring_globals.get("score_low_frequency")
        scoring_globals["score_low_frequency"] = original
        if observed is not replacement:
            raise ValueError("Exp471 captured preprocessing scorer changed")


__all__ = [
    "Exp471PreprocessingObserverAntigravity",
    "append_preprocessing_progress_antigravity",
    "instrument_captured_preprocessing_antigravity",
    "latest_preprocessing_progress_antigravity",
    "select_preprocessing_workers_antigravity",
]
