"""Narrow Task4--7 score composition for the frozen proposal population."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from time import monotonic
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from .artifacts_paper import Task4Task5SuccessAuthority, append_task4_task5_attempt_record, freeze_score_root, read_task4_task5_success_record, verify_task4_task5_attempt_namespace, verify_task4_task5_success_namespace, write_task4_task5_success_record, write_typed_proposal_shard
from .contracts_paper import FrozenTask8BackendDecision, HardRouteOracleBundle, HardRouteParityAuthority, LowFrequencyResult, RawDataset, Task8FrozenManifest
from .data_paper import PROGRESS_TICK_INTERVAL_S, ProgressCallback, emit_progress_paper, emit_strict_progress_paper
from .hard_route_paper import HardRouteResult, replay_hard_route
from .routing_paper import PolicyBundle, build_hard_evidence_surface, build_policies
from .sources_paper.canonical_paper import verify_pycatch22_environment_paper
from .sources_paper.hard_sources_paper import HardSourceBundle, score_hard_sources
from .sources_paper.low_frequency_paper import score_low_frequency
from .sources_paper.minirocket_paper import MiniRocketResult, score_minirocket_confirmations
from .sources_paper.multirocket_hydra_paper import verify_exp84_environment_paper
from .sources_paper.preprocessing_paper import DependencyUnavailablePaper
from .sources_paper.stability_paper import StabilityPartition, classify_stability


EXP137_EXACT_HEADER = (
    "experiment_id", "dataset_name", "family", "config_name", "selector_name",
    "selector_reason", "threshold_method", "score_family", "score_source_name",
    "hard_alert_indices", "standard_review_indices", "priority_review_indices",
    "hard_alert_count", "standard_review_count", "priority_review_count", "no_alert_count",
    "hard_tp", "hard_fp", "hard_fn", "hard_precision", "hard_recall", "hard_f1",
    "standard_review_tp", "standard_review_fp", "standard_review_fn", "standard_review_precision",
    "standard_review_recall", "standard_review_f1", "priority_review_tp", "priority_review_fp",
    "priority_review_fn", "priority_review_precision", "priority_review_recall", "priority_review_f1",
    "combined_tp", "combined_fp", "combined_fn", "combined_precision", "combined_recall", "combined_f1",
    "combined_metric_scope", "routing_uses_test_labels", "routing_uses_test_position",
    "routing_uses_family_performance", "priority_review_posthoc_research_rule",
    "prospective_priority_validation_required", "source_exp133_config", "source_exp135_config",
    "source_exp93_hard_count", "source_exp135_review_count", "train_normal_count",
    "sequence_length", "test_size", "anomaly_count",
)
EXP236_EXACT_ROW_KEYS = frozenset(
    {
        "contains_evaluation_metrics", "contains_test_labels", "dataset", "deployment_eligible",
        "experiment_id", "hard_indices", "hard_policy", "n_test", "new_hard_indices",
        "new_standard_review_indices", "priority_review_indices", "priority_review_policy",
        "retrospective_candidate_packaging", "standard_review_indices", "standard_review_policy",
    }
)
EXP291_EXACT_ROW_KEYS = frozenset(
    {
        "automatic_hard_indices", "automatic_normal_exact_indices", "automatic_normal_rule",
        "contains_evaluation_metrics", "contains_test_labels", "dataset", "default_review_indices",
        "experiment_id", "n_test", "priority_review_indices", "prospective_validated",
        "recovery_priority_scope", "recovery_shadow_review_indices", "research_only_recovery_priority_indices",
        "retrospective_integration", "shadow_review_indices", "source_baseline_hard_indices", "standard_review_indices",
    }
)
EXP404_EXACT_ROW_KEYS = frozenset(
    {
        "abstention_reason", "candidate_index", "confidence_by_alpha", "contains_evaluation_metrics",
        "contains_test_labels", "dataset", "eligibility", "scores_by_alpha",
    }
)


def _read_verified_file_bytes(path: Path, expected_sha256: str) -> bytes:
    """Read one immutable authority snapshot, then bind its exact bytes to SHA-256."""
    path = Path(path)
    try:
        payload = path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"oracle authority file not found: {path}") from error
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"oracle authority file SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return payload


def _oracle_json_record(payload: str, *, stage: str, line_number: int) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        record: dict[str, object] = {}
        for key, value in pairs:
            if key in record:
                raise ValueError(f"{stage} JSONL line {line_number} has duplicate key: {key}")
            record[key] = value
        return record

    try:
        record = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError(f"{stage} JSONL line {line_number} is invalid JSON") from error
    if not isinstance(record, dict):
        raise ValueError(f"{stage} JSONL line {line_number} must be an object")
    return record


def _oracle_dataset(value: object, *, stage: str, line_number: int) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{stage} dataset is invalid on line {line_number}")
    return value


def _oracle_positive_int(value: object, *, stage: str, field: str, line_number: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{stage} {field} must be a positive integer on line {line_number}")
    return value


def _oracle_indices(value: object, *, stage: str, field: str, line_number: int) -> frozenset[int]:
    if not isinstance(value, list) or any(type(index) is not int for index in value):
        raise ValueError(f"{stage} {field} must be an integer list on line {line_number}")
    result = frozenset(value)
    if len(result) != len(value):
        raise ValueError(f"{stage} {field} has duplicate coordinates on line {line_number}")
    return result


def _oracle_csv_indices(value: object, *, stage: str, field: str, line_number: int) -> frozenset[int]:
    if type(value) is not str:
        raise ValueError(f"{stage} {field} must be a space-delimited string on line {line_number}")
    text = value.strip()
    if not text:
        return frozenset()
    try:
        values = [int(item) for item in text.split()]
    except ValueError as error:
        raise ValueError(f"{stage} {field} has a non-integer coordinate on line {line_number}") from error
    result = frozenset(values)
    if len(result) != len(values):
        raise ValueError(f"{stage} {field} has duplicate coordinates on line {line_number}")
    return result


def _oracle_csv_n_test(value: object, *, line_number: int) -> int:
    if type(value) is not str:
        raise ValueError(f"Exp137 test_size must be a decimal string on line {line_number}")
    try:
        n_test = int(value)
    except ValueError as error:
        raise ValueError(f"Exp137 test_size must be an integer on line {line_number}") from error
    if n_test < 1:
        raise ValueError(f"Exp137 test_size must be positive on line {line_number}")
    return n_test


def _validate_coordinate_bounds(
    values: Mapping[str, frozenset[int]],
    n_test_by_dataset: Mapping[str, int],
    *,
    stage: str,
) -> None:
    for dataset, coordinates in values.items():
        n_test = n_test_by_dataset[dataset]
        if any(coordinate < 0 or coordinate >= n_test for coordinate in coordinates):
            raise ValueError(f"{stage} coordinate out of range for {dataset} (n_test={n_test})")


def load_hard_route_oracle_bundle(
    authority: HardRouteParityAuthority,
    *,
    selected_population: Sequence[str] | None = None,
    selected_n_test: Mapping[str, int] | None = None,
) -> HardRouteOracleBundle:
    """SHA-verify and globally validate label-free hard-route authorities before projection."""
    if not isinstance(authority, HardRouteParityAuthority):
        raise TypeError("HardRouteParityAuthority instance is required")
    exp137_bytes = _read_verified_file_bytes(authority.exp137_path, authority.exp137_sha256)
    exp236_bytes = _read_verified_file_bytes(authority.exp236_path, authority.exp236_sha256)
    exp291_bytes = _read_verified_file_bytes(authority.exp291_path, authority.exp291_sha256)
    exp404_bytes = _read_verified_file_bytes(authority.exp404_path, authority.exp404_sha256)

    exp137_global: dict[str, frozenset[int]] = {}
    exp137_n_test: dict[str, int] = {}
    with io.StringIO(exp137_bytes.decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != EXP137_EXACT_HEADER:
            raise ValueError("Exp137 CSV header schema mismatch")
        for line_number, row in enumerate(reader, 2):
            if set(row) != set(EXP137_EXACT_HEADER):
                raise ValueError(f"Exp137 CSV row schema mismatch on line {line_number}")
            dataset = _oracle_dataset(row["dataset_name"], stage="Exp137", line_number=line_number)
            if dataset in exp137_global:
                raise ValueError(f"Exp137 CSV contains duplicate dataset: {dataset}")
            exp137_global[dataset] = _oracle_csv_indices(
                row["hard_alert_indices"],
                stage="Exp137",
                field="hard_alert_indices",
                line_number=line_number,
            )
            exp137_n_test[dataset] = _oracle_csv_n_test(row["test_size"], line_number=line_number)
    if len(exp137_global) != 1117:
        raise ValueError(f"Exp137 dataset population count mismatch: expected 1117, got {len(exp137_global)}")
    if sum(map(len, exp137_global.values())) != 2005:
        raise ValueError("Exp137 total hard alert count mismatch: expected 2005")

    exp236_global: dict[str, frozenset[int]] = {}
    exp236_n_test: dict[str, int] = {}
    with io.StringIO(exp236_bytes.decode("utf-8")) as handle:
        for line_number, line in enumerate(handle, 1):
            payload = line.strip()
            if not payload:
                continue
            record = _oracle_json_record(payload, stage="Exp236", line_number=line_number)
            if set(record) != EXP236_EXACT_ROW_KEYS:
                raise ValueError(f"Exp236 JSONL line {line_number} schema mismatch")
            if record["contains_test_labels"] is not False or record["contains_evaluation_metrics"] is not False:
                raise ValueError(f"Exp236 JSONL line {line_number} violates the label-free authority boundary")
            dataset = _oracle_dataset(record["dataset"], stage="Exp236", line_number=line_number)
            if dataset in exp236_global:
                raise ValueError(f"Exp236 JSONL contains duplicate dataset: {dataset}")
            exp236_global[dataset] = _oracle_indices(
                record["hard_indices"],
                stage="Exp236",
                field="hard_indices",
                line_number=line_number,
            )
            exp236_n_test[dataset] = _oracle_positive_int(
                record["n_test"], stage="Exp236", field="n_test", line_number=line_number
            )
    if len(exp236_global) != 1117:
        raise ValueError(f"Exp236 dataset population count mismatch: expected 1117, got {len(exp236_global)}")
    if sum(map(len, exp236_global.values())) != 2116:
        raise ValueError("Exp236 total hard alert count mismatch: expected 2116")

    automatic_hard_global: dict[str, frozenset[int]] = {}
    exact_normal_removed_global: dict[str, frozenset[int]] = {}
    exp253_global: dict[str, frozenset[int]] = {}
    exp291_n_test: dict[str, int] = {}
    with io.StringIO(exp291_bytes.decode("utf-8")) as handle:
        for line_number, line in enumerate(handle, 1):
            payload = line.strip()
            if not payload:
                continue
            record = _oracle_json_record(payload, stage="Exp291", line_number=line_number)
            if set(record) != EXP291_EXACT_ROW_KEYS:
                raise ValueError(f"Exp291 JSONL line {line_number} schema mismatch")
            if record["contains_test_labels"] is not False or record["contains_evaluation_metrics"] is not False:
                raise ValueError(f"Exp291 JSONL line {line_number} violates the label-free authority boundary")
            dataset = _oracle_dataset(record["dataset"], stage="Exp291", line_number=line_number)
            if dataset in automatic_hard_global:
                raise ValueError(f"Exp291 JSONL contains duplicate dataset: {dataset}")
            automatic = _oracle_indices(
                record["automatic_hard_indices"],
                stage="Exp291",
                field="automatic_hard_indices",
                line_number=line_number,
            )
            removed = _oracle_indices(
                record["automatic_normal_exact_indices"],
                stage="Exp291",
                field="automatic_normal_exact_indices",
                line_number=line_number,
            )
            if automatic & removed:
                raise ValueError(f"Exp291 automatic and exact-normal coordinates overlap on line {line_number}")
            automatic_hard_global[dataset] = automatic
            exact_normal_removed_global[dataset] = removed
            exp253_global[dataset] = automatic | removed
            exp291_n_test[dataset] = _oracle_positive_int(
                record["n_test"], stage="Exp291", field="n_test", line_number=line_number
            )
    if len(automatic_hard_global) != 1117:
        raise ValueError(
            f"Exp291 dataset population count mismatch: expected 1117, got {len(automatic_hard_global)}"
        )
    if sum(map(len, exp253_global.values())) != 2253:
        raise ValueError("Exp253 pre-exact count mismatch: expected 2253")
    if sum(map(len, exact_normal_removed_global.values())) != 35:
        raise ValueError("Exp291 exact-normal removed count mismatch: expected 35")
    if sum(map(len, automatic_hard_global.values())) != 2218:
        raise ValueError("Exp291 automatic hard count mismatch: expected 2218")

    population_1117 = set(exp137_global)
    if set(exp236_global) != population_1117 or set(automatic_hard_global) != population_1117:
        raise ValueError("population discrepancy across Exp137, Exp236, and Exp291 oracle files")
    if exp137_n_test != exp236_n_test or exp137_n_test != exp291_n_test:
        raise ValueError("global n_test map mismatch across Exp137, Exp236, and Exp291")
    global_n_test = exp137_n_test
    _validate_coordinate_bounds(exp137_global, global_n_test, stage="Exp137")
    _validate_coordinate_bounds(exp236_global, global_n_test, stage="Exp236")
    _validate_coordinate_bounds(exp253_global, global_n_test, stage="Exp291")

    exp404_mutable: dict[str, set[int]] = {dataset: set() for dataset in population_1117}
    exp404_row_count = 0
    with io.StringIO(exp404_bytes.decode("utf-8")) as handle:
        for line_number, line in enumerate(handle, 1):
            payload = line.strip()
            if not payload:
                continue
            record = _oracle_json_record(payload, stage="Exp404", line_number=line_number)
            if set(record) != EXP404_EXACT_ROW_KEYS:
                raise ValueError(f"Exp404 JSONL line {line_number} schema mismatch")
            if record["contains_test_labels"] is not False or record["contains_evaluation_metrics"] is not False:
                raise ValueError(f"Exp404 JSONL line {line_number} violates the label-free authority boundary")
            dataset = _oracle_dataset(record["dataset"], stage="Exp404", line_number=line_number)
            if dataset not in exp404_mutable:
                raise ValueError(f"Exp404 JSONL contains unknown dataset: {dataset}")
            coordinate = record["candidate_index"]
            if type(coordinate) is not int or coordinate < 0:
                raise ValueError(f"Exp404 candidate_index must be a non-negative integer on line {line_number}")
            if coordinate < 0 or coordinate >= global_n_test[dataset]:
                raise ValueError(
                    f"Exp404 candidate coordinate out of range for {dataset} (n_test={global_n_test[dataset]})"
                )
            if coordinate in exp404_mutable[dataset]:
                raise ValueError(f"Exp404 duplicate coordinate ({dataset}, {coordinate}) on line {line_number}")
            exp404_mutable[dataset].add(coordinate)
            exp404_row_count += 1
    if exp404_row_count != 2218:
        raise ValueError(f"Exp404 row count drift: expected 2218, got {exp404_row_count}")
    exp404_global = {dataset: frozenset(coordinates) for dataset, coordinates in exp404_mutable.items()}
    if sum(map(len, exp404_global.values())) != 2218:
        raise ValueError("Exp404 total automatic hard count mismatch: expected 2218")
    if sum(1 for coordinates in exp404_global.values() if coordinates) != 888:
        raise ValueError("Exp404 nonempty dataset count mismatch: expected 888")
    if sum(1 for coordinates in exp404_global.values() if not coordinates) != 229:
        raise ValueError("Exp404 implicit-empty dataset count mismatch: expected 229")

    if selected_population is None:
        if selected_n_test is not None:
            if (
                not isinstance(selected_n_test, Mapping)
                or any(type(value) is not int for value in selected_n_test.values())
                or dict(selected_n_test) != global_n_test
            ):
                raise ValueError("selected_n_test must exactly equal the global n_test map")
        projection = tuple(exp137_global)
    else:
        if isinstance(selected_population, str):
            raise TypeError("selected_population must be a sequence of dataset names")
        projection = tuple(selected_population)
        if not projection or len(set(projection)) != len(projection) or any(type(dataset) is not str or not dataset for dataset in projection):
            raise ValueError("selected_population must contain unique nonempty dataset names")
        if selected_n_test is not None and not isinstance(selected_n_test, Mapping):
            raise TypeError("selected_n_test must be a mapping")
        if selected_n_test is not None and (
            set(selected_n_test) != set(projection)
            or any(type(value) is not int for value in selected_n_test.values())
        ):
            raise ValueError("selected_n_test must exactly match selected_population")
        for dataset in projection:
            expected_n_test = global_n_test[dataset]
            if selected_n_test is not None:
                if dataset not in selected_n_test or selected_n_test[dataset] != expected_n_test:
                    raise ValueError(f"selected_n_test mismatch for {dataset}: expected {expected_n_test}")

    return HardRouteOracleBundle(
        exp137={dataset: exp137_global[dataset] for dataset in projection},
        exp236={dataset: exp236_global[dataset] for dataset in projection},
        exp253={dataset: exp253_global[dataset] for dataset in projection},
        exact_normal_removed={dataset: exact_normal_removed_global[dataset] for dataset in projection},
        automatic_hard={dataset: automatic_hard_global[dataset] for dataset in projection},
        exp404={dataset: exp404_global[dataset] for dataset in projection},
    )


@dataclass(frozen=True, slots=True)
class ProposalScore:
    raw: Mapping[str, RawDataset]
    sources: Mapping[str, HardSourceBundle]
    routes: Mapping[str, HardRouteResult]
    low_frequency: LowFrequencyResult
    stability: StabilityPartition
    policies: PolicyBundle
    backend: FrozenTask8BackendDecision
    stage_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        raw, sources, routes, timing = dict(self.raw), dict(self.sources), dict(self.routes), dict(self.stage_seconds)
        if not raw or tuple(raw) != tuple(sorted(raw)) or set(raw) != set(sources) or set(raw) != set(routes) or set(raw) != set(self.low_frequency.datasets):
            raise ValueError("proposal score population identities differ")
        if any(not isinstance(value, RawDataset) or value.dataset != dataset for dataset, value in raw.items()) or any(not isinstance(value, HardSourceBundle) for value in sources.values()) or any(not isinstance(value, HardRouteResult) for value in routes.values()):
            raise ValueError("proposal score typed intermediates differ")
        if not isinstance(self.stability, StabilityPartition) or not isinstance(self.policies, PolicyBundle) or not isinstance(self.backend, FrozenTask8BackendDecision) or any(type(value) is not float or value < 0.0 for value in timing.values()):
            raise ValueError("proposal score result is invalid")
        object.__setattr__(self, "raw", MappingProxyType(raw))
        object.__setattr__(self, "sources", MappingProxyType(sources))
        object.__setattr__(self, "routes", MappingProxyType(routes))
        object.__setattr__(self, "stage_seconds", MappingProxyType(timing))


class ProposalScoreIncomplete(RuntimeError):
    """Typed operational result: Task4+5 lacks exact verified success coverage."""
    status = "proposal_score_incomplete"

    def __init__(self, *, completed: int, failed: int, pending: tuple[str, ...], attempts_path: str | None) -> None:
        self.completed = completed
        self.failed = failed
        self.pending = pending
        self.attempts_path = attempts_path
        super().__init__(f"proposal_score_incomplete completed={completed} failed={failed} pending={len(pending)} queue={','.join(pending)}")

    def artifact_record(self) -> dict[str, object]:
        return {"status": self.status, "completed": self.completed, "failed": self.failed, "pending": len(self.pending),
                "pending_datasets": list(self.pending), "attempts_path": self.attempts_path}


def _label_firewall() -> None:
    if "paper.equipment_anomaly_detection.label_access_paper" in sys.modules:
        raise RuntimeError("label_access_paper must remain unloaded during proposal scoring")


_NUMERICAL_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def _pin_inner_numerical_threads():
    previous = {name: os.environ.get(name) for name in _NUMERICAL_THREAD_ENV}
    try:
        for name in _NUMERICAL_THREAD_ENV:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _score_dataset_task4_task5(raw: RawDataset) -> tuple[HardSourceBundle, HardRouteResult]:
    sources = score_hard_sources(raw)
    mini = score_minirocket_confirmations(raw)
    return sources, replay_hard_route(raw, sources, mini)


def _progress(stage: str, completed: int, total: int, started: float, dataset: str | None = None) -> None:
    elapsed = float(monotonic() - started)
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    remaining = total - completed
    eta = remaining / rate if rate > 0.0 else 0.0
    suffix = f" dataset={dataset}" if dataset is not None else ""
    print(f"{stage} {completed}/{total} {100.0 * completed / total:.1f}% elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s{suffix}", file=sys.stderr, flush=True)


def make_stderr_progress_sink_paper(
    *,
    min_tick_interval_s: float = PROGRESS_TICK_INTERVAL_S,
    clock=monotonic,
    stream=None,
) -> ProgressCallback:
    """Format one flushed stderr line. Rate-gate ticks only."""
    output = sys.stderr if stream is None else stream
    last_tick: dict[str, float] = {}

    def sink(event: Mapping[str, object]) -> None:
        stage = str(event["stage"])
        phase = str(event["phase"])
        now = float(clock())
        if phase == "start":
            last_tick.pop(stage, None)
        elif phase == "tick":
            previous = last_tick.get(stage)
            if previous is not None and (now - previous) < min_tick_interval_s:
                return
            last_tick[stage] = now
        completed = int(event["completed"])
        total = int(event["total"])
        elapsed = float(event.get("elapsed_s", 0.0))
        pct = (100.0 * completed / total) if total > 0 else 0.0
        rate = (completed / elapsed) if elapsed > 0.0 else 0.0
        eta = ((total - completed) / rate) if rate > 0.0 else 0.0
        suffix = ""
        dataset = event.get("dataset")
        if type(dataset) is str:
            suffix += f" dataset={dataset}"
        detail = event.get("detail")
        if type(detail) is str:
            suffix += f" {detail}"
        print(
            f"{stage} {completed}/{total} {pct:.1f}% elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s{suffix}",
            file=output,
            flush=True,
        )

    return sink


Task4Task5SuccessPublisher = Callable[[RawDataset, HardSourceBundle, HardRouteResult], None]


def task4_task5_success_publisher(root, catalog: dict[str, object], authority: Task4Task5SuccessAuthority) -> Task4Task5SuccessPublisher:
    """Return a parent-only publisher; workers receive neither root nor authority."""
    if not isinstance(authority, Task4Task5SuccessAuthority):
        raise TypeError("Task4+5 success authority is required")
    def publish(raw: RawDataset, sources: HardSourceBundle, route: HardRouteResult) -> None:
        write_task4_task5_success_record(root, catalog, authority, raw, sources, route)
    return publish


def _risk_first_queue(raw: Mapping[str, RawDataset]) -> tuple[str, ...]:
    """Label-free operational ordering; completed records remain catalog-sorted on reconstruction."""
    def risk(values: RawDataset) -> tuple[int, str]:
        rows = tuple(values.train_rows) + tuple(values.test_rows)
        finite = tuple(bool(np.all(np.isfinite(row))) for row in rows)
        if not all(finite):
            return (0 if any(finite) else 1, values.dataset)
        if len(values.train_rows) < 4:
            return (2, values.dataset)
        if len(values.train_rows) >= 100:
            return (3, values.dataset)
        if values.metadata_length >= 1024:
            return (4, values.dataset)
        return (5, values.dataset)
    return tuple(sorted(raw, key=lambda dataset: risk(raw[dataset])))


def _collect_task4_task5(raw: Mapping[str, RawDataset], backend: FrozenTask8BackendDecision, *, root=None, context: dict[str, object] | None = None, catalog: dict[str, object] | None = None, acceleration_manifest: object | None = None, publish_success: Task4Task5SuccessPublisher | None = None, progress: ProgressCallback | None = None, strict_progress: ProgressCallback | None = None) -> tuple[dict[str, HardSourceBundle], dict[str, HardRouteResult]]:
    verify_exp84_environment_paper()
    verify_pycatch22_environment_paper()
    recovery_inputs = (root is not None, context is not None, catalog is not None)
    if any(recovery_inputs) and (not all(recovery_inputs) or acceleration_manifest is None):
        raise ValueError("Task4+5 recovery inputs are incomplete")
    authority = None
    sources: dict[str, HardSourceBundle] = {}
    routes: dict[str, HardRouteResult] = {}
    failed: list[str] = []
    pending_raw = dict(raw)
    wave = 1
    if all(recovery_inputs):
        if tuple(entry.get("dataset") for entry in catalog.get("datasets", ())) != tuple(raw) or any(not isinstance(values, RawDataset) or values.dataset != dataset for dataset, values in raw.items()):
            raise ValueError("Task4+5 catalog raw identity differs")
        authority = Task4Task5SuccessAuthority.from_verified_context(context, catalog, acceleration_manifest)
        verified = verify_task4_task5_success_namespace(root, catalog, authority, dict(raw))
        attempts = verify_task4_task5_attempt_namespace(root, catalog, authority, dict(raw))
        wave = max((record.wave for record in attempts), default=0) + 1
        for dataset in verified:
            record = read_task4_task5_success_record(root, catalog, authority, raw[dataset])
            sources[dataset], routes[dataset] = record.sources, record.route
            pending_raw.pop(dataset)
        if publish_success is None:
            publish_success = task4_task5_success_publisher(root, catalog, authority)
    queue = _risk_first_queue(pending_raw)
    if not queue:
        return dict(sorted(sources.items())), dict(sorted(routes.items()))
    started, total = monotonic(), len(queue)
    _progress("Task4+Task5 start", 0, total, started)
    emit_progress_paper(progress, {"stage": "task4_task5", "phase": "start", "completed": 0, "total": total, "elapsed_s": 0.0})
    def persist_result(dataset: str, result: tuple[HardSourceBundle, HardRouteResult]) -> None:
        sources[dataset], routes[dataset] = result
        if publish_success is not None:
            publish_success(raw[dataset], sources[dataset], routes[dataset])

    def persist_failure(dataset: str, error: Exception) -> None:
        failed.append(dataset)
        if authority is not None:
            append_task4_task5_attempt_record(root, catalog, authority, raw[dataset], wave, error)

    def publish_strict_checkpoint(dataset: str, completed: int) -> None:
        if strict_progress is None:
            return
        elapsed = float(monotonic() - started)
        emit_strict_progress_paper(strict_progress, {
            "stage": "task4_task5", "phase": "error" if dataset in failed else "tick",
            "completed": completed, "total": total, "dataset": dataset, "elapsed_s": elapsed,
        })

    if backend.dataset_workers == 1:
        for completed, dataset in enumerate(queue, 1):
            try:
                persist_result(dataset, _score_dataset_task4_task5(raw[dataset]))
            except DependencyUnavailablePaper:
                raise
            except Exception as error:
                persist_failure(dataset, error)
            publish_strict_checkpoint(dataset, completed)
            _progress("Task4+Task5", completed, total, started, dataset)
            elapsed = float(monotonic() - started)
            if dataset in failed:
                emit_progress_paper(progress, {"stage": "task4_task5", "phase": "error", "completed": completed, "total": total, "dataset": dataset, "elapsed_s": elapsed})
            elif completed == total and not failed:
                emit_progress_paper(progress, {"stage": "task4_task5", "phase": "end", "completed": completed, "total": total, "elapsed_s": elapsed})
    elif queue:
      with _pin_inner_numerical_threads():
        with ProcessPoolExecutor(max_workers=backend.dataset_workers) as executor:
            futures = {}
            iterator = iter(queue)
            window = min(len(queue), 2 * backend.dataset_workers)
            for _ in range(window):
                dataset = next(iterator)
                futures[executor.submit(_score_dataset_task4_task5, raw[dataset])] = dataset
            completed = 0
            while futures:
                future = next(iter(as_completed(tuple(futures))))
                dataset = futures.pop(future)
                completed += 1
                try:
                    persist_result(dataset, future.result())
                except DependencyUnavailablePaper:
                    raise
                except Exception as error:
                    persist_failure(dataset, error)
                publish_strict_checkpoint(dataset, completed)
                _progress("Task4+Task5", completed, total, started, dataset)
                elapsed = float(monotonic() - started)
                if dataset in failed:
                    emit_progress_paper(progress, {"stage": "task4_task5", "phase": "error", "completed": completed, "total": total, "dataset": dataset, "elapsed_s": elapsed})
                elif completed == total and not failed:
                    emit_progress_paper(progress, {"stage": "task4_task5", "phase": "end", "completed": completed, "total": total, "elapsed_s": elapsed})
                try:
                    next_dataset = next(iterator)
                except StopIteration:
                    continue
                futures[executor.submit(_score_dataset_task4_task5, raw[next_dataset])] = next_dataset
    if failed:
        pending = tuple(sorted(failed))
        attempts_path = str(root / "stage_artifacts_paper" / "proposal_work" / "task4_task5_attempts") if root is not None else None
        raise ProposalScoreIncomplete(completed=len(sources), failed=len(failed), pending=pending, attempts_path=attempts_path)
    return dict(sorted(sources.items())), dict(sorted(routes.items()))


def compose_proposal_score(raw_datasets: Mapping[str, RawDataset], acceleration_manifest: object, *, root=None, context: dict[str, object] | None = None, catalog: dict[str, object] | None = None, progress: ProgressCallback | None = None) -> ProposalScore:
    """Execute Task4+5 independently, then exactly one population-global Task6 and selected-only Task7."""
    _label_firewall()
    raw = dict(raw_datasets)
    if not raw or tuple(raw) != tuple(sorted(raw)) or any(not isinstance(values, RawDataset) or values.dataset != dataset for dataset, values in raw.items()):
        raise ValueError("proposal raw dataset mapping is invalid")
    backend = FrozenTask8BackendDecision.from_acceleration_manifest(acceleration_manifest)
    supplied = (root is not None, context is not None, catalog is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("proposal success publication inputs are incomplete")
    publisher = None
    if all(supplied):
        authority = Task4Task5SuccessAuthority.from_verified_context(context, catalog, acceleration_manifest)
        publisher = task4_task5_success_publisher(root, catalog, authority)
    started = monotonic()
    sources, routes = _collect_task4_task5(raw, backend, root=root, context=context, catalog=catalog, acceleration_manifest=acceleration_manifest, publish_success=publisher, progress=progress)
    hard_seconds = float(monotonic() - started)
    _label_firewall()
    print("Task6 start", file=sys.stderr, flush=True)
    low_started = monotonic()
    hard_lanes = {dataset: routes[dataset].exp253.hard for dataset in raw}
    low = score_low_frequency(raw, hard_lanes) if progress is None else score_low_frequency(raw, hard_lanes, progress=progress)
    low_seconds = float(monotonic() - low_started)
    print("Task6 finish", file=sys.stderr, flush=True)
    _label_firewall()
    print("Task7 start", file=sys.stderr, flush=True)
    stability_started = monotonic()
    stability = classify_stability(low, raw) if progress is None else classify_stability(low, raw, progress=progress)
    stability_seconds = float(monotonic() - stability_started)
    print("Task7 finish", file=sys.stderr, flush=True)
    surface = build_hard_evidence_surface(sources, routes, raw) if progress is None else build_hard_evidence_surface(sources, routes, raw, progress=progress)
    policies = build_policies(surface, low, stability, raw, routes) if progress is None else build_policies(surface, low, stability, raw, routes, progress=progress)
    _label_firewall()
    return ProposalScore(raw, sources, routes, low, stability, policies, backend, {"task4_task5": hard_seconds, "task6": low_seconds, "task7": stability_seconds, "total": float(monotonic() - started)})


def write_and_freeze_proposal_score(root, catalog: dict[str, object], proposal: ProposalScore, *, progress: ProgressCallback | None = None) -> Task8FrozenManifest:
    """Pass only Task4--7 typed values to Task8's create-only shard writer."""
    if not isinstance(proposal, ProposalScore) or tuple(entry["dataset"] for entry in catalog.get("datasets", ())) != tuple(proposal.raw):
        raise ValueError("proposal score catalog identity differs")
    names = tuple(proposal.raw)
    started = monotonic()
    total = len(names)
    completed = 0
    emit_progress_paper(progress, {"stage": "shard_write", "phase": "start", "completed": 0, "total": total, "elapsed_s": 0.0})
    try:
        for completed, dataset in enumerate(names, 1):
            write_typed_proposal_shard(
                root, catalog, proposal.raw[dataset], proposal.sources[dataset],
                proposal.routes[dataset], proposal.low_frequency, proposal.stability,
                proposal.policies, proposal.backend,
            )
            emit_progress_paper(progress, {
                "stage": "shard_write", "phase": "tick",
                "completed": completed, "total": total,
                "dataset": dataset, "detail": f"{dataset}_paper.json",
                "elapsed_s": float(monotonic() - started),
            })
        emit_progress_paper(progress, {"stage": "shard_write", "phase": "end", "completed": total, "total": total, "elapsed_s": float(monotonic() - started)})
    except Exception:
        emit_progress_paper(progress, {"stage": "shard_write", "phase": "error", "completed": completed if names else 0, "total": total, "elapsed_s": float(monotonic() - started)})
        raise
    return freeze_score_root(root, catalog, progress=progress)


__all__ = ["ProposalScore", "ProposalScoreIncomplete", "Task4Task5SuccessPublisher", "compose_proposal_score", "load_hard_route_oracle_bundle", "make_stderr_progress_sink_paper", "task4_task5_success_publisher", "write_and_freeze_proposal_score"]
