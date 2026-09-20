"""Single-use, post-freeze Exp448 evaluation with a closed label boundary.

This module deliberately has no SQLite, network, or dashboard import.  The only
label capability arrives through the closure-private authority in
``sources_antigravity`` after the two independent publication captures agree.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from uuid import uuid4

from .artifacts_antigravity import canonical_json_bytes_antigravity, write_create_only_antigravity
from .bootstrap_antigravity import bootstrap_draws_antigravity, paired_bootstrap_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    BootstrapComparisonAntigravity,
    ConfusionAntigravity,
    DatasetConfusionRowAntigravity,
    MetricSummaryAntigravity,
    PublicationIdentityAntigravity,
)
from .sources_antigravity import (
    capture_exp446_publication_antigravity,
    capture_hard_predictions_antigravity,
    open_test_labels_antigravity,
)

from .replay_antigravity import (
    FrozenEvaluationAntigravity,
    _DIAGNOSTIC_NAMES_ANTIGRAVITY,
    _LEAVES_ANTIGRAVITY,
    _METHODS_ANTIGRAVITY,
    _PRIMARY_METHOD_ANTIGRAVITY,
    _confusion_antigravity,
    _exact_commit_antigravity,
    _jsonl_bytes_antigravity,
    _metric_summary_antigravity,
    inspect_replay_with_dependencies_antigravity,
)

class _ProductionDependenciesAntigravity:
    """Small adapter kept separate from the deterministic evaluation algorithm."""

    root = Path(FROZEN_CONFIG_ANTIGRAVITY.run_root)

    @staticmethod
    def capture_exp446() -> PublicationIdentityAntigravity:
        return capture_exp446_publication_antigravity()

    @staticmethod
    def capture_exp404(identity: PublicationIdentityAntigravity):
        return capture_hard_predictions_antigravity(identity)

    @staticmethod
    def derive_hard(hard: object, identity: PublicationIdentityAntigravity) -> frozenset[tuple[str, int]]:
        if not isinstance(identity, PublicationIdentityAntigravity) or not hasattr(hard, "coordinates"):
            raise ValueError("verified Hard input is required")
        coordinates = getattr(hard, "coordinates")
        if not isinstance(coordinates, frozenset):
            raise ValueError("Hard coordinate authority is invalid")
        return coordinates

    @staticmethod
    def compare_exp446(first: PublicationIdentityAntigravity, second: PublicationIdentityAntigravity) -> None:
        if first != second:
            raise ValueError("Exp446 inspection drift")

    @staticmethod
    def open_labels(first: PublicationIdentityAntigravity, second: PublicationIdentityAntigravity, hard: object):
        return open_test_labels_antigravity(first, second, hard)

    @staticmethod
    def capture_after() -> PublicationIdentityAntigravity:
        return capture_exp446_publication_antigravity()

    @staticmethod
    def bootstrap(baseline: object, comparator: object, name: str, *, draws: object) -> Mapping[str, object]:
        comparison = paired_bootstrap_antigravity(baseline, comparator, name, draws=draws)
        return comparison.to_payload_antigravity()


def _coordinates_by_dataset_antigravity(coordinates: Sequence[tuple[str, int]], order: Sequence[str]) -> dict[str, frozenset[int]]:
    grouped: dict[str, set[int]] = {dataset: set() for dataset in order}
    for dataset, index in coordinates:
        if dataset not in grouped or type(index) is not int or index < 0:
            raise ValueError("candidate coordinate differs from frozen dataset population")
        grouped[dataset].add(index)
    return {dataset: frozenset(indices) for dataset, indices in grouped.items()}


def _validate_labels_antigravity(dataset: str, rows: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(rows, tuple):
        rows = tuple(rows) if isinstance(rows, Sequence) else None
    if rows is None or any(not isinstance(row, tuple) or len(row) != 2 or type(row[0]) is not int or type(row[1]) is not int or row[1] not in (0, 1) for row in rows):
        raise ValueError(f"TEST labels are invalid for {dataset}")
    if tuple(index for index, _ in rows) != tuple(range(len(rows))):
        raise ValueError(f"TEST indices are not contiguous for {dataset}")
    return rows


def _artifact_payloads_antigravity(
    result: FrozenEvaluationAntigravity,
    dataset_rows: Mapping[str, Sequence[DatasetConfusionRowAntigravity]],
    transitions: Sequence[Mapping[str, object]],
    diagnostics: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, bytes]:
    metrics_rows: list[dict[str, object]] = []
    for index, dataset in enumerate(result.identity.dataset_order):
        row: dict[str, object] = {"dataset": dataset, "dataset_order": index}
        for method in _METHODS_ANTIGRAVITY:
            confusion = dataset_rows[method][index].confusion
            row[method] = {"tp": confusion.tp, "fp": confusion.fp, "fn": confusion.fn, "tn": confusion.tn, "f1": confusion.f1}
        metrics_rows.append(row)
    evaluation = {
        "schema": "exp448", "experiment_id": "Exp448", "run_identity": FROZEN_CONFIG_ANTIGRAVITY.run_identity,
        "reviewed_commit": result.reviewed_commit, "dataset_count": result.dataset_count, "label_rows": result.label_rows,
        "evaluation_status": result.evaluation_status, "review_only": True, "operational_alert_eligible": False,
        "scientific_winner_selected": False, "same_experiment_reselection_allowed": False,
        "test_informed_reselection_allowed": False, "automatic_champion_promotion": False,
        "exp446_prediction_sha256": result.identity.prediction_sha256,
    }
    confusion = {"methods": [result.metrics[method].to_payload_antigravity() for method in _METHODS_ANTIGRAVITY]}
    source_historical_agreement = result.metrics["hard_plus_source_30"].confusion == ConfusionAntigravity(*FROZEN_CONFIG_ANTIGRAVITY.exp442_source_30_confusion)
    decision = {
        "evaluation_status": result.evaluation_status, "scientific_success_criterion": None,
        "scientific_winner_selected": False, "review_only": True, "operational_alert_eligible": False,
        "same_experiment_reselection_allowed": False, "test_informed_reselection_allowed": False,
        "automatic_champion_promotion": False,
        "historical_source_30_confusion_agreement": source_historical_agreement,
        "historical_source_30_confusion": {"tp": FROZEN_CONFIG_ANTIGRAVITY.exp442_source_30_confusion[0], "fp": FROZEN_CONFIG_ANTIGRAVITY.exp442_source_30_confusion[1], "fn": FROZEN_CONFIG_ANTIGRAVITY.exp442_source_30_confusion[2], "tn": FROZEN_CONFIG_ANTIGRAVITY.exp442_source_30_confusion[3]},
        "historical_exp442_evaluation_sha256": FROZEN_CONFIG_ANTIGRAVITY.exp442_evaluation_sha256,
        "historical_exp442_artifact_map_sha256": FROZEN_CONFIG_ANTIGRAVITY.exp442_evaluation_artifact_map_sha256,
    }
    return {
        "evaluation_antigravity.json": canonical_json_bytes_antigravity(evaluation),
        "dataset_metrics_antigravity.jsonl": _jsonl_bytes_antigravity(metrics_rows),
        "candidate_transitions_antigravity.jsonl": _jsonl_bytes_antigravity(transitions),
        "rescued_datasets_antigravity.jsonl": _jsonl_bytes_antigravity(diagnostics["rescued"]),
        "false_alarm_datasets_antigravity.jsonl": _jsonl_bytes_antigravity(diagnostics["false_alarm"]),
        "regressed_datasets_antigravity.jsonl": _jsonl_bytes_antigravity(diagnostics["regressed"]),
        "strong_regressions_antigravity.jsonl": _jsonl_bytes_antigravity(diagnostics["strong_regression"]),
        "bootstrap_antigravity.json": canonical_json_bytes_antigravity({"comparisons": list(result.bootstrap)}),
        "confusion_antigravity.json": canonical_json_bytes_antigravity(confusion),
        "decision_antigravity.json": canonical_json_bytes_antigravity(decision),
    }


def _write_publication_antigravity(root: Path, payloads: Mapping[str, bytes]) -> None:
    if root.exists() or root.is_symlink():
        raise ValueError("Exp448 evaluation root already exists and is immutable")
    root.mkdir(mode=0o700)
    try:
        for name in _LEAVES_ANTIGRAVITY[:-1]:
            write_create_only_antigravity(root / name, payloads[name])
        artifact_map = {name: sha256(payloads[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
        write_create_only_antigravity(root / _LEAVES_ANTIGRAVITY[-1], canonical_json_bytes_antigravity(artifact_map))
    except BaseException:
        raise


def _destination_state_antigravity(root: Path) -> str:
    if root.is_symlink():
        return "quarantined"
    if not root.exists():
        return "absent"
    if not root.is_dir():
        return "quarantined"
    names = {path.name for path in root.iterdir()}
    return "complete" if names == set(_LEAVES_ANTIGRAVITY) else "partial"


def _write_audit_antigravity(root: Path, phase: str, identity: PublicationIdentityAntigravity | None, *, attempts: int, completions: int, rows: int, closed: bool, session_state: str, close_error: BaseException | None, destination_state: str, error: BaseException) -> None:
    # Audit is a distinct sibling, not a mutation or completion of any destination.
    parent = root.parent
    if not parent.is_dir():
        return
    payload = {
        "schema": "exp448_evaluation_audit", "phase": phase, "query_attempts": attempts,
        "query_completions": completions, "label_rows": rows, "label_session_closed": closed,
        "label_session_state": session_state,
        "destination_state": destination_state, "error_type": type(error).__name__, "error": str(error),
        "close_error_type": None if close_error is None else type(close_error).__name__,
        "close_error": None if close_error is None else str(close_error),
        "exp446_identity": None if identity is None else {
            "root_device": identity.root_device, "root_inode": identity.root_inode,
            "manifest_sha256": identity.manifest_sha256, "artifact_map_sha256": identity.artifact_map_sha256,
            "prediction_sha256": identity.prediction_sha256, "leaf_sha256": dict(identity.leaf_sha256),
        },
    }
    write_create_only_antigravity(parent / f"evaluation_audit_{uuid4().hex}_antigravity.json", canonical_json_bytes_antigravity(payload))


def _evaluate_with_dependencies_antigravity(reviewed_commit: object, dependencies: object) -> FrozenEvaluationAntigravity:
    """Dependency-injected core used only by tests; public entrypoint is below."""
    commit = _exact_commit_antigravity(reviewed_commit)
    root = Path(getattr(dependencies, "root"))
    identity: PublicationIdentityAntigravity | None = None
    labels_opened = False
    label_boundary_entered = False
    labels_closed = False
    attempts = queries = rows_read = 0
    phase = "pre_label"
    session = None
    session_state = "pre_label"
    close_attempted = False
    close_failure: BaseException | None = None
    try:
        if root.exists() or root.is_symlink():
            raise ValueError("Exp448 evaluation root already exists and is immutable")
        identity = dependencies.capture_exp446()
        hard = dependencies.capture_exp404(identity)
        hard_coordinates = dependencies.derive_hard(hard, identity)
        second = dependencies.capture_exp446()
        dependencies.compare_exp446(identity, second)
        phase = "label_open"
        label_boundary_entered = True
        session_state = "setup_opening"
        try:
            session = dependencies.open_labels(identity, second, hard)
        except BaseException:
            session_state = "setup_failed"
            raise
        labels_opened = True
        session_state = "open"
        hard_by_dataset = _coordinates_by_dataset_antigravity(tuple(hard_coordinates), identity.dataset_order)
        source_by_dataset = _coordinates_by_dataset_antigravity(identity.source_coordinates, identity.dataset_order)
        stable_by_dataset = _coordinates_by_dataset_antigravity(identity.stable_coordinates, identity.dataset_order)
        unstable_by_dataset = _coordinates_by_dataset_antigravity(identity.unstable_coordinates, identity.dataset_order)
        rows_by_method: dict[str, list[DatasetConfusionRowAntigravity]] = {method: [] for method in _METHODS_ANTIGRAVITY}
        transitions: list[dict[str, object]] = []
        diagnostics: dict[str, list[dict[str, object]]] = {name: [] for name in _DIAGNOSTIC_NAMES_ANTIGRAVITY}
        for dataset_order, dataset in enumerate(identity.dataset_order):
            phase = "label_snapshot"
            attempts += 1
            labels = _validate_labels_antigravity(dataset, session.labels_for_dataset_antigravity(dataset))
            queries += 1
            rows_read += len(labels)
            max_index = len(labels)
            candidate_sets = {
                "hard_baseline": hard_by_dataset[dataset],
                "hard_plus_source_30": hard_by_dataset[dataset] | source_by_dataset[dataset],
                "hard_plus_stable_26": hard_by_dataset[dataset] | stable_by_dataset[dataset],
                "hard_plus_unstable_4": hard_by_dataset[dataset] | unstable_by_dataset[dataset],
            }
            for method, candidates in candidate_sets.items():
                if any(index >= max_index for index in candidates):
                    raise ValueError(f"candidate index is outside TEST range for {dataset}")
                rows_by_method[method].append(DatasetConfusionRowAntigravity(dataset, _confusion_antigravity(labels, candidates)))
            label_by_index = {index: label for index, label in labels}
            for index in sorted(source_by_dataset[dataset]):
                lane = "stable" if index in stable_by_dataset[dataset] else "unstable"
                transitions.append({"dataset": dataset, "dataset_order": dataset_order, "candidate_index": index, "lane": lane, "already_hard": index in hard_by_dataset[dataset], "transition": "hard_overlap" if index in hard_by_dataset[dataset] else ("rescued" if label_by_index[index] == 1 else "added_false_alarm")})
        if (queries, rows_read) != (FROZEN_CONFIG_ANTIGRAVITY.dataset_count, FROZEN_CONFIG_ANTIGRAVITY.label_rows):
            raise ValueError("TEST label query or row population differs from frozen contract")
        phase = "label_close"
        close_attempted = True
        try:
            session.close_antigravity()
        except BaseException as error:
            close_failure = error
            session_state = _session_close_state_antigravity(session, "close_failed")
            raise
        labels_closed = True
        session_state = _session_close_state_antigravity(session, "closed")
        phase = "post_close_inspection"
        dependencies.compare_exp446(identity, dependencies.capture_after())
        metrics = {method: _metric_summary_antigravity(method, method_rows) for method, method_rows in rows_by_method.items()}
        if metrics["hard_baseline"].confusion != ConfusionAntigravity(*FROZEN_CONFIG_ANTIGRAVITY.baseline_confusion):
            raise ValueError("Hard baseline confusion differs from frozen authority")
        if metrics["hard_plus_source_30"].confusion != ConfusionAntigravity(*FROZEN_CONFIG_ANTIGRAVITY.exp442_source_30_confusion):
            raise ValueError("source-30 historical confusion differs from frozen Exp442 authority")
        stable_rows, hard_rows = rows_by_method[_PRIMARY_METHOD_ANTIGRAVITY], rows_by_method["hard_baseline"]
        for dataset, hard_row, stable_row in zip(identity.dataset_order, hard_rows, stable_rows, strict=True):
            if stable_row.confusion.tp > hard_row.confusion.tp:
                diagnostics["rescued"].append({"dataset": dataset, "tp_increase": stable_row.confusion.tp - hard_row.confusion.tp})
            if stable_row.confusion.fp > hard_row.confusion.fp:
                diagnostics["false_alarm"].append({"dataset": dataset, "fp_increase": stable_row.confusion.fp - hard_row.confusion.fp})
            delta = stable_row.confusion.f1 - hard_row.confusion.f1
            if delta < 0.0:
                diagnostics["regressed"].append({"dataset": dataset, "f1_delta": delta})
                if hard_row.confusion.f1 >= 0.8:
                    diagnostics["strong_regression"].append({"dataset": dataset, "f1_delta": delta})
        phase = "post_compute_inspection"
        dependencies.compare_exp446(identity, dependencies.capture_after())
        draws = bootstrap_draws_antigravity(FROZEN_CONFIG_ANTIGRAVITY.dataset_count)
        comparisons = (
            dependencies.bootstrap(hard_rows, rows_by_method["hard_plus_source_30"], "hard_plus_source_30_minus_hard_baseline", draws=draws),
            dependencies.bootstrap(hard_rows, stable_rows, "hard_plus_stable_26_minus_hard_baseline", draws=draws),
            dependencies.bootstrap(rows_by_method["hard_plus_source_30"], stable_rows, "hard_plus_stable_26_minus_hard_plus_source_30", draws=draws),
        )
        transition_counts = {"source": len(identity.source_coordinates), "stable": len(identity.stable_coordinates), "unstable": len(identity.unstable_coordinates)}
        if transition_counts["source"] != transition_counts["stable"] + transition_counts["unstable"] or len(transitions) != transition_counts["source"]:
            raise ValueError("candidate transition lanes do not conserve frozen candidates")
        result = FrozenEvaluationAntigravity(commit, identity, queries, rows_read, metrics, transition_counts, {name: len(rows) for name, rows in diagnostics.items()}, comparisons)
        phase = "pre_commit_inspection"
        dependencies.compare_exp446(identity, dependencies.capture_after())
        payloads = _artifact_payloads_antigravity(result, rows_by_method, transitions, diagnostics)
        phase = "publication"
        _write_publication_antigravity(root, payloads)
        return result
    except BaseException as error:
        if session is not None and not labels_closed and not close_attempted:
            close_attempted = True
            try:
                session.close_antigravity()
                labels_closed = True
                session_state = _session_close_state_antigravity(session, "closed")
            except BaseException as close_error:
                close_failure = close_error
                session_state = _session_close_state_antigravity(session, "close_failed")
        if label_boundary_entered:
            try:
                _write_audit_antigravity(root, phase, identity, attempts=attempts, completions=queries, rows=rows_read, closed=labels_closed, session_state=session_state, close_error=close_failure, destination_state=_destination_state_antigravity(root), error=error)
            except BaseException:
                # The original failure must remain the caller-visible truth when audit storage itself fails.
                pass
        raise


def _session_close_state_antigravity(session: object, fallback: str) -> str:
    """Read the closure-private state when available without permitting retry."""
    state = getattr(session, "close_state_antigravity", None)
    if not callable(state):
        return fallback
    value = state()
    return value if value in {"open", "closed", "close_failed"} else fallback


def evaluate_antigravity(reviewed_commit: str) -> FrozenEvaluationAntigravity:
    """Evaluate immutable Exp446/Exp404 predictions exactly once after readiness."""
    return _evaluate_with_dependencies_antigravity(reviewed_commit, _ProductionDependenciesAntigravity())


def _inspect_root_with_dependencies_antigravity(dependencies: object) -> FrozenEvaluationAntigravity:
    """Compatibility wrapper around the source-free immutable replay primitive."""
    return inspect_replay_with_dependencies_antigravity(dependencies)


def inspect_evaluation_antigravity() -> FrozenEvaluationAntigravity:
    """Replay a frozen completed evaluation without opening TEST labels."""
    return inspect_replay_with_dependencies_antigravity(_ProductionDependenciesAntigravity())


__all__ = ["FrozenEvaluationAntigravity", "evaluate_antigravity", "inspect_evaluation_antigravity"]
