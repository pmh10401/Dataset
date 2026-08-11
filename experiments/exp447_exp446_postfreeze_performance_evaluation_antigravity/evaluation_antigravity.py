"""Single-use, post-freeze Exp447 evaluation with a closed label boundary.

This module deliberately has no SQLite, network, or dashboard import.  The only
label capability arrives through the closure-private authority in
``sources_antigravity`` after the two independent publication captures agree.
"""
from __future__ import annotations

from collections import defaultdict
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

_LEAVES_ANTIGRAVITY = (
    "evaluation_antigravity.json", "dataset_metrics_antigravity.jsonl",
    "candidate_transitions_antigravity.jsonl", "rescued_datasets_antigravity.jsonl",
    "false_alarm_datasets_antigravity.jsonl", "regressed_datasets_antigravity.jsonl",
    "strong_regressions_antigravity.jsonl", "bootstrap_antigravity.json",
    "confusion_antigravity.json", "decision_antigravity.json",
    "artifact_sha256_antigravity.json",
)
_METHODS_ANTIGRAVITY = (
    "hard_baseline", "hard_plus_source_30", "hard_plus_stable_26", "hard_plus_unstable_4",
)
_PRIMARY_METHOD_ANTIGRAVITY = "hard_plus_stable_26"


def _exact_commit_antigravity(value: object) -> str:
    if type(value) is not str or len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("reviewed commit must be a lowercase 40-hex SHA")
    return value


def _jsonl_bytes_antigravity(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes_antigravity(row) for row in rows)


def _confusion_antigravity(labels: Sequence[tuple[int, int]], candidates: frozenset[int]) -> ConfusionAntigravity:
    tp = fp = fn = tn = 0
    for index, label in labels:
        predicted = index in candidates
        if predicted and label == 1:
            tp += 1
        elif predicted:
            fp += 1
        elif label == 1:
            fn += 1
        else:
            tn += 1
    return ConfusionAntigravity(tp=tp, fp=fp, fn=fn, tn=tn)


def _sum_confusions_antigravity(rows: Sequence[DatasetConfusionRowAntigravity]) -> ConfusionAntigravity:
    return ConfusionAntigravity(
        tp=sum(row.confusion.tp for row in rows), fp=sum(row.confusion.fp for row in rows),
        fn=sum(row.confusion.fn for row in rows), tn=sum(row.confusion.tn for row in rows),
    )


def _metric_summary_antigravity(name: str, rows: Sequence[DatasetConfusionRowAntigravity]) -> MetricSummaryAntigravity:
    total = _sum_confusions_antigravity(rows)
    return MetricSummaryAntigravity(name, total, sum(row.confusion.f1 for row in rows) / len(rows))


@dataclass(frozen=True, slots=True)
class FrozenEvaluationAntigravity:
    """Read-only evaluated evidence returned by evaluation and inspect."""

    reviewed_commit: str
    identity: PublicationIdentityAntigravity
    dataset_count: int
    label_rows: int
    metrics: Mapping[str, MetricSummaryAntigravity]
    transition_counts: Mapping[str, int]
    diagnostic_counts: Mapping[str, int]
    bootstrap: tuple[Mapping[str, object], ...]
    evaluation_status: str = "descriptive_complete_no_promotion"

    def __post_init__(self) -> None:
        _exact_commit_antigravity(self.reviewed_commit)
        if not isinstance(self.identity, PublicationIdentityAntigravity):
            raise ValueError("frozen publication identity is required")
        if (self.dataset_count, self.label_rows) != (FROZEN_CONFIG_ANTIGRAVITY.dataset_count, FROZEN_CONFIG_ANTIGRAVITY.label_rows):
            raise ValueError("evaluated population differs from frozen contract")
        if tuple(self.metrics) != _METHODS_ANTIGRAVITY:
            raise ValueError("evaluation methods differ from frozen contract")
        if any(not isinstance(summary, MetricSummaryAntigravity) for summary in self.metrics.values()):
            raise ValueError("metric summary is invalid")
        if self.evaluation_status != "descriptive_complete_no_promotion":
            raise ValueError("Exp447 may not select or promote a winner")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "transition_counts", MappingProxyType(dict(self.transition_counts)))
        object.__setattr__(self, "diagnostic_counts", MappingProxyType(dict(self.diagnostic_counts)))
        object.__setattr__(self, "bootstrap", tuple(MappingProxyType(dict(row)) for row in self.bootstrap))


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
    def open_labels(first: PublicationIdentityAntigravity, second: PublicationIdentityAntigravity):
        return open_test_labels_antigravity(first, second)

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
        "schema": "exp447", "experiment_id": "Exp447", "run_identity": FROZEN_CONFIG_ANTIGRAVITY.run_identity,
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
        raise ValueError("Exp447 evaluation root already exists and is immutable")
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


def _write_audit_antigravity(root: Path, phase: str, identity: PublicationIdentityAntigravity | None, *, attempts: int, completions: int, rows: int, closed: bool, close_error: BaseException | None, destination_state: str, error: BaseException) -> None:
    # Audit is a distinct sibling, not a mutation or completion of any destination.
    parent = root.parent
    if not parent.is_dir():
        return
    payload = {
        "schema": "exp447_evaluation_audit", "phase": phase, "query_attempts": attempts,
        "query_completions": completions, "label_rows": rows, "label_session_closed": closed,
        "destination_state": destination_state, "error_type": type(error).__name__, "error": str(error),
        "close_error_type": None if close_error is None else type(close_error).__name__,
        "close_error": None if close_error is None else str(close_error),
        "exp446_prediction_sha256": None if identity is None else identity.prediction_sha256,
    }
    write_create_only_antigravity(parent / f"evaluation_audit_{uuid4().hex}_antigravity.json", canonical_json_bytes_antigravity(payload))


def _evaluate_with_dependencies_antigravity(reviewed_commit: object, dependencies: object) -> FrozenEvaluationAntigravity:
    """Dependency-injected core used only by tests; public entrypoint is below."""
    commit = _exact_commit_antigravity(reviewed_commit)
    root = Path(getattr(dependencies, "root"))
    identity: PublicationIdentityAntigravity | None = None
    labels_opened = False
    labels_closed = False
    attempts = queries = rows_read = 0
    phase = "pre_label"
    session = None
    close_failure: BaseException | None = None
    try:
        if root.exists() or root.is_symlink():
            raise ValueError("Exp447 evaluation root already exists and is immutable")
        identity = dependencies.capture_exp446()
        hard = dependencies.capture_exp404(identity)
        hard_coordinates = dependencies.derive_hard(hard, identity)
        second = dependencies.capture_exp446()
        dependencies.compare_exp446(identity, second)
        phase = "label_open"
        session = dependencies.open_labels(identity, second)
        labels_opened = True
        hard_by_dataset = _coordinates_by_dataset_antigravity(tuple(hard_coordinates), identity.dataset_order)
        source_by_dataset = _coordinates_by_dataset_antigravity(identity.source_coordinates, identity.dataset_order)
        stable_by_dataset = _coordinates_by_dataset_antigravity(identity.stable_coordinates, identity.dataset_order)
        unstable_by_dataset = _coordinates_by_dataset_antigravity(identity.unstable_coordinates, identity.dataset_order)
        rows_by_method: dict[str, list[DatasetConfusionRowAntigravity]] = {method: [] for method in _METHODS_ANTIGRAVITY}
        transitions: list[dict[str, object]] = []
        diagnostics: dict[str, list[dict[str, object]]] = defaultdict(list)
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
        session.close_antigravity()
        labels_closed = True
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
        if session is not None and not labels_closed:
            try:
                session.close_antigravity()
                labels_closed = True
            except BaseException as close_error:
                close_failure = close_error
        if labels_opened:
            try:
                _write_audit_antigravity(root, phase, identity, attempts=attempts, completions=queries, rows=rows_read, closed=labels_closed, close_error=close_failure, destination_state=_destination_state_antigravity(root), error=error)
            except BaseException:
                # The original failure must remain the caller-visible truth when audit storage itself fails.
                pass
        raise
    finally:
        if session is not None and not labels_closed:
            try:
                session.close_antigravity()
            except BaseException:
                pass


def evaluate_antigravity(reviewed_commit: str) -> FrozenEvaluationAntigravity:
    """Evaluate immutable Exp446/Exp404 predictions exactly once after readiness."""
    return _evaluate_with_dependencies_antigravity(reviewed_commit, _ProductionDependenciesAntigravity())


def _read_json_antigravity(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid immutable evaluation leaf: {path.name}") from error


def _json_bytes_antigravity(payload: bytes, name: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid immutable evaluation leaf: {name}") from error


def _parse_jsonl_bytes_antigravity(payload: bytes, name: str) -> tuple[object, ...]:
    try:
        return tuple(json.loads(line) for line in payload.decode("utf-8").splitlines())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid immutable JSONL leaf: {name}") from error


def _pinned_evaluation_leaves_antigravity(root: Path) -> dict[str, bytes]:
    """Read exact regular leaves through one no-follow directory descriptor."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    leaf_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(root, flags)
    try:
        before = os.fstat(descriptor)
        if set(os.listdir(descriptor)) != set(_LEAVES_ANTIGRAVITY):
            raise ValueError("Exp447 artifact inventory is not exact")
        leaves: dict[str, bytes] = {}
        for name in _LEAVES_ANTIGRAVITY:
            leaf = os.open(name, leaf_flags, dir_fd=descriptor)
            try:
                opened = os.fstat(leaf)
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                    raise ValueError("evaluation artifact leaf is not a single-link regular file")
                chunks: list[bytes] = []
                while block := os.read(leaf, 1 << 20):
                    chunks.append(block)
                if os.fstat(leaf) != opened:
                    raise ValueError("evaluation artifact leaf changed while pinned")
                leaves[name] = b"".join(chunks)
            finally:
                os.close(leaf)
        after = os.fstat(descriptor)
        namespace = os.stat(root, follow_symlinks=False)
        if after != before or (namespace.st_dev, namespace.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("evaluation artifact root changed while pinned")
        return leaves
    finally:
        os.close(descriptor)


def _read_jsonl_antigravity(path: Path) -> tuple[object, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return tuple(json.loads(line) for line in lines)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid immutable JSONL leaf: {path.name}") from error


def _inspect_root_with_dependencies_antigravity(dependencies: object) -> FrozenEvaluationAntigravity:
    """Label-free semantic replay of a completed Exp447 root."""
    root = Path(getattr(dependencies, "root"))
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Exp447 artifact inventory is not exact")
    leaves = _pinned_evaluation_leaves_antigravity(root)
    hashes = _json_bytes_antigravity(leaves["artifact_sha256_antigravity.json"], "artifact_sha256_antigravity.json")
    if not isinstance(hashes, dict) or set(hashes) != set(_LEAVES_ANTIGRAVITY[:-1]):
        raise ValueError("artifact map is invalid or self-including")
    for name in _LEAVES_ANTIGRAVITY[:-1]:
        if type(hashes[name]) is not str or sha256(leaves[name]).hexdigest() != hashes[name]:
            raise ValueError("artifact leaf digest differs")
    identity = dependencies.capture_after()
    evaluation = _json_bytes_antigravity(leaves["evaluation_antigravity.json"], "evaluation_antigravity.json")
    confusion = _json_bytes_antigravity(leaves["confusion_antigravity.json"], "confusion_antigravity.json")
    decision = _json_bytes_antigravity(leaves["decision_antigravity.json"], "decision_antigravity.json")
    if not isinstance(evaluation, dict) or not isinstance(confusion, dict) or not isinstance(decision, dict):
        raise ValueError("evaluation JSON object is invalid")
    if evaluation.get("schema") != "exp447" or evaluation.get("exp446_prediction_sha256") != identity.prediction_sha256 or evaluation.get("evaluation_status") != "descriptive_complete_no_promotion":
        raise ValueError("evaluation identity or non-promotion contract differs")
    methods = confusion.get("methods")
    if not isinstance(methods, list) or tuple(item.get("comparison_name") for item in methods if isinstance(item, dict)) != _METHODS_ANTIGRAVITY:
        raise ValueError("frozen confusion method contract differs")
    summaries: dict[str, MetricSummaryAntigravity] = {}
    for item in methods:
        if not isinstance(item, dict) or set(item) != {"comparison_name", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "false_positive_rate", "macro_f1"}:
            raise ValueError("confusion payload schema differs")
        counts = ConfusionAntigravity(item["tp"], item["fp"], item["fn"], item["tn"])
        if any(item[name] != value for name, value in (("precision", counts.precision), ("recall", counts.recall), ("f1", counts.f1), ("false_positive_rate", counts.false_positive_rate))):
            raise ValueError("confusion-derived metric is inconsistent")
        summaries[item["comparison_name"]] = MetricSummaryAntigravity(item["comparison_name"], counts, item["macro_f1"])
    if summaries["hard_baseline"].confusion != ConfusionAntigravity(*FROZEN_CONFIG_ANTIGRAVITY.baseline_confusion):
        raise ValueError("replayed Hard baseline differs")
    if summaries["hard_plus_source_30"].confusion != ConfusionAntigravity(*FROZEN_CONFIG_ANTIGRAVITY.exp442_source_30_confusion):
        raise ValueError("replayed source-30 historical confusion differs")
    expected_governance = {
        "evaluation_status": "descriptive_complete_no_promotion", "scientific_success_criterion": None,
        "scientific_winner_selected": False, "review_only": True, "operational_alert_eligible": False,
        "same_experiment_reselection_allowed": False, "test_informed_reselection_allowed": False,
        "automatic_champion_promotion": False, "historical_source_30_confusion_agreement": True,
    }
    if any(decision.get(key) != value for key, value in expected_governance.items()):
        raise ValueError("decision semantic contract differs")
    historical_counts = {"tp": 1930, "fp": 318, "fn": 1988, "tn": 191664}
    if decision.get("historical_source_30_confusion") != historical_counts or decision.get("historical_exp442_evaluation_sha256") != FROZEN_CONFIG_ANTIGRAVITY.exp442_evaluation_sha256 or decision.get("historical_exp442_artifact_map_sha256") != FROZEN_CONFIG_ANTIGRAVITY.exp442_evaluation_artifact_map_sha256:
        raise ValueError("historical source-30 authority differs")
    metric_rows = _parse_jsonl_bytes_antigravity(leaves["dataset_metrics_antigravity.jsonl"], "dataset_metrics_antigravity.jsonl")
    if len(metric_rows) != FROZEN_CONFIG_ANTIGRAVITY.dataset_count:
        raise ValueError("dataset metric population differs")
    reconstructed: dict[str, list[DatasetConfusionRowAntigravity]] = {method: [] for method in _METHODS_ANTIGRAVITY}
    for index, (dataset, row) in enumerate(zip(identity.dataset_order, metric_rows, strict=True)):
        if not isinstance(row, dict) or row.get("dataset") != dataset or row.get("dataset_order") != index or set(row) != {"dataset", "dataset_order", *_METHODS_ANTIGRAVITY}:
            raise ValueError("dataset metric identity or order differs")
        for method in _METHODS_ANTIGRAVITY:
            payload = row[method]
            if not isinstance(payload, dict) or set(payload) != {"tp", "fp", "fn", "tn", "f1"}:
                raise ValueError("dataset confusion row schema differs")
            counts = ConfusionAntigravity(payload["tp"], payload["fp"], payload["fn"], payload["tn"])
            if payload["f1"] != counts.f1:
                raise ValueError("dataset F1 differs from its confusion")
            reconstructed[method].append(DatasetConfusionRowAntigravity(dataset, counts))
    for method in _METHODS_ANTIGRAVITY:
        replayed = _metric_summary_antigravity(method, reconstructed[method])
        if replayed != summaries[method]:
            raise ValueError("aggregate confusion or macro F1 differs from dataset rows")
    transition_rows = _parse_jsonl_bytes_antigravity(leaves["candidate_transitions_antigravity.jsonl"], "candidate_transitions_antigravity.jsonl")
    source_set = set(identity.source_coordinates)
    stable_set, unstable_set = set(identity.stable_coordinates), set(identity.unstable_coordinates)
    replayed_transitions: set[tuple[str, int]] = set()
    transition_outcomes: dict[str, dict[str, int]] = {lane: {"rescued": 0, "added_false_alarm": 0} for lane in ("source", "stable", "unstable")}
    for row in transition_rows:
        if not isinstance(row, dict) or set(row) != {"dataset", "dataset_order", "candidate_index", "lane", "already_hard", "transition"}:
            raise ValueError("candidate transition schema differs")
        coordinate = (row["dataset"], row["candidate_index"])
        if type(row["dataset_order"]) is not int or row["dataset_order"] < 0 or row["dataset_order"] >= len(identity.dataset_order) or identity.dataset_order[row["dataset_order"]] != row["dataset"] or coordinate not in source_set or coordinate in replayed_transitions:
            raise ValueError("candidate transition identity differs")
        expected_lane = "stable" if coordinate in stable_set else "unstable" if coordinate in unstable_set else None
        if row["lane"] != expected_lane or type(row["already_hard"]) is not bool or row["transition"] not in {"hard_overlap", "rescued", "added_false_alarm"}:
            raise ValueError("candidate transition lane or state differs")
        if row["already_hard"] != (row["transition"] == "hard_overlap"):
            raise ValueError("candidate transition overlap state differs")
        if row["transition"] in transition_outcomes[expected_lane]:
            transition_outcomes[expected_lane][row["transition"]] += 1
        replayed_transitions.add(coordinate)
    if replayed_transitions != source_set or len(stable_set) + len(unstable_set) != len(source_set):
        raise ValueError("candidate lane conservation differs")
    for lane, method in (("source", "hard_plus_source_30"), ("stable", "hard_plus_stable_26"), ("unstable", "hard_plus_unstable_4")):
        delta_tp = summaries[method].confusion.tp - summaries["hard_baseline"].confusion.tp
        delta_fp = summaries[method].confusion.fp - summaries["hard_baseline"].confusion.fp
        if (delta_tp, delta_fp) != (transition_outcomes[lane]["rescued"], transition_outcomes[lane]["added_false_alarm"]):
            raise ValueError("candidate transitions do not reconstruct method confusion deltas")
    diagnostics: dict[str, tuple[object, ...]] = {
        "rescued": _parse_jsonl_bytes_antigravity(leaves["rescued_datasets_antigravity.jsonl"], "rescued_datasets_antigravity.jsonl"),
        "false_alarm": _parse_jsonl_bytes_antigravity(leaves["false_alarm_datasets_antigravity.jsonl"], "false_alarm_datasets_antigravity.jsonl"),
        "regressed": _parse_jsonl_bytes_antigravity(leaves["regressed_datasets_antigravity.jsonl"], "regressed_datasets_antigravity.jsonl"),
        "strong_regression": _parse_jsonl_bytes_antigravity(leaves["strong_regressions_antigravity.jsonl"], "strong_regressions_antigravity.jsonl"),
    }
    expected_diagnostics: dict[str, list[dict[str, object]]] = defaultdict(list)
    for dataset, hard_row, stable_row in zip(identity.dataset_order, reconstructed["hard_baseline"], reconstructed[_PRIMARY_METHOD_ANTIGRAVITY], strict=True):
        if stable_row.confusion.tp > hard_row.confusion.tp:
            expected_diagnostics["rescued"].append({"dataset": dataset, "tp_increase": stable_row.confusion.tp - hard_row.confusion.tp})
        if stable_row.confusion.fp > hard_row.confusion.fp:
            expected_diagnostics["false_alarm"].append({"dataset": dataset, "fp_increase": stable_row.confusion.fp - hard_row.confusion.fp})
        delta = stable_row.confusion.f1 - hard_row.confusion.f1
        if delta < 0.0:
            expected_diagnostics["regressed"].append({"dataset": dataset, "f1_delta": delta})
            if hard_row.confusion.f1 >= 0.8:
                expected_diagnostics["strong_regression"].append({"dataset": dataset, "f1_delta": delta})
    if any(list(diagnostics[name]) != expected_diagnostics[name] for name in diagnostics):
        raise ValueError("diagnostic leaves differ from reconstructed metrics")
    bootstrap_payload = _json_bytes_antigravity(leaves["bootstrap_antigravity.json"], "bootstrap_antigravity.json")
    expected_comparisons = ("hard_plus_source_30_minus_hard_baseline", "hard_plus_stable_26_minus_hard_baseline", "hard_plus_stable_26_minus_hard_plus_source_30")
    if not isinstance(bootstrap_payload, dict) or not isinstance(bootstrap_payload.get("comparisons"), list) or tuple(item.get("comparator_name") for item in bootstrap_payload["comparisons"] if isinstance(item, dict)) != expected_comparisons:
        raise ValueError("bootstrap comparison contract differs")
    draws = bootstrap_draws_antigravity(FROZEN_CONFIG_ANTIGRAVITY.dataset_count)
    expected_bootstrap = (
        paired_bootstrap_antigravity(reconstructed["hard_baseline"], reconstructed["hard_plus_source_30"], expected_comparisons[0], draws=draws).to_payload_antigravity(),
        paired_bootstrap_antigravity(reconstructed["hard_baseline"], reconstructed["hard_plus_stable_26"], expected_comparisons[1], draws=draws).to_payload_antigravity(),
        paired_bootstrap_antigravity(reconstructed["hard_plus_source_30"], reconstructed["hard_plus_stable_26"], expected_comparisons[2], draws=draws).to_payload_antigravity(),
    )
    if tuple(bootstrap_payload["comparisons"]) != expected_bootstrap:
        raise ValueError("bootstrap deltas, intervals, or shared-draw SHA differ from reconstructed metrics")
    if _pinned_evaluation_leaves_antigravity(root) != leaves:
        raise ValueError("evaluation artifact namespace changed during semantic replay")
    return FrozenEvaluationAntigravity(evaluation["reviewed_commit"], identity, evaluation["dataset_count"], evaluation["label_rows"], summaries, {"source": 30, "stable": 26, "unstable": 4}, {name: len(rows) for name, rows in diagnostics.items()}, tuple(bootstrap_payload["comparisons"]))


def inspect_evaluation_antigravity() -> FrozenEvaluationAntigravity:
    """Replay a frozen completed evaluation without importing or opening labels."""
    return _inspect_root_with_dependencies_antigravity(_ProductionDependenciesAntigravity())


__all__ = ["FrozenEvaluationAntigravity", "evaluate_antigravity", "inspect_evaluation_antigravity"]
