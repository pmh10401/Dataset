"""Source-free immutable replay primitive for Exp448 inspection.

This module owns semantic replay only.  It imports neither evaluation nor source
authority, SQLite, or the closure-private TEST-label session.  Callers must supply
an already-captured immutable identity through a minimal ``root``/``capture_after``
dependency object.
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

from .artifacts_antigravity import canonical_json_bytes_antigravity
from .bootstrap_antigravity import bootstrap_draws_antigravity, paired_bootstrap_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    ConfusionAntigravity,
    DatasetConfusionRowAntigravity,
    MetricSummaryAntigravity,
    PublicationIdentityAntigravity,
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
_DIAGNOSTIC_NAMES_ANTIGRAVITY = ("rescued", "false_alarm", "regressed", "strong_regression")


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
            raise ValueError("Exp448 may not select or promote a winner")
        if tuple(self.diagnostic_counts) != _DIAGNOSTIC_NAMES_ANTIGRAVITY or any(
            type(value) is not int or value < 0 for value in self.diagnostic_counts.values()
        ):
            raise ValueError("diagnostic counts differ from the frozen contract")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "transition_counts", MappingProxyType(dict(self.transition_counts)))
        object.__setattr__(self, "diagnostic_counts", MappingProxyType(dict(self.diagnostic_counts)))
        object.__setattr__(self, "bootstrap", tuple(MappingProxyType(dict(row)) for row in self.bootstrap))


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


def _exact_json_value_antigravity(actual: object, expected: object) -> bool:
    """Compare parsed canonical JSON without Python's ``True == 1`` coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_value_antigravity(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_value_antigravity(item, value) for item, value in zip(actual, expected, strict=True)
        )
    return actual == expected


def _integrity_identity_antigravity(snapshot: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return mutation-relevant descriptor identity without read-observable atime."""
    return (
        snapshot.st_dev, snapshot.st_ino, snapshot.st_mode, snapshot.st_nlink,
        snapshot.st_size, snapshot.st_mtime_ns, snapshot.st_ctime_ns,
    )


def _pinned_evaluation_leaves_antigravity(root: Path) -> dict[str, bytes]:
    """Read exact regular leaves through one no-follow directory descriptor."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    leaf_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(root, flags)
    try:
        before = os.fstat(descriptor)
        before_identity = _integrity_identity_antigravity(before)
        if set(os.listdir(descriptor)) != set(_LEAVES_ANTIGRAVITY):
            raise ValueError("Exp448 artifact inventory is not exact")
        leaves: dict[str, bytes] = {}
        for name in _LEAVES_ANTIGRAVITY:
            leaf = os.open(name, leaf_flags, dir_fd=descriptor)
            try:
                opened = os.fstat(leaf)
                opened_identity = _integrity_identity_antigravity(opened)
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                    raise ValueError("evaluation artifact leaf is not a single-link regular file")
                chunks: list[bytes] = []
                while block := os.read(leaf, 1 << 20):
                    chunks.append(block)
                if _integrity_identity_antigravity(os.fstat(leaf)) != opened_identity:
                    raise ValueError("evaluation artifact leaf changed while pinned")
                leaves[name] = b"".join(chunks)
            finally:
                os.close(leaf)
        after = os.fstat(descriptor)
        namespace = os.stat(root, follow_symlinks=False)
        if _integrity_identity_antigravity(after) != before_identity or (namespace.st_dev, namespace.st_ino) != (before.st_dev, before.st_ino):
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


def inspect_replay_with_dependencies_antigravity(dependencies: object) -> FrozenEvaluationAntigravity:
    """Label-free semantic replay of a completed Exp448 root."""
    root = Path(getattr(dependencies, "root"))
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Exp448 artifact inventory is not exact")
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
    expected_evaluation = {
        "schema": "exp448", "experiment_id": "Exp448", "run_identity": FROZEN_CONFIG_ANTIGRAVITY.run_identity,
        "reviewed_commit": evaluation.get("reviewed_commit"), "dataset_count": FROZEN_CONFIG_ANTIGRAVITY.dataset_count,
        "label_rows": FROZEN_CONFIG_ANTIGRAVITY.label_rows,
        "evaluation_status": "descriptive_complete_no_promotion", "review_only": True,
        "operational_alert_eligible": False, "scientific_winner_selected": False,
        "same_experiment_reselection_allowed": False, "test_informed_reselection_allowed": False,
        "automatic_champion_promotion": False, "exp446_prediction_sha256": identity.prediction_sha256,
    }
    if type(expected_evaluation["reviewed_commit"]) is not str or not _exact_json_value_antigravity(evaluation, expected_evaluation):
        raise ValueError("evaluation identity or non-promotion contract differs")
    methods = confusion.get("methods")
    if not isinstance(methods, list) or tuple(item.get("comparison_name") for item in methods if isinstance(item, dict)) != _METHODS_ANTIGRAVITY:
        raise ValueError("frozen confusion method contract differs")
    summaries: dict[str, MetricSummaryAntigravity] = {}
    for item in methods:
        if not isinstance(item, dict) or set(item) != {"comparison_name", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "false_positive_rate", "macro_f1"}:
            raise ValueError("confusion payload schema differs")
        counts = ConfusionAntigravity(item["tp"], item["fp"], item["fn"], item["tn"])
        if any(type(item[name]) is not float or item[name] != value for name, value in (("precision", counts.precision), ("recall", counts.recall), ("f1", counts.f1), ("false_positive_rate", counts.false_positive_rate))) or type(item["macro_f1"]) is not float:
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
        "historical_source_30_confusion": {"tp": 1930, "fp": 318, "fn": 1988, "tn": 191664},
        "historical_exp442_evaluation_sha256": FROZEN_CONFIG_ANTIGRAVITY.exp442_evaluation_sha256,
        "historical_exp442_artifact_map_sha256": FROZEN_CONFIG_ANTIGRAVITY.exp442_evaluation_artifact_map_sha256,
    }
    if not _exact_json_value_antigravity(decision, expected_governance):
        raise ValueError("decision semantic contract differs")
    metric_rows = _parse_jsonl_bytes_antigravity(leaves["dataset_metrics_antigravity.jsonl"], "dataset_metrics_antigravity.jsonl")
    if len(metric_rows) != FROZEN_CONFIG_ANTIGRAVITY.dataset_count:
        raise ValueError("dataset metric population differs")
    reconstructed: dict[str, list[DatasetConfusionRowAntigravity]] = {method: [] for method in _METHODS_ANTIGRAVITY}
    for index, (dataset, row) in enumerate(zip(identity.dataset_order, metric_rows, strict=True)):
        if not isinstance(row, dict) or type(row.get("dataset")) is not str or row["dataset"] != dataset or type(row.get("dataset_order")) is not int or row["dataset_order"] != index or set(row) != {"dataset", "dataset_order", *_METHODS_ANTIGRAVITY}:
            raise ValueError("dataset metric identity or order differs")
        for method in _METHODS_ANTIGRAVITY:
            payload = row[method]
            if not isinstance(payload, dict) or set(payload) != {"tp", "fp", "fn", "tn", "f1"}:
                raise ValueError("dataset confusion row schema differs")
            counts = ConfusionAntigravity(payload["tp"], payload["fp"], payload["fn"], payload["tn"])
            if type(payload["f1"]) is not float or payload["f1"] != counts.f1:
                raise ValueError("dataset F1 differs from its confusion")
            reconstructed[method].append(DatasetConfusionRowAntigravity(dataset, counts))
    for method in _METHODS_ANTIGRAVITY:
        replayed = _metric_summary_antigravity(method, reconstructed[method])
        if replayed != summaries[method]:
            raise ValueError("aggregate confusion or macro F1 differs from dataset rows")
    transition_rows = _parse_jsonl_bytes_antigravity(leaves["candidate_transitions_antigravity.jsonl"], "candidate_transitions_antigravity.jsonl")
    source_set = set(identity.source_coordinates)
    stable_set, unstable_set = set(identity.stable_coordinates), set(identity.unstable_coordinates)
    replayed_transitions: list[tuple[str, int]] = []
    transition_outcomes: dict[str, dict[str, int]] = {lane: {"rescued": 0, "added_false_alarm": 0} for lane in ("source", "stable", "unstable")}
    dataset_lane_outcomes: dict[str, dict[str, dict[str, int]]] = {
        dataset: {lane: {"rescued": 0, "added_false_alarm": 0} for lane in ("source", "stable", "unstable")}
        for dataset in identity.dataset_order
    }
    for row in transition_rows:
        if not isinstance(row, dict) or set(row) != {"dataset", "dataset_order", "candidate_index", "lane", "already_hard", "transition"}:
            raise ValueError("candidate transition schema differs")
        coordinate = (row["dataset"], row["candidate_index"])
        if type(row["dataset"]) is not str or type(row["candidate_index"]) is not int or type(row["dataset_order"]) is not int or row["dataset_order"] < 0 or row["dataset_order"] >= len(identity.dataset_order) or identity.dataset_order[row["dataset_order"]] != row["dataset"] or coordinate not in source_set or coordinate in replayed_transitions:
            raise ValueError("candidate transition identity differs")
        expected_lane = "stable" if coordinate in stable_set else "unstable" if coordinate in unstable_set else None
        if row["lane"] != expected_lane or type(row["already_hard"]) is not bool or row["transition"] not in {"hard_overlap", "rescued", "added_false_alarm"}:
            raise ValueError("candidate transition lane or state differs")
        if row["already_hard"] != (row["transition"] == "hard_overlap"):
            raise ValueError("candidate transition overlap state differs")
        if row["transition"] in transition_outcomes[expected_lane]:
            # ``source`` is the disjoint union of stable and unstable.  Keep
            # its aggregate separately from the two comparator lanes: adding
            # source to stable and unstable would incorrectly double count
            # the frozen 30-coordinate population.
            transition_outcomes["source"][row["transition"]] += 1
            transition_outcomes[expected_lane][row["transition"]] += 1
            dataset_lane_outcomes[row["dataset"]]["source"][row["transition"]] += 1
            dataset_lane_outcomes[row["dataset"]][expected_lane][row["transition"]] += 1
        replayed_transitions.append(coordinate)
    canonical_coordinates = tuple(sorted(identity.source_coordinates, key=lambda coordinate: (identity.dataset_order.index(coordinate[0]), coordinate[1])))
    if tuple(replayed_transitions) != canonical_coordinates or set(replayed_transitions) != source_set or len(stable_set) + len(unstable_set) != len(source_set):
        raise ValueError("candidate lane conservation differs")
    for lane, method in (("source", "hard_plus_source_30"), ("stable", "hard_plus_stable_26"), ("unstable", "hard_plus_unstable_4")):
        delta_tp = summaries[method].confusion.tp - summaries["hard_baseline"].confusion.tp
        delta_fp = summaries[method].confusion.fp - summaries["hard_baseline"].confusion.fp
        if (delta_tp, delta_fp) != (transition_outcomes[lane]["rescued"], transition_outcomes[lane]["added_false_alarm"]):
            raise ValueError("candidate transitions do not reconstruct method confusion deltas")
    for index, dataset in enumerate(identity.dataset_order):
        for lane, method in (("source", "hard_plus_source_30"), ("stable", "hard_plus_stable_26"), ("unstable", "hard_plus_unstable_4")):
            delta_tp = reconstructed[method][index].confusion.tp - reconstructed["hard_baseline"][index].confusion.tp
            delta_fp = reconstructed[method][index].confusion.fp - reconstructed["hard_baseline"][index].confusion.fp
            outcomes = dataset_lane_outcomes[dataset][lane]
            if (delta_tp, delta_fp) != (outcomes["rescued"], outcomes["added_false_alarm"]):
                raise ValueError("candidate transitions do not reconstruct per-dataset method deltas")
    diagnostics: dict[str, tuple[object, ...]] = {
        "rescued": _parse_jsonl_bytes_antigravity(leaves["rescued_datasets_antigravity.jsonl"], "rescued_datasets_antigravity.jsonl"),
        "false_alarm": _parse_jsonl_bytes_antigravity(leaves["false_alarm_datasets_antigravity.jsonl"], "false_alarm_datasets_antigravity.jsonl"),
        "regressed": _parse_jsonl_bytes_antigravity(leaves["regressed_datasets_antigravity.jsonl"], "regressed_datasets_antigravity.jsonl"),
        "strong_regression": _parse_jsonl_bytes_antigravity(leaves["strong_regressions_antigravity.jsonl"], "strong_regressions_antigravity.jsonl"),
    }
    expected_diagnostics: dict[str, list[dict[str, object]]] = {name: [] for name in _DIAGNOSTIC_NAMES_ANTIGRAVITY}
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
    if any(not _exact_json_value_antigravity(list(diagnostics[name]), expected_diagnostics[name]) for name in diagnostics):
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
    if not _exact_json_value_antigravity(bootstrap_payload["comparisons"], list(expected_bootstrap)):
        raise ValueError("bootstrap deltas, intervals, or shared-draw SHA differ from reconstructed metrics")
    if _pinned_evaluation_leaves_antigravity(root) != leaves:
        raise ValueError("evaluation artifact namespace changed during semantic replay")
    return FrozenEvaluationAntigravity(evaluation["reviewed_commit"], identity, evaluation["dataset_count"], evaluation["label_rows"], summaries, {"source": 30, "stable": 26, "unstable": 4}, {name: len(rows) for name, rows in diagnostics.items()}, tuple(bootstrap_payload["comparisons"]))



__all__ = [
    "FrozenEvaluationAntigravity",
    "inspect_replay_with_dependencies_antigravity",
]
