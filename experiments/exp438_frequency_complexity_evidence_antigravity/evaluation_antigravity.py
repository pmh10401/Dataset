"""Post-freeze, channel-separate retrospective evaluation for Exp438."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import CHANNELS_ANTIGRAVITY
from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.inputs import verified_read_only_connection
from .reporting_antigravity import (
    PREDICTION_FILE_ANTIGRAVITY,
    _exclusive_rename_antigravity,
    _fsync_directory_antigravity,
    _write_new_antigravity,
    inspect_publication_antigravity,
)


EVALUATION_FILE_ANTIGRAVITY = "evaluation_antigravity.json"
DATASET_METRICS_FILE_ANTIGRAVITY = "dataset_metrics_antigravity.jsonl"
RESCUED_FILE_ANTIGRAVITY = "rescued_datasets_antigravity.jsonl"
REGRESSED_FILE_ANTIGRAVITY = "regressed_datasets_antigravity.jsonl"
CONFUSION_FILE_ANTIGRAVITY = "confusion_antigravity.json"
BOOTSTRAP_FILE_ANTIGRAVITY = "bootstrap_antigravity.json"
DECISION_FILE_ANTIGRAVITY = "decision_antigravity.json"
EVALUATION_HASH_FILE_ANTIGRAVITY = "artifact_sha256_antigravity.json"
EVALUATION_FILES_ANTIGRAVITY = frozenset({
    EVALUATION_FILE_ANTIGRAVITY,
    DATASET_METRICS_FILE_ANTIGRAVITY,
    RESCUED_FILE_ANTIGRAVITY,
    REGRESSED_FILE_ANTIGRAVITY,
    CONFUSION_FILE_ANTIGRAVITY,
    BOOTSTRAP_FILE_ANTIGRAVITY,
    DECISION_FILE_ANTIGRAVITY,
    EVALUATION_HASH_FILE_ANTIGRAVITY,
})
EXP437_SCORE_CHANNELS_ANTIGRAVITY = ("prototype", "gradient", "acf")


@dataclass(frozen=True)
class ConfusionAntigravity:
    tp: int
    fp: int
    fn: int
    tn: int

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in ("tp", "fp", "fn", "tn")):
            raise ValueError("confusion counts are invalid")

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 0.0 if denominator == 0 else 2.0 * self.tp / denominator

    def to_payload(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}

    def metric_payload(self) -> dict[str, float]:
        precision_denominator = self.tp + self.fp
        recall_denominator = self.tp + self.fn
        negative_denominator = self.tn + self.fp
        return {
            "precision": 0.0 if precision_denominator == 0 else self.tp / precision_denominator,
            "recall": 0.0 if recall_denominator == 0 else self.tp / recall_denominator,
            "specificity": 0.0 if negative_denominator == 0 else self.tn / negative_denominator,
            "fpr": 0.0 if negative_denominator == 0 else self.fp / negative_denominator,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class IntervalAntigravity:
    lower: float
    upper: float


@dataclass(frozen=True)
class BootstrapAntigravity:
    samples: int
    seed: int
    micro_f1_delta: IntervalAntigravity
    macro_f1_delta: IntervalAntigravity


@dataclass(frozen=True)
class DatasetEvaluationAntigravity:
    dataset: str
    baseline_confusion: ConfusionAntigravity
    combined_confusion: ConfusionAntigravity
    outside_lane_candidate_count: int
    outside_lane_candidate_true_positives: int
    rescued: bool
    missed: bool
    regressed: bool

    def to_payload(self, channel: str) -> dict[str, object]:
        return {
            "channel": channel,
            "dataset": self.dataset,
            "baseline_confusion": self.baseline_confusion.to_payload(),
            "combined_confusion": self.combined_confusion.to_payload(),
            "baseline_metrics": self.baseline_confusion.metric_payload(),
            "combined_metrics": self.combined_confusion.metric_payload(),
            "outside_lane_candidate_count": self.outside_lane_candidate_count,
            "outside_lane_candidate_true_positives": self.outside_lane_candidate_true_positives,
            "micro_f1_delta": self.combined_confusion.f1 - self.baseline_confusion.f1,
            "rescued": self.rescued,
            "missed": self.missed,
            "regressed": self.regressed,
        }


@dataclass(frozen=True)
class ChannelEvaluationAntigravity:
    channel: str
    confusion: ConfusionAntigravity
    baseline_confusion: ConfusionAntigravity
    rescued_datasets: tuple[str, ...]
    missed_datasets: tuple[str, ...]
    regressed_datasets: tuple[str, ...]
    dataset_rows: tuple[DatasetEvaluationAntigravity, ...]
    bootstrap: BootstrapAntigravity
    full_candidate_count: int
    outside_lane_candidate_count: int
    added_false_positives: int
    micro_f1_delta: float
    macro_f1_delta: float
    candidate_true_positives: int
    candidate_precision: float


@dataclass(frozen=True)
class Exp438EvaluationAntigravity:
    channels: Mapping[str, ChannelEvaluationAntigravity]
    train_oof_spearman: Mapping[str, Mapping[str, float]] | None
    train_oof_aligned_counts: Mapping[str, Mapping[str, int]] | None
    direct_improvement_passed: bool
    integration_evidence_passed: bool
    candidate_overlap_with_exp437: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        values = dict(self.channels)
        if tuple(values) != CHANNELS_ANTIGRAVITY:
            raise ValueError("evaluation channels are not separate and ordered")
        object.__setattr__(self, "channels", MappingProxyType(values))
        if self.train_oof_spearman is not None:
            object.__setattr__(self, "train_oof_spearman", MappingProxyType({
                channel: MappingProxyType(dict(references))
                for channel, references in self.train_oof_spearman.items()
            }))
        if self.train_oof_aligned_counts is not None:
            object.__setattr__(self, "train_oof_aligned_counts", MappingProxyType({
                channel: MappingProxyType(dict(references))
                for channel, references in self.train_oof_aligned_counts.items()
            }))
        if self.candidate_overlap_with_exp437 is not None:
            object.__setattr__(self, "candidate_overlap_with_exp437", MappingProxyType(dict(self.candidate_overlap_with_exp437)))


def _counts_antigravity(selected, labels: Mapping[int, int]) -> ConfusionAntigravity:
    chosen = set(selected)
    if any(type(index) is not int or index not in labels for index in chosen):
        raise ValueError("prediction coordinate is outside TEST identity")
    return ConfusionAntigravity(
        sum(index in chosen and value == 1 for index, value in labels.items()),
        sum(index in chosen and value == 0 for index, value in labels.items()),
        sum(index not in chosen and value == 1 for index, value in labels.items()),
        sum(index not in chosen and value == 0 for index, value in labels.items()),
    )


def _sum_counts_antigravity(values) -> ConfusionAntigravity:
    rows = tuple(values)
    return ConfusionAntigravity(*(sum(getattr(row, name) for row in rows) for name in ("tp", "fp", "fn", "tn")))


def _bootstrap_antigravity(rows, samples: int, seed: int) -> BootstrapAntigravity:
    if type(samples) is not int or samples < 1 or type(seed) is not int or seed < 0:
        raise ValueError("bootstrap configuration is invalid")
    values = tuple(rows)
    if not values:
        raise ValueError("bootstrap population is empty")
    generator = np.random.default_rng(seed)
    micro = np.empty(samples, dtype=np.float64)
    macro = np.empty(samples, dtype=np.float64)
    baseline_counts = np.asarray(
        [[row[0].tp, row[0].fp, row[0].fn, row[0].tn] for row in values],
        dtype=np.int64,
    )
    current_counts = np.asarray(
        [[row[1].tp, row[1].fp, row[1].fn, row[1].tn] for row in values],
        dtype=np.int64,
    )
    per_dataset_delta = np.asarray(
        [row[1].f1 - row[0].f1 for row in values], dtype=np.float64
    )
    # Chunking preserves the exact RNG draw order while avoiding a large
    # 20,000 x 1,117 allocation in the production evaluation.
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        indices = generator.integers(
            0, len(values), size=(stop - start, len(values))
        )
        baseline = baseline_counts[indices].sum(axis=1)
        current = current_counts[indices].sum(axis=1)
        baseline_denominator = 2 * baseline[:, 0] + baseline[:, 1] + baseline[:, 2]
        current_denominator = 2 * current[:, 0] + current[:, 1] + current[:, 2]
        baseline_f1 = np.divide(
            2.0 * baseline[:, 0], baseline_denominator,
            out=np.zeros(stop - start, dtype=np.float64),
            where=baseline_denominator != 0,
        )
        current_f1 = np.divide(
            2.0 * current[:, 0], current_denominator,
            out=np.zeros(stop - start, dtype=np.float64),
            where=current_denominator != 0,
        )
        micro[start:stop] = current_f1 - baseline_f1
        macro[start:stop] = per_dataset_delta[indices].mean(axis=1)
    return BootstrapAntigravity(
        samples,
        seed,
        IntervalAntigravity(float(np.quantile(micro, 0.025)), float(np.quantile(micro, 0.975))),
        IntervalAntigravity(float(np.quantile(macro, 0.025)), float(np.quantile(macro, 0.975))),
    )


def _evaluate_frozen_coordinates_antigravity(
    channels: Mapping[str, Mapping[str, tuple[int, ...]]],
    baseline: Mapping[str, tuple[int, ...]],
    labels: Mapping[str, Mapping[int, int]],
    *,
    bootstrap_samples: int = FROZEN_CONFIG.bootstrap_samples,
    bootstrap_seed: int = FROZEN_CONFIG.bootstrap_seed,
    train_oof_spearman: Mapping[str, Mapping[str, float]] | None = None,
    train_oof_aligned_counts: Mapping[str, Mapping[str, int]] | None = None,
    candidate_overlap_with_exp437: Mapping[str, float] | None = None,
    full_candidate_counts: Mapping[str, int] | None = None,
) -> Exp438EvaluationAntigravity:
    if tuple(channels) != CHANNELS_ANTIGRAVITY or set(baseline) != set(labels) or any(set(channels[name]) != set(labels) for name in CHANNELS_ANTIGRAVITY):
        raise ValueError("evaluation population is inconsistent")
    if full_candidate_counts is not None and set(full_candidate_counts) != set(CHANNELS_ANTIGRAVITY):
        raise ValueError("full candidate channel population is inconsistent")
    evaluations = {}
    for channel in CHANNELS_ANTIGRAVITY:
        rows = []
        dataset_rows = []
        rescued = []
        missed = []
        regressed = []
        candidate_count = 0
        candidate_true_positives = 0
        for dataset in sorted(labels):
            candidate_indices = tuple(channels[channel][dataset])
            if candidate_indices != tuple(sorted(set(candidate_indices))):
                raise ValueError("outside-lane candidates must be sorted and unique")
            if set(candidate_indices) & set(baseline[dataset]):
                raise ValueError("outside-lane candidate overlaps frozen Hard")
            if any(index not in labels[dataset] for index in candidate_indices):
                raise ValueError("outside-lane candidate is outside TEST identity")
            base = _counts_antigravity(baseline[dataset], labels[dataset])
            dataset_candidate_true_positives = sum(
                labels[dataset][index] == 1 for index in candidate_indices
            )
            candidate_count += len(candidate_indices)
            candidate_true_positives += dataset_candidate_true_positives
            combined_indices = tuple(sorted(set(baseline[dataset]) | set(candidate_indices)))
            current = _counts_antigravity(combined_indices, labels[dataset])
            rows.append((base, current))
            is_rescued = current.tp > base.tp
            is_missed = current.fn > 0
            is_regressed = current.f1 + 1e-15 < base.f1
            dataset_rows.append(DatasetEvaluationAntigravity(
                dataset=dataset,
                baseline_confusion=base,
                combined_confusion=current,
                outside_lane_candidate_count=len(candidate_indices),
                outside_lane_candidate_true_positives=dataset_candidate_true_positives,
                rescued=is_rescued,
                missed=is_missed,
                regressed=is_regressed,
            ))
            if is_rescued:
                rescued.append(dataset)
            if is_missed:
                missed.append(dataset)
            if is_regressed:
                regressed.append(dataset)
        aggregate_current = _sum_counts_antigravity(row[1] for row in rows)
        aggregate_baseline = _sum_counts_antigravity(row[0] for row in rows)
        full_count = candidate_count if full_candidate_counts is None else full_candidate_counts.get(channel)
        if type(full_count) is not int or full_count < candidate_count:
            raise ValueError("full candidate count is invalid")
        evaluations[channel] = ChannelEvaluationAntigravity(
            channel=channel,
            confusion=aggregate_current,
            baseline_confusion=aggregate_baseline,
            rescued_datasets=tuple(rescued),
            missed_datasets=tuple(missed),
            regressed_datasets=tuple(regressed),
            dataset_rows=tuple(dataset_rows),
            bootstrap=_bootstrap_antigravity(rows, bootstrap_samples, bootstrap_seed),
            full_candidate_count=full_count,
            outside_lane_candidate_count=candidate_count,
            added_false_positives=aggregate_current.fp - aggregate_baseline.fp,
            micro_f1_delta=aggregate_current.f1 - aggregate_baseline.f1,
            macro_f1_delta=float(np.mean([row[1].f1 - row[0].f1 for row in rows])),
            candidate_true_positives=candidate_true_positives,
            candidate_precision=0.0 if candidate_count == 0 else candidate_true_positives / candidate_count,
        )
    direct = any(
        value.outside_lane_candidate_count <= 100
        and len(value.rescued_datasets) >= 10
        and value.added_false_positives <= 30
        and value.micro_f1_delta > 0.0
        and value.macro_f1_delta > 0.0
        and value.bootstrap.micro_f1_delta.lower > 0.0
        and value.bootstrap.macro_f1_delta.lower > 0.0
        and bool(value.rescued_datasets)
        for value in evaluations.values()
    )
    def independent(channel: str) -> bool:
        if train_oof_spearman is None or train_oof_aligned_counts is None:
            return False
        if set(train_oof_spearman) != set(CHANNELS_ANTIGRAVITY) or set(train_oof_aligned_counts) != set(CHANNELS_ANTIGRAVITY):
            return False
        correlations = train_oof_spearman.get(channel, {})
        aligned = train_oof_aligned_counts.get(channel, {})
        return (
            set(correlations) == set(EXP437_SCORE_CHANNELS_ANTIGRAVITY)
            and set(aligned) == set(EXP437_SCORE_CHANNELS_ANTIGRAVITY)
            and all(
                math.isfinite(correlations[reference])
                and abs(correlations[reference]) < 0.90
                and type(aligned[reference]) is int
                and aligned[reference] >= 20
                for reference in EXP437_SCORE_CHANNELS_ANTIGRAVITY
            )
        )
    return Exp438EvaluationAntigravity(
        evaluations,
        train_oof_spearman,
        train_oof_aligned_counts,
        direct,
        bool(any(
            independent(channel)
            and value.candidate_true_positives >= 25
            and len(value.rescued_datasets) >= 10
            and value.candidate_precision >= 0.35
            for channel, value in evaluations.items()
        )),
        candidate_overlap_with_exp437,
    )


def evaluate_after_gates_antigravity(
    *,
    gates: tuple[Callable[[], object], ...],
    coordinate_loader: Callable[[], tuple[Mapping[str, Mapping[str, tuple[int, ...]]], Mapping[str, tuple[int, ...]]]],
    label_loader: Callable[[], Mapping[str, Mapping[int, int]]],
    bootstrap_samples: int = FROZEN_CONFIG.bootstrap_samples,
) -> Exp438EvaluationAntigravity:
    for gate in gates:
        gate()
    channels, baseline = coordinate_loader()
    labels = label_loader()
    return _evaluate_frozen_coordinates_antigravity(
        channels,
        baseline,
        labels,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=FROZEN_CONFIG.bootstrap_seed,
    )


def _snapshot_tree_antigravity(paths: tuple[Path, ...]) -> dict[str, str]:
    snapshot = {}
    for root in paths:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                snapshot[str(path)] = sha256(path.read_bytes()).hexdigest()
    return snapshot


def _load_jsonl_antigravity(path: Path):
    return tuple(json.loads(line) for line in path.read_text().splitlines() if line)


def _production_coordinates_antigravity(publication, datasets, exp437_publication):
    channels = {name: {dataset: [] for dataset in datasets} for name in CHANNELS_ANTIGRAVITY}
    full_candidate_counts = {name: 0 for name in CHANNELS_ANTIGRAVITY}
    for row in _load_jsonl_antigravity(publication / PREDICTION_FILE_ANTIGRAVITY):
        if row["dataset"] not in channels[CHANNELS_ANTIGRAVITY[0]]:
            raise ValueError("Exp438 prediction contains a foreign dataset")
        for channel in CHANNELS_ANTIGRAVITY:
            evidence = row["channel_evidence"][channel]
            if evidence["available"] and evidence["abnormal"]:
                full_candidate_counts[channel] += 1
                if row["outside_all_frozen_lanes"]:
                    channels[channel][row["dataset"]].append(int(row["instance_index"]))
    exp437_candidates = {dataset: [] for dataset in datasets}
    for row in _load_jsonl_antigravity(exp437_publication / "prediction_antigravity.jsonl"):
        dataset = row.get("dataset")
        if dataset not in exp437_candidates:
            raise ValueError("Exp437 comparator contains a foreign dataset")
        exp437_candidates[dataset].append(int(row["instance_index"]))
    return (
        {channel: {dataset: tuple(indices) for dataset, indices in mapping.items()} for channel, mapping in channels.items()},
        {dataset: tuple(indices) for dataset, indices in exp437_candidates.items()},
        full_candidate_counts,
    )


def _load_labels_antigravity(datasets: tuple[str, ...]) -> dict[str, dict[int, int]]:
    labels = {}
    with verified_read_only_connection(
        Path("/Users/mac/Documents/Dataset/univariate_ts.db"),
        FROZEN_CONFIG.database_sha256,
    ) as connection:
        for dataset in datasets:
            rows = connection.execute(
                "SELECT i.instance_index,i.label FROM instances i JOIN datasets d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index",
                (dataset,),
            ).fetchall()
            if not rows:
                raise ValueError(f"TEST labels are absent: {dataset}")
            values = {}
            for index, raw in rows:
                if type(raw) is int and raw in {0, 1}:
                    value = raw
                elif isinstance(raw, str) and raw in {"0", "1"}:
                    value = int(raw)
                else:
                    raise ValueError(f"TEST label is not binary: {dataset}")
                values[int(index)] = value
            labels[dataset] = values
    return labels


def _evaluation_payload_antigravity(result: Exp438EvaluationAntigravity) -> dict[str, object]:
    channels = {}
    for name, value in result.channels.items():
        channels[name] = {
            "confusion": value.confusion.to_payload(),
            "metrics": value.confusion.metric_payload(),
            "baseline_confusion": value.baseline_confusion.to_payload(),
            "baseline_metrics": value.baseline_confusion.metric_payload(),
            "rescued_datasets": list(value.rescued_datasets),
            "missed_datasets": list(value.missed_datasets),
            "regressed_datasets": list(value.regressed_datasets),
            "full_candidate_count": value.full_candidate_count,
            "outside_lane_candidate_count": value.outside_lane_candidate_count,
            "added_false_positives": value.added_false_positives,
            "micro_f1_delta": value.micro_f1_delta,
            "macro_f1_delta": value.macro_f1_delta,
            "candidate_true_positives": value.candidate_true_positives,
            "candidate_precision": value.candidate_precision,
            "bootstrap": {
                "samples": value.bootstrap.samples,
                "seed": value.bootstrap.seed,
                "micro_f1_delta": vars(value.bootstrap.micro_f1_delta),
                "macro_f1_delta": vars(value.bootstrap.macro_f1_delta),
            },
        }
    train_oof_spearman = None if result.train_oof_spearman is None else {
        channel: dict(references)
        for channel, references in result.train_oof_spearman.items()
    }
    train_oof_aligned_counts = None if result.train_oof_aligned_counts is None else {
        channel: dict(references)
        for channel, references in result.train_oof_aligned_counts.items()
    }
    return {
        "schema": "exp438.evaluation.v1",
        "channels": channels,
        "train_oof_spearman": train_oof_spearman,
        "train_oof_aligned_counts": train_oof_aligned_counts,
        "direct_improvement_passed": result.direct_improvement_passed,
        "integration_evidence_passed": result.integration_evidence_passed,
        "candidate_overlap_with_exp437": None if result.candidate_overlap_with_exp437 is None else dict(result.candidate_overlap_with_exp437),
    }


def _json_antigravity(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _jsonl_antigravity(rows) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _evaluation_artifacts_antigravity(
    result: Exp438EvaluationAntigravity,
    *,
    prediction_sha256: str,
) -> dict[str, bytes]:
    if len(prediction_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in prediction_sha256
    ):
        raise ValueError("prediction SHA256 is invalid")
    evaluation = _evaluation_payload_antigravity(result)
    dataset_rows = [
        row.to_payload(channel)
        for channel in CHANNELS_ANTIGRAVITY
        for row in result.channels[channel].dataset_rows
    ]
    rescued_rows = [
        row for row in dataset_rows if row["rescued"]
    ]
    regressed_rows = [
        row for row in dataset_rows if row["regressed"]
    ]
    confusion = {
        "schema": "exp438.confusion.v1",
        "channels": {
            channel: {
                "baseline": result.channels[channel].baseline_confusion.to_payload(),
                "counterfactual_union": result.channels[channel].confusion.to_payload(),
                "baseline_metrics": result.channels[channel].baseline_confusion.metric_payload(),
                "counterfactual_union_metrics": result.channels[channel].confusion.metric_payload(),
            }
            for channel in CHANNELS_ANTIGRAVITY
        },
    }
    bootstrap = {
        "schema": "exp438.bootstrap.v1",
        "channels": {
            channel: {
                "samples": result.channels[channel].bootstrap.samples,
                "seed": result.channels[channel].bootstrap.seed,
                "micro_f1_delta": vars(result.channels[channel].bootstrap.micro_f1_delta),
                "macro_f1_delta": vars(result.channels[channel].bootstrap.macro_f1_delta),
            }
            for channel in CHANNELS_ANTIGRAVITY
        },
    }
    decision = {
        "schema": "exp438.decision.v1",
        "review_only": True,
        "operational_alert_eligible": False,
        "winner_selected": False,
        "direct_improvement_passed": result.direct_improvement_passed,
        "integration_evidence_passed": result.integration_evidence_passed,
        "direct_criteria": {
            "outside_lane_candidate_count_max": 100,
            "rescued_dataset_count_min": 10,
            "added_false_positives_max": 30,
            "micro_f1_delta_positive": True,
            "macro_f1_delta_positive": True,
            "micro_bootstrap_lower_positive": True,
            "macro_bootstrap_lower_positive": True,
        },
        "integration_criteria": {
            "outside_lane_true_positives_min": 25,
            "rescued_dataset_count_min": 10,
            "outside_lane_precision_min": 0.35,
            "absolute_train_oof_spearman_max_exclusive": 0.90,
            "minimum_aligned_train_oof_scores": 20,
            "unavailable_correlation_can_pass": False,
        },
        "train_oof_spearman": evaluation["train_oof_spearman"],
        "train_oof_aligned_counts": evaluation["train_oof_aligned_counts"],
        "candidate_overlap_with_exp437": evaluation["candidate_overlap_with_exp437"],
    }
    payloads = {
        EVALUATION_FILE_ANTIGRAVITY: _json_antigravity(evaluation),
        DATASET_METRICS_FILE_ANTIGRAVITY: _jsonl_antigravity(dataset_rows),
        RESCUED_FILE_ANTIGRAVITY: _jsonl_antigravity(rescued_rows),
        REGRESSED_FILE_ANTIGRAVITY: _jsonl_antigravity(regressed_rows),
        CONFUSION_FILE_ANTIGRAVITY: _json_antigravity(confusion),
        BOOTSTRAP_FILE_ANTIGRAVITY: _json_antigravity(bootstrap),
        DECISION_FILE_ANTIGRAVITY: _json_antigravity(decision),
    }
    payloads[EVALUATION_HASH_FILE_ANTIGRAVITY] = _json_antigravity({
        "schema": "exp438.evaluation-hashes.v1",
        "artifact_sha256": {
            name: sha256(raw).hexdigest() for name, raw in sorted(payloads.items())
        },
        "prediction_sha256": prediction_sha256,
        "self_excluded": EVALUATION_HASH_FILE_ANTIGRAVITY,
    })
    return payloads


def _verify_evaluation_artifacts_antigravity(
    root: Path,
    *,
    prediction_sha256: str,
) -> dict[str, object]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("evaluation root is invalid")
    inventory = {path.name for path in root.iterdir()}
    if inventory != EVALUATION_FILES_ANTIGRAVITY or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise ValueError("evaluation artifact inventory is invalid")
    try:
        hashes = json.loads((root / EVALUATION_HASH_FILE_ANTIGRAVITY).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation hash artifact is invalid") from error
    expected = EVALUATION_FILES_ANTIGRAVITY - {EVALUATION_HASH_FILE_ANTIGRAVITY}
    if (
        hashes.get("schema") != "exp438.evaluation-hashes.v1"
        or hashes.get("prediction_sha256") != prediction_sha256
        or hashes.get("self_excluded") != EVALUATION_HASH_FILE_ANTIGRAVITY
        or set(hashes.get("artifact_sha256", {})) != expected
        or any(
            sha256((root / name).read_bytes()).hexdigest() != digest
            for name, digest in hashes.get("artifact_sha256", {}).items()
        )
    ):
        raise ValueError("evaluation artifact replay failed")
    try:
        payload = json.loads((root / EVALUATION_FILE_ANTIGRAVITY).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation payload is invalid") from error
    if payload.get("schema") != "exp438.evaluation.v1":
        raise ValueError("evaluation payload schema is invalid")
    return payload


def _write_evaluation_artifacts_antigravity(
    root: Path,
    result: Exp438EvaluationAntigravity,
    *,
    prediction_sha256: str,
) -> dict[str, object]:
    root = Path(root)
    if root.exists() or root.is_symlink():
        raise ValueError("Exp438 evaluation root already exists")
    if root.parent.is_symlink() or not root.parent.is_dir():
        raise ValueError("Exp438 evaluation parent is invalid")
    payloads = _evaluation_artifacts_antigravity(
        result, prediction_sha256=prediction_sha256
    )
    staging = root.with_name(
        f".{root.name}_{uuid4().hex}_staging_antigravity"
    )
    staging.mkdir(mode=0o700)
    published = False
    try:
        for name in sorted(payloads):
            _write_new_antigravity(staging / name, payloads[name])
        _fsync_directory_antigravity(staging)
        try:
            _exclusive_rename_antigravity(staging, root)
        except FileExistsError as error:
            raise ValueError("Exp438 evaluation root already exists") from error
        _fsync_directory_antigravity(root.parent)
        published = True
        return _verify_evaluation_artifacts_antigravity(
            root, prediction_sha256=prediction_sha256
        )
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)


def evaluate_exp438_antigravity(
    *,
    publication_root: Path,
    output_root: Path,
    checkpoint_root: Path,
    evaluation_root: Path,
) -> Exp438EvaluationAntigravity:
    publication_root = Path(publication_root)
    output_root = Path(output_root)
    checkpoint_root = Path(checkpoint_root)
    evaluation_root = Path(evaluation_root)
    if evaluation_root.exists() or evaluation_root.is_symlink():
        raise ValueError("Exp438 evaluation root already exists")
    frozen = inspect_publication_antigravity(publication_root, output_root, checkpoint_root)
    datasets = tuple(json.loads((checkpoint_root / "context_antigravity.json").read_text())["datasets"])
    before = _snapshot_tree_antigravity((publication_root, output_root, checkpoint_root))

    from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import EXP437_PROFILE
    from experiments.exp436_missing_safe_strict_consensus_antigravity.reporting_antigravity import inspect_publication as inspect_exp437_publication

    exp437_output = Path("/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/output_antigravity")
    exp437_checkpoint = Path("/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/checkpoint_antigravity")
    exp437 = inspect_exp437_publication(exp437_output, exp437_checkpoint, profile=EXP437_PROFILE)
    channels, exp437_candidates, full_candidate_counts = _production_coordinates_antigravity(
        frozen.publication, datasets, exp437.publication
    )
    from experiments.exp436_missing_safe_strict_consensus_antigravity.evaluation_antigravity import _load_frozen_hard_coordinates
    from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import frozen_config_for_profile_antigravity

    baseline = _load_frozen_hard_coordinates(
        config=frozen_config_for_profile_antigravity(EXP437_PROFILE),
        profile=EXP437_PROFILE,
    )
    if set(baseline) != set(datasets):
        raise ValueError("Exp404 Hard comparator population differs from Exp438")
    overlaps = {}
    exp437_set = {(dataset, index) for dataset, indices in exp437_candidates.items() for index in indices}
    for channel, mapping in channels.items():
        current_set = {(dataset, index) for dataset, indices in mapping.items() for index in indices}
        union = current_set | exp437_set
        overlaps[channel] = 1.0 if not union else len(current_set & exp437_set) / len(union)
    # This is the first reachable TEST-label operation; every frozen source and
    # comparator gate above has already passed.
    labels = _load_labels_antigravity(datasets)
    result = _evaluate_frozen_coordinates_antigravity(
        channels,
        baseline,
        labels,
        train_oof_spearman=None,
        candidate_overlap_with_exp437=overlaps,
        full_candidate_counts=full_candidate_counts,
    )
    if _snapshot_tree_antigravity((publication_root, output_root, checkpoint_root)) != before:
        raise ValueError("frozen prediction bytes changed during evaluation")
    _write_evaluation_artifacts_antigravity(
        evaluation_root,
        result,
        prediction_sha256=frozen.prediction_sha256,
    )
    if _snapshot_tree_antigravity((publication_root, output_root, checkpoint_root)) != before:
        raise ValueError("frozen prediction bytes changed during evaluation publication")
    return result
