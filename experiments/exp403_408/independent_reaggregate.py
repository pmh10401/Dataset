"""Read-only, independent reconstruction of published Exp403-408 results.

This module deliberately does not import prediction, evaluation, consensus, or
pipeline helpers.  It consumes only files that have already been published by
the run and independently recalculates every reported aggregate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


FLOAT_TOLERANCE = 1e-12
PREDICTION_MANIFEST = "exp404_prediction_manifest.json"
PREDICTION_ROWS = "exp404_hard_confidence.jsonl"
EXP405_EVALUATION = "exp405_demotion_evaluation.json"
EXP405_ROWS = "exp405_demotion_dataset_metrics.csv"
EXP405_CANDIDATE_ROWS = "exp405_demotion_candidate_evaluation.csv"
AUDIT_ROWS = "exp403_feature_audit.jsonl"
CONSENSUS_ROWS = "exp408_consensus_results.jsonl"
EXP408_ROWS = "exp408_variant_metrics.csv"
EXP408_CANDIDATE_ROWS = "exp408_variant_candidate_evaluation.csv"
FROZEN_LABELS = "exp403_408_frozen_test_labels.csv"
INPUT_MANIFEST = "exp403_408_evaluation_input_manifest.json"
PUBLICATION_MANIFEST = "exp403_408_evaluation_publication_manifest.json"

_EXP405_INTEGERS = (
    "recheck_count",
    "false_positive_count",
    "true_positive_count",
)
_EXP405_FLOATS = (
    "false_positive_fraction",
    "baseline_true_positive_retention",
    "train_stability",
    "dataset_concentration",
    "lineage_concentration",
)
_EXP405_INTERVALS = (
    "false_positive_fraction_wilson_interval",
    "false_positive_fraction_bootstrap_interval",
    "baseline_true_positive_retention_bootstrap_interval",
)
_EXP408_INTEGERS = (
    "candidate_count",
    "tp",
    "fp",
    "fn_recovered",
    "zero_f1_before",
    "zero_f1_after",
    "standard_overlap",
    "priority_overlap",
    "review_outside_existing",
)
_BOOTSTRAP_ITERATIONS = 2_000
_WILSON_Z_95 = 1.959963984540054
_BOOTSTRAP_SEED = 20260727


@dataclass(frozen=True)
class IndependentReaggregation:
    """Successful independent reconstruction and exact official comparisons."""

    metrics: Mapping[str, object]
    comparisons: Mapping[str, Mapping[str, object]]
    mismatch_count: int


def _verify_frozen_publications(output: Path) -> Mapping[str, Mapping[int, int]]:
    """Verify every consumed immutable publication before parsing its values."""
    input_manifest = _read_object(output / INPUT_MANIFEST)
    publication_manifest = _read_object(output / PUBLICATION_MANIFEST)
    input_hashes = _artifact_hashes(input_manifest, INPUT_MANIFEST)
    publication_hashes = _artifact_hashes(publication_manifest, PUBLICATION_MANIFEST)
    required = {
        PREDICTION_MANIFEST, PREDICTION_ROWS, EXP405_EVALUATION, EXP405_ROWS,
        EXP405_CANDIDATE_ROWS, AUDIT_ROWS, CONSENSUS_ROWS, EXP408_ROWS,
        EXP408_CANDIDATE_ROWS, FROZEN_LABELS,
    }
    observed = {**input_hashes, **publication_hashes}
    missing = required - set(observed)
    if missing:
        raise ValueError("frozen publication hashes are incomplete: " + ", ".join(sorted(missing)))
    for name, expected in observed.items():
        if name in {INPUT_MANIFEST, PUBLICATION_MANIFEST}:
            continue
        if _sha256(output / name) != expected:
            raise ValueError(f"frozen publication SHA256 mismatch: {name}")
    if publication_hashes.get(INPUT_MANIFEST) != _sha256(output / INPUT_MANIFEST):
        raise ValueError("frozen publication SHA256 mismatch: evaluation input manifest")
    return _load_frozen_labels(output / FROZEN_LABELS)


def _artifact_hashes(manifest: Mapping[str, object], name: str) -> dict[str, str]:
    values = _mapping(manifest.get("artifact_sha256"), f"{name} artifact_sha256")
    result: dict[str, str] = {}
    for artifact, digest in values.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} artifact SHA256 is invalid")
        result[str(artifact)] = digest
    return result


def _load_frozen_labels(path: Path) -> Mapping[str, Mapping[int, int]]:
    rows = _read_csv(path)
    _require_columns(rows, {"dataset", "candidate_index", "is_true_anomaly"}, path.name)
    result: dict[str, dict[int, int]] = defaultdict(dict)
    for row in rows:
        dataset = _text(row.get("dataset"), "frozen label dataset")
        index = _integer(row.get("candidate_index"), "frozen label index")
        value = _binary_integer(row.get("is_true_anomaly"), "frozen label value")
        if index in result[dataset]:
            raise ValueError("frozen labels contain duplicate indices")
        result[dataset][index] = value
    if not result:
        raise ValueError("frozen labels are empty")
    return MappingProxyType({dataset: MappingProxyType(values) for dataset, values in result.items()})


def independent_reaggregate(output_dir: Path) -> IndependentReaggregation:
    """Reaggregate frozen publications and reject any metric disagreement.

    Integer fields must match exactly. Floating-point fields may differ by at
    most ``1e-12``. No database, label loader, or official metric helper is
    consulted here.
    """
    output = Path(output_dir)
    labels = _verify_frozen_publications(output)
    manifest = _read_object(output / PREDICTION_MANIFEST)
    prediction = _load_prediction(output, manifest)
    audit = _load_audit(output / AUDIT_ROWS)
    exp405, exp405_comparisons = _reaggregate_exp405(
        prediction, audit, labels, _read_object(output / EXP405_EVALUATION), output / EXP405_ROWS,
        output / EXP405_CANDIDATE_ROWS,
    )
    exp408, exp408_comparisons = _reaggregate_exp408(
        prediction, audit, labels, output / CONSENSUS_ROWS, output / EXP408_ROWS,
        output / EXP408_CANDIDATE_ROWS,
    )
    metrics: dict[str, object] = {
        "exp404": _exp404_metrics(prediction, labels),
        "exp405": exp405,
        "exp406": exp408["exp406"],
        "exp407": exp408["exp407"],
        "exp408": exp408["variants"],
        "train_size_coverage": exp408["train_size_coverage"],
        "workload": exp408["workload"],
        "concentration": exp408["concentration"],
    }
    comparisons = {**exp405_comparisons, **exp408_comparisons}
    mismatches = [
        name
        for name, comparison in comparisons.items()
        if comparison["matched"] is not True
    ]
    if mismatches:
        raise ValueError("independent reaggregation mismatch: " + ", ".join(mismatches))
    return IndependentReaggregation(
        metrics=MappingProxyType(metrics),
        comparisons=MappingProxyType(comparisons),
        mismatch_count=0,
    )


def _load_prediction(output: Path, manifest: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    filename = manifest.get("hard_confidence_file", PREDICTION_ROWS)
    if not isinstance(filename, str) or not filename:
        raise ValueError("published prediction manifest hard_confidence_file is invalid")
    path = output / filename
    expected_hash = _mapping(manifest.get("artifact_hashes"), "prediction manifest artifact_hashes").get("prediction")
    if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
        raise ValueError("published prediction SHA256 mismatch")
    frozen_lanes = _mapping(manifest.get("frozen_hard_lanes"), "prediction manifest frozen_hard_lanes")
    expected = {
        (str(dataset), int(index))
        for dataset, raw_indices in frozen_lanes.items()
        for index in _indices(raw_indices, f"frozen Hard lane {dataset}")
    }
    rows = _read_jsonl(path)
    seen: set[tuple[str, int]] = set()
    for row in rows:
        dataset = _text(row.get("dataset"), "prediction dataset")
        index = _integer(row.get("candidate_index"), "prediction candidate index")
        confidence = _mapping(row.get("confidence_by_alpha"), "prediction confidence_by_alpha")
        if not confidence or not all(value in {"high", "medium", "recheck", "abstain"} for value in confidence.values()):
            raise ValueError("published prediction confidence grades are invalid")
        identity = (dataset, index)
        if identity in seen:
            raise ValueError("published prediction contains duplicate Hard candidate")
        seen.add(identity)
    if seen != expected:
        raise ValueError("published prediction frozen Hard lane parity mismatch")
    return tuple(rows)


def _exp404_metrics(prediction: Sequence[Mapping[str, object]], labels: Mapping[str, Mapping[int, int]]) -> dict[str, object]:
    grade_counts: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    recheck_count = 0
    for row in prediction:
        grades = _mapping(row["confidence_by_alpha"], "prediction confidence_by_alpha")
        for alpha, grade in grades.items():
            grade_counts[str(alpha)][str(grade)] += 1
        if all(grade == "recheck" for grade in grades.values()):
            recheck_count += 1
    return {
        "hard_candidate_count": len(prediction),
        "recheck_count": recheck_count,
        "hard_tp": sum(labels[str(row["dataset"])][_integer(row["candidate_index"], "prediction candidate index")] for row in prediction),
        "hard_fp": len(prediction) - sum(labels[str(row["dataset"])][_integer(row["candidate_index"], "prediction candidate index")] for row in prediction),
        "hard_fn": sum(sum(values.values()) for values in labels.values()) - sum(labels[str(row["dataset"])][_integer(row["candidate_index"], "prediction candidate index")] for row in prediction),
        "grade_counts": {alpha: dict(sorted(counts.items())) for alpha, counts in sorted(grade_counts.items())},
        "frozen_lane_parity": True,
    }


def _reaggregate_exp405(
    prediction: Sequence[Mapping[str, object]],
    audit: Mapping[str, int],
    labels: Mapping[str, Mapping[int, int]],
    official: Mapping[str, object],
    path: Path,
    candidate_path: Path,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    rows = _read_csv(path)
    required = {
        "dataset", "lineage", "train_stability", "baseline_hard_tp", "recheck_count",
        "recheck_false_positive_count", "recheck_true_positive_count",
    }
    _require_columns(rows, required, path.name)
    candidate_rows = _read_candidate_csv(candidate_path)
    _require_columns(candidate_rows, {"dataset", "candidate_index", "is_recheck", "is_true_anomaly"}, candidate_path.name)
    prediction_by_identity = {
        (str(row["dataset"]), _integer(row["candidate_index"], "prediction candidate index")): row
        for row in prediction
    }
    candidate_by_dataset: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in candidate_rows:
        identity = (_text(row.get("dataset"), "Exp405 candidate dataset"), _integer(row.get("candidate_index"), "Exp405 candidate index"))
        prediction_row = prediction_by_identity.get(identity)
        if prediction_row is None or identity in seen:
            raise ValueError("Exp405 candidate evaluation coverage is invalid")
        expected_recheck = int(all(grade == "recheck" for grade in _mapping(prediction_row["confidence_by_alpha"], "confidence").values()))
        if _binary_integer(row.get("is_recheck"), "Exp405 is_recheck") != expected_recheck:
            raise ValueError("Exp405 candidate evaluation recheck status mismatch")
        _binary_integer(row.get("is_true_anomaly"), "Exp405 is_true_anomaly")
        if labels.get(identity[0], {}).get(identity[1]) != _binary_integer(row.get("is_true_anomaly"), "Exp405 is_true_anomaly"):
            raise ValueError("Exp405 candidate evaluation differs from frozen labels")
        seen.add(identity)
        candidate_by_dataset[identity[0]].append(row)
    if seen != set(prediction_by_identity):
        raise ValueError("Exp405 candidate evaluation coverage is incomplete")
    recheck = fp = tp = baseline_tp = 0
    stability_weighted = 0.0
    dataset_counts: Counter[str] = Counter()
    lineage_counts: Counter[str] = Counter()
    bootstrap_rows: list[tuple[int, int, int, int]] = []
    for row in rows:
        dataset = _text(row.get("dataset"), "Exp405 dataset")
        if dataset not in audit:
            raise ValueError(f"Exp405 dataset is absent from frozen audit: {dataset}")
        candidates = candidate_by_dataset.get(dataset, [])
        row_recheck = sum(_binary_integer(candidate.get("is_recheck"), "Exp405 is_recheck") for candidate in candidates)
        row_tp = sum(
            _binary_integer(candidate.get("is_true_anomaly"), "Exp405 is_true_anomaly")
            for candidate in candidates
            if _binary_integer(candidate.get("is_recheck"), "Exp405 is_recheck")
        )
        row_fp = row_recheck - row_tp
        for name, value in {
            "recheck_count": row_recheck,
            "recheck_false_positive_count": row_fp,
            "recheck_true_positive_count": row_tp,
        }.items():
            if _integer(row.get(name), f"Exp405 {name}") != value:
                raise ValueError(f"Exp405 candidate evaluation mismatch: {dataset}/{name}")
        if row_fp + row_tp != row_recheck:
            raise ValueError(f"Exp405 candidate outcome total mismatch: {dataset}")
        row_stability = _float(row.get("train_stability"), "Exp405 train stability")
        recheck += row_recheck
        fp += row_fp
        tp += row_tp
        actual_baseline_tp = sum(_binary_integer(candidate.get("is_true_anomaly"), "Exp405 is_true_anomaly") for candidate in candidates)
        if _integer(row.get("baseline_hard_tp"), "Exp405 baseline Hard TP") != actual_baseline_tp:
            raise ValueError(f"Exp405 candidate evaluation mismatch: {dataset}/baseline_hard_tp")
        baseline_tp += actual_baseline_tp
        stability_weighted += row_stability * row_recheck
        if row_recheck:
            dataset_counts[dataset] += row_recheck
            lineage_counts[_text(row.get("lineage"), "Exp405 lineage")] += row_recheck
        bootstrap_rows.append((row_recheck, row_fp, row_tp, actual_baseline_tp))
    fp_fraction = fp / recheck if recheck else 0.0
    tp_retention = (baseline_tp - tp) / baseline_tp if baseline_tp else 1.0
    stability = stability_weighted / recheck if recheck else 0.0
    fp_bootstrap, retention_bootstrap = _paired_bootstrap(
        bootstrap_rows,
        point_fp_fraction=fp_fraction,
        point_tp_retention=tp_retention,
    )
    result: dict[str, object] = {
        "recheck_count": recheck,
        "false_positive_count": fp,
        "true_positive_count": tp,
        "false_positive_fraction": fp_fraction,
        "baseline_true_positive_retention": tp_retention,
        "train_stability": stability,
        "false_positive_fraction_wilson_interval": _wilson_interval(fp, recheck),
        "false_positive_fraction_bootstrap_interval": fp_bootstrap,
        "baseline_true_positive_retention_bootstrap_interval": retention_bootstrap,
        "dataset_concentration": _hhi(dataset_counts),
        "lineage_concentration": _hhi(lineage_counts),
        "passes_research_gate": fp_fraction >= .80 and tp_retention >= .99 and stability >= .99,
        "lane_parity": True,
        "operational_policy_changed": False,
    }
    comparisons = _comparisons("exp405", official, result, (*_EXP405_INTEGERS, *_EXP405_FLOATS))
    for name in _EXP405_INTERVALS:
        comparisons[f"exp405.{name}"] = _interval_comparison(official.get(name), result[name])
    for name in ("lane_parity", "passes_research_gate", "operational_policy_changed"):
        comparisons[f"exp405.{name}"] = _boolean_comparison(official.get(name), bool(result[name]))
    return result, comparisons


def _reaggregate_exp408(
    prediction: Sequence[Mapping[str, object]],
    audit: Mapping[str, int],
    labels: Mapping[str, Mapping[int, int]],
    consensus_path: Path,
    metrics_path: Path,
    candidate_path: Path,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    consensus = _read_jsonl(consensus_path)
    expected_hard = {(str(row["dataset"]), _integer(row["candidate_index"], "prediction candidate index")) for row in prediction}
    candidate_sets: dict[tuple[str, str, str], set[int]] = {}
    lane_sets: dict[str, dict[str, set[int]]] = {}
    for row in consensus:
        dataset = _text(row.get("dataset"), "Exp408 dataset")
        payload = _mapping(row.get("consensus"), "Exp408 consensus")
        lanes = {
            "hard": _indices(payload.get("hard_indices"), "Exp408 hard indices"),
            "standard": _indices(payload.get("standard_indices"), "Exp408 standard indices"),
            "priority": _indices(payload.get("priority_indices"), "Exp408 priority indices"),
        }
        lane_sets[dataset] = lanes
        candidates = _mapping(payload.get("candidates_by_variant"), "Exp408 candidates_by_variant")
        for variant, alpha_map in candidates.items():
            for alpha, values in _mapping(alpha_map, "Exp408 alpha candidates").items():
                indices = {
                    _integer(_mapping(value, "Exp408 candidate").get("candidate_index"), "Exp408 candidate index")
                    for value in _sequence(values, "Exp408 candidate rows")
                }
                if indices & lanes["hard"]:
                    raise ValueError(f"Exp408 Shadow candidate overlaps frozen Hard: {dataset}")
                candidate_sets[(dataset, str(variant), str(alpha))] = indices
    if {(dataset, index) for dataset, lanes in lane_sets.items() for index in lanes["hard"]} != expected_hard:
        raise ValueError("Exp408 frozen Hard lane parity mismatch")
    candidate_rows = _read_candidate_csv(candidate_path)
    _require_columns(candidate_rows, {"dataset", "variant", "alpha", "candidate_index", "is_true_anomaly"}, candidate_path.name)
    candidate_outcomes: dict[tuple[str, str, str], dict[int, int]] = defaultdict(dict)
    for row in candidate_rows:
        key = (
            _text(row.get("dataset"), "Exp408 candidate dataset"),
            _text(row.get("variant"), "Exp408 candidate variant"),
            _text(row.get("alpha"), "Exp408 candidate alpha"),
        )
        index = _integer(row.get("candidate_index"), "Exp408 candidate index")
        outcome = _binary_integer(row.get("is_true_anomaly"), "Exp408 is_true_anomaly")
        if index in candidate_outcomes[key]:
            raise ValueError("Exp408 candidate evaluation contains duplicates")
        candidate_outcomes[key][index] = outcome
    rows = _read_csv(metrics_path)
    required = {
        "dataset", "lineage", "train_size", "variant", "alpha", *_EXP408_INTEGERS,
        "precision", "baseline_hard_tp", "baseline_hard_fp", "baseline_hard_fn",
    }
    _require_columns(rows, required, metrics_path.name)
    totals: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    zero_before: dict[tuple[str, str], set[str]] = defaultdict(set)
    zero_after: dict[tuple[str, str], set[str]] = defaultdict(set)
    concentration_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    lineage_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    comparisons: dict[str, Mapping[str, object]] = {}
    covered: set[str] = set()
    for row in rows:
        dataset = _text(row.get("dataset"), "Exp408 metric dataset")
        variant = _text(row.get("variant"), "Exp408 variant")
        alpha = _text(row.get("alpha"), "Exp408 alpha")
        key = (dataset, variant, alpha)
        if dataset not in audit or dataset not in lane_sets:
            raise ValueError(f"Exp408 metric dataset has no frozen publication: {dataset}")
        train_size = _integer(row.get("train_size"), "Exp408 train_size")
        if train_size != audit[dataset]:
            raise ValueError(f"Exp408 TRAIN size does not match frozen audit: {dataset}")
        candidate_count = _integer(row.get("candidate_count"), "Exp408 candidate_count")
        actual_candidates = candidate_sets.get(key)
        if actual_candidates is None or candidate_count != len(actual_candidates):
            raise ValueError(f"Exp408 candidate count does not match frozen consensus: {dataset}/{variant}/{alpha}")
        outcomes = candidate_outcomes.get(key, {})
        if set(outcomes) != actual_candidates:
            raise ValueError(f"Exp408 candidate evaluation coverage is invalid: {dataset}/{variant}/{alpha}")
        if dataset not in labels or not actual_candidates <= set(labels[dataset]):
            raise ValueError(f"Exp408 frozen label coverage is invalid: {dataset}")
        standard_overlap = len(actual_candidates & lane_sets[dataset]["standard"])
        priority_overlap = len(actual_candidates & lane_sets[dataset]["priority"])
        outside_existing = len(actual_candidates - lane_sets[dataset]["standard"] - lane_sets[dataset]["priority"])
        if any(outcomes[index] != labels[dataset][index] for index in actual_candidates):
            raise ValueError(f"Exp408 candidate evaluation differs from frozen labels: {dataset}/{variant}/{alpha}")
        tp = sum(labels[dataset][index] for index in actual_candidates)
        fp = candidate_count - tp
        if not lane_sets[dataset]["hard"] <= set(labels[dataset]):
            raise ValueError(f"Exp408 frozen Hard label coverage is invalid: {dataset}")
        baseline_hard_tp = sum(labels[dataset][index] for index in lane_sets[dataset]["hard"])
        baseline_hard_fn = sum(labels[dataset].values()) - baseline_hard_tp
        expected = {
            "candidate_count": candidate_count,
            "tp": tp,
            "fp": fp,
            "fn_recovered": tp,
            "zero_f1_before": int(baseline_hard_tp == 0),
            "zero_f1_after": int(baseline_hard_tp + tp == 0),
            "standard_overlap": standard_overlap,
            "priority_overlap": priority_overlap,
            "review_outside_existing": outside_existing,
            "baseline_hard_tp": baseline_hard_tp,
            "baseline_hard_fp": len(lane_sets[dataset]["hard"]) - baseline_hard_tp,
            "baseline_hard_fn": baseline_hard_fn,
        }
        for name, independent in expected.items():
            comparison = _comparison(row.get(name), independent, integer=True)
            comparisons[f"exp408.{dataset}.{variant}.{alpha}.{name}"] = comparison
            if not comparison["matched"]:
                raise ValueError(f"independent reaggregation mismatch: exp408.{dataset}.{variant}.{alpha}.{name}")
        precision = tp / candidate_count if candidate_count else 0.0
        comparison = _comparison(row.get("precision"), precision, integer=False)
        comparisons[f"exp408.{dataset}.{variant}.{alpha}.precision"] = comparison
        if not comparison["matched"]:
            raise ValueError(f"independent reaggregation mismatch: exp408.{dataset}.{variant}.{alpha}.precision")
        metric_key = (variant, alpha)
        for name in _EXP408_INTEGERS:
            totals[metric_key][name] += _integer(row.get(name), f"Exp408 {name}")
        totals[metric_key]["baseline_hard_tp"] += baseline_hard_tp
        totals[metric_key]["baseline_hard_fp"] += _integer(row.get("baseline_hard_fp"), "Exp408 baseline Hard FP")
        totals[metric_key]["baseline_hard_fn"] += _integer(row.get("baseline_hard_fn"), "Exp408 baseline Hard FN")
        if _integer(row.get("zero_f1_before"), "Exp408 zero F1 before"):
            zero_before[metric_key].add(dataset)
        if _integer(row.get("zero_f1_after"), "Exp408 zero F1 after"):
            zero_after[metric_key].add(dataset)
        concentration_counts[metric_key][dataset] += candidate_count
        lineage_counts[metric_key][_text(row.get("lineage"), "Exp408 lineage")] += candidate_count
        covered.add(dataset)
    variants: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    workload: dict[str, dict[str, int]] = defaultdict(dict)
    concentration: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (variant, alpha), values in sorted(totals.items()):
        candidate_count = values["candidate_count"]
        variants[variant][alpha] = {
            **dict(values),
            "precision": values["tp"] / candidate_count if candidate_count else 0.0,
            "zero_f1_before": len(zero_before[(variant, alpha)]),
            "zero_f1_after": len(zero_after[(variant, alpha)]),
        }
        workload[variant][alpha] = candidate_count
        concentration[variant][alpha] = {
            "dataset_hhi": _hhi(concentration_counts[(variant, alpha)]),
            "lineage_hhi": _hhi(lineage_counts[(variant, alpha)]),
        }
    return {
        "exp406": _source_summary(variants, "tri_view_only"),
        "exp407": _source_summary(variants, "synthetic_only"),
        "variants": {variant: dict(values) for variant, values in variants.items()},
        "train_size_coverage": _train_size_coverage(audit, covered),
        "workload": {variant: dict(values) for variant, values in workload.items()},
        "concentration": {variant: dict(values) for variant, values in concentration.items()},
    }, comparisons


def _source_summary(variants: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, object]:
    return dict(variants.get(name, {}))


def _train_size_coverage(audit: Mapping[str, int], covered: set[str]) -> dict[str, int]:
    result = Counter({"abstain": 0, "robust_distance": 0, "classical_committee": 0, "learned_methods": 0, "candidate_datasets": len(covered)})
    for train_size in audit.values():
        if train_size < 5:
            result["abstain"] += 1
        elif train_size < 20:
            result["robust_distance"] += 1
        elif train_size < 50:
            result["classical_committee"] += 1
        else:
            result["learned_methods"] += 1
    return dict(result)


def _comparisons(prefix: str, official: Mapping[str, object], independent: Mapping[str, object], fields: Sequence[str]) -> dict[str, Mapping[str, object]]:
    return {
        f"{prefix}.{field}": _comparison(official.get(field), independent[field], integer=field in _EXP405_INTEGERS)
        for field in fields
    }


def _comparison(official: object, independent: object, *, integer: bool) -> Mapping[str, object]:
    if integer:
        observed = _integer(official, "official integer metric")
        expected = _integer(independent, "independent integer metric")
        return MappingProxyType({"official": observed, "independent": expected, "matched": observed == expected})
    observed = _float(official, "official float metric")
    expected = _float(independent, "independent float metric")
    return MappingProxyType({"official": observed, "independent": expected, "matched": abs(observed - expected) <= FLOAT_TOLERANCE})


def _boolean_comparison(official: object, independent: bool) -> Mapping[str, object]:
    if not isinstance(official, bool):
        raise ValueError("official boolean metric must be a boolean")
    return MappingProxyType({"official": official, "independent": independent, "matched": official is independent})


def _interval_comparison(official: object, independent: object) -> Mapping[str, object]:
    observed = _interval(official, "official interval")
    expected = _interval(independent, "independent interval")
    return MappingProxyType({
        "official": observed,
        "independent": expected,
        "matched": all(abs(left - right) <= FLOAT_TOLERANCE for left, right in zip(observed, expected)),
    })


def _interval(value: object, name: str) -> tuple[float, float]:
    values = _sequence(value, name)
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    lower, upper = (_float(item, name) for item in values)
    if lower > upper:
        raise ValueError(f"{name} bounds are invalid")
    return (lower, upper)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    z_squared = _WILSON_Z_95**2
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    radius = _WILSON_Z_95 * math.sqrt(
        proportion * (1.0 - proportion) / total + z_squared / (4.0 * total**2)
    ) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def _paired_bootstrap(
    rows: Sequence[tuple[int, int, int, int]],
    *,
    point_fp_fraction: float,
    point_tp_retention: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not rows:
        return ((point_fp_fraction, point_fp_fraction), (point_tp_retention, point_tp_retention))
    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    fp_fractions: list[float] = []
    retentions: list[float] = []
    for _ in range(_BOOTSTRAP_ITERATIONS):
        sampled = [rows[index] for index in generator.integers(0, len(rows), size=len(rows))]
        recheck = sum(row[0] for row in sampled)
        if recheck:
            fp_fractions.append(sum(row[1] for row in sampled) / recheck)
        baseline_tp = sum(row[3] for row in sampled)
        if baseline_tp:
            retentions.append((baseline_tp - sum(row[2] for row in sampled)) / baseline_tp)
    return (_percentile_interval(fp_fractions, point_fp_fraction), _percentile_interval(retentions, point_tp_retention))


def _percentile_interval(values: Sequence[float], point: float) -> tuple[float, float]:
    if not values:
        return (point, point)
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))


def _load_audit(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in _read_jsonl(path):
        dataset = _text(row.get("dataset"), "audit dataset")
        train_size = _integer(row.get("train_size"), "audit train_size")
        if dataset in result or train_size < 0:
            raise ValueError("frozen audit coverage is invalid")
        result[dataset] = train_size
    if not result:
        raise ValueError("frozen audit is empty")
    return result


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read frozen JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"frozen JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read frozen JSONL artifact: {path}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"frozen JSONL artifact is invalid: {path}")
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"cannot read frozen CSV artifact: {path}") from error
    if not rows:
        raise ValueError(f"frozen CSV artifact is empty: {path}")
    return rows


def _read_candidate_csv(path: Path) -> list[dict[str, str]]:
    try:
        return _read_csv(path)
    except ValueError as error:
        raise ValueError(f"candidate evaluation artifact is invalid: {path}") from error


def _require_columns(rows: Sequence[Mapping[str, object]], required: set[str], name: str) -> None:
    actual = set(rows[0])
    if not required <= actual:
        raise ValueError(f"frozen CSV columns are incomplete: {name}")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"cannot hash frozen artifact: {path}") from error


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): child for key, child in value.items()}


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _indices(value: object, name: str) -> set[int]:
    return {_integer(item, name) for item in _sequence(value, name)}


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if str(value).strip() not in {str(result), f"+{result}"} or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _binary_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result not in {0, 1}:
        raise ValueError(f"{name} must be zero or one")
    return result


def _float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _hhi(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    return sum((count / total) ** 2 for count in counts.values()) if total else 0.0
