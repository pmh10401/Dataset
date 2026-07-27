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

_EXP405_INTEGERS = (
    "recheck_count",
    "false_positive_count",
    "true_positive_count",
)
_EXP405_FLOATS = (
    "false_positive_fraction",
    "baseline_true_positive_retention",
    "train_stability",
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


@dataclass(frozen=True)
class IndependentReaggregation:
    """Successful independent reconstruction and exact official comparisons."""

    metrics: Mapping[str, object]
    comparisons: Mapping[str, Mapping[str, object]]
    mismatch_count: int


def independent_reaggregate(output_dir: Path) -> IndependentReaggregation:
    """Reaggregate frozen publications and reject any metric disagreement.

    Integer fields must match exactly. Floating-point fields may differ by at
    most ``1e-12``. No database, label loader, or official metric helper is
    consulted here.
    """
    output = Path(output_dir)
    manifest = _read_object(output / PREDICTION_MANIFEST)
    prediction = _load_prediction(output, manifest)
    audit = _load_audit(output / AUDIT_ROWS)
    exp405, exp405_comparisons = _reaggregate_exp405(
        prediction, audit, _read_object(output / EXP405_EVALUATION), output / EXP405_ROWS,
        output / EXP405_CANDIDATE_ROWS,
    )
    exp408, exp408_comparisons = _reaggregate_exp408(
        prediction, audit, output / CONSENSUS_ROWS, output / EXP408_ROWS,
        output / EXP408_CANDIDATE_ROWS,
    )
    metrics: dict[str, object] = {
        "exp404": _exp404_metrics(prediction),
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


def _exp404_metrics(prediction: Sequence[Mapping[str, object]]) -> dict[str, object]:
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
        "grade_counts": {alpha: dict(sorted(counts.items())) for alpha, counts in sorted(grade_counts.items())},
        "frozen_lane_parity": True,
    }


def _reaggregate_exp405(
    prediction: Sequence[Mapping[str, object]],
    audit: Mapping[str, int],
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
        seen.add(identity)
        candidate_by_dataset[identity[0]].append(row)
    if seen != set(prediction_by_identity):
        raise ValueError("Exp405 candidate evaluation coverage is incomplete")
    recheck = fp = tp = baseline_tp = 0
    stability_weighted = 0.0
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
    result: dict[str, object] = {
        "recheck_count": recheck,
        "false_positive_count": fp,
        "true_positive_count": tp,
        "false_positive_fraction": fp / recheck if recheck else 0.0,
        "baseline_true_positive_retention": (baseline_tp - tp) / baseline_tp if baseline_tp else 1.0,
        "train_stability": stability_weighted / recheck if recheck else 0.0,
        "lane_parity": True,
        "operational_policy_changed": False,
    }
    comparisons = _comparisons("exp405", official, result, (*_EXP405_INTEGERS, *_EXP405_FLOATS))
    for name in ("lane_parity", "operational_policy_changed"):
        comparisons[f"exp405.{name}"] = _boolean_comparison(official.get(name), bool(result[name]))
    return result, comparisons


def _reaggregate_exp408(
    prediction: Sequence[Mapping[str, object]],
    audit: Mapping[str, int],
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
        standard_overlap = len(actual_candidates & lane_sets[dataset]["standard"])
        priority_overlap = len(actual_candidates & lane_sets[dataset]["priority"])
        outside_existing = len(actual_candidates - lane_sets[dataset]["standard"] - lane_sets[dataset]["priority"])
        tp = sum(outcomes.values())
        fp = candidate_count - tp
        baseline_hard_tp = _integer(row.get("baseline_hard_tp"), "Exp408 baseline Hard TP")
        expected = {
            "candidate_count": candidate_count,
            "fn_recovered": tp,
            "zero_f1_before": int(baseline_hard_tp == 0),
            "zero_f1_after": int(baseline_hard_tp + tp == 0),
            "standard_overlap": standard_overlap,
            "priority_overlap": priority_overlap,
            "review_outside_existing": outside_existing,
            "baseline_hard_fp": len(lane_sets[dataset]["hard"]) - baseline_hard_tp,
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
