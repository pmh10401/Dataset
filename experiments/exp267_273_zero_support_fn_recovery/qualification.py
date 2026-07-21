"""Exp271 TRAIN-only source qualification and post-freeze diagnostics.

``qualify_sources`` deliberately has no TEST label argument.  It writes an
immutable membership manifest using prediction bytes and TRAIN OOF evidence.
``evaluate_frozen_qualification`` is the separate post-freeze boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from experiments.exp239_248_diverse_score_sources.foundation import stable_json_hash

from .config import (
    CORRELATION_COLLAPSE_ABS_SPEARMAN,
    CORRELATION_MIN_ALIGNED_TRAIN_OOF,
    PRIMARY_ALPHA,
    REVIEW_RISK_LIMIT,
)
from .source_registry import registration


class MembershipHashError(ValueError):
    """Raised before labels when a frozen membership artifact is modified."""


_LABEL_FIELDS = frozenset({
    "label", "labels", "y_test", "test_label", "test_labels", "tp", "fp", "fn",
    "precision", "recall", "f1", "combined_f1", "recovered_fn", "true_indices",
})


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_prediction_boundary(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower().replace("-", "_")
            if name in _LABEL_FIELDS:
                raise ValueError(f"prediction carries evaluation field: {key}")
            if name in {"contains_test_labels", "contains_evaluation_metrics"} and child is not False:
                raise ValueError(f"prediction boundary flag must be false: {key}")
            _check_prediction_boundary(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_prediction_boundary(child)


def _read_prediction_rows(
    prediction_path: Path, manifest_path: Path, *, expected_datasets: Iterable[str]
) -> tuple[list[dict[str, object]], str, str]:
    payload = Path(prediction_path).read_bytes()
    prediction_hash = hashlib.sha256(payload).hexdigest()
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("prediction_sha256") != prediction_hash:
        raise MembershipHashError("source prediction hash does not match manifest")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("frozen source predictions must contain object rows")
    expected = tuple(sorted(str(value) for value in expected_datasets))
    if len(expected) != len(set(expected)) or not expected:
        raise ValueError("expected datasets must be non-empty and unique")
    primary_rows = [row for row in rows if float(row.get("alpha", -1.0)) == PRIMARY_ALPHA]
    seen: set[tuple[str, str, str]] = set()
    variants: set[tuple[str, str]] = set()
    for row in primary_rows:
        _check_prediction_boundary(row)
        key = (str(row.get("source")), str(row.get("variant")), str(row.get("dataset")))
        if key in seen:
            raise ValueError("duplicate primary prediction row")
        seen.add(key)
        variants.add((key[0], key[1]))
        indices = row.get("candidate_indices")
        if not isinstance(indices, list) or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("candidate indices must be non-negative integer lists")
    expected_primary = {(source, variant, dataset) for source, variant in variants for dataset in expected}
    if set((source, variant, dataset) for source, variant, dataset in seen) != expected_primary:
        raise ValueError("primary prediction coverage is incomplete or has silent omissions")
    return [dict(row) for row in primary_rows], prediction_hash, manifest_hash


def _vector(value: object) -> np.ndarray:
    if isinstance(value, Mapping):
        arrays = [np.asarray(value[key], dtype=float).reshape(-1) for key in sorted(value)]
        return np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def _aligned_train_oof(left: object, right: object) -> tuple[np.ndarray, np.ndarray]:
    """Align TRAIN OOF scores by dataset first, then by query position.

    Full runs provide dataset-keyed vectors.  Flat vectors remain supported for
    compact unit fixtures, but a mapping cannot be paired with a flat vector:
    that would silently discard the dataset alignment contract.
    """
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        common = sorted(set(str(key) for key in left).intersection(str(key) for key in right))
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        for dataset in common:
            left_values = np.asarray(left[dataset], dtype=float).reshape(-1)
            right_values = np.asarray(right[dataset], dtype=float).reshape(-1)
            count = min(len(left_values), len(right_values))
            left_parts.append(left_values[:count])
            right_parts.append(right_values[:count])
        if not left_parts:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        return np.concatenate(left_parts), np.concatenate(right_parts)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        raise ValueError("TRAIN OOF correlation requires both vectors to be dataset-keyed or both flat")
    left_values, right_values = _vector(left), _vector(right)
    count = min(len(left_values), len(right_values))
    return left_values[:count], right_values[:count]


def _registration_or_fallback(key: str) -> tuple[str, str]:
    try:
        item = registration(key)
        return item.family, item.evidence_identity
    except KeyError:
        return key, key


def _variant_key(source: object, variant: object) -> str:
    """Map persisted source/variant columns to the registry's concrete key."""
    source_name = str(source)
    variant_name = str(variant)
    if source_name == "tcn_forecast":
        return f"tcn_{variant_name}"
    if variant_name == "primary":
        return source_name
    return f"{source_name}_{variant_name}"


def _components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edges:
        union(left, right)
    result: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        result[find(node)].append(node)
    return [sorted(values) for _, values in sorted(result.items())]


def collapse_correlated_families(
    train_oof_by_variant: Mapping[str, object],
    *,
    collapse_threshold: float = CORRELATION_COLLAPSE_ABS_SPEARMAN,
    minimum_aligned: int = CORRELATION_MIN_ALIGNED_TRAIN_OOF,
) -> dict[str, object]:
    """Collapse registered evidence with only aligned finite TRAIN OOF evidence."""
    if float(collapse_threshold) != CORRELATION_COLLAPSE_ABS_SPEARMAN:
        raise ValueError("correlation collapse threshold is frozen")
    if int(minimum_aligned) != CORRELATION_MIN_ALIGNED_TRAIN_OOF:
        raise ValueError("minimum aligned TRAIN OOF count is frozen")
    nodes = sorted(str(key) for key in train_oof_by_variant)
    identities = {node: _registration_or_fallback(node) for node in nodes}
    forced: list[tuple[str, str]] = []
    correlation_edges: list[tuple[str, str]] = []
    pair_rows: list[dict[str, object]] = []
    unverified_nodes: set[str] = set()
    for offset, left in enumerate(nodes):
        for right in nodes[offset + 1 :]:
            left_family, left_identity = identities[left]
            right_family, right_identity = identities[right]
            if left_family == right_family or left_identity == right_identity:
                forced.append((left, right))
                pair_rows.append({
                    "left": left, "right": right, "aligned_count": None, "spearman": None,
                    "absolute_spearman": None, "relation": "same_registered_evidence",
                    "independence_verified": False,
                })
                continue
            left_values, right_values = _aligned_train_oof(
                train_oof_by_variant[left], train_oof_by_variant[right]
            )
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            left_values, right_values = left_values[finite], right_values[finite]
            count = len(left_values)
            if count < CORRELATION_MIN_ALIGNED_TRAIN_OOF:
                unverified_nodes.update((left, right))
                pair_rows.append({
                    "left": left, "right": right, "aligned_count": count, "spearman": None,
                    "absolute_spearman": None, "relation": "independence_unverified_low_overlap",
                    "independence_verified": False,
                })
                continue
            correlation = float(spearmanr(left_values, right_values).statistic)
            if not math.isfinite(correlation):
                unverified_nodes.update((left, right))
                pair_rows.append({
                    "left": left, "right": right, "aligned_count": count, "spearman": None,
                    "absolute_spearman": None, "relation": "independence_unverified_nonfinite",
                    "independence_verified": False,
                })
                continue
            absolute = abs(correlation)
            relation = "correlation_collapsed" if absolute >= CORRELATION_COLLAPSE_ABS_SPEARMAN else "independent"
            if relation == "correlation_collapsed":
                correlation_edges.append((left, right))
            pair_rows.append({
                "left": left, "right": right, "aligned_count": count, "spearman": correlation,
                "absolute_spearman": absolute, "relation": relation, "independence_verified": True,
            })
    groups = _components(nodes, [*forced, *correlation_edges])
    output_groups: list[dict[str, object]] = []
    group_by_variant: dict[str, str] = {}
    for index, members in enumerate(groups, 1):
        label = "+".join(members)
        has_unverified = any(member in unverified_nodes for member in members)
        group = {
            "group": label,
            "members": members,
            "families": sorted({identities[member][0] for member in members}),
            "evidence_identities": sorted({identities[member][1] for member in members}),
            "independence_verified": not has_unverified,
            "component_index": index,
        }
        output_groups.append(group)
        group_by_variant.update({member: label for member in members})
    return {
        "threshold": CORRELATION_COLLAPSE_ABS_SPEARMAN,
        "minimum_aligned": CORRELATION_MIN_ALIGNED_TRAIN_OOF,
        "pairs": pair_rows,
        "groups": output_groups,
        "group_by_variant": group_by_variant,
    }


def collapse_votes(
    candidates_by_variant: Mapping[str, Iterable[int]],
    *, group_by_variant: Mapping[str, str] | None = None,
) -> dict[int, set[str]]:
    """Count each family/correlation component once at every candidate index."""
    groups = dict(group_by_variant or {})
    votes: dict[int, set[str]] = defaultdict(set)
    for variant, indices in candidates_by_variant.items():
        default_family, _ = _registration_or_fallback(str(variant))
        group = groups.get(str(variant), default_family)
        for index in indices:
            if type(index) is not int or index < 0:
                raise ValueError("candidate index must be a non-negative integer")
            votes[index].add(group)
    return {index: values for index, values in sorted(votes.items())}


def _membership_hash(payload: Mapping[str, object]) -> str:
    return stable_json_hash({key: value for key, value in payload.items() if key != "membership_sha256"})


def _risk_by_variant(
    risk_rows: Iterable[Mapping[str, object]], expected_datasets: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in risk_rows:
        if float(row.get("alpha", -1.0)) == PRIMARY_ALPHA:
            grouped[_variant_key(row.get("source"), row.get("variant"))].append(row)
    summary: dict[str, dict[str, object]] = {}
    for variant, rows in grouped.items():
        coverage = {str(row.get("dataset")) for row in rows}
        valid = [row for row in rows if row.get("status", "ok") == "ok" and not bool(row.get("abstained", False))]
        upper_values = [float(row["wilson_upper_95"]) for row in valid if math.isfinite(float(row["wilson_upper_95"]))]
        summary[variant] = {
            "coverage_complete": coverage == set(expected_datasets) and len(rows) == len(expected_datasets),
            "risk_pass": bool(upper_values) and max(upper_values) <= REVIEW_RISK_LIMIT,
            "risk_wilson_upper_95": max(upper_values) if upper_values else math.nan,
            "abstained_datasets": sum(bool(row.get("abstained", False)) for row in rows),
        }
    return summary


def _cache_input_hashes(cache_root: Path, rows: Iterable[Mapping[str, object]]) -> dict[str, str]:
    """Read only source-cache provenance, never TEST labels or results."""
    from .source_pipeline import cache_paths

    hashes: dict[str, str] = {}
    for row in rows:
        source = str(row["source"])
        dataset = str(row["dataset"])
        key = f"{source}:{dataset}"
        if key in hashes:
            continue
        _, metadata_path = cache_paths(Path(cache_root), source, dataset)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        value = metadata.get("input_sha256") if isinstance(metadata, Mapping) else None
        if not isinstance(value, str) or not value:
            raise ValueError(f"source cache input hash is missing: {key}")
        hashes[key] = value
    return hashes


def qualify_sources(
    prediction_path: Path,
    prediction_manifest_path: Path,
    *,
    output_dir: Path,
    risk_rows: Iterable[Mapping[str, object]],
    train_oof_by_variant: Mapping[str, object],
    expected_datasets: Iterable[str],
    verify_predictions: bool = True,
    cache_root: Path | None = None,
    source_runtime: Mapping[str, Mapping[str, object]] | None = None,
    input_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Freeze Exp272 membership from prediction hashes and TRAIN-only facts.

    There is intentionally no TEST-label parameter.  ``verify_predictions``
    invokes the source-pipeline verifier before any membership bytes are made.
    """
    expected = tuple(sorted(str(item) for item in expected_datasets))
    if verify_predictions:
        from .source_pipeline import verify_prediction_hash

        verify_prediction_hash(
            prediction_path, prediction_manifest_path, expected_datasets=expected,
            cache_root=cache_root, source_runtime=source_runtime,
        )
    primary_rows, prediction_hash, manifest_hash = _read_prediction_rows(
        prediction_path, prediction_manifest_path, expected_datasets=expected,
    )
    cache_hashes = _cache_input_hashes(cache_root, primary_rows) if cache_root is not None else {}
    if input_hashes is not None and cache_hashes and dict(input_hashes) != cache_hashes:
        raise ValueError("supplied input hashes differ from source-cache provenance")
    predicted_variants = {_variant_key(row["source"], row["variant"]) for row in primary_rows}
    if set(train_oof_by_variant) != predicted_variants:
        raise ValueError("TRAIN OOF coverage differs from frozen primary prediction variants")
    risk = _risk_by_variant(risk_rows, expected)
    if set(risk) != predicted_variants:
        raise ValueError("TRAIN-risk coverage differs from frozen primary prediction variants")
    collapsed = collapse_correlated_families(train_oof_by_variant)
    selected: list[dict[str, object]] = []
    for group in collapsed["groups"]:
        assert isinstance(group, Mapping)
        members = [str(value) for value in group["members"]]
        qualified_members = [
            member for member in members
            if bool(risk[member]["coverage_complete"]) and bool(risk[member]["risk_pass"])
        ]
        if len(qualified_members) != len(members) or not bool(group["independence_verified"]):
            continue
        selected.append({
            "group": group["group"], "members": sorted(qualified_members),
            "families": group["families"], "risk_wilson_upper_95": max(
                float(risk[member]["risk_wilson_upper_95"]) for member in members
            ),
            "coverage_complete": True, "reproducible": True,
            "independence_verified": True,
        })
    payload: dict[str, object] = {
        "schema_version": 1,
        "retrospective": True,
        "review_only": True,
        "contains_test_labels": False,
        "primary_alpha": PRIMARY_ALPHA,
        "review_risk_limit": REVIEW_RISK_LIMIT,
        "correlation_collapse_abs_spearman": CORRELATION_COLLAPSE_ABS_SPEARMAN,
        "correlation_min_aligned_train_oof": CORRELATION_MIN_ALIGNED_TRAIN_OOF,
        "prediction_sha256": prediction_hash,
        "prediction_manifest_sha256": manifest_hash,
        "config_hashes": stable_json_hash(sorted(str(row.get("config_hash", "")) for row in primary_rows)),
        # A full pipeline passes source-cache input hashes.  Focused unit tests
        # use this explicit empty fixture value rather than inventing a hash.
        "input_hashes": dict(sorted((cache_hashes or input_hashes or {}).items())),
        "dataset_list_sha256": stable_json_hash(expected),
        "run_contract_sha256": stable_json_hash(sorted(
            str(row.get("run_contract_sha256", "")) for row in primary_rows
        )),
        "qualified_groups": selected,
        "risk_by_variant": risk,
        "correlation": collapsed,
    }
    payload["membership_sha256"] = _membership_hash(payload)
    _atomic_json(Path(output_dir) / "exp272_source_membership.json", payload)
    return payload


def _verify_membership(path: Path, *, prediction_hash: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MembershipHashError("cannot read frozen source membership") from error
    if not isinstance(payload, dict) or payload.get("membership_sha256") != _membership_hash(payload):
        raise MembershipHashError("frozen membership SHA256 mismatch")
    if payload.get("prediction_sha256") != prediction_hash:
        raise MembershipHashError("membership belongs to a different frozen prediction")
    if payload.get("contains_test_labels") is not False:
        raise MembershipHashError("membership must be label-free")
    return payload


def _metrics(indices: set[int], truth: set[int]) -> dict[str, int | float]:
    tp = len(indices & truth)
    fp = len(indices - truth)
    fn = len(truth - indices)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"count": len(indices), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _load_test_truth(expected: tuple[str, ...]) -> Mapping[str, set[int]]:
    from .diagnostics import load_test_truth

    truth, _ = load_test_truth(expected)
    return truth


def evaluate_frozen_qualification(
    prediction_path: Path,
    prediction_manifest_path: Path,
    membership_path: Path,
    *,
    expected_datasets: Iterable[str],
    existing_lanes: Mapping[str, Mapping[str, Iterable[int]]],
    test_truth: Mapping[str, set[int]] | None = None,
    verify_predictions: bool = True,
    cache_root: Path | None = None,
) -> dict[str, object]:
    """Read TEST labels only after source and membership hashes have passed."""
    expected = tuple(sorted(str(item) for item in expected_datasets))
    if verify_predictions:
        from .source_pipeline import verify_prediction_hash

        verify_prediction_hash(prediction_path, prediction_manifest_path, expected_datasets=expected, cache_root=cache_root)
    rows, prediction_hash, _ = _read_prediction_rows(prediction_path, prediction_manifest_path, expected_datasets=expected)
    membership = _verify_membership(membership_path, prediction_hash=prediction_hash)
    if set(existing_lanes) != set(expected):
        raise ValueError("existing lane coverage is incomplete")
    # The first label-bearing action is deliberately here, after both hashes.
    truth = test_truth if test_truth is not None else _load_test_truth(expected)
    if set(truth) != set(expected):
        raise ValueError("TEST truth coverage is incomplete")
    qualified = {member for group in membership["qualified_groups"] for member in group["members"]}
    candidates: dict[str, set[int]] = {dataset: set() for dataset in expected}
    for row in rows:
        if _variant_key(row["source"], row["variant"]) in qualified:
            candidates[str(row["dataset"])].update(int(index) for index in row["candidate_indices"])
    totals = {"hard": set(), "review": set(), "candidate": set(), "inside_hard": set(), "inside_review": set(), "outside": set(), "zero_support_recovery": set(), "truth": set()}
    for dataset in expected:
        hard = {int(index) for index in existing_lanes[dataset].get("hard", ())}
        review = {int(index) for index in existing_lanes[dataset].get("review", ())}
        existing = hard | review
        source = candidates[dataset]
        zero_support = set(truth[dataset]) - existing
        totals["hard"].update((dataset, index) for index in hard)
        totals["review"].update((dataset, index) for index in review)
        totals["candidate"].update((dataset, index) for index in source)
        totals["inside_hard"].update((dataset, index) for index in source & hard)
        totals["inside_review"].update((dataset, index) for index in source & review)
        totals["outside"].update((dataset, index) for index in source - existing)
        totals["zero_support_recovery"].update((dataset, index) for index in source & zero_support)
        totals["truth"].update((dataset, index) for index in truth[dataset])
    result = {
        "autonomous": {f"hard_{key}": value for key, value in _metrics(totals["hard"], totals["truth"]).items()},
        "human_assisted_review": {f"review_{key}": value for key, value in _metrics(totals["review"], totals["truth"]).items()},
        "source_views": {
            "standalone": _metrics(totals["candidate"], totals["truth"]),
            "inside_frozen_hard": _metrics(totals["inside_hard"], totals["truth"]),
            "inside_frozen_review": _metrics(totals["inside_review"], totals["truth"]),
            "outside_lane": _metrics(totals["outside"], totals["truth"]),
            "zero_support_recovery": _metrics(totals["zero_support_recovery"], totals["truth"]),
        },
        "prediction_sha256": prediction_hash,
        "membership_sha256": membership["membership_sha256"],
        "retrospective": True,
        "human_assisted_only": True,
    }
    return result


__all__ = [
    "MembershipHashError", "collapse_correlated_families", "collapse_votes",
    "evaluate_frozen_qualification", "qualify_sources",
]
