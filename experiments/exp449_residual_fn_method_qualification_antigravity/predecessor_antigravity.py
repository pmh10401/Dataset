"""Immutable, descriptor-pinned Exp448 authority for the Exp449 pre-label phase."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from experiments.exp448_exp447_text_label_boundary_repair_antigravity.publication_identity_antigravity import (
    _assert_directory_antigravity, _assert_pinned_leaf_antigravity, _exact_names_antigravity,
    _open_directory_antigravity, _pin_leaf_antigravity, _read_pinned_leaf_antigravity,
    capture_exp446_publication_antigravity,
)

from .config_antigravity import CANDIDATE_MATRIX_ANTIGRAVITY
from .contracts_antigravity import ConfusionAntigravity, DatasetBaselineAntigravity


_EXP448_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp448_exp447_text_label_boundary_repair_antigravity_20260812")
_LEAVES_ANTIGRAVITY = (
    "artifact_sha256_antigravity.json", "bootstrap_antigravity.json", "candidate_transitions_antigravity.jsonl",
    "confusion_antigravity.json", "dataset_metrics_antigravity.jsonl", "decision_antigravity.json",
    "evaluation_antigravity.json", "false_alarm_datasets_antigravity.jsonl", "regressed_datasets_antigravity.jsonl",
    "rescued_datasets_antigravity.jsonl", "strong_regressions_antigravity.jsonl",
)
_ARTIFACT_SHA256_ANTIGRAVITY = "ea9856c7de039697ed4f352864def0fde1c0d336714e68c9b0cd586d2c796f05"
_EVALUATION_SHA256_ANTIGRAVITY = "a58a74f69dcc13a64efb2a679ee7e93eca40ac8c5db87b29b12ed11ea068b21c"
_DECISION_SHA256_ANTIGRAVITY = "54ac58d89367ec5396dafdb180777f787384ffbd21ead1a96764bfe627bbddd5"
_TARGET_COUNTS_ANTIGRAVITY = (24, 7, 2, 2, 2, 5, 3, 3)
_TARGET_FN_ANTIGRAVITY = (246, 200, 88, 75, 63, 56, 56, 53)


@dataclass(frozen=True, slots=True)
class Exp448AuthorityAntigravity:
    root_identity: tuple[int, int, int, int, int, int, int]
    artifact_map_sha256: str
    evaluation_sha256: str
    decision_sha256: str
    population: int
    target_dataset_count: int
    target_dataset_order: tuple[int, ...]
    target_baseline_rows: tuple[DatasetBaselineAntigravity, ...]
    baseline_confusion: ConfusionAntigravity
    target_family_fn: tuple[int, ...]


def _strict_json_antigravity(payload: bytes, name: str) -> Mapping[str, object]:
    def reject_pairs_antigravity(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Exp448 {name} has duplicate JSON keys")
            result[key] = value
        return result
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_pairs_antigravity,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Exp448 {name} is not strict JSON") from error
    if type(value) is not dict:
        raise ValueError(f"Exp448 {name} is not a JSON object")
    return value


def _strict_jsonl_antigravity(payload: bytes, name: str) -> tuple[Mapping[str, object], ...]:
    if not payload.endswith(b"\n"):
        raise ValueError(f"Exp448 {name} is not complete JSONL")
    return tuple(_strict_json_antigravity(line, name) for line in payload.splitlines() if line)


def _confusion_antigravity(value: object, name: str) -> ConfusionAntigravity:
    if not isinstance(value, Mapping) or not {"tp", "fp", "fn", "tn"}.issubset(value):
        raise ValueError(f"Exp448 {name} confusion is invalid")
    try:
        return ConfusionAntigravity(*(value[key] for key in ("tp", "fp", "fn", "tn")))
    except ValueError as error:
        raise ValueError(f"Exp448 {name} confusion is invalid") from error


def _validate_antigravity(leaves: Mapping[str, bytes], digests: Mapping[str, str], root_identity: tuple[int, int, int, int, int, int, int]) -> Exp448AuthorityAntigravity:
    if digests["artifact_sha256_antigravity.json"] != _ARTIFACT_SHA256_ANTIGRAVITY or digests["evaluation_antigravity.json"] != _EVALUATION_SHA256_ANTIGRAVITY or digests["decision_antigravity.json"] != _DECISION_SHA256_ANTIGRAVITY:
        raise ValueError("Exp448 frozen leaf digest differs")
    artifact_map = _strict_json_antigravity(leaves["artifact_sha256_antigravity.json"], "artifact map")
    expected_names = set(_LEAVES_ANTIGRAVITY) - {"artifact_sha256_antigravity.json"}
    if set(artifact_map) != expected_names or any(artifact_map.get(name) != digests[name] for name in expected_names):
        raise ValueError("Exp448 artifact map does not bind the exact leaves")
    evaluation = _strict_json_antigravity(leaves["evaluation_antigravity.json"], "evaluation")
    if (evaluation.get("experiment_id"), evaluation.get("schema"), evaluation.get("dataset_count"), evaluation.get("label_rows"), evaluation.get("run_identity")) != ("Exp448", "exp448", 1117, 195900, "exp448-exp447-text-label-boundary-repair-antigravity-20260812"):
        raise ValueError("Exp448 evaluation authority is invalid")
    decision = _strict_json_antigravity(leaves["decision_antigravity.json"], "decision")
    if decision.get("evaluation_status") != "descriptive_complete_no_promotion" or decision.get("scientific_winner_selected") is not False:
        raise ValueError("Exp448 decision authority is invalid")
    confusion_payload = _strict_json_antigravity(leaves["confusion_antigravity.json"], "confusion")
    methods = confusion_payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError("Exp448 confusion methods are invalid")
    by_name = {row.get("comparison_name"): row for row in methods if isinstance(row, Mapping)}
    baseline_confusion = _confusion_antigravity(by_name.get("hard_plus_stable_26"), "stable-26")
    if (baseline_confusion.tp, baseline_confusion.fp, baseline_confusion.fn, baseline_confusion.tn) != (1930, 314, 1988, 191668):
        raise ValueError("Exp448 stable-26 confusion differs")
    metrics = _strict_jsonl_antigravity(leaves["dataset_metrics_antigravity.jsonl"], "dataset metrics")
    if len(metrics) != 1117:
        raise ValueError("Exp448 dataset population differs")
    families = tuple(name for name, _ in CANDIDATE_MATRIX_ANTIGRAVITY)
    family_rows: dict[str, list[Mapping[str, object]]] = {family: [] for family in families}
    seen: set[str] = set()
    for position, row in enumerate(metrics):
        if set(row) != {"dataset", "dataset_order", "hard_baseline", "hard_plus_source_30", "hard_plus_stable_26", "hard_plus_unstable_4"}:
            raise ValueError("Exp448 dataset metric schema is invalid")
        dataset, order = row["dataset"], row["dataset_order"]
        if type(dataset) is not str or not dataset or type(order) is not int or order != position or dataset in seen:
            raise ValueError("Exp448 dataset order is invalid")
        seen.add(dataset)
        for family in families:
            if dataset.startswith(family + "_"):
                family_rows[family].append(row)
    if tuple(len(family_rows[family]) for family in families) != _TARGET_COUNTS_ANTIGRAVITY:
        raise ValueError("Exp448 target family membership is invalid")
    first = capture_exp446_publication_antigravity()
    second = capture_exp446_publication_antigravity()
    if first is second or first != second or first.label_access_count != 0:
        raise ValueError("Exp446 stable publication authority is invalid")
    stable_by_dataset: dict[str, set[int]] = {}
    for dataset, candidate in first.stable_coordinates:
        stable_by_dataset.setdefault(dataset, set()).add(candidate)
    target_rows = tuple(row for row in metrics if any(row["dataset"].startswith(family + "_") for family in families))
    baselines: list[DatasetBaselineAntigravity] = []
    for row in target_rows:
        dataset = row["dataset"]
        family = next(family for family in families if dataset.startswith(family + "_"))
        baselines.append(DatasetBaselineAntigravity(row["dataset_order"], family, frozenset(stable_by_dataset.get(dataset, set())), _confusion_antigravity(row["hard_plus_stable_26"], dataset)))
    family_fn = tuple(sum(item.confusion.fn for item in baselines if item.family == family) for family in families)
    if family_fn != _TARGET_FN_ANTIGRAVITY or sum(item.confusion.fn for item in baselines) != sum(_TARGET_FN_ANTIGRAVITY):
        raise ValueError("Exp448 target family FN authority differs")
    return Exp448AuthorityAntigravity(root_identity, _ARTIFACT_SHA256_ANTIGRAVITY, _EVALUATION_SHA256_ANTIGRAVITY, _DECISION_SHA256_ANTIGRAVITY, 1117, 48, tuple(item.dataset_order for item in baselines), tuple(baselines), baseline_confusion, family_fn)


def capture_exp448_authority_antigravity(root: Path | None = None) -> Exp448AuthorityAntigravity:
    """Read Exp448 under one retained no-follow descriptor transaction."""
    target = _EXP448_ROOT_ANTIGRAVITY if root is None else Path(root)
    root_fd = -1
    pinned = []
    try:
        root_fd, root_identity = _open_directory_antigravity(None, target)
        _exact_names_antigravity(root_fd, _LEAVES_ANTIGRAVITY)
        leaves: dict[str, bytes] = {}
        digests: dict[str, str] = {}
        for name in _LEAVES_ANTIGRAVITY:
            leaf = _pin_leaf_antigravity(root_fd, name)
            pinned.append(leaf)
            leaves[name] = _read_pinned_leaf_antigravity(leaf)
            digests[name] = hashlib.sha256(leaves[name]).hexdigest()
        result = _validate_antigravity(leaves, digests, root_identity)
        for leaf in pinned:
            _assert_pinned_leaf_antigravity(root_fd, leaf)
        _exact_names_antigravity(root_fd, _LEAVES_ANTIGRAVITY)
        _assert_directory_antigravity(None, target, root_fd, root_identity)
        return result
    finally:
        for leaf in pinned:
            os.close(leaf.descriptor)
        if root_fd >= 0:
            os.close(root_fd)


__all__ = ["Exp448AuthorityAntigravity", "capture_exp448_authority_antigravity"]
