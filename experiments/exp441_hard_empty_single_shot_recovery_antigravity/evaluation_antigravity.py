"""Fail-closed post-freeze loader for Exp441 TEST-label facts only."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

import numpy as np

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity, canonical_jsonl_bytes_antigravity,
    exclusive_rename_antigravity, fsync_directory_antigravity,
    parse_canonical_json_bytes_antigravity, pin_exact_regular_inventory_antigravity,
    write_create_only_antigravity,
)
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    BootstrapResultAntigravity,
    DatasetEvaluationFactsAntigravity,
    EvaluationFactsAntigravity,
    FrozenPublicationAntigravity,
    FrozenEvaluationAntigravity,
    FrozenSourceSnapshotAntigravity,
    RetrospectiveEvaluationAntigravity,
    SelectionResultAntigravity,
    retrospective_status_from_evidence_antigravity,
    validate_production_selection_antigravity,
)
from .lifecycle_profile_antigravity import (
    EXP441_PROFILE_ANTIGRAVITY,
    RecoveryLifecycleProfileAntigravity,
)
from .bootstrap_antigravity import paired_dataset_bootstrap_antigravity, project_f1_antigravity
from .reporting_antigravity import (
    _inspect_at_root_antigravity,
    inspect_exp441_publication_antigravity,
)
from .runtime_antigravity import _snapshot_payload_antigravity
from .selector_antigravity import select_hard_empty_single_shot_antigravity
from .sources_antigravity import verify_frozen_sources_antigravity


_LABEL_SQL_ANTIGRAVITY = """\
SELECT i.instance_index, i.label
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index
"""


@dataclass(frozen=True)
class FrozenContextLabelsAntigravity:
    """Detached TEST labels bound to one unchanged Exp441 publication snapshot."""

    context_datasets: tuple[str, ...]
    labels_by_dataset: Mapping[str, tuple[tuple[int, int], ...]]
    publication_sha256: str
    source_audit_sha256: str
    selection: SelectionResultAntigravity
    descriptor_bound_connection_count: int
    label_query_calls: int
    out_of_scope_label_rows: int

    def __post_init__(self) -> None:
        datasets = tuple(self.context_datasets)
        if (not datasets or any(type(dataset) is not str or not dataset for dataset in datasets)
                or len(datasets) != len(set(datasets))):
            raise ValueError("context datasets are invalid")
        if not isinstance(self.labels_by_dataset, Mapping) or set(self.labels_by_dataset) != set(datasets):
            raise ValueError("label datasets are invalid")
        copied: dict[str, tuple[tuple[int, int], ...]] = {}
        for dataset in datasets:
            try:
                rows = tuple(self.labels_by_dataset[dataset])
            except (KeyError, TypeError) as error:
                raise ValueError("label rows are invalid") from error
            normalized: list[tuple[int, int]] = []
            previous = -1
            for row in rows:
                if not isinstance(row, tuple) or len(row) != 2:
                    raise ValueError("label row schema is invalid")
                index, label = row
                if type(index) is not int or index < 0 or index <= previous:
                    raise ValueError("label index is invalid")
                if type(label) is not int or label not in (0, 1):
                    raise ValueError("label value is invalid")
                normalized.append((index, label))
                previous = index
            copied[dataset] = tuple(normalized)
        if (any(type(value) is not str or len(value) != 64 for value in
                (self.publication_sha256, self.source_audit_sha256))
                or not isinstance(self.selection, SelectionResultAntigravity)):
            raise ValueError("frozen label identity is invalid")
        if (type(self.descriptor_bound_connection_count) is not int
                or self.descriptor_bound_connection_count != 1
                or type(self.label_query_calls) is not int
                or self.label_query_calls != len(datasets)
                or type(self.out_of_scope_label_rows) is not int
                or self.out_of_scope_label_rows != 0):
            raise ValueError("frozen label audit is invalid")
        object.__setattr__(self, "context_datasets", datasets)
        object.__setattr__(self, "labels_by_dataset", MappingProxyType(copied))


def _source_audit_sha256_antigravity(snapshot: FrozenSourceSnapshotAntigravity) -> str:
    return sha256(canonical_json_bytes_antigravity(_snapshot_payload_antigravity(snapshot))).hexdigest()


def _profile_publication_antigravity(
    profile: RecoveryLifecycleProfileAntigravity, run_root: Path | None,
) -> FrozenPublicationAntigravity:
    if profile is EXP441_PROFILE_ANTIGRAVITY:
        return inspect_exp441_publication_antigravity()
    return _inspect_at_root_antigravity(
        profile, profile.root_antigravity if run_root is None else Path(run_root),
    )


def _preflight_antigravity(
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
    run_root: Path | None = None,
) -> tuple[FrozenPublicationAntigravity, FrozenSourceSnapshotAntigravity, SelectionResultAntigravity, tuple[str, ...], Mapping[str, int]]:
    publication = _profile_publication_antigravity(profile, run_root)
    if not isinstance(publication, FrozenPublicationAntigravity):
        raise ValueError("publication is invalid")
    snapshot = verify_frozen_sources_antigravity()
    if not isinstance(snapshot, FrozenSourceSnapshotAntigravity) or snapshot.label_access_count != 0:
        raise ValueError("frozen sources are invalid")
    selection = select_hard_empty_single_shot_antigravity(snapshot)
    validate_production_selection_antigravity(selection)
    if selection != publication.selection:
        raise ValueError("publication selection changed")
    datasets = tuple(snapshot.context_datasets)
    if (not isinstance(snapshot.test_row_count_by_dataset, MappingProxyType)
            or set(snapshot.test_row_count_by_dataset) != set(datasets)):
        raise ValueError("frozen row-count map is invalid")
    row_counts = MappingProxyType({dataset: snapshot.test_row_count_by_dataset[dataset] for dataset in datasets})
    coordinates = set(snapshot.hard_coordinates)
    coordinates.update(item.candidate.coordinate for item in selection.selected)
    for dataset, index in coordinates:
        if dataset not in row_counts or type(index) is not int or index < 0 or index >= row_counts[dataset]:
            raise ValueError("frozen coordinate is outside row-count identity")
    if _source_audit_sha256_antigravity(snapshot) != publication.source_audit_sha256:
        raise ValueError("publication source-audit binding changed")
    final_publication = _profile_publication_antigravity(profile, run_root)
    if final_publication != publication:
        raise ValueError("publication changed before label import")
    if (not isinstance(final_publication, FrozenPublicationAntigravity)
            or final_publication.selection != selection
            or final_publication.source_audit_sha256 != _source_audit_sha256_antigravity(snapshot)):
        raise ValueError("publication final binding is invalid")
    return publication, snapshot, selection, datasets, row_counts


def _normalize_label_antigravity(value: object) -> int:
    if type(value) is int and value in (0, 1):
        return value
    if type(value) is str and value in ("0", "1"):
        return int(value)
    raise ValueError("label value is invalid")


def _load_labels_antigravity(connection: object, datasets: tuple[str, ...], row_counts: Mapping[str, int]) -> Mapping[str, tuple[tuple[int, int], ...]]:
    quick_check = connection.execute("PRAGMA quick_check").fetchall()  # type: ignore[attr-defined]
    if quick_check != [("ok",)]:
        raise ValueError("quick_check failed")
    labels: dict[str, tuple[tuple[int, int], ...]] = {}
    for dataset in datasets:
        rows = connection.execute(_LABEL_SQL_ANTIGRAVITY, (dataset,)).fetchall()  # type: ignore[attr-defined]
        normalized: list[tuple[int, int]] = []
        previous: int | None = None
        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("label row schema is invalid")
            index, value = row
            if type(index) is not int or index < 0:
                raise ValueError("label index is invalid")
            if previous is not None and index <= previous:
                raise ValueError("label index is invalid")
            normalized.append((index, _normalize_label_antigravity(value)))
            previous = index
        if len(normalized) != row_counts[dataset]:
            raise ValueError("label row count is invalid")
        labels[dataset] = tuple(normalized)
    return MappingProxyType(labels)


def _require_loaded_coordinates_antigravity(labels: Mapping[str, tuple[tuple[int, int], ...]], snapshot: FrozenSourceSnapshotAntigravity, selection: SelectionResultAntigravity) -> None:
    identities = {dataset: {index for index, _ in rows} for dataset, rows in labels.items()}
    coordinates = set(snapshot.hard_coordinates)
    coordinates.update(item.candidate.coordinate for item in selection.selected)
    for dataset, index in coordinates:
        if dataset not in identities or index not in identities[dataset]:
            raise ValueError("frozen coordinate is absent from loaded TEST identity")


def _load_frozen_context_labels_for_profile_antigravity(
    profile: RecoveryLifecycleProfileAntigravity,
    run_root: Path | None = None,
) -> FrozenContextLabelsAntigravity:
    """Load only frozen-context TEST labels after binding publication and source twice."""
    publication, snapshot, selection, datasets, row_counts = _preflight_antigravity(profile, run_root)
    from experiments.exp417_causal_multihorizon_shadow.inputs import verified_read_only_connection
    connection_entries = 0
    with verified_read_only_connection(Path(FROZEN_CONFIG.database_path), FROZEN_CONFIG.database_sha256) as connection:
        connection_entries += 1
        labels = _load_labels_antigravity(connection, datasets, row_counts)
        _require_loaded_coordinates_antigravity(labels, snapshot, selection)
    post_publication = _profile_publication_antigravity(profile, run_root)
    post_snapshot = verify_frozen_sources_antigravity()
    if (post_publication != publication
            or not isinstance(post_snapshot, FrozenSourceSnapshotAntigravity)
            or post_snapshot != snapshot
            or post_snapshot.label_access_count != 0):
        raise ValueError("post-connection frozen authority changed")
    post_selection = select_hard_empty_single_shot_antigravity(post_snapshot)
    validate_production_selection_antigravity(post_selection)
    if (post_selection != selection
            or _source_audit_sha256_antigravity(post_snapshot) != publication.source_audit_sha256):
        raise ValueError("post-connection source binding changed")
    return FrozenContextLabelsAntigravity(
        datasets, labels, publication.publication_sha256, publication.source_audit_sha256,
        selection, connection_entries, len(datasets), 0,
    )


def load_frozen_context_labels_antigravity() -> FrozenContextLabelsAntigravity:
    """Load labels only for the original fixed Exp441 post-freeze lifecycle."""
    return _load_frozen_context_labels_for_profile_antigravity(EXP441_PROFILE_ANTIGRAVITY)


def _is_regression_antigravity(union_f1: float, baseline_f1: float) -> bool:
    return union_f1 < baseline_f1


def _strong_loss_antigravity(baseline_f1: float, union_f1: float) -> float:
    if _is_regression_antigravity(union_f1, baseline_f1) and baseline_f1 >= 0.8:
        return baseline_f1 - union_f1
    return 0.0


def _confusion_for_predictions_antigravity(
    labels: tuple[tuple[int, int], ...], positives: frozenset[int],
) -> tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for index, label in labels:
        if index in positives:
            if label == 1:
                tp += 1
            else:
                fp += 1
        elif label == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _compute_sparse_retrospective_antigravity(
    snapshot: FrozenContextLabelsAntigravity,
) -> tuple[
    tuple[DatasetEvaluationFactsAntigravity, ...], tuple[str, ...], tuple[str, ...],
    tuple[tuple[int, int, int, int, int, int, int, int, float], ...],
]:
    """Calculate hand-fixture facts without the production population or bootstrap gate."""
    if not isinstance(snapshot, FrozenContextLabelsAntigravity):
        raise ValueError("frozen context labels are invalid")
    selection = snapshot.selection
    if not isinstance(selection, SelectionResultAntigravity):
        raise ValueError("frozen selection is invalid")
    hard_by_dataset: dict[str, set[int]] = {}
    selected_by_dataset: dict[str, set[int]] = {}
    for dataset, index in selection.hard_coordinates:
        hard_by_dataset.setdefault(dataset, set()).add(index)
    for ranked in selection.selected:
        selected_by_dataset.setdefault(ranked.dataset, set()).add(ranked.instance_index)
    facts: list[DatasetEvaluationFactsAntigravity] = []
    rescued: list[str] = []
    regressed: list[str] = []
    rows: list[tuple[int, int, int, int, int, int, int, int, float]] = []
    for dataset in snapshot.context_datasets:
        labels = snapshot.labels_by_dataset.get(dataset)
        if not isinstance(labels, tuple):
            raise ValueError("frozen labels are invalid")
        identity_to_label: dict[int, int] = {}
        for row in labels:
            if (not isinstance(row, tuple) or len(row) != 2 or type(row[0]) is not int
                    or row[0] < 0 or type(row[1]) is not int or row[1] not in (0, 1)
                    or row[0] in identity_to_label):
                raise ValueError("frozen labels are invalid")
            identity_to_label[row[0]] = row[1]
        hard = frozenset(hard_by_dataset.get(dataset, set()))
        selected = frozenset(selected_by_dataset.get(dataset, set()))
        if any(index not in identity_to_label for index in hard | selected):
            raise ValueError("frozen coordinate is absent from loaded TEST identity")
        baseline = _confusion_for_predictions_antigravity(labels, hard)
        union = _confusion_for_predictions_antigravity(labels, hard | selected)
        baseline_f1 = float(project_f1_antigravity(baseline))
        union_f1 = float(project_f1_antigravity(union))
        delta = float(union_f1 - baseline_f1)
        candidate_tp = sum(identity_to_label[index] for index in selected)
        candidate_fp = len(selected) - candidate_tp
        fact = DatasetEvaluationFactsAntigravity(
            dataset, baseline, union, baseline_f1, union_f1, delta, candidate_tp, candidate_fp,
        )
        facts.append(fact)
        rows.append((*baseline, *union, delta))
        if baseline[0] == 0 and union[0] >= 1:
            rescued.append(dataset)
        if _is_regression_antigravity(union_f1, baseline_f1):
            regressed.append(dataset)
    if set(hard_by_dataset).difference(snapshot.context_datasets) or set(selected_by_dataset).difference(snapshot.context_datasets):
        raise ValueError("frozen coordinate dataset is absent from context")
    return tuple(facts), tuple(rescued), tuple(regressed), tuple(rows)


def _retrospective_status_antigravity(
    facts: EvaluationFactsAntigravity, bootstrap: BootstrapResultAntigravity,
) -> str:
    """Compatibility facade for the contract-owned frozen classifier."""
    return retrospective_status_from_evidence_antigravity(facts, bootstrap)


def _is_tolerant_reproduction_value_antigravity(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)


def compute_retrospective_evaluation_antigravity(
    snapshot: FrozenContextLabelsAntigravity,
) -> RetrospectiveEvaluationAntigravity:
    """Return the fixed pure Exp441 retrospective calculation from one label snapshot."""
    if not isinstance(snapshot, FrozenContextLabelsAntigravity) or len(snapshot.context_datasets) != 1117:
        raise ValueError("production frozen context is invalid")
    validate_production_selection_antigravity(snapshot.selection)
    dataset_facts, rescued, regressed, rows = _compute_sparse_retrospective_antigravity(snapshot)
    if len(dataset_facts) != 1117 or len(rows) != 1117:
        raise ValueError("production retrospective population is invalid")
    baseline = tuple(sum(fact.baseline_confusion[index] for fact in dataset_facts) for index in range(4))
    union = tuple(sum(fact.union_confusion[index] for fact in dataset_facts) for index in range(4))
    candidate_tp = sum(fact.candidate_tp for fact in dataset_facts)
    candidate_fp = sum(fact.candidate_fp for fact in dataset_facts)
    baseline_f1 = float(project_f1_antigravity(baseline))
    union_f1 = float(project_f1_antigravity(union))
    micro_delta = float(union_f1 - baseline_f1)
    macro_delta = float(np.asarray([fact.delta for fact in dataset_facts], dtype=np.float64).mean(dtype=np.float64))
    maximum_strong_loss = float(max(
        (_strong_loss_antigravity(fact.baseline_f1, fact.union_f1) for fact in dataset_facts), default=0.0,
    ))
    aggregate = EvaluationFactsAntigravity(
        baseline, union, candidate_tp, candidate_fp, len(rescued), len(regressed), maximum_strong_loss,
        baseline_f1, union_f1, micro_delta, macro_delta,
    )
    bootstrap = paired_dataset_bootstrap_antigravity(np.asarray(rows, dtype=np.float64))
    return RetrospectiveEvaluationAntigravity(
        dataset_facts, rescued, regressed, rows, aggregate, bootstrap,
        _retrospective_status_antigravity(aggregate, bootstrap), False,
    )


_EVALUATION_FILENAMES_ANTIGRAVITY = (
    "evaluation_antigravity.json", "dataset_metrics_antigravity.jsonl",
    "rescued_datasets_antigravity.jsonl", "regressed_datasets_antigravity.jsonl",
    "confusion_antigravity.json", "bootstrap_antigravity.json", "decision_antigravity.json",
    "artifact_sha256_antigravity.json",
)
_EVALUATION_DESTINATION_ANTIGRAVITY = EXP441_PROFILE_ANTIGRAVITY.root_antigravity / "evaluation_antigravity"
_GOVERNANCE_ANTIGRAVITY: Mapping[str, object] = MappingProxyType({
    "official_run_purpose": "deterministic_retrospective_reproduction",
    "bootstrap_interpretation": "descriptive_post_selection",
    "adaptive_followup": True,
    "independent_validation": False,
    "external_generalization_supported": False,
    "confirmatory_statistical_claim": False,
    "scientific_winner_selected": False,
    "review_only": True,
    "operational_alert_eligible": False,
})
_GOVERNANCE_KEYS_ANTIGRAVITY = frozenset(_GOVERNANCE_ANTIGRAVITY)
_EVALUATION_NON_GOVERNANCE_KEYS_ANTIGRAVITY = frozenset({
    "schema", "experiment_id", "run_identity", "publication_sha256", "prediction_sha256",
    "source_audit_sha256", "dataset_count", "candidate_count", "status", "baseline_confusion",
    "union_confusion", "candidate_tp", "candidate_fp", "rescued_dataset_count",
    "regressed_dataset_count", "maximum_strong_loss", "baseline_f1", "union_f1", "micro_delta",
    "macro_delta", "dataset_metrics_sha256", "rescued_datasets_sha256",
    "regressed_datasets_sha256", "confusion_sha256", "lifecycle_profile_sha256",
})
_BOOTSTRAP_NON_GOVERNANCE_KEYS_ANTIGRAVITY = frozenset({
    "schema", "seed", "samples", "population", "numpy_version", "bit_generator", "sample_dtype",
    "endpoint", "quantile_method", "bootstrap_rows_sha256", "dataset_metrics_sha256", "micro_ci", "macro_ci",
})
_DECISION_NON_GOVERNANCE_KEYS_ANTIGRAVITY = frozenset({
    "schema", "experiment_id", "run_identity", "status", "evaluation_sha256", "bootstrap_sha256",
    "dataset_metrics_sha256", "rescued_datasets_sha256", "regressed_datasets_sha256", "confusion_sha256",
    "lifecycle_profile_sha256",
})


def _sha256_antigravity(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _hash_antigravity(value: object, name: str = "hash") -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} is invalid")
    return value


def _strict_payload_antigravity(value: object, keys: frozenset[str], name: str,
                                *, governance: bool = False) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys | (_GOVERNANCE_KEYS_ANTIGRAVITY if governance else frozenset()):
        raise ValueError(f"{name} keys are invalid")
    if governance:
        for key, expected in _GOVERNANCE_ANTIGRAVITY.items():
            if type(value[key]) is not type(expected) or value[key] != expected:
                raise ValueError(f"{name} governance is invalid")
    return value


def _confusion_antigravity(value: object, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4 or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{name} is invalid")
    return tuple(value)  # type: ignore[return-value]


def _finite_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} is invalid")
    return value


def _strict_int_antigravity(value: object, expected: int | None, name: str) -> int:
    if type(value) is not int or value < 0 or (expected is not None and value != expected):
        raise ValueError(f"{name} is invalid")
    return value


def _validate_failure_payload_antigravity(
    payload: bytes,
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
) -> dict[str, object]:
    """Validate one standalone immutable engineering-failure document exactly."""
    value = _strict_payload_antigravity(parse_canonical_json_bytes_antigravity(payload), frozenset({
        "schema", "status", "phase", "error_type", "error", "labels_opened", "destination_committed",
        "experiment_id", "run_identity", "lifecycle_profile_sha256",
    }), "failure", governance=True)
    if (
        value["schema"], value["status"], value["experiment_id"], value["run_identity"],
        value["lifecycle_profile_sha256"],
    ) != (
        profile.schema_antigravity("evaluation-failure"), "engineering_reproduction_failed",
        profile.experiment_id, profile.run_identity, profile.identity_sha256_antigravity,
    ):
        raise ValueError("failure schema is invalid")
    for name in ("phase", "error_type", "error"):
        if type(value[name]) is not str or not value[name]:
            raise ValueError("failure text is invalid")
    for name in ("labels_opened", "destination_committed"):
        if type(value[name]) is not bool:
            raise ValueError("failure flag is invalid")
    return value


def _metric_row_payload_antigravity(fact: DatasetEvaluationFactsAntigravity) -> dict[str, object]:
    return {
        "dataset": fact.dataset, "baseline_confusion": list(fact.baseline_confusion),
        "union_confusion": list(fact.union_confusion), "baseline_f1": fact.baseline_f1,
        "union_f1": fact.union_f1, "delta": fact.delta, "candidate_tp": fact.candidate_tp,
        "candidate_fp": fact.candidate_fp,
    }


def _result_artifacts_antigravity(
    result: RetrospectiveEvaluationAntigravity, publication: FrozenPublicationAntigravity,
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
) -> Mapping[str, bytes]:
    """Serialize one already frozen calculation into the closed Task 11 hash graph."""
    if not isinstance(result, RetrospectiveEvaluationAntigravity) or not isinstance(publication, FrozenPublicationAntigravity):
        raise ValueError("evaluation result is invalid")
    metrics = canonical_jsonl_bytes_antigravity(_metric_row_payload_antigravity(item) for item in result.dataset_facts)
    rescued = canonical_jsonl_bytes_antigravity({"dataset": name} for name in result.rescued_datasets)
    regressed = canonical_jsonl_bytes_antigravity({"dataset": name} for name in result.regressed_datasets)
    confusion = canonical_json_bytes_antigravity({
        "schema": profile.schema_antigravity("evaluation-confusion"), "confusion_order": ["tp", "fp", "fn", "tn"],
        "baseline_confusion": list(result.facts.baseline_confusion), "union_confusion": list(result.facts.union_confusion),
        "candidate_tp": result.facts.candidate_tp, "candidate_fp": result.facts.candidate_fp,
    })
    metrics_sha, rescued_sha, regressed_sha, confusion_sha = map(_sha256_antigravity, (metrics, rescued, regressed, confusion))
    rows = [list(row) for row in result.bootstrap_rows]
    bootstrap = canonical_json_bytes_antigravity({
        "schema": profile.schema_antigravity("bootstrap"), "seed": FROZEN_CONFIG.bootstrap_seed, "samples": FROZEN_CONFIG.bootstrap_samples,
        "population": FROZEN_CONFIG.dataset_count, "numpy_version": FROZEN_CONFIG.numpy_version,
        "bit_generator": "PCG64", "sample_dtype": "int64", "endpoint": False, "quantile_method": "linear",
        "bootstrap_rows_sha256": _sha256_antigravity(canonical_json_bytes_antigravity(rows)),
        "dataset_metrics_sha256": metrics_sha, "micro_ci": list(result.bootstrap.micro_ci), "macro_ci": list(result.bootstrap.macro_ci),
        **_GOVERNANCE_ANTIGRAVITY,
    })
    evaluation = canonical_json_bytes_antigravity({
        "schema": profile.schema_antigravity("evaluation"), "experiment_id": profile.experiment_id,
        "run_identity": profile.run_identity, "lifecycle_profile_sha256": profile.identity_sha256_antigravity,
        "publication_sha256": publication.publication_sha256,
        "prediction_sha256": publication.prediction_sha256, "source_audit_sha256": publication.source_audit_sha256,
        "dataset_count": len(result.dataset_facts), "candidate_count": result.facts.candidate_tp + result.facts.candidate_fp,
        "status": result.status, "baseline_confusion": list(result.facts.baseline_confusion),
        "union_confusion": list(result.facts.union_confusion), "candidate_tp": result.facts.candidate_tp,
        "candidate_fp": result.facts.candidate_fp, "rescued_dataset_count": len(result.rescued_datasets),
        "regressed_dataset_count": len(result.regressed_datasets), "maximum_strong_loss": result.facts.maximum_strong_loss,
        "baseline_f1": result.facts.baseline_f1, "union_f1": result.facts.union_f1,
        "micro_delta": result.facts.micro_delta, "macro_delta": result.facts.macro_delta,
        "dataset_metrics_sha256": metrics_sha, "rescued_datasets_sha256": rescued_sha,
        "regressed_datasets_sha256": regressed_sha, "confusion_sha256": confusion_sha, **_GOVERNANCE_ANTIGRAVITY,
    })
    decision = canonical_json_bytes_antigravity({
        "schema": profile.schema_antigravity("decision"), "experiment_id": profile.experiment_id,
        "run_identity": profile.run_identity, "lifecycle_profile_sha256": profile.identity_sha256_antigravity,
        "status": result.status,
        "evaluation_sha256": _sha256_antigravity(evaluation), "bootstrap_sha256": _sha256_antigravity(bootstrap),
        "dataset_metrics_sha256": metrics_sha, "rescued_datasets_sha256": rescued_sha,
        "regressed_datasets_sha256": regressed_sha, "confusion_sha256": confusion_sha, **_GOVERNANCE_ANTIGRAVITY,
    })
    payloads = {
        "evaluation_antigravity.json": evaluation, "dataset_metrics_antigravity.jsonl": metrics,
        "rescued_datasets_antigravity.jsonl": rescued, "regressed_datasets_antigravity.jsonl": regressed,
        "confusion_antigravity.json": confusion, "bootstrap_antigravity.json": bootstrap,
        "decision_antigravity.json": decision,
    }
    artifact = canonical_json_bytes_antigravity({
        "schema": profile.schema_antigravity("evaluation-artifact-map"), "files": {name: _sha256_antigravity(payload)
                 for name, payload in payloads.items()}, "self_excluded": "artifact_sha256_antigravity.json",
    })
    return MappingProxyType({**payloads, "artifact_sha256_antigravity.json": artifact})


def _parse_jsonl_antigravity(payload: bytes, name: str, *, empty_allowed: bool = False) -> list[dict[str, object]]:
    if payload == b"" and empty_allowed:
        return []
    if not payload.endswith(b"\n") or payload == b"":
        raise ValueError(f"{name} is invalid")
    rows: list[dict[str, object]] = []
    for line in payload.splitlines(keepends=True):
        parsed = parse_canonical_json_bytes_antigravity(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{name} is invalid")
        rows.append(parsed)
    return rows


def _replay_evaluation_payloads_antigravity(
    payloads: Mapping[str, bytes],
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
) -> FrozenEvaluationAntigravity:
    """Purely replay the pinned eight byte payloads; this function never opens a path."""
    if not isinstance(payloads, Mapping) or set(payloads) != set(_EVALUATION_FILENAMES_ANTIGRAVITY) or any(type(value) is not bytes for value in payloads.values()):
        raise ValueError("evaluation inventory is invalid")
    artifact = _strict_payload_antigravity(parse_canonical_json_bytes_antigravity(payloads["artifact_sha256_antigravity.json"]),
        frozenset({"schema", "files", "self_excluded"}), "artifact")
    if artifact["schema"] != profile.schema_antigravity("evaluation-artifact-map") or artifact["self_excluded"] != "artifact_sha256_antigravity.json" or not isinstance(artifact["files"], dict):
        raise ValueError("artifact map is invalid")
    expected_names = set(_EVALUATION_FILENAMES_ANTIGRAVITY) - {"artifact_sha256_antigravity.json"}
    if set(artifact["files"]) != expected_names:
        raise ValueError("artifact map is invalid")
    for name in expected_names:
        if _hash_antigravity(artifact["files"][name], "artifact hash") != _sha256_antigravity(payloads[name]):
            raise ValueError("artifact hash graph is invalid")
    metric_rows = _parse_jsonl_antigravity(payloads["dataset_metrics_antigravity.jsonl"], "metrics")
    if len(metric_rows) != FROZEN_CONFIG.dataset_count:
        raise ValueError("metrics population is invalid")
    facts: list[DatasetEvaluationFactsAntigravity] = []
    for row in metric_rows:
        if set(row) != {"dataset", "baseline_confusion", "union_confusion", "baseline_f1", "union_f1", "delta", "candidate_tp", "candidate_fp"}:
            raise ValueError("metric row keys are invalid")
        if type(row["dataset"]) is not str or type(row["candidate_tp"]) is not int or type(row["candidate_fp"]) is not int:
            raise ValueError("metric row types are invalid")
        facts.append(DatasetEvaluationFactsAntigravity(row["dataset"], _confusion_antigravity(row["baseline_confusion"], "baseline"),
            _confusion_antigravity(row["union_confusion"], "union"), _finite_float_antigravity(row["baseline_f1"], "baseline f1"),
            _finite_float_antigravity(row["union_f1"], "union f1"), _finite_float_antigravity(row["delta"], "delta"),
            row["candidate_tp"], row["candidate_fp"]))
    rescued_rows = _parse_jsonl_antigravity(payloads["rescued_datasets_antigravity.jsonl"], "rescued", empty_allowed=True)
    regressed_rows = _parse_jsonl_antigravity(payloads["regressed_datasets_antigravity.jsonl"], "regressed", empty_allowed=True)
    def names(rows: list[dict[str, object]], label: str) -> tuple[str, ...]:
        if any(set(row) != {"dataset"} or type(row["dataset"]) is not str or not row["dataset"] for row in rows):
            raise ValueError(f"{label} rows are invalid")
        return tuple(row["dataset"] for row in rows)  # type: ignore[return-value]
    rescued, regressed = names(rescued_rows, "rescued"), names(regressed_rows, "regressed")
    confusion = _strict_payload_antigravity(parse_canonical_json_bytes_antigravity(payloads["confusion_antigravity.json"]),
        frozenset({"schema", "confusion_order", "baseline_confusion", "union_confusion", "candidate_tp", "candidate_fp"}), "confusion")
    if (confusion["schema"], confusion["confusion_order"]) != (profile.schema_antigravity("evaluation-confusion"), ["tp", "fp", "fn", "tn"]):
        raise ValueError("confusion schema is invalid")
    evaluation = _strict_payload_antigravity(parse_canonical_json_bytes_antigravity(payloads["evaluation_antigravity.json"]),
        _EVALUATION_NON_GOVERNANCE_KEYS_ANTIGRAVITY, "evaluation", governance=True)
    bootstrap_payload = _strict_payload_antigravity(parse_canonical_json_bytes_antigravity(payloads["bootstrap_antigravity.json"]),
        _BOOTSTRAP_NON_GOVERNANCE_KEYS_ANTIGRAVITY, "bootstrap", governance=True)
    decision = _strict_payload_antigravity(parse_canonical_json_bytes_antigravity(payloads["decision_antigravity.json"]),
        _DECISION_NON_GOVERNANCE_KEYS_ANTIGRAVITY, "decision", governance=True)
    if (evaluation["schema"], evaluation["experiment_id"], evaluation["run_identity"], bootstrap_payload["schema"],
            decision["schema"], decision["experiment_id"], decision["run_identity"]) != (
                profile.schema_antigravity("evaluation"), profile.experiment_id, profile.run_identity,
                profile.schema_antigravity("bootstrap"), profile.schema_antigravity("decision"),
                profile.experiment_id, profile.run_identity):
        raise ValueError("evaluation schema is invalid")
    if (
        evaluation["lifecycle_profile_sha256"], decision["lifecycle_profile_sha256"],
    ) != (profile.identity_sha256_antigravity, profile.identity_sha256_antigravity):
        raise ValueError("evaluation lifecycle profile identity is invalid")
    for key, expected in (("seed", FROZEN_CONFIG.bootstrap_seed), ("samples", FROZEN_CONFIG.bootstrap_samples),
                          ("population", FROZEN_CONFIG.dataset_count), ("numpy_version", FROZEN_CONFIG.numpy_version),
                          ("bit_generator", "PCG64"), ("sample_dtype", "int64"), ("endpoint", False), ("quantile_method", "linear")):
        if type(bootstrap_payload[key]) is not type(expected) or bootstrap_payload[key] != expected:
            raise ValueError("bootstrap identity is invalid")
    rows = tuple((*fact.baseline_confusion, *fact.union_confusion, fact.delta) for fact in facts)
    if _hash_antigravity(bootstrap_payload["bootstrap_rows_sha256"], "bootstrap rows hash") != _sha256_antigravity(canonical_json_bytes_antigravity([list(row) for row in rows])):
        raise ValueError("bootstrap rows identity is invalid")
    metrics_sha = _sha256_antigravity(payloads["dataset_metrics_antigravity.jsonl"])
    if any(_hash_antigravity(value, "artifact reference") != expected for value, expected in (
        (evaluation["dataset_metrics_sha256"], metrics_sha), (bootstrap_payload["dataset_metrics_sha256"], metrics_sha),
        (decision["dataset_metrics_sha256"], metrics_sha), (evaluation["rescued_datasets_sha256"], _sha256_antigravity(payloads["rescued_datasets_antigravity.jsonl"])),
        (decision["rescued_datasets_sha256"], _sha256_antigravity(payloads["rescued_datasets_antigravity.jsonl"])),
        (evaluation["regressed_datasets_sha256"], _sha256_antigravity(payloads["regressed_datasets_antigravity.jsonl"])),
        (decision["regressed_datasets_sha256"], _sha256_antigravity(payloads["regressed_datasets_antigravity.jsonl"])),
        (evaluation["confusion_sha256"], _sha256_antigravity(payloads["confusion_antigravity.json"])),
        (decision["confusion_sha256"], _sha256_antigravity(payloads["confusion_antigravity.json"])),
    )):
        raise ValueError("evaluation hash graph is invalid")
    if _hash_antigravity(decision["evaluation_sha256"], "decision evaluation") != _sha256_antigravity(payloads["evaluation_antigravity.json"]) or _hash_antigravity(decision["bootstrap_sha256"], "decision bootstrap") != _sha256_antigravity(payloads["bootstrap_antigravity.json"]):
        raise ValueError("decision hash graph is invalid")
    bootstrap = paired_dataset_bootstrap_antigravity(np.asarray(rows, dtype=np.float64))
    for key, actual, expected in (("micro_ci", bootstrap_payload["micro_ci"], list(bootstrap.micro_ci)), ("macro_ci", bootstrap_payload["macro_ci"], list(bootstrap.macro_ci))):
        if actual != expected:
            raise ValueError(f"bootstrap {key} is inconsistent")
    for key, expected in (("dataset_count", FROZEN_CONFIG.dataset_count),):
        _strict_int_antigravity(evaluation[key], expected, key)
    candidate_count = _strict_int_antigravity(evaluation["candidate_count"], None, "candidate count")
    for key in ("candidate_tp", "candidate_fp", "rescued_dataset_count", "regressed_dataset_count"):
        _strict_int_antigravity(evaluation[key], None, key)
    if type(evaluation["status"]) is not str or evaluation["status"] not in {
        "engineering_reproduction_failed", "retrospective_reproduction_discrepancy", "retrospective_reproduction_complete",
    } or type(decision["status"]) is not str:
        raise ValueError("evaluation status is invalid")
    for key in ("micro_ci", "macro_ci"):
        if (not isinstance(bootstrap_payload[key], list) or len(bootstrap_payload[key]) != 2
                or any(type(value) is not float or not math.isfinite(value) for value in bootstrap_payload[key])):
            raise ValueError("bootstrap endpoints are invalid")
    aggregate = EvaluationFactsAntigravity(
        _confusion_antigravity(evaluation["baseline_confusion"], "evaluation baseline"),
        _confusion_antigravity(evaluation["union_confusion"], "evaluation union"),
        evaluation["candidate_tp"], evaluation["candidate_fp"], evaluation["rescued_dataset_count"],
        evaluation["regressed_dataset_count"], _finite_float_antigravity(evaluation["maximum_strong_loss"], "maximum loss"),
        _finite_float_antigravity(evaluation["baseline_f1"], "baseline f1"), _finite_float_antigravity(evaluation["union_f1"], "union f1"),
        _finite_float_antigravity(evaluation["micro_delta"], "micro delta"), _finite_float_antigravity(evaluation["macro_delta"], "macro delta"),
    )
    parsed_confusion = (_confusion_antigravity(confusion["baseline_confusion"], "confusion baseline"),
                        _confusion_antigravity(confusion["union_confusion"], "confusion union"),
                        _strict_int_antigravity(confusion["candidate_tp"], None, "confusion candidate tp"),
                        _strict_int_antigravity(confusion["candidate_fp"], None, "confusion candidate fp"))
    if (evaluation["status"] != decision["status"] or candidate_count != aggregate.candidate_tp + aggregate.candidate_fp
            or parsed_confusion != (aggregate.baseline_confusion, aggregate.union_confusion,
                                    aggregate.candidate_tp, aggregate.candidate_fp)):
        raise ValueError("evaluation aggregate identity is invalid")
    result = RetrospectiveEvaluationAntigravity(tuple(facts), rescued, regressed, rows, aggregate, bootstrap, evaluation["status"], False)
    publication_sha = _hash_antigravity(evaluation["publication_sha256"], "publication")
    prediction_sha = _hash_antigravity(evaluation["prediction_sha256"], "prediction")
    source_audit_sha = _hash_antigravity(evaluation["source_audit_sha256"], "source audit")
    return FrozenEvaluationAntigravity(_sha256_antigravity(payloads["artifact_sha256_antigravity.json"]), publication_sha,
        prediction_sha, source_audit_sha, result)


def _write_failure_antigravity(
    parent: Path, staging: Path | None, error: BaseException, phase: str,
    labels_opened: bool, committed: bool,
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
) -> None:
    """Preserve immutable failure evidence without touching a committed destination."""
    try:
        payload = canonical_json_bytes_antigravity({
            "schema": profile.schema_antigravity("evaluation-failure"), "status": "engineering_reproduction_failed", "phase": phase,
            "error_type": type(error).__name__, "error": str(error) or type(error).__name__,
            "labels_opened": labels_opened, "destination_committed": committed,
            "experiment_id": profile.experiment_id, "run_identity": profile.run_identity,
            "lifecycle_profile_sha256": profile.identity_sha256_antigravity,
            **_GOVERNANCE_ANTIGRAVITY,
        })
        _validate_failure_payload_antigravity(payload, profile)
        if committed:
            failed = parent / f"failed_evaluation_audit_{uuid4().hex}_antigravity"
            os.mkdir(failed, 0o700)
            write_create_only_antigravity(failed / "failure_antigravity.json", payload)
            fsync_directory_antigravity(failed)
            fsync_directory_antigravity(parent)
        elif staging is not None and staging.exists() and not staging.is_symlink():
            write_create_only_antigravity(staging / "failure_antigravity.json", payload)
            failed = parent / f"failed_evaluation_staging_{uuid4().hex}_antigravity"
            exclusive_rename_antigravity(staging, failed)
            fsync_directory_antigravity(parent)
    except BaseException as preservation_error:
        if hasattr(error, "add_note"):
            error.add_note(f"failure preservation failed: {type(preservation_error).__name__}: {preservation_error}")


def _preflight_matches_replay_antigravity(replayed: FrozenEvaluationAntigravity,
                                          publication: FrozenPublicationAntigravity,
                                          snapshot: FrozenSourceSnapshotAntigravity,
                                          selection: SelectionResultAntigravity,
                                          datasets: tuple[str, ...]) -> None:
    if (replayed.publication_sha256 != publication.publication_sha256
            or replayed.prediction_sha256 != publication.prediction_sha256
            or replayed.source_audit_sha256 != publication.source_audit_sha256
            or tuple(item.dataset for item in replayed.result.dataset_facts) != datasets
            or replayed.result.facts.candidate_tp + replayed.result.facts.candidate_fp != len(selection.selected)
            or snapshot.context_datasets != datasets):
        raise ValueError("replayed evaluation authority changed")


def _replay_pinned_evaluation_antigravity(
    pinned: object,
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
) -> FrozenEvaluationAntigravity:
    """Replay retained evaluation descriptors without reopening their namespace."""
    payloads = getattr(pinned, "payloads", None)
    result = (
        _replay_evaluation_payloads_antigravity(payloads)
        if profile is EXP441_PROFILE_ANTIGRAVITY
        else _replay_evaluation_payloads_antigravity(payloads, profile)
    )
    pinned.revalidate_antigravity()
    return result


def _route_preflight_antigravity(
    profile: RecoveryLifecycleProfileAntigravity, root: Path,
) -> tuple[FrozenPublicationAntigravity, FrozenSourceSnapshotAntigravity, SelectionResultAntigravity, tuple[str, ...], Mapping[str, int]]:
    if profile is EXP441_PROFILE_ANTIGRAVITY:
        return _preflight_antigravity()
    return _preflight_antigravity(profile, root)


def _route_label_load_antigravity(
    profile: RecoveryLifecycleProfileAntigravity, root: Path,
) -> FrozenContextLabelsAntigravity:
    if profile is EXP441_PROFILE_ANTIGRAVITY:
        return load_frozen_context_labels_antigravity()
    return _load_frozen_context_labels_for_profile_antigravity(profile, root)


def _route_result_artifacts_antigravity(
    profile: RecoveryLifecycleProfileAntigravity, result: RetrospectiveEvaluationAntigravity,
    publication: FrozenPublicationAntigravity,
) -> Mapping[str, bytes]:
    if profile is EXP441_PROFILE_ANTIGRAVITY:
        return _result_artifacts_antigravity(result, publication)
    return _result_artifacts_antigravity(result, publication, profile)


def _route_replay_payloads_antigravity(
    profile: RecoveryLifecycleProfileAntigravity, payloads: Mapping[str, bytes],
) -> FrozenEvaluationAntigravity:
    if profile is EXP441_PROFILE_ANTIGRAVITY:
        return _replay_evaluation_payloads_antigravity(payloads)
    return _replay_evaluation_payloads_antigravity(payloads, profile)


def _route_replay_pinned_antigravity(
    profile: RecoveryLifecycleProfileAntigravity, pinned: object,
) -> FrozenEvaluationAntigravity:
    if profile is EXP441_PROFILE_ANTIGRAVITY:
        return _replay_pinned_evaluation_antigravity(pinned)
    return _replay_pinned_evaluation_antigravity(pinned, profile)


def _evaluate_at_root_antigravity(
    profile: RecoveryLifecycleProfileAntigravity, *,
    evaluation_destination: Path,
    run_root: Path,
) -> FrozenEvaluationAntigravity:
    """Internal test seam beneath the fixed production-profile root boundary."""
    root = Path(run_root)
    destination = Path(evaluation_destination)
    parent = destination.parent
    staging: Path | None = None
    labels_opened = False
    committed = False
    phase = "entry"
    try:
        if destination.exists() or destination.is_symlink():
            phase = "existing_replay"
            committed = True
            with pin_exact_regular_inventory_antigravity(destination, _EVALUATION_FILENAMES_ANTIGRAVITY) as pinned:
                replayed = _route_replay_pinned_antigravity(profile, pinned)
                phase = "existing_preflight"
                publication, snapshot, selection, datasets, _rows = _route_preflight_antigravity(profile, root)
                _preflight_matches_replay_antigravity(replayed, publication, snapshot, selection, datasets)
                pinned.revalidate_antigravity()
                return replayed
        if not parent.is_dir() or parent.is_symlink():
            raise ValueError("evaluation parent is unavailable")
        staging = parent / f".evaluation_{uuid4().hex}_staging_antigravity"
        os.mkdir(staging, 0o700)
        phase = "preflight"
        publication, snapshot, selection, datasets, _rows = _route_preflight_antigravity(profile, root)
        phase = "labels"
        labels_opened = True
        frozen_labels = _route_label_load_antigravity(profile, root)
        if (frozen_labels.context_datasets != datasets or frozen_labels.selection != selection
                or frozen_labels.publication_sha256 != publication.publication_sha256
                or frozen_labels.source_audit_sha256 != publication.source_audit_sha256):
            raise ValueError("frozen labels changed authority")
        phase = "calculation"
        calculated = compute_retrospective_evaluation_antigravity(frozen_labels)
        phase = "post_calculation_preflight"
        post_publication, post_snapshot, post_selection, post_datasets, _post_rows = _route_preflight_antigravity(profile, root)
        if (post_publication != publication or post_snapshot != snapshot or post_selection != selection or post_datasets != datasets
                or frozen_labels.publication_sha256 != publication.publication_sha256
                or frozen_labels.source_audit_sha256 != publication.source_audit_sha256):
            raise ValueError("authority changed after labels")
        payloads = _route_result_artifacts_antigravity(profile, calculated, publication)
        phase = "staging_write"
        for name in _EVALUATION_FILENAMES_ANTIGRAVITY:
            write_create_only_antigravity(staging / name, payloads[name])
        phase = "staging_replay"
        with pin_exact_regular_inventory_antigravity(staging, _EVALUATION_FILENAMES_ANTIGRAVITY) as pinned:
            replayed = _route_replay_payloads_antigravity(profile, pinned.payloads)
            _preflight_matches_replay_antigravity(replayed, publication, snapshot, selection, datasets)
            pinned.revalidate_antigravity()
            fsync_directory_antigravity(staging)
            phase = "pre_rename_preflight"
            final_publication, final_snapshot, final_selection, final_datasets, _final_rows = _route_preflight_antigravity(profile, root)
            if (final_publication != publication or final_snapshot != snapshot or final_selection != selection or final_datasets != datasets):
                raise ValueError("authority changed before rename")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError("evaluation destination appeared during creation")
            phase = "rename"
            exclusive_rename_antigravity(staging, destination)
            committed = True
            pinned.revalidate_at_antigravity(destination, renamed_antigravity=True)
            fsync_directory_antigravity(parent)
            pinned.revalidate_at_antigravity(destination, renamed_antigravity=True)
        phase = "destination_replay"
        with pin_exact_regular_inventory_antigravity(destination, _EVALUATION_FILENAMES_ANTIGRAVITY) as pinned:
            replayed = _route_replay_pinned_antigravity(profile, pinned)
            phase = "final_preflight"
            final_publication, final_snapshot, final_selection, final_datasets, _final_rows = _route_preflight_antigravity(profile, root)
            _preflight_matches_replay_antigravity(replayed, final_publication, final_snapshot, final_selection, final_datasets)
            pinned.revalidate_antigravity()
            return replayed
    except BaseException as error:
        _write_failure_antigravity(parent, staging, error, phase, labels_opened, committed, profile)
        raise


def _evaluate_for_profile_antigravity(
    profile: RecoveryLifecycleProfileAntigravity,
) -> FrozenEvaluationAntigravity:
    """Evaluate only at the sole root and destination bound into the closed profile."""
    root = profile.root_antigravity
    return _evaluate_at_root_antigravity(
        profile,
        evaluation_destination=root / "evaluation_antigravity",
        run_root=root,
    )


def evaluate_exp441_antigravity() -> FrozenEvaluationAntigravity:
    """Create or replay the original fixed Exp441 evaluation."""
    return _evaluate_at_root_antigravity(
        EXP441_PROFILE_ANTIGRAVITY,
        evaluation_destination=_EVALUATION_DESTINATION_ANTIGRAVITY,
        run_root=EXP441_PROFILE_ANTIGRAVITY.root_antigravity,
    )


__all__ = [
    "FrozenContextLabelsAntigravity", "load_frozen_context_labels_antigravity",
    "compute_retrospective_evaluation_antigravity", "evaluate_exp441_antigravity",
]
