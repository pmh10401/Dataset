"""Post-freeze Exp443 TEST-label evaluation and immutable publication.

This is the only Exp443 module allowed to open TEST labels.  Prediction and
reporting modules never import it; the later CLI imports it only in the
``evaluate`` branch after publication inspection.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
import os
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import numpy as np

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    canonical_jsonl_bytes_antigravity,
    exclusive_rename_antigravity,
    fsync_directory_antigravity,
    parse_canonical_json_bytes_antigravity,
    read_pinned_regular_bytes_antigravity,
    write_create_only_antigravity,
)
from .bootstrap_antigravity import (
    BootstrapResultAntigravity,
    MetricSetAntigravity,
    paired_dataset_bootstrap_antigravity,
    project_metrics_antigravity,
)
from .config_antigravity import FROZEN_CONFIG, governance_payload_antigravity
from .database_antigravity import parse_dataset_family_antigravity
from .reporting_antigravity import FrozenPublicationAntigravity, inspect_publication_antigravity
from .sources_antigravity import verify_frozen_sources_antigravity


LABEL_SQL_ANTIGRAVITY = """\
SELECT i.instance_index, i.label
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index
"""

EVALUATION_NAMES_ANTIGRAVITY = (
    "evaluation_antigravity.json",
    "dataset_metrics_antigravity.jsonl",
    "coordinate_transitions_antigravity.jsonl",
    "rescued_datasets_antigravity.jsonl",
    "regressed_datasets_antigravity.jsonl",
    "family_diagnostics_antigravity.jsonl",
    "confusion_antigravity.json",
    "bootstrap_antigravity.json",
    "decision_antigravity.json",
    "artifact_sha256_antigravity.json",
)
_POLICIES_ANTIGRAVITY = tuple(FROZEN_CONFIG.policies)
_BASELINE_CONFUSION_ANTIGRAVITY = (
    FROZEN_CONFIG.baseline_tp,
    FROZEN_CONFIG.baseline_fp,
    FROZEN_CONFIG.baseline_fn,
    FROZEN_CONFIG.baseline_tn,
)
_BASELINE_F1_ANTIGRAVITY = 0.6241851368970013
_PREDICTION_ROW_KEYS_ANTIGRAVITY = frozenset({
    "dataset", "policy", "n_test", "mini1024", "mini4096",
    "inherited_hard", "inherited_standard", "inherited_priority",
    "outside_standard", "promotion", "pre_exact", "exact_normal_removed",
    "final_hard", "contains_test_labels", "contains_evaluation_metrics",
    "provenance",
})


def _hex_antigravity(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} is not a lowercase SHA256")
    return value


def _confusion_antigravity(value: object, label: str) -> tuple[int, int, int, int]:
    try:
        counts = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{label} is invalid") from error
    if len(counts) != 4 or any(type(count) is not int or count < 0 for count in counts):
        raise ValueError(f"{label} is invalid")
    return counts  # type: ignore[return-value]


def _indices_antigravity(value: object, *, n_test: int, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} is invalid")
    indices = tuple(value)
    if (
        any(type(index) is not int or index < 0 or index >= n_test for index in indices)
        or indices != tuple(sorted(set(indices)))
    ):
        raise ValueError(f"{label} is invalid")
    return indices


@dataclass(frozen=True)
class FrozenLabelsAntigravity:
    """Detached binary TEST labels bound to one unchanged publication."""

    datasets: tuple[str, ...]
    labels_by_dataset: Mapping[str, tuple[int, ...]]
    publication_manifest_sha256: str
    prediction_sha256: tuple[tuple[str, str], ...]
    database_sha256: str
    label_identity_sha256: str
    descriptor_bound_connection_count: int
    label_query_calls: int
    label_rows: int
    pre_freeze_label_access_count: int = 0

    def __post_init__(self) -> None:
        datasets = tuple(self.datasets)
        if (
            not datasets
            or datasets != tuple(sorted(set(datasets)))
            or any(type(dataset) is not str or not dataset for dataset in datasets)
        ):
            raise ValueError("frozen label datasets are invalid")
        if not isinstance(self.labels_by_dataset, Mapping) or set(self.labels_by_dataset) != set(datasets):
            raise ValueError("frozen label map is invalid")
        copied: dict[str, tuple[int, ...]] = {}
        for dataset in datasets:
            values = tuple(self.labels_by_dataset[dataset])
            if not values or any(type(value) is not int or value not in (0, 1) for value in values):
                raise ValueError("frozen labels are invalid")
            copied[dataset] = values
        _hex_antigravity(self.publication_manifest_sha256, "publication manifest SHA256")
        _hex_antigravity(self.database_sha256, "database SHA256")
        _hex_antigravity(self.label_identity_sha256, "label identity SHA256")
        if tuple(policy for policy, _digest in self.prediction_sha256) != _POLICIES_ANTIGRAVITY:
            raise ValueError("frozen prediction policy identity is invalid")
        for _policy, digest in self.prediction_sha256:
            _hex_antigravity(digest, "prediction SHA256")
        if self.descriptor_bound_connection_count != 1:
            raise ValueError("label connection count is invalid")
        if self.label_query_calls != len(datasets):
            raise ValueError("label query count is invalid")
        if self.label_rows != sum(len(values) for values in copied.values()):
            raise ValueError("label row count is invalid")
        if type(self.pre_freeze_label_access_count) is not int or self.pre_freeze_label_access_count != 0:
            raise ValueError("pre-freeze label access must remain zero")
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "labels_by_dataset", MappingProxyType(copied))


@dataclass(frozen=True)
class DatasetEvaluationRowAntigravity:
    policy: str
    dataset: str
    family: str
    affected: bool
    n_test: int
    baseline_confusion: tuple[int, int, int, int]
    policy_confusion: tuple[int, int, int, int]
    baseline_metrics: MetricSetAntigravity
    policy_metrics: MetricSetAntigravity
    added_coordinates: tuple[int, ...]
    removed_coordinates: tuple[int, ...]
    added_tp: int
    added_fp: int
    removed_tp: int
    removed_fp: int
    rescued: bool
    regressed: bool
    strong_regression: bool

    def __post_init__(self) -> None:
        if self.policy not in _POLICIES_ANTIGRAVITY or type(self.dataset) is not str or not self.dataset:
            raise ValueError("dataset evaluation identity is invalid")
        if type(self.family) is not str or not self.family:
            raise ValueError("dataset family is invalid")
        if type(self.affected) is not bool or type(self.n_test) is not int or self.n_test < 1:
            raise ValueError("dataset evaluation dimensions are invalid")
        baseline = _confusion_antigravity(self.baseline_confusion, "baseline confusion")
        policy = _confusion_antigravity(self.policy_confusion, "policy confusion")
        if sum(baseline) != self.n_test or sum(policy) != self.n_test:
            raise ValueError("dataset confusion does not conserve TEST rows")
        if self.baseline_metrics != project_metrics_antigravity(baseline) or self.policy_metrics != project_metrics_antigravity(policy):
            raise ValueError("dataset metrics disagree with confusion")
        for name in ("added_coordinates", "removed_coordinates"):
            values = tuple(getattr(self, name))
            if any(type(value) is not int or value < 0 or value >= self.n_test for value in values) or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} are invalid")
            object.__setattr__(self, name, values)
        for name in ("added_tp", "added_fp", "removed_tp", "removed_fp"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} is invalid")
        if self.added_tp + self.added_fp != len(self.added_coordinates) or self.removed_tp + self.removed_fp != len(self.removed_coordinates):
            raise ValueError("coordinate transition counts are inconsistent")
        expected_rescued = baseline[0] == 0 and policy[0] >= 1
        expected_regressed = self.policy_metrics.f1 < self.baseline_metrics.f1
        if self.rescued is not expected_rescued or self.regressed is not expected_regressed:
            raise ValueError("dataset classification is inconsistent")
        if self.strong_regression is not _is_strong_regression_antigravity(self.baseline_metrics.f1, self.policy_metrics.f1):
            raise ValueError("strong regression classification is inconsistent")
        object.__setattr__(self, "baseline_confusion", baseline)
        object.__setattr__(self, "policy_confusion", policy)


@dataclass(frozen=True)
class FamilyDiagnosticAntigravity:
    policy: str
    family: str
    dataset_count: int
    affected_dataset_count: int
    baseline_confusion: tuple[int, int, int, int]
    policy_confusion: tuple[int, int, int, int]
    baseline_f1: float
    policy_f1: float
    macro_f1_delta: float


@dataclass(frozen=True)
class PolicyEvaluationAntigravity:
    policy: str
    dataset_rows: tuple[DatasetEvaluationRowAntigravity, ...]
    family_rows: tuple[FamilyDiagnosticAntigravity, ...]
    baseline_confusion: tuple[int, int, int, int]
    policy_confusion: tuple[int, int, int, int]
    baseline_metrics: MetricSetAntigravity
    policy_metrics: MetricSetAntigravity
    baseline_macro: MetricSetAntigravity
    policy_macro: MetricSetAntigravity
    micro_f1_delta: float
    macro_f1_delta: float
    added_tp: int
    added_fp: int
    removed_tp: int
    removed_fp: int
    rescued_datasets: tuple[str, ...]
    regressed_datasets: tuple[str, ...]
    strong_regressed_datasets: tuple[str, ...]
    unaffected_identical_count: int
    bootstrap: BootstrapResultAntigravity
    gates: Mapping[str, bool]
    status: str

    def __post_init__(self) -> None:
        if self.policy not in _POLICIES_ANTIGRAVITY:
            raise ValueError("policy evaluation identity is invalid")
        rows = tuple(self.dataset_rows)
        if not rows or tuple(row.dataset for row in rows) != tuple(sorted({row.dataset for row in rows})) or any(row.policy != self.policy for row in rows):
            raise ValueError("policy dataset rows are invalid")
        if any(type(value) is not float or not math.isfinite(value) for value in (self.micro_f1_delta, self.macro_f1_delta)):
            raise ValueError("policy deltas are invalid")
        for name in ("added_tp", "added_fp", "removed_tp", "removed_fp", "unaffected_identical_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} is invalid")
        for names in (self.rescued_datasets, self.regressed_datasets, self.strong_regressed_datasets):
            if tuple(names) != tuple(sorted(set(names))):
                raise ValueError("policy dataset classification list is invalid")
        if not isinstance(self.bootstrap, BootstrapResultAntigravity) or not isinstance(self.gates, Mapping):
            raise ValueError("policy evaluation evidence is invalid")
        gates = dict(self.gates)
        if not gates or any(type(key) is not str or type(value) is not bool for key, value in gates.items()):
            raise ValueError("policy gates are invalid")
        expected_status = "retrospective_improvement_signal" if all(gates.values()) else "no_retrospective_improvement_signal"
        if self.status != expected_status:
            raise ValueError("policy status is inconsistent")
        object.__setattr__(self, "dataset_rows", rows)
        object.__setattr__(self, "family_rows", tuple(self.family_rows))
        object.__setattr__(self, "gates", MappingProxyType(gates))


@dataclass(frozen=True)
class EvaluationResultAntigravity:
    publication_manifest_sha256: str
    prediction_sha256: tuple[tuple[str, str], ...]
    database_sha256: str
    label_identity_sha256: str
    label_rows: int
    label_query_calls: int
    policy_results: tuple[PolicyEvaluationAntigravity, ...]
    scientific_winner_selected: bool = False
    review_only: bool = True
    operational_alert_eligible: bool = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.publication_manifest_sha256, "publication manifest"),
            (self.database_sha256, "database"),
            (self.label_identity_sha256, "label identity"),
        ):
            _hex_antigravity(value, label)
        if tuple(policy for policy, _digest in self.prediction_sha256) != _POLICIES_ANTIGRAVITY:
            raise ValueError("evaluation prediction identities are invalid")
        for _policy, digest in self.prediction_sha256:
            _hex_antigravity(digest, "evaluation prediction")
        if tuple(result.policy for result in self.policy_results) != _POLICIES_ANTIGRAVITY:
            raise ValueError("evaluation policy results are invalid")
        if type(self.label_rows) is not int or self.label_rows < 1 or self.label_query_calls != FROZEN_CONFIG.dataset_count:
            raise ValueError("evaluation label accounting is invalid")
        if self.scientific_winner_selected is not False or self.review_only is not True or self.operational_alert_eligible is not False:
            raise ValueError("evaluation governance is invalid")


@dataclass(frozen=True)
class FrozenEvaluationAntigravity:
    evaluation_path: str
    artifact_map_sha256: str
    result: EvaluationResultAntigravity

    def __post_init__(self) -> None:
        if type(self.evaluation_path) is not str or "antigravity" not in Path(self.evaluation_path).name:
            raise ValueError("frozen evaluation path is invalid")
        _hex_antigravity(self.artifact_map_sha256, "evaluation artifact map")
        if not isinstance(self.result, EvaluationResultAntigravity):
            raise ValueError("frozen evaluation result is invalid")


def _is_strong_regression_antigravity(baseline_f1: float, policy_f1: float) -> bool:
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in (baseline_f1, policy_f1)):
        raise ValueError("F1 values are invalid")
    return float(baseline_f1) >= 0.8 and float(policy_f1) < float(baseline_f1)


def _normalize_label_rows_antigravity(rows: object, *, expected_count: int, dataset: str) -> tuple[int, ...]:
    if type(expected_count) is not int or expected_count < 1 or type(dataset) is not str or not dataset:
        raise ValueError("label row authority is invalid")
    try:
        values = tuple(rows)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("label rows are invalid") from error
    if len(values) != expected_count:
        raise ValueError(f"TEST label row count differs: {dataset}")
    labels: list[int] = []
    for expected_index, row in enumerate(values):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError(f"TEST label row schema differs: {dataset}")
        index, raw_label = row
        if type(index) is not int or index != expected_index:
            raise ValueError(f"TEST instance identity is not contiguous: {dataset}")
        if type(raw_label) is int and raw_label in (0, 1):
            label = raw_label
        elif type(raw_label) is str and raw_label in ("0", "1"):
            label = int(raw_label)
        else:
            raise ValueError(f"TEST label is not binary: {dataset}")
        labels.append(label)
    return tuple(labels)


def _parse_prediction_line_antigravity(payload: bytes, *, policy: str) -> Mapping[str, object]:
    parsed = parse_canonical_json_bytes_antigravity(payload)
    if not isinstance(parsed, dict) or set(parsed) != _PREDICTION_ROW_KEYS_ANTIGRAVITY:
        raise ValueError("evaluation prediction row schema differs")
    if parsed.get("policy") != policy or parsed.get("contains_test_labels") is not False or parsed.get("contains_evaluation_metrics") is not False:
        raise ValueError("evaluation prediction governance differs")
    dataset = parsed.get("dataset")
    n_test = parsed.get("n_test")
    if type(dataset) is not str or not dataset or type(n_test) is not int or n_test < 1:
        raise ValueError("evaluation prediction identity differs")
    parsed["final_hard"] = _indices_antigravity(parsed.get("final_hard"), n_test=n_test, label="final Hard")
    return MappingProxyType(parsed)


def _publication_rows_antigravity(publication: FrozenPublicationAntigravity) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    if not isinstance(publication, FrozenPublicationAntigravity):
        raise ValueError("evaluation publication is invalid")
    root = Path(publication.publication_path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("evaluation publication path is invalid")
    expected_hashes = dict(publication.prediction_sha256)
    result: dict[str, tuple[Mapping[str, object], ...]] = {}
    identities: tuple[tuple[str, int], ...] | None = None
    for policy in _POLICIES_ANTIGRAVITY:
        payload, digest = read_pinned_regular_bytes_antigravity(root / f"prediction_{policy}_antigravity.jsonl")
        if digest != expected_hashes[policy] or not payload.endswith(b"\n"):
            raise ValueError("evaluation prediction SHA256 differs")
        rows = tuple(_parse_prediction_line_antigravity(line, policy=policy) for line in payload.splitlines(keepends=True))
        current = tuple((row["dataset"], row["n_test"]) for row in rows)
        if len(rows) != FROZEN_CONFIG.dataset_count or current != tuple(sorted(set(current))):
            raise ValueError("evaluation prediction population differs")
        if identities is None:
            identities = current
        elif current != identities:
            raise ValueError("evaluation policy TEST identities differ")
        result[policy] = rows
    return MappingProxyType(result)


def _open_label_connection_antigravity():
    """Import the sole label-capable connection only after publication replay."""

    from experiments.exp417_causal_multihorizon_shadow.inputs import verified_read_only_connection

    return verified_read_only_connection(Path(FROZEN_CONFIG.database_path), FROZEN_CONFIG.database_sha256)


def _source_identity_antigravity(source: object) -> str:
    """Hash the array-free source facts needed on both sides of label access."""

    datasets = tuple(getattr(source, "datasets", ()))
    if not datasets or datasets != tuple(sorted(set(datasets))):
        raise ValueError("source dataset identity is invalid")
    label_access = getattr(source, "label_access_count", None)
    if type(label_access) is not int or label_access != 0:
        raise ValueError("source label access is nonzero")
    value = {
        "datasets": list(datasets),
        "affected_datasets": list(getattr(source, "affected_datasets", ())),
        "source_hashes": dict(getattr(source, "source_hashes", {})),
        "affected_datasets_sha256": getattr(source, "affected_datasets_sha256", None),
        "hard_coordinates_exp291": [list(item) for item in sorted(getattr(source, "hard_coordinates_exp291", ()))],
        "hard_coordinates_exp404": [list(item) for item in sorted(getattr(source, "hard_coordinates_exp404", ()))],
        "test_row_count_by_dataset": dict(getattr(source, "test_row_count_by_dataset", {})),
        "target_length_by_dataset": dict(getattr(source, "target_length_by_dataset", {})),
        "routing_algebra": list(getattr(source, "routing_algebra", ())),
        "label_access_count": label_access,
    }
    return sha256(canonical_json_bytes_antigravity(value)).hexdigest()


def load_frozen_test_labels_antigravity(publication: FrozenPublicationAntigravity) -> FrozenLabelsAntigravity:
    """Open one DB snapshot only after two complete publication inspections."""

    if not isinstance(publication, FrozenPublicationAntigravity) or publication.label_access_count != 0:
        raise ValueError("label loader requires one label-free frozen publication")
    first = inspect_publication_antigravity(publication.attempt_path)
    second = inspect_publication_antigravity(publication.attempt_path)
    if first != publication or second != publication:
        raise ValueError("publication changed before TEST-label import")
    rows_by_policy = _publication_rows_antigravity(second)
    authority_rows = rows_by_policy[_POLICIES_ANTIGRAVITY[0]]
    datasets = tuple(row["dataset"] for row in authority_rows)
    n_test_by_dataset = {row["dataset"]: row["n_test"] for row in authority_rows}
    source_before = verify_frozen_sources_antigravity()
    if getattr(source_before, "label_access_count", None) != 0 or tuple(getattr(source_before, "datasets", ())) != datasets:
        raise ValueError("source authority differs before TEST-label import")
    source_identity = _source_identity_antigravity(source_before)

    labels: dict[str, tuple[int, ...]] = {}
    with _open_label_connection_antigravity() as connection:
        quick_check = connection.execute("PRAGMA quick_check").fetchall()
        if quick_check != [("ok",)]:
            raise ValueError("TEST-label database quick_check failed")
        for dataset in datasets:
            raw = connection.execute(LABEL_SQL_ANTIGRAVITY, (dataset,)).fetchall()
            labels[dataset] = _normalize_label_rows_antigravity(
                raw,
                expected_count=n_test_by_dataset[dataset],
                dataset=dataset,
            )

    source_after = verify_frozen_sources_antigravity()
    final = inspect_publication_antigravity(publication.attempt_path)
    if _source_identity_antigravity(source_after) != source_identity or final != publication:
        raise ValueError("publication or source changed after TEST-label read")
    label_identity = sha256(canonical_json_bytes_antigravity({dataset: list(labels[dataset]) for dataset in datasets})).hexdigest()
    return FrozenLabelsAntigravity(
        datasets=datasets,
        labels_by_dataset=labels,
        publication_manifest_sha256=publication.manifest_sha256,
        prediction_sha256=publication.prediction_sha256,
        database_sha256=FROZEN_CONFIG.database_sha256,
        label_identity_sha256=label_identity,
        descriptor_bound_connection_count=1,
        label_query_calls=len(datasets),
        label_rows=sum(len(values) for values in labels.values()),
        pre_freeze_label_access_count=publication.label_access_count,
    )


def _confusion_for_selection_antigravity(labels: tuple[int, ...], selected: Iterable[int]) -> tuple[int, int, int, int]:
    selected_set = set(selected)
    if any(type(index) is not int or index < 0 or index >= len(labels) for index in selected_set):
        raise ValueError("prediction coordinate is outside frozen TEST identity")
    tp = sum(labels[index] for index in selected_set)
    fp = len(selected_set) - tp
    positives = sum(labels)
    fn = positives - tp
    tn = len(labels) - tp - fp - fn
    return int(tp), int(fp), int(fn), int(tn)


def _mean_metric_antigravity(rows: Sequence[DatasetEvaluationRowAntigravity], *, policy: bool) -> MetricSetAntigravity:
    metrics = [row.policy_metrics if policy else row.baseline_metrics for row in rows]
    return MetricSetAntigravity(
        precision=float(np.asarray([item.precision for item in metrics], dtype=np.float64).mean(dtype=np.float64)),
        recall=float(np.asarray([item.recall for item in metrics], dtype=np.float64).mean(dtype=np.float64)),
        f1=float(np.asarray([item.f1 for item in metrics], dtype=np.float64).mean(dtype=np.float64)),
        false_positive_rate=float(np.asarray([item.false_positive_rate for item in metrics], dtype=np.float64).mean(dtype=np.float64)),
    )


def _success_gates_antigravity(
    *,
    micro_delta: float,
    macro_f1_delta: float,
    bootstrap: BootstrapResultAntigravity,
    policy_fp: int,
    rescued_count: int,
    strong_regression_count: int,
    unaffected_identical_count: int,
    pre_freeze_label_access_count: int,
    error_count: int,
    nonfinite_count: int,
) -> Mapping[str, bool]:
    for value in (micro_delta, macro_f1_delta):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("evaluation delta is invalid")
    gates = {
        "micro_f1_delta_gate": micro_delta > 0.0,
        "macro_f1_delta_gate": macro_f1_delta > 0.0,
        "micro_bootstrap_gate": bootstrap.micro_delta_ci[0] > 0.0,
        "macro_bootstrap_gate": bootstrap.macro_f1_delta_ci[0] > 0.0,
        "false_alarm_gate": policy_fp <= FROZEN_CONFIG.baseline_fp,
        "rescued_dataset_gate": rescued_count >= 1,
        "strong_regression_gate": strong_regression_count == 0,
        "unaffected_invariance_gate": unaffected_identical_count == FROZEN_CONFIG.dataset_count - FROZEN_CONFIG.affected_dataset_count,
        "pre_freeze_label_access_gate": pre_freeze_label_access_count == 0,
        "error_gate": error_count == 0,
        "finite_output_gate": nonfinite_count == 0,
    }
    return MappingProxyType(gates)


def _family_rows_antigravity(policy: str, rows: Sequence[DatasetEvaluationRowAntigravity]) -> tuple[FamilyDiagnosticAntigravity, ...]:
    grouped: dict[str, list[DatasetEvaluationRowAntigravity]] = defaultdict(list)
    for row in rows:
        grouped[row.family].append(row)
    result: list[FamilyDiagnosticAntigravity] = []
    for family in sorted(grouped):
        items = grouped[family]
        baseline = tuple(sum(row.baseline_confusion[index] for row in items) for index in range(4))
        current = tuple(sum(row.policy_confusion[index] for row in items) for index in range(4))
        macro_delta = float(np.asarray([row.policy_metrics.f1 - row.baseline_metrics.f1 for row in items], dtype=np.float64).mean(dtype=np.float64))
        result.append(FamilyDiagnosticAntigravity(
            policy,
            family,
            len(items),
            sum(row.affected for row in items),
            baseline,
            current,
            project_metrics_antigravity(baseline).f1,
            project_metrics_antigravity(current).f1,
            macro_delta,
        ))
    return tuple(result)


def _compute_policy_evaluation_for_test_antigravity(
    policy: str,
    baseline_by_dataset: Mapping[str, Sequence[int]],
    policy_by_dataset: Mapping[str, Sequence[int]],
    labels_by_dataset: Mapping[str, tuple[int, ...]],
    affected_datasets: set[str] | frozenset[str],
    *,
    bootstrap_runner: Callable[[np.ndarray], BootstrapResultAntigravity],
    maximum_fp: int = 303,
    expected_unaffected_count: int = 1098,
    pre_freeze_label_access_count: int = 0,
) -> PolicyEvaluationAntigravity:
    """Pure injectable-population calculator used by tests and production."""

    if policy not in _POLICIES_ANTIGRAVITY:
        raise ValueError("evaluation policy is invalid")
    datasets = tuple(sorted(labels_by_dataset))
    if not datasets or set(baseline_by_dataset) != set(datasets) or set(policy_by_dataset) != set(datasets):
        raise ValueError("evaluation dataset identities differ")
    affected = frozenset(affected_datasets)
    if not affected.issubset(datasets):
        raise ValueError("affected dataset identity differs")
    dataset_rows: list[DatasetEvaluationRowAntigravity] = []
    bootstrap_rows: list[tuple[int, int, int, int, int, int, int, int, float]] = []
    for dataset in datasets:
        labels = tuple(labels_by_dataset[dataset])
        if not labels or any(type(value) is not int or value not in (0, 1) for value in labels):
            raise ValueError("evaluation labels are invalid")
        baseline_indices = _indices_antigravity(baseline_by_dataset[dataset], n_test=len(labels), label="baseline coordinates")
        current_indices = _indices_antigravity(policy_by_dataset[dataset], n_test=len(labels), label="policy coordinates")
        baseline = _confusion_for_selection_antigravity(labels, baseline_indices)
        current = _confusion_for_selection_antigravity(labels, current_indices)
        baseline_metrics = project_metrics_antigravity(baseline)
        current_metrics = project_metrics_antigravity(current)
        added = tuple(sorted(set(current_indices) - set(baseline_indices)))
        removed = tuple(sorted(set(baseline_indices) - set(current_indices)))
        added_tp = sum(labels[index] for index in added)
        removed_tp = sum(labels[index] for index in removed)
        row = DatasetEvaluationRowAntigravity(
            policy=policy,
            dataset=dataset,
            family=parse_dataset_family_antigravity(dataset),
            affected=dataset in affected,
            n_test=len(labels),
            baseline_confusion=baseline,
            policy_confusion=current,
            baseline_metrics=baseline_metrics,
            policy_metrics=current_metrics,
            added_coordinates=added,
            removed_coordinates=removed,
            added_tp=added_tp,
            added_fp=len(added) - added_tp,
            removed_tp=removed_tp,
            removed_fp=len(removed) - removed_tp,
            rescued=baseline[0] == 0 and current[0] >= 1,
            regressed=current_metrics.f1 < baseline_metrics.f1,
            strong_regression=_is_strong_regression_antigravity(baseline_metrics.f1, current_metrics.f1),
        )
        dataset_rows.append(row)
        bootstrap_rows.append((*baseline, *current, current_metrics.f1 - baseline_metrics.f1))
    baseline_confusion = tuple(sum(row.baseline_confusion[index] for row in dataset_rows) for index in range(4))
    policy_confusion = tuple(sum(row.policy_confusion[index] for row in dataset_rows) for index in range(4))
    baseline_metrics = project_metrics_antigravity(baseline_confusion)
    policy_metrics = project_metrics_antigravity(policy_confusion)
    baseline_macro = _mean_metric_antigravity(dataset_rows, policy=False)
    policy_macro = _mean_metric_antigravity(dataset_rows, policy=True)
    micro_delta = float(policy_metrics.f1 - baseline_metrics.f1)
    macro_delta = float(policy_macro.f1 - baseline_macro.f1)
    bootstrap = bootstrap_runner(np.asarray(bootstrap_rows, dtype=np.float64))
    rescued = tuple(row.dataset for row in dataset_rows if row.rescued)
    regressed = tuple(row.dataset for row in dataset_rows if row.regressed)
    strong = tuple(row.dataset for row in dataset_rows if row.strong_regression)
    unaffected_identical = sum(
        not row.affected and not row.added_coordinates and not row.removed_coordinates
        for row in dataset_rows
    )
    # Production freezes the maximum at 303.  The narrow test seam may lower
    # it, but it must never weaken the production false-alarm limit.
    if type(maximum_fp) is not int or maximum_fp < 0 or maximum_fp > FROZEN_CONFIG.baseline_fp:
        raise ValueError("maximum FP test seam is invalid")
    gates = dict(_success_gates_antigravity(
        micro_delta=micro_delta,
        macro_f1_delta=macro_delta,
        bootstrap=bootstrap,
        policy_fp=policy_confusion[1],
        rescued_count=len(rescued),
        strong_regression_count=len(strong),
        unaffected_identical_count=unaffected_identical,
        pre_freeze_label_access_count=pre_freeze_label_access_count,
        error_count=0,
        nonfinite_count=0,
    ))
    gates["false_alarm_gate"] = policy_confusion[1] <= maximum_fp
    gates["unaffected_invariance_gate"] = unaffected_identical == expected_unaffected_count
    return PolicyEvaluationAntigravity(
        policy=policy,
        dataset_rows=tuple(dataset_rows),
        family_rows=_family_rows_antigravity(policy, dataset_rows),
        baseline_confusion=baseline_confusion,
        policy_confusion=policy_confusion,
        baseline_metrics=baseline_metrics,
        policy_metrics=policy_metrics,
        baseline_macro=baseline_macro,
        policy_macro=policy_macro,
        micro_f1_delta=micro_delta,
        macro_f1_delta=macro_delta,
        added_tp=sum(row.added_tp for row in dataset_rows),
        added_fp=sum(row.added_fp for row in dataset_rows),
        removed_tp=sum(row.removed_tp for row in dataset_rows),
        removed_fp=sum(row.removed_fp for row in dataset_rows),
        rescued_datasets=rescued,
        regressed_datasets=regressed,
        strong_regressed_datasets=strong,
        unaffected_identical_count=unaffected_identical,
        bootstrap=bootstrap,
        gates=gates,
        status="retrospective_improvement_signal" if all(gates.values()) else "no_retrospective_improvement_signal",
    )


def compute_policy_evaluation_antigravity(
    publication: FrozenPublicationAntigravity,
    labels: FrozenLabelsAntigravity,
) -> EvaluationResultAntigravity:
    """Evaluate all three frozen policies independently against one label snapshot."""

    if not isinstance(publication, FrozenPublicationAntigravity) or not isinstance(labels, FrozenLabelsAntigravity):
        raise ValueError("evaluation inputs are invalid")
    if (
        labels.publication_manifest_sha256 != publication.manifest_sha256
        or labels.prediction_sha256 != publication.prediction_sha256
        or labels.pre_freeze_label_access_count != 0
    ):
        raise ValueError("evaluation label/publication identity differs")
    rows_by_policy = _publication_rows_antigravity(publication)
    source = verify_frozen_sources_antigravity()
    datasets = labels.datasets
    if tuple(getattr(source, "datasets", ())) != datasets or getattr(source, "label_access_count", None) != 0:
        raise ValueError("evaluation source identity differs")
    hard_coordinates = frozenset(getattr(source, "hard_coordinates_exp404", ()))
    if hard_coordinates != frozenset(getattr(source, "hard_coordinates_exp291", ())):
        raise ValueError("evaluation baseline Hard authorities differ")
    affected = frozenset(getattr(source, "affected_datasets", ()))
    if len(affected) != FROZEN_CONFIG.affected_dataset_count:
        raise ValueError("evaluation affected inventory differs")
    baseline_by_dataset: dict[str, list[int]] = {dataset: [] for dataset in datasets}
    for dataset, index in hard_coordinates:
        if dataset not in baseline_by_dataset:
            raise ValueError("baseline coordinate dataset is absent")
        baseline_by_dataset[dataset].append(index)
    baseline = {dataset: tuple(sorted(indices)) for dataset, indices in baseline_by_dataset.items()}
    results: list[PolicyEvaluationAntigravity] = []
    for policy in _POLICIES_ANTIGRAVITY:
        rows = rows_by_policy[policy]
        policy_coordinates = {row["dataset"]: tuple(row["final_hard"]) for row in rows}
        if policy == _POLICIES_ANTIGRAVITY[0] and policy_coordinates != baseline:
            raise ValueError("Policy A coordinates do not reproduce the frozen baseline")
        result = _compute_policy_evaluation_for_test_antigravity(
            policy,
            baseline,
            policy_coordinates,
            labels.labels_by_dataset,
            affected,
            bootstrap_runner=paired_dataset_bootstrap_antigravity,
            maximum_fp=FROZEN_CONFIG.baseline_fp,
            expected_unaffected_count=FROZEN_CONFIG.dataset_count - FROZEN_CONFIG.affected_dataset_count,
            pre_freeze_label_access_count=labels.pre_freeze_label_access_count,
        )
        if result.baseline_confusion != _BASELINE_CONFUSION_ANTIGRAVITY or result.baseline_metrics.f1 != _BASELINE_F1_ANTIGRAVITY:
            raise ValueError("Exp404/Exp291 baseline reproduction failed")
        if policy == _POLICIES_ANTIGRAVITY[0] and result.policy_confusion != _BASELINE_CONFUSION_ANTIGRAVITY:
            raise ValueError("Policy A confusion does not reproduce the baseline")
        if len(result.dataset_rows) != FROZEN_CONFIG.dataset_count:
            raise ValueError("evaluation dataset population differs")
        results.append(result)
    return EvaluationResultAntigravity(
        publication_manifest_sha256=publication.manifest_sha256,
        prediction_sha256=publication.prediction_sha256,
        database_sha256=labels.database_sha256,
        label_identity_sha256=labels.label_identity_sha256,
        label_rows=labels.label_rows,
        label_query_calls=labels.label_query_calls,
        policy_results=tuple(results),
    )


def _dataset_payload_antigravity(row: DatasetEvaluationRowAntigravity) -> dict[str, object]:
    return {
        "policy": row.policy,
        "dataset": row.dataset,
        "family": row.family,
        "affected": row.affected,
        "n_test": row.n_test,
        "baseline_confusion": list(row.baseline_confusion),
        "policy_confusion": list(row.policy_confusion),
        "baseline_metrics": row.baseline_metrics.to_payload_antigravity(),
        "policy_metrics": row.policy_metrics.to_payload_antigravity(),
        "f1_delta": row.policy_metrics.f1 - row.baseline_metrics.f1,
        "added_tp": row.added_tp,
        "added_fp": row.added_fp,
        "removed_tp": row.removed_tp,
        "removed_fp": row.removed_fp,
        "rescued": row.rescued,
        "regressed": row.regressed,
        "strong_regression": row.strong_regression,
    }


def _transition_payload_antigravity(row: DatasetEvaluationRowAntigravity) -> dict[str, object]:
    return {
        "policy": row.policy,
        "dataset": row.dataset,
        "affected": row.affected,
        "added_coordinates": list(row.added_coordinates),
        "removed_coordinates": list(row.removed_coordinates),
        "added_tp": row.added_tp,
        "added_fp": row.added_fp,
        "removed_tp": row.removed_tp,
        "removed_fp": row.removed_fp,
    }


def _family_payload_antigravity(row: FamilyDiagnosticAntigravity) -> dict[str, object]:
    return {
        "policy": row.policy,
        "family": row.family,
        "dataset_count": row.dataset_count,
        "affected_dataset_count": row.affected_dataset_count,
        "baseline_confusion": list(row.baseline_confusion),
        "policy_confusion": list(row.policy_confusion),
        "baseline_f1": row.baseline_f1,
        "policy_f1": row.policy_f1,
        "macro_f1_delta": row.macro_f1_delta,
    }


def _subset_summary_antigravity(
    rows: Sequence[DatasetEvaluationRowAntigravity], *, affected: bool,
) -> dict[str, object]:
    selected = tuple(row for row in rows if row.affected is affected)
    if not selected:
        raise ValueError("evaluation diagnostic subset is empty")
    baseline_confusion = tuple(sum(row.baseline_confusion[index] for row in selected) for index in range(4))
    policy_confusion = tuple(sum(row.policy_confusion[index] for row in selected) for index in range(4))
    baseline_metrics = project_metrics_antigravity(baseline_confusion)
    policy_metrics = project_metrics_antigravity(policy_confusion)
    baseline_macro = _mean_metric_antigravity(selected, policy=False)
    policy_macro = _mean_metric_antigravity(selected, policy=True)
    return {
        "affected": affected,
        "dataset_count": len(selected),
        "baseline_confusion": list(baseline_confusion),
        "policy_confusion": list(policy_confusion),
        "baseline_metrics": baseline_metrics.to_payload_antigravity(),
        "policy_metrics": policy_metrics.to_payload_antigravity(),
        "baseline_macro": baseline_macro.to_payload_antigravity(),
        "policy_macro": policy_macro.to_payload_antigravity(),
        "micro_f1_delta": policy_metrics.f1 - baseline_metrics.f1,
        "macro_f1_delta": policy_macro.f1 - baseline_macro.f1,
        "added_tp": sum(row.added_tp for row in selected),
        "added_fp": sum(row.added_fp for row in selected),
        "removed_tp": sum(row.removed_tp for row in selected),
        "removed_fp": sum(row.removed_fp for row in selected),
        "rescued_dataset_count": sum(row.rescued for row in selected),
        "regressed_dataset_count": sum(row.regressed for row in selected),
        "strong_regression_count": sum(row.strong_regression for row in selected),
        "coordinate_identical_count": sum(not row.added_coordinates and not row.removed_coordinates for row in selected),
    }


def _policy_summary_antigravity(result: PolicyEvaluationAntigravity) -> dict[str, object]:
    return {
        "policy": result.policy,
        "status": result.status,
        "baseline_confusion": list(result.baseline_confusion),
        "policy_confusion": list(result.policy_confusion),
        "baseline_metrics": result.baseline_metrics.to_payload_antigravity(),
        "policy_metrics": result.policy_metrics.to_payload_antigravity(),
        "baseline_macro": result.baseline_macro.to_payload_antigravity(),
        "policy_macro": result.policy_macro.to_payload_antigravity(),
        "micro_f1_delta": result.micro_f1_delta,
        "macro_f1_delta": result.macro_f1_delta,
        "added_tp": result.added_tp,
        "added_fp": result.added_fp,
        "removed_tp": result.removed_tp,
        "removed_fp": result.removed_fp,
        "rescued_dataset_count": len(result.rescued_datasets),
        "regressed_dataset_count": len(result.regressed_datasets),
        "strong_regression_count": len(result.strong_regressed_datasets),
        "unaffected_identical_count": result.unaffected_identical_count,
        "affected_diagnostic": _subset_summary_antigravity(result.dataset_rows, affected=True),
        "unaffected_diagnostic": _subset_summary_antigravity(result.dataset_rows, affected=False),
        "pre_freeze_label_access_count": 0,
        "error_count": 0,
        "nonfinite_count": 0,
        "gates": dict(result.gates),
    }


def _artifact_map_bytes_antigravity(payloads: Mapping[str, bytes]) -> bytes:
    expected = set(EVALUATION_NAMES_ANTIGRAVITY) - {"artifact_sha256_antigravity.json"}
    if set(payloads) != expected or any(type(payload) is not bytes for payload in payloads.values()):
        raise ValueError("evaluation artifact-map inputs are invalid")
    return canonical_json_bytes_antigravity({
        "schema": "exp443.evaluation-artifact-map.v1",
        "files": {name: sha256(payloads[name]).hexdigest() for name in EVALUATION_NAMES_ANTIGRAVITY[:-1]},
        "self_excluded": "artifact_sha256_antigravity.json",
    })


def _evaluation_payloads_antigravity(result: EvaluationResultAntigravity) -> Mapping[str, bytes]:
    dataset_rows = [row for policy in result.policy_results for row in policy.dataset_rows]
    family_rows = [row for policy in result.policy_results for row in policy.family_rows]
    metrics = canonical_jsonl_bytes_antigravity(_dataset_payload_antigravity(row) for row in dataset_rows)
    transitions = canonical_jsonl_bytes_antigravity(_transition_payload_antigravity(row) for row in dataset_rows)
    rescued = canonical_jsonl_bytes_antigravity(
        {"policy": policy.policy, "dataset": dataset}
        for policy in result.policy_results for dataset in policy.rescued_datasets
    )
    regressed = canonical_jsonl_bytes_antigravity(
        {
            "policy": policy.policy,
            "dataset": dataset,
            "strong_regression": dataset in policy.strong_regressed_datasets,
        }
        for policy in result.policy_results for dataset in policy.regressed_datasets
    )
    families = canonical_jsonl_bytes_antigravity(_family_payload_antigravity(row) for row in family_rows)
    confusion = canonical_json_bytes_antigravity({
        "schema": "exp443.confusion.v1",
        "order": ["tp", "fp", "fn", "tn"],
        "policies": {
            policy.policy: {
                "baseline": list(policy.baseline_confusion),
                "policy": list(policy.policy_confusion),
            }
            for policy in result.policy_results
        },
    })
    bootstrap = canonical_json_bytes_antigravity({
        "schema": "exp443.bootstrap.v1",
        "policies": {policy.policy: policy.bootstrap.to_payload_antigravity() for policy in result.policy_results},
        **governance_payload_antigravity(),
    })
    leaves = {
        "dataset_metrics_antigravity.jsonl": metrics,
        "coordinate_transitions_antigravity.jsonl": transitions,
        "rescued_datasets_antigravity.jsonl": rescued,
        "regressed_datasets_antigravity.jsonl": regressed,
        "family_diagnostics_antigravity.jsonl": families,
        "confusion_antigravity.json": confusion,
        "bootstrap_antigravity.json": bootstrap,
    }
    evaluation = canonical_json_bytes_antigravity({
        "schema": "exp443.evaluation.v1",
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "run_identity": FROZEN_CONFIG.run_identity,
        "publication_manifest_sha256": result.publication_manifest_sha256,
        "prediction_sha256": dict(result.prediction_sha256),
        "database_sha256": result.database_sha256,
        "label_identity_sha256": result.label_identity_sha256,
        "label_rows": result.label_rows,
        "label_query_calls": result.label_query_calls,
        "dataset_count": FROZEN_CONFIG.dataset_count,
        "policy_summaries": [_policy_summary_antigravity(policy) for policy in result.policy_results],
        "artifact_sha256": {name: sha256(payload).hexdigest() for name, payload in leaves.items()},
        **governance_payload_antigravity(),
    })
    decision = canonical_json_bytes_antigravity({
        "schema": "exp443.decision.v1",
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "policy_status": {policy.policy: policy.status for policy in result.policy_results},
        "evaluation_sha256": sha256(evaluation).hexdigest(),
        "bootstrap_sha256": sha256(bootstrap).hexdigest(),
        "scientific_winner_selected": False,
        "review_only": True,
        "operational_alert_eligible": False,
        "test_informed_reselection_allowed": False,
    })
    payloads: dict[str, bytes] = {
        "evaluation_antigravity.json": evaluation,
        **leaves,
        "decision_antigravity.json": decision,
    }
    if set(payloads) != set(EVALUATION_NAMES_ANTIGRAVITY[:-1]):
        raise ValueError("evaluation inventory construction differs")
    ordered = {name: payloads[name] for name in EVALUATION_NAMES_ANTIGRAVITY[:-1]}
    ordered["artifact_sha256_antigravity.json"] = _artifact_map_bytes_antigravity(ordered)
    return MappingProxyType(ordered)


def _read_evaluation_inventory_antigravity(root: Path) -> Mapping[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("evaluation root is invalid")
    entries = {entry.name: entry for entry in root.iterdir()}
    if set(entries) != set(EVALUATION_NAMES_ANTIGRAVITY):
        raise ValueError("evaluation inventory differs")
    return MappingProxyType({name: read_pinned_regular_bytes_antigravity(entries[name])[0] for name in EVALUATION_NAMES_ANTIGRAVITY})


def _verify_artifact_map_antigravity(payloads: Mapping[str, bytes]) -> None:
    artifact = parse_canonical_json_bytes_antigravity(payloads["artifact_sha256_antigravity.json"])
    if not isinstance(artifact, dict) or set(artifact) != {"schema", "files", "self_excluded"}:
        raise ValueError("evaluation artifact map is invalid")
    if artifact["schema"] != "exp443.evaluation-artifact-map.v1" or artifact["self_excluded"] != "artifact_sha256_antigravity.json":
        raise ValueError("evaluation artifact map identity differs")
    files = artifact["files"]
    if not isinstance(files, dict) or set(files) != set(EVALUATION_NAMES_ANTIGRAVITY[:-1]):
        raise ValueError("evaluation artifact map inventory differs")
    for name in EVALUATION_NAMES_ANTIGRAVITY[:-1]:
        if files[name] != sha256(payloads[name]).hexdigest():
            raise ValueError("evaluation artifact hash differs")


def _evaluation_result_from_payloads_antigravity(
    payloads: Mapping[str, bytes], *, _enforce_official_antigravity: bool = True,
) -> EvaluationResultAntigravity:
    """Replay aggregate authority without reopening TEST labels."""

    _verify_artifact_map_antigravity(payloads)
    evaluation = parse_canonical_json_bytes_antigravity(payloads["evaluation_antigravity.json"])
    decision = parse_canonical_json_bytes_antigravity(payloads["decision_antigravity.json"])
    bootstrap_payload = parse_canonical_json_bytes_antigravity(payloads["bootstrap_antigravity.json"])
    if not isinstance(evaluation, dict) or not isinstance(decision, dict) or not isinstance(bootstrap_payload, dict):
        raise ValueError("evaluation JSON payload is invalid")
    if evaluation.get("schema") != "exp443.evaluation.v1" or evaluation.get("experiment_id") != FROZEN_CONFIG.experiment_id:
        raise ValueError("evaluation identity differs")
    if decision.get("schema") != "exp443.decision.v1" or decision.get("evaluation_sha256") != sha256(payloads["evaluation_antigravity.json"]).hexdigest():
        raise ValueError("evaluation decision identity differs")
    if decision.get("bootstrap_sha256") != sha256(payloads["bootstrap_antigravity.json"]).hexdigest():
        raise ValueError("evaluation bootstrap decision identity differs")
    for name, expected in governance_payload_antigravity().items():
        if evaluation.get(name) != expected or bootstrap_payload.get(name) != expected:
            raise ValueError("evaluation governance differs")
    if (
        decision.get("scientific_winner_selected") is not False
        or decision.get("review_only") is not True
        or decision.get("operational_alert_eligible") is not False
        or decision.get("test_informed_reselection_allowed") is not False
    ):
        raise ValueError("evaluation decision governance differs")
    # A byte-for-byte replay is authoritative because the artifact map binds
    # all row-level TEST-derived evidence.  Task 13 additionally reconstructs
    # confusion/bootstrap in a private temporary destination.
    summaries = evaluation.get("policy_summaries")
    bootstrap_policies = bootstrap_payload.get("policies")
    if not isinstance(summaries, list) or not isinstance(bootstrap_policies, dict):
        raise ValueError("evaluation policy evidence is invalid")
    # Full dataclass reconstruction deliberately consumes the canonical rows
    # so aggregate self-reports cannot stand alone.
    metric_lines = payloads["dataset_metrics_antigravity.jsonl"].splitlines(keepends=True)
    transition_lines = payloads["coordinate_transitions_antigravity.jsonl"].splitlines(keepends=True)
    if len(metric_lines) != FROZEN_CONFIG.dataset_count * len(_POLICIES_ANTIGRAVITY) or len(transition_lines) != len(metric_lines):
        raise ValueError("evaluation dataset evidence population differs")
    metric_by_key: dict[tuple[str, str], dict[str, object]] = {}
    transition_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for line, target in ((line, metric_by_key) for line in metric_lines):
        row = parse_canonical_json_bytes_antigravity(line)
        if not isinstance(row, dict) or type(row.get("policy")) is not str or type(row.get("dataset")) is not str:
            raise ValueError("evaluation metric row is invalid")
        key = (row["policy"], row["dataset"])
        if key in target:
            raise ValueError("evaluation metric row is duplicated")
        target[key] = row
    for line in transition_lines:
        row = parse_canonical_json_bytes_antigravity(line)
        if not isinstance(row, dict) or type(row.get("policy")) is not str or type(row.get("dataset")) is not str:
            raise ValueError("evaluation transition row is invalid")
        key = (row["policy"], row["dataset"])
        if key in transition_by_key:
            raise ValueError("evaluation transition row is duplicated")
        transition_by_key[key] = row
    if set(metric_by_key) != set(transition_by_key):
        raise ValueError("evaluation metric/transition identities differ")

    policy_results: list[PolicyEvaluationAntigravity] = []
    summary_by_policy = {item.get("policy"): item for item in summaries if isinstance(item, dict)}
    for policy in _POLICIES_ANTIGRAVITY:
        rows: list[DatasetEvaluationRowAntigravity] = []
        for key in sorted(item for item in metric_by_key if item[0] == policy):
            metric = metric_by_key[key]
            transition = transition_by_key[key]
            if any(metric.get(name) != transition.get(name) for name in ("policy", "dataset", "affected", "added_tp", "added_fp", "removed_tp", "removed_fp")):
                raise ValueError("evaluation transition facts differ")
            baseline_metrics_raw = metric.get("baseline_metrics")
            policy_metrics_raw = metric.get("policy_metrics")
            if not isinstance(baseline_metrics_raw, dict) or not isinstance(policy_metrics_raw, dict):
                raise ValueError("evaluation metric set is invalid")
            def metric_set(raw: Mapping[str, object]) -> MetricSetAntigravity:
                if set(raw) != {"precision", "recall", "f1", "false_positive_rate"}:
                    raise ValueError("evaluation metric set schema differs")
                return MetricSetAntigravity(*(float(raw[name]) for name in ("precision", "recall", "f1", "false_positive_rate")))
            rows.append(DatasetEvaluationRowAntigravity(
                policy=policy,
                dataset=str(metric["dataset"]),
                family=str(metric["family"]),
                affected=bool(metric["affected"]),
                n_test=int(metric["n_test"]),
                baseline_confusion=_confusion_antigravity(metric["baseline_confusion"], "baseline confusion"),
                policy_confusion=_confusion_antigravity(metric["policy_confusion"], "policy confusion"),
                baseline_metrics=metric_set(baseline_metrics_raw),
                policy_metrics=metric_set(policy_metrics_raw),
                added_coordinates=tuple(transition["added_coordinates"]),
                removed_coordinates=tuple(transition["removed_coordinates"]),
                added_tp=int(metric["added_tp"]),
                added_fp=int(metric["added_fp"]),
                removed_tp=int(metric["removed_tp"]),
                removed_fp=int(metric["removed_fp"]),
                rescued=bool(metric["rescued"]),
                regressed=bool(metric["regressed"]),
                strong_regression=bool(metric["strong_regression"]),
            ))
        if len(rows) != FROZEN_CONFIG.dataset_count:
            raise ValueError("evaluation policy dataset population differs")
        summary = summary_by_policy.get(policy)
        boot = bootstrap_policies.get(policy)
        if not isinstance(summary, dict) or not isinstance(boot, dict):
            raise ValueError("evaluation policy summary is invalid")
        bootstrap = BootstrapResultAntigravity(
            micro_delta_ci=tuple(boot["micro_delta_ci"]),
            macro_f1_delta_ci=tuple(boot["macro_f1_delta_ci"]),
            seed=boot["seed"],
            samples=boot["samples"],
            population=boot["population"],
            numpy_version=boot["numpy_version"],
            sample_indices_sha256=boot["sample_indices_sha256"],
            bit_generator=boot["bit_generator"],
            sample_dtype=boot["sample_dtype"],
            endpoint=boot["endpoint"],
            quantile_method=boot["quantile_method"],
        )
        baseline_confusion = tuple(sum(row.baseline_confusion[index] for row in rows) for index in range(4))
        policy_confusion = tuple(sum(row.policy_confusion[index] for row in rows) for index in range(4))
        baseline_metrics = project_metrics_antigravity(baseline_confusion)
        policy_metrics = project_metrics_antigravity(policy_confusion)
        baseline_macro = _mean_metric_antigravity(rows, policy=False)
        policy_macro = _mean_metric_antigravity(rows, policy=True)
        gates = _success_gates_antigravity(
            micro_delta=float(policy_metrics.f1 - baseline_metrics.f1),
            macro_f1_delta=float(policy_macro.f1 - baseline_macro.f1),
            bootstrap=bootstrap,
            policy_fp=policy_confusion[1],
            rescued_count=sum(row.rescued for row in rows),
            strong_regression_count=sum(row.strong_regression for row in rows),
            unaffected_identical_count=sum(not row.affected and not row.added_coordinates and not row.removed_coordinates for row in rows),
            pre_freeze_label_access_count=0,
            error_count=0,
            nonfinite_count=0,
        )
        bootstrap_rows = np.asarray([
            (*row.baseline_confusion, *row.policy_confusion, row.policy_metrics.f1 - row.baseline_metrics.f1)
            for row in rows
        ], dtype=np.float64)
        if paired_dataset_bootstrap_antigravity(bootstrap_rows) != bootstrap:
            raise ValueError("evaluation bootstrap disagrees with dataset rows")
        reconstructed = PolicyEvaluationAntigravity(
            policy,
            tuple(rows),
            _family_rows_antigravity(policy, rows),
            baseline_confusion,
            policy_confusion,
            baseline_metrics,
            policy_metrics,
            baseline_macro,
            policy_macro,
            float(policy_metrics.f1 - baseline_metrics.f1),
            float(policy_macro.f1 - baseline_macro.f1),
            sum(row.added_tp for row in rows),
            sum(row.added_fp for row in rows),
            sum(row.removed_tp for row in rows),
            sum(row.removed_fp for row in rows),
            tuple(row.dataset for row in rows if row.rescued),
            tuple(row.dataset for row in rows if row.regressed),
            tuple(row.dataset for row in rows if row.strong_regression),
            sum(not row.affected and not row.added_coordinates and not row.removed_coordinates for row in rows),
            bootstrap,
            gates,
            "retrospective_improvement_signal" if all(gates.values()) else "no_retrospective_improvement_signal",
        )
        if _policy_summary_antigravity(reconstructed) != summary:
            raise ValueError("evaluation policy summary disagrees with row evidence")
        policy_results.append(reconstructed)
    statuses = {result.policy: result.status for result in policy_results}
    if decision.get("policy_status") != statuses:
        raise ValueError("evaluation decision status differs")
    prediction = evaluation.get("prediction_sha256")
    if not isinstance(prediction, dict) or set(prediction) != set(_POLICIES_ANTIGRAVITY):
        raise ValueError("evaluation prediction identity differs")
    result = EvaluationResultAntigravity(
        publication_manifest_sha256=_hex_antigravity(evaluation.get("publication_manifest_sha256"), "publication manifest"),
        prediction_sha256=tuple((policy, _hex_antigravity(prediction[policy], "prediction")) for policy in _POLICIES_ANTIGRAVITY),
        database_sha256=_hex_antigravity(evaluation.get("database_sha256"), "database"),
        label_identity_sha256=_hex_antigravity(evaluation.get("label_identity_sha256"), "label identity"),
        label_rows=int(evaluation.get("label_rows")),
        label_query_calls=int(evaluation.get("label_query_calls")),
        policy_results=tuple(policy_results),
    )
    if type(_enforce_official_antigravity) is not bool:
        raise ValueError("evaluation replay mode is invalid")
    if _enforce_official_antigravity:
        if result.database_sha256 != FROZEN_CONFIG.database_sha256:
            raise ValueError("evaluation database SHA256 differs")
        if result.label_rows != sum(_BASELINE_CONFUSION_ANTIGRAVITY):
            raise ValueError("evaluation TEST-label population differs")
        for policy_result in result.policy_results:
            if (
                policy_result.baseline_confusion != _BASELINE_CONFUSION_ANTIGRAVITY
                or policy_result.baseline_metrics.f1 != _BASELINE_F1_ANTIGRAVITY
                or policy_result.unaffected_identical_count
                != FROZEN_CONFIG.dataset_count - FROZEN_CONFIG.affected_dataset_count
            ):
                raise ValueError("evaluation baseline or unaffected authority differs")
        policy_a = result.policy_results[0]
        if (
            policy_a.policy_confusion != _BASELINE_CONFUSION_ANTIGRAVITY
            or policy_a.policy_metrics.f1 != _BASELINE_F1_ANTIGRAVITY
            or any(row.added_coordinates or row.removed_coordinates for row in policy_a.dataset_rows)
        ):
            raise ValueError("evaluation Policy A reproduction differs")
    expected_payloads = _evaluation_payloads_antigravity(result)
    if any(expected_payloads[name] != payloads[name] for name in EVALUATION_NAMES_ANTIGRAVITY):
        raise ValueError("evaluation replay bytes disagree with row evidence")
    return result


def inspect_evaluation_antigravity(
    attempt: Path | str,
    evaluation_destination: Path | str | None = None,
) -> FrozenEvaluationAntigravity:
    publication = inspect_publication_antigravity(attempt)
    destination = Path(FROZEN_CONFIG.run_root) / "evaluation_antigravity" if evaluation_destination is None else Path(evaluation_destination)
    payloads = _read_evaluation_inventory_antigravity(destination)
    result = _evaluation_result_from_payloads_antigravity(payloads)
    if result.publication_manifest_sha256 != publication.manifest_sha256 or result.prediction_sha256 != publication.prediction_sha256:
        raise ValueError("evaluation no longer matches the frozen publication")
    return FrozenEvaluationAntigravity(
        str(destination),
        sha256(payloads["artifact_sha256_antigravity.json"]).hexdigest(),
        result,
    )


def _evaluation_phase_antigravity(_phase: str) -> None:
    """Private deterministic failure-injection seam."""


def _preserve_failed_evaluation_antigravity(
    parent: Path,
    staging: Path | None,
    error: BaseException,
    *,
    phase: str,
    labels_opened: bool,
    destination_committed: bool,
) -> None:
    failure = canonical_json_bytes_antigravity({
        "schema": "exp443.evaluation-failure.v1",
        "status": "failed",
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error) or type(error).__name__,
        "labels_opened": labels_opened,
        "destination_committed": destination_committed,
        "scientific_winner_selected": False,
        "review_only": True,
        "operational_alert_eligible": False,
    })
    failed = parent / f"_failed_evaluation_{uuid4().hex}_antigravity"
    if staging is not None and staging.exists() and not staging.is_symlink():
        write_create_only_antigravity(staging / "failure_antigravity.json", failure)
        fsync_directory_antigravity(staging)
        exclusive_rename_antigravity(staging, failed)
    else:
        failed.mkdir(mode=0o700)
        write_create_only_antigravity(failed / "failure_antigravity.json", failure)
        fsync_directory_antigravity(failed)
    fsync_directory_antigravity(parent)


def evaluate_publication_antigravity(
    attempt: Path | str | None = None,
    evaluation_destination: Path | str | None = None,
) -> FrozenEvaluationAntigravity:
    """Create one immutable evaluation, or replay it without reopening labels."""

    attempt_path = (
        Path(FROZEN_CONFIG.run_root) / "attempts_antigravity" / "attempt_0000_antigravity"
        if attempt is None else Path(attempt)
    )
    destination = (
        Path(FROZEN_CONFIG.run_root) / "evaluation_antigravity"
        if evaluation_destination is None else Path(evaluation_destination)
    )
    publication = inspect_publication_antigravity(attempt_path)
    if destination.exists() or destination.is_symlink():
        return inspect_evaluation_antigravity(attempt_path, destination)
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("evaluation parent is invalid")
    staging: Path | None = None
    phase = "pre_label"
    labels_opened = False
    committed = False
    try:
        labels = load_frozen_test_labels_antigravity(publication)
        labels_opened = True
        phase = "compute"
        result = compute_policy_evaluation_antigravity(publication, labels)
        payloads = _evaluation_payloads_antigravity(result)
        staging = parent / f".evaluation_{uuid4().hex}_antigravity"
        staging.mkdir(mode=0o700)
        phase = "before_file_write"
        _evaluation_phase_antigravity(phase)
        for name in EVALUATION_NAMES_ANTIGRAVITY:
            write_create_only_antigravity(staging / name, payloads[name])
        fsync_directory_antigravity(staging)
        staged = _read_evaluation_inventory_antigravity(staging)
        if any(staged[name] != payloads[name] for name in EVALUATION_NAMES_ANTIGRAVITY):
            raise ValueError("evaluation staging bytes differ")
        phase = "before_rename"
        _evaluation_phase_antigravity(phase)
        exclusive_rename_antigravity(staging, destination)
        committed = True
        fsync_directory_antigravity(parent)
        phase = "after_rename_verification"
        frozen = inspect_evaluation_antigravity(attempt_path, destination)
        _evaluation_phase_antigravity(phase)
        return frozen
    except BaseException as error:
        try:
            _preserve_failed_evaluation_antigravity(
                parent,
                staging,
                error,
                phase=phase,
                labels_opened=labels_opened,
                destination_committed=committed,
            )
        except BaseException as preservation_error:
            error.add_note(
                "evaluation failure preservation also failed: "
                f"{type(preservation_error).__name__}: {preservation_error}"
            )
        raise


__all__ = [
    "LABEL_SQL_ANTIGRAVITY",
    "EVALUATION_NAMES_ANTIGRAVITY",
    "FrozenLabelsAntigravity",
    "DatasetEvaluationRowAntigravity",
    "FamilyDiagnosticAntigravity",
    "PolicyEvaluationAntigravity",
    "EvaluationResultAntigravity",
    "FrozenEvaluationAntigravity",
    "load_frozen_test_labels_antigravity",
    "compute_policy_evaluation_antigravity",
    "evaluate_publication_antigravity",
    "inspect_evaluation_antigravity",
]
