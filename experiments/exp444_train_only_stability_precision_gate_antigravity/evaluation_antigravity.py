"""Post-freeze Exp444 evaluator; TEST labels stay unreachable until Task8 matches twice."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from uuid import uuid4
from typing import Mapping, Protocol, Sequence

from .artifacts_antigravity import canonical_json_bytes_antigravity, canonical_jsonl_bytes_antigravity, parse_canonical_json_bytes_antigravity, write_create_only_antigravity
from .bootstrap_antigravity import paired_bootstrap_antigravity, project_f1_antigravity
from .config_antigravity import FROZEN_CONFIG
from .reporting_antigravity import (
    DATASET_SUMMARY_FILE_ANTIGRAVITY, SOURCE_AUDIT_FILE_ANTIGRAVITY,
    inspect_publication_antigravity, _pinned_publication_payloads_antigravity,
)


EVALUATION_NAMES_ANTIGRAVITY = (
    "evaluation_antigravity.json", "dataset_metrics_antigravity.jsonl",
    "candidate_transitions_antigravity.jsonl", "rescued_datasets_antigravity.jsonl",
    "regressed_datasets_antigravity.jsonl", "strong_regressions_antigravity.jsonl",
    "bootstrap_antigravity.json", "confusion_antigravity.json", "decision_antigravity.json",
    "artifact_sha256_antigravity.json",
)
_LEAVES_ANTIGRAVITY = EVALUATION_NAMES_ANTIGRAVITY[:-1]


class _LabelSessionAntigravity(Protocol):
    def labels_for_dataset_antigravity(self, dataset: str) -> Sequence[tuple[int, int]]: ...
    def close_antigravity(self) -> None: ...


class _FrozenDependenciesAntigravity:
    """The only production dependency boundary; configuration cannot be overridden."""
    dataset_count = FROZEN_CONFIG.dataset_count
    label_count = 195900

    def _publication_root_antigravity(self) -> Path:
        return Path(FROZEN_CONFIG.run_root) / "attempts_antigravity" / "attempt_0000_antigravity" / "publication_antigravity"

    def inspect_publication_antigravity(self):
        return inspect_publication_antigravity(self._publication_root_antigravity())

    def publication_identity_antigravity(self, publication: object) -> dict[str, object]:
        root = Path(getattr(publication, "publication_root"))
        root_stat = root.stat(follow_symlinks=False)
        if root.is_symlink() or not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("evaluation publication root is invalid")
        manifest = dict(getattr(publication, "manifest"))
        leaves = manifest.get("leaf_sha256")
        if not isinstance(leaves, Mapping) or set(leaves) != {
            "dataset_summary_antigravity.jsonl", "source_candidates_antigravity.jsonl", "candidate_stability_antigravity.jsonl", "view_evidence_antigravity.jsonl", "stable_review_prediction_antigravity.jsonl", "unstable_review_prediction_antigravity.jsonl", "control_replay_antigravity.json", "source_audit_antigravity.json", "backend_decision_antigravity.json", "runtime_antigravity.json",
        }:
            raise ValueError("evaluation publication leaf map is invalid")
        with _pinned_publication_payloads_antigravity(root) as payloads:
            actual = {name: sha256(payloads[name]).hexdigest() for name in leaves}
            if any(actual[name] != leaves[name] for name in leaves):
                raise ValueError("evaluation publication leaf changed")
            summary = payloads[DATASET_SUMMARY_FILE_ANTIGRAVITY]
            audit = payloads[SOURCE_AUDIT_FILE_ANTIGRAVITY]
        root_after = root.stat(follow_symlinks=False)
        if (root_stat.st_dev, root_stat.st_ino) != (root_after.st_dev, root_after.st_ino):
            raise ValueError("evaluation publication root changed")
        for name in (DATASET_SUMMARY_FILE_ANTIGRAVITY, SOURCE_AUDIT_FILE_ANTIGRAVITY):
            if name not in actual:
                raise ValueError("evaluation Task8 authority is incomplete")
        selection = getattr(publication, "selection")
        def coordinates(items: object) -> list[list[object]]:
            return [[getattr(item, "dataset"), getattr(item, "instance_index")] for item in tuple(items)]
        source, stable, unstable = coordinates(selection.source), coordinates(selection.stable), coordinates(selection.unstable)
        summary_rows = tuple(line for line in summary.splitlines() if line)
        dataset_order: list[str] = []
        for line in summary_rows:
            row = parse_canonical_json_bytes_antigravity(line + b"\n")
            if not isinstance(row, Mapping) or type(row.get("dataset")) is not str:
                raise ValueError("evaluation Task8 dataset summary is invalid")
            dataset_order.append(row["dataset"])
        if len(dataset_order) != FROZEN_CONFIG.dataset_count or len(set(dataset_order)) != len(dataset_order):
            raise ValueError("evaluation Task8 dataset summary order is invalid")
        source_audit = parse_canonical_json_bytes_antigravity(audit)
        if not isinstance(source_audit, Mapping) or source_audit.get("label_access_count") != 0:
            raise ValueError("evaluation Task8 source audit is invalid")
        if len(source) != 30 or len(stable) + len(unstable) != 30 or stable + unstable == []:
            raise ValueError("evaluation Task8 selection is invalid")
        if {tuple(row) for row in stable}.intersection(tuple(row) for row in unstable) or {tuple(row) for row in stable + unstable} != {tuple(row) for row in source}:
            raise ValueError("evaluation Task8 lane selection is invalid")
        return {
            "root_device": root_stat.st_dev, "root_inode": root_stat.st_ino,
            "manifest_sha256": sha256(canonical_json_bytes_antigravity(manifest)).hexdigest(),
            "prediction_sha256": manifest.get("prediction_sha256"), "leaf_sha256": dict(sorted(actual.items())),
            "dataset_summary_sha256": actual[DATASET_SUMMARY_FILE_ANTIGRAVITY],
            "source_audit_sha256": actual[SOURCE_AUDIT_FILE_ANTIGRAVITY],
            "dataset_order": dataset_order, "source_order": source, "stable": stable, "unstable": unstable,
        }

    def hard_predictions_antigravity(self, identity: Mapping[str, object]) -> dict[str, set[int]]:
        from .artifacts_antigravity import read_pinned_regular_bytes_antigravity
        path = Path(FROZEN_CONFIG.exp404_hard_prediction_path)
        encoded, digest = read_pinned_regular_bytes_antigravity(path)
        if digest != FROZEN_CONFIG.exp404_hard_prediction_sha256:
            raise ValueError("Exp404 Hard baseline hash changed")
        values: dict[str, set[int]] = {}
        for line in encoded.decode("utf-8").splitlines():
            row = json.loads(line)
            if not isinstance(row, Mapping) or type(row.get("dataset")) is not str:
                raise ValueError("Exp404 Hard baseline row is invalid")
            raw = row.get("instance_index", row.get("indices", ()))
            indices = (raw,) if type(raw) is int else tuple(raw) if isinstance(raw, list) else ()
            if any(type(index) is not int or index < 0 for index in indices):
                raise ValueError("Exp404 Hard baseline index is invalid")
            values.setdefault(row["dataset"], set()).update(indices)
        return values

    def open_labels_antigravity(self, _identity: Mapping[str, object]) -> _LabelSessionAntigravity:
        # Deliberately local: importing this evaluator or replaying evaluation cannot import SQLite.
        sqlite3 = importlib.import_module("sqlite3")
        database = Path(FROZEN_CONFIG.database_path)
        connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
        return _SqliteLabelsAntigravity(connection)

    def evaluation_root_antigravity(self) -> Path:
        return Path(FROZEN_CONFIG.run_root) / "evaluation_antigravity"


class _SqliteLabelsAntigravity:
    def __init__(self, connection: object) -> None:
        self._connection = connection

    def labels_for_dataset_antigravity(self, dataset: str) -> tuple[tuple[int, int], ...]:
        cursor = self._connection.execute("""SELECT i.instance_index, i.label
FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST' ORDER BY i.instance_index""", (dataset,))
        return tuple(cursor.fetchall())

    def close_antigravity(self) -> None:
        self._connection.close()


@dataclass(frozen=True)
class FrozenEvaluationAntigravity:
    status: str
    scientific_winner_selected: bool
    evaluation_sha256: str
    publication_identity: Mapping[str, object]
    query_attempts: int
    query_completions: int
    rows: int
    baseline_confusion: tuple[int, int, int, int]
    frozen_source_confusion: tuple[int, int, int, int]
    stable_confusion: tuple[int, int, int, int]
    unstable_confusion: tuple[int, int, int, int]
    phase: str = "committed"
    session_state: str = "closed"

    def __post_init__(self) -> None:
        if self.status != "descriptive_complete_no_promotion" or self.scientific_winner_selected is not False:
            raise ValueError("evaluation governance is invalid")
        if len(self.evaluation_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.evaluation_sha256):
            raise ValueError("evaluation hash is invalid")
        if any(type(value) is not int or value < 0 for value in (self.query_attempts, self.query_completions, self.rows)) or self.query_completions > self.query_attempts:
            raise ValueError("evaluation query accounting is invalid")
        for confusion in (self.baseline_confusion, self.frozen_source_confusion, self.stable_confusion, self.unstable_confusion):
            if len(confusion) != 4 or any(type(value) is not int or value < 0 for value in confusion):
                raise ValueError("evaluation confusion is invalid")
        object.__setattr__(self, "publication_identity", MappingProxyType(dict(self.publication_identity)))


def _identity_equal_antigravity(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    # Root device/inode are deliberately compared separately from content hashes.
    keys = {"root_device", "root_inode", "manifest_sha256", "prediction_sha256", "leaf_sha256", "dataset_summary_sha256", "source_audit_sha256", "dataset_order", "source_order", "stable", "unstable"}
    return set(left) >= keys and set(right) >= keys and all(
        canonical_json_bytes_antigravity(left[key]) == canonical_json_bytes_antigravity(right[key])
        for key in keys
    )


def _labels_antigravity(rows: Sequence[object], dataset: str) -> tuple[int, ...]:
    labels: list[int] = []
    for expected, row in enumerate(tuple(rows)):
        if not isinstance(row, (tuple, list)) or len(row) != 2 or type(row[0]) is not int or row[0] != expected or type(row[1]) is not int or row[1] not in (0, 1):
            raise ValueError(f"TEST indices must be contiguous and unique: {dataset}")
        labels.append(row[1])
    if not labels:
        raise ValueError("TEST label query returned no rows")
    return tuple(labels)


def _confusion_antigravity(labels: Sequence[int], predicted: set[int]) -> tuple[int, int, int, int]:
    if any(index < 0 or index >= len(labels) for index in predicted):
        raise ValueError("prediction index is outside TEST authority")
    tp = sum(labels[index] for index in predicted)
    fp = len(predicted) - tp
    fn = sum(labels) - tp
    return (tp, fp, fn, len(labels) - tp - fp - fn)


def _sum_confusion_antigravity(rows: Sequence[Mapping[str, object]], field: str) -> tuple[int, int, int, int]:
    return tuple(sum(row[field][index] for row in rows) for index in range(4))  # type: ignore[index,return-value]


def _payloads_antigravity(identity: Mapping[str, object], rows: Sequence[Mapping[str, object]], transitions: Sequence[Mapping[str, object]]) -> dict[str, bytes]:
    baseline, all_rows, stable, unstable = (_sum_confusion_antigravity(rows, name) for name in ("baseline", "frozen_source", "stable", "unstable"))
    bootstrap = paired_bootstrap_antigravity(
        tuple({"dataset": row["dataset"], **dict(zip(("tp", "fp", "fn", "tn"), row["baseline"], strict=True))} for row in rows),
        tuple({"dataset": row["dataset"], **dict(zip(("tp", "fp", "fn", "tn"), row["stable"], strict=True))} for row in rows),
    )
    rescued = tuple(row for row in rows if row["stable"][0] > row["baseline"][0])
    regressed = tuple(row for row in rows if project_f1_antigravity(row["stable"]) < project_f1_antigravity(row["baseline"]))
    strong = tuple(row for row in regressed if project_f1_antigravity(row["baseline"]) >= 0.8)
    confusion = {"schema": "exp444.confusion.v1", "order": ["tp", "fp", "fn", "tn"], "baseline_hard": list(baseline), "hard_plus_frozen_source": list(all_rows), "hard_plus_stable": list(stable), "hard_plus_unstable": list(unstable)}
    bootstrap_payload = {"schema": "exp444.paired-bootstrap.v1", "comparator": "hard_plus_stable", **bootstrap.to_payload_antigravity()}
    metrics = canonical_jsonl_bytes_antigravity(rows)
    result: dict[str, bytes] = {
        "dataset_metrics_antigravity.jsonl": metrics,
        "candidate_transitions_antigravity.jsonl": canonical_jsonl_bytes_antigravity(transitions),
        "rescued_datasets_antigravity.jsonl": canonical_jsonl_bytes_antigravity({"dataset": row["dataset"]} for row in rescued),
        "regressed_datasets_antigravity.jsonl": canonical_jsonl_bytes_antigravity({"dataset": row["dataset"]} for row in regressed),
        "strong_regressions_antigravity.jsonl": canonical_jsonl_bytes_antigravity({"dataset": row["dataset"], "baseline_f1": project_f1_antigravity(row["baseline"]), "stable_f1": project_f1_antigravity(row["stable"])} for row in strong),
        "bootstrap_antigravity.json": canonical_json_bytes_antigravity(bootstrap_payload),
        "confusion_antigravity.json": canonical_json_bytes_antigravity(confusion),
    }
    facts = {"baseline_confusion": list(baseline), "frozen_source_confusion": list(all_rows), "stable_confusion": list(stable), "unstable_confusion": list(unstable)}
    evaluation = {"schema": "exp444.evaluation.v1", "status": "descriptive_complete_no_promotion", "scientific_winner_selected": False, "scientific_success_criterion": None, "publication_identity": dict(identity), "dataset_count": len(rows), "label_rows": sum(sum(row[name]) for row in rows for name in ("baseline",)), "facts": facts, "artifact_sha256": {name: sha256(payload).hexdigest() for name, payload in result.items()}}
    result["evaluation_antigravity.json"] = canonical_json_bytes_antigravity(evaluation)
    decision = {"schema": "exp444.decision.v1", "status": "descriptive_complete_no_promotion", "scientific_success_criterion": None, "scientific_winner_selected": False, "review_only": True, "operational_alert_eligible": False, "same_experiment_reselection_allowed": False, "publication_manifest_sha256": identity["manifest_sha256"], "prediction_sha256": identity["prediction_sha256"], "evaluation_sha256": sha256(result["evaluation_antigravity.json"]).hexdigest()}
    result["decision_antigravity.json"] = canonical_json_bytes_antigravity(decision)
    result["artifact_sha256_antigravity.json"] = canonical_json_bytes_antigravity({"schema": "exp444.evaluation-artifact-sha256.v1", "self_excluded": "artifact_sha256_antigravity.json", "files": {name: sha256(result[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY}})
    return result


def _publish_create_only_antigravity(root: Path, payloads: Mapping[str, bytes]) -> None:
    if root.exists() or root.is_symlink():
        raise FileExistsError("evaluation destination already exists")
    root.mkdir(mode=0o700, parents=False)
    try:
        for name in EVALUATION_NAMES_ANTIGRAVITY:
            write_create_only_antigravity(root / name, payloads[name])
    except BaseException:
        # A partial new root is evidence; never replace or amend it.
        raise


def _audit_antigravity(dependencies: object, *, phase: str, identity: Mapping[str, object] | None, attempts: int, completions: int, rows: int, session_state: str, error: BaseException) -> None:
    root = Path(dependencies.evaluation_root_antigravity()).parent
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / f"evaluation_audit_{uuid4().hex}_antigravity.json"
    payload = {"schema": "exp444.evaluation-audit.v1", "phase": phase, "publication_identity": dict(identity or {}), "query_attempts": attempts, "query_completions": completions, "rows": rows, "session_state": session_state, "commit_state": "uncommitted", "error": f"{type(error).__name__}: {error}"}
    write_create_only_antigravity(path, canonical_json_bytes_antigravity(payload))


def _result_from_payloads_antigravity(payloads: Mapping[str, bytes]) -> FrozenEvaluationAntigravity:
    if set(payloads) != set(EVALUATION_NAMES_ANTIGRAVITY):
        raise ValueError("evaluation inventory is invalid")
    artifact = parse_canonical_json_bytes_antigravity(payloads["artifact_sha256_antigravity.json"])
    if not isinstance(artifact, Mapping) or artifact.get("self_excluded") != "artifact_sha256_antigravity.json" or set(artifact.get("files", {})) != set(_LEAVES_ANTIGRAVITY):
        raise ValueError("evaluation artifact map is invalid")
    if any(artifact["files"][name] != sha256(payloads[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY):
        raise ValueError("evaluation artifact changed")
    evaluation = parse_canonical_json_bytes_antigravity(payloads["evaluation_antigravity.json"])
    decision = parse_canonical_json_bytes_antigravity(payloads["decision_antigravity.json"])
    if not isinstance(evaluation, Mapping) or not isinstance(decision, Mapping) or evaluation.get("status") != "descriptive_complete_no_promotion" or decision.get("status") != "descriptive_complete_no_promotion" or decision.get("scientific_winner_selected") is not False:
        raise ValueError("evaluation governance changed")
    facts, identity = evaluation.get("facts"), evaluation.get("publication_identity")
    if not isinstance(facts, Mapping) or not isinstance(identity, Mapping):
        raise ValueError("evaluation summary is invalid")
    metrics = payloads["dataset_metrics_antigravity.jsonl"].splitlines()
    rows = [parse_canonical_json_bytes_antigravity(line + b"\n") for line in metrics]
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("evaluation metrics are invalid")
    return FrozenEvaluationAntigravity("descriptive_complete_no_promotion", False, sha256(payloads["evaluation_antigravity.json"]).hexdigest(), identity, len(rows), len(rows), sum(sum(row["baseline"]) for row in rows), tuple(facts["baseline_confusion"]), tuple(facts["frozen_source_confusion"]), tuple(facts["stable_confusion"]), tuple(facts["unstable_confusion"]))


def inspect_evaluation_antigravity(*, dependencies: object | None = None) -> FrozenEvaluationAntigravity:
    """Replay an existing ten-leaf evaluation before any label-capable import."""
    dependencies = _FrozenDependenciesAntigravity() if dependencies is None else dependencies
    root = Path(dependencies.evaluation_root_antigravity())
    if root.is_symlink() or not root.is_dir():
        raise ValueError("evaluation publication is absent")
    from .artifacts_antigravity import pin_exact_regular_inventory_antigravity
    with pin_exact_regular_inventory_antigravity(root, EVALUATION_NAMES_ANTIGRAVITY) as pinned:
        payloads = dict(pinned.payloads)
    result = _result_from_payloads_antigravity(payloads)
    publication = dependencies.inspect_publication_antigravity()
    identity = dependencies.publication_identity_antigravity(publication)
    if not _identity_equal_antigravity(result.publication_identity, identity):
        raise ValueError("evaluation Task8 publication identity changed")
    return result


def evaluate_frozen_publication_antigravity(*, dependencies: object | None = None) -> FrozenEvaluationAntigravity:
    """Evaluate frozen lanes only after independent full Task8 identity agreement."""
    dependencies = _FrozenDependenciesAntigravity() if dependencies is None else dependencies
    root = Path(dependencies.evaluation_root_antigravity())
    if root.exists() or root.is_symlink():
        return inspect_evaluation_antigravity(dependencies=dependencies)
    phase, identity, attempts, completions, row_count, session_state = "pre_label_inspection", None, 0, 0, 0, "not_opened"
    try:
        first = dependencies.inspect_publication_antigravity(); first_identity = dependencies.publication_identity_antigravity(first)
        # Exp404 is hash-bound before the independent repeat; labels remain unreachable.
        hard = dependencies.hard_predictions_antigravity(first_identity)
        phase = "second_inspection"
        second = dependencies.inspect_publication_antigravity(); identity = dependencies.publication_identity_antigravity(second)
        if not _identity_equal_antigravity(first_identity, identity):
            raise ValueError("Task8 publication content/root identity changed")
        phase = "labels_open"
        session = dependencies.open_labels_antigravity(identity); session_state = "open"
        try:
            labels_by_dataset: dict[str, tuple[int, ...]] = {}
            for dataset in identity["dataset_order"]:
                attempts += 1
                labels = _labels_antigravity(session.labels_for_dataset_antigravity(dataset), dataset)
                completions += 1; row_count += len(labels); labels_by_dataset[dataset] = labels
        finally:
            session.close_antigravity(); session_state = "closed"
        if len(labels_by_dataset) != getattr(dependencies, "dataset_count") or row_count != getattr(dependencies, "label_count"):
            raise ValueError("evaluation label population differs from frozen Task8 summary")
        source, stable, unstable = ({tuple(value) for value in identity[name]} for name in ("source_order", "stable", "unstable"))
        metric_rows: list[dict[str, object]] = []
        transitions: list[dict[str, object]] = []
        for dataset in labels_by_dataset:
            labels = labels_by_dataset[dataset]
            baseline = set(hard.get(dataset, set()))
            added = {index for name, index in source if name == dataset}
            stable_added = {index for name, index in stable if name == dataset}
            unstable_added = {index for name, index in unstable if name == dataset}
            values = {"baseline": _confusion_antigravity(labels, baseline), "frozen_source": _confusion_antigravity(labels, baseline | added), "stable": _confusion_antigravity(labels, baseline | stable_added), "unstable": _confusion_antigravity(labels, baseline | unstable_added)}
            metric_rows.append({"dataset": dataset, **{name: list(value) for name, value in values.items()}})
            for index in sorted(added):
                transitions.append({"dataset": dataset, "instance_index": index, "lane": "stable_review" if (dataset, index) in stable else "unstable_review", "label": labels[index], "baseline_predicted": index in baseline, "frozen_source_predicted": True, "stable_predicted": index in stable_added, "unstable_predicted": index in unstable_added})
        hook = getattr(dependencies, "after_labels_antigravity", None)
        if hook is not None: hook()
        phase = "post_compute_reinspection"
        final_publication = dependencies.inspect_publication_antigravity(); final_identity = dependencies.publication_identity_antigravity(final_publication)
        if not _identity_equal_antigravity(identity, final_identity):
            raise ValueError("Task8 publication changed after labels")
        payloads = _payloads_antigravity(identity, metric_rows, transitions)
        phase = "pre_commit_reinspection"
        final_publication = dependencies.inspect_publication_antigravity(); final_identity = dependencies.publication_identity_antigravity(final_publication)
        if not _identity_equal_antigravity(identity, final_identity):
            raise ValueError("Task8 publication changed before evaluation commit")
        _publish_create_only_antigravity(root, payloads)
        result = _result_from_payloads_antigravity(payloads)
        return FrozenEvaluationAntigravity(result.status, result.scientific_winner_selected, result.evaluation_sha256, result.publication_identity, attempts, completions, row_count, result.baseline_confusion, result.frozen_source_confusion, result.stable_confusion, result.unstable_confusion)
    except BaseException as error:
        if attempts:
            _audit_antigravity(dependencies, phase=phase, identity=identity, attempts=attempts, completions=completions, rows=row_count, session_state=session_state, error=error)
        raise


__all__ = ["FrozenEvaluationAntigravity", "EVALUATION_NAMES_ANTIGRAVITY", "evaluate_frozen_publication_antigravity", "inspect_evaluation_antigravity"]
