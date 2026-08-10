"""Post-freeze Exp444 evaluator; TEST labels stay unreachable until Task8 matches twice."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
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
from .profiles_antigravity import (
    EXP444_PROFILE_ANTIGRAVITY,
    EXP445_PROFILE_ANTIGRAVITY,
    LifecycleProfileAntigravity,
)
from .reporting_antigravity import (
    DATASET_SUMMARY_FILE_ANTIGRAVITY, SOURCE_AUDIT_FILE_ANTIGRAVITY,
    inspect_publication_antigravity, _inspect_exp445_publication_antigravity,
    _pinned_publication_payloads_antigravity,
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

    def __init__(
        self, profile: LifecycleProfileAntigravity = EXP444_PROFILE_ANTIGRAVITY,
    ) -> None:
        self._profile = _require_evaluation_profile_antigravity(profile)

    def _publication_root_antigravity(self) -> Path:
        return self._profile.run_root / "attempts_antigravity" / "attempt_0000_antigravity" / "publication_antigravity"

    def inspect_publication_antigravity(self):
        inspector = (
            inspect_publication_antigravity
            if self._profile is EXP444_PROFILE_ANTIGRAVITY
            else _inspect_exp445_publication_antigravity
        )
        return inspector(self._publication_root_antigravity())

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

    def open_labels_antigravity(self, identity: Mapping[str, object]) -> _LabelSessionAntigravity:
        return _open_verified_labels_antigravity(Path(FROZEN_CONFIG.database_path), FROZEN_CONFIG.database_sha256, tuple(identity["dataset_order"]))

    def evaluation_root_antigravity(self) -> Path:
        return self._profile.run_root / "evaluation_antigravity"


def _require_evaluation_profile_antigravity(
    profile: object,
) -> LifecycleProfileAntigravity:
    """Keep evaluation routing closed to its two reviewed lifecycle identities."""
    if profile is EXP444_PROFILE_ANTIGRAVITY:
        return EXP444_PROFILE_ANTIGRAVITY
    if profile is EXP445_PROFILE_ANTIGRAVITY:
        return EXP445_PROFILE_ANTIGRAVITY
    raise ValueError("evaluation lifecycle profile is invalid")


class _SqliteLabelsAntigravity:
    """Zero-state facade: the verified connection remains closure-owned."""
    __slots__ = ()


@contextmanager
def _open_verified_labels_antigravity(database: Path, expected_sha256: str, dataset_order: tuple[str, ...]):
    """Open exactly one descriptor-bound, hash-bound TEST-label session lazily."""
    inputs = importlib.import_module("experiments.exp417_causal_multihorizon_shadow.inputs")
    sqlite3 = importlib.import_module("sqlite3")
    with inputs.verified_read_only_connection(Path(database), expected_sha256) as connection:
        allowed_reads = {("instances", "instance_index"), ("instances", "label"), ("instances", "split"), ("instances", "dataset_id"), ("datasets", "id"), ("datasets", "name")}
        def authorizer(action: int, arg1: str | None, arg2: str | None, _database: str | None, _source: str | None) -> int:
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and (arg1, arg2) in allowed_reads:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        connection.set_authorizer(authorizer)
        if not dataset_order or any(type(dataset) is not str or not dataset for dataset in dataset_order) or len(set(dataset_order)) != len(dataset_order):
            raise ValueError("label dataset order is invalid")
        next_index = 0
        closed = False
        def query_next(dataset: str) -> tuple[tuple[int, int], ...]:
            nonlocal next_index
            if closed or next_index >= len(dataset_order) or dataset != dataset_order[next_index]:
                raise ValueError("label query order is invalid")
            cursor = connection.execute("""SELECT i.instance_index, i.label
FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST' ORDER BY i.instance_index""", (dataset,))
            next_index += 1
            return tuple(cursor.fetchall())
        def close_facade() -> None:
            nonlocal closed
            closed = True
        class Facade(_SqliteLabelsAntigravity):
            __slots__ = ()
            def labels_for_dataset_antigravity(self, dataset: str) -> tuple[tuple[int, int], ...]:
                return query_next(dataset)
            def close_antigravity(self) -> None:
                close_facade()
        try:
            yield Facade()
        finally:
            connection.set_authorizer(None)


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


def _validate_prepublication_antigravity(identity: Mapping[str, object], rows: Sequence[Mapping[str, object]], transitions: Sequence[Mapping[str, object],], *, attempts: int, completions: int, label_rows: int, production: bool) -> None:
    """Validate frozen counts and the complete source/stable/unstable delta partition."""
    if attempts != completions:
        raise ValueError("evaluation dataset/query count is invalid")
    if production and (attempts, completions, label_rows) != (1117, 1117, 195900):
        raise ValueError("evaluation frozen dataset/query/rows count is invalid")
    if sum(sum(row["baseline"]) for row in rows) != label_rows:
        raise ValueError("evaluation label rows are invalid")
    baseline, all_rows, stable, unstable = (_sum_confusion_antigravity(rows, field) for field in ("baseline", "frozen_source", "stable", "unstable"))
    if production and baseline != (1915, 303, 2003, 191679):
        raise ValueError("evaluation baseline confusion is invalid")
    if len(rows) != attempts:
        raise ValueError("evaluation dataset/query count is invalid")
    source = tuple(tuple(item) for item in identity.get("source_order", ()))
    stable_coordinates = tuple(tuple(item) for item in identity.get("stable", ()))
    unstable_coordinates = tuple(tuple(item) for item in identity.get("unstable", ()))
    if len(source) != len(set(source)) or set(stable_coordinates).intersection(unstable_coordinates) or set(stable_coordinates) | set(unstable_coordinates) != set(source):
        raise ValueError("evaluation source lane partition is invalid")
    if production and len(source) != 30:
        raise ValueError("evaluation source count is invalid")
    transition_coordinates = tuple((row.get("dataset"), row.get("instance_index")) for row in transitions)
    if set(transition_coordinates) != set(source) or len(transition_coordinates) != len(set(transition_coordinates)):
        raise ValueError("evaluation transition partition is invalid")
    lane_delta = {"stable_review": [0, 0], "unstable_review": [0, 0]}
    for transition in transitions:
        lane, label, already = transition.get("lane"), transition.get("label"), transition.get("baseline_predicted")
        if lane not in lane_delta or type(label) is not int or label not in (0, 1) or type(already) is not bool:
            raise ValueError("evaluation transition is invalid")
        coordinate = (transition.get("dataset"), transition.get("instance_index"))
        expected_lane = "stable_review" if coordinate in set(stable_coordinates) else "unstable_review"
        if lane != expected_lane or transition.get("stable_predicted") is not (lane == "stable_review") or transition.get("unstable_predicted") is not (lane == "unstable_review"):
            raise ValueError("evaluation transition lane is invalid")
        if not already:
            lane_delta[lane][0 if label else 1] += 1
    def delta(confusion: tuple[int, int, int, int]) -> tuple[int, int]:
        return (confusion[0] - baseline[0], confusion[1] - baseline[1])
    all_delta = delta(all_rows)
    if (lane_delta["stable_review"][0] + lane_delta["unstable_review"][0], lane_delta["stable_review"][1] + lane_delta["unstable_review"][1]) != all_delta or tuple(lane_delta["stable_review"]) != delta(stable) or tuple(lane_delta["unstable_review"]) != delta(unstable):
        raise ValueError("evaluation lane TP/FP conservation is invalid")


def _payloads_antigravity(
    identity: Mapping[str, object], rows: Sequence[Mapping[str, object]],
    transitions: Sequence[Mapping[str, object]], *,
    profile: LifecycleProfileAntigravity = EXP444_PROFILE_ANTIGRAVITY,
) -> dict[str, bytes]:
    profile = _require_evaluation_profile_antigravity(profile)
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
    evaluation = {"schema": f"{profile.schema}.evaluation.v1", "status": "descriptive_complete_no_promotion", "scientific_winner_selected": False, "scientific_success_criterion": None, "publication_identity": dict(identity), "dataset_count": len(rows), "label_rows": sum(sum(row[name]) for row in rows for name in ("baseline",)), "facts": facts, "artifact_sha256": {name: sha256(payload).hexdigest() for name, payload in result.items()}}
    result["evaluation_antigravity.json"] = canonical_json_bytes_antigravity(evaluation)
    decision = {"schema": f"{profile.schema}.decision.v1", "status": "descriptive_complete_no_promotion", "scientific_success_criterion": None, "scientific_winner_selected": False, "review_only": True, "operational_alert_eligible": False, "same_experiment_reselection_allowed": False, "publication_manifest_sha256": identity["manifest_sha256"], "prediction_sha256": identity["prediction_sha256"], "evaluation_sha256": sha256(result["evaluation_antigravity.json"]).hexdigest()}
    result["decision_antigravity.json"] = canonical_json_bytes_antigravity(decision)
    result["artifact_sha256_antigravity.json"] = canonical_json_bytes_antigravity({"schema": f"{profile.schema}.evaluation-artifact-sha256.v1", "self_excluded": "artifact_sha256_antigravity.json", "files": {name: sha256(result[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY}})
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


def _audit_antigravity(
    dependencies: object, *, phase: str, identity: Mapping[str, object] | None,
    attempts: int, completions: int, rows: int, session_state: str,
    error: BaseException,
    profile: LifecycleProfileAntigravity = EXP444_PROFILE_ANTIGRAVITY,
) -> None:
    profile = _require_evaluation_profile_antigravity(profile)
    root = Path(dependencies.evaluation_root_antigravity()).parent
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / f"evaluation_audit_{uuid4().hex}_antigravity.json"
    destination = Path(dependencies.evaluation_root_antigravity())
    if not destination.exists() and not destination.is_symlink():
        commit_state = "none"
    elif destination.is_dir() and not destination.is_symlink() and set(path.name for path in destination.iterdir()) == set(EVALUATION_NAMES_ANTIGRAVITY):
        commit_state = "complete"
    else:
        commit_state = "partial"
    payload = {"schema": f"{profile.schema}.evaluation-audit.v1", "phase": phase, "publication_identity": dict(identity or {}), "query_attempts": attempts, "query_completions": completions, "rows": rows, "session_state": session_state, "commit_state": commit_state, "error": f"{type(error).__name__}: {error}"}
    write_create_only_antigravity(path, canonical_json_bytes_antigravity(payload))


def _result_from_payloads_antigravity(
    payloads: Mapping[str, bytes], *, production: bool = False,
    profile: LifecycleProfileAntigravity = EXP444_PROFILE_ANTIGRAVITY,
) -> FrozenEvaluationAntigravity:
    profile = _require_evaluation_profile_antigravity(profile)
    if set(payloads) != set(EVALUATION_NAMES_ANTIGRAVITY):
        raise ValueError("evaluation inventory is invalid")
    artifact = parse_canonical_json_bytes_antigravity(payloads["artifact_sha256_antigravity.json"])
    if not isinstance(artifact, Mapping) or artifact.get("schema") != f"{profile.schema}.evaluation-artifact-sha256.v1" or artifact.get("self_excluded") != "artifact_sha256_antigravity.json" or set(artifact.get("files", {})) != set(_LEAVES_ANTIGRAVITY):
        raise ValueError("evaluation artifact map is invalid")
    if any(artifact["files"][name] != sha256(payloads[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY):
        raise ValueError("evaluation artifact changed")
    evaluation = parse_canonical_json_bytes_antigravity(payloads["evaluation_antigravity.json"])
    decision = parse_canonical_json_bytes_antigravity(payloads["decision_antigravity.json"])
    if not isinstance(evaluation, Mapping) or not isinstance(decision, Mapping) or evaluation.get("schema") != f"{profile.schema}.evaluation.v1" or decision.get("schema") != f"{profile.schema}.decision.v1" or evaluation.get("status") != "descriptive_complete_no_promotion" or decision.get("status") != "descriptive_complete_no_promotion" or decision.get("scientific_winner_selected") is not False:
        raise ValueError("evaluation governance changed")
    facts, identity = evaluation.get("facts"), evaluation.get("publication_identity")
    if not isinstance(facts, Mapping) or not isinstance(identity, Mapping):
        raise ValueError("evaluation summary is invalid")
    metrics = payloads["dataset_metrics_antigravity.jsonl"].splitlines()
    rows = [parse_canonical_json_bytes_antigravity(line + b"\n") for line in metrics]
    metric_keys = {"dataset", "baseline", "frozen_source", "stable", "unstable"}
    if not all(isinstance(row, Mapping) and set(row) == metric_keys and type(row["dataset"]) is str and all(isinstance(value, list) and len(value) == 4 and all(type(item) is int and item >= 0 for item in value) for value in (row["baseline"], row["frozen_source"], row["stable"], row["unstable"])) for row in rows) or tuple(row["dataset"] for row in rows) != tuple(identity.get("dataset_order", ())):
        raise ValueError("evaluation metrics are invalid")
    transitions = [parse_canonical_json_bytes_antigravity(line + b"\n") for line in payloads["candidate_transitions_antigravity.jsonl"].splitlines()]
    transition_keys = {"dataset", "instance_index", "lane", "label", "baseline_predicted", "frozen_source_predicted", "stable_predicted", "unstable_predicted"}
    source = tuple(tuple(value) for value in identity.get("source_order", ()))
    expected_transition_order = tuple((dataset, index) for dataset in identity.get("dataset_order", ()) for name, index in sorted(source) if name == dataset)
    if not all(isinstance(row, Mapping) and set(row) == transition_keys and type(row["dataset"]) is str and type(row["instance_index"]) is int and row["instance_index"] >= 0 and type(row["label"]) is int and row["label"] in (0, 1) and all(type(row[name]) is bool for name in ("baseline_predicted", "frozen_source_predicted", "stable_predicted", "unstable_predicted")) and row["frozen_source_predicted"] is True for row in transitions) or tuple((row["dataset"], row["instance_index"]) for row in transitions) != expected_transition_order:
        raise ValueError("evaluation transitions are invalid")
    _validate_prepublication_antigravity(identity, rows, transitions, attempts=len(rows), completions=len(rows), label_rows=sum(sum(row["baseline"]) for row in rows), production=production)
    reconstructed = _payloads_antigravity(identity, rows, transitions, profile=profile)
    if any(payloads[name] != reconstructed[name] for name in EVALUATION_NAMES_ANTIGRAVITY):
        raise ValueError("evaluation derived replay changed")
    return FrozenEvaluationAntigravity("descriptive_complete_no_promotion", False, sha256(payloads["evaluation_antigravity.json"]).hexdigest(), identity, len(rows), len(rows), sum(sum(row["baseline"]) for row in rows), tuple(facts["baseline_confusion"]), tuple(facts["frozen_source_confusion"]), tuple(facts["stable_confusion"]), tuple(facts["unstable_confusion"]))


def _inspect_evaluation_for_profile_antigravity(
    profile: LifecycleProfileAntigravity, *, dependencies: object | None = None,
) -> FrozenEvaluationAntigravity:
    """Replay an existing ten-leaf evaluation before any label-capable import."""
    profile = _require_evaluation_profile_antigravity(profile)
    dependencies = _FrozenDependenciesAntigravity(profile) if dependencies is None else dependencies
    root = Path(dependencies.evaluation_root_antigravity())
    if root.is_symlink() or not root.is_dir():
        raise ValueError("evaluation publication is absent")
    from .artifacts_antigravity import pin_exact_regular_inventory_antigravity
    with pin_exact_regular_inventory_antigravity(root, EVALUATION_NAMES_ANTIGRAVITY) as pinned:
        payloads = dict(pinned.payloads)
    result = _result_from_payloads_antigravity(
        payloads, production=isinstance(dependencies, _FrozenDependenciesAntigravity), profile=profile,
    )
    publication = dependencies.inspect_publication_antigravity()
    identity = dependencies.publication_identity_antigravity(publication)
    if not _identity_equal_antigravity(result.publication_identity, identity):
        raise ValueError("evaluation Task8 publication identity changed")
    return result


def inspect_evaluation_antigravity(*, dependencies: object | None = None) -> FrozenEvaluationAntigravity:
    """Preserve Exp444's public frozen evaluation replay wrapper."""
    return _inspect_evaluation_for_profile_antigravity(
        EXP444_PROFILE_ANTIGRAVITY, dependencies=dependencies,
    )


def _evaluate_for_profile_antigravity(
    profile: LifecycleProfileAntigravity, *, dependencies: object | None = None,
) -> FrozenEvaluationAntigravity:
    """Evaluate frozen lanes only after independent full Task8 identity agreement."""
    profile = _require_evaluation_profile_antigravity(profile)
    dependencies = _FrozenDependenciesAntigravity(profile) if dependencies is None else dependencies
    root = Path(dependencies.evaluation_root_antigravity())
    if root.exists() or root.is_symlink():
        return _inspect_evaluation_for_profile_antigravity(profile, dependencies=dependencies)
    phase, identity, attempts, completions, row_count, session_state, label_boundary_entered = "pre_label_inspection", None, 0, 0, 0, "not_opened", False
    try:
        first = dependencies.inspect_publication_antigravity(); first_identity = dependencies.publication_identity_antigravity(first)
        # Exp404 is hash-bound before the independent repeat; labels remain unreachable.
        hard = dependencies.hard_predictions_antigravity(first_identity)
        phase = "second_inspection"
        second = dependencies.inspect_publication_antigravity(); identity = dependencies.publication_identity_antigravity(second)
        if not _identity_equal_antigravity(first_identity, identity):
            raise ValueError("Task8 publication content/root identity changed")
        phase = "labels_open"; label_boundary_entered = True
        opened = dependencies.open_labels_antigravity(identity)
        manager = opened if hasattr(opened, "__enter__") and hasattr(opened, "__exit__") else None
        session = manager.__enter__() if manager is not None else opened
        session_state = "open"
        try:
            labels_by_dataset: dict[str, tuple[int, ...]] = {}
            for dataset in identity["dataset_order"]:
                attempts += 1
                labels = _labels_antigravity(session.labels_for_dataset_antigravity(dataset), dataset)
                completions += 1; row_count += len(labels); labels_by_dataset[dataset] = labels
        finally:
            try:
                if manager is not None:
                    manager.__exit__(None, None, None)
                else:
                    session.close_antigravity()
            except BaseException:
                session_state = "close_failed"
                raise
            else:
                session_state = "closed"
        phase = "post_close_reinspection"
        post_close = dependencies.inspect_publication_antigravity(); post_close_identity = dependencies.publication_identity_antigravity(post_close)
        if not _identity_equal_antigravity(identity, post_close_identity):
            raise ValueError("Task8 publication changed after label close")
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
        _validate_prepublication_antigravity(identity, metric_rows, transitions, attempts=attempts, completions=completions, label_rows=row_count, production=isinstance(dependencies, _FrozenDependenciesAntigravity))
        hook = getattr(dependencies, "after_labels_antigravity", None)
        if hook is not None: hook()
        payloads = _payloads_antigravity(identity, metric_rows, transitions, profile=profile)
        phase = "post_compute_reinspection"
        final_publication = dependencies.inspect_publication_antigravity(); final_identity = dependencies.publication_identity_antigravity(final_publication)
        if not _identity_equal_antigravity(identity, final_identity):
            raise ValueError("Task8 publication changed after labels")
        phase = "pre_commit_reinspection"
        final_publication = dependencies.inspect_publication_antigravity(); final_identity = dependencies.publication_identity_antigravity(final_publication)
        if not _identity_equal_antigravity(identity, final_identity):
            raise ValueError("Task8 publication changed before evaluation commit")
        phase = "commit_write"
        _publish_create_only_antigravity(root, payloads)
        result = _result_from_payloads_antigravity(payloads, profile=profile)
        return FrozenEvaluationAntigravity(result.status, result.scientific_winner_selected, result.evaluation_sha256, result.publication_identity, attempts, completions, row_count, result.baseline_confusion, result.frozen_source_confusion, result.stable_confusion, result.unstable_confusion)
    except BaseException as error:
        if label_boundary_entered:
            _audit_antigravity(dependencies, phase=phase, identity=identity, attempts=attempts, completions=completions, rows=row_count, session_state=session_state, error=error, profile=profile)
        raise


def evaluate_frozen_publication_antigravity(*, dependencies: object | None = None) -> FrozenEvaluationAntigravity:
    """Preserve Exp444's public post-freeze evaluator."""
    return _evaluate_for_profile_antigravity(EXP444_PROFILE_ANTIGRAVITY, dependencies=dependencies)


def evaluate_exp445_frozen_publication_antigravity() -> FrozenEvaluationAntigravity:
    """Evaluate only the fixed Exp445 publication after its CLI inspection gate."""
    return _evaluate_for_profile_antigravity(EXP445_PROFILE_ANTIGRAVITY)


__all__ = [
    "FrozenEvaluationAntigravity", "EVALUATION_NAMES_ANTIGRAVITY",
    "evaluate_frozen_publication_antigravity", "evaluate_exp445_frozen_publication_antigravity",
    "inspect_evaluation_antigravity",
]
