"""Aggregate fixed Exp457 routes over immutable Exp456 confusion evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
import re

from experiments.exp456_exp455_dataset_key_order_repair_antigravity.authority_antigravity import (
    capture_routing_authority_antigravity,
)
from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
    ComparisonMetricAntigravity,
    ConfusionAntigravity,
    DatasetRoutingAuthorityAntigravity,
    DatasetRoutingFeaturesAntigravity,
    FrozenRoutingAuthorityAntigravity,
    RoutingEvaluationAntigravity,
)
from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import (
    _comparison_antigravity,
    _prefer_candidate_antigravity,
)
from experiments.exp456_exp455_dataset_key_order_repair_antigravity.features_antigravity import (
    build_label_free_features_antigravity,
)
from experiments.exp456_exp455_dataset_key_order_repair_antigravity.replay_antigravity import (
    _inspect_replay_with_dependencies_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    CriterionOutcomeAntigravity,
    EvaluationResultAntigravity,
    Exp456ResultAuthorityAntigravity,
    FamilyMetricAntigravity,
    RouteDecisionAntigravity,
    TrainQualificationAntigravity,
)
from .router_antigravity import (
    _evidence_for_route_decision_antigravity,
    route_ecg_antigravity,
)


_EXP456_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp456_exp455_dataset_key_order_repair_antigravity_20260817"
)
_EXP456_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "features_antigravity.jsonl",
    "fold_decisions_antigravity.jsonl",
    "comparison_metrics_antigravity.json",
    "family_metrics_antigravity.jsonl",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)
_EXP456_IDENTITIES_ANTIGRAVITY = {
    "artifact_sha256_antigravity.json": "3c1321f74016d363899b41357609edca62ecf410d71708a9c90a6b803a4445e0",
    "comparison_metrics_antigravity.json": "6117a5f2ef28a3d543efcf4b499dd715508a4345a5e92679e1c14f0b5e1087cc",
    "decision_antigravity.json": "ea49b140475d4c9d8245a85ec0479a60f1ba6c46e8763b0c35877ec16ca6dc01",
}
_COMPARISON_NAMES_ANTIGRAVITY = (
    "always_baseline",
    "family_wide_local_gap",
    "oracle_per_dataset",
    "ecg_train_confidence_router",
)
_FAMILIES_ANTIGRAVITY = ("ECG5000", "ChlorineConcentration")
_EVIDENCE_BY_RESULT_ID_ANTIGRAVITY: dict[
    int,
    tuple[
        EvaluationResultAntigravity,
        Exp456ResultAuthorityAntigravity,
        tuple[TrainQualificationAntigravity, ...],
    ],
] = {}


def _expected_predecessor_antigravity() -> tuple[ComparisonMetricAntigravity, ...]:
    return (
        _comparison_antigravity(
            "always_baseline", (ConfusionAntigravity(609, 66, 837, 70788),)
        ),
        _comparison_antigravity(
            "family_wide_local_gap", (ConfusionAntigravity(611, 63, 835, 70791),)
        ),
        _comparison_antigravity(
            "oracle_per_dataset", (ConfusionAntigravity(618, 64, 828, 70790),)
        ),
    )


def _capture_exp456_result_authority_with_dependencies_antigravity(
    root: Path,
    expected_identities: Mapping[str, str],
    inspect: Callable[[Path], RoutingEvaluationAntigravity],
    capture: Callable[[], FrozenRoutingAuthorityAntigravity],
    feature_builder: Callable[
        [FrozenRoutingAuthorityAntigravity], tuple[DatasetRoutingFeaturesAntigravity, ...]
    ],
) -> Exp456ResultAuthorityAntigravity:
    """Dependency seam for identity-bound, label-query-free Exp456 replay."""
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise ValueError("Exp456 result root is invalid")
    if {entry.name for entry in root.iterdir()} != set(_EXP456_LEAVES_ANTIGRAVITY):
        raise ValueError("Exp456 result inventory differs")
    if not isinstance(expected_identities, Mapping) or set(expected_identities) != set(
        _EXP456_IDENTITIES_ANTIGRAVITY
    ):
        raise ValueError("Exp456 expected identities are invalid")
    observed: dict[str, str] = {}
    for name in _EXP456_IDENTITIES_ANTIGRAVITY:
        digest = expected_identities[name]
        path = root / name
        if (
            type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError("Exp456 result identity is invalid")
        observed[name] = sha256(path.read_bytes()).hexdigest()
        if observed[name] != digest:
            raise ValueError("Exp456 result identity differs")
    replayed = inspect(root)
    if type(replayed) is not RoutingEvaluationAntigravity or replayed.comparisons[:3] != _expected_predecessor_antigravity():
        raise ValueError("Exp456 semantic replay differs")
    authority = capture()
    if type(authority) is not FrozenRoutingAuthorityAntigravity or authority.label_query_count != 0:
        raise ValueError("Exp456 authority queried labels")
    features = feature_builder(authority)
    return Exp456ResultAuthorityAntigravity(
        authority,
        features,
        replayed.comparisons[:3],
        observed["artifact_sha256_antigravity.json"],
        observed["comparison_metrics_antigravity.json"],
        observed["decision_antigravity.json"],
        0,
    )


def capture_exp456_result_authority_antigravity() -> Exp456ResultAuthorityAntigravity:
    """Replay the immutable Exp456 result without opening TEST labels."""
    authority = capture_routing_authority_antigravity()
    features = build_label_free_features_antigravity(authority)

    def features_antigravity(
        received: FrozenRoutingAuthorityAntigravity,
    ) -> tuple[DatasetRoutingFeaturesAntigravity, ...]:
        if received is not authority:
            raise ValueError("Exp456 authority identity changed during capture")
        return features

    return _capture_exp456_result_authority_with_dependencies_antigravity(
        _EXP456_ROOT_ANTIGRAVITY,
        _EXP456_IDENTITIES_ANTIGRAVITY,
        lambda root: _inspect_replay_with_dependencies_antigravity(
            root, authority, features
        ),
        lambda: authority,
        features_antigravity,
    )


def _confusion_antigravity(value: object) -> ConfusionAntigravity:
    if type(value) is not ConfusionAntigravity or any(
        type(item) is not int or item < 0
        for item in (value.tp, value.fp, value.fn, value.tn)
    ):
        raise ValueError("confusion is invalid")
    return value


def _validate_authority_antigravity(
    value: object,
) -> tuple[
    tuple[DatasetRoutingAuthorityAntigravity, ...],
    dict[str, DatasetRoutingFeaturesAntigravity],
]:
    if type(value) is not Exp456ResultAuthorityAntigravity:
        raise ValueError("Exp456 result authority is invalid")
    if (
        value.artifact_map_sha256
        != _EXP456_IDENTITIES_ANTIGRAVITY["artifact_sha256_antigravity.json"]
        or value.comparison_sha256
        != _EXP456_IDENTITIES_ANTIGRAVITY["comparison_metrics_antigravity.json"]
        or value.decision_sha256
        != _EXP456_IDENTITIES_ANTIGRAVITY["decision_antigravity.json"]
        or value.predecessor_comparisons != _expected_predecessor_antigravity()
        or value.label_query_count != 0
    ):
        raise ValueError("Exp456 immutable result binding differs")
    rows = value.authority.rows
    if type(rows) is not tuple or len(rows) != 48 or any(
        type(row) is not DatasetRoutingAuthorityAntigravity for row in rows
    ):
        raise ValueError("Exp456 authority must contain exactly 48 rows")
    if len({row.dataset for row in rows}) != 48 or any(
        type(row.dataset) is not str or type(row.family) is not str for row in rows
    ):
        raise ValueError("Exp456 authority dataset keys are invalid")
    ecg = []
    chlorine = []
    targets: dict[str, DatasetRoutingAuthorityAntigravity] = {}
    for row in rows:
        _confusion_antigravity(row.baseline_confusion)
        if row.family == "ECG5000":
            ecg.append(row.dataset)
            expected_method = "hydra_local_gap"
        elif row.family == "ChlorineConcentration":
            chlorine.append(row.dataset)
            expected_method = "multirocket_local_gap"
        else:
            expected_method = None
        if expected_method is None:
            if row.method is not None or row.candidate_confusion is not None:
                raise ValueError("non-target Exp456 row has candidate evidence")
        elif row.method != expected_method or row.candidate_confusion is None:
            raise ValueError("target Exp456 row lacks its frozen candidate")
        if row.candidate_confusion is not None:
            _confusion_antigravity(row.candidate_confusion)
            if type(row.target_length) is not int or row.target_length < 0:
                raise ValueError("target Exp456 length is invalid")
            targets[row.dataset] = row
    if tuple(ecg) != FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets or tuple(chlorine) != tuple(
        f"ChlorineConcentration_normal_{index}" for index in range(1, 4)
    ):
        raise ValueError("Exp456 target dataset order differs")
    if len(value.features) != 8 or len({row.dataset for row in value.features}) != 8:
        raise ValueError("Exp456 feature inventory differs")
    features = {row.dataset: row for row in value.features}
    if set(features) != set(ecg + chlorine):
        raise ValueError("Exp456 feature dataset keys differ")
    for row in value.features:
        if type(row) is not DatasetRoutingFeaturesAntigravity:
            raise ValueError("Exp456 feature type differs")
        numeric = (
            row.target_length,
            row.baseline_count,
            row.candidate_count,
            row.overlap_count,
            row.union_count,
            row.added_count,
            row.removed_count,
            row.overlap_ppm,
        )
        if any(type(item) is not int or item < 0 for item in numeric):
            raise ValueError("Exp456 feature counts are invalid")
        target = targets[row.dataset]
        if row.family != target.family or row.target_length != target.target_length:
            raise ValueError("Exp456 feature identity differs from authority")
    return rows, features


def _validate_decisions_antigravity(
    decisions: object,
    features: Mapping[str, DatasetRoutingFeaturesAntigravity],
) -> tuple[
    dict[str, RouteDecisionAntigravity], tuple[TrainQualificationAntigravity, ...]
]:
    if type(decisions) is not tuple or len(decisions) != 5 or any(
        type(item) is not RouteDecisionAntigravity for item in decisions
    ):
        raise ValueError("route decisions must be the exact five-record tuple")
    if tuple(item.dataset for item in decisions) != FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets:
        raise ValueError("route decision dataset order differs")
    qualifications = []
    for item in decisions:
        source_feature, qualification = _evidence_for_route_decision_antigravity(item)
        if source_feature != features[item.dataset] or qualification.dataset != item.dataset:
            raise ValueError("route decision provenance differs from Exp456 feature authority")
        expected_structure = features[item.dataset].added_count > features[item.dataset].removed_count
        expected_route = "candidate" if item.train_gate and expected_structure else "baseline"
        if item.structure_gate is not expected_structure or item.route != expected_route:
            raise ValueError("route decision differs from recomputed gates")
        qualifications.append(qualification)
    return {item.dataset: item for item in decisions}, tuple(qualifications)


def evaluate_routes_antigravity(
    exp456_authority: Exp456ResultAuthorityAntigravity,
    decisions: tuple[RouteDecisionAntigravity, ...],
) -> EvaluationResultAntigravity:
    """Compute the exact three Exp456 comparators and conservative Exp457 route."""
    rows, features = _validate_authority_antigravity(exp456_authority)
    by_dataset, qualifications = _validate_decisions_antigravity(decisions, features)
    baseline = tuple(row.baseline_confusion for row in rows)
    family_wide = tuple(
        row.candidate_confusion
        if row.candidate_confusion is not None
        else row.baseline_confusion
        for row in rows
    )
    oracle = tuple(
        row.candidate_confusion
        if row.candidate_confusion is not None
        and _prefer_candidate_antigravity(row.baseline_confusion, row.candidate_confusion)
        else row.baseline_confusion
        for row in rows
    )
    routed = tuple(
        row.candidate_confusion
        if row.family == "ChlorineConcentration"
        or (
            row.family == "ECG5000"
            and by_dataset[row.dataset].route == "candidate"
        )
        else row.baseline_confusion
        for row in rows
    )
    comparisons = (
        _comparison_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[0], baseline),
        _comparison_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[1], family_wide),
        _comparison_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[2], oracle),
        _comparison_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[3], routed),
    )
    if comparisons[:3] != _expected_predecessor_antigravity():
        raise ValueError("computed Exp456 comparisons differ from immutable values")
    routed_by_dataset = dict(zip((row.dataset for row in rows), routed, strict=True))
    family_metrics = tuple(
        FamilyMetricAntigravity(
            family,
            sum(
                routed_by_dataset[row.dataset].fn - row.baseline_confusion.fn
                for row in rows
                if row.family == family
            ),
            sum(
                routed_by_dataset[row.dataset].fp - row.baseline_confusion.fp
                for row in rows
                if row.family == family
            ),
        )
        for family in _FAMILIES_ANTIGRAVITY
    )
    final = comparisons[-1]
    criteria = (
        CriterionOutcomeAntigravity(
            "final_fn",
            "<=",
            FROZEN_CONFIG_ANTIGRAVITY.final_fn_max,
            final.confusion.fn,
            final.confusion.fn <= FROZEN_CONFIG_ANTIGRAVITY.final_fn_max,
        ),
        CriterionOutcomeAntigravity(
            "final_fp",
            "<=",
            FROZEN_CONFIG_ANTIGRAVITY.final_fp_max,
            final.confusion.fp,
            final.confusion.fp <= FROZEN_CONFIG_ANTIGRAVITY.final_fp_max,
        ),
        CriterionOutcomeAntigravity(
            "final_f1",
            ">",
            FROZEN_CONFIG_ANTIGRAVITY.final_f1_strictly_above,
            final.f1,
            final.f1 > FROZEN_CONFIG_ANTIGRAVITY.final_f1_strictly_above,
        ),
    )
    passed = all(item.passed for item in criteria)
    result = EvaluationResultAntigravity(
        comparisons,
        family_metrics,
        decisions,
        criteria,
        passed,
        "routing_candidate_qualified_for_external_validation"
        if passed
        else "routing_candidate_not_qualified_for_external_validation",
    )
    _EVIDENCE_BY_RESULT_ID_ANTIGRAVITY[id(result)] = (
        result,
        exp456_authority,
        qualifications,
    )
    return result


def validate_evaluation_result_semantics_antigravity(
    result: EvaluationResultAntigravity,
) -> tuple[
    Exp456ResultAuthorityAntigravity, tuple[TrainQualificationAntigravity, ...]
]:
    """Recompute a result from its exact authority, Gate 1, and router provenance."""
    if type(result) is not EvaluationResultAntigravity:
        raise ValueError("evaluation result type is invalid")
    evidence = _EVIDENCE_BY_RESULT_ID_ANTIGRAVITY.get(id(result))
    if evidence is None or evidence[0] is not result:
        raise ValueError("evaluation result lacks semantic provenance")
    _stored, authority, qualifications = evidence
    _rows, features = _validate_authority_antigravity(authority)
    decisions = tuple(
        route_ecg_antigravity(features[qualification.dataset], qualification)
        for qualification in qualifications
    )
    expected = evaluate_routes_antigravity(authority, decisions)
    if expected != result:
        raise ValueError("evaluation result differs from semantic recomputation")
    return authority, qualifications


__all__ = (
    "capture_exp456_result_authority_antigravity",
    "evaluate_routes_antigravity",
    "validate_evaluation_result_semantics_antigravity",
)
