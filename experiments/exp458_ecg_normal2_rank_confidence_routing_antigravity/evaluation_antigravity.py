"""Evaluate the Exp458 normal_2 route over immutable Exp456 confusion rows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from hashlib import sha256
import json
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
    RankQualificationAntigravity,
    RouteDecisionAntigravity,
    validate_frozen_task3_qualification_antigravity,
)
from .router_antigravity import (
    _evidence_for_route_decision_antigravity,
    route_normal2_antigravity,
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
_EXP456_AUTHORITY_ROWS_SHA256_ANTIGRAVITY = "b1f6399b12a618967b4001468f77d27896bcd49575b12c84ac500e9ffb772adf"
_EXP456_INNER_AUTHORITY_INVENTORY_SHA256_ANTIGRAVITY = "213e39e9ef0a7072735b900e91f931214b549e18561e643da998665e5b9b97b2"
_EXP456_FEATURE_ROWS_SHA256_ANTIGRAVITY = "a7dc1130afa6a1fb54ba7573e07084b6432a240b7c0392676bc48e5e6b0f632c"
_EXP456_NORMAL2_FEATURE_ANTIGRAVITY = {
    "dataset": "ECG5000_normal_2",
    "family": "ECG5000",
    "target_length": 1800,
    "baseline_count": 13,
    "candidate_count": 22,
    "overlap_count": 8,
    "union_count": 27,
    "added_count": 14,
    "removed_count": 5,
    "overlap_ppm": 296296,
}
_COMPARISON_NAMES_ANTIGRAVITY = (
    "always_baseline",
    "family_wide_local_gap",
    "oracle_per_dataset",
    "normal2_rank_router",
)
_FAMILIES_ANTIGRAVITY = ("ECG5000", "ChlorineConcentration")
_EXPECTED_ECG_ANTIGRAVITY = tuple(f"ECG5000_normal_{index}" for index in range(1, 6))
_EXPECTED_CHLORINE_ANTIGRAVITY = tuple(
    f"ChlorineConcentration_normal_{index}" for index in range(1, 4)
)
_FINAL_FN_MAX_ANTIGRAVITY = 834
_FINAL_FP_MAX_ANTIGRAVITY = 66
_FINAL_F1_STRICTLY_ABOVE_ANTIGRAVITY = 0.5764150943396227
_EVIDENCE_BY_RESULT_ID_ANTIGRAVITY: dict[
    int,
    tuple[
        EvaluationResultAntigravity,
        Exp456ResultAuthorityAntigravity,
        RankQualificationAntigravity,
        DatasetRoutingFeaturesAntigravity,
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


def _json_bytes_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _confusion_payload_antigravity(value: ConfusionAntigravity) -> dict[str, int]:
    return {"tp": value.tp, "fp": value.fp, "fn": value.fn, "tn": value.tn}


def _index_payload_antigravity(value: object, name: str) -> list[int]:
    if type(value) is not frozenset or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{name} indices are invalid")
    return sorted(value)


def _authority_rows_payload_antigravity(
    rows: tuple[DatasetRoutingAuthorityAntigravity, ...],
) -> list[dict[str, object]]:
    return [
        {
            "dataset": row.dataset,
            "family": row.family,
            "target_length": row.target_length,
            "method": row.method,
            "baseline_indices": _index_payload_antigravity(row.baseline_indices, "baseline"),
            "candidate_indices": None
            if row.candidate_indices is None
            else _index_payload_antigravity(row.candidate_indices, "candidate"),
            "baseline_confusion": _confusion_payload_antigravity(row.baseline_confusion),
            "candidate_confusion": None
            if row.candidate_confusion is None
            else _confusion_payload_antigravity(row.candidate_confusion),
        }
        for row in rows
    ]


def _inventory_payload_antigravity(value: object) -> list[list[str]]:
    if type(value) is not tuple:
        raise ValueError("Exp456 inner authority inventory type differs")
    payload = []
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", item[1]) is None
        ):
            raise ValueError("Exp456 inner authority inventory differs")
        payload.append([item[0], item[1]])
    return payload


def _feature_payload_antigravity(row: DatasetRoutingFeaturesAntigravity) -> dict[str, object]:
    return {
        "dataset": row.dataset,
        "family": row.family,
        "target_length": row.target_length,
        "baseline_count": row.baseline_count,
        "candidate_count": row.candidate_count,
        "overlap_count": row.overlap_count,
        "union_count": row.union_count,
        "added_count": row.added_count,
        "removed_count": row.removed_count,
        "overlap_ppm": row.overlap_ppm,
    }


def _capture_exp456_result_authority_with_dependencies_antigravity(
    root: Path,
    expected_identities: Mapping[str, str],
    inspect: Callable[[Path], RoutingEvaluationAntigravity],
    capture: Callable[[], FrozenRoutingAuthorityAntigravity],
    feature_builder: Callable[
        [FrozenRoutingAuthorityAntigravity], tuple[DatasetRoutingFeaturesAntigravity, ...]
    ],
) -> Exp456ResultAuthorityAntigravity:
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise ValueError("Exp456 result root is invalid")
    if {entry.name for entry in root.iterdir()} != set(_EXP456_LEAVES_ANTIGRAVITY):
        raise ValueError("Exp456 result inventory differs")
    if not isinstance(expected_identities, Mapping) or set(expected_identities) != set(_EXP456_IDENTITIES_ANTIGRAVITY):
        raise ValueError("Exp456 expected identities are invalid")
    observed: dict[str, str] = {}
    for name in _EXP456_IDENTITIES_ANTIGRAVITY:
        digest = expected_identities[name]
        path = root / name
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None or path.is_symlink() or not path.is_file():
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
        lambda root: _inspect_replay_with_dependencies_antigravity(root, authority, features),
        lambda: authority,
        features_antigravity,
    )


def _confusion_antigravity(value: object) -> ConfusionAntigravity:
    if type(value) is not ConfusionAntigravity or any(
        type(item) is not int or item < 0 for item in (value.tp, value.fp, value.fn, value.tn)
    ):
        raise ValueError("confusion is invalid")
    return value


def _sum_antigravity(rows: Sequence[ConfusionAntigravity]) -> ConfusionAntigravity:
    return ConfusionAntigravity(
        sum(row.tp for row in rows),
        sum(row.fp for row in rows),
        sum(row.fn for row in rows),
        sum(row.tn for row in rows),
    )


def _f1_antigravity(confusion: ConfusionAntigravity) -> Fraction:
    denominator = 2 * confusion.tp + confusion.fp + confusion.fn
    return Fraction(0, 1) if denominator == 0 else Fraction(2 * confusion.tp, denominator)


def _comparison_local_antigravity(
    name: str, rows: Sequence[ConfusionAntigravity]
) -> ComparisonMetricAntigravity:
    confusion = _sum_antigravity(rows)
    return ComparisonMetricAntigravity(name, confusion, float(_f1_antigravity(confusion)))


def _validate_authority_antigravity(
    value: object,
) -> tuple[
    tuple[DatasetRoutingAuthorityAntigravity, ...],
    dict[str, DatasetRoutingFeaturesAntigravity],
]:
    if type(value) is not Exp456ResultAuthorityAntigravity:
        raise ValueError("Exp456 result authority is invalid")
    if (
        value.artifact_map_sha256 != _EXP456_IDENTITIES_ANTIGRAVITY["artifact_sha256_antigravity.json"]
        or value.comparison_sha256 != _EXP456_IDENTITIES_ANTIGRAVITY["comparison_metrics_antigravity.json"]
        or value.decision_sha256 != _EXP456_IDENTITIES_ANTIGRAVITY["decision_antigravity.json"]
        or value.predecessor_comparisons != _expected_predecessor_antigravity()
        or value.label_query_count != 0
        or type(value.authority.label_query_count) is not int
        or value.authority.label_query_count != 0
    ):
        raise ValueError("Exp456 immutable result binding differs")
    if sha256(_json_bytes_antigravity(_inventory_payload_antigravity(value.authority.exp454_leaf_sha256))).hexdigest() != _EXP456_INNER_AUTHORITY_INVENTORY_SHA256_ANTIGRAVITY:
        raise ValueError("Exp456 inner authority inventory differs")
    rows = value.authority.rows
    if type(rows) is not tuple or len(rows) != 48 or any(type(row) is not DatasetRoutingAuthorityAntigravity for row in rows):
        raise ValueError("Exp456 authority must contain exactly 48 rows")
    if sha256(_json_bytes_antigravity(_authority_rows_payload_antigravity(rows))).hexdigest() != _EXP456_AUTHORITY_ROWS_SHA256_ANTIGRAVITY:
        raise ValueError("Exp456 authority row content or order differs")
    if len({row.dataset for row in rows}) != 48:
        raise ValueError("Exp456 authority dataset keys are invalid")
    ecg = []
    chlorine = []
    targets: dict[str, DatasetRoutingAuthorityAntigravity] = {}
    for row in rows:
        if type(row.dataset) is not str or type(row.family) is not str:
            raise ValueError("Exp456 authority identity is invalid")
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
    if tuple(ecg) != _EXPECTED_ECG_ANTIGRAVITY or tuple(chlorine) != _EXPECTED_CHLORINE_ANTIGRAVITY:
        raise ValueError("Exp456 target dataset order differs")
    if type(value.features) is not tuple or len(value.features) != 8 or len({row.dataset for row in value.features}) != 8:
        raise ValueError("Exp456 feature inventory differs")
    feature_payload = [_feature_payload_antigravity(row) for row in value.features]
    if sha256(_json_bytes_antigravity(feature_payload)).hexdigest() != _EXP456_FEATURE_ROWS_SHA256_ANTIGRAVITY:
        raise ValueError("Exp456 feature content or order differs")
    features = {row.dataset: row for row in value.features}
    if set(features) != set(ecg + chlorine):
        raise ValueError("Exp456 feature dataset keys differ")
    if _feature_payload_antigravity(features[FROZEN_CONFIG_ANTIGRAVITY.dataset]) != _EXP456_NORMAL2_FEATURE_ANTIGRAVITY:
        raise ValueError("Exp456 normal_2 feature row differs")
    for row in value.features:
        if type(row) is not DatasetRoutingFeaturesAntigravity:
            raise ValueError("Exp456 feature type differs")
        if row.dataset not in targets or row.family != targets[row.dataset].family or row.target_length != targets[row.dataset].target_length:
            raise ValueError("Exp456 feature identity differs from authority")
    return rows, features


def _validate_decision_antigravity(
    decision: object,
    features: Mapping[str, DatasetRoutingFeaturesAntigravity],
) -> tuple[RouteDecisionAntigravity, RankQualificationAntigravity, DatasetRoutingFeaturesAntigravity]:
    if type(decision) is not RouteDecisionAntigravity or decision.dataset != FROZEN_CONFIG_ANTIGRAVITY.dataset:
        raise ValueError("normal_2 route decision is invalid")
    source_feature, qualification = _evidence_for_route_decision_antigravity(decision)
    if source_feature != features[FROZEN_CONFIG_ANTIGRAVITY.dataset] or qualification.dataset != FROZEN_CONFIG_ANTIGRAVITY.dataset:
        raise ValueError("route decision provenance differs from Exp456 feature authority")
    validate_frozen_task3_qualification_antigravity(qualification)
    expected_structure = source_feature.added_count > source_feature.removed_count
    expected_route = "candidate" if decision.rank_gate and expected_structure else "baseline"
    if decision.structure_gate is not expected_structure or decision.route != expected_route:
        raise ValueError("route decision differs from recomputed gates")
    return decision, qualification, source_feature


def _criteria_for_final_antigravity(
    final: ComparisonMetricAntigravity,
) -> tuple[tuple[CriterionOutcomeAntigravity, ...], bool, str]:
    criteria = (
        CriterionOutcomeAntigravity(
            "final_fn",
            "<=",
            _FINAL_FN_MAX_ANTIGRAVITY,
            final.confusion.fn,
            final.confusion.fn <= _FINAL_FN_MAX_ANTIGRAVITY,
        ),
        CriterionOutcomeAntigravity(
            "final_fp",
            "<=",
            _FINAL_FP_MAX_ANTIGRAVITY,
            final.confusion.fp,
            final.confusion.fp <= _FINAL_FP_MAX_ANTIGRAVITY,
        ),
        CriterionOutcomeAntigravity(
            "final_f1",
            ">",
            _FINAL_F1_STRICTLY_ABOVE_ANTIGRAVITY,
            final.f1,
            final.f1 > _FINAL_F1_STRICTLY_ABOVE_ANTIGRAVITY,
        ),
    )
    passed = all(item.passed for item in criteria)
    status = (
        "routing_candidate_qualified_for_external_validation"
        if passed
        else "routing_candidate_not_qualified_for_external_validation"
    )
    return criteria, passed, status


def evaluate_routes_antigravity(
    authority: Exp456ResultAuthorityAntigravity,
    decision: RouteDecisionAntigravity,
) -> EvaluationResultAntigravity:
    rows, features = _validate_authority_antigravity(authority)
    decision, qualification, feature = _validate_decision_antigravity(decision, features)
    baseline = tuple(row.baseline_confusion for row in rows)
    family_wide = tuple(
        row.candidate_confusion if row.candidate_confusion is not None else row.baseline_confusion
        for row in rows
    )
    oracle = tuple(
        row.candidate_confusion
        if row.candidate_confusion is not None and _prefer_candidate_antigravity(row.baseline_confusion, row.candidate_confusion)
        else row.baseline_confusion
        for row in rows
    )
    routed = tuple(
        row.candidate_confusion
        if (row.family == "ChlorineConcentration" or (row.dataset == FROZEN_CONFIG_ANTIGRAVITY.dataset and decision.route == "candidate"))
        else row.baseline_confusion
        for row in rows
    )
    comparisons = (
        _comparison_local_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[0], baseline),
        _comparison_local_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[1], family_wide),
        _comparison_local_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[2], oracle),
        _comparison_local_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[3], routed),
    )
    if comparisons[:3] != _expected_predecessor_antigravity():
        raise ValueError("computed Exp456 comparisons differ from immutable values")
    routed_by_dataset = dict(zip((row.dataset for row in rows), routed, strict=True))
    family_metrics = tuple(
        FamilyMetricAntigravity(
            family,
            sum(routed_by_dataset[row.dataset].fn - row.baseline_confusion.fn for row in rows if row.family == family),
            sum(routed_by_dataset[row.dataset].fp - row.baseline_confusion.fp for row in rows if row.family == family),
        )
        for family in _FAMILIES_ANTIGRAVITY
    )
    criteria, passed, status = _criteria_for_final_antigravity(comparisons[-1])
    result = EvaluationResultAntigravity(
        comparisons,
        family_metrics,
        (decision,),
        criteria,
        passed,
        status,
    )
    _EVIDENCE_BY_RESULT_ID_ANTIGRAVITY[id(result)] = (result, authority, qualification, feature)
    return result


def validate_evaluation_result_semantics_antigravity(
    result: EvaluationResultAntigravity,
) -> EvaluationResultAntigravity:
    if type(result) is not EvaluationResultAntigravity:
        raise ValueError("evaluation result type is invalid")
    evidence = _EVIDENCE_BY_RESULT_ID_ANTIGRAVITY.get(id(result))
    if evidence is None or evidence[0] is not result:
        raise ValueError("evaluation result lacks semantic provenance")
    _stored, authority, qualification, feature = evidence
    decision = route_normal2_antigravity(feature, qualification)
    expected = evaluate_routes_antigravity(authority, decision)
    if expected != result:
        raise ValueError("evaluation result differs from semantic recomputation")
    return result


def _evidence_for_result_antigravity(
    result: EvaluationResultAntigravity,
) -> tuple[Exp456ResultAuthorityAntigravity, RankQualificationAntigravity, DatasetRoutingFeaturesAntigravity]:
    validate_evaluation_result_semantics_antigravity(result)
    _stored, authority, qualification, feature = _EVIDENCE_BY_RESULT_ID_ANTIGRAVITY[id(result)]
    return authority, qualification, feature


__all__ = (
    "capture_exp456_result_authority_antigravity",
    "evaluate_routes_antigravity",
    "validate_evaluation_result_semantics_antigravity",
)
