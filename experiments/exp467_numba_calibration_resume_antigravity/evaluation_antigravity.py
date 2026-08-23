"""Exp467 label gate delegating metric definitions to frozen Exp450 evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from weakref import WeakSet

from experiments.exp450_resampling_image_ablation_antigravity import evaluation_antigravity as _frozen_evaluation
from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import (
    METHOD_ORDER,
    Exp450EvaluationAntigravity,
    curve_tables_exp450_antigravity,
    metric_tables_exp450_antigravity,
)
from paper.equipment_anomaly_detection.numba_calibration_antigravity import (
    upper_tail_pvalues_numba_antigravity,
)

from .config_antigravity import EVALUATION_ROOT, EXP467_TOTAL_COORDINATES
from .contracts_antigravity import canonical_json_bytes
from .verifier_antigravity import (
    Exp467ScoreManifestsAntigravity,
    Exp467VerificationAntigravity,
    _score_manifest_authority_is_issued_antigravity,
    _score_manifests_digest_antigravity,
    _validate_persisted_verification_authority_antigravity,
    _validate_verification_authority_antigravity,
    provenance_summary_antigravity,
)


def _plain_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_antigravity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_antigravity(item) for item in value]
    return value


def _validated_score_manifests_antigravity(
    verification: Exp467VerificationAntigravity,
    score_authority: object,
) -> Mapping[str, Mapping[str, object]]:
    """Bind score vectors to each verified checkpoint manifest before labels."""
    if (
        not isinstance(score_authority, Exp467ScoreManifestsAntigravity)
        or not _score_manifest_authority_is_issued_antigravity(score_authority)
        or score_authority.inventory_sha256 != verification.inventory_sha256
        or score_authority.context_sha256 != verification.context_sha256
        or score_authority.score_manifests_sha256 != _score_manifests_digest_antigravity(score_authority.manifests)
    ):
        raise ValueError("Exp467 score manifest authority is invalid")
    score_manifests = score_authority.manifests
    if tuple(score_manifests) != METHOD_ORDER[:4]:
        raise ValueError("Exp467 score manifest order differs")
    checked: dict[str, Mapping[str, object]] = {}
    for condition in METHOD_ORDER[:4]:
        value = score_manifests[condition]
        base = verification.condition_manifests[condition]
        required = set(base) | {"score_schema", "base_manifest_sha256", "score_rows", "score_rows_sha256", "score_manifest_sha256"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("Exp467 score manifest schema differs")
        if any(value[key] != base[key] for key in base):
            raise ValueError("Exp467 score manifest base binding differs")
        rows = value["score_rows"]
        score_bare = {
            "score_schema": "exp467-verified-score-manifest-antigravity.v1",
            "base_manifest_sha256": base["manifest_sha256"],
            "score_rows": rows,
            "score_rows_sha256": value["score_rows_sha256"],
        }
        if (
            value.get("score_schema") != "exp467-verified-score-manifest-antigravity.v1"
            or value.get("base_manifest_sha256") != base["manifest_sha256"]
            or value.get("score_rows_sha256") != sha256(canonical_json_bytes(rows)).hexdigest()
            or value.get("score_manifest_sha256") != sha256(canonical_json_bytes(score_bare)).hexdigest()
            or not isinstance(rows, Mapping)
        ):
            raise ValueError("Exp467 score manifest hash differs")
        datasets = [entry["dataset"] for entry in base["datasets"]]
        if tuple(rows) != tuple(datasets):
            raise ValueError("Exp467 score manifest dataset identity differs")
        for entry in base["datasets"]:
            scores = rows[entry["dataset"]]
            expected = entry["test"]["count"]
            if (
                not isinstance(scores, list) or len(scores) != expected
                or any(type(score) is not float or not math.isfinite(score) for score in scores)
            ):
                raise ValueError("Exp467 score manifest score rows differ")
        checked[condition] = value
    return checked


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class Exp467EvaluationAntigravity:
    """Frozen metric result cryptographically bound to one verified inventory."""

    frozen_evaluation: Exp450EvaluationAntigravity
    inventory_sha256: str
    context_sha256: str
    score_manifests_sha256: str = ""
    evaluation_sha256: str = ""


def _evaluation_digest_antigravity(value: Exp450EvaluationAntigravity) -> str:
    return sha256(canonical_json_bytes(_plain_antigravity({
        "manifests": value.manifests,
        "metric_tables": {
            "method_order": value.metric_tables.method_order,
            "per_dataset": value.metric_tables.per_dataset,
            "macro": value.metric_tables.macro,
            "micro": value.metric_tables.micro,
            "paired_statistics": value.metric_tables.paired_statistics,
        },
        "curve_tables": value.curve_tables,
    }))).hexdigest()


def _validate_evaluation_antigravity(
    value: object, verification: Exp467VerificationAntigravity,
) -> Exp467EvaluationAntigravity:
    if (
        not isinstance(value, Exp467EvaluationAntigravity)
        or not _evaluation_authority_is_issued_antigravity(value)
        or value.inventory_sha256 != verification.inventory_sha256
        or value.context_sha256 != verification.context_sha256
        or not isinstance(value.frozen_evaluation, Exp450EvaluationAntigravity)
        or value.evaluation_sha256 != _evaluation_digest_antigravity(value.frozen_evaluation)
        or not isinstance(value.score_manifests_sha256, str) or len(value.score_manifests_sha256) != 64
    ):
        raise ValueError("Exp467 evaluation authority is not bound to verified evidence")
    return value


def require_exp467_evaluation_ready_antigravity(
    verification: Exp467VerificationAntigravity,
) -> None:
    """Close the TEST-label boundary until all results and four manifests verify."""
    verification = _validate_persisted_verification_authority_antigravity(verification)
    if verification.total_successes != EXP467_TOTAL_COORDINATES:
        raise ValueError("Exp467 evaluation requires exactly 4468 verified successes")
    if tuple(verification.condition_manifests) != ("A", "B", "C", "D"):
        raise ValueError("Exp467 evaluation requires four frozen condition manifests")
    summary = provenance_summary_antigravity(verification)
    if summary["imported_total"] != 4234 or summary["native_total"] != 234:
        raise ValueError("Exp467 evaluation origin totals differ")
    if {
        condition: summary["imported_by_condition"][condition] + summary["native_by_condition"][condition]
        for condition in ("A", "B", "C", "D")
    } != {"A": 1117, "B": 1117, "C": 1117, "D": 1117}:
        raise ValueError("Exp467 evaluation condition totals differ")


def _load_baseline_scores_numba_antigravity(
    manifests: Mapping[str, object], datasets: tuple[str, ...],
    test_identities: Mapping[str, Mapping[str, object]], database_sha256: str,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Local copy of only frozen baseline loading, with exact Numba float64 calibration."""
    raw_scores: dict[str, dict[str, object]] = {}
    micro_scores: dict[str, dict[str, object]] = {}
    for root_name in _frozen_evaluation._BASELINE_ROOT_ORDER:
        manifest = manifests[root_name]
        identity, db, shards = _frozen_evaluation._baseline_identity_and_shards(manifest)
        if db != database_sha256 or tuple(identity) != datasets:
            raise ValueError("baseline TEST identity differs")
        if isinstance(manifest, Mapping):
            if identity != test_identities:
                raise ValueError("baseline TEST identity differs")
        elif not _frozen_evaluation._same_test_coordinates(identity, test_identities):
            raise ValueError("baseline TEST identity differs")
        for method in _frozen_evaluation._BASELINE_METHODS[root_name]:
            raw_scores[method], micro_scores[method] = {}, {}
            for dataset in datasets:
                shard = shards.get((method, dataset))
                if shard is None:
                    raise ValueError("baseline score coverage differs")
                if shard.get("status") != "complete":
                    continue
                test = _frozen_evaluation._finite_vector(shard.get("test_scores"), "baseline TEST scores")
                train = _frozen_evaluation._finite_vector(shard.get("train_oof_scores"), "baseline TRAIN OOF scores")
                expected, indices = test_identities[dataset]["count"], shard.get("test_indices")
                if test.size != expected or indices != list(range(expected)):
                    raise ValueError("baseline TEST identity differs")
                raw_scores[method][dataset] = test
                micro_scores[method][dataset] = 1.0 - upper_tail_pvalues_numba_antigravity(train, test)
    return raw_scores, micro_scores


def evaluate_exp467_frozen_metrics_adapter_antigravity(
    root: Path, *, baseline_roots: Mapping[str, object] | None = None,
    verifier: Callable[[str], Mapping[str, object]] | None = None,
    label_loader: Callable[[tuple[str, ...]], Mapping[str, Sequence[object]]] | None = None,
    baseline_verifier: Callable[[Path], object] | None = None,
) -> Exp450EvaluationAntigravity:
    """Use frozen metric/curve definitions with the sole legacy calibration seam replaced."""
    if verifier is None:
        raise ValueError("Exp450 manifest verifier is required")
    manifests = {condition: verifier(condition) for condition in _frozen_evaluation._CONDITIONS}
    if any(not isinstance(value, Mapping) for value in manifests.values()):
        raise ValueError("Exp450 manifest verifier returned an invalid manifest")
    datasets, identities, database_sha256 = _frozen_evaluation._exp450_identity(manifests)
    roots = _frozen_evaluation.BASELINE_ROOTS if baseline_roots is None else baseline_roots
    verify_baseline = _frozen_evaluation.verify_score_root if baseline_verifier is None else baseline_verifier
    baseline_manifests = _frozen_evaluation._verify_baseline_roots(roots, verify_baseline)
    baseline_scores, baseline_micro = _load_baseline_scores_numba_antigravity(
        baseline_manifests, datasets, identities, database_sha256
    )
    exp_scores = _frozen_evaluation._load_exp450_scores(Path(root), manifests, datasets, identities)
    if label_loader is None:
        from experiments.paper_public_baselines.label_access import load_test_labels
        active_label_loader = load_test_labels
    else:
        active_label_loader = label_loader
    labels = _frozen_evaluation._normalize_labels(active_label_loader(datasets), datasets)
    all_scores = {**exp_scores, **baseline_scores}
    all_micro = {**exp_scores, **baseline_micro}
    return Exp450EvaluationAntigravity(
        manifests=_frozen_evaluation.MappingProxyType(dict(manifests)),
        metric_tables=metric_tables_exp450_antigravity(labels, all_scores, micro_scores=all_micro),
        curve_tables=curve_tables_exp450_antigravity(labels, all_micro),
    )


def _evaluate_exp467_impl_antigravity(
    root: Path, *, verification: Exp467VerificationAntigravity,
    score_manifests: Exp467ScoreManifestsAntigravity,
    baseline_roots: Mapping[str, object] | None = None,
    label_loader: Callable[[tuple[str, ...]], Mapping[str, Sequence[object]]] | None = None,
    baseline_verifier: Callable[[Path], object] | None = None, issue: object,
) -> Exp467EvaluationAntigravity:
    """Evaluate only after the Exp467 gate, using unchanged Exp450 metric definitions."""
    require_exp467_evaluation_ready_antigravity(verification)
    if Path(root) != EVALUATION_ROOT:
        raise ValueError("Exp467 evaluation root differs from the frozen configured root")
    frozen = _validated_score_manifests_antigravity(verification, score_manifests)
    def verifier(condition: str) -> Mapping[str, object]:
        manifest = frozen.get(condition)
        if not isinstance(manifest, Mapping):
            raise ValueError("Exp467 score manifest differs")
        return manifest
    kwargs: dict[str, object] = {
        "baseline_roots": baseline_roots,
        "verifier": verifier,
        "label_loader": label_loader,
    }
    if baseline_verifier is not None:
        kwargs["baseline_verifier"] = baseline_verifier
    frozen_evaluation = evaluate_exp467_frozen_metrics_adapter_antigravity(Path(root), **kwargs)
    score_digest = sha256(canonical_json_bytes(_plain_antigravity(frozen))).hexdigest()
    authority = Exp467EvaluationAntigravity(
        frozen_evaluation, verification.inventory_sha256, verification.context_sha256,
        score_digest, _evaluation_digest_antigravity(frozen_evaluation),
    )
    assert callable(issue)
    return issue(authority)


def _make_evaluation_api_antigravity():
    """Issue evaluation evidence only from the full label-gated evaluator."""
    issued: WeakSet[object] = WeakSet()
    def is_issued(value: object) -> bool:
        return value in issued
    def evaluate(
        root: Path, *, verification: Exp467VerificationAntigravity,
        score_manifests: Exp467ScoreManifestsAntigravity,
        baseline_roots: Mapping[str, object] | None = None,
        label_loader: Callable[[tuple[str, ...]], Mapping[str, Sequence[object]]] | None = None,
        baseline_verifier: Callable[[Path], object] | None = None,
    ) -> Exp467EvaluationAntigravity:
        def issue(value: Exp467EvaluationAntigravity) -> Exp467EvaluationAntigravity:
            issued.add(value)
            return value
        return _evaluate_exp467_impl_antigravity(
            root, verification=verification, score_manifests=score_manifests,
            baseline_roots=baseline_roots, label_loader=label_loader,
            baseline_verifier=baseline_verifier, issue=issue,
        )
    return evaluate, is_issued


evaluate_exp467_antigravity, _evaluation_authority_is_issued_antigravity = _make_evaluation_api_antigravity()
del _make_evaluation_api_antigravity


def curve_tables_exp467_antigravity(
    labels: Mapping[str, Sequence[object]], scores: Mapping[str, Mapping[str, Sequence[object]]],
):
    """Frozen ROC/PR implementation: ROC=(FPR,TPR), PR=(recall,precision)."""
    return curve_tables_exp450_antigravity(labels, scores)


def metric_tables_exp467_antigravity(*args: object, **kwargs: object):
    """Expose the unchanged frozen AP/AUROC implementation under Exp467 identity."""
    return metric_tables_exp450_antigravity(*args, **kwargs)


__all__ = [
    "METHOD_ORDER", "Exp450EvaluationAntigravity", "Exp467EvaluationAntigravity", "curve_tables_exp467_antigravity",
    "evaluate_exp467_antigravity", "metric_tables_exp467_antigravity",
    "require_exp467_evaluation_ready_antigravity",
]
