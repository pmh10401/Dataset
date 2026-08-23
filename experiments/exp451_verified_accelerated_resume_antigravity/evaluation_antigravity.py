"""Exp451 label gate delegating metric definitions to frozen Exp450 evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from weakref import WeakSet

from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import (
    METHOD_ORDER,
    Exp450EvaluationAntigravity,
    curve_tables_exp450_antigravity,
    evaluate_exp450_antigravity,
    metric_tables_exp450_antigravity,
)

from .config_antigravity import EVALUATION_ROOT, EXP451_TOTAL_COORDINATES
from .contracts_antigravity import canonical_json_bytes
from .verifier_antigravity import (
    Exp451ScoreManifestsAntigravity,
    Exp451VerificationAntigravity,
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
    verification: Exp451VerificationAntigravity,
    score_authority: object,
) -> Mapping[str, Mapping[str, object]]:
    """Bind score vectors to each verified checkpoint manifest before labels."""
    if (
        not isinstance(score_authority, Exp451ScoreManifestsAntigravity)
        or not _score_manifest_authority_is_issued_antigravity(score_authority)
        or score_authority.inventory_sha256 != verification.inventory_sha256
        or score_authority.context_sha256 != verification.context_sha256
        or score_authority.score_manifests_sha256 != _score_manifests_digest_antigravity(score_authority.manifests)
    ):
        raise ValueError("Exp451 score manifest authority is invalid")
    score_manifests = score_authority.manifests
    if tuple(score_manifests) != METHOD_ORDER[:4]:
        raise ValueError("Exp451 score manifest order differs")
    checked: dict[str, Mapping[str, object]] = {}
    for condition in METHOD_ORDER[:4]:
        value = score_manifests[condition]
        base = verification.condition_manifests[condition]
        required = set(base) | {"score_schema", "base_manifest_sha256", "score_rows", "score_rows_sha256", "score_manifest_sha256"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("Exp451 score manifest schema differs")
        if any(value[key] != base[key] for key in base):
            raise ValueError("Exp451 score manifest base binding differs")
        rows = value["score_rows"]
        score_bare = {
            "score_schema": "exp451-verified-score-manifest-antigravity.v1",
            "base_manifest_sha256": base["manifest_sha256"],
            "score_rows": rows,
            "score_rows_sha256": value["score_rows_sha256"],
        }
        if (
            value.get("score_schema") != "exp451-verified-score-manifest-antigravity.v1"
            or value.get("base_manifest_sha256") != base["manifest_sha256"]
            or value.get("score_rows_sha256") != sha256(canonical_json_bytes(rows)).hexdigest()
            or value.get("score_manifest_sha256") != sha256(canonical_json_bytes(score_bare)).hexdigest()
            or not isinstance(rows, Mapping)
        ):
            raise ValueError("Exp451 score manifest hash differs")
        datasets = [entry["dataset"] for entry in base["datasets"]]
        if tuple(rows) != tuple(datasets):
            raise ValueError("Exp451 score manifest dataset identity differs")
        for entry in base["datasets"]:
            scores = rows[entry["dataset"]]
            expected = entry["test"]["count"]
            if (
                not isinstance(scores, list) or len(scores) != expected
                or any(type(score) is not float or not math.isfinite(score) for score in scores)
            ):
                raise ValueError("Exp451 score manifest score rows differ")
        checked[condition] = value
    return checked


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class Exp451EvaluationAntigravity:
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
    value: object, verification: Exp451VerificationAntigravity,
) -> Exp451EvaluationAntigravity:
    if (
        not isinstance(value, Exp451EvaluationAntigravity)
        or not _evaluation_authority_is_issued_antigravity(value)
        or value.inventory_sha256 != verification.inventory_sha256
        or value.context_sha256 != verification.context_sha256
        or not isinstance(value.frozen_evaluation, Exp450EvaluationAntigravity)
        or value.evaluation_sha256 != _evaluation_digest_antigravity(value.frozen_evaluation)
        or not isinstance(value.score_manifests_sha256, str) or len(value.score_manifests_sha256) != 64
    ):
        raise ValueError("Exp451 evaluation authority is not bound to verified evidence")
    return value


def require_exp451_evaluation_ready_antigravity(
    verification: Exp451VerificationAntigravity,
) -> None:
    """Close the TEST-label boundary until all results and four manifests verify."""
    verification = _validate_persisted_verification_authority_antigravity(verification)
    if verification.total_successes != EXP451_TOTAL_COORDINATES:
        raise ValueError("Exp451 evaluation requires exactly 4468 verified successes")
    if tuple(verification.condition_manifests) != ("A", "B", "C", "D"):
        raise ValueError("Exp451 evaluation requires four frozen condition manifests")
    summary = provenance_summary_antigravity(verification)
    if summary["imported_total"] != 4234 or summary["native_total"] != 234:
        raise ValueError("Exp451 evaluation origin totals differ")
    if {
        condition: summary["imported_by_condition"][condition] + summary["native_by_condition"][condition]
        for condition in ("A", "B", "C", "D")
    } != {"A": 1117, "B": 1117, "C": 1117, "D": 1117}:
        raise ValueError("Exp451 evaluation condition totals differ")


def _evaluate_exp451_impl_antigravity(
    root: Path, *, verification: Exp451VerificationAntigravity,
    score_manifests: Exp451ScoreManifestsAntigravity,
    baseline_roots: Mapping[str, object] | None = None,
    label_loader: Callable[[tuple[str, ...]], Mapping[str, Sequence[object]]] | None = None,
    baseline_verifier: Callable[[Path], object] | None = None, issue: object,
) -> Exp451EvaluationAntigravity:
    """Evaluate only after the Exp451 gate, using unchanged Exp450 metric definitions."""
    require_exp451_evaluation_ready_antigravity(verification)
    if Path(root) != EVALUATION_ROOT:
        raise ValueError("Exp451 evaluation root differs from the frozen configured root")
    frozen = _validated_score_manifests_antigravity(verification, score_manifests)
    def verifier(condition: str) -> Mapping[str, object]:
        manifest = frozen.get(condition)
        if not isinstance(manifest, Mapping):
            raise ValueError("Exp451 score manifest differs")
        return manifest
    kwargs: dict[str, object] = {
        "baseline_roots": baseline_roots,
        "verifier": verifier,
        "label_loader": label_loader,
    }
    if baseline_verifier is not None:
        kwargs["baseline_verifier"] = baseline_verifier
    frozen_evaluation = evaluate_exp450_antigravity(Path(root), **kwargs)
    score_digest = sha256(canonical_json_bytes(_plain_antigravity(frozen))).hexdigest()
    authority = Exp451EvaluationAntigravity(
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
        root: Path, *, verification: Exp451VerificationAntigravity,
        score_manifests: Exp451ScoreManifestsAntigravity,
        baseline_roots: Mapping[str, object] | None = None,
        label_loader: Callable[[tuple[str, ...]], Mapping[str, Sequence[object]]] | None = None,
        baseline_verifier: Callable[[Path], object] | None = None,
    ) -> Exp451EvaluationAntigravity:
        def issue(value: Exp451EvaluationAntigravity) -> Exp451EvaluationAntigravity:
            issued.add(value)
            return value
        return _evaluate_exp451_impl_antigravity(
            root, verification=verification, score_manifests=score_manifests,
            baseline_roots=baseline_roots, label_loader=label_loader,
            baseline_verifier=baseline_verifier, issue=issue,
        )
    return evaluate, is_issued


evaluate_exp451_antigravity, _evaluation_authority_is_issued_antigravity = _make_evaluation_api_antigravity()
del _make_evaluation_api_antigravity


def curve_tables_exp451_antigravity(
    labels: Mapping[str, Sequence[object]], scores: Mapping[str, Mapping[str, Sequence[object]]],
):
    """Frozen ROC/PR implementation: ROC=(FPR,TPR), PR=(recall,precision)."""
    return curve_tables_exp450_antigravity(labels, scores)


def metric_tables_exp451_antigravity(*args: object, **kwargs: object):
    """Expose the unchanged frozen AP/AUROC implementation under Exp451 identity."""
    return metric_tables_exp450_antigravity(*args, **kwargs)


__all__ = [
    "METHOD_ORDER", "Exp450EvaluationAntigravity", "Exp451EvaluationAntigravity", "curve_tables_exp451_antigravity",
    "evaluate_exp451_antigravity", "metric_tables_exp451_antigravity",
    "require_exp451_evaluation_ready_antigravity",
]
