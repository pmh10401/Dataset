"""Exp451 label gate delegating metric definitions to frozen Exp450 evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import (
    METHOD_ORDER,
    Exp450EvaluationAntigravity,
    curve_tables_exp450_antigravity,
    evaluate_exp450_antigravity,
    metric_tables_exp450_antigravity,
)

from .config_antigravity import EVALUATION_ROOT, EXP451_TOTAL_COORDINATES
from .verifier_antigravity import (
    Exp451VerificationAntigravity,
    _validate_verification_authority_antigravity,
    provenance_summary_antigravity,
)


@dataclass(frozen=True, slots=True)
class Exp451EvaluationAntigravity:
    """Frozen metric result cryptographically bound to one verified inventory."""

    frozen_evaluation: Exp450EvaluationAntigravity
    inventory_sha256: str
    context_sha256: str


def require_exp451_evaluation_ready_antigravity(
    verification: Exp451VerificationAntigravity,
) -> None:
    """Close the TEST-label boundary until all results and four manifests verify."""
    verification = _validate_verification_authority_antigravity(verification)
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


def evaluate_exp451_antigravity(
    root: Path, *, verification: Exp451VerificationAntigravity,
    score_manifests: Mapping[str, Mapping[str, object]],
    baseline_roots: Mapping[str, object] | None = None,
    label_loader: Callable[[tuple[str, ...]], Mapping[str, Sequence[object]]] | None = None,
    baseline_verifier: Callable[[Path], object] | None = None,
) -> Exp451EvaluationAntigravity:
    """Evaluate only after the Exp451 gate, using unchanged Exp450 metric definitions."""
    require_exp451_evaluation_ready_antigravity(verification)
    if Path(root) != EVALUATION_ROOT:
        raise ValueError("Exp451 evaluation root differs from the frozen configured root")
    if tuple(score_manifests) != METHOD_ORDER[:4]:
        raise ValueError("Exp451 score manifest order differs")
    frozen = dict(score_manifests)
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
    return Exp451EvaluationAntigravity(
        frozen_evaluation, verification.inventory_sha256, verification.context_sha256,
    )


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
