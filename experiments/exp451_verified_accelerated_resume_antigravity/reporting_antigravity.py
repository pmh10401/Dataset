"""Exp451 report wrapper preserving frozen Exp450 tables and curve rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity import (
    Exp450ReportArtifactAntigravity,
    write_exp450_report_antigravity,
)

from .contracts_antigravity import canonical_json_bytes
from .artifacts_antigravity import _checked_target_path_antigravity, _publish_create_only_antigravity
from .config_antigravity import REPORT_ROOT
from .evaluation_antigravity import Exp451EvaluationAntigravity, require_exp451_evaluation_ready_antigravity
from .verifier_antigravity import Exp451VerificationAntigravity, _validate_verification_authority_antigravity, provenance_summary_antigravity


@dataclass(frozen=True, slots=True)
class Exp451ReportArtifactAntigravity:
    """Frozen Exp450 figure/table leaves plus an Exp451 origin-provenance leaf."""

    frozen_report: Exp450ReportArtifactAntigravity
    provenance_path: Path


def write_exp451_report_antigravity(
    evaluation: Exp451EvaluationAntigravity, root: Path, *, verification: Exp451VerificationAntigravity,
) -> Exp451ReportArtifactAntigravity:
    """Write unchanged comparison figures only after the 4,468-result gate closes."""
    require_exp451_evaluation_ready_antigravity(verification)
    verification = _validate_verification_authority_antigravity(verification)
    if (
        not isinstance(evaluation, Exp451EvaluationAntigravity)
        or evaluation.inventory_sha256 != verification.inventory_sha256
        or evaluation.context_sha256 != verification.context_sha256
    ):
        raise ValueError("Exp451 report evaluation is not bound to the verified inventory")
    if Path(root) != REPORT_ROOT or Path(root).is_symlink():
        raise ValueError("Exp451 report root differs from the frozen configured root")
    frozen = write_exp450_report_antigravity(evaluation.frozen_evaluation, Path(root))
    summary = provenance_summary_antigravity(verification)
    payload = {
        "schema": "exp451-report-provenance-antigravity.v1",
        "experiment": "Exp451",
        "frozen_metric_definition": "exp450_resampling_image_ablation_antigravity",
        "method_order": list(evaluation.frozen_evaluation.metric_tables.method_order),
        "curve_axes": {
            "roc": {"x": "false_positive_rate", "y": "true_positive_rate"},
            "precision_recall": {"x": "recall", "y": "precision"},
        },
        **summary,
    }
    data = canonical_json_bytes(payload)
    path = _checked_target_path_antigravity(
        frozen.root, "exp451_report_provenance_antigravity.json", create=True
    )
    _publish_create_only_antigravity(path, data)
    return Exp451ReportArtifactAntigravity(frozen, path)


__all__ = ["Exp451ReportArtifactAntigravity", "write_exp451_report_antigravity"]
