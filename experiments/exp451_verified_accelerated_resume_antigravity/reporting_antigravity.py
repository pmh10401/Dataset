"""Exp451 report wrapper preserving frozen Exp450 tables and curve rendering."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path

from experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity import (
    Exp450ReportArtifactAntigravity,
    write_exp450_report_antigravity,
)

from .contracts_antigravity import canonical_json_bytes
from .evaluation_antigravity import require_exp451_evaluation_ready_antigravity
from .verifier_antigravity import Exp451VerificationAntigravity, provenance_summary_antigravity


@dataclass(frozen=True, slots=True)
class Exp451ReportArtifactAntigravity:
    """Frozen Exp450 figure/table leaves plus an Exp451 origin-provenance leaf."""

    frozen_report: Exp450ReportArtifactAntigravity
    provenance_path: Path


def _write_create_only_antigravity(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("Exp451 report artifact already exists with different bytes") from None
        return
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Exp451 report artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exp451_report_antigravity(
    evaluation: object, root: Path, *, verification: Exp451VerificationAntigravity,
) -> Exp451ReportArtifactAntigravity:
    """Write unchanged comparison figures only after the 4,468-result gate closes."""
    require_exp451_evaluation_ready_antigravity(verification)
    frozen = write_exp450_report_antigravity(evaluation, Path(root))
    summary = provenance_summary_antigravity(verification)
    payload = {
        "schema": "exp451-report-provenance-antigravity.v1",
        "experiment": "Exp451",
        "frozen_metric_definition": "exp450_resampling_image_ablation_antigravity",
        "method_order": list(evaluation.metric_tables.method_order),
        "curve_axes": {
            "roc": {"x": "false_positive_rate", "y": "true_positive_rate"},
            "precision_recall": {"x": "recall", "y": "precision"},
        },
        **summary,
    }
    data = canonical_json_bytes(payload)
    path = frozen.root / "exp451_report_provenance_antigravity.json"
    _write_create_only_antigravity(path, data)
    return Exp451ReportArtifactAntigravity(frozen, path)


__all__ = ["Exp451ReportArtifactAntigravity", "write_exp451_report_antigravity"]
