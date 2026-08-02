"""Post-hoc, prediction-first TEST evaluation for Exp422."""

from .audit import compute_confusion_audit, metrics_from_counts
from .config import FROZEN_CONFIG, Exp422TestEvalConfig
from .contracts import (
    AuditResult,
    ConfusionCounts,
    ConfusionMetrics,
    FrozenLabels,
    FrozenPrediction,
    LabelRow,
)

__all__ = [
    "AuditResult",
    "ConfusionCounts",
    "ConfusionMetrics",
    "FROZEN_CONFIG",
    "FrozenLabels",
    "FrozenPrediction",
    "Exp422TestEvalConfig",
    "LabelRow",
    "compute_confusion_audit",
    "metrics_from_counts",
]
