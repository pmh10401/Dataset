"""Frozen data contracts for the Exp403-408 validation workflow.

The prediction validator is deliberately independent of experiment-local
loaders and evaluators.  Prediction artifacts may describe TEST scores, but
they must not contain labels or metrics computed from those labels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Real
import re
from typing import Any


SEED = 20260727
ALPHAS = (0.005, 0.01, 0.02, 0.05)
EXPECTED_DATASETS = 1117
EXPECTED_HARD = 2218
EXPECTED_STANDARD = 806
EXPECTED_PRIORITY = 31
ALPHA_KEYS = tuple(str(alpha) for alpha in ALPHAS)


@dataclass(frozen=True)
class TrainEligibility:
    level: str
    robust_distance: bool
    classical_committee: bool
    learned_methods: bool


@dataclass(frozen=True)
class DatasetInput:
    dataset: str
    train_normal: Any
    test_values: Any
    target_length: int
    hard_indices: frozenset[int] | Sequence[int] = field(default_factory=frozenset)
    standard_indices: frozenset[int] | Sequence[int] = field(default_factory=frozenset)
    priority_indices: frozenset[int] | Sequence[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Fold:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    group: str
    train_oof: Any
    test_values: Any
    source_hash: str
    available: bool = True
    reason: str = ""


@dataclass(frozen=True)
class FeatureStability:
    """TRAIN OOF measurements used to retain one stable feature per component."""

    feature_name: str
    group: str
    finite_rate: float
    non_finite_rate: float
    median: float
    mad: float
    robust_variance: float
    fold_center_drift: float
    bootstrap_rank_stability: float
    coverage: float
    empirical_p_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class FeatureGroupSummary:
    group: str
    feature_names: tuple[str, ...] = ()
    minimum: float = 0.0
    median: float = 0.0
    robust_mean: float = 0.0
    maximum: float = 0.0
    exceedance_count: int = 0
    top_two_gap: float = 0.0
    stability: float = 0.0


@dataclass(frozen=True)
class StabilityAudit:
    excluded: Mapping[str, str] = field(default_factory=dict)
    correlation_components: tuple[tuple[str, ...], ...] = ()
    correlation_threshold: float = 0.90
    representatives: tuple[str, ...] = ()
    feature_stability: Mapping[str, FeatureStability] = field(default_factory=dict)
    group_summaries: Mapping[str, FeatureGroupSummary] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitteeScores:
    scores_by_alpha: Mapping[str, Sequence[float]] = field(default_factory=dict)
    available_models: tuple[str, ...] = ()
    confidence_by_alpha: Mapping[str, Sequence[str]] = field(default_factory=dict)
    hard_indices: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class FrozenConfig:
    seed: int = SEED
    alphas: tuple[float, ...] = ALPHAS
    target_length: int = 512


@dataclass(frozen=True)
class PredictionRow:
    dataset: str
    candidate_index: int
    scores_by_alpha: Mapping[str, float]
    contains_test_labels: bool = False
    contains_evaluation_metrics: bool = False
    confidence_by_alpha: Mapping[str, str] = field(default_factory=dict)
    eligibility: str = ""
    abstention_reason: str = ""


@dataclass(frozen=True)
class DemotionEvaluation:
    recheck_count: int = 0
    false_positive_count: int = 0
    true_positive_count: int = 0
    false_positive_fraction: float = 0.0
    baseline_true_positive_retention: float = 0.0
    train_stability: float = 0.0
    operational_policy_changed: bool = False


@dataclass(frozen=True)
class LearnedSourceScores:
    scores_by_alpha: Mapping[str, Sequence[float]] = field(default_factory=dict)
    available: bool = False
    abstention_reason: str = ""


@dataclass(frozen=True)
class ShadowSourcePrediction:
    dataset: str
    candidate_index: int
    scores_by_alpha: Mapping[str, float]
    source: str = "self_supervised_shadow"


def train_eligibility(n_train: int) -> TrainEligibility:
    """Return the frozen method lane allowed by the TRAIN sample count."""
    if n_train < 5:
        return TrainEligibility("abstain", False, False, False)
    if n_train < 20:
        return TrainEligibility("robust_distance", True, False, False)
    if n_train < 50:
        return TrainEligibility("classical_committee", True, True, False)
    return TrainEligibility("learned_methods", True, True, True)


_FORBIDDEN_EXACT = frozenset(
    {
        "y_test",
        "test_labels",
        "true_label",
        "is_true_anomaly",
        "actual_anomaly_positions",
        "labels",
        "hard_metrics",
        "evaluation_metrics",
    }
)
_FORBIDDEN_SUFFIXES = (
    "_tp",
    "_fp",
    "_fn",
    "_f1",
    "_precision",
    "_recall",
)
_TEST_EVALUATION_TERMS = frozenset(
    {
        "label",
        "labels",
        "accuracy",
        "auc",
        "precision",
        "recall",
        "f1",
        "loss",
        "metric",
        "metrics",
    }
)
_ALLOWED_TEST_DECLARATIONS = frozenset({"contains_test_labels"})


def _is_forbidden_key(key: object) -> bool:
    lowered = str(key).lower()
    if lowered in _ALLOWED_TEST_DECLARATIONS:
        return False
    if lowered in _FORBIDDEN_EXACT or lowered.endswith(_FORBIDDEN_SUFFIXES):
        return True
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    return "test" in tokens and bool(tokens & _TEST_EVALUATION_TERMS)


def _scan_forbidden(value: object, path: str = "prediction") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_forbidden_key(key):
                raise ValueError(f"forbidden prediction field: {path}.{key}")
            _scan_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_forbidden(child, f"{path}[{index}]")


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{path} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _scan_finite_numbers(value: object, path: str = "prediction") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _scan_finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_finite_numbers(child, f"{path}[{index}]")
    elif isinstance(value, (Real, Decimal)) and not isinstance(value, bool):
        _finite_number(value, path)


def validate_prediction_row(row: Mapping[str, object]) -> dict[str, object]:
    """Validate and return a copy of a label-free prediction row."""
    if not isinstance(row, Mapping):
        raise ValueError("prediction row must be a mapping")
    _scan_forbidden(row)
    _scan_finite_numbers(row)
    if row.get("contains_test_labels") is not False:
        raise ValueError("prediction must declare contains_test_labels=false")
    if row.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction must declare contains_evaluation_metrics=false")

    scores = row.get("scores_by_alpha")
    if not isinstance(scores, Mapping) or set(scores) != set(ALPHA_KEYS):
        raise ValueError("scores_by_alpha must contain exactly the four alpha keys")
    for alpha in ALPHA_KEYS:
        _finite_number(scores[alpha], f"scores_by_alpha.{alpha}")
    return dict(row)
