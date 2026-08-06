"""Hash-gated retrospective evaluation for Exp436.

This module is the only place in Exp436 that can open TEST labels.  It is
deliberately a post-freeze boundary: a publication is replayed from its
read-only checkpoint, every prediction/config/input identity is checked, and
only then is the private SQLite label loader called.  The evaluator writes
separate artifacts and never mutates the prediction publication.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import stat
from hashlib import sha256
from tempfile import mkdtemp
from types import MappingProxyType
from typing import Any

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.inputs import (
    verified_read_only_connection,
)

from .checkpoint_antigravity import Exp436CheckpointStore
from .config_antigravity import FROZEN_CONFIG, Exp436FrozenConfig
from .reporting_antigravity import (
    HASH_FILE as PUBLICATION_HASH_FILE,
    MANIFEST_FILE,
    PREDICTION_FILE,
    PUBLICATION_DIRECTORY,
    STATUS_FILE,
    SUMMARY_FILE,
    FrozenPublication,
    _PublicationExpectations,
    _inspect_publication,
    _exclusive_rename,
    inspect_publication,
)


DATABASE_PATH = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
EXP404_HARD_PATH = Path(
    "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/"
    "exp404_hard_confidence.jsonl"
)
EXP291_HARD_PATH = Path(
    "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/"
    "exp291_frozen_operational_routing.jsonl"
)
EVALUATION_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp436_missing_safe_strict_consensus_antigravity_20260806/"
    "evaluation_antigravity"
)

EVALUATION_FILE = "evaluation_antigravity.json"
DATASET_FILE = "dataset_metrics_antigravity.jsonl"
RESCUED_FILE = "rescued_datasets_antigravity.jsonl"
REGRESSED_FILE = "regressed_datasets_antigravity.jsonl"
CONFUSION_FILE = "confusion_antigravity.json"
BOOTSTRAP_FILE = "bootstrap_antigravity.json"
DECISION_FILE = "decision_antigravity.json"
EVALUATION_HASH_FILE = "artifact_sha256_antigravity.json"
_EVALUATION_FILES = frozenset(
    {
        EVALUATION_FILE,
        DATASET_FILE,
        RESCUED_FILE,
        REGRESSED_FILE,
        CONFUSION_FILE,
        BOOTSTRAP_FILE,
        DECISION_FILE,
        EVALUATION_HASH_FILE,
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_LABEL_LOADER_DEFAULT = object()
_EXP404_EXPECTED_COORDINATES = 2_218
_EXP404_EXPECTED_DATASETS = 888
_PINNED_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _jsonl(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _sha256_file(path: Path) -> str:
    """Compatibility wrapper for callers that need only a pinned digest."""

    return _read_pinned_bytes(Path(path))[1]


def _stable_inode_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_pinned_bytes(path: Path) -> tuple[bytes, str]:
    """Capture bytes and SHA256 from one stable, non-symlink regular inode."""

    path = Path(path)
    try:
        descriptor = os.open(path, _PINNED_READ_FLAGS)
    except OSError as error:
        raise ValueError(f"frozen artifact is absent or invalid: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            raise ValueError(f"frozen artifact is not a linked regular file: {path}")
        digest = sha256()
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        if _stable_inode_metadata(before) != _stable_inode_metadata(after):
            raise ValueError(f"frozen artifact inode changed while reading: {path}")
        return b"".join(chunks), digest.hexdigest()
    except OSError as error:
        raise ValueError(f"frozen artifact cannot be read: {path}") from error
    finally:
        os.close(descriptor)


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _f1(counts: "ConfusionCounts") -> float:
    denominator = 2 * counts.tp + counts.fp + counts.fn
    return 0.0 if denominator == 0 else (2.0 * counts.tp) / denominator


@dataclass(frozen=True)
class ConfusionCounts:
    """Point-level confusion counts for one dataset or an aggregate."""

    tp: int
    fp: int
    fn: int
    tn: int

    def __post_init__(self) -> None:
        for name in ("tp", "fp", "fn", "tn"):
            _require_int(getattr(self, name), name)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def f1(self) -> float:
        return _f1(self)

    def to_payload(self) -> dict[str, object]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = _require_float(self.lower, "confidence interval lower")
        upper = _require_float(self.upper, "confidence interval upper")
        if lower > upper:
            raise ValueError("confidence interval lower exceeds upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_payload(self) -> dict[str, float]:
        return {"lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class DatasetComparison:
    """Paired Hard versus Hard+Exp436 evidence for one dataset."""

    dataset: str
    test_size: int
    baseline: ConfusionCounts
    combined: ConfusionCounts

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError("dataset must be a non-empty string")
        _require_int(self.test_size, "test_size")
        if not isinstance(self.baseline, ConfusionCounts) or not isinstance(self.combined, ConfusionCounts):
            raise ValueError("baseline and combined must be ConfusionCounts")
        if self.baseline.total != self.test_size or self.combined.total != self.test_size:
            raise ValueError("confusion counts must conserve test_size")
        if (
            self.combined.tp < self.baseline.tp
            or self.combined.fp < self.baseline.fp
            or self.combined.fn > self.baseline.fn
            or self.combined.tn > self.baseline.tn
        ):
            raise ValueError("combined confusion counts violate baseline monotonic invariants")

    @property
    def baseline_f1(self) -> float:
        return self.baseline.f1

    @property
    def baseline_counts(self) -> ConfusionCounts:
        return self.baseline

    @property
    def combined_f1(self) -> float:
        return self.combined.f1

    @property
    def combined_counts(self) -> ConfusionCounts:
        return self.combined

    @property
    def rescued(self) -> bool:
        return self.baseline.tp == 0 and self.combined.tp >= 1

    @property
    def regressed(self) -> bool:
        return self.combined_f1 < self.baseline_f1

    @property
    def strong_baseline(self) -> bool:
        return self.baseline_f1 >= 0.8

    @property
    def added_fp(self) -> int:
        return self.combined.fp - self.baseline.fp

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "test_size": self.test_size,
            "baseline": self.baseline.to_payload(),
            "combined": self.combined.to_payload(),
            "baseline_f1": self.baseline_f1,
            "combined_f1": self.combined_f1,
            "f1_delta": self.combined_f1 - self.baseline_f1,
            "rescued": self.rescued,
            "regressed": self.regressed,
            "strong_baseline": self.strong_baseline,
            "added_fp": self.added_fp,
        }


@dataclass(frozen=True)
class BootstrapResult:
    """Deterministic paired dataset-cluster bootstrap intervals."""

    repetitions: int
    seed: int
    micro_f1_delta_ci: ConfidenceInterval
    macro_f1_delta_ci: ConfidenceInterval

    def __post_init__(self) -> None:
        _require_int(self.repetitions, "bootstrap repetitions", minimum=1)
        _require_int(self.seed, "bootstrap seed", minimum=0)
        if not isinstance(self.micro_f1_delta_ci, ConfidenceInterval) or not isinstance(self.macro_f1_delta_ci, ConfidenceInterval):
            raise ValueError("bootstrap confidence intervals are invalid")

    @property
    def micro_f1_ci(self) -> ConfidenceInterval:
        return self.micro_f1_delta_ci

    @property
    def repeat_count(self) -> int:
        return self.repetitions

    @property
    def iterations(self) -> int:
        return self.repetitions

    @property
    def macro_f1_ci(self) -> ConfidenceInterval:
        return self.macro_f1_delta_ci

    def to_payload(self) -> dict[str, object]:
        return {
            "repetitions": self.repetitions,
            "seed": self.seed,
            "confidence": "percentile_2.5_97.5",
            "micro_f1_delta_ci": self.micro_f1_delta_ci.to_payload(),
            "macro_f1_delta_ci": self.macro_f1_delta_ci.to_payload(),
            "micro_f1_ci": self.micro_f1_delta_ci.to_payload(),
            "macro_f1_ci": self.macro_f1_delta_ci.to_payload(),
        }


@dataclass(frozen=True)
class SuccessDecision:
    """Stable all-gates result; an empty reason list means success."""

    passed: bool
    reasons: tuple[str, ...]
    gates: Mapping[str, bool]

    def __post_init__(self) -> None:
        _require_bool(self.passed, "decision passed")
        reasons = tuple(self.reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("decision reasons are invalid")
        object.__setattr__(self, "reasons", reasons)
        if not isinstance(self.gates, Mapping):
            raise ValueError("decision gates are invalid")
        gates = dict(self.gates)
        if any(not isinstance(key, str) or type(value) is not bool for key, value in gates.items()):
            raise ValueError("decision gates are invalid")
        object.__setattr__(self, "gates", MappingProxyType(gates))
        if self.passed != (not reasons):
            raise ValueError("decision passed/reasons mismatch")

    @property
    def success(self) -> bool:
        return self.passed

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return self.reasons

    def to_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "success": self.passed,
            "reasons": list(self.reasons),
            "gates": dict(sorted(self.gates.items())),
        }


@dataclass(frozen=True)
class Exp436Evaluation:
    """Complete paired retrospective evaluation and provenance."""

    dataset_count: int
    candidate_count: int
    added_fp: int
    dataset_rows: tuple[DatasetComparison, ...]
    baseline_confusion: ConfusionCounts
    combined_confusion: ConfusionCounts
    micro_f1_delta: float
    macro_f1_delta: float
    micro_f1_ci: ConfidenceInterval
    macro_f1_ci: ConfidenceInterval
    rescued_datasets: tuple[str, ...]
    regressed_datasets: tuple[str, ...]
    strong_baseline_regressions: tuple[str, ...]
    prediction_sha256: str
    manifest_sha256: str
    prediction_label_access_count: int = 0
    label_rows: int = 0
    label_identity_sha256: str = ""
    database_sha256: str = ""
    code_commit: str = ""
    bootstrap: BootstrapResult | None = None

    def __post_init__(self) -> None:
        _require_int(self.dataset_count, "dataset_count")
        _require_int(self.candidate_count, "candidate_count")
        _require_int(self.added_fp, "added_fp")
        rows = tuple(self.dataset_rows)
        if any(not isinstance(row, DatasetComparison) for row in rows):
            raise ValueError("dataset_rows are invalid")
        if tuple(row.dataset for row in rows) != tuple(sorted(row.dataset for row in rows)) or len({row.dataset for row in rows}) != len(rows):
            raise ValueError("dataset_rows must be sorted and unique")
        if len(rows) != self.dataset_count:
            raise ValueError("dataset_count does not match dataset_rows")
        if self.baseline_confusion.total != sum(row.test_size for row in rows) or self.combined_confusion.total != sum(row.test_size for row in rows):
            raise ValueError("aggregate confusion counts do not conserve labels")
        if self.added_fp != self.combined_confusion.fp - self.baseline_confusion.fp:
            raise ValueError("added_fp does not match aggregate confusion counts")
        for name in ("micro_f1_delta", "macro_f1_delta"):
            _require_float(getattr(self, name), name)
        if not isinstance(self.micro_f1_ci, ConfidenceInterval) or not isinstance(self.macro_f1_ci, ConfidenceInterval):
            raise ValueError("evaluation confidence intervals are invalid")
        for name in ("rescued_datasets", "regressed_datasets", "strong_baseline_regressions"):
            values = tuple(getattr(self, name))
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise ValueError(f"{name} must be sorted and unique")
            object.__setattr__(self, name, values)
        for name in ("prediction_sha256", "manifest_sha256"):
            _require_digest(getattr(self, name), name)
        _require_int(self.prediction_label_access_count, "prediction_label_access_count")
        _require_int(self.label_rows, "label_rows")
        if self.label_identity_sha256:
            _require_digest(self.label_identity_sha256, "label_identity_sha256")
        if self.database_sha256:
            _require_digest(self.database_sha256, "database_sha256")
        if self.code_commit and _HEX40.fullmatch(self.code_commit) is None:
            raise ValueError("code_commit is invalid")
        if self.bootstrap is not None and not isinstance(self.bootstrap, BootstrapResult):
            raise ValueError("bootstrap result is invalid")
        object.__setattr__(self, "dataset_rows", rows)

    @property
    def macro_f1_baseline(self) -> float:
        return float(np.mean([row.baseline_f1 for row in self.dataset_rows])) if self.dataset_rows else 0.0

    @property
    def baseline(self) -> ConfusionCounts:
        return self.baseline_confusion

    @property
    def macro_f1_combined(self) -> float:
        return float(np.mean([row.combined_f1 for row in self.dataset_rows])) if self.dataset_rows else 0.0

    @property
    def combined(self) -> ConfusionCounts:
        return self.combined_confusion

    @property
    def micro_f1_baseline(self) -> float:
        return self.baseline_confusion.f1

    @property
    def micro_f1_combined(self) -> float:
        return self.combined_confusion.f1

    @property
    def candidate_datasets(self) -> tuple[str, ...]:
        return tuple(sorted({row.dataset for row in self.dataset_rows if row.combined.fp != row.baseline.fp or row.combined.tp != row.baseline.tp}))

    @property
    def new_candidate_count(self) -> int:
        return self.candidate_count

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "exp436.evaluation.v1",
            "state": "evaluated",
            "dataset_count": self.dataset_count,
            "candidate_count": self.candidate_count,
            "added_fp": self.added_fp,
            "label_rows": self.label_rows,
            "prediction_label_access_count": self.prediction_label_access_count,
            "prediction_sha256": self.prediction_sha256,
            "manifest_sha256": self.manifest_sha256,
            "database_sha256": self.database_sha256,
            "label_identity_sha256": self.label_identity_sha256,
            "code_commit": self.code_commit,
            "baseline_confusion": self.baseline_confusion.to_payload(),
            "combined_confusion": self.combined_confusion.to_payload(),
            "micro_f1_baseline": self.micro_f1_baseline,
            "micro_f1_combined": self.micro_f1_combined,
            "micro_f1_delta": self.micro_f1_delta,
            "micro_f1_ci": self.micro_f1_ci.to_payload(),
            "macro_f1_baseline": self.macro_f1_baseline,
            "macro_f1_combined": self.macro_f1_combined,
            "macro_f1_delta": self.macro_f1_delta,
            "macro_f1_ci": self.macro_f1_ci.to_payload(),
            "rescued_datasets": list(self.rescued_datasets),
            "regressed_datasets": list(self.regressed_datasets),
            "strong_baseline_regressions": list(self.strong_baseline_regressions),
            "bootstrap": self.bootstrap.to_payload() if self.bootstrap is not None else None,
        }


def _normalise_comparison(value: object) -> DatasetComparison:
    if isinstance(value, DatasetComparison):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("bootstrap row is invalid")
    dataset = value.get("dataset")
    test_size = value.get("test_size", value.get("n_test"))
    baseline = value.get("baseline", value.get("hard", value.get("baseline_counts")))
    combined = value.get("combined", value.get("hard_plus_exp436", value.get("combined_counts")))
    if baseline is None and all(f"baseline_{name}" in value for name in ("tp", "fp", "fn", "tn")):
        baseline = {name: value[f"baseline_{name}"] for name in ("tp", "fp", "fn", "tn")}
    if combined is None and all(f"combined_{name}" in value for name in ("tp", "fp", "fn", "tn")):
        combined = {name: value[f"combined_{name}"] for name in ("tp", "fp", "fn", "tn")}
    def counts(raw: object, label: str) -> ConfusionCounts:
        if isinstance(raw, ConfusionCounts):
            return raw
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} counts are invalid")
        return ConfusionCounts(
            tp=_require_int(raw.get("tp"), f"{label}.tp"),
            fp=_require_int(raw.get("fp"), f"{label}.fp"),
            fn=_require_int(raw.get("fn"), f"{label}.fn"),
            tn=_require_int(raw.get("tn"), f"{label}.tn"),
        )
    if not isinstance(dataset, str) or type(test_size) is not int:
        raise ValueError("bootstrap row identity is invalid")
    return DatasetComparison(dataset, test_size, counts(baseline, "baseline"), counts(combined, "combined"))


def _aggregate_counts(rows: Sequence[DatasetComparison], *, combined: bool) -> ConfusionCounts:
    selected = [row.combined if combined else row.baseline for row in rows]
    return ConfusionCounts(
        tp=sum(row.tp for row in selected),
        fp=sum(row.fp for row in selected),
        fn=sum(row.fn for row in selected),
        tn=sum(row.tn for row in selected),
    )


def _paired_dataset_bootstrap_for_test(
    rows: Iterable[DatasetComparison | Mapping[str, object]],
    *,
    repetitions: int,
    seed: int,
) -> BootstrapResult:
    """Return a paired dataset-cluster percentile CI for F1 deltas.

    The same sampled dataset indices are used for Hard and Hard+Exp436 in each
    replicate.  Dataset F1 is defined as zero when its denominator is zero.
    """

    values = tuple(sorted((_normalise_comparison(row) for row in rows), key=lambda row: row.dataset))
    if not values:
        raise ValueError("bootstrap requires at least one dataset")
    _require_int(repetitions, "bootstrap repetitions", minimum=1)
    _require_int(seed, "bootstrap seed", minimum=0)
    rng = np.random.default_rng(seed)
    baseline = np.asarray([[row.baseline.tp, row.baseline.fp, row.baseline.fn, row.baseline.tn] for row in values], dtype=np.int64)
    combined = np.asarray([[row.combined.tp, row.combined.fp, row.combined.fn, row.combined.tn] for row in values], dtype=np.int64)
    baseline_f1 = np.asarray([row.baseline_f1 for row in values], dtype=np.float64)
    combined_f1 = np.asarray([row.combined_f1 for row in values], dtype=np.float64)
    micro_deltas = np.empty(repetitions, dtype=np.float64)
    macro_deltas = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(values), size=len(values), endpoint=False)
        base_counts = baseline[sampled].sum(axis=0)
        combo_counts = combined[sampled].sum(axis=0)
        base_denominator = 2 * int(base_counts[0]) + int(base_counts[1]) + int(base_counts[2])
        combo_denominator = 2 * int(combo_counts[0]) + int(combo_counts[1]) + int(combo_counts[2])
        base_f1 = 0.0 if base_denominator == 0 else 2.0 * int(base_counts[0]) / base_denominator
        combo_f1 = 0.0 if combo_denominator == 0 else 2.0 * int(combo_counts[0]) / combo_denominator
        micro_deltas[index] = combo_f1 - base_f1
        macro_deltas[index] = float(np.mean(combined_f1[sampled] - baseline_f1[sampled]))
    micro_low, micro_high = np.quantile(micro_deltas, [0.025, 0.975], method="linear")
    macro_low, macro_high = np.quantile(macro_deltas, [0.025, 0.975], method="linear")
    return BootstrapResult(
        repetitions=repetitions,
        seed=seed,
        micro_f1_delta_ci=ConfidenceInterval(float(micro_low), float(micro_high)),
        macro_f1_delta_ci=ConfidenceInterval(float(macro_low), float(macro_high)),
    )


def _paired_dataset_bootstrap(
    rows: Iterable[DatasetComparison | Mapping[str, object]],
) -> BootstrapResult:
    """Run the immutable production bootstrap without caller-controlled knobs."""

    return _paired_dataset_bootstrap_for_test(
        rows,
        repetitions=FROZEN_CONFIG.bootstrap_repetitions,
        seed=FROZEN_CONFIG.bootstrap_seed,
    )


def decide_exp436_success(
    evaluation: Exp436Evaluation | object | None = None,
    *,
    dataset_count: int | None = None,
    candidate_count: int | None = None,
    rescued_datasets: Iterable[str] | None = None,
    added_fp: int | None = None,
    micro_f1_delta: float | None = None,
    macro_f1_delta: float | None = None,
    micro_f1_ci: ConfidenceInterval | None = None,
    macro_f1_ci: ConfidenceInterval | None = None,
    prediction_label_access_count: int | None = None,
) -> SuccessDecision:
    """Apply every preregistered success gate without tuning from TEST."""

    source = evaluation
    def coerce_ci(value: object, label: str) -> ConfidenceInterval:
        if isinstance(value, ConfidenceInterval):
            return value
        if isinstance(value, Mapping) and "lower" in value and "upper" in value:
            return ConfidenceInterval(value["lower"], value["upper"])  # type: ignore[arg-type]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            return ConfidenceInterval(value[0], value[1])  # type: ignore[arg-type]
        raise ValueError(f"{label} is invalid")
    def get(name: str, supplied: object | None) -> object:
        if supplied is not None:
            return supplied
        if source is None or not hasattr(source, name):
            raise ValueError(f"success decision requires {name}")
        return getattr(source, name)
    dataset_count_value = _require_int(get("dataset_count", dataset_count), "dataset_count")
    candidate_count_value = _require_int(get("candidate_count", candidate_count), "candidate_count")
    rescued = tuple(sorted(set(str(value) for value in (get("rescued_datasets", rescued_datasets) or ()))))
    added_fp_value = _require_int(get("added_fp", added_fp), "added_fp")
    micro_delta_value = _require_float(get("micro_f1_delta", micro_f1_delta), "micro_f1_delta")
    macro_delta_value = _require_float(get("macro_f1_delta", macro_f1_delta), "macro_f1_delta")
    micro_ci_value = get("micro_f1_ci", micro_f1_ci)
    macro_ci_value = get("macro_f1_ci", macro_f1_ci)
    micro_ci_value = coerce_ci(micro_ci_value, "micro_f1_ci")
    macro_ci_value = coerce_ci(macro_ci_value, "macro_f1_ci")
    access_count = _require_int(get("prediction_label_access_count", prediction_label_access_count), "prediction_label_access_count")
    gates: dict[str, bool] = {
        "dataset_population": dataset_count_value == FROZEN_CONFIG.expected_datasets,
        "candidate_budget": candidate_count_value <= FROZEN_CONFIG.global_candidate_budget,
        "rescued_dataset_gate": len(rescued) >= FROZEN_CONFIG.rescued_dataset_minimum,
        "added_false_positive_gate": 0 <= added_fp_value <= FROZEN_CONFIG.added_false_positive_maximum,
        "micro_f1_point_gate": micro_delta_value > 0.0,
        "micro_f1_ci_gate": micro_ci_value.lower > 0.0,
        "macro_f1_point_gate": macro_delta_value > 0.0,
        "macro_f1_ci_gate": macro_ci_value.lower > 0.0,
        "prediction_label_access_gate": access_count == 0,
    }
    reason_names = {
        "dataset_population": "dataset_count_gate",
        "candidate_budget": "candidate_budget_gate",
    }
    reasons = tuple(reason_names.get(key, key) for key, passed in gates.items() if not passed)
    return SuccessDecision(not reasons, reasons, gates)


def _parse_hard_file(path: Path) -> dict[str, tuple[int, ...]]:
    """Compatibility wrapper that parses bytes captured from one descriptor."""

    payload, _ = _read_pinned_bytes(Path(path))
    return _parse_hard_payload(payload)


def _parse_hard_payload(payload: bytes) -> dict[str, tuple[int, ...]]:
    """Parse already-pinned Exp291 rows without reopening their pathname."""

    rows: dict[str, tuple[int, ...]] = {}
    for raw in payload.splitlines():
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Exp291 Hard file is invalid JSON") from error
        if not isinstance(value, Mapping):
            raise ValueError("Exp291 Hard row is invalid")
        dataset = value.get("dataset")
        indices = value.get("automatic_hard_indices", value.get("source_baseline_hard_indices"))
        if not isinstance(dataset, str) or not isinstance(indices, list):
            raise ValueError("Exp291 Hard row is incomplete")
        if value.get("contains_test_labels") is not False:
            raise ValueError("Exp291 Hard artifact is not label-free")
        parsed = tuple(sorted(int(index) for index in indices))
        if any(type(index) is not int or index < 0 for index in parsed) or len(parsed) != len(set(parsed)):
            raise ValueError("Exp291 Hard indices are invalid")
        if dataset in rows:
            raise ValueError("Exp291 Hard dataset is duplicated")
        rows[dataset] = parsed
    if not rows:
        raise ValueError("Exp291 Hard file is empty")
    return dict(sorted(rows.items()))


def _parse_exp404_hard_file(path: Path) -> dict[str, tuple[int, ...]]:
    """Compatibility wrapper that parses bytes captured from one descriptor."""

    payload, _ = _read_pinned_bytes(Path(path))
    return _parse_exp404_hard_payload(payload)


def _parse_exp404_hard_payload(payload: bytes) -> dict[str, tuple[int, ...]]:
    """Parse already-pinned Exp404 candidate-index rows without TEST labels."""

    rows: dict[str, list[int]] = {}
    coordinates: set[tuple[str, int]] = set()
    for raw in payload.splitlines():
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Exp404 Hard prediction is invalid JSON") from error
        if not isinstance(value, Mapping):
            raise ValueError("Exp404 Hard prediction row is invalid")
        dataset = value.get("dataset")
        candidate_index = value.get("candidate_index")
        if not isinstance(dataset, str) or not dataset.strip() or type(candidate_index) is not int or candidate_index < 0:
            raise ValueError("Exp404 Hard prediction row is incomplete")
        if value.get("contains_test_labels") is not False or value.get("contains_evaluation_metrics") is not False:
            raise ValueError("Exp404 Hard prediction is not label-free")
        coordinate = (dataset, candidate_index)
        if coordinate in coordinates:
            raise ValueError("Exp404 Hard prediction contains duplicate coordinates")
        coordinates.add(coordinate)
        rows.setdefault(dataset, []).append(candidate_index)
    if len(coordinates) != _EXP404_EXPECTED_COORDINATES or len(rows) != _EXP404_EXPECTED_DATASETS:
        raise ValueError("Exp404 Hard prediction population is invalid")
    if any(not values for values in rows.values()):
        raise ValueError("Exp404 Hard prediction contains an empty candidate dataset")
    return {dataset: tuple(sorted(values)) for dataset, values in sorted(rows.items())}


def _load_frozen_hard_coordinates(
    *,
    config: Exp436FrozenConfig = FROZEN_CONFIG,
) -> dict[str, tuple[int, ...]]:
    """Verify Exp404 and Exp291 identities and return their exact shared map."""

    config.validate()
    exp404_payload, exp404_sha = _read_pinned_bytes(EXP404_HARD_PATH)
    if exp404_sha != config.hard_prediction_sha256:
        raise ValueError("Exp404 Hard prediction SHA256 does not match frozen input")
    exp291_payload, exp291_sha = _read_pinned_bytes(EXP291_HARD_PATH)
    if exp291_sha != config.exp291_prediction_sha256:
        raise ValueError("Exp291 Hard prediction SHA256 does not match frozen input")
    exp404 = _parse_exp404_hard_payload(exp404_payload)
    exp291 = _parse_hard_payload(exp291_payload)
    if len(exp291) != config.expected_datasets:
        raise ValueError("Exp291 Hard population is invalid")
    if set(exp404) - set(exp291):
        raise ValueError("Exp404 Hard dataset population is outside Exp291")
    if any(tuple(exp404.get(dataset, ())) != tuple(exp291[dataset]) for dataset in exp291):
        raise ValueError("Exp404 Hard coordinates do not equal Exp291 automatic Hard")
    if sum(bool(indices) for indices in exp291.values()) != _EXP404_EXPECTED_DATASETS:
        raise ValueError("Exp291 Hard nonempty population is invalid")
    if sum(len(indices) for indices in exp291.values()) != _EXP404_EXPECTED_COORDINATES:
        raise ValueError("Exp291 Hard coordinate population is invalid")
    return dict(sorted(exp291.items()))


def _read_publication_json(publication: FrozenPublication, name: str) -> dict[str, object]:
    path = publication.publication / name
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"publication {name} is invalid JSON") from error
    if not isinstance(value, Mapping) or raw != _canonical_json(value):
        raise ValueError(f"publication {name} is not canonical JSON")
    return dict(value)


def _validate_verified_publication(
    publication: FrozenPublication,
    store: object,
    *,
    expected_total: int,
    config: Exp436FrozenConfig,
) -> tuple[dict[str, object], tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Validate all replayed publication identities before labels are opened."""

    manifest = dict(publication.manifest)
    status = _read_publication_json(publication, STATUS_FILE)
    artifact_map = _read_publication_json(publication, PUBLICATION_HASH_FILE)
    if status.get("state") != "complete_review_only":
        raise ValueError("publication status is not complete_review_only")
    if status.get("review_only") is not True or status.get("operational_alert_eligible") is not False:
        raise ValueError("publication review-only governance is invalid")
    if status.get("contains_test_labels") is not False or status.get("test_labels_opened") is not False or type(status.get("label_access_count")) is not int or status.get("label_access_count") != 0:
        raise ValueError("publication label governance is invalid")
    if status.get("prediction_sha256") != manifest.get("prediction_sha256"):
        raise ValueError("publication status prediction SHA256 does not match manifest")
    if manifest.get("review_only") is not True or manifest.get("operational_alert_eligible") is not False:
        raise ValueError("publication manifest governance is invalid")
    if manifest.get("contains_test_labels") is not False or manifest.get("test_labels_opened") is not False or type(manifest.get("label_access_count")) is not int or manifest.get("label_access_count") != 0:
        raise ValueError("publication manifest label governance is invalid")
    if manifest.get("completed_datasets") != expected_total or manifest.get("failed_datasets") != 0:
        raise ValueError("publication population is invalid")
    if not isinstance(manifest.get("prediction_sha256"), str) or manifest["prediction_sha256"] != _digest((publication.publication / PREDICTION_FILE).read_bytes()):
        raise ValueError("prediction SHA256 does not match manifest")
    config_hash = _digest(canonical_json_bytes(asdict(config)))
    if manifest.get("config_sha256") != config_hash:
        raise ValueError("prediction config SHA256 does not match frozen config")
    frozen_population = expected_total == config.expected_datasets
    if frozen_population:
        if manifest.get("database_sha256") != config.database_sha256:
            raise ValueError("prediction input database SHA256 does not match frozen input")
        expected_inputs = {
            "database": config.database_sha256,
            "exp291_prediction_sha256": config.exp291_prediction_sha256,
            "exp291_manifest_sha256": config.exp291_manifest_sha256,
            "exp292_prediction_sha256": config.exp292_prediction_sha256,
            "exp292_manifest_sha256": config.exp292_manifest_sha256,
        }
        input_hashes = manifest.get("input_sha256")
        if not isinstance(input_hashes, Mapping) or any(input_hashes.get(key) != value for key, value in expected_inputs.items()):
            raise ValueError("prediction input SHA256 does not match frozen lanes")
        lane_hashes = manifest.get("lane_artifact_sha256")
        if not isinstance(lane_hashes, Mapping) or any(lane_hashes.get(key) != value for key, value in expected_inputs.items() if key != "database"):
            raise ValueError("prediction lane SHA256 does not match frozen lanes")
    else:
        _require_digest(manifest.get("database_sha256"), "prediction database SHA256")
        input_hashes = manifest.get("input_sha256")
        if not isinstance(input_hashes, Mapping):
            raise ValueError("prediction input SHA256 map is invalid")
        for value in input_hashes.values():
            _require_digest(value, "prediction input SHA256")
    reviewed_commit = manifest.get("reviewed_commit")
    if not isinstance(reviewed_commit, str) or _HEX40.fullmatch(reviewed_commit) is None:
        raise ValueError("prediction reviewed commit is invalid")
    if manifest.get("reviewed_commit_sha256") != _digest(reviewed_commit.encode("utf-8")):
        raise ValueError("prediction reviewed commit SHA256 does not match")
    candidate_count = manifest.get("candidate_count")
    if type(candidate_count) is not int or candidate_count < 0:
        raise ValueError("prediction candidate count is invalid")
    if candidate_count > config.global_candidate_budget:
        raise ValueError("prediction candidate budget exceeded")
    summary_rows: list[dict[str, object]] = []
    for raw in (publication.publication / SUMMARY_FILE).read_bytes().splitlines():
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("publication summary is invalid JSON") from error
        if not isinstance(value, Mapping):
            raise ValueError("publication summary row is invalid")
        summary_rows.append(dict(value))
    if len(summary_rows) != expected_total:
        raise ValueError("prediction summary population is invalid")
    datasets = tuple(str(row.get("dataset")) for row in summary_rows)
    if tuple(sorted(datasets)) != datasets or len(set(datasets)) != len(datasets) or any(not name or name == "None" for name in datasets):
        raise ValueError("prediction summary dataset identity is invalid")
    prediction_rows: list[dict[str, object]] = []
    for raw in (publication.publication / PREDICTION_FILE).read_bytes().splitlines():
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("prediction artifact is invalid JSON") from error
        if not isinstance(value, Mapping):
            raise ValueError("prediction artifact row is invalid")
        dataset = value.get("dataset")
        index = value.get("instance_index")
        if not isinstance(dataset, str) or type(index) is not int or index < 0:
            raise ValueError("prediction coordinate is invalid")
        prediction_rows.append(dict(value))
    if len(prediction_rows) != candidate_count:
        raise ValueError("prediction candidate count does not match manifest")
    coordinates = [(row["dataset"], row["instance_index"]) for row in prediction_rows]
    if len(coordinates) != len(set(coordinates)) or coordinates != sorted(coordinates):
        raise ValueError("prediction coordinates are not sorted and unique")
    if any(dataset not in set(datasets) for dataset, _ in coordinates):
        raise ValueError("prediction coordinate is outside population")
    if artifact_map.get("self_excluded") != PUBLICATION_HASH_FILE or artifact_map.get("self_hash_excluded") is not True:
        raise ValueError("publication artifact SHA256 self-exclusion is invalid")
    hashes = artifact_map.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("publication artifact SHA256 map is invalid")
    for name, value in hashes.items():
        if name == PUBLICATION_HASH_FILE or name not in {item.name for item in publication.publication.iterdir()} or value != _digest((publication.publication / name).read_bytes()):
            raise ValueError("publication artifact SHA256 map does not match")
    return manifest, tuple(summary_rows), tuple(prediction_rows)


def _label_payload(value: object, *, dataset: str) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """Normalise a test-label loader result while preserving instance identity."""

    indices: object = None
    labels: object = value
    supplied_digest: object = None
    if isinstance(value, Mapping):
        labels = value.get("labels", value.get("values"))
        indices = value.get("instance_indices", value.get("indices"))
        supplied_digest = value.get("labels_sha256", value.get("identity_sha256"))
    elif isinstance(value, tuple) and len(value) == 2:
        indices, labels = value
    array = np.asarray(labels)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"TEST labels are invalid: dataset={dataset}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"TEST labels are not integer binary values: dataset={dataset}")
    labels_tuple = tuple(int(item) for item in array.tolist())
    if any(item not in (0, 1) for item in labels_tuple):
        raise ValueError(f"TEST labels are not binary: dataset={dataset}")
    if indices is None:
        index_tuple = tuple(range(len(labels_tuple)))
    else:
        try:
            index_tuple = tuple(int(item) for item in indices)
        except (TypeError, ValueError) as error:
            raise ValueError(f"TEST instance identity is invalid: dataset={dataset}") from error
        if (
            len(index_tuple) != len(labels_tuple)
            or any(type(index) is not int or index < 0 for index in index_tuple)
            or index_tuple != tuple(sorted(index_tuple))
            or len(index_tuple) != len(set(index_tuple))
        ):
            raise ValueError(f"TEST instance identity is not canonical: dataset={dataset}")
    identity = {
        "dataset": dataset,
        "instance_indices": list(index_tuple),
        "labels": list(labels_tuple),
    }
    digest = _digest(canonical_json_bytes(identity))
    if supplied_digest is not None and supplied_digest != digest:
        raise ValueError(f"TEST label SHA256 does not match identity: dataset={dataset}")
    return index_tuple, labels_tuple, digest


def _load_test_labels_from_connection(connection: object, dataset: str) -> Mapping[str, object]:
    rows = connection.execute(
        """
        SELECT i.instance_index, i.label
        FROM instances AS i
        JOIN datasets AS d ON d.id = i.dataset_id
        WHERE d.name = ? AND i.split = 'TEST'
        ORDER BY i.instance_index
        """,
        (dataset,),
    ).fetchall()
    if not rows:
        raise ValueError(f"TEST labels are absent: dataset={dataset}")
    labels: list[int] = []
    for row in rows:
        raw_label = row[1]
        if type(raw_label) is int:
            label = raw_label
        elif isinstance(raw_label, str) and raw_label in {"0", "1"}:
            label = int(raw_label)
        else:
            raise ValueError(f"TEST label schema is invalid: dataset={dataset}")
        labels.append(label)
    return {"instance_indices": [int(row[0]) for row in rows], "labels": labels}


def _load_test_labels(
    dataset: str,
    *,
    database_path: Path = DATABASE_PATH,
    expected_sha256: str | None = None,
) -> Mapping[str, object]:
    """Load one dataset through Exp417's descriptor-bound verified session."""

    expected = _sha256_file(database_path) if expected_sha256 is None else _require_digest(
        expected_sha256, "TEST label database SHA256"
    )
    with verified_read_only_connection(Path(database_path), expected) as connection:
        return _load_test_labels_from_connection(connection, dataset)


def _load_labels(
    loader: Callable[[str], object],
    datasets: Sequence[str],
    summary_by_dataset: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[int, int]], dict[str, str]]:
    labels_by_dataset: dict[str, dict[int, int]] = {}
    identities: dict[str, str] = {}
    for dataset in datasets:
        try:
            raw = loader(dataset)
        except TypeError as error:
            raise ValueError(f"TEST label loader must accept dataset: {dataset}") from error
        indices, labels, digest = _label_payload(raw, dataset=dataset)
        expected_size = summary_by_dataset[dataset].get("test_size")
        if type(expected_size) is not int or expected_size != len(labels):
            raise ValueError(f"TEST label size does not match prediction: dataset={dataset}")
        labels_by_dataset[dataset] = dict(zip(indices, labels, strict=True))
        identities[dataset] = digest
    return labels_by_dataset, identities


def _counts(selected: Iterable[int], labels: Mapping[int, int]) -> ConfusionCounts:
    selected_set = set(int(index) for index in selected)
    identities = set(labels)
    if any(type(index) is not int or index < 0 or index not in identities for index in selected_set):
        raise ValueError("prediction coordinate is outside TEST identity")
    if any(type(index) is not int or index < 0 or label not in (0, 1) for index, label in labels.items()):
        raise ValueError("TEST label identity map is invalid")
    truth = {index for index, label in labels.items() if label == 1}
    tp = len(selected_set & truth)
    fp = len(selected_set - truth)
    fn = len(truth - selected_set)
    tn = len(labels) - tp - fp - fn
    if tn < 0:
        raise ValueError("confusion counts are invalid")
    return ConfusionCounts(tp, fp, fn, tn)


def _resolve_store(publication: FrozenPublication | Path | str, checkpoint: object | None) -> tuple[Path, object]:
    if checkpoint is None:
        if isinstance(publication, FrozenPublication):
            checkpoint = publication.publication.parent.parent / "checkpoint_antigravity"
        else:
            root = Path(publication)
            output = root.parent if root.name == PUBLICATION_DIRECTORY else root
            checkpoint = output.parent / "checkpoint_antigravity"
    store: object
    if isinstance(checkpoint, (Path, str)):
        store = Exp436CheckpointStore.open_read_only(Path(checkpoint))
    else:
        store = checkpoint
    if isinstance(publication, FrozenPublication):
        output = publication.publication.parent
    else:
        root = Path(publication)
        output = root.parent if root.name == PUBLICATION_DIRECTORY else root
    return output, store


def _label_identity_hash(identities: Mapping[str, str]) -> str:
    return _digest(canonical_json_bytes(dict(sorted(identities.items()))))


def _snapshot_frozen_tree(root: Path) -> dict[str, tuple[str, bytes | str | None]]:
    """Capture every relative path, entry type, and byte payload without following links."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("frozen output root is absent or invalid")
    snapshot: dict[str, tuple[str, bytes | str | None]] = {".": ("directory", None)}

    def visit(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as error:
            raise ValueError("frozen output tree cannot be read") from error
        for entry in entries:
            relative = entry.relative_to(root).as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError("frozen output tree cannot be inspected") from error
            if stat.S_ISLNK(metadata.st_mode):
                snapshot[relative] = ("symlink", os.readlink(entry))
            elif stat.S_ISDIR(metadata.st_mode):
                snapshot[relative] = ("directory", None)
                visit(entry)
            elif stat.S_ISREG(metadata.st_mode):
                try:
                    payload = entry.read_bytes()
                except OSError as error:
                    raise ValueError("frozen output tree cannot be read") from error
                snapshot[relative] = ("file", payload)
            else:
                snapshot[relative] = ("other", None)

    visit(root)
    return snapshot


def _verify_frozen_source(
    output: Path,
    store: object,
    *,
    expected_total: int,
    snapshot: Mapping[str, tuple[str, bytes | str | None]],
) -> None:
    """Replay publication and compare the complete source tree before publish."""

    try:
        _inspect_publication(
            Path(output),
            store,
            _expectations=_PublicationExpectations(total=expected_total),
        )
    except ValueError as error:
        raise ValueError(f"frozen output source replay failed: {error}") from error
    current = _snapshot_frozen_tree(Path(output))
    if dict(current) != dict(snapshot):
        raise ValueError("frozen output source tree changed after label access")


def _validate_evaluation_root(output: Path, store: object, evaluation_root: Path) -> Path:
    """Require a new sibling destination outside the frozen source/checkpoint."""

    output = Path(output)
    destination = Path(evaluation_root)
    if destination.exists() or destination.is_symlink():
        raise ValueError("evaluation root already exists")
    checkpoint = Path(getattr(store, "root", output.parent / "checkpoint_antigravity"))
    publication = output / PUBLICATION_DIRECTORY
    if destination.parent != output.parent:
        raise ValueError("evaluation root must be an absent sibling of output")
    if destination == output or output in destination.parents:
        raise ValueError("evaluation root cannot be equal to or nested under output")
    if destination == publication or publication in destination.parents:
        raise ValueError("evaluation root cannot be equal to or nested under publication")
    if destination == checkpoint or checkpoint in destination.parents:
        raise ValueError("evaluation root cannot be equal to or nested under checkpoint")
    return destination


def _extract_candidate_indices(prediction_rows: Sequence[Mapping[str, object]]) -> dict[str, tuple[int, ...]]:
    output: dict[str, list[int]] = {}
    for row in prediction_rows:
        dataset = str(row["dataset"])
        output.setdefault(dataset, []).append(int(row["instance_index"]))
    return {dataset: tuple(values) for dataset, values in sorted(output.items())}


def _normalise_hard_coordinates(value: object) -> Mapping[str, Iterable[int]]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: dict[str, tuple[int, ...]] = {}
        for row in value:
            if not isinstance(row, Mapping) or not isinstance(row.get("dataset"), str):
                raise ValueError("Exp291 Hard coordinates are invalid")
            indices = row.get("hard_indices", row.get("automatic_hard_indices"))
            if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
                raise ValueError("Exp291 Hard coordinates are invalid")
            result[str(row["dataset"])] = tuple(int(index) for index in indices)
        return result
    raise ValueError("Exp291 Hard coordinates are invalid")


def _evaluate_frozen_prediction_for_test(
    publication: FrozenPublication | Path | str,
    hard_coordinates: Mapping[str, Iterable[int]] | None = None,
    label_loader: Callable[[str], object] | None = None,
    *,
    checkpoint: object | None = None,
    completed_store: object | None = None,
    evaluation_root: Path | None = None,
    output_root: Path | None = None,
    expected_datasets: int | Sequence[str] | None = None,
    database_path: Path = DATABASE_PATH,
    config: Exp436FrozenConfig = FROZEN_CONFIG,
    before_publish: Callable[[], None] | None = None,
) -> Exp436Evaluation:
    """Replay and evaluate a frozen publication after the hash gate.

    ``completed_store`` is an alias for ``checkpoint`` for callers that make
    the read-only boundary explicit.  ``expected_datasets`` may be an integer
    or exact sorted names for deterministic unit fixtures; production uses the
    frozen 1,117 value.
    """

    config.validate()
    if before_publish is not None and not callable(before_publish):
        raise ValueError("before_publish callback is invalid")
    if checkpoint is not None and completed_store is not None:
        raise ValueError("checkpoint and completed_store are mutually exclusive")
    store_argument = completed_store if completed_store is not None else checkpoint
    output, store = _resolve_store(publication, store_argument)
    if evaluation_root is None and output_root is not None:
        evaluation_root = Path(output_root)
    if evaluation_root is not None:
        evaluation_root = _validate_evaluation_root(output, store, Path(evaluation_root))
    if expected_datasets is None:
        expected_total = config.expected_datasets
        expected_names: tuple[str, ...] | None = None
    elif type(expected_datasets) is int:
        expected_total = _require_int(expected_datasets, "expected_datasets", minimum=1)
        expected_names = None
    else:
        expected_names = tuple(str(value) for value in expected_datasets)
        if expected_names != tuple(sorted(expected_names)) or len(set(expected_names)) != len(expected_names) or not expected_names:
            raise ValueError("expected dataset population is invalid")
        expected_total = len(expected_names)
    # This is intentionally the first operation that can inspect publication
    # bytes.  No TEST-label loader is reachable before this replay completes.
    try:
        verified = _inspect_publication(
            output,
            store,
            _expectations=_PublicationExpectations(total=expected_total),
        )
    except ValueError as error:
        # Keep the first post-freeze failure stable and make the hash-gate
        # boundary obvious to callers.  Most importantly, the label loader is
        # still unreachable on this path.
        if "prediction" in str(error).lower():
            raise ValueError(f"prediction SHA256 verification failed: {error}") from error
        raise
    manifest, summary_rows, prediction_rows = _validate_verified_publication(
        verified,
        store,
        expected_total=expected_total,
        config=config,
    )
    datasets = tuple(str(row["dataset"]) for row in summary_rows)
    if expected_names is not None and datasets != expected_names:
        raise ValueError("prediction population does not match expected names")
    summary_by_dataset = {str(row["dataset"]): row for row in summary_rows}
    source_snapshot = _snapshot_frozen_tree(output)
    if hard_coordinates is None:
        if expected_total != config.expected_datasets:
            raise ValueError("fixture evaluation must provide Hard coordinates")
        hard_coordinates = _load_frozen_hard_coordinates(config=config)
    hard_source = _normalise_hard_coordinates(hard_coordinates)
    hard = {str(dataset): tuple(sorted(set(int(index) for index in indices))) for dataset, indices in hard_source.items()}
    if tuple(sorted(hard)) != datasets:
        raise ValueError("Exp291 Hard population does not match prediction")
    if any(any(index < 0 for index in indices) for indices in hard.values()):
        raise ValueError("Exp291 Hard coordinates are invalid")
    candidate = _extract_candidate_indices(prediction_rows)
    if any(dataset not in summary_by_dataset for dataset in candidate):
        raise ValueError("prediction candidate population is invalid")
    if any(set(hard[dataset]) & set(candidate.get(dataset, ())) for dataset in datasets):
        raise ValueError("Exp436 candidate overlaps frozen Hard")
    # Exp404/Exp291 pinning and equality checks occur after the complete source
    # snapshot.  Replay once more before the single label-bearing connection.
    _verify_frozen_source(output, store, expected_total=expected_total, snapshot=source_snapshot)
    loader = label_loader
    private_loader = loader is None
    if loader is None:
        # Exp417 verifies the frozen digest from the same descriptor used for
        # the read-only SQLite session.  All artifact gates above have already
        # completed, so TEST labels remain unreachable on any earlier failure.
        database_sha = config.database_sha256
    else:
        database_sha = str(manifest.get("database_sha256", ""))
        if database_sha and _HEX64.fullmatch(database_sha) is None:
            raise ValueError("publication database SHA256 is invalid")
    if private_loader:
        with verified_read_only_connection(Path(database_path), database_sha) as connection:
            labels_by_dataset, identities = _load_labels(
                lambda dataset: _load_test_labels_from_connection(connection, dataset),
                datasets,
                summary_by_dataset,
            )
    else:
        labels_by_dataset, identities = _load_labels(loader, datasets, summary_by_dataset)  # type: ignore[arg-type]
    if not private_loader and database_sha and database_sha != config.database_sha256 and expected_total == config.expected_datasets:
        raise ValueError("TEST label database SHA256 does not match frozen input")
    rows: list[DatasetComparison] = []
    for dataset in datasets:
        labels = labels_by_dataset[dataset]
        baseline = _counts(hard[dataset], labels)
        combined = _counts(set(hard[dataset]) | set(candidate.get(dataset, ())), labels)
        rows.append(DatasetComparison(dataset, len(labels), baseline, combined))
    comparisons = tuple(rows)
    baseline_total = _aggregate_counts(comparisons, combined=False)
    combined_total = _aggregate_counts(comparisons, combined=True)
    bootstrap = _paired_dataset_bootstrap(comparisons)
    micro_delta = combined_total.f1 - baseline_total.f1
    macro_delta = float(np.mean([row.combined_f1 - row.baseline_f1 for row in comparisons])) if comparisons else 0.0
    rescued = tuple(row.dataset for row in comparisons if row.rescued)
    regressed = tuple(row.dataset for row in comparisons if row.regressed)
    strong_regressed = tuple(row.dataset for row in comparisons if row.regressed and row.strong_baseline)
    evaluation = Exp436Evaluation(
        dataset_count=len(comparisons),
        candidate_count=len(prediction_rows),
        added_fp=combined_total.fp - baseline_total.fp,
        dataset_rows=comparisons,
        baseline_confusion=baseline_total,
        combined_confusion=combined_total,
        micro_f1_delta=micro_delta,
        macro_f1_delta=macro_delta,
        micro_f1_ci=bootstrap.micro_f1_ci,
        macro_f1_ci=bootstrap.macro_f1_ci,
        rescued_datasets=rescued,
        regressed_datasets=regressed,
        strong_baseline_regressions=strong_regressed,
        prediction_sha256=str(manifest["prediction_sha256"]),
        manifest_sha256=verified.manifest_sha256,
        prediction_label_access_count=int(manifest["label_access_count"]),
        label_rows=sum(len(values) for values in labels_by_dataset.values()),
        label_identity_sha256=_label_identity_hash(identities),
        database_sha256=database_sha,
        code_commit=str(manifest.get("reviewed_commit")),
        bootstrap=bootstrap,
    )
    if evaluation_root is not None:
        def verify_before_publish() -> None:
            if before_publish is not None:
                before_publish()
            _verify_frozen_source(
                output,
                store,
                expected_total=expected_total,
                snapshot=source_snapshot,
            )

        # This gate must complete before the writer allocates its staging
        # directory.  The same gate is retained immediately before rename to
        # close the later staging-to-publication interval.
        verify_before_publish()
        _write_evaluation_artifacts(
            Path(evaluation_root),
            evaluation,
            before_publish=verify_before_publish,
        )
    else:
        _verify_frozen_source(
            output,
            store,
            expected_total=expected_total,
            snapshot=source_snapshot,
        )
    return evaluation


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        os.fsync(directory)
        os.close(directory)
    except OSError:
        pass


def _evaluation_payloads(evaluation: Exp436Evaluation) -> dict[str, bytes]:
    decision = decide_exp436_success(evaluation)
    dataset_rows = [row.to_payload() for row in evaluation.dataset_rows]
    rescued_rows = [row.to_payload() for row in evaluation.dataset_rows if row.rescued]
    regressed_rows = [row.to_payload() for row in evaluation.dataset_rows if row.regressed]
    confusion = {
        "baseline": evaluation.baseline_confusion.to_payload(),
        "combined": evaluation.combined_confusion.to_payload(),
        "baseline_f1": evaluation.micro_f1_baseline,
        "combined_f1": evaluation.micro_f1_combined,
    }
    payloads: dict[str, bytes] = {
        EVALUATION_FILE: _canonical_json(evaluation.to_payload()),
        DATASET_FILE: _jsonl(dataset_rows),
        RESCUED_FILE: _jsonl(rescued_rows),
        REGRESSED_FILE: _jsonl(regressed_rows),
        CONFUSION_FILE: _canonical_json(confusion),
        BOOTSTRAP_FILE: _canonical_json(evaluation.bootstrap.to_payload() if evaluation.bootstrap else {}),
        DECISION_FILE: _canonical_json(decision.to_payload()),
    }
    hashes = {name: _digest(raw) for name, raw in sorted(payloads.items())}
    payloads[EVALUATION_HASH_FILE] = _canonical_json({
        "artifact_sha256": hashes,
        "self_excluded": EVALUATION_HASH_FILE,
        "self_hash_excluded": True,
    })
    return payloads


def _verify_evaluation_artifacts(root: Path, expected: Mapping[str, bytes]) -> None:
    """Reopen the finished evaluation tree and verify its exact inventory."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir() or {item.name for item in root.iterdir()} != _EVALUATION_FILES:
        raise ValueError("evaluation artifact inventory is invalid")
    actual = {name: _read_pinned_bytes(root / name)[0] for name in _EVALUATION_FILES}
    if dict(actual) != dict(expected):
        raise ValueError("evaluation artifact replay does not match frozen evaluation")
    try:
        hash_map = json.loads(actual[EVALUATION_HASH_FILE])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation artifact hash map is invalid") from error
    if (
        not isinstance(hash_map, Mapping)
        or hash_map.get("self_excluded") != EVALUATION_HASH_FILE
        or hash_map.get("self_hash_excluded") is not True
        or hash_map.get("artifact_sha256") != {name: _digest(raw) for name, raw in sorted(expected.items()) if name != EVALUATION_HASH_FILE}
    ):
        raise ValueError("evaluation artifact hash map is invalid")


def _write_evaluation_artifacts(
    root: Path,
    evaluation: Exp436Evaluation,
    *,
    before_publish: Callable[[], None] | None = None,
) -> Exp436Evaluation:
    root = Path(root)
    if root.exists() or root.is_symlink():
        payloads = _evaluation_payloads(evaluation)
        _verify_evaluation_artifacts(root, payloads)
        return evaluation
    parent = root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("evaluation parent is absent or invalid")
    payloads = _evaluation_payloads(evaluation)
    staging = Path(mkdtemp(prefix=f".{root.name}.", dir=str(parent)))
    published = False
    try:
        for name in sorted(payloads):
            _write_new(staging / name, payloads[name])
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
        os.close(descriptor)
        if before_publish is not None:
            before_publish()
        if root.exists() or root.is_symlink():
            raise ValueError("evaluation root appeared during publish")
        # Darwin's RENAME_EXCL is the same no-clobber primitive used for the
        # prediction publication.  A concurrent destination is never replaced.
        _exclusive_rename(staging, root)
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
        os.close(descriptor)
        _verify_evaluation_artifacts(root, payloads)
        published = True
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            for child in tuple(staging.iterdir()):
                if child.is_file() and not child.is_symlink():
                    child.unlink()
            staging.rmdir()
    return evaluation


def _read_existing_evaluation(root: Path) -> Exp436Evaluation:
    """Reconstruct and byte-verify a prior immutable evaluation without labels."""
    raw, _ = _read_pinned_bytes(root / EVALUATION_FILE)
    dataset_raw, _ = _read_pinned_bytes(root / DATASET_FILE)
    bootstrap_raw, _ = _read_pinned_bytes(root / BOOTSTRAP_FILE)
    try:
        payload = json.loads(raw)
        rows = tuple(_normalise_comparison(json.loads(line)) for line in dataset_raw.decode("utf-8").splitlines() if line)
        bootstrap_payload = json.loads(bootstrap_raw)
        bootstrap = BootstrapResult(
            int(bootstrap_payload["repetitions"]), int(bootstrap_payload["seed"]),
            ConfidenceInterval(**bootstrap_payload["micro_f1_delta_ci"]), ConfidenceInterval(**bootstrap_payload["macro_f1_delta_ci"]),
        )
        result = Exp436Evaluation(
            dataset_count=int(payload["dataset_count"]), candidate_count=int(payload["candidate_count"]), added_fp=int(payload["added_fp"]),
            dataset_rows=rows, baseline_confusion=ConfusionCounts(**payload["baseline_confusion"]), combined_confusion=ConfusionCounts(**payload["combined_confusion"]),
            micro_f1_delta=float(payload["micro_f1_delta"]), macro_f1_delta=float(payload["macro_f1_delta"]),
            micro_f1_ci=ConfidenceInterval(**payload["micro_f1_ci"]), macro_f1_ci=ConfidenceInterval(**payload["macro_f1_ci"]),
            rescued_datasets=tuple(payload["rescued_datasets"]), regressed_datasets=tuple(payload["regressed_datasets"]), strong_baseline_regressions=tuple(payload["strong_baseline_regressions"]),
            prediction_sha256=str(payload["prediction_sha256"]), manifest_sha256=str(payload["manifest_sha256"]), prediction_label_access_count=int(payload["prediction_label_access_count"]),
            label_rows=int(payload["label_rows"]), label_identity_sha256=str(payload["label_identity_sha256"]), database_sha256=str(payload["database_sha256"]), code_commit=str(payload["code_commit"]), bootstrap=bootstrap,
        )
    except (KeyError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("existing evaluation is invalid") from error
    _verify_evaluation_artifacts(root, _evaluation_payloads(result))
    return result


def _validate_existing_evaluation(
    result: Exp436Evaluation,
    publication: FrozenPublication,
    summary_rows: Sequence[Mapping[str, object]],
) -> None:
    """Require a stored evaluation to be the exact frozen full-population result."""
    if result.dataset_count != FROZEN_CONFIG.expected_datasets or len(result.dataset_rows) != FROZEN_CONFIG.expected_datasets:
        raise ValueError("existing evaluation population is not frozen")
    if result.candidate_count != publication.manifest.get("candidate_count"):
        raise ValueError("existing evaluation candidate count does not bind publication")
    expected_identity = tuple((str(row.get("dataset")), row.get("test_size")) for row in summary_rows)
    actual_identity = tuple((row.dataset, row.test_size) for row in result.dataset_rows)
    if (len(expected_identity) != FROZEN_CONFIG.expected_datasets
            or any(type(test_size) is not int for _dataset, test_size in expected_identity)
            or actual_identity != expected_identity):
        raise ValueError("existing evaluation rows do not bind publication summaries")
    if result.label_rows != sum(row.test_size for row in result.dataset_rows) or result.prediction_label_access_count != 0:
        raise ValueError("existing evaluation label accounting is invalid")
    if _HEX64.fullmatch(result.label_identity_sha256) is None:
        raise ValueError("existing evaluation label identity is invalid")
    baseline = _aggregate_counts(result.dataset_rows, combined=False)
    combined = _aggregate_counts(result.dataset_rows, combined=True)
    if baseline != result.baseline_confusion or combined != result.combined_confusion or result.added_fp != combined.fp - baseline.fp:
        raise ValueError("existing evaluation confusion accounting is invalid")
    positives = 0
    for row in result.dataset_rows:
        tp_delta, fp_delta = row.combined.tp - row.baseline.tp, row.combined.fp - row.baseline.fp
        if tp_delta < 0 or fp_delta < 0:
            raise ValueError("existing evaluation candidate accounting is invalid")
        positives += tp_delta + fp_delta
    if positives != result.candidate_count:
        raise ValueError("existing evaluation candidate accounting does not bind publication")
    expected_micro = combined.f1 - baseline.f1
    expected_macro = float(np.mean([row.combined_f1 - row.baseline_f1 for row in result.dataset_rows]))
    if not math.isclose(result.micro_f1_delta, expected_micro, rel_tol=0.0, abs_tol=1e-15) or not math.isclose(result.macro_f1_delta, expected_macro, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("existing evaluation metrics are invalid")
    rescued = tuple(row.dataset for row in result.dataset_rows if row.rescued)
    regressed = tuple(row.dataset for row in result.dataset_rows if row.regressed)
    strong = tuple(row.dataset for row in result.dataset_rows if row.regressed and row.strong_baseline)
    if (result.rescued_datasets, result.regressed_datasets, result.strong_baseline_regressions) != (rescued, regressed, strong):
        raise ValueError("existing evaluation dataset reports are invalid")
    bootstrap = result.bootstrap
    if bootstrap is None or bootstrap.repetitions != FROZEN_CONFIG.bootstrap_repetitions or bootstrap.seed != FROZEN_CONFIG.bootstrap_seed:
        raise ValueError("existing evaluation bootstrap is not frozen")
    if result.micro_f1_ci != bootstrap.micro_f1_ci or result.macro_f1_ci != bootstrap.macro_f1_ci:
        raise ValueError("existing evaluation confidence intervals do not bind bootstrap")
    if _paired_dataset_bootstrap(result.dataset_rows) != bootstrap:
        raise ValueError("existing evaluation bootstrap intervals are not reproducible")


def evaluate_exp436(
    output_root: Path,
    evaluation_root: Path,
) -> Exp436Evaluation:
    """Fixed CLI entrypoint for the post-freeze ``evaluate`` mode."""

    # Import the CLI constants lazily so prediction modes never import this
    # evaluator, while this production entrypoint cannot be redirected to a
    # fixture or an alternate output directory.
    from . import cli_antigravity

    output_root = Path(output_root)
    evaluation_root = Path(evaluation_root)
    fixed_output = Path(cli_antigravity.FROZEN_OUTPUT)
    fixed_checkpoint = Path(cli_antigravity.FROZEN_CHECKPOINT)
    fixed_evaluation = Path(cli_antigravity.FROZEN_EVALUATION)
    run_parent = Path(cli_antigravity.RUN_PARENT)
    if any(path.is_symlink() for path in (run_parent, fixed_output, fixed_checkpoint, fixed_evaluation)):
        raise ValueError("production evaluation paths cannot be symlinks")
    if output_root != fixed_output or evaluation_root != fixed_evaluation:
        raise ValueError("production evaluation requires the exact fixed paths")
    if fixed_output.parent != run_parent or fixed_evaluation.parent != run_parent:
        raise ValueError("production evaluation paths must be fixed siblings")
    if fixed_checkpoint != run_parent / "checkpoint_antigravity":
        raise ValueError("production checkpoint path is not fixed")
    if fixed_output == fixed_evaluation or fixed_evaluation in fixed_output.parents or fixed_output in fixed_evaluation.parents:
        raise ValueError("production evaluation paths must be fixed siblings")
    store = Exp436CheckpointStore.open_read_only(fixed_checkpoint)
    if evaluation_root.exists() or evaluation_root.is_symlink():
        if evaluation_root.is_symlink() or not evaluation_root.is_dir():
            raise ValueError("evaluation root is invalid")
        publication = inspect_publication(output_root, store)
        _manifest, summary_rows, _prediction_rows = _validate_verified_publication(
            publication, store, expected_total=FROZEN_CONFIG.expected_datasets, config=FROZEN_CONFIG,
        )
        snapshot = _snapshot_frozen_tree(output_root)
        _verify_frozen_source(output_root, store, expected_total=FROZEN_CONFIG.expected_datasets, snapshot=snapshot)
        result = _read_existing_evaluation(evaluation_root)
        _validate_existing_evaluation(result, publication, summary_rows)
        if (result.prediction_sha256 != publication.prediction_sha256
                or result.manifest_sha256 != publication.manifest_sha256):
            raise ValueError("existing evaluation does not bind the verified publication")
        if result.database_sha256 != FROZEN_CONFIG.database_sha256:
            raise ValueError("existing evaluation database hash is not frozen")
        if result.code_commit != str(publication.manifest["reviewed_commit"]):
            raise ValueError("existing evaluation code commit is not the reviewed publication commit")
        return result
    publication = FrozenPublication(output_root / PUBLICATION_DIRECTORY, {})
    # The placeholder object above is intentionally not trusted; the evaluator
    # performs a fresh inspect_publication replay from disk before labels.
    return _evaluate_frozen_prediction_for_test(
        publication,
        checkpoint=store,
        evaluation_root=Path(evaluation_root),
    )


__all__ = [
    "evaluate_exp436",
    "decide_exp436_success",
]
