from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenAuditInputs:
    labels_by_dataset: Mapping[str, tuple[int, ...]]
    hard_by_dataset: Mapping[str, frozenset[int]]
    input_sha256: Mapping[str, str]


@dataclass(frozen=True)
class DatasetProfile:
    dataset: str
    family: str
    train_normal_count: int
    test_total_count: int
    test_anomaly_count: int
    test_normal_count: int
    exact_overlap_count: int
    exact_overlap_fraction: float


@dataclass(frozen=True)
class DatasetMetric:
    dataset: str
    family: str
    hard_count: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float
    f1: float


@dataclass(frozen=True)
class FamilyMetric:
    family: str
    dataset_count: int
    mean_dataset_f1: float
    mean_dataset_fp: float
    pooled_tp: int
    pooled_fp: int
    pooled_fn: int
    pooled_precision: float
    pooled_recall: float
    pooled_f1: float


@dataclass(frozen=True)
class AuditResult:
    dataset_metrics: tuple[DatasetMetric, ...]
    family_metrics: tuple[FamilyMetric, ...]
    weight_sensitivity: tuple[Mapping[str, object], ...]
    family_influence: tuple[Mapping[str, object], ...]
    stratified_metrics: tuple[Mapping[str, object], ...]
    summary: Mapping[str, object]


def family_from_dataset(dataset: str) -> str:
    if not isinstance(dataset, str) or dataset.count("_normal_") != 1:
        raise ValueError("Exp410 dataset must contain one _normal_ marker")
    family, suffix = dataset.split("_normal_", 1)
    if not family or not suffix.isdecimal():
        raise ValueError("Exp410 dataset family and numeric class suffix are required")
    return family
