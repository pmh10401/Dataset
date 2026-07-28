from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainRun:
    dataset: str
    instance_index: int
    values: tuple[float, ...]
    values_sha256: str


@dataclass(frozen=True)
class IdentityFold:
    fit_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    query_identity_sha256: tuple[str, ...]


@dataclass(frozen=True)
class DatasetQualification:
    dataset: str
    family: str
    identity_group_count: int
    normal_total: int
    normal_abstained: int
    normal_near: Mapping[str, int]
    synthetic_total: Mapping[str, int]
    synthetic_near: Mapping[str, int]
    perturbation_total: Mapping[str, Mapping[str, int]]
    perturbation_near: Mapping[str, Mapping[str, int]]
