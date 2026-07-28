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
