from dataclasses import dataclass

from experiments.exp411_near_repeat_qualification.contracts import TrainRun


@dataclass(frozen=True)
class RunRepairAudit:
    dataset: str
    instance_index: int
    raw_values_sha256: str
    repaired_values_sha256: str | None
    length: int
    nan_count: int
    posinf_count: int
    neginf_count: int
    missing_positions_sha256: str
    repair_method: str


@dataclass(frozen=True)
class DatasetRepairAudit:
    dataset: str
    runs: tuple[RunRepairAudit, ...]
    raw_train_identity_sha256: str
    repaired_train_identity_sha256: str | None
    status: str


@dataclass(frozen=True)
class MissingInventory:
    datasets: int
    affected_datasets: int
    affected_runs: int
    nonfinite_values: int
    all_missing_runs: int
    affected_datasets_sha256: str


@dataclass(frozen=True)
class PreparedDataset:
    dataset: str
    runs: tuple[TrainRun, ...]
    repair_audit: DatasetRepairAudit
