"""Low-cost, label-free readiness gate for the paper proposal r4 run.

This active-Exp448 boundary treats Exp443 as the immutable inventory/archive
authority and the standalone paper namespace as the implementation under test.
It creates no run root and imports no label-capable or evaluation module.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import ContextManager

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import (
    FROZEN_CONFIG as EXP443_FROZEN_CONFIG,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.database_antigravity import (
    load_raw_dataset_antigravity,
    pin_database_snapshot_antigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.preprocessing_antigravity import (
    POLICY_A_ANTIGRAVITY,
    prepare_dataset_antigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.sources_antigravity import (
    verify_frozen_sources_antigravity,
)
from paper.equipment_anomaly_detection.calibration_paper import upper_tail_pvalues
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import (
    MiniRocketConfig,
    score_configuration_paper,
)
from paper.equipment_anomaly_detection.sources_paper.multirocket_hydra_paper import (
    PINNED_TORCH_VERSION_PAPER,
    verify_exp84_environment_paper,
)
from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import (
    align_series_lengths_paper,
    repair_linear_edge_nearest_paper,
    z_normalize_paper,
)


READINESS_SCHEMA_ANTIGRAVITY = "paper-r4-readiness-antigravity.v1"
EXPECTED_DATASETS_ANTIGRAVITY = 1117
EXPECTED_AFFECTED_ROWS_ANTIGRAVITY = 164
EXPECTED_NONFINITE_SCALARS_ANTIGRAVITY = 983
EXPECTED_ALL_MISSING_ANTIGRAVITY = 0
EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY = (
    "DodgerLoopDay_normal_1", "DodgerLoopDay_normal_2",
    "DodgerLoopDay_normal_4", "DodgerLoopDay_normal_5",
    "DodgerLoopDay_normal_6", "DodgerLoopDay_normal_7",
    "DodgerLoopGame_normal_1", "DodgerLoopGame_normal_2",
    "DodgerLoopWeekend_normal_1", "DodgerLoopWeekend_normal_2",
    "MelbournePedestrian_normal_1", "MelbournePedestrian_normal_2",
    "MelbournePedestrian_normal_3", "MelbournePedestrian_normal_4",
    "MelbournePedestrian_normal_5", "MelbournePedestrian_normal_6",
    "MelbournePedestrian_normal_7", "MelbournePedestrian_normal_8",
    "MelbournePedestrian_normal_9",
)
EXPECTED_AFFECTED_SHA256_ANTIGRAVITY = "ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9"
EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY = "MelbournePedestrian_normal_10"
EXPECTED_UNION_DATASETS_ANTIGRAVITY = tuple(sorted((*EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY, EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY)))
EXPECTED_UNION_DATASET_COUNT_ANTIGRAVITY = 20
EXPECTED_UNION_TEST_ONLY_ROWS_ANTIGRAVITY = 3
EXPECTED_UNION_TEST_ONLY_SCALARS_ANTIGRAVITY = 3
EXPECTED_UNION_SHA256_ANTIGRAVITY = sha256(
    json.dumps(list(EXPECTED_UNION_DATASETS_ANTIGRAVITY), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()
FROZEN_ALPHAS_ANTIGRAVITY = (0.005,)
KERNEL_DIMENSIONS_ANTIGRAVITY = {1024: 1008, 4096: 4032}
SEEDS_ANTIGRAVITY = (20260721, 20261721)
SCORE_TOLERANCE_ANTIGRAVITY = 1e-6
UNKNOWN_SHA256_ANTIGRAVITY = "0" * 64
FORBIDDEN_LABEL_MODULES_ANTIGRAVITY = (
    "paper.equipment_anomaly_detection.label_access_paper",
)


class ReadinessFailureAntigravity(RuntimeError):
    """One or more bounded readiness checks failed."""

    def __init__(self, failures: tuple[str, ...]):
        values = tuple(failures)
        if not values or any(type(item) is not str or not item for item in values):
            raise ValueError("readiness failures must be nonempty strings")
        self.failures = values
        super().__init__("; ".join(values))


def _exact_int(value: object, expected: int, name: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} differs: expected={expected} actual={value}")
    return value


@dataclass(frozen=True, slots=True)
class InventoryGateAntigravity:
    dataset_count: int
    affected_datasets: tuple[str, ...]
    affected_rows: int
    nonfinite_scalars: int
    all_missing_rows: int
    affected_datasets_sha256: str
    label_access_count: int
    repaired_row_count: int

    def __post_init__(self) -> None:
        _exact_int(self.dataset_count, EXPECTED_DATASETS_ANTIGRAVITY, "dataset_count")
        affected = tuple(self.affected_datasets)
        if type(self.affected_datasets) is not tuple or affected != EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY:
            raise ValueError("affected_datasets differ from frozen authority")
        _exact_int(self.affected_rows, EXPECTED_AFFECTED_ROWS_ANTIGRAVITY, "affected_rows")
        _exact_int(self.nonfinite_scalars, EXPECTED_NONFINITE_SCALARS_ANTIGRAVITY, "nonfinite_scalars")
        _exact_int(self.all_missing_rows, EXPECTED_ALL_MISSING_ANTIGRAVITY, "all_missing_rows")
        if self.affected_datasets_sha256 != EXPECTED_AFFECTED_SHA256_ANTIGRAVITY:
            raise ValueError("affected_datasets_sha256 differs")
        _exact_int(self.label_access_count, 0, "label_access_count")
        _exact_int(self.repaired_row_count, EXPECTED_AFFECTED_ROWS_ANTIGRAVITY, "repaired_row_count")

    def to_payload_antigravity(self) -> dict[str, object]:
        return {
            "dataset_count": self.dataset_count,
            "affected_datasets": list(self.affected_datasets),
            "affected_rows": self.affected_rows,
            "nonfinite_scalars": self.nonfinite_scalars,
            "all_missing_rows": self.all_missing_rows,
            "affected_datasets_sha256": self.affected_datasets_sha256,
            "label_access_count": self.label_access_count,
            "repaired_row_count": self.repaired_row_count,
        }


@dataclass(frozen=True, slots=True)
class UnionInventoryGateAntigravity:
    dataset_count: int
    datasets: tuple[str, ...]
    test_only_dataset: str
    test_only_rows: int
    test_only_scalars: int
    label_access_count: int

    def __post_init__(self) -> None:
        _exact_int(self.dataset_count, EXPECTED_UNION_DATASET_COUNT_ANTIGRAVITY, "union_dataset_count")
        datasets = tuple(self.datasets)
        if type(self.datasets) is not tuple or datasets != EXPECTED_UNION_DATASETS_ANTIGRAVITY:
            raise ValueError("union datasets differ from official TRAIN-or-TEST authority")
        if self.test_only_dataset != EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY:
            raise ValueError("union test-only dataset differs")
        _exact_int(self.test_only_rows, EXPECTED_UNION_TEST_ONLY_ROWS_ANTIGRAVITY, "union_test_only_rows")
        _exact_int(self.test_only_scalars, EXPECTED_UNION_TEST_ONLY_SCALARS_ANTIGRAVITY, "union_test_only_scalars")
        _exact_int(self.label_access_count, 0, "union_label_access_count")

    def to_payload_antigravity(self) -> dict[str, object]:
        return {
            "dataset_count": self.dataset_count,
            "datasets": list(self.datasets),
            "datasets_sha256": EXPECTED_UNION_SHA256_ANTIGRAVITY,
            "test_only_dataset": self.test_only_dataset,
            "test_only_rows": self.test_only_rows,
            "test_only_scalars": self.test_only_scalars,
            "label_access_count": self.label_access_count,
        }


@dataclass(frozen=True, slots=True)
class ParityEvidenceAntigravity:
    dataset: str
    requested_kernels: int
    seed: int
    train_shape: tuple[int, ...]
    test_shape: tuple[int, ...]
    max_abs_difference: float
    candidate_equal: bool

    def __post_init__(self) -> None:
        if self.dataset not in EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY:
            raise ValueError("parity dataset is outside affected authority")
        if self.requested_kernels not in KERNEL_DIMENSIONS_ANTIGRAVITY or self.seed not in SEEDS_ANTIGRAVITY:
            raise ValueError("parity kernel/seed differs")
        for name, shape in (("train_shape", self.train_shape), ("test_shape", self.test_shape)):
            if type(shape) is not tuple or len(shape) != 1 or type(shape[0]) is not int or shape[0] < 1:
                raise ValueError(f"{name} is invalid")
        difference = float(self.max_abs_difference)
        if type(self.max_abs_difference) not in (int, float) or not np.isfinite(difference) or difference < 0.0:
            raise ValueError("max_abs_difference is invalid")
        if type(self.candidate_equal) is not bool:
            raise ValueError("candidate_equal must be bool")
        object.__setattr__(self, "max_abs_difference", difference)

    def to_payload_antigravity(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "requested_kernels": self.requested_kernels,
            "seed": self.seed,
            "train_shape": list(self.train_shape),
            "test_shape": list(self.test_shape),
            "max_abs_difference": self.max_abs_difference,
            "candidate_equal": self.candidate_equal,
        }


@dataclass(frozen=True, slots=True)
class R4ReadinessResultAntigravity:
    status: str
    inventory: InventoryGateAntigravity | None
    parity: tuple[ParityEvidenceAntigravity, ...]
    failures: tuple[str, ...]
    database_sha256: str
    code_closure_sha256: str
    environment: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.status not in {"ready", "not_ready"}:
            raise ValueError("readiness status is invalid")
        parity, failures = tuple(self.parity), tuple(self.failures)
        if any(not isinstance(item, ParityEvidenceAntigravity) for item in parity):
            raise ValueError("readiness parity evidence is invalid")
        if any(type(item) is not str or not item for item in failures):
            raise ValueError("readiness failures are invalid")
        if self.status == "ready" and (
            not isinstance(self.inventory, InventoryGateAntigravity)
            or len(parity) != 76
            or failures
            or self.database_sha256 != EXP443_FROZEN_CONFIG.database_sha256
        ):
            raise ValueError("ready result is incomplete")
        if self.status == "not_ready" and not failures:
            raise ValueError("not_ready result requires failures")
        for name, value in (("database_sha256", self.database_sha256), ("code_closure_sha256", self.code_closure_sha256)):
            if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} is invalid")
        if not isinstance(self.environment, Mapping):
            raise ValueError("readiness environment is invalid")
        object.__setattr__(self, "parity", parity)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))

    def bare_payload_antigravity(self) -> dict[str, object]:
        return {
            "schema": READINESS_SCHEMA_ANTIGRAVITY,
            "status": self.status,
            "inventory": None if self.inventory is None else self.inventory.to_payload_antigravity(),
            "parity": [item.to_payload_antigravity() for item in self.parity],
            "failures": list(self.failures),
            "database_sha256": self.database_sha256,
            "code_closure_sha256": self.code_closure_sha256,
            "environment": dict(self.environment),
        }


@dataclass(frozen=True, slots=True)
class R4ReadinessDependenciesAntigravity:
    verify_sources: Callable[[], object]
    affected_raw_context: Callable[[object], ContextManager[Mapping[str, object]]]
    prepare_policy_a: Callable[[object], object]
    score_configuration: Callable[..., object]
    verify_union: Callable[[object, Mapping[str, object]], UnionInventoryGateAntigravity]
    verify_environment: Callable[[], Mapping[str, object]]

    def __post_init__(self) -> None:
        if any(not callable(getattr(self, name)) for name in self.__dataclass_fields__):
            raise ValueError("readiness dependency is not callable")


def _canonical_bytes_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def canonical_result_bytes_antigravity(result: R4ReadinessResultAntigravity) -> bytes:
    if not isinstance(result, R4ReadinessResultAntigravity):
        raise ValueError("readiness result type is invalid")
    bare = result.bare_payload_antigravity()
    digest = sha256(_canonical_bytes_antigravity(bare)).hexdigest()
    return _canonical_bytes_antigravity({**bare, "result_sha256": digest})


def _package_version_antigravity(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _environment_antigravity() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "aeon": _package_version_antigravity("aeon"),
        "scikit_learn": _package_version_antigravity("scikit-learn"),
        "canonical_backend": "cpu",
        "pytorch": "non_canonical_frozen_aeon_transformer",
        "mlx": "non_canonical_frozen_aeon_transformer",
        "mps": "non_canonical_frozen_aeon_transformer",
        "outer_workers": 1,
        "inner_threads": 1,
    }


def _repo_root_antigravity() -> Path:
    return Path(__file__).resolve().parents[2]


def _code_closure_sha256_antigravity() -> str:
    root = _repo_root_antigravity()
    names = (
        "experiments/exp448_exp447_text_label_boundary_repair_antigravity/paper_r4_readiness_antigravity.py",
        "experiments/exp443_missing_normal_train_policy_comparison_antigravity/config_antigravity.py",
        "experiments/exp443_missing_normal_train_policy_comparison_antigravity/database_antigravity.py",
        "experiments/exp443_missing_normal_train_policy_comparison_antigravity/sources_antigravity.py",
        "experiments/exp443_missing_normal_train_policy_comparison_antigravity/preprocessing_antigravity.py",
        "experiments/exp443_missing_normal_train_policy_comparison_antigravity/minirocket_antigravity.py",
        "experiments/exp443_missing_normal_train_policy_comparison_antigravity/control_antigravity.py",
        "paper/equipment_anomaly_detection/calibration_paper.py",
        "paper/equipment_anomaly_detection/contracts_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/preprocessing_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/minirocket_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/multirocket_hydra_paper.py",
        "paper/equipment_anomaly_detection/hard_route_paper.py",
        "paper/equipment_anomaly_detection/proposal_score_paper.py",
    )
    hashes = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise ValueError(f"readiness code closure file is absent: {name}")
        hashes[name] = sha256(path.read_bytes()).hexdigest()
    return sha256(_canonical_bytes_antigravity(hashes)).hexdigest()


@contextmanager
def _production_raw_context_antigravity(snapshot: object):
    with TemporaryDirectory(prefix="paper_r4_readiness_antigravity_") as temporary:
        attempt = Path(temporary) / "attempt_antigravity"
        pinned = pin_database_snapshot_antigravity(
            EXP443_FROZEN_CONFIG.database_path,
            attempt,
            expected_sha256=EXP443_FROZEN_CONFIG.database_sha256,
        )
        try:
            yield {
                dataset: load_raw_dataset_antigravity(pinned, dataset)
                for dataset in EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY
            }
            if pinned.label_access_count != 0:
                raise ValueError("label_access_count differs after raw load")
        finally:
            pinned.close(terminal_evidence_durable=True)


def _prepare_policy_a_antigravity(raw: object) -> object:
    return prepare_dataset_antigravity(raw, POLICY_A_ANTIGRAVITY)


def _score_configuration_antigravity(dataset: str, train: np.ndarray, test: np.ndarray, **kwargs: object) -> object:
    del dataset
    return score_configuration_paper(train, test, config=MiniRocketConfig(), **kwargs)


def _union_gate_from_rows_antigravity(raw_values: Mapping[str, object], *, label_access_count: int) -> UnionInventoryGateAntigravity:
    affected: set[str] = set()
    test_only_rows = 0
    test_only_scalars = 0
    forbidden_fields = ("label", "raw_label", "ground_truth", "anomaly")
    for dataset, raw in raw_values.items():
        if any(any(hasattr(row, field) for field in forbidden_fields) for row in tuple(getattr(raw, "test", ()))):
            raise ValueError(f"TEST label field crossed the values-only boundary: {dataset}")
        train_missing = False
        for row in tuple(getattr(raw, "train", ())):
            values = np.asarray(getattr(row, "values", ()), dtype="<f4")
            if tuple(int(index) for index in np.flatnonzero(~np.isfinite(values))):
                train_missing = True
                affected.add(dataset)
        for row in tuple(getattr(raw, "test", ())):
            values = np.asarray(getattr(row, "values", ()), dtype="<f4")
            missing = tuple(int(index) for index in np.flatnonzero(~np.isfinite(values)))
            if missing:
                affected.add(dataset)
                if dataset == EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY and not train_missing:
                    test_only_rows += 1
                    test_only_scalars += len(missing)
    return UnionInventoryGateAntigravity(
        dataset_count=len(tuple(sorted(affected))),
        datasets=tuple(sorted(affected)),
        test_only_dataset=EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY,
        test_only_rows=test_only_rows,
        test_only_scalars=test_only_scalars,
        label_access_count=label_access_count,
    )


def _production_union_antigravity(snapshot: object, train_raw: Mapping[str, object]) -> UnionInventoryGateAntigravity:
    del train_raw
    with TemporaryDirectory(prefix="paper_r4_union_antigravity_") as temporary:
        attempt = Path(temporary) / "attempt_antigravity"
        pinned = pin_database_snapshot_antigravity(
            EXP443_FROZEN_CONFIG.database_path,
            attempt,
            expected_sha256=EXP443_FROZEN_CONFIG.database_sha256,
        )
        try:
            raw_values = {
                dataset: load_raw_dataset_antigravity(pinned, dataset)
                for dataset in EXPECTED_UNION_DATASETS_ANTIGRAVITY
            }
            if pinned.label_access_count != 0:
                raise ValueError("label_access_count differs after union raw load")
            return _union_gate_from_rows_antigravity(raw_values, label_access_count=0)
        finally:
            pinned.close(terminal_evidence_durable=True)


def _default_dependencies_antigravity() -> R4ReadinessDependenciesAntigravity:
    return R4ReadinessDependenciesAntigravity(
        verify_sources=verify_frozen_sources_antigravity,
        affected_raw_context=_production_raw_context_antigravity,
        prepare_policy_a=_prepare_policy_a_antigravity,
        score_configuration=_score_configuration_antigravity,
        verify_union=_production_union_antigravity,
        verify_environment=verify_exp84_environment_paper,
    )


def _validate_snapshot_antigravity(snapshot: object) -> None:
    datasets = tuple(getattr(snapshot, "datasets", ()))
    affected = tuple(getattr(snapshot, "affected_datasets", ()))
    if len(datasets) != EXPECTED_DATASETS_ANTIGRAVITY or datasets != tuple(sorted(set(datasets))):
        raise ValueError(f"dataset_count differs: expected=1117 actual={len(datasets)}")
    if affected != EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY or not set(affected).issubset(datasets):
        raise ValueError("affected_datasets differ from frozen authority")
    _exact_int(getattr(snapshot, "affected_rows", None), EXPECTED_AFFECTED_ROWS_ANTIGRAVITY, "affected_rows")
    _exact_int(getattr(snapshot, "nonfinite_scalars", None), EXPECTED_NONFINITE_SCALARS_ANTIGRAVITY, "nonfinite_scalars")
    _exact_int(getattr(snapshot, "all_missing_rows", None), EXPECTED_ALL_MISSING_ANTIGRAVITY, "all_missing_rows")
    _exact_int(getattr(snapshot, "label_access_count", None), 0, "label_access_count")
    if getattr(snapshot, "affected_datasets_sha256", None) != EXPECTED_AFFECTED_SHA256_ANTIGRAVITY:
        raise ValueError("affected_datasets_sha256 differs")
    recomputed_affected_sha256 = sha256(
        json.dumps(list(affected), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if recomputed_affected_sha256 != EXPECTED_AFFECTED_SHA256_ANTIGRAVITY:
        raise ValueError("affected_datasets identity differs")
    source_hashes = getattr(snapshot, "source_hashes", None)
    if not isinstance(source_hashes, Mapping) or source_hashes.get("database") != EXP443_FROZEN_CONFIG.database_sha256:
        raise ValueError("database SHA256 differs from frozen authority")
    cache_inventory = getattr(snapshot, "cache_inventory", None)
    if not isinstance(cache_inventory, Mapping) or not set(affected).issubset(cache_inventory):
        raise ValueError("cache_inventory is incomplete")


def _validate_label_import_firewall_antigravity() -> None:
    loaded = tuple(name for name in FORBIDDEN_LABEL_MODULES_ANTIGRAVITY if name in sys.modules)
    if loaded:
        raise ValueError(f"label-capable module is loaded: {loaded[0]}")


def _paper_matrices_antigravity(raw: object) -> tuple[np.ndarray, np.ndarray]:
    target = getattr(raw, "target_length", None)
    if type(target) is not int or not 8 <= target <= 2048:
        raise ValueError("paper target_length is invalid")
    train_rows = tuple(repair_linear_edge_nearest_paper(row.values) for row in getattr(raw, "train", ()))
    test_rows = tuple(repair_linear_edge_nearest_paper(row.values) for row in getattr(raw, "test", ()))
    train = z_normalize_paper(align_series_lengths_paper(train_rows, target))
    test = z_normalize_paper(align_series_lengths_paper(test_rows, target))
    if train.dtype != np.dtype("<f4") or test.dtype != np.dtype("<f4") or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("paper prepared matrices are not finite little-endian float32")
    return train, test


def _inventory_from_snapshot_antigravity(snapshot: object, raw_values: Mapping[str, object]) -> InventoryGateAntigravity:
    if set(raw_values) != set(EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY):
        raise ValueError("affected raw population differs")
    repaired_rows = 0
    nonfinite_scalars = 0
    all_missing_rows = 0
    for dataset in EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY:
        raw = raw_values[dataset]
        if getattr(raw, "dataset", None) != dataset:
            raise ValueError(f"affected raw dataset identity differs: {dataset}")
        test_rows = tuple(getattr(raw, "test", ()))
        if not test_rows:
            raise ValueError(f"TEST rows are absent: {dataset}")
        forbidden_fields = ("label", "raw_label", "ground_truth", "anomaly")
        if any(any(hasattr(row, field) for field in forbidden_fields) for row in test_rows):
            raise ValueError(f"TEST label field crossed the values-only boundary: {dataset}")
        for row in getattr(raw, "train", ()):
            values = np.asarray(getattr(row, "values", ()), dtype="<f4")
            missing = tuple(int(index) for index in np.flatnonzero(~np.isfinite(values)))
            if missing:
                repaired_rows += 1
                nonfinite_scalars += len(missing)
                if not np.isfinite(values).any():
                    all_missing_rows += 1
                repaired = repair_linear_edge_nearest_paper(values)
                if repaired.dtype != np.dtype("<f4") or repaired.shape != values.shape or not np.isfinite(repaired).all():
                    raise ValueError(f"repair remains nonfinite: dataset={dataset} row={getattr(row, 'instance_index', None)}")
    _exact_int(nonfinite_scalars, EXPECTED_NONFINITE_SCALARS_ANTIGRAVITY, "raw nonfinite_scalars")
    _exact_int(all_missing_rows, EXPECTED_ALL_MISSING_ANTIGRAVITY, "raw all_missing_rows")
    return InventoryGateAntigravity(
        dataset_count=len(tuple(getattr(snapshot, "datasets"))),
        affected_datasets=tuple(getattr(snapshot, "affected_datasets")),
        affected_rows=getattr(snapshot, "affected_rows"),
        nonfinite_scalars=getattr(snapshot, "nonfinite_scalars"),
        all_missing_rows=getattr(snapshot, "all_missing_rows"),
        affected_datasets_sha256=getattr(snapshot, "affected_datasets_sha256"),
        label_access_count=getattr(snapshot, "label_access_count"),
        repaired_row_count=repaired_rows,
    )


def _candidate_indices_antigravity(train: np.ndarray, test: np.ndarray) -> tuple[int, ...]:
    values = np.flatnonzero(upper_tail_pvalues(train, test) <= FROZEN_ALPHAS_ANTIGRAVITY[0])
    return tuple(int(index) for index in values)


def _database_sha256_antigravity(snapshot: object | None) -> str:
    source_hashes = getattr(snapshot, "source_hashes", {}) if snapshot is not None else {}
    value = source_hashes.get("database") if isinstance(source_hashes, Mapping) else None
    return value if type(value) is str and len(value) == 64 else UNKNOWN_SHA256_ANTIGRAVITY


def run_r4_readiness_antigravity(
    *,
    dependencies: R4ReadinessDependenciesAntigravity | None = None,
    progress: Callable[[str], None] | None = None,
) -> R4ReadinessResultAntigravity:
    """Run the inventory gate, then the bounded 19-dataset score parity gate."""
    deps = _default_dependencies_antigravity() if dependencies is None else dependencies
    if not isinstance(deps, R4ReadinessDependenciesAntigravity):
        raise ValueError("readiness dependencies have the wrong type")
    emit = (lambda message: print(message, file=sys.stderr, flush=True)) if progress is None else progress
    closure = _code_closure_sha256_antigravity()
    environment = _environment_antigravity()
    snapshot = None
    emit("inventory start")
    try:
        _validate_label_import_firewall_antigravity()
        snapshot = deps.verify_sources()
        _validate_snapshot_antigravity(snapshot)
        with deps.affected_raw_context(snapshot) as raw_values:
            inventory = _inventory_from_snapshot_antigravity(snapshot, raw_values)
            detached_raw = dict(raw_values)
    except (OSError, RuntimeError, ValueError) as error:
        failure = f"inventory: {type(error).__name__}: {error}"
        return R4ReadinessResultAntigravity(
            "not_ready", None, (), (failure,), _database_sha256_antigravity(snapshot), closure, environment,
        )
    emit("inventory ready")
    try:
        union = deps.verify_union(snapshot, detached_raw)
        if not isinstance(union, UnionInventoryGateAntigravity):
            raise ValueError("union inventory type is invalid")
        environment = {**environment, "union_inventory": union.to_payload_antigravity(), "pinned_torch": PINNED_TORCH_VERSION_PAPER}
        deps.verify_environment()
    except (OSError, RuntimeError, ValueError) as error:
        failure = f"union_or_environment: {type(error).__name__}: {error}"
        return R4ReadinessResultAntigravity(
            "not_ready", inventory, (), (failure,), _database_sha256_antigravity(snapshot), closure, environment,
        )

    evidence: list[ParityEvidenceAntigravity] = []
    failures: list[str] = []
    for ordinal, dataset in enumerate(EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY, 1):
        emit(f"parity {ordinal}/19 {dataset}")
        raw = detached_raw[dataset]
        try:
            paper_train, paper_test = _paper_matrices_antigravity(raw)
            prepared = deps.prepare_policy_a(raw)
            reference_train = np.asarray(getattr(prepared, "train_values", ()), dtype="<f4")
            reference_test = np.asarray(getattr(prepared, "test_values", ()), dtype="<f4")
            if paper_train.dtype != reference_train.dtype or paper_test.dtype != reference_test.dtype or paper_train.shape != reference_train.shape or paper_test.shape != reference_test.shape or paper_train.tobytes() != reference_train.tobytes() or paper_test.tobytes() != reference_test.tobytes():
                raise ValueError("paper matrices differ from frozen Policy A")
            arrays = getattr(snapshot, "cache_inventory")[dataset]["arrays"]
            for kernels, expected_dimension in KERNEL_DIMENSIONS_ANTIGRAVITY.items():
                for seed_index, seed in enumerate(SEEDS_ANTIGRAVITY):
                    paper_score = deps.score_configuration(
                        dataset,
                        paper_train,
                        paper_test,
                        requested_kernels=kernels,
                        seed=seed,
                        expected_dimension=expected_dimension,
                    )
                    archive_train = np.asarray(arrays[f"kernels_{kernels}__train_by_seed"][seed_index], dtype="<f8")
                    archive_test = np.asarray(arrays[f"kernels_{kernels}__test_by_seed"][seed_index], dtype="<f8")
                    paper_oof = np.asarray(getattr(paper_score, "train_oof"), dtype="<f8")
                    paper_query = np.asarray(getattr(paper_score, "test_score"), dtype="<f8")
                    if paper_oof.shape != archive_train.shape or paper_query.shape != archive_test.shape or not all(np.isfinite(item).all() for item in (paper_oof, paper_query, archive_train, archive_test)):
                        raise ValueError(f"score shape/finite mismatch kernel={kernels} seed={seed}")
                    train_difference = float(np.max(np.abs(paper_oof - archive_train))) if paper_oof.size else 0.0
                    test_difference = float(np.max(np.abs(paper_query - archive_test))) if paper_query.size else 0.0
                    difference = max(train_difference, test_difference)
                    candidate_equal = _candidate_indices_antigravity(paper_oof, paper_query) == _candidate_indices_antigravity(archive_train, archive_test)
                    evidence.append(ParityEvidenceAntigravity(dataset, kernels, seed, tuple(paper_oof.shape), tuple(paper_query.shape), difference, candidate_equal))
                    if difference > SCORE_TOLERANCE_ANTIGRAVITY:
                        failures.append(f"{dataset}/kernel{kernels}/seed{seed} score tolerance exceeded: {difference}")
                    if not candidate_equal:
                        failures.append(f"{dataset}/kernel{kernels}/seed{seed} candidate mismatch")
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append(f"{dataset}: {type(error).__name__}: {error}")

    try:
        _validate_label_import_firewall_antigravity()
    except ValueError as error:
        failures.append(f"label firewall: {error}")

    status = "ready" if not failures and len(evidence) == 76 else "not_ready"
    if len(evidence) != 76 and not failures:
        failures.append(f"parity comparison count differs: {len(evidence)}")
        status = "not_ready"
    return R4ReadinessResultAntigravity(
        status,
        inventory,
        tuple(evidence),
        tuple(failures),
        _database_sha256_antigravity(snapshot),
        closure,
        environment,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit("paper r4 readiness accepts no arguments")
    result = run_r4_readiness_antigravity()
    sys.stdout.write(canonical_result_bytes_antigravity(result).decode("utf-8"))
    sys.stdout.flush()
    return 0 if result.status == "ready" else 2


__all__ = [
    "EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY", "EXPECTED_AFFECTED_SHA256_ANTIGRAVITY",
    "EXPECTED_UNION_DATASETS_ANTIGRAVITY", "EXPECTED_UNION_DATASET_COUNT_ANTIGRAVITY",
    "EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY", "EXPECTED_UNION_TEST_ONLY_ROWS_ANTIGRAVITY",
    "EXPECTED_UNION_TEST_ONLY_SCALARS_ANTIGRAVITY", "EXPECTED_UNION_SHA256_ANTIGRAVITY",
    "InventoryGateAntigravity", "UnionInventoryGateAntigravity", "ParityEvidenceAntigravity", "R4ReadinessDependenciesAntigravity",
    "R4ReadinessResultAntigravity", "ReadinessFailureAntigravity",
    "canonical_result_bytes_antigravity", "main", "run_r4_readiness_antigravity",
]


if __name__ == "__main__":
    raise SystemExit(main())
