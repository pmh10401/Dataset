from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


REQUIRED_ARRAYS = (
    "spectral_train",
    "spectral_test",
    "shapelet_train",
    "shapelet_test",
    "interval_train",
    "interval_test",
)


@dataclass(frozen=True)
class DatasetContract:
    dataset: str
    n_train: int
    n_test: int

    @property
    def train_bin(self) -> str:
        return train_size_bin(self.n_train)

    @property
    def minimum_attainable_p(self) -> float | None:
        if self.n_train < 5:
            return None
        return 1.0 / (self.n_train + 1)


@dataclass(frozen=True)
class CacheAuditRow:
    dataset: str
    file_path: str
    declared_hash: str
    actual_hash: str
    n_train_expected: int | None
    n_test_expected: int | None
    source_config_hash: str
    contains_test_labels: bool | None
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class CacheAuditResult:
    rows: tuple[CacheAuditRow, ...]
    expected_dataset_count: int
    duplicate_datasets: tuple[str, ...]
    missing_datasets: tuple[str, ...]
    unexpected_datasets: tuple[str, ...]

    @property
    def valid_count(self) -> int:
        return sum(row.valid for row in self.rows)

    @property
    def complete(self) -> bool:
        return (
            len(self.rows) == self.expected_dataset_count
            and self.valid_count == self.expected_dataset_count
            and not self.duplicate_datasets
            and not self.missing_datasets
            and not self.unexpected_datasets
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_size_bin(n_train: int) -> str:
    if n_train < 5:
        return "1-4"
    if n_train < 20:
        return "5-19"
    if n_train < 50:
        return "20-49"
    if n_train < 100:
        return "50-99"
    if n_train < 200:
        return "100-199"
    return "ge200"


def alpha_resolution(n_train: int, alpha: float) -> str:
    if n_train < 5:
        return "abstained_n_train_lt5"
    minimum_p = 1.0 / (n_train + 1)
    if minimum_p <= alpha or math.isclose(minimum_p, alpha, abs_tol=1e-12):
        return "attainable"
    return "resolution_blocked"


def _resolve_cache_path(file_path: str, repository_root: Path) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else repository_root / path


def audit_exp162_cache(
    manifest_path: Path,
    dataset_contracts: Mapping[str, tuple[int, int]],
    *,
    expected_dataset_count: int,
    repository_root: Path,
    expected_source_config_hash: str | None = None,
) -> CacheAuditResult:
    manifest_path = Path(manifest_path)
    repository_root = Path(repository_root)
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))

    names = [row.get("dataset", "") for row in manifest_rows]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)

    expected_names = set(dataset_contracts)
    manifest_names = set(names)
    missing = tuple(sorted(expected_names - manifest_names))
    unexpected = tuple(sorted(manifest_names - expected_names))

    audit_rows: list[CacheAuditRow] = []
    for manifest_row in manifest_rows:
        dataset = manifest_row.get("dataset", "")
        file_path_text = manifest_row.get("file_path", "")
        declared_hash = manifest_row.get("file_hash", "")
        cache_path = _resolve_cache_path(file_path_text, repository_root)
        expected_lengths = dataset_contracts.get(dataset)
        n_train_expected = expected_lengths[0] if expected_lengths else None
        n_test_expected = expected_lengths[1] if expected_lengths else None
        errors: list[str] = []
        actual_hash = ""
        source_config_hash = ""
        contains_test_labels: bool | None = None

        if dataset in duplicates:
            errors.append("duplicate_dataset")
        if expected_lengths is None:
            errors.append("unexpected_dataset")
        if not cache_path.is_file():
            errors.append("missing_file")
        else:
            actual_hash = sha256_file(cache_path)
            if actual_hash != declared_hash:
                errors.append("file_hash_mismatch")
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    available_arrays = set(cache.files)
                    for array_name in REQUIRED_ARRAYS:
                        if array_name not in available_arrays:
                            errors.append(f"missing_array:{array_name}")
                            continue
                        values = np.asarray(cache[array_name])
                        if not np.isfinite(values).all():
                            errors.append(f"nonfinite_array:{array_name}")
                        expected_length = (
                            n_train_expected
                            if array_name.endswith("_train")
                            else n_test_expected
                        )
                        if (
                            expected_length is not None
                            and values.size != expected_length
                        ):
                            errors.append(f"length_mismatch:{array_name}")
                    if expected_source_config_hash is not None:
                        if "metadata_json" not in available_arrays:
                            errors.append("missing_metadata_json")
                        else:
                            metadata = json.loads(str(cache["metadata_json"].item()))
                            source_config_hash = str(
                                metadata.get("source_config_hash", "")
                            )
                            contains_test_labels = metadata.get(
                                "contains_test_labels"
                            )
                            if source_config_hash != expected_source_config_hash:
                                errors.append("source_config_hash_mismatch")
                            if metadata.get("dataset") != dataset:
                                errors.append("metadata_dataset_mismatch")
                            if metadata.get("train_rows") != n_train_expected:
                                errors.append("metadata_train_length_mismatch")
                            if metadata.get("test_rows") != n_test_expected:
                                errors.append("metadata_test_length_mismatch")
                            if contains_test_labels is not False:
                                errors.append("metadata_contains_test_labels")
            except (OSError, ValueError) as exc:
                errors.append(f"npz_read_error:{type(exc).__name__}")

        audit_rows.append(
            CacheAuditRow(
                dataset=dataset,
                file_path=str(cache_path),
                declared_hash=declared_hash,
                actual_hash=actual_hash,
                n_train_expected=n_train_expected,
                n_test_expected=n_test_expected,
                source_config_hash=source_config_hash,
                contains_test_labels=contains_test_labels,
                errors=tuple(sorted(set(errors))),
            )
        )

    return CacheAuditResult(
        rows=tuple(audit_rows),
        expected_dataset_count=expected_dataset_count,
        duplicate_datasets=tuple(sorted(duplicates)),
        missing_datasets=missing,
        unexpected_datasets=unexpected,
    )
