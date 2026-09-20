"""Pinned verification of Exp443's database and inherited source artifacts.

This module is deliberately source-only.  It parses the frozen Exp249/291/404
JSON and cache authority, compares the Exp414 raw audit with a fresh database
scan, and proves the frozen routing algebra before any score code is imported.
No TEST label or evaluation table is read.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config_antigravity import FROZEN_CONFIG
from .database_antigravity import (
    MissingInventoryAntigravity,
    parse_dataset_family_antigravity,
    pin_database_snapshot_antigravity,
    scan_missing_inventory_antigravity,
)


_CACHE_FIELDS = frozenset({
    "abstained", "abstention_reason", "cache_file", "cache_sha256", "config_hash",
    "contains_test_labels", "dataset", "experiment", "feature_dimensions", "metadata_sha256",
    "n_test", "n_train", "source", "target_length",
})
_EXP249_REQUIRED_FIELDS = frozenset({
    "alpha", "attainable_p", "calibration_abstained", "candidate_indices",
    "contains_evaluation_metrics", "contains_test_labels", "dataset", "experiment",
    "experiment_id", "frozen_hard_indices", "frozen_priority_indices", "frozen_standard_indices",
    "hard_confirmation_indices", "hard_policy_changed", "n_test", "outside_lane_indices",
    "priority_confirmation_indices", "resolution_blocked", "source", "standard_priority_indices", "variant",
})
_EXP291_REQUIRED_FIELDS = frozenset({
    "automatic_hard_indices", "automatic_normal_exact_indices", "automatic_normal_rule",
    "contains_evaluation_metrics", "contains_test_labels", "dataset", "default_review_indices",
    "experiment_id", "n_test", "priority_review_indices", "prospective_validated", "recovery_priority_scope",
    "recovery_shadow_review_indices", "research_only_recovery_priority_indices", "retrospective_integration",
    "shadow_review_indices", "source_baseline_hard_indices", "standard_review_indices",
})
_EXP414_FIELDS = frozenset({
    "dataset", "instance_index", "raw_values_sha256", "repaired_values_sha256", "length",
    "nan_count", "posinf_count", "neginf_count", "missing_positions_sha256", "repair_method",
})
EXP249_CACHE_CONFIG_HASH_ANTIGRAVITY = "8ae2ab74764db26d76e563030ad99346fa3c1a6ef97164c3b5d590b443735921"

_SOURCE_PRE_OPEN_HOOK_ANTIGRAVITY: Callable[[Path], None] | None = None


class DuplicateJsonKeyError(ValueError):
    """Raised rather than silently accepting the last duplicate JSON key."""


def parse_json_object_antigravity(payload: bytes | str | Path, name: str = "JSON") -> Mapping[str, object]:
    if isinstance(payload, Path):
        payload = payload.read_bytes()
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} is not UTF-8") from error
    elif isinstance(payload, str):
        text = payload
    else:
        raise ValueError(f"{name} payload is invalid")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise DuplicateJsonKeyError(f"{name} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"{name} contains {value}")))
    except DuplicateJsonKeyError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(result, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return result


def _read_regular_bytes_antigravity(path: Path, expected_sha256: str | None = None) -> tuple[bytes, str]:
    path = Path(path)
    try:
        initial = path.lstat()
    except OSError as error:
        raise ValueError(f"source is unavailable: {path}") from error
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode) or initial.st_nlink < 1:
        raise ValueError(f"source is not a regular linked file: {path}")
    hook = _SOURCE_PRE_OPEN_HOOK_ANTIGRAVITY
    if hook is not None:
        hook(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError as error:
        raise ValueError(f"source open failed: {path}") from error
    try:
        before = os.fstat(descriptor)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (initial.st_dev, initial.st_ino):
            raise ValueError(f"source inode changed before read: {path}")
        digest = sha256()
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        named = path.lstat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"source inode changed while reading: {path}")
        if (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns, named.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or stat.S_ISLNK(named.st_mode):
            raise ValueError(f"source pathname changed while reading: {path}")
        observed = digest.hexdigest()
        if expected_sha256 is not None and observed != expected_sha256:
            raise ValueError(f"source SHA256 mismatch: {path.name}")
        return b"".join(chunks), observed
    except OSError as error:
        raise ValueError(f"source read failed: {path}") from error
    finally:
        os.close(descriptor)


def _read_expected(path: Path, expected: str) -> bytes:
    return _read_regular_bytes_antigravity(path, expected)[0]


def validate_coordinate_antigravity(dataset: str, index: object, n_test: int) -> tuple[str, int]:
    if type(dataset) is not str or not dataset or type(index) is not int or index < 0 or type(n_test) is not int or n_test < 1 or index >= n_test:
        raise ValueError("coordinate is outside TEST identity")
    return dataset, index


def _coordinates(value: object, dataset: str, n_test: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError("coordinate field must be a list")
    result = tuple(validate_coordinate_antigravity(dataset, item, n_test)[1] for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError("coordinate field is not sorted and unique")
    return result


def _sha256_canonical(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _family(dataset: str) -> str:
    return parse_dataset_family_antigravity(dataset)


def _freeze_array_antigravity(array: np.ndarray) -> np.ndarray:
    """Return a read-only ndarray backed by an immutable bytes object."""
    contiguous = np.array(array, copy=True, order="C")
    payload = contiguous.tobytes(order="C")
    frozen = np.frombuffer(payload, dtype=contiguous.dtype).reshape(contiguous.shape)
    if frozen.flags.writeable:
        frozen.setflags(write=False)
    return frozen


def load_pinned_npz_antigravity(path: Path, payload: bytes | None = None) -> Mapping[str, np.ndarray]:
    if payload is None:
        payload, _ = _read_regular_bytes_antigravity(Path(path))
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {str(name): _freeze_array_antigravity(np.asarray(archive[name])) for name in archive.files}
    except (ValueError, OSError, EOFError) as error:
        raise ValueError(f"NPZ cache is invalid: {Path(path).name}") from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("NPZ object arrays are forbidden")
    return MappingProxyType(arrays)


def parse_cache_manifest_antigravity(
    manifest_path: Path,
    source_root: Path | None = None,
    *,
    expected_sha256: str | None = None,
    expected_dataset_count: int = 1117,
) -> Mapping[str, Mapping[str, object]]:
    manifest_path = Path(manifest_path)
    if type(expected_dataset_count) is not int or expected_dataset_count < 1:
        raise ValueError("cache dataset authority is invalid")
    payload, _ = _read_regular_bytes_antigravity(manifest_path, expected_sha256)
    root = Path(source_root) if source_root is not None else manifest_path.parent
    try:
        text = payload.decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("cache manifest is invalid CSV") from error
    if not rows or any(set(row) != _CACHE_FIELDS for row in rows):
        raise ValueError("cache manifest schema is invalid")
    result: dict[str, Mapping[str, object]] = {}
    config_hash: str | None = None
    cache_files: set[str] = set()
    metadata_files: set[str] = set()
    for row in rows:
        dataset = row["dataset"]
        if not dataset or dataset in result:
            raise ValueError("cache manifest has duplicate dataset")
        if row["experiment"] != "249" or row["source"] != "official_minirocket" or row["contains_test_labels"] != "False":
            raise ValueError("cache manifest authority fields are invalid")
        if row["abstained"] == "True":
            # The frozen cache records the official minimum-train abstention
            # (162 datasets with four normal TRAIN rows).  It is still a
            # required, hash-pinned NPZ/metadata pair and carries no labels.
            if row["abstention_reason"] != "n_train_below_5":
                raise ValueError("Exp249 cache abstention reason is invalid")
        elif row["abstained"] != "False" or row["abstention_reason"]:
            raise ValueError("Exp249 cache abstention flag is invalid")
        if config_hash is None:
            config_hash = row["config_hash"]
        if row["config_hash"] != config_hash or config_hash != EXP249_CACHE_CONFIG_HASH_ANTIGRAVITY or not re.fullmatch(r"[0-9a-f]{64}", config_hash):
            raise ValueError("cache config hash is inconsistent")
        n_train, n_test, target = (int(row[key]) for key in ("n_train", "n_test", "target_length"))
        if n_train < 1 or n_test < 1 or target < 1:
            raise ValueError("cache dimensions are invalid")
        relative = Path(row["cache_file"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npz":
            raise ValueError("cache path is invalid")
        cache_path = (root / relative).resolve(strict=False)
        cache_root = (root / "cache").resolve(strict=False)
        if cache_root not in cache_path.parents:
            raise ValueError("cache path escapes cache root")
        metadata_path = cache_path.with_suffix(".json")
        cache_files.add(cache_path.name)
        metadata_files.add(metadata_path.name)
        if row["dataset"] in result:
            raise ValueError("cache manifest dataset is duplicated")
        npz_payload, npz_digest = _read_regular_bytes_antigravity(cache_path, row["cache_sha256"])
        metadata_payload, metadata_digest = _read_regular_bytes_antigravity(metadata_path, row["metadata_sha256"])
        metadata = parse_json_object_antigravity(metadata_payload, metadata_path.name)
        if set(metadata) != {"abstained", "abstention_reason", "cache_file", "cache_sha256", "config", "config_hash", "contains_test_labels", "dataset", "experiment", "feature_dimensions", "n_test", "n_train", "source", "target_length"}:
            raise ValueError("cache metadata schema is invalid")
        if (
            metadata["dataset"] != dataset
            or metadata["cache_file"] != row["cache_file"]
            or metadata["cache_sha256"] != npz_digest
            or metadata["config_hash"] != config_hash
            or metadata["contains_test_labels"] is not False
            or metadata["abstained"] is not (row["abstained"] == "True")
            or (metadata["abstention_reason"] or "") != row["abstention_reason"]
            or metadata["experiment"] != 249
            or metadata["source"] != "official_minirocket"
            or metadata["n_train"] != n_train
            or metadata["n_test"] != n_test
            or metadata["target_length"] != target
        ):
            raise ValueError("cache metadata identity is invalid")
        arrays = load_pinned_npz_antigravity(cache_path, npz_payload)
        expected_arrays = {"kernels_1024__train_by_seed", "kernels_1024__test_by_seed", "kernels_4096__train_by_seed", "kernels_4096__test_by_seed"}
        if set(arrays) != expected_arrays:
            raise ValueError("cache NPZ schema is invalid")
        expected_train = 0 if row["abstained"] == "True" else n_train
        expected_test = 0 if row["abstained"] == "True" else n_test
        for name, expected_columns in (
            ("kernels_1024__train_by_seed", expected_train),
            ("kernels_1024__test_by_seed", expected_test),
            ("kernels_4096__train_by_seed", expected_train),
            ("kernels_4096__test_by_seed", expected_test),
        ):
            array = arrays[name]
            if array.dtype != np.dtype("<f8") or array.ndim != 2 or array.shape != (2, expected_columns) or not np.isfinite(array).all():
                raise ValueError("cache NPZ type, shape, or finite contract is invalid")
        result[dataset] = MappingProxyType({
            "cache_file": cache_path, "metadata_file": metadata_path,
            "cache_sha256": npz_digest, "metadata_sha256": metadata_digest,
            "config_hash": config_hash, "n_train": n_train, "n_test": n_test,
            "target_length": target, "abstained": row["abstained"] == "True", "arrays": arrays,
        })
    cache_dir = root / "cache"
    try:
        entries = {item.name for item in cache_dir.iterdir() if not item.name.startswith("._")}
    except OSError as error:
        raise ValueError("cache directory is invalid") from error
    expected_entries = cache_files | metadata_files
    if entries != expected_entries:
        raise ValueError("cache inventory has missing or extra artifacts")
    if len(result) != expected_dataset_count:
        raise ValueError("Exp249 cache manifest dataset count disagrees with authority")
    return MappingProxyType(result)


def _parse_jsonl(path: Path, expected_sha256: str, name: str) -> tuple[Mapping[str, object], ...]:
    payload = _read_expected(path, expected_sha256)
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{name} has a blank row")
        rows.append(parse_json_object_antigravity(line, f"{name} row {line_number}"))
    return tuple(rows)


def _validate_exp249_prediction(rows: Sequence[Mapping[str, object]], manifest: Mapping[str, object]) -> tuple[tuple[str, ...], Mapping[str, Mapping[int, tuple[int, ...]]], Mapping[str, Mapping[str, tuple[int, ...]]]]:
    if len(rows) != 17872 or len({row.get("dataset") for row in rows}) != 1117:
        raise ValueError("Exp249 prediction population is invalid")
    if set(manifest) != {"contains_evaluation_metrics", "contains_test_labels", "datasets", "experiment", "hard_policy_changed", "prediction_sha256", "rows", "source"} or manifest["rows"] != 17872 or manifest["datasets"] != 1117 or manifest["contains_test_labels"] is not False or manifest["contains_evaluation_metrics"] is not False:
        raise ValueError("Exp249 prediction manifest is invalid")
    seen: set[tuple[str, str, float]] = set()
    candidates: dict[str, dict[int, tuple[int, ...]]] = {}
    lanes: dict[str, dict[str, tuple[int, ...]]] = {}
    for row in rows:
        if set(row) != _EXP249_REQUIRED_FIELDS or row["experiment"] != 249 or row["experiment_id"] != "experiment_249_official_minirocket" or row["source"] != "official_minirocket" or row["contains_test_labels"] is not False or row["contains_evaluation_metrics"] is not False or row["hard_policy_changed"] is not False:
            raise ValueError("Exp249 prediction row schema/flags are invalid")
        dataset, variant, alpha = row["dataset"], row["variant"], row["alpha"]
        n_test = row["n_test"]
        if type(dataset) is not str or not dataset or variant not in {"kernels_1024_seed_intersection", "kernels_1024_seed_median", "kernels_4096_seed_intersection", "kernels_4096_seed_median"} or type(alpha) not in (int, float) or alpha not in {0.005, 0.01, 0.02, 0.05} or type(n_test) is not int or n_test < 1:
            raise ValueError("Exp249 prediction identity is invalid")
        key = (dataset, variant, float(alpha))
        if key in seen:
            raise ValueError("Exp249 prediction has duplicate variant/alpha")
        seen.add(key)
        for field in ("candidate_indices", "frozen_hard_indices", "frozen_priority_indices", "frozen_standard_indices", "hard_confirmation_indices", "outside_lane_indices", "priority_confirmation_indices", "standard_priority_indices"):
            _coordinates(row[field], dataset, n_test)
        if alpha == 0.005:
            kernel = 1024 if variant.startswith("kernels_1024") else 4096
            if variant.endswith("seed_intersection"):
                candidates.setdefault(dataset, {})[kernel] = tuple(row["candidate_indices"])
            lanes.setdefault(dataset, {})["hard"] = tuple(row["frozen_hard_indices"])
            lanes[dataset]["standard"] = tuple(row["frozen_standard_indices"])
            lanes[dataset]["priority"] = tuple(row["frozen_priority_indices"])
            if variant.endswith("seed_intersection"):
                previous_promotion = set(lanes[dataset].get("promotion", ()))
                previous_promotion.update(row["standard_priority_indices"])
                lanes[dataset]["promotion"] = tuple(sorted(previous_promotion))
    if len(seen) != 17872 or any(len(value) != 2 for value in candidates.values()) or len(candidates) != 1117 or len(lanes) != 1117:
        raise ValueError("Exp249 prediction variants are incomplete")
    return tuple(sorted(lanes)), MappingProxyType({name: MappingProxyType(values) for name, values in sorted(candidates.items())}), MappingProxyType({name: MappingProxyType(values) for name, values in sorted(lanes.items())})


def _parse_exp291(rows: Sequence[Mapping[str, object]], manifest: Mapping[str, object]) -> tuple[Mapping[str, tuple[int, ...]], Mapping[str, int], frozenset[tuple[str, int]]]:
    if len(rows) != 1117 or len({row.get("dataset") for row in rows}) != 1117:
        raise ValueError("Exp291 routing population is invalid")
    if set(manifest) != {"contains_evaluation_metrics", "contains_test_labels", "default_review_excludes_recovery_priority", "exp272_prediction_sha256", "exp290_prediction_sha256", "experiment_id", "lane_counts", "prediction_sha256", "recovery_priority_scope", "retrospective_integration", "rows"} or manifest.get("rows") != 1117 or manifest.get("prediction_sha256") != FROZEN_CONFIG.exp291_routing_sha256 or manifest.get("contains_test_labels") is not False or manifest.get("contains_evaluation_metrics") is not False or manifest.get("lane_counts") != {"automatic_hard_indices": 2218, "automatic_normal_exact_indices": 35, "priority_review_indices": 31, "recovery_shadow_review_indices": 42, "research_only_recovery_priority_indices": 137, "shadow_review_indices": 2, "standard_review_indices": 806}:
        raise ValueError("Exp291 manifest is invalid")
    lanes: dict[str, tuple[int, ...]] = {}
    normal: dict[str, tuple[int, ...]] = {}
    standard: dict[str, tuple[int, ...]] = {}
    priority: dict[str, tuple[int, ...]] = {}
    n_test: dict[str, int] = {}
    for row in rows:
        if set(row) != _EXP291_REQUIRED_FIELDS or row["contains_test_labels"] is not False or row["contains_evaluation_metrics"] is not False or row["experiment_id"] != "experiment_291_final_operational_routing_freeze" or row["automatic_normal_rule"] != "sqlite_values_blob_byte_exact":
            raise ValueError("Exp291 row schema/flags are invalid")
        dataset, count = row["dataset"], row["n_test"]
        if type(dataset) is not str or not dataset or type(count) is not int or count < 1:
            raise ValueError("Exp291 row identity is invalid")
        n_test[dataset] = count
        lanes[dataset] = _coordinates(row["automatic_hard_indices"], dataset, count)
        normal[dataset] = _coordinates(row["automatic_normal_exact_indices"], dataset, count)
        standard[dataset] = _coordinates(row["standard_review_indices"], dataset, count)
        priority[dataset] = _coordinates(row["priority_review_indices"], dataset, count)
    counts = (sum(map(len, lanes.values())), sum(map(len, normal.values())), sum(map(len, standard.values())), sum(map(len, priority.values())))
    if counts != (2218, 35, 806, 31):
        raise ValueError("Exp291 lane counts are invalid")
    hard = frozenset((dataset, index) for dataset, values in lanes.items() for index in values)
    return MappingProxyType({"automatic_hard_indices": MappingProxyType(lanes), "automatic_normal_exact_indices": MappingProxyType(normal), "standard_review_indices": MappingProxyType(standard), "priority_review_indices": MappingProxyType(priority)}), MappingProxyType(n_test), hard


def _hard_identity(coords: Iterable[tuple[str, int]]) -> str:
    values = tuple(coords)
    if len(values) != len(set(values)):
        raise ValueError("Hard coordinates are duplicated")
    return sha256(json.dumps([[dataset, index] for dataset, index in sorted(values)], separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _parse_exp404(rows: Sequence[Mapping[str, object]], manifest: Mapping[str, object], n_test: Mapping[str, int]) -> frozenset[tuple[str, int]]:
    if len(rows) != 2218 or manifest.get("rows") != 2218 or manifest.get("datasets") != 888 or manifest.get("contains_test_labels") is not False or manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("Exp404 manifest/population is invalid")
    coords: set[tuple[str, int]] = set()
    for row in rows:
        if set(row) != {"abstention_reason", "candidate_index", "confidence_by_alpha", "contains_evaluation_metrics", "contains_test_labels", "dataset", "eligibility", "scores_by_alpha"} or row["contains_test_labels"] is not False or row["contains_evaluation_metrics"] is not False:
            raise ValueError("Exp404 row schema/flags are invalid")
        dataset = row["dataset"]
        if dataset not in n_test:
            raise ValueError("Exp404 dataset is outside Exp291 identity")
        coordinate = validate_coordinate_antigravity(dataset, row["candidate_index"], n_test[dataset])
        if coordinate in coords:
            raise ValueError("Exp404 coordinate is duplicated")
        coords.add(coordinate)
    lanes = manifest.get("frozen_hard_lanes")
    if not isinstance(lanes, Mapping) or len(lanes) != 1117:
        raise ValueError("Exp404 frozen lane map is invalid")
    manifest_coords: set[tuple[str, int]] = set()
    for dataset, values in lanes.items():
        if dataset not in n_test or not isinstance(values, list):
            raise ValueError("Exp404 frozen lane identity is invalid")
        ordered = _coordinates(values, dataset, n_test[dataset])
        manifest_coords.update((dataset, index) for index in ordered)
    if manifest_coords != coords or _hard_identity(coords) != sha256(json.dumps(sorted([[dataset, index] for dataset, index in coords]), separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest():
        raise ValueError("Exp404 manifest coordinates disagree with rows")
    return frozenset(coords)


def _parse_exp414(path: Path, expected_sha256: str) -> Mapping[tuple[str, int], Mapping[str, object]]:
    payload = _read_expected(path, expected_sha256)
    try:
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("Exp414 audit CSV is invalid") from error
    if not rows or any(set(row) != _EXP414_FIELDS for row in rows):
        raise ValueError("Exp414 audit schema is invalid")
    result: dict[tuple[str, int], Mapping[str, object]] = {}
    affected_datasets: set[str] = set()
    affected_rows = 0
    scalars = 0
    all_missing = 0
    for row in rows:
        dataset, index = row["dataset"], int(row["instance_index"])
        if not dataset or index < 0 or (dataset, index) in result:
            raise ValueError("Exp414 audit identity is duplicated or invalid")
        length, nan, posinf, neginf = (int(row[key]) for key in ("length", "nan_count", "posinf_count", "neginf_count"))
        if length < 1 or min(nan, posinf, neginf) < 0 or nan + posinf + neginf > length:
            raise ValueError("Exp414 audit counts are invalid")
        for key in ("raw_values_sha256", "missing_positions_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("Exp414 audit hash is invalid")
        missing = nan + posinf + neginf
        if missing:
            affected_datasets.add(dataset)
            affected_rows += 1
            scalars += missing
            if missing == length:
                all_missing += 1
        result[(dataset, index)] = MappingProxyType({"dataset": dataset, "instance_index": index, "length": length, "nan_count": nan, "posinf_count": posinf, "neginf_count": neginf, "raw_values_sha256": row["raw_values_sha256"], "missing_positions_sha256": row["missing_positions_sha256"]})
    if (len(affected_datasets), affected_rows, scalars, all_missing) != (19, 164, 983, 0):
        raise ValueError("Exp414 missing inventory totals are invalid")
    return MappingProxyType(result)


_CACHE_ARRAY_NAMES_ANTIGRAVITY = frozenset({
    "kernels_1024__train_by_seed", "kernels_1024__test_by_seed",
    "kernels_4096__train_by_seed", "kernels_4096__test_by_seed",
})


def _validate_coordinate_tuple_antigravity(value: object, dataset: str, n_test: int) -> tuple[int, ...]:
    if type(value) is not tuple or value != tuple(sorted(set(value))) or any(type(index) is not int or index < 0 or index >= n_test for index in value):
        raise ValueError("frozen coordinate lane is invalid")
    return value


@dataclass(frozen=True)
class FrozenSourceSnapshotAntigravity:
    datasets: tuple[str, ...]
    affected_datasets: tuple[str, ...]
    affected_families: tuple[str, ...]
    affected_rows: int
    nonfinite_scalars: int
    all_missing_rows: int
    automatic_hard_count: int
    exact_normal_count: int
    standard_count: int
    priority_count: int
    hard_coordinates_exp291: frozenset[tuple[str, int]]
    hard_coordinates_exp404: frozenset[tuple[str, int]]
    routing_algebra: tuple[int, int, int, int]
    exp236_lanes: Mapping[str, Mapping[str, tuple[int, ...]]]
    exp249_candidate_maps: Mapping[str, Mapping[int, tuple[int, ...]]]
    cache_inventory: Mapping[str, Mapping[str, object]]
    test_row_count_by_dataset: Mapping[str, int]
    target_length_by_dataset: Mapping[str, int]
    source_hashes: Mapping[str, str]
    affected_datasets_sha256: str
    label_access_count: int = 0

    def __post_init__(self) -> None:
        datasets = tuple(self.datasets)
        if type(self.datasets) is not tuple or any(type(dataset) is not str or not dataset for dataset in datasets) or datasets != tuple(sorted(set(datasets))) or len(datasets) != 1117:
            raise ValueError("source dataset population is invalid")
        affected_datasets = tuple(self.affected_datasets)
        if type(self.affected_datasets) is not tuple or any(type(dataset) is not str for dataset in affected_datasets) or affected_datasets != tuple(sorted(set(affected_datasets))):
            raise ValueError("source affected population is invalid")
        if len(affected_datasets) != 19 or not set(affected_datasets).issubset(datasets):
            raise ValueError("source affected population is invalid")
        affected_families = tuple(self.affected_families)
        if affected_families != tuple(sorted(set(affected_families))) or len(affected_families) != 4 or set(affected_families) != {parse_dataset_family_antigravity(dataset) for dataset in affected_datasets}:
            raise ValueError("source affected families are invalid")
        scalar_fields = ("affected_rows", "nonfinite_scalars", "all_missing_rows", "automatic_hard_count", "exact_normal_count", "standard_count", "priority_count", "label_access_count")
        if any(type(getattr(self, field)) is not int or getattr(self, field) < 0 for field in scalar_fields):
            raise ValueError("source scalar evidence types are invalid")
        if self.routing_algebra != (2116, 137, 35, 2218) or (self.automatic_hard_count, self.exact_normal_count, self.standard_count, self.priority_count) != (2218, 35, 806, 31):
            raise ValueError("source routing counts are invalid")
        if type(self.routing_algebra) is not tuple or any(type(value) is not int for value in self.routing_algebra):
            raise ValueError("source routing algebra types are invalid")
        if type(self.hard_coordinates_exp291) is not frozenset or type(self.hard_coordinates_exp404) is not frozenset or any(type(dataset) is not str or type(index) is not int or index < 0 for dataset, index in self.hard_coordinates_exp291 | self.hard_coordinates_exp404):
            raise ValueError("source Hard coordinate types are invalid")
        if self.hard_coordinates_exp291 != self.hard_coordinates_exp404 or len(self.hard_coordinates_exp291) != 2218:
            raise ValueError("Exp291/Exp404 Hard coordinates disagree")
        if self.label_access_count != 0:
            raise ValueError("source verification accessed labels")
        if not isinstance(self.test_row_count_by_dataset, Mapping) or set(self.test_row_count_by_dataset) != set(datasets) or any(type(value) is not int or value < 1 for value in self.test_row_count_by_dataset.values()):
            raise ValueError("source TEST identity authority is invalid")
        if not isinstance(self.target_length_by_dataset, Mapping) or set(self.target_length_by_dataset) != set(datasets) or any(type(value) is not int or value < 1 for value in self.target_length_by_dataset.values()):
            raise ValueError("source target-length authority is invalid")
        if not isinstance(self.source_hashes, Mapping) or any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in self.source_hashes.values()):
            raise ValueError("source hashes are invalid")
        expected_hash_names = {"exp249_prediction", "exp249_prediction_manifest", "exp249_artifact_manifest", "exp249_cache_manifest", "exp249_audit", "exp291_routing", "exp291_manifest", "exp414_repair_audit", "exp414_manifest", "exp404_hard", "exp404_manifest", "database"}
        if set(self.source_hashes) != expected_hash_names or type(self.affected_datasets_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", self.affected_datasets_sha256) is None:
            raise ValueError("source hash authority schema is invalid")

        n_test = {str(dataset): int(value) for dataset, value in self.test_row_count_by_dataset.items()}
        lane_copy: dict[str, Mapping[str, tuple[int, ...]]] = {}
        if not isinstance(self.exp236_lanes, Mapping) or set(self.exp236_lanes) != set(datasets):
            raise ValueError("Exp236 lane authority is incomplete")
        for dataset in datasets:
            lanes = self.exp236_lanes[dataset]
            if not isinstance(lanes, Mapping) or set(lanes) != {"hard", "standard", "priority", "promotion"}:
                raise ValueError("Exp236 lane schema is invalid")
            lane_copy[dataset] = MappingProxyType({name: _validate_coordinate_tuple_antigravity(lanes[name], dataset, n_test[dataset]) for name in ("hard", "standard", "priority", "promotion")})

        candidate_copy: dict[str, Mapping[int, tuple[int, ...]]] = {}
        if not isinstance(self.exp249_candidate_maps, Mapping) or set(self.exp249_candidate_maps) != set(datasets):
            raise ValueError("Exp249 candidate authority is incomplete")
        for dataset in datasets:
            candidates = self.exp249_candidate_maps[dataset]
            if not isinstance(candidates, Mapping) or set(candidates) != {1024, 4096}:
                raise ValueError("Exp249 candidate schema is invalid")
            candidate_copy[dataset] = MappingProxyType({kernel: _validate_coordinate_tuple_antigravity(candidates[kernel], dataset, n_test[dataset]) for kernel in (1024, 4096)})

        for coordinate in self.hard_coordinates_exp291 | self.hard_coordinates_exp404:
            dataset, index = coordinate
            if dataset not in n_test or index >= n_test[dataset]:
                raise ValueError("Hard coordinate is outside TEST identity authority")

        cache_copy: dict[str, Mapping[str, object]] = {}
        if not isinstance(self.cache_inventory, Mapping) or set(self.cache_inventory) != set(datasets):
            raise ValueError("cache authority population is incomplete")
        for dataset in datasets:
            entry = self.cache_inventory[dataset]
            if not isinstance(entry, Mapping):
                raise ValueError("cache authority entry is not a mapping")
            required = {"cache_file", "metadata_file", "cache_sha256", "metadata_sha256", "config_hash", "n_train", "n_test", "target_length", "abstained", "arrays"}
            if set(entry) != required or not isinstance(entry["cache_file"], Path) or not isinstance(entry["metadata_file"], Path):
                raise ValueError("cache authority entry schema is invalid")
            if any(type(entry[field]) is not str or re.fullmatch(r"[0-9a-f]{64}", entry[field]) is None for field in ("cache_sha256", "metadata_sha256", "config_hash")):
                raise ValueError("cache authority hashes are invalid")
            if type(entry["abstained"]) is not bool or any(type(entry[field]) is not int or entry[field] < 1 for field in ("n_train", "n_test", "target_length")):
                raise ValueError("cache authority dimensions are invalid")
            if entry["n_test"] != n_test[dataset] or entry["target_length"] != self.target_length_by_dataset[dataset]:
                raise ValueError("cache authority disagrees with TEST/target identity")
            arrays = entry["arrays"]
            if not isinstance(arrays, Mapping) or set(arrays) != _CACHE_ARRAY_NAMES_ANTIGRAVITY:
                raise ValueError("cache authority arrays are incomplete")
            expected_train = 0 if entry["abstained"] else entry["n_train"]
            frozen_arrays: dict[str, np.ndarray] = {}
            for name, expected_columns in (
                ("kernels_1024__train_by_seed", expected_train),
                ("kernels_1024__test_by_seed", entry["n_test"] if not entry["abstained"] else 0),
                ("kernels_4096__train_by_seed", expected_train),
                ("kernels_4096__test_by_seed", entry["n_test"] if not entry["abstained"] else 0),
            ):
                array = arrays[name]
                if type(array) is not np.ndarray or array.dtype != np.dtype("<f8") or array.ndim != 2 or array.shape != (2, expected_columns) or not np.isfinite(array).all():
                    raise ValueError("cache authority array type, shape, or finite contract is invalid")
                frozen_arrays[name] = _freeze_array_antigravity(array)
            frozen_entry = dict(entry)
            frozen_entry["arrays"] = MappingProxyType(frozen_arrays)
            cache_copy[dataset] = MappingProxyType(frozen_entry)

        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "affected_datasets", affected_datasets)
        object.__setattr__(self, "affected_families", affected_families)
        object.__setattr__(self, "exp236_lanes", MappingProxyType(lane_copy))
        object.__setattr__(self, "exp249_candidate_maps", MappingProxyType(candidate_copy))
        object.__setattr__(self, "cache_inventory", MappingProxyType(cache_copy))
        object.__setattr__(self, "test_row_count_by_dataset", MappingProxyType(n_test))
        object.__setattr__(self, "target_length_by_dataset", MappingProxyType({str(k): int(v) for k, v in self.target_length_by_dataset.items()}))
        object.__setattr__(self, "source_hashes", MappingProxyType({str(k): str(v) for k, v in self.source_hashes.items()}))

    @property
    def context_datasets(self) -> tuple[str, ...]:
        return self.datasets

    @property
    def affected_runs(self) -> int:
        return self.affected_rows

    @property
    def nonfinite_values(self) -> int:
        return self.nonfinite_scalars

    @property
    def all_missing_runs(self) -> int:
        return self.all_missing_rows

    @property
    def hard_coordinates(self) -> frozenset[tuple[str, int]]:
        return self.hard_coordinates_exp291

    @property
    def cache_manifest(self) -> Mapping[str, Mapping[str, object]]:
        return self.cache_inventory


def _compare_inventory(inventory: MissingInventoryAntigravity, exp414_rows: Mapping[tuple[str, int], Mapping[str, object]]) -> None:
    if (len(inventory.affected_datasets), inventory.affected_rows, inventory.nonfinite_scalars, inventory.all_missing_rows) != (19, 164, 983, 0):
        raise ValueError("database missing inventory totals disagree with Exp414")
    for key, positions in inventory.missing_positions_by_row.items():
        dataset, text_index = key.rsplit("/", 1)
        index = int(text_index)
        if not positions:
            continue
        audit = exp414_rows.get((dataset, index))
        if audit is None or audit["raw_values_sha256"] != inventory.raw_blob_sha256_by_row[key] or int(audit["length"]) <= max(positions):
            raise ValueError("Exp414 raw audit does not match database blob")
        if _sha256_canonical(list(positions)) != audit["missing_positions_sha256"]:
            raise ValueError("Exp414 missing-position audit does not match database mask")


def verify_frozen_sources_antigravity() -> FrozenSourceSnapshotAntigravity:
    """Verify every frozen source and the real 1,117-dataset raw inventory."""
    cfg = FROZEN_CONFIG
    source_paths = {
        "exp249_prediction": Path(cfg.exp249_prediction_path), "exp249_prediction_manifest": Path(cfg.exp249_prediction_manifest_path),
        "exp249_artifact_manifest": Path(cfg.exp249_artifact_manifest_path), "exp249_cache_manifest": Path(cfg.exp249_cache_manifest_path),
        "exp249_audit": Path(cfg.exp249_audit_path), "exp291_routing": Path(cfg.exp291_routing_path),
        "exp291_manifest": Path(cfg.exp291_manifest_path), "exp414_repair_audit": Path(cfg.exp414_repair_audit_path),
        "exp414_manifest": Path(cfg.exp414_manifest_path), "exp404_hard": Path(cfg.exp404_hard_path), "exp404_manifest": Path(cfg.exp404_manifest_path),
    }
    hashes = {name: getattr(cfg, f"{name}_sha256") for name in source_paths}
    exp249_rows = _parse_jsonl(source_paths["exp249_prediction"], hashes["exp249_prediction"], "Exp249 prediction")
    exp249_manifest = parse_json_object_antigravity(_read_expected(source_paths["exp249_prediction_manifest"], hashes["exp249_prediction_manifest"]), "Exp249 prediction manifest")
    datasets, candidate_maps, exp236_lanes = _validate_exp249_prediction(exp249_rows, exp249_manifest)
    artifact_manifest = parse_json_object_antigravity(_read_expected(source_paths["exp249_artifact_manifest"], hashes["exp249_artifact_manifest"]), "Exp249 artifact manifest")
    if artifact_manifest.get("experiment") != 249 or artifact_manifest.get("scope") != "retrospective_score_source_qualification" or artifact_manifest.get("autonomous_policy_changed") is not False:
        raise ValueError("Exp249 artifact manifest is invalid")
    exp249_audit = parse_json_object_antigravity(_read_expected(source_paths["exp249_audit"], hashes["exp249_audit"]), "Exp249 audit")
    if exp249_audit.get("datasets") != 1117 or exp249_audit.get("contains_test_labels") is not False or exp249_audit.get("database_sha256") != cfg.database_sha256 or exp249_audit.get("aeon_version") != "1.5.0" or exp249_audit.get("numpy_version") != "2.3.5" or exp249_audit.get("scikit_learn_version") != "1.8.0":
        raise ValueError("Exp249 audit is invalid")
    cache_inventory = parse_cache_manifest_antigravity(source_paths["exp249_cache_manifest"], source_paths["exp249_cache_manifest"].parent, expected_sha256=hashes["exp249_cache_manifest"])
    exp291_rows = _parse_jsonl(source_paths["exp291_routing"], hashes["exp291_routing"], "Exp291 routing")
    exp291_manifest = parse_json_object_antigravity(_read_expected(source_paths["exp291_manifest"], hashes["exp291_manifest"]), "Exp291 manifest")
    lanes, n_test, hard291 = _parse_exp291(exp291_rows, exp291_manifest)
    exp404_rows = _parse_jsonl(source_paths["exp404_hard"], hashes["exp404_hard"], "Exp404 Hard")
    exp404_manifest = parse_json_object_antigravity(_read_expected(source_paths["exp404_manifest"], hashes["exp404_manifest"]), "Exp404 manifest")
    hard404 = _parse_exp404(exp404_rows, exp404_manifest, n_test)
    if hard291 != hard404:
        raise ValueError("Exp291/Exp404 Hard coordinate mismatch")
    exp414_rows = _parse_exp414(source_paths["exp414_repair_audit"], hashes["exp414_repair_audit"])
    exp414_manifest = parse_json_object_antigravity(_read_expected(source_paths["exp414_manifest"], hashes["exp414_manifest"]), "Exp414 manifest")
    if exp414_manifest.get("experiment_id") != "Exp414" or not isinstance(exp414_manifest.get("artifact_sha256"), Mapping):
        raise ValueError("Exp414 manifest is invalid")
    target_lengths = {dataset: int(cache_inventory[dataset]["target_length"]) for dataset in datasets}
    with __import__("tempfile").TemporaryDirectory(prefix="exp443_sources_antigravity_") as temporary:
        pinned = pin_database_snapshot_antigravity(cfg.database_path, Path(temporary) / "attempt_antigravity")
        try:
            inventory = scan_missing_inventory_antigravity(
                pinned,
                datasets,
                expected_test_rows_by_dataset=n_test,
                expected_target_length_by_dataset=target_lengths,
            )
            _compare_inventory(inventory, exp414_rows)
        finally:
            pinned.close(terminal_evidence_durable=True)
    if inventory.affected_datasets_sha256 != "ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9":
        raise ValueError("affected dataset identity SHA mismatch")
    if set(datasets) != set(n_test) or set(datasets) != set(cache_inventory):
        raise ValueError("inherited source populations disagree")
    # Exp249's embedded Exp236 lanes are immutable and independent of the two
    # MiniROCKET candidate maps.  The routing proof is set algebra only.
    hard236 = {(dataset, index) for dataset, lanes_for_dataset in exp236_lanes.items() for index in lanes_for_dataset["hard"]}
    promotions = {(dataset, index) for dataset, values in exp236_lanes.items() for index in values["promotion"]}
    exact_normal = {(dataset, index) for dataset, values in lanes["automatic_normal_exact_indices"].items() for index in values}
    if len(hard236) != 2116 or len(promotions) != 137 or len(exact_normal) != 35 or len((hard236 | promotions) - exact_normal) != 2218:
        raise ValueError("frozen routing algebra is invalid")
    if (hard236 | promotions) - exact_normal != hard291:
        raise ValueError("frozen routing coordinates disagree")
    source_hashes = {name: hashes[name] for name in source_paths}
    source_hashes["database"] = cfg.database_sha256
    return FrozenSourceSnapshotAntigravity(
        datasets=datasets,
        affected_datasets=inventory.affected_datasets,
        affected_families=inventory.affected_families,
        affected_rows=inventory.affected_rows,
        nonfinite_scalars=inventory.nonfinite_scalars,
        all_missing_rows=inventory.all_missing_rows,
        automatic_hard_count=len(hard291), exact_normal_count=len(exact_normal),
        standard_count=sum(map(len, lanes["standard_review_indices"].values())),
        priority_count=sum(map(len, lanes["priority_review_indices"].values())),
        hard_coordinates_exp291=hard291, hard_coordinates_exp404=hard404,
        routing_algebra=(len(hard236), len(promotions), len(exact_normal), len(hard291)),
        exp236_lanes=exp236_lanes, exp249_candidate_maps=candidate_maps,
        cache_inventory=cache_inventory, test_row_count_by_dataset=n_test,
        target_length_by_dataset=target_lengths, source_hashes=source_hashes,
        affected_datasets_sha256=inventory.affected_datasets_sha256,
        label_access_count=inventory.label_access_count,
    )


__all__ = [
    "DuplicateJsonKeyError", "FrozenSourceSnapshotAntigravity", "parse_json_object_antigravity",
    "parse_cache_manifest_antigravity", "load_pinned_npz_antigravity", "validate_coordinate_antigravity",
    "verify_frozen_sources_antigravity",
]
