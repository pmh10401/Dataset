"""Qualification-only orchestration for label-free policy comparison."""
from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
from html import escape
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import roc_auc_score

from .anomaly_detection import (
    ALPHA, FIXED_SEEDS, empirical_upper_pvalues, run_analysis_policy, run_method_detection,
)
from .dataset_repository import (
    PREPARATION_VALUE_SQL, PreparedDatasetValues,
    open_source_database, prepare_dataset_values, resample_series,
)
from .routing_policy import FEATURE_KEYS, MethodMetric, RouterTrainingRecord, extract_train_features, fit_train_feature_router
from .score_sources import METHOD_NAMES, ScoreResult, score_method

SERVICE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SERVICE_ROOT.parents[1]
RUNTIME_ROOT = SERVICE_ROOT / "runtime" / "live_policy_qualification"
EVIDENCE_ROOT = SERVICE_ROOT / "evidence"
QUALIFICATION_DASHBOARD = SERVICE_ROOT / "docs" / "live_analysis_qualification_dashboard.html"
OFFICIAL_SOURCE = REPOSITORY_ROOT / "experiments/exp481_strict_evidence_residual_correction_antigravity/results_antigravity.json"
OFFICIAL_SOURCE_SHA256 = "7038b994e422dc93a18927c810ce5e3bdb7601039ba01d2b388895444e09956f"
POLICY_ORDER = ("single_minirocket", "three_method_consensus", "train_feature_router")


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _frame(hasher, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big")); hasher.update(value)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _array_record(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    if not np.isfinite(array).all():
        raise ValueError("arrays must be finite")
    return {"shape": list(array.shape), "dtype": array.dtype.str, "bytes": array.tobytes().hex()}


def _valid_test_indices(values: Any) -> bool:
    return (type(values) in (list, tuple)
            and all(type(value) is int and value >= 0 for value in values)
            and len(set(values)) == len(values))


def _array_from_record(record: Mapping[str, Any], *, ndim: int) -> np.ndarray:
    try:
        if set(record) != {"shape", "dtype", "bytes"}:
            raise ValueError("array record schema mismatch")
        dtype = np.dtype(record["dtype"]); shape = tuple(record["shape"])
        values = np.frombuffer(bytes.fromhex(record["bytes"]), dtype=dtype).reshape(shape)
    except Exception as error:
        raise ValueError("malformed array record") from error
    if values.ndim != ndim or not np.isfinite(values).all():
        raise ValueError("array record must be finite and aligned")
    return values.copy()


@dataclass(frozen=True, slots=True)
class DatasetValueStatus:
    name: str
    eligible: bool
    failure: str | None
    raw_value_digest: str
    prepared_digest: str | None
    preparation_mode: str | None
    train_count: int
    test_count: int
    target_length: int
    test_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ValueManifest:
    requested_names: tuple[str, ...]
    statuses: tuple[DatasetValueStatus, ...]
    digest: str

    @property
    def total_requested(self) -> int: return len(self.requested_names)
    @property
    def eligible_names(self) -> tuple[str, ...]: return tuple(row.name for row in self.statuses if row.eligible)
    @property
    def eligible_count(self) -> int: return len(self.eligible_names)

    def as_dict(self) -> dict[str, Any]:
        return {"requested_names": list(self.requested_names), "total_requested": self.total_requested,
                "eligible_names": list(self.eligible_names), "eligible_count": self.eligible_count,
                "statuses": [asdict(row) for row in self.statuses], "digest": self.digest}

    def to_json(self) -> str: return _canonical_bytes(self.as_dict()).decode()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValueManifest":
        expected_keys = {"requested_names", "total_requested", "eligible_names", "eligible_count", "statuses", "digest"}
        if set(payload) != expected_keys or type(payload["statuses"]) is not list:
            raise ValueError("value manifest schema mismatch")
        status_keys = {field.name for field in fields(DatasetValueStatus)}
        if any(type(row) is not dict or set(row) != status_keys for row in payload["statuses"]):
            raise ValueError("value manifest status schema mismatch")
        try:
            statuses = tuple(DatasetValueStatus(**{**row, "test_indices": tuple(row["test_indices"])})
                             for row in payload["statuses"])
        except Exception as error:
            raise ValueError("value manifest status schema mismatch") from error
        result = cls(tuple(payload["requested_names"]), statuses, payload["digest"])
        core = {"requested_names": list(result.requested_names), "statuses": [asdict(row) for row in statuses]}
        names = tuple(row.name for row in statuses)
        if (result.requested_names != tuple(sorted(result.requested_names))
                or len(set(result.requested_names)) != len(result.requested_names)
                or names != result.requested_names
                or result.digest != _digest(core)
                or payload["total_requested"] != result.total_requested
                or payload["eligible_names"] != list(result.eligible_names)
                or payload["eligible_count"] != result.eligible_count
                or any(type(row.eligible) is not bool or type(row.name) is not str
                       or type(row.raw_value_digest) is not str or len(row.raw_value_digest) != 64
                       or any(type(value) is not int or value < 0 for value in
                              (row.train_count, row.test_count, row.target_length))
                       or not _valid_test_indices(row.test_indices)
                       or (row.eligible and (row.failure is not None or type(row.prepared_digest) is not str
                                             or len(row.prepared_digest) != 64
                                             or type(row.preparation_mode) is not str or not row.preparation_mode
                                             or row.train_count < 5 or row.test_count != len(row.test_indices)
                                             or len(set(row.test_indices)) != len(row.test_indices)
                                             or row.target_length < 1))
                       or (not row.eligible and (type(row.failure) is not str or not row.failure
                                                or row.prepared_digest is not None or row.preparation_mode is not None
                                                or any((row.train_count, row.test_count, row.target_length, len(row.test_indices)))))
                       for row in statuses)):
            raise ValueError("value manifest digest mismatch")
        return result


def _raw_value_digest(connection: sqlite3.Connection, name: str) -> str:
    hasher = sha256()
    for split, index, raw_blob in connection.execute(PREPARATION_VALUE_SQL, (name,)):
        if type(raw_blob) is not bytes:
            raw_blob = f"{type(raw_blob).__name__}:{raw_blob!r}".encode()
        for value in (name.encode(), str(split).encode(), str(index).encode(), raw_blob):
            _frame(hasher, value)
    return hasher.hexdigest()


def _prepared_digest(prepared: PreparedDatasetValues) -> str:
    payload = {"name": prepared.dataset_name, "mode": prepared.preparation_mode,
               "train": _array_record(prepared.train_values), "test": _array_record(prepared.test_values),
               "raw_train": [_array_record(row) for row in prepared.raw_train_rows or ()],
               "test_indices": list(prepared.test_indices)}
    return _digest(payload)


def build_value_manifest(connection: sqlite3.Connection, names: Iterable[str]) -> ValueManifest:
    requested = tuple(names)
    if any(type(name) is not str or not name for name in requested) or len(set(requested)) != len(requested):
        raise ValueError("names must be unique non-empty strings")
    requested = tuple(sorted(requested))
    statuses = []
    for name in requested:
        raw_digest = _raw_value_digest(connection, name)
        try:
            prepared = prepare_dataset_values(connection, name)
            if not _valid_test_indices(prepared.test_indices):
                raise ValueError("duplicate TEST indices or non-integer/negative TEST index")
            status = DatasetValueStatus(name, True, None, raw_digest, _prepared_digest(prepared),
                                        prepared.preparation_mode,
                                        len(prepared.train_values), len(prepared.test_values), prepared.target_length,
                                        prepared.test_indices)
        except Exception as error:
            status = DatasetValueStatus(name, False, f"{type(error).__name__}: {error}", raw_digest,
                                        None, None, 0, 0, 0, ())
        statuses.append(status)
    core = {"requested_names": list(requested), "statuses": [asdict(row) for row in statuses]}
    return ValueManifest(requested, tuple(statuses), _digest(core))


def load_official_population(path: Path = OFFICIAL_SOURCE, expected_digest: str = OFFICIAL_SOURCE_SHA256,
                             *, expected_count: int = 1117) -> tuple[tuple[str, ...], dict[str, str]]:
    if sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("official population source digest mismatch")
    try: decisions = json.loads(path.read_text())["evaluation"]["decisions"]
    except Exception as error: raise ValueError("malformed official population source") from error
    names = [row.get("dataset") for row in decisions]
    if len(names) != expected_count or len(set(names)) != expected_count or any(type(name) is not str or not name for name in names):
        raise ValueError("official population must have exact unique coverage")
    families = {row["dataset"]: row.get("held_family") for row in decisions}
    if set(families) != set(names) or any(type(value) is not str or not value for value in families.values()):
        raise ValueError("official family map must have one-to-one coverage")
    return tuple(sorted(names)), {name: families[name] for name in sorted(names)}


def _version(package: str) -> str | None:
    try: return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: return None


def _file_digest(path: Path) -> str: return sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutionContract:
    payload: dict[str, Any]
    canonical_payload: bytes
    sha256: str


def build_execution_contract_record(manifest: ValueManifest, population_source_digest: str,
                                    family_by_dataset: Mapping[str, str], backend_assessment: Mapping[str, Any],
                                    *, workers: int) -> ExecutionContract:
    payload = {"value_manifest_digest": manifest.digest, "population_source_digest": population_source_digest,
               "family_map_digest": _digest(dict(sorted(family_by_dataset.items()))),
               "code_digests": {name: _file_digest(SERVICE_ROOT / name) for name in
                                ("dataset_repository.py", "score_sources.py", "anomaly_detection.py", "routing_policy.py", "policy_qualification.py")},
               "constants": {"alpha": ALPHA, "fixed_seeds": FIXED_SEEDS, "methods": METHOD_NAMES,
                             "score": {"folds": 5, "max_reference": 1000, "chunk": 128,
                                       "kernels": dict(zip(METHOD_NAMES, (4096, 1024, 8), strict=True)),
                                       "neighbors": 3}, "floor_tolerance": 1e-15,
                             "router": {"minimum_groups": 20, "maximum_neighbors": 31,
                                        "non_degradation_rate": 0.9, "auc_quantile": 0.2}},
               "versions": {name: _version(name) for name in
                            ("numpy", "numba", "aeon", "scikit-learn", "mlx", "mlx-metal")},
               "hardware": {"machine": platform.machine(), "processor": platform.processor()},
               "python_version": tuple(sys.version_info[:3]),
               "execution": {"workers": workers, "cpu_count": os.cpu_count(), "backend_identity": {
                   key: value.get("selected") for key, value in backend_assessment.items()
                   if isinstance(value, Mapping) and "selected" in value}},
               "backend_assessment": dict(backend_assessment)}
    canonical = _canonical_bytes(payload)
    payload = json.loads(canonical)
    return ExecutionContract(payload, canonical, sha256(canonical).hexdigest())


def build_execution_contract(manifest: ValueManifest, population_source_digest: str,
                             family_by_dataset: Mapping[str, str], backend_assessment: Mapping[str, Any]) -> str:
    return build_execution_contract_record(manifest, population_source_digest, family_by_dataset,
                                           backend_assessment, workers=1).sha256


def _score_row(prepared: PreparedDatasetValues, scorer) -> dict[str, Any]:
    if prepared.raw_train_rows is None: raise ValueError("authoritative raw TRAIN rows are required")
    scores = {}
    for method in METHOD_NAMES:
        scores[method] = {}
        for seed in FIXED_SEEDS:
            try:
                result = scorer(prepared.train_values, prepared.test_values, method, seed)
            except Exception as error:
                raise ScoreStageError(prepared.dataset_name, method, seed, error) from error
            scores[method][str(seed)] = {"train_oof": _array_record(result.train_oof),
                                         "test_scores": _array_record(result.test_scores),
                                         "feature_dimension": result.feature_dimension, "seed": result.seed}
    row = {"dataset": prepared.dataset_name, "prepared_digest": _prepared_digest(prepared),
           "metadata": {"mode": prepared.preparation_mode, "train_count": len(prepared.train_values),
                        "test_count": len(prepared.test_values), "target_length": prepared.target_length,
                        "test_indices": list(prepared.test_indices)},
           "features": extract_train_features(prepared.raw_train_rows), "scores": scores}
    return {**row, "row_sha256": _digest(row)}


class ScoreStageError(RuntimeError):
    def __init__(self, dataset: str, method: str, seed: int, error: Exception):
        self.dataset = dataset
        self.method = method
        self.seed = seed
        self.detail = f"{type(error).__name__}: {error}"
        super().__init__(f"{dataset}/{method}/{seed}: {self.detail}")


def _failure_row(error: ScoreStageError) -> dict[str, Any]:
    core = {"dataset": error.dataset, "method": error.method, "seed": error.seed,
            "status": "failed", "error": error.detail}
    return {**core, "row_sha256": _digest(core)}


def _write_score_failures(path: Path, contract_sha: str, manifest: ValueManifest,
                          failures: Iterable[Mapping[str, Any]]) -> None:
    rows = sorted((dict(row) for row in failures), key=lambda row: (row["dataset"], row["method"], row["seed"]))
    header = {"schema": "score_failures_v1", "contract_sha256": contract_sha,
              "manifest_digest": manifest.digest, "manifest_requested": manifest.total_requested,
              "manifest_eligible": manifest.eligible_count}
    _atomic_write(path, b"".join([_canonical_bytes(header), *(_canonical_bytes(row) for row in rows)]))


def _load_score_failures(path: Path, contract_sha: str, manifest: ValueManifest) -> list[dict[str, Any]]:
    try:
        documents = [json.loads(line) for line in path.read_text().splitlines()]
    except Exception as error:
        raise ValueError("malformed score failure evidence") from error
    header = {"schema": "score_failures_v1", "contract_sha256": contract_sha,
              "manifest_digest": manifest.digest, "manifest_requested": manifest.total_requested,
              "manifest_eligible": manifest.eligible_count}
    if not documents or documents[0] != header:
        raise ValueError("score failure evidence identity mismatch")
    failures = documents[1:]
    for row in failures:
        core = {key: value for key, value in row.items() if key != "row_sha256"}
        if (set(row) != {"dataset", "method", "seed", "status", "error", "row_sha256"}
                or row["dataset"] not in manifest.eligible_names or row["method"] not in METHOD_NAMES
                or row["seed"] not in FIXED_SEEDS or row["status"] != "failed"
                or row["row_sha256"] != _digest(core)):
            raise ValueError("score failure row mismatch")
    return failures


def _validate_score_row(row: Mapping[str, Any], status: DatasetValueStatus) -> None:
    if set(row) != {"dataset", "prepared_digest", "metadata", "features", "scores", "row_sha256"}:
        raise ValueError("score row schema mismatch")
    if row.get("row_sha256") != _digest({key: value for key, value in row.items() if key != "row_sha256"}):
        raise ValueError("score row digest mismatch")
    if row.get("dataset") != status.name or row.get("prepared_digest") != status.prepared_digest:
        raise ValueError("score row input identity mismatch")
    metadata = row.get("metadata", {})
    if set(metadata) != {"mode", "train_count", "test_count", "target_length", "test_indices"}:
        raise ValueError("score metadata schema mismatch")
    if (metadata.get("train_count"), metadata.get("test_count"), metadata.get("target_length")) != (status.train_count, status.test_count, status.target_length):
        raise ValueError("score row metadata mismatch")
    indices = metadata.get("test_indices")
    if (metadata.get("mode") != status.preparation_mode or type(indices) is not list
            or not _valid_test_indices(indices) or tuple(indices) != status.test_indices):
        raise ValueError("score row preparation metadata mismatch")
    if set(row.get("features", {})) != set(FEATURE_KEYS) or not np.isfinite(list(row["features"].values())).all():
        raise ValueError("score row features are incomplete")
    if set(row.get("scores", {})) != set(METHOD_NAMES): raise ValueError("score method coverage mismatch")
    for method in METHOD_NAMES:
        if set(row["scores"][method]) != {str(seed) for seed in FIXED_SEEDS}: raise ValueError("score seed coverage mismatch")
        for seed in FIXED_SEEDS:
            record = row["scores"][method][str(seed)]
            if set(record) != {"train_oof", "test_scores", "feature_dimension", "seed"}:
                raise ValueError("score record schema mismatch")
            train = _array_from_record(record["train_oof"], ndim=1); test = _array_from_record(record["test_scores"], ndim=1)
            if len(train) != status.train_count or len(test) != status.test_count or record.get("seed") != seed or type(record.get("feature_dimension")) is not int or record["feature_dimension"] < 1:
                raise ValueError("score arrays, seed, or dimension mismatch")


def _read_score_checkpoint(path: Path, contract_sha: str, manifest: ValueManifest, *, require_complete: bool) -> dict[str, dict[str, Any]]:
    try: lines = path.read_text().splitlines(); header = json.loads(lines[0]); rows = [json.loads(line) for line in lines[1:]]
    except Exception as error: raise ValueError("malformed score checkpoint") from error
    if header != {"schema": "score_checkpoint_v1", "contract_sha256": contract_sha}: raise ValueError("checkpoint contract mismatch")
    expected = {row.name: row for row in manifest.statuses if row.eligible}
    names = [row.get("dataset") for row in rows]
    if len(names) != len(set(names)) or not set(names) <= set(expected) or (require_complete and set(names) != set(expected)):
        raise ValueError("checkpoint dataset coverage mismatch")
    result = {}
    for row in rows:
        _validate_score_row(row, expected[row["dataset"]])
        row["features"] = {key: row["features"][key] for key in FEATURE_KEYS}
        result[row["dataset"]] = row
    return result


def load_score_checkpoint(path: Path, contract_sha: str, manifest: ValueManifest) -> dict[str, dict[str, Any]]:
    return _read_score_checkpoint(path, contract_sha, manifest, require_complete=True)


def _write_checkpoint(path: Path, contract_sha: str, rows: Mapping[str, Mapping[str, Any]]) -> None:
    values = [_canonical_bytes({"schema": "score_checkpoint_v1", "contract_sha256": contract_sha})]
    values.extend(_canonical_bytes(rows[name]) for name in sorted(rows))
    _atomic_write(path, b"".join(values))


def score_with_checkpoint(connection: sqlite3.Connection, manifest: ValueManifest, contract_sha: str,
                          path: Path, *, scorer=score_method, resume: bool = False,
                          failure_path: Path | None = None) -> dict[str, dict[str, Any]]:
    if resume and build_value_manifest(connection, manifest.requested_names).digest != manifest.digest:
        raise ValueError("value manifest changed before score resume")
    rows = _read_score_checkpoint(path, contract_sha, manifest, require_complete=False) if resume and path.exists() else {}
    failures = []
    for name in manifest.eligible_names:
        if name in rows: continue
        try:
            rows[name] = _score_row(prepare_dataset_values(connection, name), scorer)
            _write_checkpoint(path, contract_sha, rows)
        except ScoreStageError as error:
            failures.append(_failure_row(error))
            if failure_path is not None:
                _write_score_failures(failure_path, contract_sha, manifest, failures)
    if failures:
        raise RuntimeError(f"{len(failures)} score failures were durably recorded")
    return load_score_checkpoint(path, contract_sha, manifest)


def _cached_scorer(row: Mapping[str, Any]):
    def scorer(_train, _test, method, seed):
        record = row["scores"][method][str(seed)]
        return ScoreResult(_array_from_record(record["train_oof"], ndim=1), _array_from_record(record["test_scores"], ndim=1), record["feature_dimension"], seed)
    return scorer


def _prediction(prepared: PreparedDatasetValues, result) -> dict[str, Any]:
    by_index = {row["series_index"]: row for row in result.rows}
    ordered = [by_index[index] for index in prepared.test_indices]
    return {"policy_id": result.policy_id or "current_baseline", "dataset": prepared.dataset_name,
            "test_indices": list(prepared.test_indices), "evaluation_scores": [row["score"] for row in ordered],
            "p_values": [row["p_value"] for row in ordered], "automatic": [row["automatic_anomaly"] for row in ordered],
            "review": [index in result.review_indices for index in prepared.test_indices],
            "selected_method": result.selected_method,
            "method_p_values": [row.get("method_p_values", {}) for row in ordered],
            "method_seed_p_values": [row.get("method_seed_p_values", {}) for row in ordered]}


def _validate_predictions(predictions: Iterable[Mapping[str, Any]], manifest: ValueManifest) -> list[dict[str, Any]]:
    rows = [dict(row) for row in predictions]; expected = set(manifest.eligible_names)
    row_keys = {"policy_id", "dataset", "test_indices", "evaluation_scores", "p_values",
                "automatic", "review", "selected_method", "method_p_values", "method_seed_p_values"}
    forbidden = {"label", "labels", "test_labels", "metrics"}
    if any(set(row) != row_keys or set(row) & forbidden for row in rows):
        raise ValueError("frozen prediction schema mismatch")
    if len(rows) != len(expected) or {row.get("dataset") for row in rows} != expected: raise ValueError("prediction dataset coverage mismatch")
    if len({row["dataset"] for row in rows}) != len(rows): raise ValueError("duplicate prediction dataset")
    if len({row["policy_id"] for row in rows}) != 1 or next(iter(rows))["policy_id"] not in ("current_baseline", *POLICY_ORDER):
        raise ValueError("predictions require one known policy")
    status = {row.name: row for row in manifest.statuses}
    for row in rows:
        indices = row.get("test_indices", []); count = status[row["dataset"]].test_count
        arrays = [row.get(key, []) for key in ("evaluation_scores", "p_values", "automatic", "review", "method_p_values", "method_seed_p_values")]
        if (type(indices) is not list or not _valid_test_indices(indices)
                or tuple(indices) != status[row["dataset"]].test_indices
                or any(len(values) != count for values in arrays)):
            raise ValueError("prediction rows are incomplete or misaligned")
        if any(type(value) not in (int, float) for key in ("evaluation_scores", "p_values") for value in row[key]):
            raise ValueError("prediction values must be numeric")
        if not np.isfinite(row["evaluation_scores"]).all() or not np.isfinite(row["p_values"]).all(): raise ValueError("prediction values must be finite")
        if any(not 0.0 <= value <= 1.0 for value in row["p_values"]): raise ValueError("p-values must be in [0, 1]")
        if not np.allclose(np.asarray(row["evaluation_scores"]), 1.0 - np.asarray(row["p_values"]), atol=1e-15, rtol=0):
            raise ValueError("evaluation scores must replay from p-values")
        if any(type(value) is not bool for key in ("automatic", "review") for value in row[key]): raise ValueError("prediction flags must be booleans")
        if any(a and b for a, b in zip(row["automatic"], row["review"], strict=True)): raise ValueError("automatic and review flags must be disjoint")
        selected = row["selected_method"]
        methods = set(METHOD_NAMES) if row["policy_id"] == "three_method_consensus" else {selected}
        if (row["policy_id"] == "three_method_consensus" and selected is not None) or (row["policy_id"] != "three_method_consensus" and selected not in METHOD_NAMES):
            raise ValueError("selected method is invalid")
        for final, seed_values in zip(row["method_p_values"], row["method_seed_p_values"], strict=True):
            if set(final) != methods or set(seed_values) != methods:
                raise ValueError("method evidence coverage mismatch")
            values = [*final.values(), *(value for pairs in seed_values.values() for value in pairs)]
            if (any(len(seed_values[method]) != len(FIXED_SEEDS) for method in methods)
                    or any(type(value) not in (int, float) for value in values)
                    or not np.isfinite(values).all() or any(not 0 <= value <= 1 for value in values)):
                raise ValueError("method evidence must be finite complete p-values")
        floor = 1.0 / (status[row["dataset"]].train_count + 1.0)
        review_enabled = ALPHA + 1e-15 < floor
        for position, (p_value, automatic, review) in enumerate(
                zip(row["p_values"], row["automatic"], row["review"], strict=True)):
            final = row["method_p_values"][position]
            seeds = row["method_seed_p_values"][position]
            expected_p = sorted(final.values())[1] if row["policy_id"] == "three_method_consensus" else final[selected]
            expected_automatic = (sum(value <= ALPHA for value in final.values()) >= 2
                                  if row["policy_id"] == "three_method_consensus" else expected_p <= ALPHA)
            if row["policy_id"] == "current_baseline":
                expected_review = review_enabled and abs(expected_p - floor) <= 1e-15
            else:
                seed_support = any(all(abs(value - floor) <= 1e-15 for value in seeds[method])
                                   for method in methods)
                method_support = (row["policy_id"] == "three_method_consensus"
                                  and sum(abs(value - floor) <= 1e-15 for value in final.values()) >= 2)
                expected_review = review_enabled and (seed_support or method_support)
            expected_review = bool(expected_review and not expected_automatic)
            if (abs(p_value - expected_p) > 1e-15 or automatic != expected_automatic
                    or review != expected_review):
                raise ValueError("prediction decision evidence mismatch")
    rows.sort(key=lambda row: row["dataset"])
    return rows


def freeze_predictions(predictions: Iterable[Mapping[str, Any]], manifest: ValueManifest, path: Path, *,
                       contract_sha: str, binding_sha: str | None = None) -> dict[str, Any]:
    rows = _validate_predictions(predictions, manifest); digest = _digest(rows)
    payload = {"schema": "frozen_predictions_v1", "contract_sha256": contract_sha,
               "binding_sha256": binding_sha, "dataset_count": len(rows),
               "row_count": sum(len(row["test_indices"]) for row in rows), "prediction_sha256": digest, "predictions": rows}
    _atomic_write(path, _canonical_bytes(payload)); return payload


def load_frozen_predictions(path: Path, manifest: ValueManifest, *, contract_sha: str,
                            binding_sha: str | None = None) -> dict[str, Any]:
    try: payload = json.loads(path.read_text()); rows = _validate_predictions(payload["predictions"], manifest)
    except Exception as error: raise ValueError("malformed frozen predictions") from error
    expected_keys = {"schema", "contract_sha256", "binding_sha256", "dataset_count", "row_count",
                     "prediction_sha256", "predictions"}
    if (set(payload) != expected_keys or payload.get("schema") != "frozen_predictions_v1"
            or payload.get("contract_sha256") != contract_sha or payload.get("binding_sha256") != binding_sha
            or payload.get("dataset_count") != len(rows)
            or payload.get("row_count") != sum(len(row["test_indices"]) for row in rows)
            or payload.get("prediction_sha256") != _digest(rows)):
        raise ValueError("frozen prediction digest or counts mismatch")
    return payload


@dataclass(frozen=True, slots=True)
class PolicyMetrics:
    policy_id: str; complete: bool; failure: str | None; prediction_digest: str
    dataset_count: int; row_count: int; automatic_count: int; residual_fn_total: int
    residual_fn_rate: float; family_balanced_residual_fn_rate: float
    review_inclusive_fp_total: int; automatic_fp_total: int
    dataset_mean_auc: float; family_balanced_auc: float; elapsed_seconds: float

def evaluate_frozen_predictions(predictions: Iterable[Mapping[str, Any]], labels_by_dataset: Mapping[str, Mapping[int, int]],
                                family_by_dataset: Mapping[str, str], elapsed_seconds: float = 0.0) -> PolicyMetrics:
    rows = [dict(row) for row in predictions]; policy = rows[0].get("policy_id", "unknown") if rows else "unknown"
    try:
        if not rows or len({row["dataset"] for row in rows}) != len(rows): raise ValueError("unique prediction rows are required")
        if {row["dataset"] for row in rows} != set(labels_by_dataset) or set(labels_by_dataset) != set(family_by_dataset): raise ValueError("evaluation coverage mismatch")
        total_anomalies = residual = review_fp = automatic_fp = automatic_count = row_count = 0
        aucs = []; fn_rates = []; families: dict[str, list[int]] = {}
        for row in rows:
            indices = row["test_indices"]
            if len(indices) != len(row["evaluation_scores"]) or any(index not in labels_by_dataset[row["dataset"]] for index in indices): raise ValueError("label alignment mismatch")
            labels = np.asarray([labels_by_dataset[row["dataset"]][index] for index in indices], dtype=int)
            if set(labels.tolist()) != {0, 1}: raise ValueError("each dataset needs both derived TEST classes and an anomaly")
            scores = np.asarray(row["evaluation_scores"], dtype=float); automatic = np.asarray(row["automatic"], dtype=bool); review = np.asarray(row["review"], dtype=bool)
            if not np.isfinite(scores).all() or any(len(values) != len(labels) for values in (automatic, review)): raise ValueError("prediction alignment failure")
            positives = labels == 1; negatives = ~positives; missed = positives & ~automatic & ~review
            dataset_residual = int(missed.sum()); dataset_anomalies = int(positives.sum())
            total_anomalies += dataset_anomalies; residual += dataset_residual
            review_fp += int(((automatic | review) & negatives).sum()); automatic_fp += int((automatic & negatives).sum())
            automatic_count += int(automatic.sum()); row_count += len(labels)
            auc = float(roc_auc_score(labels, scores)); aucs.append(auc); fn_rates.append(dataset_residual / dataset_anomalies)
            families.setdefault(family_by_dataset[row["dataset"]], []).append(len(aucs) - 1)
        family_fn = np.mean([np.mean([fn_rates[i] for i in indices]) for indices in families.values()])
        family_auc = np.mean([np.mean([aucs[i] for i in indices]) for indices in families.values()])
        return PolicyMetrics(policy, True, None, _digest(rows), len(rows), row_count, automatic_count,
                             residual, residual / total_anomalies, float(family_fn), review_fp, automatic_fp,
                             float(np.mean(aucs)), float(family_auc), float(elapsed_seconds))
    except Exception as error:
        return PolicyMetrics(policy, False, f"{type(error).__name__}: {error}", _digest(rows), len(rows), 0, 0,
                             0, 0.0, 0.0, 0, 0, 0.0, 0.0, float(elapsed_seconds))


def select_qualified_policy(baseline: PolicyMetrics, candidates: Iterable[PolicyMetrics]) -> str | None:
    if not baseline.complete: return None
    eligible = [row for row in candidates if row.complete and row.policy_id in POLICY_ORDER
                and row.residual_fn_total < baseline.residual_fn_total
                and row.family_balanced_residual_fn_rate < baseline.family_balanced_residual_fn_rate
                and row.review_inclusive_fp_total <= baseline.review_inclusive_fp_total
                and row.automatic_fp_total <= baseline.automatic_fp_total]
    if not eligible: return None
    order = {name: index for index, name in enumerate(POLICY_ORDER)}
    return min(eligible, key=lambda row: (row.residual_fn_total, row.family_balanced_residual_fn_rate,
                                          row.review_inclusive_fp_total, row.automatic_fp_total, order[row.policy_id])).policy_id


@dataclass(frozen=True, slots=True)
class BackendResult:
    backend: str; finite: bool; parity: bool; decisions_equal: bool; tolerance: float
    cold_seconds: float; warm_seconds: float; speedup: float; fallback_reason: str


def select_backend(reference: str, result: BackendResult) -> str:
    return result.backend if result.finite and result.parity and result.decisions_equal and result.speedup >= 1.1 else reference


def _backend_document(reference: str, result: BackendResult) -> dict[str, Any]:
    selected = select_backend(reference, result)
    reasons = []
    if not result.finite: reasons.append("non-finite output")
    if not result.parity: reasons.append(f"numerical parity exceeded atol {result.tolerance}")
    if not result.decisions_equal: reasons.append("route/candidate decisions differed")
    if result.speedup < 1.1: reasons.append(f"complete-stage speedup {result.speedup:.6g}x is below 1.1x")
    return {**asdict(result), "selected": selected,
            "fallback_reason": "; ".join(reasons) if selected == reference else ""}


def _router_decision_for_order(router, ordered: np.ndarray) -> str:
    selected = []
    groups = set()
    for index in ordered:
        group = router._records[int(index)].group_id
        if group not in groups:
            groups.add(group); selected.append(int(index))
            if len(selected) == 31: break
    evidence = [router._candidate_evidence(method, selected) for method in METHOD_NAMES[1:]]
    passed = [item for item in evidence if item.passed]
    if not passed: return METHOD_NAMES[0]
    fixed_order = {name: -index for index, name in enumerate(METHOD_NAMES[1:])}
    return max(passed, key=lambda item: (item.mean_zf_gain, item.auc_gain_quantile,
                                         fixed_order[item.method_name])).method_name


def _numpy_pvalues(reference, query):
    reference = np.asarray(reference, dtype=float)
    query = np.asarray(query, dtype=float)
    if reference.ndim != 1 or query.ndim != 1 or not np.isfinite(reference).all() or not np.isfinite(query).all() or len(reference) == 0:
        raise ValueError("finite one-dimensional non-empty reference and query are required")
    reference = np.frombuffer(reference.tobytes(), dtype=float)
    query = np.frombuffer(query.tobytes(), dtype=float)
    return np.asarray([(1 + np.count_nonzero(reference >= value)) / (len(reference) + 1) for value in query])


def benchmark_backends() -> dict[str, Any]:
    resample_calls = 100_000
    pvalue_calls = 1_000
    rng = np.random.default_rng(20260721); row = rng.normal(size=997); targets = 521
    reference_resample = lambda: np.interp(np.linspace(0, len(row) - 1, targets), np.arange(len(row)), row)
    start = perf_counter(); actual = resample_series(row, targets); resample_cold = perf_counter() - start
    start = perf_counter(); resample_series(row, targets); resample_warm = perf_counter() - start
    start = perf_counter(); expected = reference_resample()
    for _ in range(resample_calls - 1): reference_resample()
    numpy_resample = perf_counter() - start
    start = perf_counter()
    for _ in range(resample_calls - 1): resample_series(row, targets)
    numba_resample_total = resample_cold + perf_counter() - start
    resample_result = BackendResult("numba", bool(np.isfinite(actual).all()), bool(np.allclose(expected, actual, atol=1e-12, rtol=0)), True, 1e-12,
                                    resample_cold, resample_warm, numpy_resample / max(numba_resample_total, 1e-12), "")
    train = rng.normal(size=1000); test = rng.normal(size=2000)
    start = perf_counter(); actual_p = empirical_upper_pvalues(train, test); numba_p_cold = perf_counter() - start
    start = perf_counter(); empirical_upper_pvalues(train, test); numba_p_warm = perf_counter() - start
    start = perf_counter(); expected_p = _numpy_pvalues(train, test)
    for _ in range(pvalue_calls - 1): _numpy_pvalues(train, test)
    numpy_p = perf_counter() - start
    start = perf_counter()
    for _ in range(pvalue_calls - 1): empirical_upper_pvalues(train, test)
    numba_p_total = numba_p_cold + perf_counter() - start
    decisions = all(np.array_equal(expected_p <= alpha, actual_p <= alpha) for alpha in (0.005, 0.01, 0.05))
    p_result = BackendResult("numba", bool(np.isfinite(actual_p).all()), bool(np.allclose(expected_p, actual_p, atol=1e-12, rtol=0)), decisions, 1e-12,
                             numba_p_cold, numba_p_warm, numpy_p / max(numba_p_total, 1e-12), "")
    try: import mlx.core as mx  # type: ignore
    except ImportError:
        mlx = {"backend": "mlx", "installed": False, "mlx_version": _version("mlx"),
               "mlx_metal_version": _version("mlx-metal"), "device": "unavailable",
               "finite": None, "parity": None, "route_decisions_equal": None,
               "tolerance": 1e-12, "cold_seconds": None, "warm_seconds": None,
               "speedup": None, "selected": "numpy",
               "fallback_reason": "MLX and mlx-metal are not installed; router distance remains NumPy"}
    else:
        route_records = tuple(RouterTrainingRecord(i, dict(zip(FEATURE_KEYS, (i + 2, 40, i / 40, 0, 0, 1, 0.5), strict=True)),
                                                   (MethodMetric(METHOD_NAMES[0], 0.6, 1),
                                                    MethodMetric(METHOD_NAMES[1], 0.7, 2),
                                                    MethodMetric(METHOD_NAMES[2], 0.5, 1)))
                              for i in range(40))
        route_router = fit_train_feature_router(route_records)
        matrix = route_router._matrix; query = matrix[0]
        route_calls = 1_000
        start = perf_counter(); expected_d = np.linalg.norm(matrix - query, axis=1); numpy_cold = perf_counter() - start
        start = perf_counter()
        for _ in range(route_calls - 1): np.linalg.norm(matrix - query, axis=1)
        numpy_total = numpy_cold + perf_counter() - start
        start = perf_counter(); device_values = mx.sqrt(mx.sum((mx.array(matrix) - mx.array(query)) ** 2, axis=1)); mx.eval(device_values); actual_d = np.asarray(device_values); mlx_cold = perf_counter() - start
        start = perf_counter(); device_values = mx.sqrt(mx.sum((mx.array(matrix) - mx.array(query)) ** 2, axis=1)); mx.eval(device_values); actual_d = np.asarray(device_values); mlx_warm = perf_counter() - start
        start = perf_counter()
        for _ in range(route_calls - 1):
            device_values = mx.sqrt(mx.sum((mx.array(matrix) - mx.array(query)) ** 2, axis=1)); mx.eval(device_values); np.asarray(device_values)
        mlx_total = mlx_cold + perf_counter() - start
        expected_method = route_router.route(route_records[0].features).selected_method
        actual_method = _router_decision_for_order(route_router, np.argsort(actual_d, kind="stable"))
        result = BackendResult("mlx", bool(np.isfinite(actual_d).all()), bool(np.allclose(expected_d, actual_d, atol=1e-12, rtol=0)),
                               bool(np.array_equal(np.argsort(expected_d, kind="stable")[:31], np.argsort(actual_d, kind="stable")[:31])
                                    and expected_method == actual_method), 1e-12,
                               mlx_cold, mlx_warm, numpy_total / max(mlx_total, 1e-12), "")
        mlx = {**_backend_document("numpy", result), "installed": True,
               "mlx_version": _version("mlx"), "mlx_metal_version": _version("mlx-metal"),
               "device": str(mx.default_device()),
               "route_decisions_equal": result.decisions_equal,
               "decision_check": "neighbor order and final route method"}
    return {"benchmark_calls": {"resampling": resample_calls, "empirical_p": pvalue_calls},
            "resampling": _backend_document("numpy", resample_result),
            "empirical_p": {**_backend_document("numpy", p_result),
                            "measured_path": "anomaly_detection.empirical_upper_pvalues"},
            "router_distance": mlx, "transformers": "frozen_aeon_cpu"}


def render_progress_dashboard(path: Path, *, phase: str, status: str, requested: int, eligible: int,
                              completed: int, failures: int, backend: str, elapsed_seconds: float) -> None:
    fields = (("Phase", phase), ("Status", status), ("Requested", requested), ("Eligible", eligible),
              ("Completed", completed), ("Failures", failures), ("Backend", backend), ("Elapsed seconds", elapsed_seconds))
    html = "<!doctype html><meta charset=utf-8><title>Policy qualification progress</title><h1>Policy qualification progress</h1><dl>" + "".join(f"<dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd>" for key, value in fields) + "</dl>"
    _atomic_write(path, html.encode())


def _evaluate_policy_stage(predictions_by_policy: Mapping[str, list[dict[str, Any]]],
                           labels_by_policy: Mapping[str, Mapping[str, Mapping[int, int]]],
                           families: Mapping[str, str], *, dashboard_path: Path,
                           requested: int, eligible: int, clock=perf_counter,
                           renderer=render_progress_dashboard) -> tuple[list[PolicyMetrics], float]:
    stage_started = clock()
    metrics = []
    for policy_id, predictions in predictions_by_policy.items():
        started = clock()
        metric = evaluate_frozen_predictions(predictions, labels_by_policy[policy_id], families)
        metrics.append(replace(metric, elapsed_seconds=max(0.0, clock() - started)))
    elapsed = max(0.0, clock() - stage_started)
    renderer(dashboard_path, phase="evaluation", status="complete", requested=requested,
             eligible=eligible, completed=eligible,
             failures=sum(not row.complete for row in metrics), backend="cpu",
             elapsed_seconds=elapsed)
    return metrics, elapsed


def _load_baseline_methods(names: Iterable[str]) -> dict[str, str]:
    if sha256(OFFICIAL_SOURCE.read_bytes()).hexdigest() != OFFICIAL_SOURCE_SHA256: raise ValueError("official source digest mismatch")
    decisions = json.loads(OFFICIAL_SOURCE.read_text())["evaluation"]["decisions"]
    mapping = {row["dataset"]: row["method"] for row in decisions}
    if any(name not in mapping or mapping[name] not in METHOD_NAMES for name in names): raise ValueError("baseline route missing")
    return {name: mapping[name] for name in names}


def _baseline_prediction(prepared: PreparedDatasetValues, method: str, scorer) -> dict[str, Any]:
    detection = run_method_detection(prepared, method, scorer=scorer); floor = 1 / (len(prepared.train_values) + 1)
    automatic = detection.final_pvalues <= ALPHA
    review = (~automatic) & (floor > ALPHA) & np.isclose(detection.final_pvalues, floor, atol=1e-15, rtol=0)
    return {"policy_id": "current_baseline", "dataset": prepared.dataset_name, "test_indices": list(prepared.test_indices),
            "evaluation_scores": (1 - detection.final_pvalues).tolist(), "p_values": detection.final_pvalues.tolist(),
            "automatic": automatic.tolist(), "review": review.tolist(), "selected_method": method,
            "method_p_values": [{method: float(detection.final_pvalues[i])} for i in range(len(prepared.test_indices))],
            "method_seed_p_values": [{method: tuple(float(values[i]) for values in detection.seed_pvalues)}
                                     for i in range(len(prepared.test_indices))]}


def _labels(connection: sqlite3.Connection, prepared: PreparedDatasetValues) -> dict[int, int]:
    from .dataset_repository import load_test_labels
    return load_test_labels(connection, prepared.dataset_name, prepared.test_indices)


def _router_settings_sha() -> str:
    return _digest({"minimum_groups": 20, "maximum_neighbors": 31,
                    "non_degradation_rate": 0.9, "auc_quantile": 0.2,
                    "method_order": METHOD_NAMES})


def _router_payload(router, records: Iterable[RouterTrainingRecord], *, contract_sha: str,
                    source_sha: str) -> dict[str, Any]:
    anonymous = []
    for record in records:
        by_method = {metric.method_name: metric for metric in record.metrics}
        baseline = by_method[METHOD_NAMES[0]]
        anonymous.append({"group_id": record.group_id,
                          "features": {key: record.features[key] for key in FEATURE_KEYS},
                          "performance_differences": {
                              method: {"auc_gain": by_method[method].auc - baseline.auc,
                                       "zf_gain": by_method[method].zf_true_positives - baseline.zf_true_positives}
                              for method in METHOD_NAMES[1:]}})
    core = {"schema": "anonymous_router_v1", "contract_sha256": contract_sha,
            "source_sha256": source_sha, "feature_schema_sha256": _digest(FEATURE_KEYS),
            "settings_sha256": _router_settings_sha(), "input_keys": list(router.input_keys),
            "normalization": {"medians": router._medians.tolist(), "means": router._means.tolist(),
                              "scales": router._scales.tolist()},
            "records": anonymous}
    return {**core, "router_sha256": _digest(core)}


def _router_from_payload(payload: Mapping[str, Any], *, contract_sha: str, source_sha: str):
    try:
        expected_keys = {"schema", "contract_sha256", "source_sha256", "feature_schema_sha256",
                         "settings_sha256", "input_keys", "normalization", "records", "router_sha256"}
        if set(payload) != expected_keys:
            raise ValueError("router schema mismatch")
        core = {key: value for key, value in payload.items() if key != "router_sha256"}
        if (payload["schema"] != "anonymous_router_v1" or payload["contract_sha256"] != contract_sha
                or payload["source_sha256"] != source_sha
                or payload["feature_schema_sha256"] != _digest(FEATURE_KEYS)
                or payload["settings_sha256"] != _router_settings_sha()
                or payload["input_keys"] != list(FEATURE_KEYS)
                or payload["router_sha256"] != _digest(core)
                or set(payload["normalization"]) != {"medians", "means", "scales"}):
            raise ValueError("router identity mismatch")
        records = []
        for row in payload["records"]:
            if set(row) != {"group_id", "features", "performance_differences"}:
                raise ValueError("router record schema mismatch")
            differences = row["performance_differences"]
            if set(differences) != set(METHOD_NAMES[1:]) or any(
                    set(differences[method]) != {"auc_gain", "zf_gain"} for method in METHOD_NAMES[1:]):
                raise ValueError("router performance difference schema mismatch")
            auc_gains = [float(differences[method]["auc_gain"]) for method in METHOD_NAMES[1:]]
            zf_gains = [int(differences[method]["zf_gain"]) for method in METHOD_NAMES[1:]]
            if not np.isfinite(auc_gains).all() or any(type(differences[method]["zf_gain"]) is not int for method in METHOD_NAMES[1:]):
                raise ValueError("router differences must be finite")
            baseline_auc = max(0.0, -min([0.0, *auc_gains]))
            baseline_zf = max(0, -min([0, *zf_gains]))
            metrics = [MethodMetric(METHOD_NAMES[0], baseline_auc, baseline_zf)]
            metrics.extend(MethodMetric(method, baseline_auc + auc_gain, baseline_zf + zf_gain)
                           for method, auc_gain, zf_gain in zip(METHOD_NAMES[1:], auc_gains, zf_gains, strict=True))
            features = {key: row["features"][key] for key in FEATURE_KEYS}
            if set(row["features"]) != set(FEATURE_KEYS):
                raise ValueError("router feature schema mismatch")
            records.append(RouterTrainingRecord(row["group_id"], features, tuple(metrics)))
        router = fit_train_feature_router(records)
        normalized = payload["normalization"]
        if not all(np.array_equal(getattr(router, f"_{name}"), np.asarray(normalized[name], dtype=float))
                   for name in ("medians", "means", "scales")):
            raise ValueError("router normalization mismatch")
        if _router_payload(router, records, contract_sha=contract_sha, source_sha=source_sha) != payload:
            raise ValueError("router replay mismatch")
        return router
    except Exception as error:
        raise ValueError("malformed anonymous router artifact") from error


def _router_binding(records: Iterable[RouterTrainingRecord], router) -> str:
    return _router_payload(router, records, contract_sha="binding", source_sha="binding")["router_sha256"]


def _run_outer_family_unit(*, outer_manifest: ValueManifest, unit_path: Path, contract_sha: str,
                           load_inner_labels, build_records, build_predictions, load_outer_labels,
                           resume: bool, events: list[str]):
    inner_labels = load_inner_labels(); events.append("inner_labels")
    records = tuple(build_records(inner_labels)); router = fit_train_feature_router(records); events.append("router_fit")
    binding = _router_binding(records, router)
    if resume and unit_path.exists():
        unit = load_frozen_predictions(unit_path, outer_manifest, contract_sha=contract_sha, binding_sha=binding)
        events.append("outer_resume_verify")
    else:
        unit = freeze_predictions(build_predictions(router), outer_manifest, unit_path,
                                  contract_sha=contract_sha, binding_sha=binding)
        events.append("outer_freeze_hash")
    outer_labels = load_outer_labels(); events.append("outer_labels")
    return unit, outer_labels, router, binding


def _synthetic_c_check() -> dict[str, Any]:
    metrics = (MethodMetric(METHOD_NAMES[0], 0.7, 0), MethodMetric(METHOD_NAMES[1], 0.8, 1),
               MethodMetric(METHOD_NAMES[2], 0.6, 0))
    records = tuple(RouterTrainingRecord(i, dict(zip(FEATURE_KEYS, (i + 2, 20, i, i, i, i, i), strict=True)), metrics)
                    for i in range(20))
    status = DatasetValueStatus("outer", True, None, "raw", "prepared", "value_only_numba",
                                20, 2, 3, (0, 1))
    manifest = ValueManifest(("outer",), (status,), _digest({"requested_names": ["outer"], "statuses": [asdict(status)]}))
    row = {**{"policy_id": "train_feature_router", "dataset": "outer", "test_indices": [0, 1],
              "evaluation_scores": [0.1, 0.9], "p_values": [0.9, 0.1], "automatic": [False, False],
              "review": [False, False], "selected_method": METHOD_NAMES[0]},
           "method_p_values": [{METHOD_NAMES[0]: 0.9}, {METHOD_NAMES[0]: 0.1}],
           "method_seed_p_values": [{METHOD_NAMES[0]: [0.9, 0.9]}, {METHOD_NAMES[0]: [0.1, 0.1]}]}
    builds = []
    def build_predictions(_router):
        builds.append(1)
        return [row]
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "outer.json"
        events = ["global_score_freeze"]
        first, _, router, _ = _run_outer_family_unit(
            outer_manifest=manifest, unit_path=path, contract_sha="contract",
            load_inner_labels=lambda: {i: (0, 1) for i in range(20)}, build_records=lambda _labels: records,
            build_predictions=build_predictions, load_outer_labels=lambda: {"outer": {0: 0, 1: 1}},
            resume=False, events=events)
        first_builds = len(builds)
        resume_events = ["global_score_freeze"]
        resumed, _, _, _ = _run_outer_family_unit(
            outer_manifest=manifest, unit_path=path, contract_sha="contract",
            load_inner_labels=lambda: {i: (0, 1) for i in range(20)}, build_records=lambda _labels: records,
            build_predictions=build_predictions, load_outer_labels=lambda: {"outer": {0: 0, 1: 1}},
            resume=True, events=resume_events)
    return {"passed": router.route(records[0].features).support == 20,
            "events": events, "resume_events": resume_events,
            "first_prediction_builds": first_builds, "resumed_prediction_builds": len(builds) - first_builds,
            "first_prediction_digest": first["prediction_sha256"],
            "resumed_prediction_digest": resumed["prediction_sha256"],
            "scope": "synthetic 20 anonymous groups; real-data C deferred to Task 5"}


def _freeze_then_load_labels(rows_by_policy: Mapping[str, Iterable[Mapping[str, Any]]],
                             manifest: ValueManifest, contract_sha: str, prediction_path,
                             load_labels, events: list[str]):
    frozen = {policy: freeze_predictions(rows, manifest, prediction_path(policy),
                                         contract_sha=contract_sha)
              for policy, rows in rows_by_policy.items()}
    events.append("freeze")
    labels = load_labels()
    events.append("labels")
    return frozen, labels


def run_smoke(names: list[str]) -> dict[str, Any]:
    if names != ["Trace_normal_2", "AllGestureWiimoteX_normal_1", "EOGVerticalSignal_normal_12"]: raise ValueError("smoke requires the approved three datasets in order")
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True); connection = open_source_database()
    try:
        events = []
        manifest = build_value_manifest(connection, names); events.append("values")
        families = {name: name.split("_normal_")[0] for name in names}
        assessment_path = RUNTIME_ROOT / "backend_assessment.json"
        assessment = json.loads(assessment_path.read_text()) if assessment_path.exists() else benchmark_backends()
        contract_record = build_execution_contract_record(manifest, OFFICIAL_SOURCE_SHA256, families,
                                                          assessment, workers=1)
        checkpoint = RUNTIME_ROOT / "smoke_scores.jsonl"; calls = []
        def counting(train, test, method, seed): calls.append((method, seed)); return score_method(train, test, method, seed)
        started_from_checkpoint = checkpoint.exists()
        contract_record = _persist_execution_contract(RUNTIME_ROOT / "smoke_execution_contract.json",
                                                      contract_record, resume=started_from_checkpoint)
        contract = contract_record.sha256
        first = score_with_checkpoint(connection, manifest, contract, checkpoint, scorer=counting, resume=started_from_checkpoint)
        first_calls = len(calls); second = score_with_checkpoint(connection, manifest, contract, checkpoint, scorer=counting, resume=True)
        resumed_calls = len(calls) - first_calls
        events.append("scores")
        baseline_methods = _load_baseline_methods(names); prepared = {name: prepare_dataset_values(connection, name) for name in names}
        rows_by_policy = {}; results = {}
        for policy in ("current_baseline", "single_minirocket", "three_method_consensus"):
            rows = []
            for name in names:
                scorer = _cached_scorer(second[name])
                result = (_baseline_prediction(prepared[name], baseline_methods[name], scorer) if policy == "current_baseline"
                          else _prediction(prepared[name], run_analysis_policy(prepared[name], policy, scorer=scorer)))
                rows.append(result)
            rows_by_policy[policy] = rows
        def load_labels():
            label_connection = open_source_database()
            try: return {name: _labels(label_connection, prepared[name]) for name in names}
            finally: label_connection.close()
        frozen, labels = _freeze_then_load_labels(
            rows_by_policy, manifest, contract,
            lambda policy: RUNTIME_ROOT / f"smoke_{policy}_predictions.json",
            load_labels, events)
        for policy, payload in frozen.items(): results[policy] = asdict(evaluate_frozen_predictions(payload["predictions"], labels, families))
        trace_a = next(row for row in frozen["single_minirocket"]["predictions"] if row["dataset"] == "Trace_normal_2")
        position = trace_a["test_indices"].index(49)
        trace_ok = not trace_a["automatic"][position] and trace_a["review"][position]
        output = {"contract_sha256": contract, "manifest_digest": manifest.digest,
                  "started_from_checkpoint": started_from_checkpoint, "first_scorer_calls": first_calls,
                  "resumed_scorer_calls": resumed_calls, "checkpoint_identical": first == second,
                  "prediction_digests": {key: value["prediction_sha256"] for key, value in frozen.items()},
                  "trace_49_seed_consensus": trace_ok, "label_access_order": events,
                  "synthetic_c": _synthetic_c_check(), "metrics": results}
        evidence_name = "smoke_resume_result.json" if started_from_checkpoint else "smoke_first_result.json"
        _atomic_write(RUNTIME_ROOT / evidence_name, _canonical_bytes(output))
        if started_from_checkpoint and (RUNTIME_ROOT / "smoke_first_result.json").exists():
            first_evidence = json.loads((RUNTIME_ROOT / "smoke_first_result.json").read_text())
            _atomic_write(RUNTIME_ROOT / "smoke_repeatability.json", _canonical_bytes({
                "first_scorer_calls": first_evidence["first_scorer_calls"],
                "resumed_scorer_calls": output["first_scorer_calls"],
                "identical_contract": first_evidence["contract_sha256"] == output["contract_sha256"],
                "identical_predictions": first_evidence["prediction_digests"] == output["prediction_digests"],
                "identical_metrics": first_evidence["metrics"] == output["metrics"],
            }))
        return output
    finally: connection.close()


def _load_execution_contract(path: Path) -> ExecutionContract:
    try: document = json.loads(path.read_text()); payload = document["payload"]
    except Exception as error: raise ValueError("malformed execution contract") from error
    canonical = _canonical_bytes(payload); digest = sha256(canonical).hexdigest()
    if set(document) != {"schema", "payload", "sha256"} or document["schema"] != "execution_contract_v1" or document["sha256"] != digest:
        raise ValueError("execution contract digest mismatch")
    return ExecutionContract(payload, canonical, digest)


def _persist_execution_contract(path: Path, expected: ExecutionContract, *, resume: bool) -> ExecutionContract:
    if not resume:
        _atomic_write(path, _canonical_bytes({"schema": "execution_contract_v1",
                                              "payload": expected.payload, "sha256": expected.sha256}))
    loaded = _load_execution_contract(path)
    if loaded.sha256 != expected.sha256 or loaded.payload != expected.payload:
        raise ValueError("stored execution contract no longer matches authoritative inputs")
    return loaded


def _official_context(*, workers: int | None = None) -> tuple[ValueManifest, dict[str, str], ExecutionContract]:
    names, families = load_official_population()
    manifest_path = RUNTIME_ROOT / "official_value_manifest.json"
    connection = open_source_database()
    try: fresh = build_value_manifest(connection, names)
    finally: connection.close()
    if manifest_path.exists():
        manifest = ValueManifest.from_dict(json.loads(manifest_path.read_text()))
        if manifest.requested_names != names or manifest.digest != fresh.digest:
            raise ValueError("stored official value manifest is stale or has wrong population")
    else:
        manifest = fresh
        _atomic_write(manifest_path, manifest.to_json().encode())
    assessment_path = RUNTIME_ROOT / "backend_assessment.json"
    if not assessment_path.exists(): raise ValueError("backend assessment is required before official scoring")
    assessment = json.loads(assessment_path.read_text())
    contract_path = RUNTIME_ROOT / "official_execution_contract.json"
    if workers is not None:
        expected = build_execution_contract_record(manifest, OFFICIAL_SOURCE_SHA256, families,
                                                   assessment, workers=workers)
        contract = _persist_execution_contract(contract_path, expected, resume=contract_path.exists())
    else:
        contract = _load_execution_contract(contract_path)
        expected = build_execution_contract_record(manifest, OFFICIAL_SOURCE_SHA256, families, assessment,
                                                   workers=contract.payload["execution"]["workers"])
        if contract.sha256 != expected.sha256:
            raise ValueError("stored execution contract no longer matches authoritative inputs")
    return manifest, families, contract


def _invoke_score_job(score_one, name: str):
    try: return score_one(name), None
    except ScoreStageError as error: return None, _failure_row(error)


def _score_official_dataset(name: str) -> tuple[str, dict[str, Any]]:
    connection = open_source_database()
    try: return name, _score_row(prepare_dataset_values(connection, name), score_method)
    finally: connection.close()


def _score_jobs(names: Iterable[str], workers: int, score_one):
    if workers == 1:
        for name in names:
            yield _invoke_score_job(score_one, name)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_invoke_score_job, score_one, name) for name in names]
        for future in as_completed(futures):
            yield future.result()


def run_official(*, resume: bool, workers: int) -> dict[str, Any]:
    manifest, _, contract_record = _official_context(workers=workers)
    contract = contract_record.sha256
    official_names, _ = load_official_population()
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_write(EVIDENCE_ROOT / "official_dataset_population.json",
                  _canonical_bytes({"contract_sha256": contract, "source_sha256": OFFICIAL_SOURCE_SHA256, "count": len(official_names),
                                    "datasets": list(official_names)}))
    _atomic_write(EVIDENCE_ROOT / "policy_qualification_manifest.json",
                  _canonical_bytes({"contract_sha256": contract, "value_manifest": manifest.as_dict()}))
    checkpoint_path = RUNTIME_ROOT / "official_scores.jsonl"
    failure_path = RUNTIME_ROOT / "official_score_failures.jsonl"
    rows = (_read_score_checkpoint(checkpoint_path, contract, manifest, require_complete=False)
            if resume and checkpoint_path.exists() else {})
    failures: list[dict[str, Any]] = []
    _write_score_failures(failure_path, contract, manifest, failures)
    missing = [name for name in manifest.eligible_names if name not in rows]
    for result, failure in _score_jobs(missing, workers, _score_official_dataset):
        if failure is not None:
            failures.append(failure); _write_score_failures(failure_path, contract, manifest, failures)
        else:
            name, row = result; rows[name] = row; _write_checkpoint(checkpoint_path, contract, rows)
    if failures:
        render_progress_dashboard(RUNTIME_ROOT / "progress.html", phase="scores", status="incomplete",
                                  requested=manifest.total_requested, eligible=manifest.eligible_count,
                                  completed=len(rows), failures=len(failures), backend="cpu", elapsed_seconds=0.0)
        return {"contract_sha256": contract, "requested": manifest.total_requested,
                "eligible": manifest.eligible_count, "completed": len(rows),
                "score_failures": len(failures), "workers": workers, "status": "incomplete",
                "policy_status": {policy: {"complete": False, "failure": "score stage incomplete"}
                                  for policy in ("current_baseline", *POLICY_ORDER)}}
    cache = load_score_checkpoint(checkpoint_path, contract, manifest)
    connection = open_source_database()
    try:
        baseline_methods = _load_baseline_methods(manifest.eligible_names)
        policies = ("current_baseline", "single_minirocket", "three_method_consensus")
        digests = {}
        for policy in policies:
            rows = []
            for name in manifest.eligible_names:
                prepared = prepare_dataset_values(connection, name); scorer = _cached_scorer(cache[name])
                rows.append(_baseline_prediction(prepared, baseline_methods[name], scorer) if policy == "current_baseline"
                            else _prediction(prepared, run_analysis_policy(prepared, policy, scorer=scorer)))
            frozen = freeze_predictions(rows, manifest, RUNTIME_ROOT / f"official_{policy}_predictions.json", contract_sha=contract)
            digests[policy] = frozen["prediction_sha256"]
        result = {"contract_sha256": contract, "requested": manifest.total_requested,
                  "eligible": manifest.eligible_count, "prediction_digests": digests,
                  "workers": workers, "status": "predictions_frozen"}
        render_progress_dashboard(RUNTIME_ROOT / "progress.html", phase="freeze", status="complete",
                                  requested=manifest.total_requested, eligible=manifest.eligible_count,
                                  completed=manifest.eligible_count, failures=manifest.total_requested - manifest.eligible_count,
                                  backend="cpu", elapsed_seconds=0.0)
        render_progress_dashboard(QUALIFICATION_DASHBOARD, phase="freeze", status="complete",
                                  requested=manifest.total_requested, eligible=manifest.eligible_count,
                                  completed=manifest.eligible_count, failures=manifest.total_requested - manifest.eligible_count,
                                  backend="cpu", elapsed_seconds=0.0)
        return result
    finally: connection.close()


def _method_metrics(row: Mapping[str, Any], prepared: PreparedDatasetValues,
                    labels: Mapping[int, int]) -> tuple[MethodMetric, ...]:
    values = []
    for method in METHOD_NAMES:
        detection = run_method_detection(prepared, method, scorer=_cached_scorer(row))
        actual = np.asarray([labels[index] for index in prepared.test_indices], dtype=int)
        if set(actual.tolist()) != {0, 1}: raise ValueError("router training requires both TEST classes")
        scores = 1 - detection.final_pvalues; normal_max = float(np.max(scores[actual == 0]))
        zf = int(np.count_nonzero((actual == 1) & (scores > normal_max)))
        values.append(MethodMetric(method, float(roc_auc_score(actual, scores)), zf))
    return tuple(values)


def evaluate_official(*, resume: bool) -> dict[str, Any]:
    manifest, families, contract_record = _official_context()
    contract = contract_record.sha256
    score_failures = _load_score_failures(RUNTIME_ROOT / "official_score_failures.jsonl", contract, manifest)
    if score_failures:
        raise ValueError("official scores are incomplete; qualification is prohibited")
    cache = load_score_checkpoint(RUNTIME_ROOT / "official_scores.jsonl", contract, manifest)
    predictions = {policy: load_frozen_predictions(RUNTIME_ROOT / f"official_{policy}_predictions.json", manifest, contract_sha=contract)["predictions"]
                   for policy in ("current_baseline", "single_minirocket", "three_method_consensus")}
    values_connection = open_source_database()
    try: prepared = {name: prepare_dataset_values(values_connection, name) for name in manifest.eligible_names}
    finally: values_connection.close()
    c_rows = []; c_labels = {}
    for outer_family in sorted(set(families[name] for name in manifest.eligible_names)):
        outer_names = tuple(name for name in manifest.eligible_names if families[name] == outer_family)
        inner_names = tuple(name for name in manifest.eligible_names if families[name] != outer_family)
        groups = {family: index for index, family in enumerate(sorted({families[name] for name in inner_names}))}
        outer_manifest = ValueManifest(tuple(outer_names), tuple(row for row in manifest.statuses if row.name in outer_names),
                                       _digest({"requested_names": list(outer_names), "statuses": [asdict(row) for row in manifest.statuses if row.name in outer_names]}))
        unit_path = RUNTIME_ROOT / "outer_families" / f"{sha256(outer_family.encode()).hexdigest()}.json"
        def load_inner():
            connection = open_source_database()
            try: return {name: _labels(connection, prepared[name]) for name in inner_names}
            finally: connection.close()
        def build_records(inner_labels):
            return [RouterTrainingRecord(groups[families[name]], cache[name]["features"],
                                         _method_metrics(cache[name], prepared[name], inner_labels[name]))
                    for name in inner_names]
        def build_predictions(router):
            return [_prediction(prepared[name], run_analysis_policy(prepared[name], "train_feature_router", router,
                                                                    scorer=_cached_scorer(cache[name])))
                    for name in outer_names]
        def load_outer():
            connection = open_source_database()
            try: return {name: _labels(connection, prepared[name]) for name in outer_names}
            finally: connection.close()
        unit, outer_labels, _, _ = _run_outer_family_unit(
            outer_manifest=outer_manifest, unit_path=unit_path, contract_sha=contract,
            load_inner_labels=load_inner, build_records=build_records,
            build_predictions=build_predictions, load_outer_labels=load_outer,
            resume=resume, events=[])
        c_rows.extend(unit["predictions"])
        c_labels.update(outer_labels)
    c_frozen = freeze_predictions(c_rows, manifest, RUNTIME_ROOT / "official_train_feature_router_predictions.json", contract_sha=contract)
    predictions["train_feature_router"] = c_frozen["predictions"]
    labels_connection = open_source_database()
    try: labels = {name: _labels(labels_connection, prepared[name]) for name in manifest.eligible_names}
    finally: labels_connection.close()
    metric_family = {name: families[name] for name in manifest.eligible_names}
    labels_by_policy = {policy_id: (c_labels if policy_id == "train_feature_router" else labels)
                        for policy_id in predictions}
    metrics, stage_elapsed = _evaluate_policy_stage(
        predictions, labels_by_policy, metric_family, dashboard_path=QUALIFICATION_DASHBOARD,
        requested=manifest.total_requested, eligible=manifest.eligible_count)
    baseline = next(row for row in metrics if row.policy_id == "current_baseline")
    winner = select_qualified_policy(baseline, [row for row in metrics if row.policy_id in POLICY_ORDER])
    result = {"contract_sha256": contract, "metrics": [asdict(row) for row in metrics], "qualified_policy": winner,
              "stage_elapsed_seconds": stage_elapsed,
              "prediction_digests": {row.policy_id: row.prediction_digest for row in metrics}}
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    result_lines = [_canonical_bytes({"schema": "policy_qualification_results_v1", "contract_sha256": contract,
                                      "qualified_policy": winner, "prediction_digests": result["prediction_digests"]})]
    result_lines.extend(_canonical_bytes({"type": "policy_metrics", **asdict(row)}) for row in metrics)
    _atomic_write(EVIDENCE_ROOT / "policy_qualification_results.jsonl", b"".join(result_lines))
    policy_artifact: dict[str, Any] = {"contract_sha256": contract, "policy_id": winner}
    if winner == "train_feature_router":
        groups = {family: index for index, family in enumerate(sorted(set(families[name] for name in manifest.eligible_names)))}
        records = [RouterTrainingRecord(groups[families[name]], cache[name]["features"],
                                        _method_metrics(cache[name], prepared[name], labels[name])) for name in manifest.eligible_names]
        router = fit_train_feature_router(records)
        router_payload = _router_payload(router, records, contract_sha=contract,
                                         source_sha=OFFICIAL_SOURCE_SHA256)
        policy_artifact["router"] = router_payload
    _atomic_write(EVIDENCE_ROOT / "qualified_policy.json", _canonical_bytes(policy_artifact))
    if winner == "train_feature_router":
        committed = json.loads((EVIDENCE_ROOT / "qualified_policy.json").read_text())
        replayed = _router_from_payload(committed["router"], contract_sha=contract,
                                        source_sha=OFFICIAL_SOURCE_SHA256)
        if any(router.route(row.features).selected_method != replayed.route(row.features).selected_method
               for row in records):
            raise ValueError("qualified router artifact changes routing")
    return result


def validate_results_bundle(*, results_path: Path, policy_path: Path, population_path: Path,
                            manifest_path: Path, prediction_paths: Mapping[str, Path],
                            authoritative_names: tuple[str, ...], source_sha: str,
                            fresh_manifest: ValueManifest, contract_sha: str,
                            labels: Mapping[str, Mapping[int, int]], families: Mapping[str, str],
                            authoritative_router_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        lines = [json.loads(line) for line in results_path.read_text().splitlines()]
        header, metric_documents = lines[0], lines[1:]
        population = json.loads(population_path.read_text())
        manifest_document = json.loads(manifest_path.read_text())
        policy = json.loads(policy_path.read_text())
    except Exception as error:
        raise ValueError("malformed official evidence") from error
    policy_ids = ("current_baseline", *POLICY_ORDER)
    if (set(header) != {"schema", "contract_sha256", "qualified_policy", "prediction_digests"}
            or header["schema"] != "policy_qualification_results_v1"
            or header["contract_sha256"] != contract_sha
            or set(header["prediction_digests"]) != set(policy_ids)):
        raise ValueError("results header mismatch")
    if population != {"contract_sha256": contract_sha, "source_sha256": source_sha,
                      "count": len(authoritative_names), "datasets": list(authoritative_names)}:
        raise ValueError("official population evidence mismatch")
    if set(manifest_document) != {"contract_sha256", "value_manifest"} or manifest_document["contract_sha256"] != contract_sha:
        raise ValueError("official manifest evidence mismatch")
    committed_manifest = ValueManifest.from_dict(manifest_document["value_manifest"])
    if committed_manifest.requested_names != authoritative_names or committed_manifest != fresh_manifest:
        raise ValueError("committed value manifest mismatch")
    metric_by_policy = {}
    expected_metric_keys = {"type", *PolicyMetrics.__dataclass_fields__}
    for document in metric_documents:
        if set(document) != expected_metric_keys or document.get("type") != "policy_metrics":
            raise ValueError("metric row schema mismatch")
        policy_id = document.get("policy_id")
        if policy_id in metric_by_policy or policy_id not in policy_ids:
            raise ValueError("metric row coverage mismatch")
        metric_by_policy[policy_id] = document
    if set(metric_by_policy) != set(policy_ids):
        raise ValueError("metric row coverage mismatch")
    recomputed = {}
    for policy_id in policy_ids:
        frozen = load_frozen_predictions(prediction_paths[policy_id], fresh_manifest,
                                         contract_sha=contract_sha)
        metric = evaluate_frozen_predictions(frozen["predictions"], labels, families)
        expected = {"type": "policy_metrics", **asdict(metric)}
        committed = dict(metric_by_policy[policy_id])
        elapsed = committed.pop("elapsed_seconds")
        expected.pop("elapsed_seconds")
        if (type(elapsed) not in (int, float) or not np.isfinite(elapsed) or elapsed < 0
                or committed != expected):
            raise ValueError("metric replay mismatch")
        if header["prediction_digests"][policy_id] != frozen["prediction_sha256"]:
            raise ValueError("prediction evidence digest mismatch")
        recomputed[policy_id] = metric
    winner = select_qualified_policy(recomputed["current_baseline"],
                                     [recomputed[name] for name in POLICY_ORDER])
    if header["qualified_policy"] != winner or policy.get("contract_sha256") != contract_sha or policy.get("policy_id") != winner:
        raise ValueError("qualified policy mismatch")
    expected_policy_keys = {"contract_sha256", "policy_id", "router"} if winner == "train_feature_router" else {"contract_sha256", "policy_id"}
    if set(policy) != expected_policy_keys:
        raise ValueError("qualified policy artifact schema mismatch")
    if winner == "train_feature_router":
        router = _router_from_payload(policy["router"], contract_sha=contract_sha, source_sha=source_sha)
        for row in policy["router"]["records"]:
            router.route(row["features"])
        if authoritative_router_payload is None or policy["router"] != authoritative_router_payload:
            raise ValueError("qualified router does not match authoritative full-data training")
    return {"status": "valid", "contract_sha256": contract_sha, "qualified_policy": winner}


def validate_official_results() -> dict[str, Any]:
    manifest, families, contract_record = _official_context()
    names, _ = load_official_population()
    connection = open_source_database()
    try:
        prepared = {name: prepare_dataset_values(connection, name) for name in manifest.eligible_names}
        labels = {name: _labels(connection, prepared[name]) for name in manifest.eligible_names}
    finally:
        connection.close()
    metric_families = {name: families[name] for name in manifest.eligible_names}
    policy_path = EVIDENCE_ROOT / "qualified_policy.json"
    router_payload = None
    if json.loads(policy_path.read_text()).get("policy_id") == "train_feature_router":
        cache = load_score_checkpoint(RUNTIME_ROOT / "official_scores.jsonl", contract_record.sha256, manifest)
        groups = {family: index for index, family in enumerate(sorted(set(metric_families.values())))}
        records = [RouterTrainingRecord(groups[metric_families[name]], cache[name]["features"],
                                        _method_metrics(cache[name], prepared[name], labels[name]))
                   for name in manifest.eligible_names]
        router_payload = _router_payload(fit_train_feature_router(records), records,
                                         contract_sha=contract_record.sha256,
                                         source_sha=OFFICIAL_SOURCE_SHA256)
    return validate_results_bundle(
        results_path=EVIDENCE_ROOT / "policy_qualification_results.jsonl",
        policy_path=policy_path,
        population_path=EVIDENCE_ROOT / "official_dataset_population.json",
        manifest_path=EVIDENCE_ROOT / "policy_qualification_manifest.json",
        prediction_paths={policy: RUNTIME_ROOT / f"official_{policy}_predictions.json"
                          for policy in ("current_baseline", *POLICY_ORDER)},
        authoritative_names=names, source_sha=OFFICIAL_SOURCE_SHA256, fresh_manifest=manifest,
        contract_sha=contract_record.sha256, labels=labels, families=metric_families,
        authoritative_router_payload=router_payload)


def _parser() -> ArgumentParser:
    parser = ArgumentParser(); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--benchmark-backends", action="store_true"); modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--official", action="store_true"); modes.add_argument("--evaluate-frozen", action="store_true")
    modes.add_argument("--validate-results", action="store_true")
    parser.add_argument("--datasets", nargs="+"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--workers", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser(); args = parser.parse_args(argv)
    if args.smoke and not args.datasets: parser.error("--smoke requires --datasets")
    if not args.smoke and args.datasets: parser.error("--datasets is smoke-only")
    if args.official and not args.resume: parser.error("--official requires --resume")
    if args.evaluate_frozen and not args.resume: parser.error("--evaluate-frozen requires --resume")
    if args.official and args.workers is None: parser.error("--official requires --workers")
    if not args.official and args.workers is not None: parser.error("--workers is official-only")
    if args.resume and not (args.official or args.evaluate_frozen): parser.error("--resume is official/evaluate-only")
    if args.workers is not None and not 1 <= args.workers <= 7: parser.error("--workers must be between 1 and 7")
    if args.benchmark_backends:
        result = benchmark_backends(); _atomic_write(RUNTIME_ROOT / "backend_assessment.json", _canonical_bytes(result)); print(json.dumps(result, sort_keys=True)); return 0
    if args.smoke: print(json.dumps(run_smoke(args.datasets), sort_keys=True)); return 0
    if args.official: print(json.dumps(run_official(resume=args.resume, workers=args.workers), sort_keys=True)); return 0
    if args.evaluate_frozen: print(json.dumps(evaluate_official(resume=args.resume), sort_keys=True)); return 0
    if args.validate_results: print(json.dumps(validate_official_results(), sort_keys=True)); return 0
    return 0


if __name__ == "__main__": raise SystemExit(main())
