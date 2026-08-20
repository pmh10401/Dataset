"""Independent scorer-free verification for immutable Exp450 score roots."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import stat

import numpy as np


_CONDITION_IDS = ("A", "B", "C", "D")
_POLICY_ORDER = ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")
_SHARD_NAME = "condition_score_shard_antigravity.json"
_MANIFEST_NAME = "condition_score_manifest_antigravity.json"
_SHARD_SCHEMA = "exp450-condition-score-shard-antigravity.v1"
_MANIFEST_SCHEMA = "exp450-condition-score-manifest-antigravity.v1"
_CONDITIONS = {
    "A": ("linear", "linear"), "B": ("polyphase_pchip", "linear"),
    "C": ("linear", "area"), "D": ("polyphase_pchip", "area"),
}
_SOURCES = ("rocket_exp40", "exp55_best", "exp56_best", "exp84", "rocket_block_b", "canonical_normal_state")
_ALPHAS = {"rocket_exp40": .03, "exp55_best": .03, "exp56_best": .02, "exp84": .02, "rocket_block_b": .015, "canonical_normal_state": .02}
_ORIGIN_IDS = ("rocket_exp40", "exp55_best", "exp56_best", "exp84_family_guard", "rocket_block_b", "canonical_normal_state")
_ORIGIN_SHA = ("5d9f3ff6e74d32b46d865f8b32b2ea2370acdd9d86aeb36f144c816384180fa3",)*4 + ("8d38067291d78defeb40410f1dde8dac44e9e3bd9e2a2636eb4c33ce29932fc2", "49f15e39c39a801842abca43cf67f73b79ded0a273029071e079f29100d2cc43")


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("score root data is noncanonical") from error


def _read(path: Path) -> tuple[object, bytes]:
    details = path.lstat() if path.exists() else None
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("score root artifact is missing")
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("score root artifact is invalid") from error
    if data != _canonical_bytes(value):
        raise ValueError("score root artifact is noncanonical")
    return value, data


def _directory_children(path: Path) -> set[str]:
    details = path.lstat() if path.exists() else None
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("score root inventory differs")
    return {child.name for child in path.iterdir()}


def _root(context: object, root: Path) -> Path:
    bound = getattr(context, "run_root", None)
    if not isinstance(bound, Path) or Path(root) != bound or not bound.is_absolute():
        raise ValueError("score root context differs")
    details = bound.lstat() if bound.exists() else None
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("score root is invalid")
    return bound


def _context_sha256(context: object) -> str:
    value = getattr(context, "sha256", None)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("score root context differs")
    return value


def _backend(context: object) -> dict[str, object]:
    backend = getattr(context, "backend", None)
    artifact_record = getattr(backend, "artifact_record", None)
    if not callable(artifact_record):
        raise ValueError("score root backend differs")
    value = artifact_record()
    if not isinstance(value, dict):
        raise ValueError("score root backend differs")
    return value


def _catalog(catalog: object, context: object) -> list[dict[str, object]]:
    if not isinstance(catalog, dict) or set(catalog) != {"schema", "mode", "database", "datasets", "catalog_sha256"} or catalog.get("catalog_sha256") != _digest({key: value for key, value in catalog.items() if key != "catalog_sha256"}) or catalog.get("catalog_sha256") != getattr(context, "catalog_sha256", None):
        raise ValueError("score root catalog differs")
    entries = catalog.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("score root catalog differs")
    datasets = [entry.get("dataset") if isinstance(entry, dict) else None for entry in entries]
    if any(type(dataset) is not str or not dataset for dataset in datasets) or len(set(datasets)) != len(datasets):
        raise ValueError("score root catalog differs")
    return entries


def _json_float_vector(value: object, size: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != size or any(type(item) not in {int, float} for item in value):
        raise ValueError("score root continuous score differs")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("score root continuous score differs") from error
    if not np.all(np.isfinite(result)):
        raise ValueError("score root continuous score differs")
    return result


def _check_checkpoint(value: object, entry: dict[str, object], condition_id: str, context_sha256: str, catalog_sha256: str, backend: dict[str, object]) -> None:
    required = {"schema", "context_sha256", "catalog_sha256", "condition_id", "dataset", "raw_identity", "backend", "payload", "payload_sha256"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != "exp450-dataset-condition-success-antigravity.v1" or value["dataset"] != entry["dataset"] or value["condition_id"] != condition_id or value["raw_identity"] != entry or value["context_sha256"] != context_sha256 or value["catalog_sha256"] != catalog_sha256 or value["backend"] != backend:
        raise ValueError("score root checkpoint binding differs")
    payload = value["payload"]
    if not isinstance(payload, dict) or set(payload) != {"sources", "route"} or value["payload_sha256"] != _digest(payload):
        raise ValueError("score root checkpoint payload differs")
    n_train, n_test = entry["train"]["count"], entry["test"]["count"]
    _check_sources(payload["sources"], n_train, n_test)
    _check_route(payload["route"], n_test)


def _numbers(value: object, size: int, *, pvalue: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != size or any(type(item) not in {int, float} for item in value):
        raise ValueError("score root checkpoint payload differs")
    values = [float(item) for item in value]
    if not all(np.isfinite(values)) or (pvalue and any(item <= 0.0 or item > 1.0 for item in values)):
        raise ValueError("score root checkpoint payload differs")
    return values


def _indices(value: object, size: int) -> set[int]:
    if not isinstance(value, list) or value != sorted(set(value)) or any(type(item) is not int or item < 0 or item >= size for item in value):
        raise ValueError("score root checkpoint payload differs")
    return set(value)


def _lanes(value: object, size: int) -> tuple[set[int], set[int], set[int]]:
    if not isinstance(value, dict) or set(value) != {"hard", "standard", "priority"}:
        raise ValueError("score root checkpoint payload differs")
    lanes = tuple(_indices(value[key], size) for key in ("hard", "standard", "priority"))
    if lanes[0] & lanes[1] or lanes[0] & lanes[2] or lanes[1] & lanes[2]:
        raise ValueError("score root checkpoint payload differs")
    return lanes


def _check_sources(value: object, n_train: int, n_test: int) -> None:
    required = {"source_id", "order", "origin", "train_oof_score", "test_score", "test_pvalue", "threshold", "test_native_pass", "eligible", "eligibility_reason"}
    if not isinstance(value, list) or len(value) != len(_SOURCES):
        raise ValueError("score root checkpoint payload differs")
    for order, row in enumerate(value):
        origin = row.get("origin") if isinstance(row, dict) else None
        if not isinstance(row, dict) or set(row) != required or row["source_id"] != _SOURCES[order] or row["order"] != order or type(row["threshold"]) not in {int, float} or row["threshold"] != _ALPHAS[_SOURCES[order]] or type(row["eligible"]) is not bool or type(row["eligibility_reason"]) is not str or not row["eligibility_reason"] or not isinstance(origin, dict) or set(origin) != {"node_id", "stage", "input_kind", "module", "symbols", "source_sha256", "paper_module", "label_access", "parity_oracle"} or origin["node_id"] != _ORIGIN_IDS[order] or origin["source_sha256"] != _ORIGIN_SHA[order] or origin["label_access"] is not False or not all(isinstance(origin[key], str) and origin[key] for key in ("stage", "input_kind", "module", "paper_module", "parity_oracle")) or not isinstance(origin["symbols"], list) or not all(isinstance(item, str) and item for item in origin["symbols"]):
            raise ValueError("score root checkpoint payload differs")
        pvalues = _numbers(row["test_pvalue"], n_test, pvalue=True)
        native = row["test_native_pass"]
        if not isinstance(native, list) or len(native) != n_test or any(type(item) is not bool for item in native):
            raise ValueError("score root checkpoint payload differs")
        if row["eligible"]:
            train, test = _numbers(row["train_oof_score"], n_train), _numbers(row["test_score"], n_test)
            expected = [(1 + sum(item >= query for item in train)) / (len(train) + 1) for query in test]
            if row["eligibility_reason"] != "eligible" or pvalues != expected or native != [item <= row["threshold"] for item in pvalues]:
                raise ValueError("score root checkpoint payload differs")
        elif row["train_oof_score"] != [] or row["test_score"] != [] or pvalues != [1.0] * n_test or any(native):
            raise ValueError("score root checkpoint payload differs")


def _check_route(value: object, size: int) -> None:
    required = {"n_test", "exp74d", "exp89", "exp90", "exp93", "exp137", "exp234", "exp236", "promoted", "exp253", "exact_normal_removed", "automatic_hard"}
    if not isinstance(value, dict) or set(value) != required or value["n_test"] != size:
        raise ValueError("score root checkpoint payload differs")
    _lanes(value["exp74d"], size); _indices(value["exp89"], size); _indices(value["exp90"], size); _indices(value["exp93"], size); _lanes(value["exp137"], size)
    if not isinstance(value["exp234"], dict) or set(value["exp234"]) != {"minimum_two", "minimum_three"}:
        raise ValueError("score root checkpoint payload differs")
    two, three = _indices(value["exp234"]["minimum_two"], size), _indices(value["exp234"]["minimum_three"], size)
    if not three <= two: raise ValueError("score root checkpoint payload differs")
    hard236, standard236, priority236 = _lanes(value["exp236"], size)
    promoted = _indices(value["promoted"], size)
    hard253, standard253, priority253 = _lanes(value["exp253"], size)
    removed, automatic = _indices(value["exact_normal_removed"], size), _indices(value["automatic_hard"], size)
    if not promoted <= standard236 or hard253 != hard236 | promoted or standard253 != standard236 - promoted or priority253 != priority236 or not removed <= hard253 or automatic != hard253 - removed:
        raise ValueError("score root checkpoint payload differs")


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _check_shard(
    root: Path,
    condition: dict[str, object],
    context_sha256: str,
    catalog_sha256: str,
    backend: dict[str, object],
    entry: dict[str, object],
    descriptor: dict[str, object],
) -> None:
    dataset = entry["dataset"]
    token = sha256(dataset.encode("utf-8")).hexdigest()
    expected_file = f"score_roots_antigravity/{condition['condition_id']}_antigravity/{token}_antigravity/{_SHARD_NAME}"
    if descriptor.get("dataset") != dataset or descriptor.get("file") != expected_file:
        raise ValueError("score root shard inventory differs")
    path = root.joinpath(*expected_file.split("/"))
    if _directory_children(path.parent) != {_SHARD_NAME}:
        raise ValueError("score root shard inventory differs")
    value, data = _read(path)
    if descriptor.get("sha256") != sha256(data).hexdigest():
        raise ValueError("score root shard hash differs")
    required = {"schema", "context_sha256", "catalog_sha256", "condition", "backend", "dataset", "test_identity", "policy_order", "policies", "shard_sha256"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != _SHARD_SCHEMA:
        raise ValueError("score root shard schema differs")
    bare = {key: item for key, item in value.items() if key != "shard_sha256"}
    if value["shard_sha256"] != _digest(bare):
        raise ValueError("score root shard payload hash differs")
    if value["context_sha256"] != context_sha256 or value["catalog_sha256"] != catalog_sha256 or value["condition"] != condition or value["backend"] != backend or value["dataset"] != dataset or value["test_identity"] != entry.get("test"):
        raise ValueError("score root shard binding differs")
    if value["policy_order"] != list(_POLICY_ORDER) or not isinstance(value["policies"], list):
        raise ValueError("score root policy order differs")
    policies = value["policies"]
    if [item.get("policy_id") if isinstance(item, dict) else None for item in policies] != list(_POLICY_ORDER):
        raise ValueError("score root policy order differs")
    size = entry.get("test", {}).get("count") if isinstance(entry.get("test"), dict) else None
    if type(size) is not int or size < 0:
        raise ValueError("score root TEST identity differs")
    for policy in policies:
        if not isinstance(policy, dict) or set(policy) != {"policy_id", "alerts", "score_dtype", "scores", "score_mask"} or policy["score_dtype"] != "float64":
            raise ValueError("score root policy schema differs")
        scores = _json_float_vector(policy["scores"], size)
        mask = np.asarray(policy["score_mask"])
        alerts = policy["alerts"]
        if scores.ndim != 1 or scores.size != size or not np.all(np.isfinite(scores)):
            raise ValueError("score root continuous score differs")
        if mask.dtype != np.dtype(bool) or mask.ndim != 1 or mask.size != size:
            raise ValueError("score root score mask differs")
        if not isinstance(alerts, list) or alerts != sorted(set(alerts)) or any(type(index) is not int or index < 0 or index >= size for index in alerts):
            raise ValueError("score root alerts differ")
        threshold_alerts = np.flatnonzero(scores >= 0.5).tolist()
        if alerts != threshold_alerts or not np.array_equal(mask, scores >= 0.5):
            raise ValueError("score root alert threshold differs")


def verify_exp450_score_root_antigravity(
    root: Path, *, context: object, catalog: dict[str, object], condition_id: str
) -> dict[str, object]:
    """Verify one frozen condition root without importing score or label science."""
    if condition_id not in _CONDITION_IDS:
        raise ValueError("score root condition differs")
    root = _root(context, Path(root))
    context_sha256 = _context_sha256(context)
    backend = _backend(context)
    entries = _catalog(catalog, context)
    manifest_path = root / "score_roots_antigravity" / f"{condition_id}_antigravity" / _MANIFEST_NAME
    condition_root = manifest_path.parent
    expected_children = {_MANIFEST_NAME} | {
        f"{sha256(entry['dataset'].encode('utf-8')).hexdigest()}_antigravity"
        for entry in entries
    }
    if _directory_children(condition_root) != expected_children:
        raise ValueError("score root inventory differs")
    manifest, _data = _read(manifest_path)
    required = {"schema", "context_sha256", "catalog_sha256", "condition", "backend", "datasets", "checkpoint_hashes", "shards", "manifest_sha256"}
    if not isinstance(manifest, dict) or set(manifest) != required or manifest["schema"] != _MANIFEST_SCHEMA:
        raise ValueError("score root manifest schema differs")
    bare = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    if manifest["manifest_sha256"] != _digest(bare):
        raise ValueError("score root manifest hash differs")
    condition = manifest["condition"]
    if not isinstance(condition, dict) or set(condition) != {"condition_id", "length_mode", "image_mode"} or condition["condition_id"] != condition_id or (condition["length_mode"], condition["image_mode"]) != _CONDITIONS[condition_id]:
        raise ValueError("score root condition differs")
    if manifest["context_sha256"] != context_sha256 or manifest["catalog_sha256"] != catalog["catalog_sha256"] or manifest["backend"] != backend or manifest["datasets"] != entries:
        raise ValueError("score root manifest binding differs")
    expected_datasets = [entry["dataset"] for entry in entries]
    checkpoint_hashes = manifest["checkpoint_hashes"]
    if not isinstance(checkpoint_hashes, list) or [item.get("dataset") if isinstance(item, dict) else None for item in checkpoint_hashes] != expected_datasets or any(not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64 for item in checkpoint_hashes):
        raise ValueError("score root checkpoint hashes differ")
    for entry, descriptor in zip(entries, checkpoint_hashes, strict=True):
        dataset = entry["dataset"]
        token = sha256(dataset.encode("utf-8")).hexdigest()
        checkpoint = root / "checkpoints_antigravity" / f"{condition_id}_antigravity" / f"{token}_antigravity" / "dataset_condition_success_antigravity.json"
        checkpoint_value, checkpoint_data = _read(checkpoint)
        if sha256(checkpoint_data).hexdigest() != descriptor["sha256"]:
            raise ValueError("score root checkpoint hash differs")
        _check_checkpoint(checkpoint_value, entry, condition_id, context_sha256, catalog["catalog_sha256"], backend)
    shards = manifest["shards"]
    if not isinstance(shards, list) or len(shards) != len(entries):
        raise ValueError("score root shard inventory differs")
    if [item.get("dataset") if isinstance(item, dict) else None for item in shards] != expected_datasets:
        raise ValueError("score root shard inventory differs")
    for entry, descriptor in zip(entries, shards, strict=True):
        _check_shard(root, condition, context_sha256, catalog["catalog_sha256"], backend, entry, descriptor)
    return manifest


__all__ = ["verify_exp450_score_root_antigravity"]
