"""Path-only, read-only Exp471 authority for Exp472 evaluation.

Exp471 exposes public source/cache/shard readers.  It has no public
label-free evaluation/report reader, so this module locally verifies those
canonical, no-symlink JSON leaves without changing Exp471.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType

from .contracts_antigravity import canonical_json_bytes_exp472_antigravity

_SHA = re.compile(r"^[0-9a-f]{64}$")
_METHODS = ("A", "B", "C", "D", "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d")
_REPORT_MARKER = ".exp471_report_complete_antigravity.json"
_REPORT_ARTIFACT_MAP = "artifact_sha256_antigravity.json"
_TYPED_REPORT_ROOT = "exp471_typed_report_artifacts_antigravity"
_TYPED_REPORT_MANIFEST = "exp471_typed_report_manifest_antigravity.json"

def _digest(value: object, name: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None: raise ValueError(f"Exp472 {name} digest differs")
    return value

def _coordinate(value: object) -> Mapping[str, object]:
    """Project the shared catalog/Task4 TEST identity, rejecting unknown fields."""
    allowed = {"dataset", "dataset_id", "series_length", "split", "indices", "values_blob_sha256", "row_digest", "count"}
    if not isinstance(value, Mapping) or not {"dataset", "series_length", "split", "indices", "values_blob_sha256", "row_digest", "count"}.issubset(value) or set(value) - allowed: raise ValueError("Exp472 TEST coordinate differs")
    if type(value["dataset"]) is not str or not value["dataset"] or value["split"] != "TEST" or type(value["series_length"]) is not int or value["series_length"] < 1: raise ValueError("Exp472 TEST coordinate differs")
    count, indices, blobs = value["count"], value["indices"], value["values_blob_sha256"]
    if type(count) is not int or count < 1 or not isinstance(indices, (tuple, list)) or len(indices) != count or any(type(x) is not int or x < 0 for x in indices) or len(set(indices)) != count: raise ValueError("Exp472 TEST coordinate differs")
    if not isinstance(blobs, (tuple, list)) or len(blobs) != count: raise ValueError("Exp472 TEST blob coordinate differs")
    return MappingProxyType({"dataset": value["dataset"], "series_length": value["series_length"], "split": "TEST", "count": count, "indices": tuple(indices), "values_blob_sha256": tuple(_digest(blob, "blob") for blob in blobs), "row_digest": _digest(value["row_digest"], "row")})

def _project_catalog_test_identity_antigravity(value: object) -> Mapping[str, object]:
    """Project Exp467's dataset-id-bearing TEST identity into Task4's schema."""
    coordinate = _coordinate(value)
    bare = {
        "dataset": coordinate["dataset"], "series_length": coordinate["series_length"],
        "split": "TEST", "indices": list(coordinate["indices"]),
        "values_blob_sha256": list(coordinate["values_blob_sha256"]),
    }
    return MappingProxyType({**dict(coordinate), "row_digest": sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()})

def _coordinate_digest(order: tuple[str, ...], coordinates: Mapping[str, Mapping[str, object]]) -> str:
    return sha256(canonical_json_bytes_exp472_antigravity([{**dict(_coordinate(coordinates[d])), "indices": list(_coordinate(coordinates[d])["indices"]), "values_blob_sha256": list(_coordinate(coordinates[d])["values_blob_sha256"])} for d in order])).hexdigest()

@dataclass(frozen=True, slots=True)
class Exp472ReuseAuthorityAntigravity:
    exp471_source_authority_sha256: str; exp471_cache_manifest_sha256: str; exp471_evaluation_sha256: str; exp471_report_manifest_sha256: str; database_sha256: str; coordinate_inventory_sha256: str; method_order: tuple[str, ...]; total: int; scored: int; abstained: int; failed: int; dataset_count: int; common_support: int
    def __post_init__(self) -> None:
        for name in ("exp471_source_authority_sha256", "exp471_cache_manifest_sha256", "exp471_evaluation_sha256", "exp471_report_manifest_sha256", "database_sha256", "coordinate_inventory_sha256"): _digest(getattr(self, name), name)
        if tuple(self.method_order) != _METHODS: raise ValueError("Exp472 method order differs")
        if (self.total, self.scored, self.abstained, self.failed) != (11170, 10775, 395, 0): raise ValueError("Exp472 terminal counts differ")
        if (self.dataset_count, self.common_support) != (1117, 922): raise ValueError("Exp472 dataset support differs")

@dataclass(frozen=True, slots=True)
class Exp472VerifiedExp471BundleAntigravity:
    root: Path; authority: Exp472ReuseAuthorityAntigravity; dataset_order: tuple[str, ...]; coordinates: Mapping[str, Mapping[str, object]]; scores: Mapping[str, Mapping[str, tuple[float, ...] | None]]; statuses: Mapping[str, Mapping[str, str]]; path_sha256: str
    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not self.root.is_absolute() or len(self.dataset_order) != 1117 or len(set(self.dataset_order)) != 1117: raise ValueError("Exp472 Exp471 dataset inventory differs")
        if self.authority.dataset_count != len(self.dataset_order) or tuple(self.coordinates) != self.dataset_order: raise ValueError("Exp472 Exp471 coordinate inventory differs")
        coords = MappingProxyType({d: _coordinate(self.coordinates[d]) for d in self.dataset_order})
        if _coordinate_digest(self.dataset_order, coords) != self.authority.coordinate_inventory_sha256: raise ValueError("Exp472 Exp471 coordinate digest differs")
        if tuple(self.scores) != _METHODS or tuple(self.statuses) != _METHODS: raise ValueError("Exp472 Exp471 score inventory differs")
        scored = abstained = 0
        for method in _METHODS:
            if tuple(self.scores[method]) != self.dataset_order or tuple(self.statuses[method]) != self.dataset_order: raise ValueError("Exp472 Exp471 score order differs")
            for dataset in self.dataset_order:
                status, values = self.statuses[method][dataset], self.scores[method][dataset]
                if status == "scored":
                    if not isinstance(values, tuple) or len(values) != coords[dataset]["count"] or not all(type(x) in (int,float) and float("-inf") < float(x) < float("inf") for x in values): raise ValueError("Exp472 scored shard differs")
                    scored += 1
                elif status == "abstained":
                    if values is not None: raise ValueError("Exp472 typed abstention differs")
                    abstained += 1
                else: raise ValueError("Exp472 shard status differs")
        if (scored, abstained) != (10775, 395) or sum(all(self.statuses[m][d] == "scored" for m in _METHODS) for d in self.dataset_order) != 922: raise ValueError("Exp472 Exp471 live inventory differs")
        _digest(self.path_sha256, "path")
        object.__setattr__(self, "coordinates", coords)

def _safe_relative_report_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise ValueError("Exp472 frozen report path is unsafe")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Exp472 frozen report path is unsafe")
    return value

def _safe_report_directory(root: Path) -> None:
    details = root.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("Exp472 frozen report directory is unsafe")

def _regular_report_leaf_bytes(root: Path, relative: object) -> bytes:
    """Read one report leaf through no-symlink, regular-file path components."""
    safe = _safe_relative_report_path(relative)
    _safe_report_directory(root)
    parent = root
    for component in safe.split("/")[:-1]:
        parent = parent / component
        _safe_report_directory(parent)
    path = root.joinpath(*safe.split("/"))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("Exp472 frozen report leaf is unsafe") from error
    try:
        opened, linked = os.fstat(descriptor), path.lstat()
        if (not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(linked.st_mode)
                or opened.st_dev != linked.st_dev or opened.st_ino != linked.st_ino
                or opened.st_nlink != 1):
            raise ValueError("Exp472 frozen report leaf is unsafe")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(descriptor)

def _canonical_report_json(root: Path, relative: object) -> tuple[Mapping[str, object], bytes]:
    from experiments.exp471_exp467_evaluation_recovery_antigravity.contracts_antigravity import canonical_json_bytes_exp471_antigravity
    raw = _regular_report_leaf_bytes(root, relative)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp472 frozen report JSON is malformed") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes_exp471_antigravity(value):
        raise ValueError("Exp472 frozen report JSON is noncanonical")
    return MappingProxyType(value), raw

def _report_leaf_names(root: Path) -> tuple[str, ...]:
    _safe_report_directory(root)
    names: list[str] = []
    for directory, subdirectories, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        _safe_report_directory(directory_path)
        for name in subdirectories:
            _safe_report_directory(directory_path / name)
        for name in files:
            relative = (directory_path / name).relative_to(root).as_posix()
            _regular_report_leaf_bytes(root, relative)
            names.append(relative)
    if len(names) != len(set(names)):
        raise ValueError("Exp472 frozen report leaf inventory differs")
    return tuple(sorted(names))

def _verified_exp471_report_digest_antigravity(
    root: Path, *, source_authority_sha256: str, cache_manifest_sha256: str,
    evaluation_sha256: str,
) -> str:
    """Verify Exp471's published marker -> artifact-map -> typed-manifest chain."""
    report_root = root / "report_antigravity"
    marker, _ = _canonical_report_json(report_root, _REPORT_MARKER)
    required_marker = {
        "schema", "source_authority_sha256", "cache_manifest_sha256", "evaluation_sha256",
        "report_file_sha256", "report_manifest_sha256", "artifact_map",
    }
    if (set(marker) != required_marker or marker.get("schema") != "exp471-report-publication-antigravity.v2"
            or marker.get("artifact_map") != _REPORT_ARTIFACT_MAP
            or marker.get("source_authority_sha256") != source_authority_sha256
            or marker.get("cache_manifest_sha256") != cache_manifest_sha256
            or marker.get("evaluation_sha256") != evaluation_sha256
            or not isinstance(marker.get("report_file_sha256"), Mapping)):
        raise ValueError("Exp472 Exp471 report complete marker differs")
    expected_files = dict(marker["report_file_sha256"])
    for relative, digest in expected_files.items():
        _safe_relative_report_path(relative); _digest(digest, "report leaf")
    artifact_map, artifact_map_raw = _canonical_report_json(report_root, _REPORT_ARTIFACT_MAP)
    if (set(artifact_map) != {"schema", "self_excluded", "files"}
            or artifact_map.get("schema") != "exp471-report-artifact-map-antigravity.v2"
            or artifact_map.get("self_excluded") != _REPORT_ARTIFACT_MAP
            or not isinstance(artifact_map.get("files"), Mapping)
            or dict(artifact_map["files"]) != expected_files
            or sha256(artifact_map_raw).hexdigest() != _digest(marker.get("report_manifest_sha256"), "report manifest")):
        raise ValueError("Exp472 Exp471 report artifact map digest differs")
    expected_names = set(expected_files) | {_REPORT_MARKER, _REPORT_ARTIFACT_MAP}
    if set(_report_leaf_names(report_root)) != expected_names:
        raise ValueError("Exp472 Exp471 report leaf inventory differs")
    for relative, expected in expected_files.items():
        if sha256(_regular_report_leaf_bytes(report_root, relative)).hexdigest() != expected:
            raise ValueError("Exp472 Exp471 report leaf digest differs")
    typed_relative = f"{_TYPED_REPORT_ROOT}/{_TYPED_REPORT_MANIFEST}"
    if typed_relative not in expected_files:
        raise ValueError("Exp472 Exp471 typed report manifest is absent")
    typed_manifest, _ = _canonical_report_json(report_root, typed_relative)
    if (set(typed_manifest) != {"schema", "files"}
            or typed_manifest.get("schema") != "exp471-typed-report-manifest-antigravity.v1"
            or not isinstance(typed_manifest.get("files"), Mapping)):
        raise ValueError("Exp472 Exp471 typed report manifest differs")
    typed_files = dict(typed_manifest["files"])
    expected_typed_files = {
        relative.removeprefix(f"{_TYPED_REPORT_ROOT}/"): digest
        for relative, digest in expected_files.items()
        if relative.startswith(f"{_TYPED_REPORT_ROOT}/") and relative != typed_relative
    }
    if typed_files != expected_typed_files:
        raise ValueError("Exp472 Exp471 typed report manifest differs")
    for relative, digest in typed_files.items():
        _safe_relative_report_path(relative); _digest(digest, "typed report leaf")
    return _digest(marker["report_manifest_sha256"], "report manifest")

def _canonical_evaluation_leaf(root: Path) -> Mapping[str, object]:
    from experiments.exp471_exp467_evaluation_recovery_antigravity.contracts_antigravity import canonical_json_bytes_exp471_antigravity
    value, _ = _canonical_report_json(root, "evaluation_antigravity/evaluation_antigravity.json")
    digest = value.get("evaluation_sha256")
    bare = {key: item for key, item in value.items() if key != "evaluation_sha256"}
    if _digest(digest, "evaluation") != sha256(canonical_json_bytes_exp471_antigravity(bare)).hexdigest():
        raise ValueError("Exp472 frozen evaluation digest differs")
    return value

def _read_verified_exp471_path_antigravity(root: Path) -> Exp472VerifiedExp471BundleAntigravity:
    """Production reader: public Exp471 source/cache/shard APIs plus local JSON leaf checks."""
    from experiments.exp471_exp467_evaluation_recovery_antigravity import cache_artifacts_antigravity as cache, source_authority_antigravity as source
    from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import EXP467_SOURCE_ROOT_ANTIGRAVITY
    root = Path(root)
    if not root.is_absolute(): raise ValueError("Exp472 Exp471 root must be absolute")
    repository = Path(__file__).resolve().parents[2]
    session = source.open_exp467_source_session_antigravity(repository, EXP467_SOURCE_ROOT_ANTIGRAVITY)
    source_authority = source.read_source_authority_antigravity(root, repository, EXP467_SOURCE_ROOT_ANTIGRAVITY)
    if session.authority != source_authority or not isinstance(session.catalog, Mapping): raise ValueError("Exp472 Exp471 source session differs")
    manifest = cache.read_cache_manifest_antigravity(root / "score_cache_antigravity")
    evaluation = _canonical_evaluation_leaf(root)
    if evaluation.get("source_authority_sha256") != source_authority.authority_sha256 or evaluation.get("cache_manifest_sha256") != manifest.manifest_sha256: raise ValueError("Exp472 Exp471 evaluation digest chain differs")
    report_manifest_sha256 = _verified_exp471_report_digest_antigravity(root, source_authority_sha256=source_authority.authority_sha256, cache_manifest_sha256=manifest.manifest_sha256, evaluation_sha256=str(evaluation["evaluation_sha256"]))
    catalog_rows = session.catalog.get("datasets")
    if not isinstance(catalog_rows, (tuple, list)) or len(catalog_rows) != 1117: raise ValueError("Exp472 Exp471 catalog differs")
    catalog_coordinates = {row.get("dataset"): _project_catalog_test_identity_antigravity(row.get("test")) for row in catalog_rows if isinstance(row, Mapping)}
    coordinates, scores, statuses = {}, {m:{} for m in _METHODS}, {m:{} for m in _METHODS}
    for dataset in manifest.dataset_order:
        first = cache.read_score_shard_antigravity(root / "score_cache_antigravity", "A", dataset)
        coordinate = catalog_coordinates.get(dataset)
        if coordinate is None or first.test_count != coordinate["count"] or first.test_indices != coordinate["indices"]: raise ValueError("Exp472 Exp471 catalog TEST identity differs")
        coordinates[dataset] = coordinate
        for method in _METHODS:
            shard = cache.read_score_shard_antigravity(root / "score_cache_antigravity", method, dataset)
            if shard.test_count != first.test_count or shard.test_indices != first.test_indices: raise ValueError("Exp472 Exp471 TEST coordinate differs")
            statuses[method][dataset] = shard.status; scores[method][dataset] = tuple(shard.raw_scores) if shard.status == "scored" else None
    order = tuple(manifest.dataset_order); coordinate_sha = _coordinate_digest(order, coordinates)
    common = sum(all(statuses[m][d] == "scored" for m in _METHODS) for d in order)
    authority = Exp472ReuseAuthorityAntigravity(source_authority.authority_sha256, manifest.manifest_sha256, str(evaluation["evaluation_sha256"]), report_manifest_sha256, source_authority.database_sha256, coordinate_sha, tuple(manifest.method_order), manifest.total_shards, manifest.scored_shards, manifest.abstained_shards, manifest.failed_shards, len(order), common)
    return Exp472VerifiedExp471BundleAntigravity(root, authority, order, coordinates, scores, statuses, sha256(root.as_posix().encode() + bytes.fromhex(coordinate_sha)).hexdigest())

def verify_exp471_reuse_exp472_antigravity(root: Path) -> Exp472VerifiedExp471BundleAntigravity:
    if not isinstance(root, Path): raise ValueError("Exp472 Exp471 authority requires a Path")
    return _read_verified_exp471_path_antigravity(root)

def bind_e_to_exp471_exp472_antigravity(evidence: object, bundle: Exp472VerifiedExp471BundleAntigravity) -> tuple[str, ...]:
    if not isinstance(bundle, Exp472VerifiedExp471BundleAntigravity): raise ValueError("Exp472 authority bundle differs")
    if getattr(evidence, "database_sha256", None) != bundle.authority.database_sha256 or tuple(getattr(evidence, "dataset_order", ())) != bundle.dataset_order: raise ValueError("Exp472 E database or dataset order differs")
    for dataset in bundle.dataset_order:
        if _coordinate(evidence.coordinates[dataset]) != bundle.coordinates[dataset]: raise ValueError("Exp472 E-vs-D TEST coordinate differs")
    return bundle.dataset_order

__all__ = ["Exp472ReuseAuthorityAntigravity", "Exp472VerifiedExp471BundleAntigravity", "bind_e_to_exp471_exp472_antigravity", "verify_exp471_reuse_exp472_antigravity"]
