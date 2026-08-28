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
from pathlib import Path
import re
import stat
from types import MappingProxyType

from .contracts_antigravity import canonical_json_bytes_exp472_antigravity

_SHA = re.compile(r"^[0-9a-f]{64}$")
_METHODS = ("A", "B", "C", "D", "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d")

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

def _canonical_leaf(path: Path, expected_digest_field: str) -> Mapping[str, object]:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode): raise ValueError("Exp472 frozen leaf is unsafe")
    raw = path.read_bytes(); value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_json_bytes_exp472_antigravity(value): raise ValueError("Exp472 frozen leaf is noncanonical")
    digest = value.get(expected_digest_field); bare = {k:v for k,v in value.items() if k != expected_digest_field}
    if _digest(digest, expected_digest_field) != sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest(): raise ValueError("Exp472 frozen leaf digest differs")
    return MappingProxyType(value)

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
    evaluation = _canonical_leaf(root / "evaluation_antigravity" / "evaluation_antigravity.json", "evaluation_sha256")
    report = _canonical_leaf(root / "report_antigravity" / "exp471_typed_report_manifest_antigravity.json", "report_manifest_sha256")
    if evaluation.get("source_authority_sha256") != source_authority.authority_sha256 or evaluation.get("cache_manifest_sha256") != manifest.manifest_sha256 or report.get("evaluation_sha256") != evaluation["evaluation_sha256"]: raise ValueError("Exp472 Exp471 digest chain differs")
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
    authority = Exp472ReuseAuthorityAntigravity(source_authority.authority_sha256, manifest.manifest_sha256, str(evaluation["evaluation_sha256"]), str(report["report_manifest_sha256"]), source_authority.database_sha256, coordinate_sha, tuple(manifest.method_order), manifest.total_shards, manifest.scored_shards, manifest.abstained_shards, manifest.failed_shards, len(order), common)
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
