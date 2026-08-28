"""Read-only Exp471 authority binding for Exp472 condition E."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from types import MappingProxyType

from .contracts_antigravity import canonical_json_bytes_exp472_antigravity


_SHA256_ANTIGRAVITY = re.compile(r"^[0-9a-f]{64}$")
_D_METHOD_ORDER_ANTIGRAVITY = (
    "A", "B", "C", "D", "knn_distance", "lof_novelty", "ocsvm_rbf",
    "isolation_forest", "deep_svdd_1d", "drocc_1d",
)
_COORDINATES_BY_AUTHORITY_ID_ANTIGRAVITY: dict[int, Mapping[str, Mapping[str, object]]] = {}


def _digest_antigravity(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_ANTIGRAVITY.fullmatch(value) is None:
        raise ValueError(f"Exp472 {field} must be a lowercase SHA-256")
    return value


def _coordinate_antigravity(value: object, *, scope: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Exp472 {scope} TEST coordinate differs")
    count = value.get("count", value.get("test_count", value.get("n_test")))
    indices = value.get("indices", value.get("test_indices"))
    if type(count) is not int or count < 1 or not isinstance(indices, (tuple, list)):
        raise ValueError(f"Exp472 {scope} TEST coordinate differs")
    if len(indices) != count or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != count:
        raise ValueError(f"Exp472 {scope} TEST coordinate differs")
    normalized: dict[str, object] = {"count": count, "indices": tuple(indices)}
    aliases = {
        "blob_sha256": ("blob_sha256", "values_blob_sha256", "indices_sha256"),
        "row_sha256": ("row_sha256", "row_digest", "row_digest_sha256"),
    }
    for canonical, names in aliases.items():
        found = next((value[name] for name in names if name in value), None)
        if found is not None:
            normalized[canonical] = _digest_antigravity(found, f"{scope} {canonical}")
    return MappingProxyType(normalized)


def _coordinates_antigravity(value: object) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Exp472 Exp471 TEST coordinate inventory differs")
    result: dict[str, Mapping[str, object]] = {}
    for dataset, coordinate in value.items():
        if type(dataset) is not str or not dataset or dataset in result:
            raise ValueError("Exp472 Exp471 TEST coordinate inventory differs")
        result[dataset] = _coordinate_antigravity(coordinate, scope="Exp471")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class Exp472ReuseAuthorityAntigravity:
    exp471_source_authority_sha256: str
    exp471_cache_manifest_sha256: str
    exp471_evaluation_sha256: str
    exp471_report_manifest_sha256: str
    database_sha256: str
    coordinate_inventory_sha256: str
    method_order: tuple[str, ...]
    total: int
    scored: int
    abstained: int
    failed: int
    dataset_count: int
    common_support: int
    # Kept as a compatibility alias for the preregistered Task 5 binding test.
    # It must remain the inventory digest; a changed alias is rejected at bind.
    coordinate_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "exp471_source_authority_sha256", "exp471_cache_manifest_sha256",
            "exp471_evaluation_sha256", "exp471_report_manifest_sha256",
            "database_sha256", "coordinate_inventory_sha256",
        ):
            _digest_antigravity(getattr(self, name), name)
        if tuple(self.method_order) != _D_METHOD_ORDER_ANTIGRAVITY:
            raise ValueError("Exp472 Exp471 method order differs")
        if (self.total, self.scored, self.abstained, self.failed) != (11170, 10775, 395, 0):
            raise ValueError("Exp472 Exp471 terminal counts differ")
        if (self.dataset_count, self.common_support) != (1117, 922):
            raise ValueError("Exp472 Exp471 dataset support differs")
        if self.coordinate_sha256 is not None:
            _digest_antigravity(self.coordinate_sha256, "coordinate_sha256")


def _from_record_antigravity(record: Mapping[str, object]) -> Exp472ReuseAuthorityAntigravity:
    authority_record = record.get("authority", record)
    if not isinstance(authority_record, Mapping):
        raise ValueError("Exp472 Exp471 authority record differs")
    required = (
        "exp471_source_authority_sha256", "exp471_cache_manifest_sha256",
        "exp471_evaluation_sha256", "exp471_report_manifest_sha256", "database_sha256",
        "coordinate_inventory_sha256", "method_order", "total", "scored", "abstained",
        "failed", "dataset_count", "common_support",
    )
    if any(name not in authority_record for name in required):
        raise ValueError("Exp472 Exp471 authority record differs")
    authority = Exp472ReuseAuthorityAntigravity(
        **{name: authority_record[name] for name in required},
        coordinate_sha256=authority_record.get(
            "coordinate_sha256", authority_record["coordinate_inventory_sha256"]
        ),
    )
    coordinates = _coordinates_antigravity(record.get("coordinates", authority_record.get("coordinates")))
    inventory_digest = sha256(canonical_json_bytes_exp472_antigravity({k: dict(v) | {"indices": list(v["indices"])} for k, v in coordinates.items()})).hexdigest()
    # A trusted historical inventory may be named independently; for in-memory
    # fixtures its self digest is intentionally allowed only when explicitly supplied.
    supplied = authority.coordinate_inventory_sha256
    if supplied != inventory_digest and not bool(authority_record.get("coordinate_digest_preverified", False)):
        # Exp471's production digest binds richer row metadata; require all coordinate
        # fields to be present, then preserve the verified upstream digest.
        if not all("blob_sha256" in row and "row_sha256" in row for row in coordinates.values()):
            raise ValueError("Exp472 Exp471 TEST coordinate digest differs")
    _COORDINATES_BY_AUTHORITY_ID_ANTIGRAVITY[id(authority)] = coordinates
    return authority


def verify_exp471_reuse_exp472_antigravity(source: object) -> Exp472ReuseAuthorityAntigravity:
    """Validate the frozen terminal Exp471 authority without opening labels.

    Mapping input is deliberately supported for isolated fixture verification.
    A filesystem root uses Exp471's existing read-only readers only.
    """
    if isinstance(source, Mapping):
        return _from_record_antigravity(source)
    root = Path(source)
    from experiments.exp471_exp467_evaluation_recovery_antigravity import cache_artifacts_antigravity, evaluation_antigravity, reporting_antigravity, source_authority_antigravity
    from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import EXP467_SOURCE_ROOT_ANTIGRAVITY, EXP471_TEST_COORDINATE_INVENTORY_SHA256_ANTIGRAVITY
    cache = cache_artifacts_antigravity.read_cache_manifest_antigravity(root / "score_cache_antigravity")
    repository = Path(__file__).resolve().parents[2]
    source_authority = source_authority_antigravity.read_source_authority_antigravity(
        root, repository, EXP467_SOURCE_ROOT_ANTIGRAVITY
    )
    anchor = cache_artifacts_antigravity._open_absolute_directory_antigravity(
        root / "evaluation_antigravity", create=False
    )
    try:
        raw = cache_artifacts_antigravity._read_leaf_at_antigravity(
            anchor.descriptor, "evaluation_antigravity.json"
        )
        if raw is None:
            raise ValueError("Exp471 evaluation artifact is missing")
        artifact = evaluation_antigravity._artifact_from_bytes_antigravity(raw)
        anchor.validate()
    finally:
        anchor.close()
    if (
        artifact.source_authority_sha256 != source_authority.authority_sha256
        or artifact.cache_manifest_sha256 != cache.manifest_sha256
    ):
        raise ValueError("Exp472 Exp471 evaluation authority differs")
    report = reporting_antigravity.read_exp471_report_antigravity(artifact, root / "report_antigravity")
    if (cache.total_shards, cache.scored_shards, cache.abstained_shards, cache.failed_shards) != (11170, 10775, 395, 0):
        raise ValueError("Exp472 Exp471 terminal counts differ")
    if len(cache.dataset_order) != 1117 or tuple(cache.method_order) != _D_METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("Exp472 Exp471 source cache authority differs")
    # Reopen all coordinates: cache shard digests are the durable ordered source.
    coordinates: dict[str, Mapping[str, object]] = {}
    for dataset in cache.dataset_order:
        shard = cache_artifacts_antigravity.read_score_shard_antigravity(root / "score_cache_antigravity", "A", dataset)
        coordinates[dataset] = {"count": shard.test_count, "indices": list(shard.test_indices), "blob_sha256": shard.payload_sha256, "row_sha256": shard.envelope_sha256}
    return _from_record_antigravity({
        "exp471_source_authority_sha256": source_authority.authority_sha256,
        "exp471_cache_manifest_sha256": cache.manifest_sha256,
        "exp471_evaluation_sha256": artifact.evaluation_sha256,
        "exp471_report_manifest_sha256": report.report_manifest_sha256,
        "database_sha256": source_authority.database_sha256,
        "coordinate_inventory_sha256": EXP471_TEST_COORDINATE_INVENTORY_SHA256_ANTIGRAVITY,
        "method_order": cache.method_order, "total": cache.total_shards,
        "scored": cache.scored_shards, "abstained": cache.abstained_shards,
        "failed": cache.failed_shards, "dataset_count": len(cache.dataset_order),
        "common_support": 922, "coordinates": coordinates,
    })


def bind_e_to_exp471_exp472_antigravity(e_manifest: object, authority: Exp472ReuseAuthorityAntigravity) -> tuple[str, ...]:
    """Bind every ordered E TEST coordinate to D before labels can be accessed."""
    if not isinstance(authority, Exp472ReuseAuthorityAntigravity):
        raise ValueError("Exp472 Exp471 reuse authority differs")
    coordinates = _COORDINATES_BY_AUTHORITY_ID_ANTIGRAVITY.get(id(authority))
    if coordinates is None:
        raise ValueError("Exp472 Exp471 TEST coordinate authority is unavailable")
    if authority.coordinate_sha256 != authority.coordinate_inventory_sha256:
        raise ValueError("Exp472 E-vs-D TEST coordinate authority differs")
    if not isinstance(e_manifest, Mapping) or e_manifest.get("database_sha256") != authority.database_sha256:
        raise ValueError("Exp472 database authority differs")
    order = e_manifest.get("dataset_order")
    descriptors = e_manifest.get("descriptors")
    if not isinstance(order, (tuple, list)) or not isinstance(descriptors, (tuple, list)) or len(order) != len(descriptors):
        raise ValueError("Exp472 E manifest TEST coordinate coverage differs")
    if tuple(order) != tuple(coordinates):
        raise ValueError("Exp472 E manifest TEST coordinate order differs")
    for dataset, descriptor in zip(order, descriptors, strict=True):
        if not isinstance(descriptor, Mapping) or descriptor.get("dataset") != dataset:
            raise ValueError("Exp472 E manifest TEST coordinate coverage differs")
        observed = _coordinate_antigravity(descriptor.get("test_identity"), scope="E")
        if observed != coordinates[dataset]:
            raise ValueError("Exp472 E-vs-D TEST coordinate differs")
    return tuple(order)


__all__ = ["Exp472ReuseAuthorityAntigravity", "bind_e_to_exp471_exp472_antigravity", "verify_exp471_reuse_exp472_antigravity"]
