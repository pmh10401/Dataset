"""Fail-closed migration of verified clean Exp411 checkpoint records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp411_near_repeat_qualification.checkpoints import (
    CheckpointProvenance,
    CheckpointStore,
)
from experiments.exp411_near_repeat_qualification.config import (
    FROZEN_CONFIG as EXP411_FROZEN_CONFIG,
)
from experiments.exp411_near_repeat_qualification.qualification import (
    qualify_dataset,
)
from experiments.exp411_near_repeat_qualification.runner import (
    dataset_train_identity_sha256,
)

from .checkpoints import Exp414CheckpointStore
from .config import FROZEN_CONFIG
from .contracts import PreparedDataset


EXP411_MIGRATION_REPRESENTATIVES = (
    "ACSF1_normal_0",
    "CBF_normal_2",
    "DistalPhalanxTW_normal_8",
)

_SOURCE_ROOT_KEYS = {
    "schema_version",
    "provenance",
    "expected_datasets",
}
_SOURCE_PROVENANCE_KEYS = {
    "git_commit",
    "database_snapshot",
    "population_manifest_sha256",
    "frozen_config_sha256",
    "perturbation_contract_sha256",
    "acceleration_contract_sha256",
}


def open_exp411_checkpoint(
    root: Path,
    datasets: Sequence[str],
) -> CheckpointStore:
    """Open the pinned Exp411 source and validate every present record."""
    source_root = Path(root)
    path = source_root / "checkpoint_provenance.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Exp411 migration source provenance is invalid") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != _SOURCE_ROOT_KEYS
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or not isinstance(payload["provenance"], dict)
        or set(payload["provenance"]) != _SOURCE_PROVENANCE_KEYS
    ):
        raise ValueError("Exp411 migration source provenance is invalid")
    try:
        provenance = CheckpointProvenance(**payload["provenance"])
        provenance.to_payload()
    except (TypeError, ValueError) as error:
        raise ValueError("Exp411 migration source provenance is invalid") from error
    _require_pinned_provenance(provenance)
    requested = tuple(str(dataset) for dataset in datasets)
    store = CheckpointStore(source_root, provenance, requested)
    store.load_completed()
    return store


def verify_migration_representatives(
    source: CheckpointStore,
    prepared_by_dataset: Mapping[str, PreparedDataset],
) -> Mapping[str, str]:
    """Recompute the fixed clean representatives with the NumPy oracle."""
    validated = _revalidate_source(source)
    missing = [
        dataset
        for dataset in EXP411_MIGRATION_REPRESENTATIVES
        if dataset not in prepared_by_dataset
    ]
    if missing:
        raise ValueError(
            "Exp411 migration representative coverage is incomplete"
        )
    evidence: dict[str, str] = {}
    for dataset in EXP411_MIGRATION_REPRESENTATIVES:
        prepared = prepared_by_dataset[dataset]
        if prepared.dataset != dataset or not _is_clean(prepared):
            raise ValueError(
                f"Exp411 migration representative is not clean: {dataset}"
            )
        record = validated.load_dataset_record(dataset)
        raw_digest = prepared.repair_audit.raw_train_identity_sha256
        repaired_digest = (
            prepared.repair_audit.repaired_train_identity_sha256
        )
        if (
            record["dataset_train_identity_sha256"] != raw_digest
            or record["dataset_train_identity_sha256"] != repaired_digest
        ):
            raise ValueError(
                f"Exp411 migration representative TRAIN identity mismatch: "
                f"{dataset}"
            )
        observed = qualify_dataset(
            dataset=dataset,
            family=dataset.split("_normal_", 1)[0],
            runs=prepared.runs,
            backend="numpy_cpu",
            accelerated=False,
            config=EXP411_FROZEN_CONFIG,
        )
        if observed != record["qualification"]:
            raise ValueError(
                f"Exp411 migration representative qualification mismatch: "
                f"{dataset}"
            )
        evidence[dataset] = str(record["payload_sha256"])
    return MappingProxyType(evidence)


def migrate_if_eligible(
    prepared: PreparedDataset,
    source: CheckpointStore,
    target: Exp414CheckpointStore,
    representative_evidence: Mapping[str, str],
) -> bool:
    """Copy one clean, identity-matching Exp411 record into Exp414."""
    validated = _revalidate_source(source)
    _require_representative_evidence(validated, representative_evidence)
    if not _is_clean(prepared):
        return False
    completed = validated.load_completed()
    if prepared.dataset not in completed:
        return False
    record = validated.load_dataset_record(prepared.dataset)
    raw_digest = prepared.repair_audit.raw_train_identity_sha256
    repaired_digest = prepared.repair_audit.repaired_train_identity_sha256
    if (
        record["dataset_train_identity_sha256"] != raw_digest
        or record["dataset_train_identity_sha256"] != repaired_digest
    ):
        return False
    repair_digest = sha256(
        canonical_json_bytes(asdict(prepared.repair_audit))
    ).hexdigest()
    target.write_dataset(
        record["qualification"],
        record["backend_audit"],
        raw_train_identity_sha256=raw_digest,
        repaired_train_identity_sha256=repaired_digest,
        repair_audit_sha256=repair_digest,
        execution_source="migrated_exp411_verified",
        elapsed_seconds=float(record["elapsed_seconds"]),
    )
    return True


def _revalidate_source(source: CheckpointStore) -> CheckpointStore:
    if not isinstance(source, CheckpointStore):
        raise ValueError("Exp411 migration source store is invalid")
    _require_pinned_provenance(source.provenance)
    validated = CheckpointStore(
        source.root,
        source.provenance,
        source.expected_datasets,
    )
    validated.load_completed()
    return validated


def _require_pinned_provenance(provenance: CheckpointProvenance) -> None:
    provenance.to_payload()
    if provenance.git_commit != FROZEN_CONFIG.exp411_source_commit:
        raise ValueError("Exp411 migration source Git commit mismatch")
    if dict(provenance.database_snapshot) != {
        "db": FROZEN_CONFIG.database_sha256
    }:
        raise ValueError("Exp411 migration source database snapshot mismatch")
    if (
        provenance.population_manifest_sha256
        != FROZEN_CONFIG.population_manifest_sha256
    ):
        raise ValueError("Exp411 migration source population manifest mismatch")


def _require_representative_evidence(
    source: CheckpointStore,
    evidence: Mapping[str, str],
) -> None:
    if (
        not isinstance(evidence, Mapping)
        or tuple(evidence) != EXP411_MIGRATION_REPRESENTATIVES
    ):
        raise ValueError("Exp411 migration representative evidence is invalid")
    for dataset in EXP411_MIGRATION_REPRESENTATIVES:
        record = source.load_dataset_record(dataset)
        if evidence[dataset] != record["payload_sha256"]:
            raise ValueError(
                "Exp411 migration representative evidence is invalid"
            )


def _is_clean(prepared: PreparedDataset) -> bool:
    audit = prepared.repair_audit
    repaired_digest = audit.repaired_train_identity_sha256
    try:
        prepared_identity = dataset_train_identity_sha256(
            tuple(
                sorted(
                    prepared.runs,
                    key=lambda run: run.instance_index,
                )
            )
        )
    except ValueError:
        return False
    return (
        prepared.dataset == audit.dataset
        and audit.status == "ready"
        and bool(audit.runs)
        and len(prepared.runs) == len(audit.runs)
        and all(row.dataset == prepared.dataset for row in audit.runs)
        and all(row.repair_method == "unchanged" for row in audit.runs)
        and repaired_digest is not None
        and audit.raw_train_identity_sha256 == repaired_digest
        and prepared_identity == repaired_digest
    )


__all__ = [
    "EXP411_MIGRATION_REPRESENTATIVES",
    "migrate_if_eligible",
    "open_exp411_checkpoint",
    "verify_migration_representatives",
]
