"""Missing-aware qualification orchestration without TEST or outcome access."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_file
from experiments.exp411_near_repeat_qualification.acceleration import (
    acceleration_contract_sha256,
    configure_runtime,
)
from experiments.exp411_near_repeat_qualification.config import (
    FROZEN_CONFIG as EXP411_FROZEN_CONFIG,
)
from experiments.exp411_near_repeat_qualification.contracts import (
    DatasetBackendAudit,
    DatasetQualification,
)
from experiments.exp411_near_repeat_qualification.inputs import database_snapshot
from experiments.exp411_near_repeat_qualification.perturbations import (
    perturbation_contract_provenance,
    perturbation_types,
)
from experiments.exp411_near_repeat_qualification.qualification import (
    aggregate_qualification,
    qualify_dataset,
)
from experiments.exp411_near_repeat_qualification.runner import (
    dataset_train_identity_sha256,
)

from .checkpoints import Exp414CheckpointProvenance, Exp414CheckpointStore
from .config import FROZEN_CONFIG, Exp414FrozenConfig
from .contracts import DatasetRepairAudit, MissingInventory, PreparedDataset
from .inputs import iter_prepared_train_datasets, scan_missing_inventory
from .migration import (
    EXP411_MIGRATION_REPRESENTATIVES,
    migrate_if_eligible,
    open_exp411_checkpoint,
    verify_migration_representatives,
)
from .repair import build_dataset_repair_audit, repair_train_blob


_REPO_ROOT = Path(__file__).resolve().parents[2]
_QUANTILE_KEYS = ("0.01", "0.05", "0.10")
_SUBGROUPS = (
    "all",
    "nonmissing",
    "missing_repaired",
    "DodgerLoop",
    "MelbournePedestrian",
    "missing_support_abstain",
)
_SMOKE_COVERAGE = (
    "clean",
    "internal_gap",
    "edge_gap",
    "repaired_identity_collision",
    "all_missing",
)


def missing_support_abstention(
    prepared: PreparedDataset,
) -> DatasetQualification:
    """Return a deterministic dataset-level abstention for absent support."""
    _validate_prepared(prepared)
    audit = prepared.repair_audit
    if (
        audit.status != "missing_support_abstain"
        or audit.repaired_train_identity_sha256 is not None
        or prepared.runs
        or not any(
            row.repair_method == "all_missing_abstain"
            for row in audit.runs
        )
    ):
        raise ValueError("Exp414 missing-support abstention input is invalid")
    types = perturbation_types()
    zeros = {key: 0 for key in _QUANTILE_KEYS}
    nested_zeros = {
        key: {name: 0 for name in types}
        for key in _QUANTILE_KEYS
    }
    total = len(audit.runs)
    return DatasetQualification(
        dataset=prepared.dataset,
        family=_family_from_dataset(prepared.dataset),
        identity_group_count=len(
            {row.raw_values_sha256 for row in audit.runs}
        ),
        normal_total=total,
        normal_abstained=total,
        normal_near=dict(zeros),
        synthetic_total=dict(zeros),
        synthetic_near=dict(zeros),
        perturbation_total={
            key: dict(values)
            for key, values in nested_zeros.items()
        },
        perturbation_near={
            key: dict(values)
            for key, values in nested_zeros.items()
        },
    )


def qualify_prepared_dataset(
    prepared: PreparedDataset,
    *,
    accelerated: bool,
    backend_audit: list[DatasetBackendAudit] | None = None,
) -> DatasetQualification:
    """Qualify repaired TRAIN runs under the unchanged Exp411 algorithm."""
    _validate_prepared(prepared)
    if prepared.repair_audit.status == "missing_support_abstain":
        return missing_support_abstention(prepared)
    return qualify_dataset(
        dataset=prepared.dataset,
        family=_family_from_dataset(prepared.dataset),
        runs=prepared.runs,
        accelerated=bool(accelerated),
        backend_audit=backend_audit,
        config=EXP411_FROZEN_CONFIG,
    )


def aggregate_subgroups(
    rows: Sequence[DatasetQualification],
    audits: Sequence[DatasetRepairAudit],
    *,
    config: Exp414FrozenConfig = FROZEN_CONFIG,
) -> Mapping[str, object]:
    """Aggregate the global decision and non-authoritative diagnostics."""
    config.validate()
    values = tuple(rows)
    audit_rows = tuple(audits)
    if not values or len(values) != len(audit_rows):
        raise ValueError("Exp414 subgroup inputs are empty or misaligned")
    by_dataset: dict[str, DatasetQualification] = {}
    audit_by_dataset: dict[str, DatasetRepairAudit] = {}
    for row in values:
        if row.dataset in by_dataset:
            raise ValueError("Exp414 qualification dataset is duplicated")
        by_dataset[row.dataset] = row
    for audit in audit_rows:
        if audit.dataset in audit_by_dataset:
            raise ValueError("Exp414 repair audit dataset is duplicated")
        audit_by_dataset[audit.dataset] = audit
    if set(by_dataset) != set(audit_by_dataset):
        raise ValueError("Exp414 qualification and repair audit coverage mismatch")

    groups: dict[str, tuple[DatasetQualification, ...]] = {
        "all": tuple(by_dataset[name] for name in sorted(by_dataset)),
        "nonmissing": _select_rows(
            by_dataset,
            audit_by_dataset,
            lambda audit: (
                audit.status == "ready"
                and all(
                    row.nan_count == 0
                    and row.posinf_count == 0
                    and row.neginf_count == 0
                    for row in audit.runs
                )
            ),
        ),
        "missing_repaired": _select_rows(
            by_dataset,
            audit_by_dataset,
            lambda audit: (
                audit.status == "ready"
                and any(
                    row.nan_count or row.posinf_count or row.neginf_count
                    for row in audit.runs
                )
            ),
        ),
        "DodgerLoop": tuple(
            row
            for row in values
            if row.family.startswith("DodgerLoop")
        ),
        "MelbournePedestrian": tuple(
            row
            for row in values
            if row.family == "MelbournePedestrian"
        ),
        "missing_support_abstain": _select_rows(
            by_dataset,
            audit_by_dataset,
            lambda audit: audit.status == "missing_support_abstain",
        ),
    }

    output: dict[str, object] = {}
    for name in _SUBGROUPS:
        selected = groups[name]
        if selected:
            summary = aggregate_qualification(
                selected,
                config=EXP411_FROZEN_CONFIG,
            )
        else:
            summary = {
                "status": "not_applicable",
                "selected_quantile": None,
                "quantiles": {},
                "dataset_count": 0,
                "family_count": 0,
            }
        output[name] = {
            **summary,
            "selection_authority": name == "all",
        }
    return output


def _run_checkpointed_stream(
    stream: Iterable[PreparedDataset],
    target_store: Exp414CheckpointStore,
    *,
    source_store: object | None = None,
    representative_evidence: Mapping[str, str] | None = None,
    migration_enabled: bool,
    expected_families: int,
) -> tuple[
    tuple[DatasetQualification, ...],
    tuple[DatasetRepairAudit, ...],
    Mapping[str, object],
]:
    """Execute one record at a time and aggregate only after exact coverage."""
    if migration_enabled and (
        source_store is None or representative_evidence is None
    ):
        raise ValueError("Exp414 migration inputs are incomplete")
    observed: set[str] = set()
    repair_audits: dict[str, DatasetRepairAudit] = {}
    for prepared in stream:
        _validate_prepared(prepared)
        dataset = prepared.dataset
        if dataset in observed:
            raise ValueError(f"Exp414 streamed dataset is duplicated: {dataset}")
        observed.add(dataset)
        audit = prepared.repair_audit
        raw_digest = audit.raw_train_identity_sha256
        repaired_digest = audit.repaired_train_identity_sha256
        repair_digest = _repair_audit_sha256(audit)
        if target_store.has_dataset(dataset):
            record = target_store.load_dataset_record(dataset)
            _require_record_identity(
                record,
                dataset=dataset,
                raw_digest=raw_digest,
                repaired_digest=repaired_digest,
                repair_digest=repair_digest,
            )
        elif migration_enabled and migrate_if_eligible(
            prepared,
            source_store,
            target_store,
            representative_evidence,
        ):
            pass
        elif audit.status == "missing_support_abstain":
            target_store.write_dataset(
                missing_support_abstention(prepared),
                DatasetBackendAudit(
                    backend="numpy_cpu",
                    sampled_pairs=(),
                    max_abs_error=0.0,
                    fallback_reason="missing_support_abstain",
                ),
                raw_train_identity_sha256=raw_digest,
                repaired_train_identity_sha256=repaired_digest,
                repair_audit_sha256=repair_digest,
                execution_source="exp414_computed",
                elapsed_seconds=0.0,
            )
        else:
            target_store.write_progress(current_dataset=dataset)
            backend_audits: list[DatasetBackendAudit] = []
            started = perf_counter()
            result = qualify_prepared_dataset(
                prepared,
                accelerated=True,
                backend_audit=backend_audits,
            )
            if len(backend_audits) != 1:
                raise ValueError(
                    "Exp414 dataset backend audit coverage is invalid"
                )
            target_store.write_dataset(
                result,
                backend_audits[0],
                raw_train_identity_sha256=raw_digest,
                repaired_train_identity_sha256=repaired_digest,
                repair_audit_sha256=repair_digest,
                execution_source="exp414_computed",
                elapsed_seconds=perf_counter() - started,
            )
        record = target_store.load_dataset_record(dataset)
        _require_record_identity(
            record,
            dataset=dataset,
            raw_digest=raw_digest,
            repaired_digest=repaired_digest,
            repair_digest=repair_digest,
        )
        repair_audits[dataset] = audit

    if observed != set(target_store.expected_datasets):
        raise ValueError("Exp414 streamed dataset coverage is incomplete")
    rows = target_store.require_complete(expected_families)
    if set(repair_audits) != set(target_store.expected_datasets):
        raise ValueError("Exp414 repair audit coverage is incomplete")
    ordered_audits = tuple(
        repair_audits[dataset]
        for dataset in target_store.expected_datasets
    )
    subgroups = aggregate_subgroups(rows, ordered_audits)
    target_store.write_progress(current_dataset=None)
    return rows, ordered_audits, subgroups


def run_representative_smoke(output_dir: Path) -> Mapping[str, str]:
    """Run synthetic missingness coverage and hand results to publication."""
    FROZEN_CONFIG.validate()
    output = _require_new_output(output_dir)
    prepared = _build_smoke_prepared()
    cpu_rows: list[DatasetQualification] = []
    cpu_traces: dict[str, tuple[Mapping[str, object], ...]] = {}
    for row in prepared:
        trace: list[dict[str, object]] = []
        cpu_rows.append(
            _qualify_smoke_row(
                row,
                accelerated=False,
                backend_audit=None,
                parity_trace=trace,
            )
        )
        if row.repair_audit.status == "ready":
            if not trace:
                raise ValueError("Exp414 smoke NumPy tag trace is empty")
            cpu_traces[row.dataset] = tuple(trace)
    accelerated_rows: list[DatasetQualification] = []
    accelerated_traces: dict[str, tuple[Mapping[str, object], ...]] = {}
    backend_audits: list[DatasetBackendAudit] = []
    for row in prepared:
        audits: list[DatasetBackendAudit] = []
        trace = []
        observed = _qualify_smoke_row(
            row,
            accelerated=True,
            backend_audit=audits,
            parity_trace=trace,
        )
        accelerated_rows.append(observed)
        if row.repair_audit.status == "missing_support_abstain":
            backend_audits.append(
                DatasetBackendAudit(
                    backend="numpy_cpu",
                    sampled_pairs=(),
                    max_abs_error=0.0,
                    fallback_reason="missing_support_abstain",
                )
            )
        else:
            if not trace:
                raise ValueError(
                    "Exp414 smoke accelerated tag trace is empty"
                )
            accelerated_traces[row.dataset] = tuple(trace)
            if len(audits) != 1:
                raise ValueError(
                    "Exp414 smoke backend audit coverage is invalid"
                )
            backend_audits.append(audits[0])
    cpu_values = tuple(cpu_rows)
    accelerated_values = tuple(accelerated_rows)
    if accelerated_values != cpu_values:
        raise ValueError("Exp414 smoke NumPy/accelerated parity failed")
    if accelerated_traces != cpu_traces:
        raise ValueError("Exp414 smoke NumPy/accelerated tag parity failed")
    parity_trace_sha256 = sha256(
        canonical_json_bytes(cpu_traces)
    ).hexdigest()
    repair_audits = tuple(row.repair_audit for row in prepared)
    subgroups = aggregate_subgroups(accelerated_values, repair_audits)
    aggregate = subgroups["all"]
    provenance = {
        "mode": "representative_smoke",
        "git_commit": _git_commit(allow_dirty=True),
        "synthetic_dataset_count": len(prepared),
        "contains_real_dataset_rows": False,
        "smoke_coverage": _SMOKE_COVERAGE,
        "test_values_used": False,
        "test_labels_used": False,
        "contracts": _contract_values(),
    }
    records = tuple(
        {
            "qualification": row,
            "execution_source": "exp414_computed",
            "repair_status": prepared[index].repair_audit.status,
            "backend_audit": backend_audits[index],
        }
        for index, row in enumerate(accelerated_values)
    )
    return _finalize_outputs(
        records=records,
        repair_audits=repair_audits,
        aggregate=aggregate,
        subgroups=subgroups,
        backend_assessment={
            "canonical_backend": "adaptive_numpy_numba_cpu",
            "qualification_identity": True,
            "tag_identity": True,
            "parity_trace_sha256": parity_trace_sha256,
            "dataset_audits": tuple(asdict(value) for value in backend_audits),
            "mlx_status": "excluded_by_user_decision",
        },
        provenance=provenance,
        full_qualification_executed=False,
        output_dir=output,
        pre_publish_verify=lambda: _require_new_output(output),
    )


def run_full_qualification(
    db_path: Path,
    population_manifest: Path,
    output_dir: Path,
    *,
    exp411_checkpoint_dir: Path,
    checkpoint_dir: Path,
    benchmark_report: Path,
) -> Mapping[str, str]:
    """Run the separately authorized full stream behind fail-closed gates."""
    FROZEN_CONFIG.validate()
    output = _require_new_output(output_dir)
    starting_commit = _require_clean_git()
    runtime = configure_runtime()
    _require_runtime_contract(runtime)
    datasets = _load_population_manifest(population_manifest)
    manifest_sha256 = sha256_file(population_manifest)
    pinned_snapshot = database_snapshot(db_path)
    if dict(pinned_snapshot) != {"db": FROZEN_CONFIG.database_sha256}:
        raise ValueError("Exp414 database snapshot mismatch")
    inventory = scan_missing_inventory(db_path, datasets)
    _require_inventory(inventory)
    contracts = _contract_values()
    benchmark = _load_benchmark_report(
        benchmark_report,
        expected_commit=starting_commit,
        expected_snapshot=pinned_snapshot,
        expected_contracts=contracts,
    )
    benchmark_sha256 = sha256_file(benchmark_report)

    source_store = open_exp411_checkpoint(exp411_checkpoint_dir, datasets)
    representatives = tuple(
        iter_prepared_train_datasets(
            db_path,
            EXP411_MIGRATION_REPRESENTATIVES,
        )
    )
    prepared_by_dataset = {
        row.dataset: row
        for row in representatives
    }
    representative_evidence = verify_migration_representatives(
        source_store,
        prepared_by_dataset,
    )
    migration_source_sha256 = _migration_source_sha256(
        source_store,
        representative_evidence,
    )
    provenance = Exp414CheckpointProvenance(
        git_commit=starting_commit,
        database_snapshot=dict(pinned_snapshot),
        population_manifest_sha256=manifest_sha256,
        frozen_config_sha256=contracts["frozen_config_sha256"],
        perturbation_contract_sha256=contracts[
            "perturbation_contract_sha256"
        ],
        acceleration_contract_sha256=contracts[
            "acceleration_contract_sha256"
        ],
        missing_repair_contract_sha256=contracts[
            "missing_repair_contract_sha256"
        ],
        missing_inventory_sha256=_missing_inventory_sha256(inventory),
        qualification_contract_sha256=contracts[
            "qualification_contract_sha256"
        ],
        exp411_migration_source_sha256=migration_source_sha256,
    )
    store = Exp414CheckpointStore(checkpoint_dir, provenance, datasets)
    rows, repair_audits, subgroups = _run_checkpointed_stream(
        iter_prepared_train_datasets(db_path, datasets),
        store,
        source_store=source_store,
        representative_evidence=representative_evidence,
        migration_enabled=True,
        expected_families=FROZEN_CONFIG.expected_families,
    )
    aggregate = subgroups["all"]
    final_state = lambda: _require_full_state(
        db_path=db_path,
        population_manifest=population_manifest,
        output_dir=output,
        benchmark_report=benchmark_report,
        checkpoint_store=store,
        datasets=datasets,
        expected_commit=starting_commit,
        expected_snapshot=pinned_snapshot,
        expected_manifest_sha256=manifest_sha256,
        expected_benchmark_sha256=benchmark_sha256,
        expected_inventory=inventory,
        expected_contracts=contracts,
    )
    final_state()
    records = tuple(
        store.load_dataset_record(dataset)
        for dataset in datasets
    )
    execution_counts: dict[str, int] = {}
    for record in records:
        source = str(record["execution_source"])
        execution_counts[source] = execution_counts.get(source, 0) + 1
    publication_provenance = {
        "mode": "full_qualification",
        "git_commit": starting_commit,
        "database_snapshot": dict(pinned_snapshot),
        "population_manifest_sha256": manifest_sha256,
        "checkpoint_provenance": provenance.to_payload(),
        "benchmark_report_sha256": benchmark_sha256,
        "benchmark": benchmark,
        "runtime": dict(runtime),
        "contracts": contracts,
        "missing_inventory": asdict(inventory),
        "migration_representative_payload_sha256": dict(
            representative_evidence
        ),
        "execution_source_counts": execution_counts,
        "test_values_used": False,
        "test_labels_used": False,
    }
    return _finalize_outputs(
        records=records,
        repair_audits=repair_audits,
        aggregate=aggregate,
        subgroups=subgroups,
        backend_assessment={
            "canonical_backend": "adaptive_numpy_numba_cpu",
            "accelerated_backend_adopted": True,
            "benchmark_report_sha256": benchmark_sha256,
            "mlx_status": "excluded_by_user_decision",
        },
        provenance=publication_provenance,
        full_qualification_executed=True,
        output_dir=output,
        pre_publish_verify=final_state,
    )


def missing_repair_contract_sha256() -> str:
    return _source_contract_sha256(("config.py", "contracts.py", "repair.py"))


def qualification_contract_sha256() -> str:
    package = _REPO_ROOT / "experiments" / "exp411_near_repeat_qualification"
    names = (
        "config.py",
        "qualification.py",
        "folds.py",
        "distances.py",
        "perturbations.py",
        "acceleration.py",
    )
    payload = {
        name: sha256((package / name).read_bytes()).hexdigest()
        for name in names
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def execution_contract_sha256() -> str:
    return _source_contract_sha256(
        (
            "checkpoints.py",
            "inputs.py",
            "migration.py",
            "runner.py",
        )
    )


def _build_smoke_prepared() -> tuple[PreparedDataset, ...]:
    length = 16

    def base(index: int) -> np.ndarray:
        x = np.arange(length, dtype=np.float32)
        return np.asarray(
            np.sin(x * (0.13 + index * 0.01)) + index * 0.17,
            dtype="<f4",
        )

    payloads: list[tuple[str, tuple[np.ndarray, ...]]] = []
    payloads.append(
        ("SmokeClean_normal_0", tuple(base(index) for index in range(6)))
    )
    internal = [base(index) for index in range(6)]
    internal[0] = internal[0].copy()
    internal[0][5:8] = np.nan
    payloads.append(("SmokeInternal_normal_0", tuple(internal)))
    edge = [base(index) for index in range(6)]
    edge[0] = edge[0].copy()
    edge[0][:2] = np.inf
    edge[0][-2:] = -np.inf
    payloads.append(("SmokeEdge_normal_0", tuple(edge)))
    collision = [base(index) for index in range(6)]
    collision[1] = collision[0].copy()
    collision[0] = collision[0].copy()
    collision[0][7] = np.nan
    collision[1][7] = (
        collision[1][6] + collision[1][8]
    ) / np.float32(2.0)
    payloads.append(("SmokeCollision_normal_0", tuple(collision)))
    all_missing = [base(index) for index in range(6)]
    all_missing[-1] = np.full(length, np.nan, dtype="<f4")
    payloads.append(("SmokeAllMissing_normal_0", tuple(all_missing)))

    prepared = []
    for dataset, arrays in payloads:
        runs = []
        audits = []
        for index, values in enumerate(arrays):
            run, audit = repair_train_blob(
                dataset,
                index,
                np.asarray(values, dtype="<f4").tobytes(),
            )
            if run is not None:
                runs.append(run)
            audits.append(audit)
        repair_audit = build_dataset_repair_audit(dataset, runs, audits)
        prepared.append(
            PreparedDataset(
                dataset=dataset,
                runs=(
                    tuple(runs)
                    if repair_audit.status == "ready"
                    else ()
                ),
                repair_audit=repair_audit,
            )
        )
    return tuple(prepared)


def _qualify_smoke_row(
    prepared: PreparedDataset,
    *,
    accelerated: bool,
    backend_audit: list[DatasetBackendAudit] | None,
    parity_trace: list[dict[str, object]],
) -> DatasetQualification:
    _validate_prepared(prepared)
    if prepared.repair_audit.status == "missing_support_abstain":
        return missing_support_abstention(prepared)
    return qualify_dataset(
        dataset=prepared.dataset,
        family=_family_from_dataset(prepared.dataset),
        runs=prepared.runs,
        accelerated=accelerated,
        backend_audit=backend_audit,
        parity_trace=parity_trace,
        config=EXP411_FROZEN_CONFIG,
    )


def _require_full_state(
    *,
    db_path: Path,
    population_manifest: Path,
    output_dir: Path,
    benchmark_report: Path,
    checkpoint_store: Exp414CheckpointStore,
    datasets: Sequence[str],
    expected_commit: str,
    expected_snapshot: Mapping[str, str],
    expected_manifest_sha256: str,
    expected_benchmark_sha256: str,
    expected_inventory: MissingInventory,
    expected_contracts: Mapping[str, str],
) -> None:
    _require_new_output(output_dir)
    if _require_clean_git() != expected_commit:
        raise ValueError("Exp414 Git commit changed during qualification")
    if dict(database_snapshot(db_path)) != dict(expected_snapshot):
        raise ValueError("Exp414 database snapshot changed before publication")
    if sha256_file(population_manifest) != expected_manifest_sha256:
        raise ValueError(
            "Exp414 population manifest changed before publication"
        )
    if sha256_file(benchmark_report) != expected_benchmark_sha256:
        raise ValueError("Exp414 benchmark report changed before publication")
    observed_inventory = scan_missing_inventory(db_path, datasets)
    if observed_inventory != expected_inventory:
        raise ValueError("Exp414 missing inventory changed before publication")
    _require_inventory(observed_inventory)
    _load_benchmark_report(
        benchmark_report,
        expected_commit=expected_commit,
        expected_snapshot=expected_snapshot,
        expected_contracts=expected_contracts,
    )
    checkpoint_store.require_complete(FROZEN_CONFIG.expected_families)
    _require_checkpoint_identities(
        db_path,
        datasets,
        checkpoint_store,
    )


def _require_checkpoint_identities(
    db_path: Path,
    datasets: Sequence[str],
    store: Exp414CheckpointStore,
) -> None:
    observed: set[str] = set()
    for prepared in iter_prepared_train_datasets(db_path, datasets):
        dataset = prepared.dataset
        if dataset in observed:
            raise ValueError("Exp414 prepublication dataset is duplicated")
        observed.add(dataset)
        record = store.load_dataset_record(dataset)
        _require_record_identity(
            record,
            dataset=dataset,
            raw_digest=prepared.repair_audit.raw_train_identity_sha256,
            repaired_digest=(
                prepared.repair_audit.repaired_train_identity_sha256
            ),
            repair_digest=_repair_audit_sha256(prepared.repair_audit),
        )
    if observed != set(datasets):
        raise ValueError(
            "Exp414 prepublication TRAIN identity coverage is incomplete"
        )


def _finalize_outputs(
    *,
    records: Sequence[Mapping[str, object]],
    repair_audits: Sequence[DatasetRepairAudit],
    aggregate: Mapping[str, object],
    subgroups: Mapping[str, object],
    backend_assessment: Mapping[str, object],
    provenance: Mapping[str, object],
    full_qualification_executed: bool,
    output_dir: Path,
    pre_publish_verify: Callable[[], None] | None,
) -> Mapping[str, str]:
    """Render and atomically publish the complete Exp414 artifact set."""
    try:
        from .publication import publish_payloads
        from .reporting import render_payloads
    except ImportError as error:
        raise RuntimeError(
            "Exp414 publication components are not installed"
        ) from error
    payloads = render_payloads(
        records,
        repair_audits,
        aggregate,
        subgroups,
        backend_assessment,
        provenance,
        full_qualification_executed=full_qualification_executed,
        config=FROZEN_CONFIG,
    )
    return publish_payloads(
        payloads,
        output_dir,
        provenance,
        pre_publish_verify=pre_publish_verify,
    )


def _validate_prepared(prepared: PreparedDataset) -> None:
    if not isinstance(prepared, PreparedDataset):
        raise ValueError("Exp414 prepared dataset is invalid")
    audit = prepared.repair_audit
    if prepared.dataset != audit.dataset or not audit.runs:
        raise ValueError("Exp414 prepared dataset audit is invalid")
    if len({row.instance_index for row in audit.runs}) != len(audit.runs):
        raise ValueError("Exp414 prepared dataset audit indices are duplicated")
    raw_identity = sha256(
        canonical_json_bytes(
            [
                {
                    "dataset": row.dataset,
                    "instance_index": int(row.instance_index),
                    "values_sha256": row.raw_values_sha256,
                }
                for row in sorted(
                    audit.runs,
                    key=lambda value: value.instance_index,
                )
            ]
        )
    ).hexdigest()
    if (
        any(row.dataset != prepared.dataset for row in audit.runs)
        or raw_identity != audit.raw_train_identity_sha256
    ):
        raise ValueError("Exp414 raw TRAIN identity is invalid")
    if audit.status == "ready":
        audit_by_index = {
            row.instance_index: row
            for row in audit.runs
        }
        if (
            len(prepared.runs) != len(audit.runs)
            or audit.repaired_train_identity_sha256 is None
            or {run.instance_index for run in prepared.runs}
            != set(audit_by_index)
            or any(
                run.dataset != prepared.dataset
                or not np.isfinite(run.values).all()
                for run in prepared.runs
            )
        ):
            raise ValueError("Exp414 repaired dataset support is invalid")
        for run in prepared.runs:
            values_digest = sha256(
                np.asarray(run.values, dtype="<f4").tobytes()
            ).hexdigest()
            row = audit_by_index[run.instance_index]
            if (
                run.values_sha256 != values_digest
                or row.repaired_values_sha256 != values_digest
            ):
                raise ValueError("Exp414 repaired TRAIN identity is invalid")
        repaired_identity = dataset_train_identity_sha256(
            tuple(
                sorted(
                    prepared.runs,
                    key=lambda run: run.instance_index,
                )
            )
        )
        if repaired_identity != audit.repaired_train_identity_sha256:
            raise ValueError("Exp414 repaired TRAIN identity is invalid")
    elif audit.status == "missing_support_abstain":
        if prepared.runs or audit.repaired_train_identity_sha256 is not None:
            raise ValueError("Exp414 missing-support dataset is invalid")
    else:
        raise ValueError("Exp414 repair status is invalid")


def _select_rows(
    rows: Mapping[str, DatasetQualification],
    audits: Mapping[str, DatasetRepairAudit],
    predicate: Callable[[DatasetRepairAudit], bool],
) -> tuple[DatasetQualification, ...]:
    return tuple(
        rows[dataset]
        for dataset in sorted(rows)
        if predicate(audits[dataset])
    )


def _repair_audit_sha256(audit: DatasetRepairAudit) -> str:
    return sha256(canonical_json_bytes(asdict(audit))).hexdigest()


def _require_record_identity(
    record: Mapping[str, object],
    *,
    dataset: str,
    raw_digest: str,
    repaired_digest: str | None,
    repair_digest: str,
) -> None:
    if (
        record["raw_train_identity_sha256"] != raw_digest
        or record["repaired_train_identity_sha256"] != repaired_digest
        or record["repair_audit_sha256"] != repair_digest
    ):
        raise ValueError(
            f"Exp414 checkpoint TRAIN identity mismatch: {dataset}"
        )


def _family_from_dataset(dataset: str) -> str:
    if dataset.count("_normal_") != 1:
        raise ValueError("Exp414 dataset must contain one _normal_ marker")
    family, suffix = dataset.split("_normal_", 1)
    if not family or not (
        suffix.isdecimal()
        or (suffix.startswith("-") and suffix[1:].isdecimal())
    ):
        raise ValueError(
            "Exp414 dataset family and numeric class suffix are required"
        )
    return family


def _load_population_manifest(path: Path) -> tuple[str, ...]:
    source = Path(path)
    if sha256_file(source) != FROZEN_CONFIG.population_manifest_sha256:
        raise ValueError("Exp414 population manifest SHA256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Exp414 population manifest is invalid") from error
    lanes = payload.get("frozen_hard_lanes") if isinstance(payload, dict) else None
    if not isinstance(lanes, dict):
        raise ValueError("Exp414 population manifest lacks frozen_hard_lanes")
    datasets = tuple(sorted(str(dataset) for dataset in lanes))
    families = {_family_from_dataset(dataset) for dataset in datasets}
    if (
        len(datasets) != FROZEN_CONFIG.expected_datasets
        or len(set(datasets)) != len(datasets)
        or len(families) != FROZEN_CONFIG.expected_families
    ):
        raise ValueError("Exp414 population manifest universe mismatch")
    return datasets


def _load_benchmark_report(
    path: Path,
    *,
    expected_commit: str,
    expected_snapshot: Mapping[str, str],
    expected_contracts: Mapping[str, str],
) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Exp414 benchmark report is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("Exp414 benchmark report is invalid")
    try:
        from .benchmark import require_acceleration_adoption
    except ImportError as error:
        raise ValueError("Exp414 benchmark validator is unavailable") from error
    require_acceleration_adoption(payload, expected_commit)
    if dict(payload.get("database_snapshot", {})) != dict(expected_snapshot):
        raise ValueError("Exp414 benchmark database snapshot mismatch")
    for name, expected in expected_contracts.items():
        if payload.get(name) != expected:
            raise ValueError(f"Exp414 benchmark {name} mismatch")
    return payload


def _require_runtime_contract(runtime: Mapping[str, object]) -> None:
    if (
        runtime.get("numba_threads") != FROZEN_CONFIG.numba_threads
        or runtime.get("nested_threads")
        != FROZEN_CONFIG.nested_numerical_threads
    ):
        raise ValueError("Exp414 full runtime thread contract is invalid")


def _require_inventory(inventory: MissingInventory) -> None:
    expected = (
        FROZEN_CONFIG.expected_datasets,
        FROZEN_CONFIG.expected_missing_datasets,
        FROZEN_CONFIG.expected_missing_runs,
        FROZEN_CONFIG.expected_nonfinite_values,
        FROZEN_CONFIG.expected_all_missing_runs,
        FROZEN_CONFIG.affected_datasets_sha256,
    )
    observed = (
        inventory.datasets,
        inventory.affected_datasets,
        inventory.affected_runs,
        inventory.nonfinite_values,
        inventory.all_missing_runs,
        inventory.affected_datasets_sha256,
    )
    if observed != expected:
        raise ValueError("Exp414 missing inventory mismatch")


def _missing_inventory_sha256(inventory: MissingInventory) -> str:
    return sha256(canonical_json_bytes(asdict(inventory))).hexdigest()


def _contract_values() -> dict[str, str]:
    return {
        "frozen_config_sha256": sha256(
            canonical_json_bytes(FROZEN_CONFIG.to_payload())
        ).hexdigest(),
        "perturbation_contract_sha256": sha256(
            canonical_json_bytes(perturbation_contract_provenance())
        ).hexdigest(),
        "acceleration_contract_sha256": acceleration_contract_sha256(),
        "missing_repair_contract_sha256": missing_repair_contract_sha256(),
        "qualification_contract_sha256": qualification_contract_sha256(),
        "execution_contract_sha256": execution_contract_sha256(),
    }


def _source_contract_sha256(names: Sequence[str]) -> str:
    package = Path(__file__).resolve().parent
    payload = {
        name: sha256((package / name).read_bytes()).hexdigest()
        for name in names
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _migration_source_sha256(
    source_store: object,
    evidence: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": 1,
        "source_checkpoint_provenance": source_store.provenance.to_payload(),
        "source_expected_datasets": list(source_store.expected_datasets),
        "representative_payload_sha256": dict(evidence),
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _require_new_output(output_dir: Path) -> Path:
    output = Path(output_dir)
    if not output.parent.is_dir():
        raise FileNotFoundError("Exp414 output parent does not exist")
    if output.exists():
        raise FileExistsError("Exp414 output directory must not exist")
    return output


def _git_commit(*, allow_dirty: bool) -> str:
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        if allow_dirty:
            return "unavailable"
        raise ValueError("Exp414 cannot resolve Git commit")
    return result.stdout.strip()


def _require_clean_git() -> str:
    status = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        raise ValueError("Exp414 cannot inspect Git worktree")
    if status.stdout.strip():
        raise ValueError(
            "Exp414 full qualification requires a clean Git worktree"
        )
    return _git_commit(allow_dirty=False)


__all__ = [
    "aggregate_subgroups",
    "execution_contract_sha256",
    "missing_repair_contract_sha256",
    "missing_support_abstention",
    "qualification_contract_sha256",
    "qualify_prepared_dataset",
    "run_full_qualification",
    "run_representative_smoke",
]
