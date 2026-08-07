#!/usr/bin/env python3
"""Non-official, label-free Exp417 readiness smoke."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
import json
import os
import sqlite3
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.backend import load_frozen_backend_audit
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    DatasetResult,
    DatasetTerminalEvidence,
    FrozenRunContext,
)
from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    _result_payload,
)
from experiments.exp417_causal_multihorizon_shadow.inputs import (
    assert_label_free_sql_contract,
    freeze_lane_inputs,
    freeze_run_context,
    verify_input_snapshots,
)
from experiments.exp417_causal_multihorizon_shadow.runner import (
    RunnerResources,
    RunnerRuntime,
    run_dataset,
)
from scratch.run_exp417_causal_multihorizon_shadow import (
    _backend_audit_path,
    _existing_file,
    _new_output_dir,
    require_clean_git,
)


SMOKE_FILE = "EXP417_NON_OFFICIAL_SMOKE.json"
SMOKE_SCHEMA_KEYS = frozenset(
    {
        "experiment_id",
        "artifact_kind",
        "official_prediction",
        "publication_eligible",
        "contains_test_labels",
        "official_checkpoint_reusable",
        "selection_source",
        "selection_rule",
        "selected_datasets",
        "statuses",
        "backend_audit",
        "elapsed_seconds",
        "estimated_full_seconds",
        "disk_projection",
        "estimate_method",
    }
)


def _exact_three(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("smoke dataset count must be exactly 3") from error
    if count != 3:
        raise argparse.ArgumentTypeError("smoke dataset count must be exactly 3")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a non-official three-dataset Exp417 label-free smoke.",
        allow_abbrev=False,
    )
    parser.add_argument("--database", type=_existing_file, required=True, metavar="PATH")
    parser.add_argument("--exp291-prediction", type=_existing_file, required=True, metavar="PATH")
    parser.add_argument("--exp291-manifest", type=_existing_file, required=True, metavar="PATH")
    parser.add_argument("--exp292-prediction", type=_existing_file, required=True, metavar="PATH")
    parser.add_argument("--exp292-manifest", type=_existing_file, required=True, metavar="PATH")
    parser.add_argument("--smoke-datasets", type=_exact_three, required=True, metavar="3")
    parser.add_argument("--output-dir", type=_new_output_dir, required=True, metavar="NEW_SMOKE_PATH")
    return parser


@dataclass(frozen=True)
class TrainMetadata:
    dataset: str
    normal_count: int
    minimum_length: int
    maximum_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("TRAIN metadata dataset is invalid")
        if any(
            type(value) is not int or value < 1
            for value in (self.normal_count, self.minimum_length, self.maximum_length)
        ):
            raise ValueError("TRAIN metadata counts and lengths must be positive integers")
        if self.minimum_length > self.maximum_length:
            raise ValueError("TRAIN metadata length range is invalid")


@dataclass(frozen=True)
class SmokeContext:
    """A governance type that is structurally incompatible with official output."""

    selected_datasets: tuple[str, ...]
    output_dir: Path
    official_prediction: bool = field(default=False, init=False)
    publication_eligible: bool = field(default=False, init=False)
    contains_test_labels: bool = field(default=False, init=False)
    official_checkpoint_reusable: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        datasets = tuple(self.selected_datasets)
        if (
            len(datasets) != 3
            or datasets != tuple(sorted(datasets))
            or len(set(datasets)) != 3
            or any(not isinstance(name, str) or not name for name in datasets)
        ):
            raise ValueError("smoke context requires three sorted unique datasets")
        output = Path(self.output_dir)
        if output.exists() or not output.parent.is_dir():
            raise ValueError("smoke output must be a new path with an existing parent")
        object.__setattr__(self, "selected_datasets", datasets)
        object.__setattr__(self, "output_dir", output)

    @property
    def smoke_datasets(self) -> int:
        return len(self.selected_datasets)


@dataclass(frozen=True)
class DiskProjection:
    """Measured terminal projection plus a schema-derived candidate upper bound."""

    measured_smoke_artifact_bytes: int
    measured_sample_checkpoint_bytes: int
    failed_terminal_projection_bytes: int
    candidate_bearing_conservative_upper_bytes: int
    frozen_dataset_count: int
    frozen_test_instance_count: int
    normal_train_series_count: int
    sample_basis: str

    def __post_init__(self) -> None:
        numeric = (
            self.measured_smoke_artifact_bytes,
            self.measured_sample_checkpoint_bytes,
            self.failed_terminal_projection_bytes,
            self.candidate_bearing_conservative_upper_bytes,
            self.frozen_dataset_count,
            self.frozen_test_instance_count,
            self.normal_train_series_count,
        )
        if any(type(value) is not int or value < 0 for value in numeric):
            raise ValueError("disk projection values must be non-negative integers")
        if self.frozen_dataset_count != FROZEN_CONFIG.expected_datasets:
            raise ValueError("disk projection must use the frozen dataset count")
        if self.candidate_bearing_conservative_upper_bytes < self.failed_terminal_projection_bytes:
            raise ValueError("disk projection upper bound is below its terminal projection")
        if self.sample_basis not in {
            "actual_canonical_smoke_results",
            "reconstructed_minimum_terminal_schema",
        }:
            raise ValueError("disk projection sample basis is invalid")


def project_disk_bounds(
    *,
    measured_smoke_artifact_bytes: int,
    measured_sample_checkpoint_bytes: int,
    sample_dataset_count: int,
    frozen_dataset_count: int,
    frozen_test_instance_count: int,
    normal_train_series_count: int,
    worst_decision_bytes: int,
    worst_group_bytes: int,
    sample_basis: str = "actual_canonical_smoke_results",
) -> DiskProjection:
    """Project terminal bytes and bound candidate-bearing canonical evidence."""
    integers = (
        measured_smoke_artifact_bytes,
        measured_sample_checkpoint_bytes,
        sample_dataset_count,
        frozen_dataset_count,
        frozen_test_instance_count,
        normal_train_series_count,
        worst_decision_bytes,
        worst_group_bytes,
    )
    if any(type(value) is not int or value < 0 for value in integers) or sample_dataset_count < 1:
        raise ValueError("disk bound inputs must be non-negative integers")
    terminal = measured_smoke_artifact_bytes + (
        measured_sample_checkpoint_bytes * frozen_dataset_count
        + sample_dataset_count - 1
    ) // sample_dataset_count
    upper = (
        terminal
        + frozen_test_instance_count * worst_decision_bytes
        + normal_train_series_count * worst_group_bytes
    )
    return DiskProjection(
        measured_smoke_artifact_bytes=measured_smoke_artifact_bytes,
        measured_sample_checkpoint_bytes=measured_sample_checkpoint_bytes,
        failed_terminal_projection_bytes=terminal,
        candidate_bearing_conservative_upper_bytes=upper,
        frozen_dataset_count=frozen_dataset_count,
        frozen_test_instance_count=frozen_test_instance_count,
        normal_train_series_count=normal_train_series_count,
        sample_basis=sample_basis,
    )


def _checkpoint_record_size(result: DatasetResult) -> int:
    """Measure canonical Task 8 record bytes without creating a checkpoint."""
    if not isinstance(result, DatasetResult):
        raise ValueError("checkpoint measurement requires a DatasetResult")
    terminal = isinstance(result.decision_evidence, DatasetTerminalEvidence)
    lengths = tuple(group.length for group in result.groups)
    body: dict[str, Any] = {
        "schema_version": 2,
        "dataset": result.dataset,
        "raw_train_identity_sha256": None if terminal else "0" * 64,
        "repaired_train_identity_sha256": None if terminal else "1" * 64,
        "length_group_identity_sha256": (
            {} if terminal else {str(length): "2" * 64 for length in lengths}
        ),
        "canonical_model_hashes": (
            {}
            if terminal
            else {
                str(length): {
                    str(seed): "3" * 64 for seed in FROZEN_CONFIG.model_seeds
                }
                for length in lengths
            }
        ),
        "score_calibration_hashes": (
            {} if terminal else {str(length): "4" * 64 for length in lengths}
        ),
        "qualification": _result_payload(result),
        "candidate_coordinates": [
            [candidate.dataset, candidate.instance_index]
            for candidate in result.candidates
        ],
        "backend_audit": (
            None
            if terminal
            else asdict(result.decision_evidence.frozen_train_contract.backend)
        ),
        "elapsed_seconds": 0.0,
    }
    body["payload_sha256"] = sha256(canonical_json_bytes(body)).hexdigest()
    return len(canonical_json_bytes(body)) + 1


def _schema_upper_units(context: FrozenRunContext) -> tuple[int, int]:
    """Return canonical per-TEST and per-TRAIN conservative byte units."""
    if not isinstance(context, FrozenRunContext):
        raise ValueError("disk units require a FrozenRunContext")
    longest_dataset = max(context.datasets, key=lambda name: len(name.encode("utf-8")))
    maximum_index = max(
        index
        for entries in context.test_identity_manifest.values()
        for index, _ in entries
    )
    maximum_length = max(
        length
        for entries in context.test_identity_manifest.values()
        for _, length in entries
    )
    score = {
        "horizon": maximum_length,
        "score": 1.7976931348623157e308,
        "p_value": 0.9999999999999999,
        "persistence": 0.9999999999999999,
    }
    seed = {
        "seed": max(FROZEN_CONFIG.model_seeds),
        "scores": {
            str(maximum_length): score,
            str(maximum_length - 1): {**score, "horizon": maximum_length - 1},
            str(maximum_length - 2): {**score, "horizon": maximum_length - 2},
        },
        "candidate": True,
    }
    decision = {
        "dataset": longest_dataset,
        "instance_index": maximum_index,
        "length": maximum_length,
        "seed_decisions": [seed, seed, seed],
    }
    # Worst case retains the same all-seed evidence in decision evidence and
    # candidate evidence, then renders a prediction/report row.  The 1024-byte
    # allowance covers canonical separators, CSV fields, and manifest hashes.
    worst_decision = (
        2 * len(canonical_json_bytes(decision)) + 1024
    )
    group = {
        "length": maximum_length,
        "series_count": 9_999_999_999,
        "status": "qualified",
        "oof_candidate_rate": 0.9999999999999999,
        "wilson_upper": 0.9999999999999999,
        "pooled_synthetic_sensitivity": 0.9999999999999999,
        "per_type_synthetic_sensitivity": {
            name: 0.9999999999999999
            for name in (
                "amplitude_scaling",
                "gradual_drift",
                "level_shift",
                "period_change",
            )
        },
        "seed_agreement": True,
        "finite": True,
        "backend_fallback": False,
    }
    # Every normal TRAIN series as its own length-group is deliberately much
    # looser than reality.  Four copies cover result, contract, CSV, and the
    # per-length digest/model maps with an additional 2048-byte allowance.
    worst_group = 4 * len(canonical_json_bytes(group)) + 2048
    return worst_decision, worst_group


def estimate_disk_projection(
    context: FrozenRunContext,
    results: Sequence[DatasetResult],
    *,
    normal_train_series_count: int,
    measured_smoke_artifact_bytes: int,
    sample_basis: str = "actual_canonical_smoke_results",
) -> DiskProjection:
    frozen_results = tuple(results)
    if not frozen_results or any(not isinstance(result, DatasetResult) for result in frozen_results):
        raise ValueError("disk projection requires concrete dataset results")
    if any(result.context != context for result in frozen_results):
        raise ValueError("disk projection results do not share the frozen context")
    decision_bytes, group_bytes = _schema_upper_units(context)
    return project_disk_bounds(
        measured_smoke_artifact_bytes=measured_smoke_artifact_bytes,
        measured_sample_checkpoint_bytes=sum(
            _checkpoint_record_size(result) for result in frozen_results
        ),
        sample_dataset_count=len(frozen_results),
        frozen_dataset_count=len(context.datasets),
        frozen_test_instance_count=sum(context.test_sizes.values()),
        normal_train_series_count=normal_train_series_count,
        worst_decision_bytes=decision_bytes,
        worst_group_bytes=group_bytes,
        sample_basis=sample_basis,
    )


def _tertile(value: int, ordered: Sequence[int]) -> int:
    lower = ordered[(len(ordered) - 1) // 3]
    upper = ordered[(2 * (len(ordered) - 1)) // 3]
    if value <= lower:
        return 0
    if value <= upper:
        return 1
    return 2


def select_train_strata(
    rows: Sequence[TrainMetadata], count: int
) -> tuple[str, ...]:
    """Select low/middle/high count×length strata using TRAIN metadata only."""
    frozen = tuple(rows)
    if count != 3 or len(frozen) < 3 or any(not isinstance(row, TrainMetadata) for row in frozen):
        raise ValueError("TRAIN smoke strata require at least three records and count 3")
    if len({row.dataset for row in frozen}) != len(frozen):
        raise ValueError("TRAIN smoke metadata datasets must be unique")
    counts = sorted(row.normal_count for row in frozen)
    lengths = sorted(row.maximum_length for row in frozen)
    bins = {
        row.dataset: (
            _tertile(row.normal_count, counts),
            _tertile(row.maximum_length, lengths),
        )
        for row in frozen
    }
    selected: list[str] = []
    for target in ((0, 0), (1, 1), (2, 2)):
        candidates = sorted(
            row.dataset
            for row in frozen
            if row.dataset not in selected and bins[row.dataset] == target
        )
        if not candidates:
            candidates = [
                row.dataset
                for row in sorted(
                    (item for item in frozen if item.dataset not in selected),
                    key=lambda item: (
                        abs(bins[item.dataset][0] - target[0])
                        + abs(bins[item.dataset][1] - target[1]),
                        item.dataset,
                    ),
                )
            ]
        selected.append(candidates[0])
    return tuple(sorted(selected))


def load_train_metadata(database: Path, datasets: Sequence[str]) -> tuple[TrainMetadata, ...]:
    """Read only normal TRAIN counts and raw byte lengths from SQLite."""
    path = Path(database).resolve()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT d.name, COUNT(*), MIN(length(i.values_blob) / 4),
                   MAX(length(i.values_blob) / 4)
            FROM instances AS i
            JOIN datasets AS d ON d.id = i.dataset_id
            WHERE i.split = 'TRAIN' AND i.label = '0'
            GROUP BY d.name
            ORDER BY d.name
            """
        ).fetchall()
    finally:
        connection.close()
    allowed = set(datasets)
    metadata = tuple(
        TrainMetadata(name, count, minimum, maximum)
        for name, count, minimum, maximum in rows
        if name in allowed
    )
    if tuple(row.dataset for row in metadata) != tuple(sorted(allowed)):
        raise ValueError("normal TRAIN metadata does not cover the frozen population")
    return metadata


def render_smoke_context(
    context: SmokeContext,
    *,
    statuses: Mapping[str, str],
    backend: Mapping[str, Any],
    elapsed_seconds: float,
    estimated_full_seconds: float,
    disk_projection: DiskProjection,
) -> dict[str, Any]:
    if not isinstance(context, SmokeContext):
        raise ValueError("smoke rendering requires SmokeContext")
    if tuple(statuses) != context.selected_datasets:
        raise ValueError("smoke statuses do not cover exact selected datasets")
    if not isinstance(disk_projection, DiskProjection):
        raise ValueError("smoke rendering requires a DiskProjection")
    payload = {
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "artifact_kind": "non_official_label_free_readiness_smoke",
        "official_prediction": context.official_prediction,
        "publication_eligible": context.publication_eligible,
        "contains_test_labels": context.contains_test_labels,
        "official_checkpoint_reusable": context.official_checkpoint_reusable,
        "selection_source": "normal_TRAIN_count_and_raw_length_only",
        "selection_rule": "count_length_tertiles_lexicographic_tie_break",
        "selected_datasets": list(context.selected_datasets),
        "statuses": dict(statuses),
        "backend_audit": dict(backend),
        "elapsed_seconds": float(elapsed_seconds),
        "estimated_full_seconds": float(estimated_full_seconds),
        "disk_projection": asdict(disk_projection),
        "estimate_method": "linear_time_and_schema_bounded_disk",
    }
    if set(payload) != SMOKE_SCHEMA_KEYS:
        raise ValueError("smoke artifact schema is invalid")
    return payload


def _write_smoke(output: Path, payload: Mapping[str, Any]) -> int:
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    smoke_file = output / SMOKE_FILE
    try:
        descriptor = os.open(
            smoke_file,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            raw = canonical_json_bytes(payload) + b"\n"
            written = 0
            while written < len(raw):
                count = os.write(descriptor, raw[written:])
                if count <= 0:
                    raise OSError("Exp417 smoke write made no progress")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return len(raw)
    except BaseException:
        try:
            if smoke_file.exists():
                smoke_file.unlink()
            os.rmdir(output)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verified_head = require_clean_git()
    assert_label_free_sql_contract()
    paths = (
        args.database,
        args.exp291_prediction,
        args.exp291_manifest,
        args.exp292_prediction,
        args.exp292_manifest,
    )
    lanes = freeze_lane_inputs(*paths[1:])
    frozen_context = freeze_run_context(
        paths[0], lanes, verified_code_commit=verified_head
    )
    metadata = load_train_metadata(paths[0], frozen_context.datasets)
    selected = select_train_strata(metadata, args.smoke_datasets)
    smoke = SmokeContext(selected, args.output_dir)
    backend_path = _backend_audit_path(paths[0])
    backend = load_frozen_backend_audit(backend_path)
    resources = RunnerResources(*paths, backend_audit_path=backend_path)
    runtime = RunnerRuntime.production(resources)
    started = perf_counter()
    results = tuple(
        run_dataset(dataset, frozen_context, runtime=runtime)
        for dataset in smoke.selected_datasets
    )
    elapsed = perf_counter() - started
    statuses = {result.dataset: result.status for result in results}
    estimated_seconds = elapsed * FROZEN_CONFIG.expected_datasets / len(results)
    projection = estimate_disk_projection(
        frozen_context,
        results,
        normal_train_series_count=sum(item.normal_count for item in metadata),
        measured_smoke_artifact_bytes=0,
    )
    for _ in range(4):
        payload = render_smoke_context(
            smoke,
            statuses=statuses,
            backend=asdict(backend),
            elapsed_seconds=elapsed,
            estimated_full_seconds=estimated_seconds,
            disk_projection=projection,
        )
        measured = len(canonical_json_bytes(payload)) + 1
        if measured == projection.measured_smoke_artifact_bytes:
            break
        projection = estimate_disk_projection(
            frozen_context,
            results,
            normal_train_series_count=sum(item.normal_count for item in metadata),
            measured_smoke_artifact_bytes=measured,
        )
    else:
        raise ValueError("smoke artifact byte projection did not stabilize")
    verify_input_snapshots(*paths)
    _write_smoke(smoke.output_dir, payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
