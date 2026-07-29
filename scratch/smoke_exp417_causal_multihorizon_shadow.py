#!/usr/bin/env python3
"""Non-official, label-free Exp417 readiness smoke."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
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
    estimated_full_disk_bytes: int,
) -> dict[str, Any]:
    if not isinstance(context, SmokeContext):
        raise ValueError("smoke rendering requires SmokeContext")
    if tuple(statuses) != context.selected_datasets:
        raise ValueError("smoke statuses do not cover exact selected datasets")
    return {
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
        "estimated_full_disk_bytes": int(estimated_full_disk_bytes),
        "estimate_method": "linear_per_dataset_from_three_dataset_smoke",
    }


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
    require_clean_git()
    assert_label_free_sql_contract()
    paths = (
        args.database,
        args.exp291_prediction,
        args.exp291_manifest,
        args.exp292_prediction,
        args.exp292_manifest,
    )
    lanes = freeze_lane_inputs(*paths[1:])
    frozen_context = freeze_run_context(paths[0], lanes)
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
    provisional = render_smoke_context(
        smoke,
        statuses=statuses,
        backend=asdict(backend),
        elapsed_seconds=elapsed,
        estimated_full_seconds=estimated_seconds,
        estimated_full_disk_bytes=0,
    )
    representative_bytes = len(canonical_json_bytes(provisional)) + sum(
        len(repr(result).encode("utf-8")) for result in results
    )
    estimated_disk = round(
        representative_bytes * FROZEN_CONFIG.expected_datasets / len(results)
    )
    payload = render_smoke_context(
        smoke,
        statuses=statuses,
        backend=asdict(backend),
        elapsed_seconds=elapsed,
        estimated_full_seconds=estimated_seconds,
        estimated_full_disk_bytes=estimated_disk,
    )
    verify_input_snapshots(*paths)
    _write_smoke(smoke.output_dir, payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
