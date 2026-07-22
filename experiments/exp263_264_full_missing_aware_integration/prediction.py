from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .inputs import EXPECTED_POPULATION, FrozenSourceMaps, SOURCE_NAMES, load_source_maps
from .routing import RoutingInputs, route_dataset


EXPERIMENT = 263
EXPERIMENT_ID = "experiment_263_full_missing_aware_integration"
SOURCE = "full_missing_aware_integration"


def _json_ready(value: object) -> object:
    """Convert NumPy containers and scalars before serializing frozen JSON."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_ready(item) for item in sorted(value)]
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _sorted_indices(values: frozenset[int]) -> list[int]:
    return sorted(values)


def _source_indices_for(
    dataset: str, sources: FrozenSourceMaps
) -> tuple[dict[str, frozenset[int]], frozenset[int], bool]:
    if dataset in sources.missing_override_rows:
        return (
            dict(sources.override_source_indices[dataset]),
            sources.quality_flags[dataset],
            True,
        )
    return (
        {
            name: sources.source_indices[name].get(dataset, frozenset())
            for name in SOURCE_NAMES
        },
        frozenset(),
        False,
    )


def _route_row(dataset: str, sources: FrozenSourceMaps) -> dict[str, object]:
    baseline = sources.baseline_rows[dataset]
    source_indices, quality_flags, missing_override = _source_indices_for(dataset, sources)
    baseline_standard = set(baseline["final_standard_indices"])
    quality_blocked = {
        index
        for index in baseline_standard
        if sum(index in source_indices[name] for name in SOURCE_NAMES) >= 3
        and index in quality_flags
    }
    routed = route_dataset(
        RoutingInputs(
            dataset=dataset,
            n_train=sources.n_train[dataset],
            n_test=sources.n_test[dataset],
            baseline_hard=set(baseline["final_hard_indices"]),
            baseline_standard=set(baseline["final_standard_indices"]),
            baseline_priority=set(baseline["final_priority_indices"]),
            sources={name: set(indices) for name, indices in source_indices.items()},
            quality_flags=set(quality_flags),
        )
    )
    return {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "source": SOURCE,
        **routed,
        "source_indices": {
            name: _sorted_indices(source_indices[name]) for name in SOURCE_NAMES
        },
        "quality_flag_indices": _sorted_indices(quality_flags),
        "quality_blocked_indices": sorted(quality_blocked),
        "quality_flag_source": (
            "exp260_train_q99" if missing_override else "verified_nonmissing_empty"
        ),
        "missing_override": missing_override,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
    }


def _validate_rows(rows: list[dict[str, object]], sources: FrozenSourceMaps) -> None:
    names = [row["dataset"] for row in rows]
    if len(rows) != EXPECTED_POPULATION:
        raise ValueError(f"prediction row count mismatch: {len(rows)}")
    if any(type(name) is not str for name in names):
        raise ValueError("prediction dataset name is invalid")
    if tuple(names) != sources.population or len(set(names)) != EXPECTED_POPULATION:
        raise ValueError("prediction dataset coverage mismatch")

    for row in rows:
        dataset = str(row["dataset"])
        n_test = row["n_test"]
        if type(n_test) is not int:
            raise ValueError(f"prediction n_test is invalid: dataset={dataset}")
        lanes = [
            set(row[field])
            for field in (
                "final_hard_indices",
                "final_standard_indices",
                "final_priority_indices",
            )
        ]
        if any(
            lanes[left] & lanes[right]
            for left in range(len(lanes))
            for right in range(left + 1, len(lanes))
        ):
            raise ValueError(f"prediction lane overlap: dataset={dataset}")
        source_indices = row["source_indices"]
        if type(source_indices) is not dict or set(source_indices) != set(SOURCE_NAMES):
            raise ValueError(f"prediction source map is invalid: dataset={dataset}")
        bounded_indices = [
            *lanes,
            set(row["quality_flag_indices"]),
            set(row["quality_blocked_indices"]),
            *(set(source_indices[name]) for name in SOURCE_NAMES),
        ]
        if any(
            type(index) is not int or index < 0 or index >= n_test
            for lane in bounded_indices
            for index in lane
        ):
            raise ValueError(f"prediction index outside TEST range: dataset={dataset}")
        if not set(row["baseline_hard_indices"]) <= set(row["final_hard_indices"]):
            raise ValueError(f"baseline Hard was removed: dataset={dataset}")


def _agreement_diagnostic(row: Mapping[str, object]) -> dict[str, object]:
    source_indices = row["source_indices"]
    if type(source_indices) is not dict:
        raise ValueError(f"prediction source map is invalid: dataset={row['dataset']}")
    standard = set(row["baseline_standard_indices"])
    agreement = {
        index: sum(index in source_indices[name] for name in SOURCE_NAMES)
        for index in standard
    }
    return {
        "dataset": row["dataset"],
        "n_train": row["n_train"],
        "n_test": row["n_test"],
        "missing_override": row["missing_override"],
        "quality_flag_count": len(row["quality_flag_indices"]),
        "baseline_hard_count": len(row["baseline_hard_indices"]),
        "baseline_standard_count": len(standard),
        "baseline_priority_count": len(row["baseline_priority_indices"]),
        **{
            f"{name}_candidate_count": len(source_indices[name])
            for name in SOURCE_NAMES
        },
        **{
            f"agreement_{count}_count": sum(
                value == count for value in agreement.values()
            )
            for count in range(len(SOURCE_NAMES) + 1)
        },
        "promoted_count": len(row["promoted_indices"]),
        "quality_blocked_count": len(row["quality_blocked_indices"]),
        "final_hard_count": len(row["final_hard_indices"]),
        "final_standard_count": len(row["final_standard_indices"]),
        "final_priority_count": len(row["final_priority_indices"]),
    }


def _write_diagnostics(path: Path, rows: list[dict[str, object]]) -> None:
    diagnostics = [_agreement_diagnostic(row) for row in rows]
    fields = list(diagnostics[0])
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, newline="", encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(diagnostics)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def predict_all(output_root: Path) -> Path:
    """Build and freeze the full label-free Exp263 prediction artifact."""

    sources = load_source_maps()
    rows = [_route_row(dataset, sources) for dataset in sources.population]
    _validate_rows(rows, sources)
    data = b"".join(
        (
            json.dumps(_json_ready(row), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    prediction_path = output_root / "frozen_predictions.jsonl"
    _atomic_write(prediction_path, data)
    digest = hashlib.sha256(data).hexdigest()
    manifest = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "rows": len(rows),
        "datasets": len(rows),
        "missing_override_datasets": len(sources.missing_override_rows),
        "prediction_sha256": digest,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
    }
    _atomic_write(
        output_root / "prediction_manifest.json",
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    _write_diagnostics(output_root / "source_agreement_diagnostics.csv", rows)
    return prediction_path
