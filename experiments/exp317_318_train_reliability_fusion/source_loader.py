from __future__ import annotations

from pathlib import Path

from experiments.exp309_316_fn_recovery.config import (
    EXP291_MANIFEST,
    EXP291_PREDICTION,
)
from experiments.exp309_316_fn_recovery.contracts import verify_prediction_rows

from .config import EXPECTED_DATASETS, SOURCE_EXPERIMENTS, SOURCE_ROOTS
from .contracts import sha256_file


def validate_source_dataset_sets(
    source_sets: dict[str, set[str]], *, frozen_datasets: set[str], expected: int,
) -> None:
    if len(frozen_datasets) != int(expected):
        raise ValueError(f"frozen evaluation coverage mismatch: {len(frozen_datasets)}/{expected}")
    for source, datasets in source_sets.items():
        if datasets != frozen_datasets:
            missing = sorted(frozen_datasets - datasets)[:5]
            extra = sorted(datasets - frozen_datasets)[:5]
            raise ValueError(
                f"source coverage mismatch: source={source} rows={len(datasets)}/{expected} "
                f"missing={missing} extra={extra}"
            )


def load_verified_sources(
    source_roots: dict[str, Path] | None = None,
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, str], set[str]]:
    roots = source_roots or SOURCE_ROOTS
    baseline_rows = verify_prediction_rows(EXP291_PREDICTION, EXP291_MANIFEST)
    frozen_datasets = {str(row["dataset"]) for row in baseline_rows}
    sources: dict[str, dict[str, dict[str, object]]] = {}
    hashes: dict[str, str] = {}
    for source, number in SOURCE_EXPERIMENTS.items():
        root = Path(roots[source])
        prediction = root / f"exp{number}_frozen_predictions.jsonl"
        manifest = root / f"exp{number}_prediction_manifest.json"
        rows = verify_prediction_rows(prediction, manifest)
        by_dataset = {str(row["dataset"]): row for row in rows}
        if len(by_dataset) != len(rows):
            raise ValueError(f"duplicate source dataset rows: source={source}")
        sources[source] = by_dataset
        hashes[source] = sha256_file(prediction)
    validate_source_dataset_sets(
        {source: set(rows) for source, rows in sources.items()},
        frozen_datasets=frozen_datasets, expected=EXPECTED_DATASETS,
    )
    return sources, hashes, frozen_datasets
