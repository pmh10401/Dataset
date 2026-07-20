from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .config import DB_PATH, OUTPUT_ROOT, SOURCE_CONTRACT, SourceSpec


EXPECTED_POPULATION = 1117
EXPECTED_MISSING_OVERRIDE_DATASETS = 20
SOURCE_NAMES = ("statistical", "structural", "change_point", "reconstruction")


@dataclass(frozen=True)
class FrozenSourceMaps:
    """Validated, label-free inputs consumed by the Exp263 routing stage."""

    population: tuple[str, ...]
    n_train: Mapping[str, int]
    n_test: Mapping[str, int]
    baseline_rows: Mapping[str, Mapping[str, object]]
    source_indices: Mapping[str, Mapping[str, frozenset[int]]]
    missing_override_rows: Mapping[str, Mapping[str, object]]
    override_source_indices: Mapping[str, Mapping[str, frozenset[int]]]
    quality_flags: Mapping[str, frozenset[int]]


class SourceContractError(ValueError):
    """Raised when a frozen source cannot safely feed the label-free pipeline."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceContractError(f"cannot read source artifact: {path}") from exc


def select_rows(path: Path, variant: str, alpha: float) -> dict[str, dict[str, object]]:
    """Select one exact `(dataset, variant, alpha)` row from a JSONL artifact."""

    selected: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(path):
        if row.get("variant") != variant:
            continue
        try:
            matches_alpha = abs(float(row.get("alpha")) - alpha) <= 1e-12
        except (TypeError, ValueError):
            matches_alpha = False
        if not matches_alpha:
            continue
        dataset = str(row.get("dataset", ""))
        if not dataset:
            raise SourceContractError(f"source row has no dataset: {path}")
        if dataset in selected:
            raise SourceContractError(f"duplicate source row: {dataset}")
        selected[dataset] = row
    return selected


def _database_metadata() -> dict[str, tuple[int, int]]:
    uri = f"file:{DB_PATH}?mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True)
        rows = connection.execute(
            "SELECT name, train_normal_count, test_total_count "
            "FROM datasets ORDER BY name"
        ).fetchall()
    except sqlite3.Error as exc:
        raise SourceContractError(f"cannot read source database: {DB_PATH}") from exc
    finally:
        if connection is not None:
            connection.close()
    return {str(name): (int(n_train), int(n_test)) for name, n_train, n_test in rows}


def _validate_manifest(name: str, spec: SourceSpec) -> dict[str, object]:
    manifest_path = spec.artifact_path.parent / "prediction_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceContractError(f"cannot read source manifest: {manifest_path}") from exc
    expected_hash = manifest.get("prediction_sha256", manifest.get("sha256"))
    actual_hash = sha256_file(spec.artifact_path)
    if expected_hash != actual_hash:
        raise SourceContractError(
            f"source SHA256 mismatch: {name} expected={expected_hash} actual={actual_hash}"
        )
    if manifest.get("contains_test_labels") is not False:
        raise SourceContractError(f"source manifest contains TEST labels: {name}")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise SourceContractError(f"source manifest contains evaluation metrics: {name}")
    return manifest


def _validate_label_free_row(name: str, dataset: str, row: Mapping[str, object]) -> None:
    if row.get("contains_test_labels") is not False:
        raise SourceContractError(f"source row contains TEST labels: {name} dataset={dataset}")
    if row.get("contains_evaluation_metrics") is not False:
        raise SourceContractError(
            f"source row contains evaluation metrics: {name} dataset={dataset}"
        )


def _indices(value: object, *, source: str, dataset: str) -> frozenset[int]:
    if not isinstance(value, list):
        raise SourceContractError(f"source indices are not a list: {source} dataset={dataset}")
    try:
        return frozenset(int(index) for index in value)
    except (TypeError, ValueError) as exc:
        raise SourceContractError(f"source has non-integer index: {source} dataset={dataset}") from exc


def _validate_index_range(
    *, source: str, dataset: str, n_test: int, indices: frozenset[int]
) -> None:
    if any(index < 0 or index >= n_test for index in indices):
        raise SourceContractError(
            f"candidate outside TEST range: source={source} dataset={dataset} n_test={n_test}"
        )


def _validate_row_indices(
    *, source: str, dataset: str, n_test: int, row: Mapping[str, object]
) -> None:
    for field, value in row.items():
        if field.endswith("_indices") and isinstance(value, list):
            _validate_index_range(
                source=source,
                dataset=dataset,
                n_test=n_test,
                indices=_indices(value, source=source, dataset=dataset),
            )
    nested = row.get("source_indices")
    if isinstance(nested, Mapping):
        for source_name, value in nested.items():
            _validate_index_range(
                source=f"{source}.{source_name}",
                dataset=dataset,
                n_test=n_test,
                indices=_indices(value, source=source, dataset=dataset),
            )


def _validate_n_test(
    *, source: str, dataset: str, row: Mapping[str, object], expected_n_test: int
) -> None:
    try:
        row_n_test = int(row["n_test"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceContractError(f"source n_test missing: {source} dataset={dataset}") from exc
    if row_n_test != expected_n_test:
        raise SourceContractError(
            f"source n_test mismatch: {source} dataset={dataset} "
            f"expected={expected_n_test} actual={row_n_test}"
        )


def _expected_experiment(spec: SourceSpec) -> int:
    suffix = spec.experiment.removeprefix("exp").split("_")[0]
    return int(suffix)


def _validate_experiment(name: str, spec: SourceSpec, row: Mapping[str, object]) -> None:
    if "experiment" not in row:
        return
    try:
        actual = int(row["experiment"])
    except (TypeError, ValueError) as exc:
        raise SourceContractError(f"source experiment is invalid: {name}") from exc
    expected = _expected_experiment(spec)
    if actual != expected:
        raise SourceContractError(
            f"source experiment mismatch: {name} expected={expected} actual={actual}"
        )


def _validate_coverage(
    *, source: str, actual: set[str], expected: set[str]
) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)[:5]
    extra = sorted(actual - expected)[:5]
    raise SourceContractError(
        f"source coverage mismatch: {source} missing={missing} extra={extra}"
    )


def _baseline_rows() -> dict[str, dict[str, object]]:
    spec = SOURCE_CONTRACT["baseline"]
    rows = _read_jsonl(spec.artifact_path)
    selected: dict[str, dict[str, object]] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        if not dataset:
            raise SourceContractError("baseline source row has no dataset")
        if dataset in selected:
            raise SourceContractError(f"duplicate source row: {dataset}")
        _validate_label_free_row("baseline", dataset, row)
        _validate_experiment("baseline", spec, row)
        selected[dataset] = row
    if len(selected) != EXPECTED_POPULATION:
        raise SourceContractError(
            f"baseline must contain {EXPECTED_POPULATION} unique datasets, found {len(selected)}"
        )
    return selected


def load_population() -> tuple[str, ...]:
    """Return the frozen 1,117-dataset population in deterministic name order."""

    metadata = _database_metadata()
    baseline = _baseline_rows()
    population = tuple(sorted(baseline))
    if len(population) != EXPECTED_POPULATION:
        raise SourceContractError(
            f"population must contain {EXPECTED_POPULATION} unique datasets, found {len(population)}"
        )
    missing_from_db = sorted(set(population) - set(metadata))
    if missing_from_db:
        raise SourceContractError(f"source dataset absent from DB: {missing_from_db[:5]}")
    return population


def _finalized_rows(name: str) -> dict[str, dict[str, object]]:
    spec = SOURCE_CONTRACT[name]
    selected: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(spec.artifact_path):
        dataset = str(row.get("dataset", ""))
        if not dataset:
            raise SourceContractError(f"source row has no dataset: {name}")
        if dataset in selected:
            raise SourceContractError(f"duplicate source row: {dataset}")
        _validate_label_free_row(name, dataset, row)
        _validate_experiment(name, spec, row)
        selected[dataset] = row
    return selected


def load_source_maps() -> FrozenSourceMaps:
    """Load and validate every frozen artifact without reading TEST labels."""

    manifests = {
        name: _validate_manifest(name, spec) for name, spec in SOURCE_CONTRACT.items()
    }
    del manifests  # Manifest validation is a contract gate, not routing input.
    metadata = _database_metadata()
    baseline = _baseline_rows()
    population = tuple(sorted(baseline))
    if len(population) != EXPECTED_POPULATION:
        raise SourceContractError(f"population coverage mismatch: {len(population)}")
    _validate_coverage(source="baseline", actual=set(baseline), expected=set(population))

    n_train = {dataset: metadata[dataset][0] for dataset in population}
    n_test = {dataset: metadata[dataset][1] for dataset in population}
    for dataset, row in baseline.items():
        _validate_n_test(
            source="baseline", dataset=dataset, row=row, expected_n_test=n_test[dataset]
        )
        _validate_row_indices(
            source="baseline", dataset=dataset, n_test=n_test[dataset], row=row
        )

    selected_rows: dict[str, dict[str, dict[str, object]]] = {}
    for name in SOURCE_NAMES:
        spec = SOURCE_CONTRACT[name]
        if spec.alpha is None:
            raise SourceContractError(f"source alpha is required: {name}")
        rows = select_rows(spec.artifact_path, spec.variant, spec.alpha)
        selected_rows[name] = rows
        expected = (
            set(population)
            if name != "reconstruction"
            else {dataset for dataset in population if n_train[dataset] >= 100}
        )
        _validate_coverage(source=name, actual=set(rows), expected=expected)
        for dataset, row in rows.items():
            _validate_label_free_row(name, dataset, row)
            _validate_experiment(name, spec, row)
            _validate_n_test(
                source=name, dataset=dataset, row=row, expected_n_test=n_test[dataset]
            )
            _validate_row_indices(
                source=name, dataset=dataset, n_test=n_test[dataset], row=row
            )

    override = _finalized_rows("missing_override")
    if len(override) != EXPECTED_MISSING_OVERRIDE_DATASETS:
        raise SourceContractError(
            "missing override must contain "
            f"{EXPECTED_MISSING_OVERRIDE_DATASETS} unique datasets, found {len(override)}"
        )
    _validate_coverage(
        source="missing_override", actual=set(override), expected=set(override)
    )
    for dataset, row in override.items():
        if dataset not in n_test:
            raise SourceContractError(f"source dataset absent from DB: {dataset}")
        _validate_n_test(
            source="missing_override", dataset=dataset, row=row, expected_n_test=n_test[dataset]
        )
        _validate_row_indices(
            source="missing_override", dataset=dataset, n_test=n_test[dataset], row=row
        )
        nested = row.get("source_indices")
        if not isinstance(nested, Mapping) or set(nested) != set(SOURCE_NAMES):
            raise SourceContractError(f"missing override has invalid source map: {dataset}")

    source_indices: dict[str, dict[str, frozenset[int]]] = {
        name: {
            dataset: _indices(row["candidate_indices"], source=name, dataset=dataset)
            for dataset, row in rows.items()
        }
        for name, rows in selected_rows.items()
    }
    quality_flags: dict[str, frozenset[int]] = {}
    override_source_indices: dict[str, dict[str, frozenset[int]]] = {}
    for dataset, row in override.items():
        nested = row["source_indices"]
        assert isinstance(nested, Mapping)
        override_source_indices[dataset] = {
            name: _indices(
                nested[name], source=f"missing_override.{name}", dataset=dataset
            )
            for name in SOURCE_NAMES
        }
        quality_flags[dataset] = _indices(
            row["quality_flag_indices"], source="missing_override", dataset=dataset
        )

    return FrozenSourceMaps(
        population=population,
        n_train=MappingProxyType(n_train),
        n_test=MappingProxyType(n_test),
        baseline_rows=MappingProxyType(
            {dataset: MappingProxyType(row) for dataset, row in baseline.items()}
        ),
        source_indices=MappingProxyType(
            {name: MappingProxyType(values) for name, values in source_indices.items()}
        ),
        missing_override_rows=MappingProxyType(
            {dataset: MappingProxyType(row) for dataset, row in override.items()}
        ),
        override_source_indices=MappingProxyType(
            {
                dataset: MappingProxyType(values)
                for dataset, values in override_source_indices.items()
            }
        ),
        quality_flags=MappingProxyType(quality_flags),
    )


def _write_audit_files(audit: Mapping[str, object], sources: FrozenSourceMaps) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "00_source_contract_audit.json").write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "dataset",
        "n_train",
        "n_test",
        "baseline",
        "statistical",
        "structural",
        "change_point",
        "reconstruction",
        "missing_override",
    ]
    with (OUTPUT_ROOT / "01_source_coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        overrides = set(sources.missing_override_rows)
        for dataset in sources.population:
            writer.writerow(
                {
                    "dataset": dataset,
                    "n_train": sources.n_train[dataset],
                    "n_test": sources.n_test[dataset],
                    "baseline": True,
                    "statistical": dataset in sources.source_indices["statistical"],
                    "structural": dataset in sources.source_indices["structural"],
                    "change_point": dataset in sources.source_indices["change_point"],
                    "reconstruction": dataset in sources.source_indices["reconstruction"],
                    "missing_override": dataset in overrides,
                }
            )


def audit_source_contract() -> dict[str, object]:
    """Validate frozen inputs and materialize the label-free source audit."""

    sources = load_source_maps()
    coverage = {
        name: len(values) for name, values in sources.source_indices.items()
    }
    overrides = set(sources.missing_override_rows)
    audit: dict[str, object] = {
        "dataset_count": len(sources.population),
        "database_population_count": len(sources.population),
        "baseline_coverage": len(sources.baseline_rows),
        "statistical_coverage": coverage["statistical"],
        "structural_coverage": coverage["structural"],
        "change_point_coverage": coverage["change_point"],
        "reconstruction_coverage": coverage["reconstruction"],
        "missing_override_coverage": len(overrides),
        "missing_datasets_without_override": [],
        "input_sha256": {
            name: sha256_file(spec.artifact_path)
            for name, spec in SOURCE_CONTRACT.items()
        },
        "source_manifests": {
            name: str(spec.artifact_path.parent / "prediction_manifest.json")
            for name, spec in SOURCE_CONTRACT.items()
        },
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "database_read_only": True,
    }
    _write_audit_files(audit, sources)
    return audit


def main() -> int:
    try:
        audit_source_contract()
    except SourceContractError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
