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
EXPECTED_RAW_DATABASE_DATASETS = 1119
FROZEN_EXCLUDED_DATASETS = frozenset({"CornellWhaleChallenge", "Wafer_normal_1"})
SOURCE_NAMES = ("statistical", "structural", "change_point", "reconstruction")

BASELINE_SOURCE = "guarded_standard_to_hard_expansion"
BASELINE_EXPERIMENT_ID = "experiment_253_guarded_standard_to_hard_expansion"
EXP260_AUDIT_PATH = (
    SOURCE_CONTRACT["missing_override"].artifact_path.parent / "audit.json"
)

BASELINE_SCHEMA = {
    "baseline_hard_indices": list,
    "baseline_priority_indices": list,
    "baseline_standard_indices": list,
    "contains_evaluation_metrics": bool,
    "contains_test_labels": bool,
    "dataset": str,
    "deployment_eligible": bool,
    "experiment": int,
    "experiment_id": str,
    "final_hard_indices": list,
    "final_priority_indices": list,
    "final_standard_indices": list,
    "grade_a_indices": list,
    "grade_b_indices": list,
    "grade_s_indices": list,
    "mini1024_confirmation_indices": list,
    "mini4096_confirmation_indices": list,
    "n_test": int,
    "promoted_indices": list,
    "retrospective": bool,
    "selected_from_prior_test_review": bool,
    "source": str,
}

MISSING_OVERRIDE_SCHEMA = {
    "baseline_hard_indices": list,
    "contains_evaluation_metrics": bool,
    "contains_test_labels": bool,
    "dataset": str,
    "final_hard_indices": list,
    "final_priority_indices": list,
    "final_standard_indices": list,
    "n_test": int,
    "n_train": int,
    "promoted_indices": list,
    "quality_blocked_indices": list,
    "quality_boundary": list,
    "quality_flag_indices": list,
    "retrospective": bool,
    "source_indices": dict,
}


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
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceContractError(f"cannot read source artifact: {path}") from exc
    if any(type(row) is not dict for row in rows):
        raise SourceContractError(f"source artifact row is not an object: {path}")
    return rows


def _read_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceContractError(f"cannot read {description}: {path}") from exc
    if type(value) is not dict:
        raise SourceContractError(f"{description} is not an object: {path}")
    return value


def _canonical_alpha(value: object, *, source: str, expected: float) -> bool:
    if type(value) is not float:
        raise SourceContractError(f"alpha is not canonical: {source}")
    return value == expected


def select_rows(path: Path, variant: str, alpha: float) -> dict[str, dict[str, object]]:
    """Select one exact `(dataset, variant, alpha)` row from a JSONL artifact."""

    selected: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(path):
        if row.get("variant") != variant:
            continue
        matches_alpha = _canonical_alpha(
            row.get("alpha"), source=str(path), expected=alpha
        )
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
    metadata: dict[str, tuple[int, int]] = {}
    for name, n_train, n_test in rows:
        if type(name) is not str or type(n_train) is not int or type(n_test) is not int:
            raise SourceContractError("database metadata has invalid types")
        if name in metadata:
            raise SourceContractError(f"duplicate DB dataset name: {name}")
        metadata[name] = (n_train, n_test)
    return metadata


def _validate_manifest(
    name: str, spec: SourceSpec, *, artifact_row_count: int, dataset_count: int
) -> dict[str, object]:
    manifest_path = spec.artifact_path.parent / "prediction_manifest.json"
    manifest = _read_json_object(manifest_path, description="source manifest")
    expected_hash = manifest.get("prediction_sha256", manifest.get("sha256"))
    if type(expected_hash) is not str:
        raise SourceContractError(f"source manifest SHA256 missing: {name}")
    actual_hash = sha256_file(spec.artifact_path)
    if expected_hash != actual_hash:
        raise SourceContractError(
            f"source SHA256 mismatch: {name} expected={expected_hash} actual={actual_hash}"
        )
    if manifest.get("contains_test_labels") is not False:
        raise SourceContractError(f"source manifest contains TEST labels: {name}")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise SourceContractError(f"source manifest contains evaluation metrics: {name}")
    if "rows" not in manifest and "datasets" not in manifest:
        raise SourceContractError(f"source manifest row count missing: {name}")
    if "rows" in manifest and (
        type(manifest["rows"]) is not int or manifest["rows"] != artifact_row_count
    ):
        raise SourceContractError(
            f"source manifest row count mismatch: {name} "
            f"expected={artifact_row_count} actual={manifest['rows']}"
        )
    if "datasets" in manifest and (
        type(manifest["datasets"]) is not int
        or manifest["datasets"] != dataset_count
    ):
        raise SourceContractError(
            f"source manifest dataset count mismatch: {name} "
            f"expected={dataset_count} actual={manifest['datasets']}"
        )
    return manifest


def _validate_label_free_row(name: str, dataset: str, row: Mapping[str, object]) -> None:
    if row.get("contains_test_labels") is not False:
        raise SourceContractError(f"source row contains TEST labels: {name} dataset={dataset}")
    if row.get("contains_evaluation_metrics") is not False:
        raise SourceContractError(
            f"source row contains evaluation metrics: {name} dataset={dataset}"
        )


def _validate_row_schema(
    *, source: str, dataset: str, row: Mapping[str, object], schema: Mapping[str, type]
) -> None:
    actual_keys = set(row)
    expected_keys = set(schema)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise SourceContractError(
            f"source schema mismatch: {source} dataset={dataset} "
            f"missing={missing} extra={extra}"
        )
    for field, expected_type in schema.items():
        if type(row[field]) is not expected_type:
            raise SourceContractError(
                f"source schema type mismatch: {source} dataset={dataset} field={field}"
            )


def _indices(value: object, *, source: str, dataset: str) -> frozenset[int]:
    if not isinstance(value, list):
        raise SourceContractError(f"source indices are not a list: {source} dataset={dataset}")
    if any(type(index) is not int for index in value):
        raise SourceContractError(f"source has non-integral index: {source} dataset={dataset}")
    return frozenset(value)


def _required_indices(
    row: Mapping[str, object], *, field: str, source: str, dataset: str
) -> frozenset[int]:
    if field not in row:
        raise SourceContractError(f"missing required field: {field}")
    return _indices(row[field], source=source, dataset=dataset)


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
        if field.endswith("_indices") and field != "source_indices":
            _validate_index_range(
                source=source,
                dataset=dataset,
                n_test=n_test,
                indices=_indices(value, source=source, dataset=dataset),
            )
    nested = row.get("source_indices")
    if nested is None:
        return
    if type(nested) is not dict:
        raise SourceContractError(f"source indices are not a mapping: {source} dataset={dataset}")
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
    if "n_test" not in row:
        raise SourceContractError(f"source n_test missing: {source} dataset={dataset}")
    row_n_test = row["n_test"]
    if type(row_n_test) is not int:
        raise SourceContractError(f"source n_test is not an integer: {source} dataset={dataset}")
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
        raise SourceContractError(f"source experiment missing: {name}")
    actual = row["experiment"]
    if type(actual) is not int:
        raise SourceContractError(f"source experiment is invalid: {name}")
    expected = _expected_experiment(spec)
    if actual != expected:
        raise SourceContractError(
            f"source experiment mismatch: {name} expected={expected} actual={actual}"
        )


def _validate_baseline_identity(row: Mapping[str, object]) -> None:
    if row["source"] != BASELINE_SOURCE:
        raise SourceContractError("baseline source identity mismatch")
    if row["experiment_id"] != BASELINE_EXPERIMENT_ID:
        raise SourceContractError("baseline experiment identity mismatch")
    if row["selected_from_prior_test_review"] is not True:
        raise SourceContractError("baseline review provenance mismatch")
    if row["deployment_eligible"] is not False:
        raise SourceContractError("baseline deployment provenance mismatch")


def _frozen_database_names(metadata: Mapping[str, tuple[int, int]]) -> set[str]:
    raw_names = set(metadata)
    if len(raw_names) != EXPECTED_RAW_DATABASE_DATASETS:
        raise SourceContractError(
            "raw DB dataset count mismatch: "
            f"expected={EXPECTED_RAW_DATABASE_DATASETS} actual={len(raw_names)}"
        )
    if not FROZEN_EXCLUDED_DATASETS <= raw_names:
        missing = sorted(FROZEN_EXCLUDED_DATASETS - raw_names)
        raise SourceContractError(f"frozen exclusion missing from raw DB: {missing}")
    frozen_names = raw_names - FROZEN_EXCLUDED_DATASETS
    if len(frozen_names) != EXPECTED_POPULATION:
        raise SourceContractError(
            f"frozen DB population count mismatch: expected={EXPECTED_POPULATION} "
            f"actual={len(frozen_names)}"
        )
    return frozen_names


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
        _validate_row_schema(
            source="baseline", dataset=dataset, row=row, schema=BASELINE_SCHEMA
        )
        _validate_label_free_row("baseline", dataset, row)
        _validate_experiment("baseline", spec, row)
        _validate_baseline_identity(row)
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
    frozen_names = _frozen_database_names(metadata)
    missing_from_db = sorted(set(baseline) - set(metadata))
    if missing_from_db:
        raise SourceContractError(f"source dataset absent from DB: {missing_from_db[:5]}")
    _validate_coverage(
        source="baseline", actual=set(baseline), expected=frozen_names
    )
    return tuple(sorted(frozen_names))


def _finalized_rows(name: str) -> dict[str, dict[str, object]]:
    spec = SOURCE_CONTRACT[name]
    selected: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(spec.artifact_path):
        dataset = str(row.get("dataset", ""))
        if not dataset:
            raise SourceContractError(f"source row has no dataset: {name}")
        if dataset in selected:
            raise SourceContractError(f"duplicate source row: {dataset}")
        if name == "missing_override":
            _validate_row_schema(
                source=name,
                dataset=dataset,
                row=row,
                schema=MISSING_OVERRIDE_SCHEMA,
            )
        _validate_label_free_row(name, dataset, row)
        selected[dataset] = row
    return selected


def _validate_baseline_provenance() -> None:
    spec = SOURCE_CONTRACT["baseline"]
    audit = _read_json_object(spec.artifact_path.parent / "audit.json", description="baseline audit")
    if audit.get("datasets") != EXPECTED_POPULATION:
        raise SourceContractError("baseline audit dataset count mismatch")
    if not _canonical_alpha(audit.get("alpha"), source="baseline audit", expected=0.005):
        raise SourceContractError("baseline audit alpha mismatch")
    for field, expected in (
        ("contains_test_labels", False),
        ("retrospective", True),
        ("deployment_eligible", False),
        ("selected_from_prior_test_review", True),
    ):
        if audit.get(field) is not expected:
            raise SourceContractError(f"baseline audit provenance mismatch: {field}")


def _validate_missing_override_provenance() -> None:
    spec = SOURCE_CONTRACT["missing_override"]
    if spec.experiment != "exp260_262" or spec.variant != "corrected_four_sources":
        raise SourceContractError("missing override config provenance mismatch")
    if type(spec.alpha) is not float or spec.alpha != 0.005:
        raise SourceContractError("missing override config alpha mismatch")
    audit = _read_json_object(EXP260_AUDIT_PATH, description="missing override audit")
    if audit.get("experiments") != [260, 261, 262]:
        raise SourceContractError("missing override experiment provenance mismatch")
    if audit.get("datasets") != EXPECTED_MISSING_OVERRIDE_DATASETS:
        raise SourceContractError("missing override audit dataset count mismatch")
    if not _canonical_alpha(
        audit.get("alpha"), source="missing override audit", expected=spec.alpha
    ):
        raise SourceContractError("missing override audit alpha mismatch")
    if audit.get("contains_test_labels") is not False or audit.get("retrospective") is not True:
        raise SourceContractError("missing override audit flags mismatch")


def load_source_maps() -> FrozenSourceMaps:
    """Load and validate every frozen artifact without reading TEST labels."""

    raw_rows = {
        name: _read_jsonl(spec.artifact_path)
        for name, spec in SOURCE_CONTRACT.items()
    }
    manifests = {
        name: _validate_manifest(
            name,
            spec,
            artifact_row_count=len(raw_rows[name]),
            dataset_count=len({str(row.get("dataset", "")) for row in raw_rows[name]}),
        )
        for name, spec in SOURCE_CONTRACT.items()
    }
    del manifests  # Manifest validation is a contract gate, not routing input.
    _validate_baseline_provenance()
    _validate_missing_override_provenance()
    metadata = _database_metadata()
    frozen_names = _frozen_database_names(metadata)
    baseline = _baseline_rows()
    missing_from_db = sorted(set(baseline) - set(metadata))
    if missing_from_db:
        raise SourceContractError(f"source dataset absent from DB: {missing_from_db[:5]}")
    _validate_coverage(source="baseline", actual=set(baseline), expected=frozen_names)
    population = tuple(sorted(frozen_names))

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
            _validate_index_range(
                source=name,
                dataset=dataset,
                n_test=n_test[dataset],
                indices=_required_indices(
                    row, field="candidate_indices", source=name, dataset=dataset
                ),
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
        _validate_index_range(
            source="missing_override",
            dataset=dataset,
            n_test=n_test[dataset],
            indices=_required_indices(
                row,
                field="quality_flag_indices",
                source="missing_override",
                dataset=dataset,
            ),
        )
        nested = row.get("source_indices")
        if type(nested) is not dict or set(nested) != set(SOURCE_NAMES):
            raise SourceContractError(f"missing override has invalid source map: {dataset}")

    source_indices: dict[str, dict[str, frozenset[int]]] = {
        name: {
            dataset: _required_indices(
                row, field="candidate_indices", source=name, dataset=dataset
            )
            for dataset, row in rows.items()
        }
        for name, rows in selected_rows.items()
    }
    quality_flags: dict[str, frozenset[int]] = {}
    override_source_indices: dict[str, dict[str, frozenset[int]]] = {}
    for dataset, row in override.items():
        nested = row["source_indices"]
        assert type(nested) is dict
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
    metadata = _database_metadata()
    raw_database_dataset_count = len(metadata)
    _frozen_database_names(metadata)
    coverage = {
        name: len(values) for name, values in sources.source_indices.items()
    }
    overrides = set(sources.missing_override_rows)
    audit: dict[str, object] = {
        "raw_database_dataset_count": raw_database_dataset_count,
        "frozen_excluded_datasets": sorted(FROZEN_EXCLUDED_DATASETS),
        "frozen_population_count": len(sources.population),
        "dataset_count": len(sources.population),
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
        "source_contract": {
            "baseline": {
                "experiment": SOURCE_CONTRACT["baseline"].experiment,
                "variant": SOURCE_CONTRACT["baseline"].variant,
                "provenance_audit": str(
                    SOURCE_CONTRACT["baseline"].artifact_path.parent / "audit.json"
                ),
            },
            "missing_override": {
                "experiment": SOURCE_CONTRACT["missing_override"].experiment,
                "variant": SOURCE_CONTRACT["missing_override"].variant,
                "alpha": SOURCE_CONTRACT["missing_override"].alpha,
                "provenance_audit": str(EXP260_AUDIT_PATH),
            },
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
