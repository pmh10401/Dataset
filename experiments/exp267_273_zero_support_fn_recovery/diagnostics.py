from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from .config import DB_PATH, OUTPUT_ROOT
from .dependencies import dependency_report
from .population import (
    EXPECTED_DATASETS,
    EXP266_OUTPUT_ROOT,
    PopulationContractError,
    frozen_input_hashes,
    frozen_input_paths,
    load_frozen_population,
    read_frozen_prediction_rows,
    validate_population,
)


EXPECTED_COMBINED_FN = 1553
EXPECTED_ZERO_SUPPORT_FN = 1405
EXPECTED_SINGLE_SOURCE_FN = 148
LANE_FIELDS = (
    "hard_indices",
    "shadow_indices",
    "priority_indices",
    "standard_indices",
    "recovery_shadow_indices",
    "recovery_priority_indices",
    "data_quality_review_indices",
)
TRAIN_TIERS = (
    ("<5", 0, 4),
    ("5-19", 5, 19),
    ("20-99", 20, 99),
    ("100-199", 100, 199),
    ("200-499", 200, 499),
    (">=500", 500, None),
)
TEST_SIZE_SLICES = (
    ("<100", 0, 99),
    ("100-249", 100, 249),
    ("250-499", 250, 499),
    (">=500", 500, None),
)


def _indices(value: object, field_name: str) -> set[int]:
    if value is None:
        return set()
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"{field_name} must be an index collection")
    values = set(value)
    if any(type(index) is not int or index < 0 for index in values):
        raise ValueError(f"{field_name} must contain non-negative integers")
    if len(values) != len(value):
        raise ValueError(f"{field_name} contains duplicate indices")
    return values


def _support_count(value: object) -> tuple[int, tuple[str, ...]]:
    if value is None:
        return 0, ()
    if type(value) is int:
        if value < 0:
            raise ValueError("support count must be non-negative")
        return value, ()
    if isinstance(value, (list, tuple, set, frozenset)):
        sources = tuple(sorted({str(source) for source in value}))
        return len(sources), sources
    raise ValueError("support must be an integer count or source collection")


def _support_mapping(
    support: Mapping[object, object] | None,
) -> dict[int, tuple[int, tuple[str, ...]]]:
    if support is None:
        return {}
    if not isinstance(support, Mapping):
        raise ValueError("support must be a mapping")
    normalized: dict[int, tuple[int, tuple[str, ...]]] = {}
    for raw_index, raw_value in support.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("support index must be an integer") from exc
        if index < 0:
            raise ValueError("support index must be non-negative")
        normalized[index] = _support_count(raw_value)
    return normalized


def _combined_indices(row: Mapping[str, object]) -> set[int]:
    if "combined_indices" in row:
        return _indices(row["combined_indices"], "combined_indices")
    combined: set[int] = set()
    for field in LANE_FIELDS:
        combined.update(_indices(row.get(field), field))
    return combined


def _support_from_prediction(row: Mapping[str, object]) -> dict[int, object]:
    source_values = row.get("support_sources")
    if isinstance(source_values, Mapping):
        by_index: dict[int, list[str]] = {}
        for raw_source, raw_indices in source_values.items():
            for index in _indices(raw_indices, f"support_sources.{raw_source}"):
                by_index.setdefault(index, []).append(str(raw_source))
        return {index: tuple(sources) for index, sources in by_index.items()}
    support = row.get("support_by_index", {})
    if support is None:
        return {}
    if not isinstance(support, Mapping):
        raise ValueError("support_by_index must be a mapping")
    return dict(support)


def diagnostic_row(
    *,
    true_indices: Iterable[int],
    combined_indices: Iterable[int],
    support: Mapping[object, object],
    dataset: str = "",
    n_train: int | None = None,
    n_test: int | None = None,
) -> dict[str, object]:
    """Build one label-backed diagnostic row from frozen candidate support."""

    truth = _indices(list(true_indices), "true_indices")
    combined = _indices(list(combined_indices), "combined_indices")
    support_by_index = _support_mapping(support)
    remaining = truth - combined
    support_counts = {
        index: support_by_index.get(index, (0, ()))[0] for index in remaining
    }
    support_sources = {
        index: list(support_by_index.get(index, (0, ()))[1])
        for index in remaining
    }
    zero_support = sorted(index for index in remaining if support_counts[index] == 0)
    single_source = sorted(index for index in remaining if support_counts[index] == 1)
    return {
        "dataset": dataset,
        "n_train": n_train,
        "n_test": n_test,
        "true_anomaly_count": len(truth),
        "combined_candidate_count": len(combined),
        "remaining_fn_indices": sorted(remaining),
        "remaining_fn_count": len(remaining),
        "zero_support_fn_indices": zero_support,
        "zero_support_fn_count": len(zero_support),
        "single_source_fn_indices": single_source,
        "single_source_fn_count": len(single_source),
        "support_count_by_index": support_counts,
        "support_sources_by_index": support_sources,
    }


def _as_int(row: Mapping[str, object], field: str, default: int = 0) -> int:
    value = row.get(field, default)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _summary_row(global_summary: Mapping[str, object] | Iterable[Mapping[str, object]] | None) -> Mapping[str, object] | None:
    if global_summary is None:
        return None
    if isinstance(global_summary, Mapping):
        return global_summary
    rows = list(global_summary)
    if len(rows) != 1:
        raise ValueError("06_global_summary.csv must contain exactly one row")
    return rows[0]


def build_zero_support_diagnostics(
    prediction_rows: Iterable[Mapping[str, object]],
    result_rows: Iterable[Mapping[str, object]] | None = None,
    true_indices_by_dataset: Mapping[str, Iterable[int]] | None = None,
    global_summary: Mapping[str, object] | Iterable[Mapping[str, object]] | None = None,
    expected_population: int | None = None,
    metadata_by_dataset: Mapping[str, Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Combine frozen candidates with separately loaded TEST truth.

    This function intentionally refuses to infer labels from prediction or result
    rows.  ``true_indices_by_dataset`` must be supplied by a separate diagnostic
    evaluation step after the frozen input audit has been persisted.
    """

    if true_indices_by_dataset is None:
        raise ValueError("label-backed diagnostics require explicit TEST truth")
    predictions = list(prediction_rows)
    names = [row.get("dataset") for row in predictions]
    if any(type(name) is not str or not name for name in names):
        raise ValueError("prediction dataset field is invalid")
    if len(names) != len(set(names)):
        raise PopulationContractError("prediction population contains duplicate datasets")
    population = validate_population(names, expected=expected_population) if expected_population is not None else tuple(sorted(names))
    truth_names = set(true_indices_by_dataset)
    if truth_names != set(population):
        raise PopulationContractError("label-backed population does not match predictions")

    results = {str(row.get("dataset")): row for row in (result_rows or ())}
    if results and set(results) != set(population):
        raise PopulationContractError("Exp266 result population does not match predictions")
    metadata = metadata_by_dataset or {}
    diagnostics: list[dict[str, object]] = []
    for row in predictions:
        dataset = str(row["dataset"])
        result = results.get(dataset, {})
        meta = metadata.get(dataset, {})
        diagnostic = diagnostic_row(
            true_indices=true_indices_by_dataset[dataset],
            combined_indices=_combined_indices(row),
            support=_support_from_prediction(row),
            dataset=dataset,
            n_train=_as_int(result, "n_train", _as_int(meta, "n_train", 0)),
            n_test=_as_int(result, "total_test_runs", _as_int(meta, "n_test", 0)),
        )
        if "combined_fn" in result and _as_int(result, "combined_fn") != diagnostic["remaining_fn_count"]:
            raise ValueError(f"Exp266 combined_fn mismatch: dataset={dataset}")
        diagnostics.append(diagnostic)
    diagnostics.sort(key=lambda row: str(row["dataset"]))

    summary = _summary_row(global_summary)
    totals = summarize_zero_support_diagnostics(diagnostics)
    if summary is not None:
        if "combined_fn" in summary and _as_int(summary, "combined_fn") != totals["combined_fn"]:
            raise ValueError("Exp266 global combined_fn does not independently reproduce")
        if expected_population == EXPECTED_DATASETS:
            if "datasets" in summary and _as_int(summary, "datasets") != EXPECTED_DATASETS:
                raise ValueError("Exp266 global summary population is incomplete")
            expected = {
                "combined_fn": EXPECTED_COMBINED_FN,
                "zero_support_fn": EXPECTED_ZERO_SUPPORT_FN,
                "single_source_fn": EXPECTED_SINGLE_SOURCE_FN,
            }
            if any(totals[key] != value for key, value in expected.items()):
                raise ValueError(
                    "Exp267 support totals failed contract: "
                    f"actual={totals} expected={expected}"
                )
    return diagnostics


def summarize_zero_support_diagnostics(
    diagnostics: Iterable[Mapping[str, object]],
) -> dict[str, int]:
    rows = list(diagnostics)
    return {
        "datasets": len(rows),
        "combined_fn": sum(_as_int(row, "remaining_fn_count") for row in rows),
        "zero_support_fn": sum(_as_int(row, "zero_support_fn_count") for row in rows),
        "single_source_fn": sum(_as_int(row, "single_source_fn_count") for row in rows),
        "datasets_with_remaining_fn": sum(
            _as_int(row, "remaining_fn_count") > 0 for row in rows
        ),
        "datasets_with_zero_support_fn": sum(
            _as_int(row, "zero_support_fn_count") > 0 for row in rows
        ),
    }


def load_test_truth(
    population: Iterable[str], *, database_path: Path = DB_PATH
) -> tuple[dict[str, set[int]], dict[str, dict[str, int]]]:
    """Read TEST labels for diagnostics only, using a read-only SQLite handle."""

    names = tuple(population)
    truth: dict[str, set[int]] = {}
    metadata: dict[str, dict[str, int]] = {}
    resolved = Path(database_path).resolve()
    uri = f"file:{resolved}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise ValueError(f"cannot open diagnostic database: {database_path}") from exc
    try:
        connection.execute("PRAGMA query_only = ON")
        for dataset in names:
            dataset_row = connection.execute(
                "SELECT train_normal_count, test_total_count FROM datasets WHERE name=?",
                (dataset,),
            ).fetchone()
            if dataset_row is None:
                raise ValueError(f"diagnostic dataset is missing from database: {dataset}")
            rows = connection.execute(
                """SELECT instance_index, CAST(label AS INTEGER)
                   FROM instances i JOIN datasets d ON d.id=i.dataset_id
                   WHERE d.name=? AND i.split='TEST' ORDER BY instance_index""",
                (dataset,),
            ).fetchall()
            indices = [int(index) for index, _ in rows]
            if indices != list(range(len(indices))):
                raise ValueError(f"TEST instance indices are not contiguous: {dataset}")
            labels = [int(label) for _, label in rows]
            if any(label not in (0, 1) for label in labels):
                raise ValueError(f"TEST labels are not binary: {dataset}")
            truth[dataset] = {index for index, label in enumerate(labels) if label == 1}
            metadata[dataset] = {
                "n_train": int(dataset_row[0]),
                "n_test": int(dataset_row[1]),
            }
    finally:
        connection.close()
    return truth, metadata


def _read_exp266_csvs(
    exp266_output_root: Path,
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]]:
    root = Path(exp266_output_root)
    result_path = root / "05_dataset_results.csv"
    summary_path = root / "06_global_summary.csv"
    with result_path.open(newline="", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    if len(summary_rows) != 1:
        raise ValueError("06_global_summary.csv must contain exactly one row")
    return result_rows, summary_rows[0], frozen_input_hashes(root)


def _write_csv(path: Path, rows: list[Mapping[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tier(value: int, definitions: tuple[tuple[str, int, int | None], ...]) -> str:
    for label, minimum, maximum in definitions:
        if value >= minimum and (maximum is None or value <= maximum):
            return label
    raise ValueError(f"value does not fit a frozen tier: {value}")


def _write_diagnostic_outputs(
    output_root: Path,
    diagnostics: list[dict[str, object]],
    audit: Mapping[str, object],
) -> dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    expanded: list[dict[str, object]] = []
    for row in diagnostics:
        for index in row["remaining_fn_indices"]:  # type: ignore[union-attr]
            sources = row["support_sources_by_index"][index]  # type: ignore[index]
            expanded.append(
                {
                    "dataset": row["dataset"],
                    "fn_index": index,
                    "support_count": row["support_count_by_index"][index],  # type: ignore[index]
                    "support_sources": ";".join(sources),
                    "zero_support": int(index in row["zero_support_fn_indices"]),  # type: ignore[operator]
                    "single_source": int(index in row["single_source_fn_indices"]),  # type: ignore[operator]
                    "n_train": row["n_train"],
                    "n_test": row["n_test"],
                    "train_tier": _tier(int(row["n_train"]), TRAIN_TIERS),
                    "test_size_slice": _tier(int(row["n_test"]), TEST_SIZE_SLICES),
                }
            )
    expanded.sort(key=lambda row: (str(row["dataset"]), int(row["fn_index"])))
    paths = {
        "dependency_audit": root / "00_dependency_audit.json",
        "population_contract": root / "01_population_contract.json",
        "dataset": root / "02_zero_support_fn_dataset.csv",
        "summary": root / "03_zero_support_fn_summary.csv",
        "train_tiers": root / "04_zero_support_fn_train_tiers.csv",
        "test_sizes": root / "05_zero_support_fn_test_size_diagnostic.csv",
        "report": root / "06_exp267_report.md",
    }
    paths["dependency_audit"].write_text(
        json.dumps(audit["dependency_audit"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["population_contract"].write_text(
        json.dumps(audit["population_contract"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(paths["dataset"], expanded, list(expanded[0]) if expanded else [
        "dataset", "fn_index", "support_count", "support_sources", "zero_support",
        "single_source", "n_train", "n_test", "train_tier", "test_size_slice",
    ])
    totals = summarize_zero_support_diagnostics(diagnostics)
    _write_csv(paths["summary"], [totals], list(totals))

    train_groups: dict[str, Counter[str]] = {}
    for row in expanded:
        train_groups.setdefault(str(row["train_tier"]), Counter())
        train_groups[str(row["train_tier"])]["remaining_fn"] += 1
        train_groups[str(row["train_tier"])]["zero_support_fn"] += int(row["zero_support"])
        train_groups[str(row["train_tier"])]["single_source_fn"] += int(row["single_source"])
    train_rows = [
        {
            "train_tier": label,
            "remaining_fn": train_groups.get(label, {}).get("remaining_fn", 0),
            "zero_support_fn": train_groups.get(label, {}).get("zero_support_fn", 0),
            "single_source_fn": train_groups.get(label, {}).get("single_source_fn", 0),
        }
        for label, _, _ in TRAIN_TIERS
    ]
    _write_csv(paths["train_tiers"], train_rows, list(train_rows[0]))

    test_groups: dict[str, Counter[str]] = {}
    for row in expanded:
        test_groups.setdefault(str(row["test_size_slice"]), Counter())
        test_groups[str(row["test_size_slice"])]["remaining_fn"] += 1
        test_groups[str(row["test_size_slice"])]["zero_support_fn"] += int(row["zero_support"])
        test_groups[str(row["test_size_slice"])]["single_source_fn"] += int(row["single_source"])
    test_rows = [
        {
            "test_size_slice": label,
            "remaining_fn": test_groups.get(label, {}).get("remaining_fn", 0),
            "zero_support_fn": test_groups.get(label, {}).get("zero_support_fn", 0),
            "single_source_fn": test_groups.get(label, {}).get("single_source_fn", 0),
        }
        for label, _, _ in TEST_SIZE_SLICES
    ]
    _write_csv(paths["test_sizes"], test_rows, list(test_rows[0]))

    paths["report"].write_text(
        "\n".join(
            [
                "# Exp267 Zero-Support FN Diagnostic",
                "",
                "This is a retrospective, label-backed diagnostic of the frozen Exp266 prediction population.",
                "It does not change predictions, lanes, thresholds, or routing.",
                "",
                "| Metric | Count |",
                "|---|---:|",
                f"| Frozen datasets | {totals['datasets']} |",
                f"| Remaining combined FN | {totals['combined_fn']} |",
                f"| Zero-support FN | {totals['zero_support_fn']} |",
                f"| Single-source FN | {totals['single_source_fn']} |",
                "",
                "Prediction hashes and input paths were written before TEST labels were read.",
                "TEST-size and TRAIN-tier slices are diagnostic views only.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


def write_exp267_outputs(
    *,
    output_root: Path = OUTPUT_ROOT,
    exp266_output_root: Path = EXP266_OUTPUT_ROOT,
    database_path: Path = DB_PATH,
) -> dict[str, Path]:
    """Run the read-only Exp267 freeze, then materialize its diagnostic outputs."""

    prediction_path, results_path, summary_path = frozen_input_paths(exp266_output_root)
    prediction_rows, prediction_sha256, manifest = read_frozen_prediction_rows(prediction_path)
    population = load_frozen_population(prediction_path)
    result_rows, global_summary, hashes = _read_exp266_csvs(exp266_output_root)
    if global_summary.get("prediction_sha256") != prediction_sha256:
        raise ValueError("Exp266 global summary is not bound to frozen predictions")
    if any(row.get("prediction_sha256") != prediction_sha256 for row in result_rows):
        raise ValueError("Exp266 result rows are not bound to frozen predictions")
    audit = {
        "dependency_audit": dependency_report(),
        "population_contract": {
            "population_count": len(population),
            "unique_population_count": len(set(population)),
            "expected_population": EXPECTED_DATASETS,
            "prediction_path": str(prediction_path),
            "results_path": str(results_path),
            "summary_path": str(summary_path),
            "input_sha256": hashes,
            "prediction_sha256": prediction_sha256,
            "prediction_manifest": manifest,
            "labels_loaded_after_audit": True,
        },
    }
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "00_dependency_audit.json").write_text(
        json.dumps(audit["dependency_audit"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "01_population_contract.json").write_text(
        json.dumps(audit["population_contract"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    truth, metadata = load_test_truth(population, database_path=database_path)
    diagnostics = build_zero_support_diagnostics(
        prediction_rows=prediction_rows,
        result_rows=result_rows,
        true_indices_by_dataset=truth,
        global_summary=global_summary,
        expected_population=EXPECTED_DATASETS,
        metadata_by_dataset=metadata,
    )
    return _write_diagnostic_outputs(root, diagnostics, audit)


__all__ = [
    "EXPECTED_COMBINED_FN",
    "EXPECTED_SINGLE_SOURCE_FN",
    "EXPECTED_ZERO_SUPPORT_FN",
    "PopulationContractError",
    "build_zero_support_diagnostics",
    "diagnostic_row",
    "load_test_truth",
    "summarize_zero_support_diagnostics",
    "write_exp267_outputs",
]
