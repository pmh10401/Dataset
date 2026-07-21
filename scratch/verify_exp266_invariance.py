from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp266_fn_recovery_shadow import inputs, prediction
from experiments.exp266_fn_recovery_shadow.config import (
    DB_PATH,
    FULL_RUN_WORKERS,
    OUTPUT_ROOT,
)


CHECKS = (
    "label-permutation",
    "anomaly-position-mutation",
    "display-name-remapping",
    "input-row-permutation",
    "repeat-prediction-hash",
)
PREDICTION_FILENAME = "03_frozen_predictions.jsonl"
DEFAULT_TEMP_ROOT = Path("/Users/minho/Documents/Dataset")
DATABASE_SCHEMA_SQL = """
CREATE TABLE datasets AS SELECT * FROM src.datasets;
CREATE TABLE instances (
  id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL, split TEXT NOT NULL,
  instance_index INTEGER NOT NULL, label TEXT NOT NULL, values_blob BLOB NOT NULL,
  labels_blob BLOB
);
"""
DATABASE_INDEX_SQL = """
CREATE UNIQUE INDEX idx_instances_dataset_split_index
ON instances(dataset_id, split, instance_index);
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_temp_root(temp_root: Path, baseline_root: Path) -> Path:
    resolved = temp_root.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if _is_within(resolved, REPO_ROOT):
        raise ValueError("temporary invariance outputs must be outside the repository")
    if _is_within(resolved, baseline_root):
        raise ValueError("temporary invariance outputs must be separate from full results")
    return resolved


def _baseline_hash(baseline_root: Path) -> str:
    manifest = prediction.verify_prediction_checkpoint(baseline_root)
    path = baseline_root / PREDICTION_FILENAME
    actual = _sha256(path)
    if manifest.get("prediction_sha256") != actual:
        raise ValueError("baseline prediction hash does not match its manifest")
    return actual


def _sqlite_uri(path: Path, mode: str) -> str:
    return f"file:{quote(str(path.resolve()), safe='/')}?mode={mode}"


def _build_mutation_database(
    source: Path,
    destination: Path,
    *,
    mutation: str,
) -> dict[str, int]:
    connection = sqlite3.connect(_sqlite_uri(destination, "rwc"), uri=True)
    try:
        connection.execute("ATTACH DATABASE ? AS src", (_sqlite_uri(source, "ro"),))
        connection.executescript(DATABASE_SCHEMA_SQL)
        if mutation == "label-permutation":
            connection.executescript(
                """
                INSERT INTO instances
                SELECT i.id, i.dataset_id, i.split, i.instance_index,
                       CASE WHEN i.split='TEST' THEN shifted.label ELSE i.label END,
                       zeroblob(0), NULL
                FROM src.instances AS i
                JOIN src.datasets AS d ON d.id=i.dataset_id
                LEFT JOIN src.instances AS shifted
                  ON i.split='TEST'
                 AND shifted.dataset_id=i.dataset_id
                 AND shifted.split='TEST'
                 AND shifted.instance_index=(
                       (i.instance_index + 1) % d.test_total_count
                 );
                """
            )
            changed = connection.execute(
                """SELECT COUNT(*)
                   FROM instances AS mutated
                   JOIN src.instances AS original ON original.id=mutated.id
                   WHERE mutated.split='TEST' AND mutated.label<>original.label"""
            ).fetchone()[0]
        elif mutation == "anomaly-position-mutation":
            connection.executescript(
                """
                INSERT INTO instances
                SELECT i.id, i.dataset_id, i.split,
                       CASE WHEN i.split='TEST'
                            THEN ((i.instance_index + 1) % d.test_total_count)
                            ELSE i.instance_index END,
                       i.label, zeroblob(0), NULL
                FROM src.instances AS i
                JOIN src.datasets AS d ON d.id=i.dataset_id;
                """
            )
            changed = connection.execute(
                """SELECT COUNT(*)
                   FROM instances AS mutated
                   JOIN src.instances AS original ON original.id=mutated.id
                   WHERE mutated.split='TEST'
                     AND mutated.instance_index<>original.instance_index"""
            ).fetchone()[0]
        else:
            raise ValueError(f"unknown database mutation: {mutation}")
        connection.executescript(DATABASE_INDEX_SQL)
        total = connection.execute(
            "SELECT COUNT(*) FROM instances WHERE split='TEST'"
        ).fetchone()[0]
        connection.commit()
        connection.execute("DETACH DATABASE src")
        connection.execute("VACUUM")
    finally:
        connection.close()
    return {"changed_test_rows": int(changed), "total_test_rows": int(total)}


def _predict_with_database(
    database: Path,
    output_root: Path,
    *,
    workers: int,
) -> Path:
    original = inputs.DB_PATH
    inputs.DB_PATH = database
    try:
        return prediction.predict_all(output_root, workers=workers)
    finally:
        inputs.DB_PATH = original


def _database_mutation_check(
    check: str,
    *,
    source_database: Path,
    temporary: Path,
    baseline_sha256: str,
    workers: int,
) -> dict[str, object]:
    database = temporary / f"{check}.db"
    details = _build_mutation_database(
        source_database,
        database,
        mutation=check,
    )
    output = _predict_with_database(
        database,
        temporary / "prediction",
        workers=workers,
    )
    output_sha256 = _sha256(output)
    return {
        "baseline_sha256": baseline_sha256,
        "check": check,
        "details": details,
        "input_sha256": _sha256(database),
        "output_sha256": output_sha256,
        "passed": output_sha256 == baseline_sha256,
    }


def _display_name_check(
    temporary: Path,
    *,
    baseline_sha256: str,
) -> dict[str, object]:
    recovery = inputs.load_frozen_inputs(output_root=temporary / "audit")
    name_map = {
        name: f"display-name-{index:04d}"
        for index, name in enumerate(recovery.population)
    }
    reverse_map = {display: name for name, display in name_map.items()}
    remapped = replace(
        recovery,
        population=tuple(name_map[name] for name in recovery.population),
        datasets=MappingProxyType(
            {
                name_map[name]: replace(item, dataset=name_map[name])
                for name, item in recovery.datasets.items()
            }
        ),
    )
    rows = []
    for item in prediction.ordered_dataset_inputs(remapped, None):
        row = dict(prediction._route_prediction_row(item, remapped))
        row["dataset"] = reverse_map[str(row["dataset"])]
        rows.append(row)
    output = prediction.freeze_rows(temporary / "prediction", rows)
    output_sha256 = _sha256(output)
    input_sha256 = _canonical_hash(
        {
            "baseline_prediction_sha256": baseline_sha256,
            "display_name_map": name_map,
        }
    )
    return {
        "baseline_sha256": baseline_sha256,
        "check": "display-name-remapping",
        "details": {"remapped_datasets": len(name_map)},
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "passed": output_sha256 == baseline_sha256,
    }


def _input_row_check(
    temporary: Path,
    *,
    baseline_sha256: str,
) -> dict[str, object]:
    recovery = inputs.load_frozen_inputs(output_root=temporary / "audit")
    reversed_items = list(reversed(tuple(recovery.datasets.items())))
    permuted = replace(
        recovery,
        population=tuple(reversed(recovery.population)),
        datasets=MappingProxyType(dict(reversed_items)),
    )
    rows = [
        prediction._route_prediction_row(item, permuted)
        for item in prediction.ordered_dataset_inputs(permuted, None)
    ]
    output = prediction.freeze_rows(temporary / "prediction", rows)
    output_sha256 = _sha256(output)
    input_sha256 = _canonical_hash(
        {
            "baseline_prediction_sha256": baseline_sha256,
            "population_order": list(permuted.population),
            "dataset_iteration_order": [name for name, _item in reversed_items],
        }
    )
    return {
        "baseline_sha256": baseline_sha256,
        "check": "input-row-permutation",
        "details": {"permuted_datasets": len(permuted.population)},
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "passed": output_sha256 == baseline_sha256,
    }


def _repeat_prediction_check(
    temporary: Path,
    *,
    baseline_sha256: str,
    workers: int,
) -> dict[str, object]:
    output = prediction.predict_all(temporary / "prediction", workers=workers)
    output_sha256 = _sha256(output)
    return {
        "baseline_sha256": baseline_sha256,
        "check": "repeat-prediction-hash",
        "details": {"workers": workers},
        "input_sha256": baseline_sha256,
        "output_sha256": output_sha256,
        "passed": output_sha256 == baseline_sha256,
    }


def run_check(
    check: str,
    *,
    baseline_root: Path,
    source_database: Path,
    temp_root: Path,
    workers: int,
) -> dict[str, object]:
    baseline_sha256 = _baseline_hash(baseline_root)
    with tempfile.TemporaryDirectory(prefix=f"exp266-{check}-", dir=temp_root) as name:
        temporary = Path(name)
        if check in {"label-permutation", "anomaly-position-mutation"}:
            return _database_mutation_check(
                check,
                source_database=source_database,
                temporary=temporary,
                baseline_sha256=baseline_sha256,
                workers=workers,
            )
        if check == "display-name-remapping":
            return _display_name_check(temporary, baseline_sha256=baseline_sha256)
        if check == "input-row-permutation":
            return _input_row_check(temporary, baseline_sha256=baseline_sha256)
        if check == "repeat-prediction-hash":
            return _repeat_prediction_check(
                temporary,
                baseline_sha256=baseline_sha256,
                workers=workers,
            )
    raise ValueError(f"unknown check: {check}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Exp266 prediction invariance evidence"
    )
    parser.add_argument("--check", choices=(*CHECKS, "all"), required=True)
    parser.add_argument("--baseline-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--source-database", type=Path, default=DB_PATH)
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--workers", type=int, default=FULL_RUN_WORKERS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = CHECKS if args.check == "all" else (args.check,)
    try:
        baseline_root = args.baseline_root.expanduser().resolve()
        source_database = args.source_database.expanduser().resolve()
        temp_root = _validate_temp_root(args.temp_root, baseline_root)
    except BaseException as exc:
        for check in selected:
            print(
                json.dumps(
                    {
                        "check": check,
                        "error": f"{type(exc).__name__}: {exc}",
                        "passed": False,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return 1
    exit_code = 0
    for check in selected:
        try:
            evidence = run_check(
                check,
                baseline_root=baseline_root,
                source_database=source_database,
                temp_root=temp_root,
                workers=args.workers,
            )
        except BaseException as exc:
            evidence = {
                "check": check,
                "error": f"{type(exc).__name__}: {exc}",
                "passed": False,
            }
        if not evidence["passed"]:
            exit_code = 1
        print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
