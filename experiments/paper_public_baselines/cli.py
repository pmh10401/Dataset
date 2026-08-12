"""Closed label-free CLI for prepare, acceleration assessment, score, and freeze."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from .acceleration import AccelerationRecord, assess_mlx
from .artifacts import (
    canonical_json_bytes,
    read_strict_json,
    sha256_file,
    write_create_only_bytes,
)
from .config import (
    ALPHAS,
    DEEP_METHOD_IDS,
    EXPECTED_DATASETS,
    EXPECTED_DB_SHA256,
    METHOD_IDS,
    SEED,
)
from .data import (
    load_dataset_values,
    load_frozen_population,
    verify_frozen_population,
)
from .preprocessing import prepare_dataset
from .runner import ScoreRunConfig, freeze_score_root, run_methods


PREPARED_FIELDS = {
    "schema",
    "database_path",
    "database_sha256",
    "population_authority",
    "population_sha256",
    "full_dataset_count",
    "datasets",
    "limited_for_smoke",
    "methods",
    "alphas",
    "seed",
    "contains_test_labels",
    "contains_evaluation_metrics",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare":
        result = _prepare(
            Path(arguments.db_path), Path(arguments.run_root), arguments.limit
        )
    elif arguments.command == "assess-mlx":
        result = _assess_mlx(Path(arguments.run_root))
    elif arguments.command == "score":
        result = _score(
            Path(arguments.run_root),
            parse_method_list(arguments.methods),
            arguments.workers,
            arguments.epochs,
            Path(arguments.db_path) if arguments.db_path else None,
        )
    elif arguments.command == "freeze":
        _load_acceleration(Path(arguments.run_root))
        manifest = freeze_score_root(Path(arguments.run_root))
        result = {"pair_count": manifest.pair_count, "status": "frozen"}
    else:  # pragma: no cover - argparse closes this branch
        raise AssertionError(arguments.command)
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


def parse_method_list(value: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if (
        not methods
        or len(methods) != len(set(methods))
        or any(method not in METHOD_IDS for method in methods)
    ):
        raise ValueError("method list must be a unique, non-empty published-method subset")
    return methods


def _prepare(db_path: Path, run_root: Path, limit: int | None) -> dict[str, object]:
    resolved_db = Path(db_path).resolve(strict=True)
    database_sha = sha256_file(resolved_db)
    if database_sha != EXPECTED_DB_SHA256:
        raise ValueError("database SHA256 differs from the frozen paper authority")
    population = load_frozen_population()
    verify_frozen_population(resolved_db, population)
    if len(population) != EXPECTED_DATASETS:
        raise ValueError("frozen paper population count is not 1,117")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(population)):
        raise ValueError("limit must be between one and the frozen population size")
    selected = population[:limit] if limit is not None else population
    population_bytes = canonical_json_bytes(list(population))
    payload = {
        "schema": "paper-public-baselines.prepared.v1",
        "database_path": str(resolved_db),
        "database_sha256": database_sha,
        "population_authority": "Exp446 label-free publication dataset order",
        "population_sha256": hashlib.sha256(population_bytes).hexdigest(),
        "full_dataset_count": len(population),
        "datasets": list(selected),
        "limited_for_smoke": limit is not None,
        "methods": list(METHOD_IDS),
        "alphas": list(ALPHAS),
        "seed": SEED,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    encoded = canonical_json_bytes(payload)
    write_create_only_bytes(Path(run_root) / "prepared.json", encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    write_create_only_bytes(
        Path(run_root) / "prepared.sha256", (digest + "\n").encode("ascii")
    )
    return {
        "database_sha256": database_sha,
        "full_dataset_count": len(population),
        "selected_dataset_count": len(selected),
        "prepared_sha256": digest,
    }


def _assess_mlx(run_root: Path) -> dict[str, object]:
    prepared, prepared_sha = _load_prepared(run_root)
    database = Path(str(prepared["database_path"]))
    first_dataset = str(prepared["datasets"][0])
    values = prepare_dataset(load_dataset_values(database, first_dataset))
    smoke_input = values.train[: min(2, len(values.train))]
    records = [_record_payload(assess_mlx(method, smoke_input)) for method in METHOD_IDS]
    payload = {
        "schema": "paper-public-baselines.acceleration.v1",
        "prepared_sha256": prepared_sha,
        "database_sha256": prepared["database_sha256"],
        "records": records,
        "source_sha256": sha256_file(Path(__file__).resolve().parent / "acceleration.py"),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    encoded = canonical_json_bytes(payload)
    write_create_only_bytes(run_root / "acceleration.json", encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    write_create_only_bytes(
        run_root / "acceleration.sha256", (digest + "\n").encode("ascii")
    )
    return {"acceleration_sha256": digest, "selected_backend": "cpu", "methods": len(records)}


def _score(
    run_root: Path,
    selected_methods: tuple[str, ...],
    workers: int,
    epochs: int,
    db_path: Path | None,
) -> dict[str, object]:
    prepared, _ = _load_prepared(run_root)
    _load_acceleration(run_root)
    prepared_db = Path(str(prepared["database_path"]))
    if db_path is not None and db_path.resolve(strict=True) != prepared_db.resolve(strict=True):
        raise ValueError("score database path differs from prepared authority")
    if any(method in DEEP_METHOD_IDS for method in selected_methods) and workers != 1:
        raise ValueError("deep methods require exactly one worker")
    if sha256_file(prepared_db) != prepared["database_sha256"]:
        raise ValueError("database SHA256 changed after prepare")
    config = ScoreRunConfig(
        db_path=prepared_db,
        run_root=run_root,
        datasets=tuple(str(value) for value in prepared["datasets"]),
        methods=tuple(str(value) for value in prepared["methods"]),
        workers=workers,
        database_sha256=str(prepared["database_sha256"]),
        seed=int(prepared["seed"]),
        epochs=epochs,
    )
    summary = run_methods(config, selected_methods=selected_methods)
    return {
        "selected_methods": list(selected_methods),
        "completed": summary.completed,
        "reused": summary.reused,
        "abstained": summary.abstained,
        "failed": summary.failed,
    }


def _load_prepared(run_root: Path) -> tuple[dict[str, object], str]:
    root = Path(run_root)
    path = root / "prepared.json"
    digest_path = root / "prepared.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise ValueError("prepared authority is absent")
    expected = digest_path.read_text(encoding="ascii").strip()
    actual = sha256_file(path)
    if expected != actual:
        raise ValueError("prepared authority SHA256 mismatch")
    payload = read_strict_json(path)
    if set(payload) != PREPARED_FIELDS or payload.get("schema") != "paper-public-baselines.prepared.v1":
        raise ValueError("prepared authority schema is invalid")
    datasets = payload.get("datasets")
    methods = payload.get("methods")
    if not isinstance(datasets, list) or not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("prepared dataset population is invalid")
    if tuple(methods) != METHOD_IDS:
        raise ValueError("prepared method population is invalid")
    if payload.get("alphas") != list(ALPHAS) or payload.get("seed") != SEED:
        raise ValueError("prepared scientific configuration is invalid")
    if payload.get("contains_test_labels") is not False or payload.get("contains_evaluation_metrics") is not False:
        raise ValueError("prepared authority violates label boundary")
    return payload, actual


def _load_acceleration(run_root: Path) -> dict[str, object]:
    prepared, prepared_sha = _load_prepared(run_root)
    path = Path(run_root) / "acceleration.json"
    digest_path = Path(run_root) / "acceleration.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise ValueError("MLX assessment is absent")
    if digest_path.read_text(encoding="ascii").strip() != sha256_file(path):
        raise ValueError("MLX assessment SHA256 mismatch")
    payload = read_strict_json(path)
    if (
        payload.get("schema") != "paper-public-baselines.acceleration.v1"
        or payload.get("prepared_sha256") != prepared_sha
        or payload.get("database_sha256") != prepared.get("database_sha256")
        or payload.get("contains_test_labels") is not False
        or payload.get("contains_evaluation_metrics") is not False
    ):
        raise ValueError("MLX assessment identity is invalid")
    records = payload.get("records")
    if not isinstance(records, list) or tuple(record.get("method_id") for record in records if isinstance(record, dict)) != METHOD_IDS:
        raise ValueError("MLX assessment method coverage is invalid")
    if any(
        record.get("selected_backend") != "cpu"
        or record.get("condition_1_eligible") is not False
        or not record.get("fallback_reason")
        for record in records
    ):
        raise ValueError("MLX assessment does not bind the CPU fallback")
    return payload


def _record_payload(record: AccelerationRecord) -> dict[str, object]:
    return {
        "method_id": record.method_id,
        "exact_stage": record.exact_stage,
        "condition_1_eligible": record.condition_1_eligible,
        "cpu_finite": record.cpu_finite,
        "mlx_finite": record.mlx_finite,
        "parity_passed": record.parity_passed,
        "candidate_indices_equal_by_alpha": dict(record.candidate_indices_equal_by_alpha),
        "numerical_tolerance": record.numerical_tolerance,
        "cpu_seconds": record.cpu_seconds,
        "mlx_seconds": record.mlx_seconds,
        "speed_ratio": record.speed_ratio,
        "selected_backend": record.selected_backend,
        "fallback_reason": record.fallback_reason,
        "versions": dict(record.versions),
        "device": record.device,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_paper_public_baselines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--db-path", required=True)
    prepare.add_argument("--run-root", required=True)
    prepare.add_argument("--limit", type=int)

    assess = subparsers.add_parser("assess-mlx")
    assess.add_argument("--run-root", required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--run-root", required=True)
    score.add_argument("--methods", required=True)
    score.add_argument("--workers", type=int, required=True)
    score.add_argument("--epochs", type=int, default=50)
    score.add_argument("--db-path")

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--run-root", required=True)
    return parser


__all__ = ["main", "parse_method_list"]
