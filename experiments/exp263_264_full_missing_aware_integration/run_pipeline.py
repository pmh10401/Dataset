from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import MappingProxyType

from . import evaluation, prediction, reporting
from .config import DB_PATH, OUTPUT_ROOT, SOURCE_CONTRACT
from .inputs import (
    EXPECTED_MISSING_OVERRIDE_DATASETS,
    EXPECTED_POPULATION,
    EXPECTED_RAW_DATABASE_DATASETS,
    EXP260_AUDIT_PATH,
    FROZEN_EXCLUDED_DATASETS,
    FrozenSourceMaps,
    SOURCE_NAMES,
    load_source_maps,
    sha256_file,
)


EXP263_ID = "experiment_263_full_missing_aware_integration"
EXP264_ID = "experiment_264_full_missing_aware_posthoc_evaluation"
PHASES = ("audit", "predict", "evaluate", "report")


def output_paths(output_root: Path) -> dict[str, str]:
    return {
        "status": str(output_root / "status.json"),
        "stdout_log": str(output_root / "full.log"),
        "audit": str(output_root / "00_source_contract_audit.json"),
        "coverage": str(output_root / "01_source_coverage.csv"),
        "predictions": str(output_root / "frozen_predictions.jsonl"),
        "prediction_manifest": str(output_root / "prediction_manifest.json"),
        "source_diagnostics": str(output_root / "source_agreement_diagnostics.csv"),
        "results": str(output_root / "results.csv"),
        "lane_transitions": str(output_root / "lane_transitions.csv"),
        "summary": str(output_root / "summary.json"),
        "evaluation_manifest": str(output_root / "evaluation_manifest.json"),
        "report": str(output_root / "RESULT_REPORT.md"),
        "manifest": str(output_root / "MANIFEST.json"),
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write(
        path,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


class StatusReporter:
    def __init__(self, *, phase: str, output_root: Path, expected: int) -> None:
        self.output_root = Path(output_root)
        self.path = self.output_root / "status.json"
        now = time.time()
        paths = output_paths(self.output_root)
        experiment_id = EXP263_ID if phase in {"audit", "predict"} else EXP264_ID
        self.payload: dict[str, object] = {
            "id": experiment_id,
            "label": "Exp263/264 - full missing-aware integration",
            "status": "running",
            "phase": phase,
            "pid": os.getpid(),
            "done": 0,
            "datasets_done": 0,
            "expected": expected,
            "expected_datasets": expected,
            "current_dataset": None,
            "errors": [],
            "error_count": 0,
            "output_root": str(self.output_root),
            "output_paths": paths,
            "detail_csv": paths["results"],
            "stdout_log": paths["stdout_log"],
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "started_unix": now,
            "updated_unix": now,
        }
        self.update()

    def update(self, **changes: object) -> None:
        self.payload.update(changes)
        if "done" in changes:
            self.payload["datasets_done"] = changes["done"]
        if "expected" in changes:
            self.payload["expected_datasets"] = changes["expected"]
        errors = self.payload.get("errors")
        self.payload["error_count"] = len(errors) if isinstance(errors, list) else 0
        self.payload["updated_unix"] = time.time()
        _atomic_write_json(self.path, self.payload)

    def fail(self, exc: BaseException) -> None:
        errors = list(self.payload.get("errors") or [])
        errors.append(f"{type(exc).__name__}: {exc}")
        self.update(status="failed", errors=errors)


def _subset_sources(sources: FrozenSourceMaps, dataset_limit: int | None) -> FrozenSourceMaps:
    if dataset_limit is None:
        return sources
    population = sources.population[:dataset_limit]
    selected = set(population)
    return FrozenSourceMaps(
        population=population,
        n_train=MappingProxyType({name: sources.n_train[name] for name in population}),
        n_test=MappingProxyType({name: sources.n_test[name] for name in population}),
        baseline_rows=MappingProxyType(
            {name: sources.baseline_rows[name] for name in population}
        ),
        source_indices=MappingProxyType(
            {
                source: MappingProxyType(
                    {
                        name: indices
                        for name, indices in sources.source_indices[source].items()
                        if name in selected
                    }
                )
                for source in SOURCE_NAMES
            }
        ),
        missing_override_rows=MappingProxyType(
            {
                name: row
                for name, row in sources.missing_override_rows.items()
                if name in selected
            }
        ),
        override_source_indices=MappingProxyType(
            {
                name: values
                for name, values in sources.override_source_indices.items()
                if name in selected
            }
        ),
        quality_flags=MappingProxyType(
            {
                name: values
                for name, values in sources.quality_flags.items()
                if name in selected
            }
        ),
    )


def _load_selected_sources(dataset_limit: int | None) -> FrozenSourceMaps:
    return _subset_sources(load_source_maps(), dataset_limit)


def _write_coverage(
    output_root: Path, sources: FrozenSourceMaps, reporter: StatusReporter
) -> None:
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
    rows = []
    overrides = set(sources.missing_override_rows)
    for done, dataset in enumerate(sources.population, start=1):
        rows.append(
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
        reporter.update(done=done, current_dataset=dataset)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output_root, suffix=".tmp", delete=False, newline="", encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, output_root / "01_source_coverage.csv")


def _run_audit(args: argparse.Namespace, reporter: StatusReporter) -> dict[str, object]:
    sources = _load_selected_sources(args.dataset_limit)
    coverage = {name: len(values) for name, values in sources.source_indices.items()}
    audit = {
        "raw_database_dataset_count": EXPECTED_RAW_DATABASE_DATASETS,
        "frozen_excluded_datasets": sorted(FROZEN_EXCLUDED_DATASETS),
        "frozen_population_count": len(sources.population),
        "production_expected_population": EXPECTED_POPULATION,
        "production_expected_missing_override_datasets": EXPECTED_MISSING_OVERRIDE_DATASETS,
        "expected_population": len(sources.population),
        "dataset_limit": args.dataset_limit,
        "smoke_contract_override": args.dataset_limit is not None,
        "selected_datasets": list(sources.population),
        "dataset_count": len(sources.population),
        "baseline_coverage": len(sources.baseline_rows),
        "statistical_coverage": coverage["statistical"],
        "structural_coverage": coverage["structural"],
        "change_point_coverage": coverage["change_point"],
        "reconstruction_coverage": coverage["reconstruction"],
        "missing_override_coverage": len(sources.missing_override_rows),
        "missing_datasets_without_override": [],
        "input_sha256": {
            name: sha256_file(spec.artifact_path) for name, spec in SOURCE_CONTRACT.items()
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
        "errors": [],
        "error_count": 0,
    }
    _atomic_write_json(args.output_root / "00_source_contract_audit.json", audit)
    _write_coverage(args.output_root, sources, reporter)
    return {
        "done": len(sources.population),
        "current_dataset": sources.population[-1] if sources.population else None,
    }


def _validate_prediction_rows(
    rows: list[dict[str, object]], sources: FrozenSourceMaps
) -> None:
    names = [row.get("dataset") for row in rows]
    if tuple(names) != sources.population or len(set(names)) != len(sources.population):
        raise ValueError("prediction dataset coverage mismatch")
    for row in rows:
        dataset = str(row["dataset"])
        n_test = row.get("n_test")
        if type(n_test) is not int:
            raise ValueError(f"prediction n_test is invalid: dataset={dataset}")
        lanes = [
            set(row[field])
            for field in ("final_hard_indices", "final_standard_indices", "final_priority_indices")
        ]
        if any(
            lanes[left] & lanes[right]
            for left in range(len(lanes))
            for right in range(left + 1, len(lanes))
        ):
            raise ValueError(f"prediction lane overlap: dataset={dataset}")
        source_indices = row.get("source_indices")
        if type(source_indices) is not dict or set(source_indices) != set(SOURCE_NAMES):
            raise ValueError(f"prediction source map is invalid: dataset={dataset}")
        bounded = [
            *lanes,
            set(row["quality_flag_indices"]),
            set(row["quality_blocked_indices"]),
            *(set(source_indices[name]) for name in SOURCE_NAMES),
        ]
        if any(
            type(index) is not int or index < 0 or index >= n_test
            for indices in bounded
            for index in indices
        ):
            raise ValueError(f"prediction index outside TEST range: dataset={dataset}")
        if not set(row["baseline_hard_indices"]) <= set(row["final_hard_indices"]):
            raise ValueError(f"baseline Hard was removed: dataset={dataset}")


def _run_predict(args: argparse.Namespace, reporter: StatusReporter) -> dict[str, object]:
    sources = _load_selected_sources(args.dataset_limit)
    rows = []
    for done, dataset in enumerate(sources.population, start=1):
        rows.append(prediction._route_row(dataset, sources))
        reporter.update(done=done, current_dataset=dataset)
    if args.dataset_limit is None:
        prediction._validate_rows(rows, sources)
    else:
        _validate_prediction_rows(rows, sources)
    data = b"".join(
        (
            json.dumps(prediction._json_ready(row), sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    prediction_path = args.output_root / "frozen_predictions.jsonl"
    prediction._atomic_write(prediction_path, data)
    digest = hashlib.sha256(data).hexdigest()
    manifest = {
        "experiment": prediction.EXPERIMENT,
        "experiment_id": prediction.EXPERIMENT_ID,
        "rows": len(rows),
        "datasets": len(rows),
        "missing_override_datasets": len(sources.missing_override_rows),
        "prediction_sha256": digest,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
        "production_expected_population": EXPECTED_POPULATION,
        "smoke_contract_override": args.dataset_limit is not None,
    }
    prediction._atomic_write(
        args.output_root / "prediction_manifest.json",
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    prediction._write_diagnostics(args.output_root / "source_agreement_diagnostics.csv", rows)
    return {
        "done": len(rows),
        "current_dataset": str(rows[-1]["dataset"]) if rows else None,
    }


def _prediction_contract(args: argparse.Namespace) -> tuple[str, tuple[str, ...] | None]:
    manifest = json.loads((args.output_root / "prediction_manifest.json").read_text())
    expected_sha256 = str(manifest["prediction_sha256"])
    if args.dataset_limit is None:
        return expected_sha256, None
    rows = evaluation._parse_prediction_payload(
        (args.output_root / "frozen_predictions.jsonl").read_bytes()
    )
    population = tuple(str(row["dataset"]) for row in rows)
    if len(population) != args.dataset_limit:
        raise ValueError(
            f"smoke population mismatch: expected={args.dataset_limit} actual={len(population)}"
        )
    return expected_sha256, population


def _run_evaluate(args: argparse.Namespace, reporter: StatusReporter) -> dict[str, object]:
    expected_sha256, expected_population = _prediction_contract(args)
    evaluated, summary = evaluation.evaluate_all(
        args.output_root / "frozen_predictions.jsonl",
        expected_sha256=expected_sha256,
        expected_population=expected_population,
        expected_total_anomalies=args.expected_total_anomalies,
    )
    manifest = {
        "experiment": 264,
        "experiment_id": EXP264_ID,
        "prediction_sha256": expected_sha256,
        "datasets": len(evaluated),
        "expected_total_anomalies": (
            evaluation.EXPECTED_TOTAL_ANOMALIES
            if args.expected_total_anomalies is None
            else args.expected_total_anomalies
        ),
        "actual_total_anomalies": summary["autonomous_hard"]["total_anomalies"],
        "smoke_contract_override": args.dataset_limit is not None,
        "results_sha256": sha256_file(args.output_root / "results.csv"),
        "summary_sha256": sha256_file(args.output_root / "summary.json"),
    }
    _atomic_write_json(args.output_root / "evaluation_manifest.json", manifest)
    current = str(evaluated[-1]["dataset"]) if evaluated else None
    reporter.update(done=len(evaluated), current_dataset=current)
    return {"done": len(evaluated), "current_dataset": current}


def _run_report(args: argparse.Namespace, reporter: StatusReporter) -> dict[str, object]:
    report = reporting.generate_report(args.output_root, database_path=DB_PATH)
    done = int(report["reaggregation"]["datasets"])
    rows = reporting.read_results(args.output_root / "results.csv")
    current = str(rows[-1]["dataset"]) if rows else None
    reporter.update(done=done, current_dataset=current)
    return {"done": done, "current_dataset": current}


PHASE_RUNNERS = {
    "audit": _run_audit,
    "predict": _run_predict,
    "evaluate": _run_evaluate,
    "report": _run_report,
}


def run_phase(args: argparse.Namespace, reporter: StatusReporter) -> dict[str, object]:
    return PHASE_RUNNERS[args.phase](args, reporter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--expected-total-anomalies", type=int)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset_limit is not None and not 1 <= args.dataset_limit <= EXPECTED_POPULATION:
        raise ValueError(f"dataset-limit must be between 1 and {EXPECTED_POPULATION}")
    if args.expected_total_anomalies is not None and args.expected_total_anomalies < 0:
        raise ValueError("expected-total-anomalies must be nonnegative")
    if (
        args.phase == "evaluate"
        and args.dataset_limit is not None
        and args.expected_total_anomalies is None
    ):
        raise ValueError(
            "smoke evaluation requires --expected-total-anomalies as an explicit contract override"
        )


def _append_log(output_root: Path, message: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "full.log").open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    expected = args.dataset_limit or EXPECTED_POPULATION
    reporter = StatusReporter(phase=args.phase, output_root=args.output_root, expected=expected)
    try:
        validate_args(args)
        _append_log(args.output_root, f"START phase={args.phase} expected={expected}")
        result = run_phase(args, reporter)
        reporter.update(
            status="completed",
            done=int(result.get("done", expected)),
            current_dataset=result.get("current_dataset"),
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        _append_log(args.output_root, f"DONE phase={args.phase} done={result.get('done', expected)}")
        return 0
    except BaseException as exc:
        reporter.fail(exc)
        _append_log(args.output_root, traceback.format_exc())
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
