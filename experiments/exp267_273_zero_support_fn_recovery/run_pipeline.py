"""Sequential Exp267-273 stage runner with dashboard-visible heartbeats."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from experiments.exp239_248_diverse_score_sources.foundation import stable_json_hash

from .config import OUTPUT_ROOT
from .diagnostics import write_exp267_outputs
from .evaluation import evaluate_frozen_predictions
from .forecast_tcn_source import benchmark_device
from .population import load_frozen_population
from .promotion_audit import audit_promotion
from .qualification import build_verified_qualification_evidence, qualify_sources
from .recovery_routing import build_and_freeze_verified_recovery_predictions
from .reporting import build_status, publish_status
from .source_pipeline import cache_source, freeze_source_predictions
from .train_risk import write_train_risk_audit


STAGES = (
    "audit", "267", "268", "269", "270a", "270b", "270c", "271", "272",
    "273", "evaluate", "all", "device-benchmark",
)
STAGE_EXPERIMENT = {
    "audit": "experiment_267_zero_support_fn_diagnostic",
    "267": "experiment_267_zero_support_fn_diagnostic",
    "268": "experiment_268_multiscale_scattering_deviation",
    "269": "experiment_269_ordered_path_signature_deviation",
    "270a": "experiment_270a_rff_normal_forecast",
    "270b": "experiment_270b_causal_tcn_normal_forecast",
    "270c": "experiment_270c_forecast_risk_audit",
    "271": "experiment_271_new_source_qualification",
    "272": "experiment_272_recovery_review_selector",
    "273": "experiment_273_autonomous_promotion_audit",
    "evaluate": "experiment_273_autonomous_promotion_audit",
    "device-benchmark": "experiment_270b_causal_tcn_normal_forecast",
}


class StatusPublisher:
    def __init__(self, root: Path, experiment_id: str, phase: str, expected: int) -> None:
        self.root, self.experiment_id, self.phase = Path(root), experiment_id, phase
        self.expected, self.done, self.current = expected, 0, None
        self.started, self.timestamps, self.errors = time.time(), [], []
        self.log = self.root / "logs" / f"{experiment_id}.log"

    def emit(self, status: str = "running", prediction_sha256: str | None = None) -> None:
        row = build_status(
            self.experiment_id, status, self.phase, self.done, self.expected, self.current,
            errors=self.errors, timestamps=self.timestamps, stdout_log=self.log,
            started_unix=self.started, prediction_sha256=prediction_sha256,
        )
        try:
            publish_status(row, output_root=self.root)
        except OSError as error:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.log.open("a", encoding="utf-8") as handle:
                handle.write(f"dashboard status write failed: {error}\n")

    def progress(self, done: int, _expected: int, dataset: str, errors: int) -> None:
        self.done, self.current = int(done), str(dataset)
        self.timestamps.append((time.time(), self.done))
        self.timestamps = self.timestamps[-20:]
        if errors:
            self.errors = [f"source cache errors: {errors}"]
        self.emit()


def _paths(root: Path) -> dict[str, Path]:
    return {
        "cache": root / "source_cache",
        "source_prediction": root / "source_predictions/frozen_source_predictions.jsonl",
        "source_manifest": root / "source_predictions/source_prediction_manifest.json",
        "membership": root / "exp272_source_membership.json",
        "qualification_summary": root / "exp271_qualification_summary.csv",
        "reproducibility": root / "exp271_reproducibility_manifest.json",
        "recovery_prediction": root / "exp272_frozen_predictions.jsonl",
        "recovery_manifest": root / "exp272_prediction_manifest.json",
        "recovery_summary": root / "exp272_recovery_summary.csv",
        "results": root / "exp273_dataset_results.csv",
        "summary": root / "exp273_evaluation_summary.json",
    }


def _write_one_row_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)


def _runtime(root: Path, requested: str) -> dict[str, dict[str, object]]:
    benchmark_path = root / "device/tcn_device_benchmark.json"
    if not benchmark_path.exists():
        return {}
    record = json.loads(benchmark_path.read_text(encoding="utf-8"))
    selected = str(record.get("selected_device", "cpu")) if requested != "cpu" else "cpu"
    return {"tcn_forecast": {"device": selected, "benchmark_record": record}}


def _freeze_reproducibility(root: Path, datasets: tuple[str, ...], runtime) -> Path:
    paths = _paths(root)
    first = build_verified_qualification_evidence(
        paths["cache"], expected_datasets=datasets, source_runtime=runtime,
    )
    second = build_verified_qualification_evidence(
        paths["cache"], expected_datasets=datasets, source_runtime=runtime,
    )
    first_hash = hashlib.sha256(paths["source_prediction"].read_bytes()).hexdigest()
    temp_prediction, _ = freeze_source_predictions(
        paths["cache"], datasets=datasets, expected_dataset_count=len(datasets),
        output_dir=root / "reproducibility_read_2", source_runtime=runtime,
    )
    second_hash = hashlib.sha256(temp_prediction.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1, "reproducible": True,
        "verification_scope": "two_verified_reads_and_prediction_freezes_of_the_same_frozen_cache",
        "prediction_sha256_run_1": first_hash, "prediction_sha256_run_2": second_hash,
        "evidence_sha256_run_1": first.evidence_sha256,
        "evidence_sha256_run_2": second.evidence_sha256,
        "cache_sha256_run_1": dict(sorted(first.cache_sha256.items())),
        "cache_sha256_run_2": dict(sorted(second.cache_sha256.items())),
    }
    payload["reproducibility_manifest_sha256"] = stable_json_hash(payload)
    paths["reproducibility"].write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8",
    )
    return paths["reproducibility"]


def run_stage(
    stage: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    dataset_limit: int | None = None,
    workers: int = 7,
    device: str = "auto",
    reset: bool = False,
) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    population = load_frozen_population()
    datasets = population if dataset_limit is None else population[: int(dataset_limit)]
    paths = _paths(root)
    publisher = StatusPublisher(root, STAGE_EXPERIMENT.get(stage, "experiment_267_273_pipeline"), stage, len(datasets))
    publisher.emit("running")
    try:
        runtime = _runtime(root, device)
        if stage == "audit":
            (root / "00_population_audit.json").write_text(json.dumps({
                "datasets": len(datasets), "dataset_list_sha256": stable_json_hash(datasets),
                "contains_test_labels": False,
            }, sort_keys=True) + "\n", encoding="utf-8")
        elif stage == "267":
            if dataset_limit is None:
                write_exp267_outputs(output_root=root / "exp267")
            else:
                (root / "exp267_smoke.json").write_text(json.dumps({
                    "datasets": len(datasets), "smoke": True,
                    "note": "full zero-support diagnostic is restricted to the frozen 1,117 population",
                }, sort_keys=True) + "\n", encoding="utf-8")
        elif stage == "device-benchmark":
            benchmark_device(root / "device", requested=device)
        elif stage in {"268", "269", "270a", "270b"}:
            source = {"268": "scattering", "269": "signature", "270a": "rff_forecast", "270b": "tcn_forecast"}[stage]
            if source == "tcn_forecast" and not runtime:
                benchmark_device(root / "device", requested=device)
                runtime = _runtime(root, device)
            source_device = str(runtime.get(source, {}).get("device", "cpu"))
            benchmark = runtime.get(source, {}).get("benchmark_record")
            cache_source(
                paths["cache"], source, datasets=datasets,
                workers=1 if source == "tcn_forecast" else workers,
                device=source_device, reset=reset, benchmark_record=benchmark,
                expected_dataset_count=len(datasets), progress=publisher.progress,
            )
        elif stage == "270c":
            freeze_source_predictions(
                paths["cache"], datasets=datasets, expected_dataset_count=len(datasets),
                output_dir=paths["source_prediction"].parent, source_runtime=runtime,
            )
            evidence = build_verified_qualification_evidence(
                paths["cache"], expected_datasets=datasets, source_runtime=runtime,
            )
            write_train_risk_audit(root, evidence.risk_rows)
        elif stage == "271":
            evidence = build_verified_qualification_evidence(
                paths["cache"], expected_datasets=datasets, source_runtime=runtime,
            )
            proof = _freeze_reproducibility(root, datasets, runtime)
            membership = qualify_sources(
                paths["source_prediction"], paths["source_manifest"], output_dir=root,
                evidence=evidence, expected_datasets=datasets,
                reproducibility_manifest_path=proof, cache_root=paths["cache"],
                source_runtime=runtime,
            )
            _write_one_row_csv(paths["qualification_summary"], {
                "datasets": len(datasets),
                "qualified_groups": len(membership.get("qualified_groups", [])),
                "contains_test_labels": False,
                "retrospective": True,
            })
        elif stage == "272":
            build_and_freeze_verified_recovery_predictions(
                paths["source_prediction"], paths["source_manifest"], paths["membership"],
                cache_root=paths["cache"], output_dir=root, source_runtime=runtime,
                expected_datasets=datasets,
            )
            rows = [
                json.loads(line)
                for line in paths["recovery_prediction"].read_text(encoding="utf-8").splitlines()
                if line
            ]
            _write_one_row_csv(paths["recovery_summary"], {
                "datasets": len(datasets),
                "hard_count": sum(len(row["hard_indices"]) for row in rows),
                "recovery_shadow_count": sum(len(row["recovery_shadow_indices"]) for row in rows),
                "recovery_priority_count": sum(len(row["recovery_priority_indices"]) for row in rows),
                "contains_test_labels": False,
                "retrospective": True,
            })
        elif stage == "evaluate":
            evaluate_frozen_predictions(
                paths["recovery_prediction"], paths["recovery_manifest"],
                expected_datasets=datasets, output_dir=root,
                source_prediction_path=paths["source_prediction"],
                source_prediction_manifest_path=paths["source_manifest"],
                membership_path=paths["membership"], cache_root=paths["cache"],
                source_runtime=runtime,
            )
        elif stage == "273":
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            autonomous = summary["autonomous"]
            membership = json.loads(paths["membership"].read_text(encoding="utf-8"))
            qualified = membership.get("qualified_groups", [])
            train_u95 = max((float(group["risk_wilson_upper_95"]) for group in qualified), default=1.0)
            with paths["results"].open(encoding="utf-8", newline="") as handle:
                result_rows = list(csv.DictReader(handle))
            recovery_count = sum(
                int(row["recovery_shadow_count"]) + int(row["recovery_priority_count"])
                for row in result_rows
            )
            recovery_tp = sum(
                int(row["recovery_shadow_tp"]) + int(row["recovery_priority_tp"])
                for row in result_rows
            )
            recovery_fp = recovery_count - recovery_tp
            precision = recovery_tp / recovery_count if recovery_count else 0.0
            overall = (
                (int(autonomous["hard_tp"]) + recovery_tp)
                / (int(autonomous["hard_alerts"]) + recovery_count)
                if int(autonomous["hard_alerts"]) + recovery_count else 0.0
            )
            audit = audit_promotion(
                independent_groups=len(qualified), train_u95=train_u95,
                precision=precision, overall_precision=overall, invariant=True,
            )
            (root / "exp273_promotion_audit.json").write_text(
                json.dumps({
                    **audit.__dict__,
                    "audited_candidate_scope": "Exp272 recovery shadow plus recovery priority only",
                    "audited_candidate_count": recovery_count,
                    "audited_candidate_tp": recovery_tp,
                    "audited_candidate_fp": recovery_fp,
                    "hypothetical_only": True,
                    "hard_activated": False,
                }, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif stage == "all":
            for item in ("audit", "267", "268", "269", "270a", "device-benchmark", "270b", "270c", "271", "272", "evaluate", "273"):
                run_stage(item, output_root=root, dataset_limit=dataset_limit, workers=workers, device=device, reset=reset)
        publisher.done, publisher.current = len(datasets), None
        prediction_hash = hashlib.sha256(paths["recovery_prediction"].read_bytes()).hexdigest() if paths["recovery_prediction"].exists() else None
        publisher.emit("completed", prediction_sha256=prediction_hash)
    except Exception as error:
        publisher.errors = [f"{type(error).__name__}: {error}"]
        publisher.emit("failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run_stage(
        args.stage, output_root=args.output_root, dataset_limit=args.dataset_limit,
        workers=args.workers, device=args.device, reset=args.reset_cache,
    )


if __name__ == "__main__":
    main()
