from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.exp206_210.core import empirical_upper_pvalues, leave_one_out_pvalues
from experiments.exp220_223.evaluation import wilson_bound
from experiments.exp226_235_hard_alert_quality.baseline_exp137 import (
    BaselineLanes,
    load_exp137_baseline,
    verify_exp137_exact_replay,
)
from experiments.exp226_235_hard_alert_quality.config import DB_PATH, OUTPUT_ROOT, REPO_OUTPUT_ROOT, SEED
from experiments.exp226_235_hard_alert_quality.contracts import PredictionRow
from experiments.exp226_235_hard_alert_quality.contracts import ScoreBundle
from experiments.exp226_235_hard_alert_quality.database import (
    audit_database,
    load_test_labels_for_evaluation,
    sha256_file,
)
from experiments.exp226_235_hard_alert_quality.exp227_independent_confirmation import confirmation_tiers
from experiments.exp226_235_hard_alert_quality.exp226_train_risk import audit_frozen_train_risk
from experiments.exp226_235_hard_alert_quality.exp228_margin_confidence import conformal_margin
from experiments.exp226_235_hard_alert_quality.exp229_correlation_aware_agreement import (
    correlation_adjusted_weights,
    spearman_matrix,
)
from experiments.exp226_235_hard_alert_quality.exp231_normal_mode_calibration import (
    fit_normal_modes,
    local_empirical_pvalues,
)
from experiments.exp226_235_hard_alert_quality.exp232_calibration_reliability_gate import reliability_state
from experiments.exp226_235_hard_alert_quality.exp233_conformal_fdr import FDR_LEVELS, bh_indices
from experiments.exp226_235_hard_alert_quality.exp234_precision_first_rescue import (
    research_rescue_candidates,
)
from experiments.exp226_235_hard_alert_quality.exp235_shadow_export import build_shadow_rows
from experiments.exp226_235_hard_alert_quality.reporting import (
    dashboard_policy_summaries,
    evaluate_prediction_rows,
    summarize_evaluated_rows,
    write_csv,
)
from experiments.exp226_235_hard_alert_quality.independent_reaggregate import verify_experiment
from experiments.exp226_235_hard_alert_quality.score_cache import FrozenScoreRepository


FORBIDDEN_PREDICTION_KEYS = {
    "y_test", "labels", "test_labels", "anomaly_positions", "tp", "fp",
    "fn", "f1", "precision", "recall", "combined_f1",
}
CONFIRMATION_ALPHA = 0.01
MARGIN_LEVELS = (0.0, 0.5, 0.8)
WEIGHT_LEVELS = (0.50, 0.67, 0.75)
HEARTBEAT_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_dashboard_heartbeat.json")


def parse_experiment_range(value: str) -> tuple[int, ...]:
    text = str(value).strip()
    if "-" in text:
        left, right = text.split("-", 1)
        start, stop = int(left), int(right)
        experiments = tuple(range(start, stop + 1))
    else:
        experiments = tuple(int(item) for item in text.split(",") if item.strip())
    if not experiments or min(experiments) < 226 or max(experiments) > 235:
        raise ValueError("experiment selection must stay within Exp226-235")
    return experiments


def _jsonable_prediction(row: PredictionRow) -> dict[str, object]:
    payload = dataclasses.asdict(row)
    if FORBIDDEN_PREDICTION_KEYS & set(payload):
        raise ValueError("prediction payload contains evaluation fields")
    return payload


def _atomic_write(path: Path, content: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def freeze_prediction_rows(
    rows: Iterable[PredictionRow], prediction_path: Path, manifest_path: Path
) -> dict[str, object]:
    payloads = sorted(
        (_jsonable_prediction(row) for row in rows),
        key=lambda row: (str(row["experiment_id"]), str(row["dataset"]), str(row["variant"])),
    )
    serialized = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in payloads
    )
    _atomic_write(Path(prediction_path), serialized)
    digest = hashlib.sha256(serialized).hexdigest()
    manifest = {
        "schema": "exp226_235_prediction_manifest_v1",
        "prediction_rows": len(payloads),
        "prediction_sha256": digest,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    _atomic_write(
        Path(manifest_path),
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    return manifest


def load_verified_prediction_rows(
    prediction_path: Path, manifest_path: Path
) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual = sha256_file(Path(prediction_path))
    if actual != manifest.get("prediction_sha256"):
        raise ValueError(
            f"prediction SHA-256 mismatch: expected={manifest.get('prediction_sha256')} actual={actual}"
        )
    rows = [
        json.loads(line)
        for line in Path(prediction_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("prediction_rows", -1)):
        raise ValueError("prediction row count mismatch")
    for row in rows:
        if FORBIDDEN_PREDICTION_KEYS & set(row):
            raise ValueError("verified prediction contains evaluation fields")
    return rows


def save_prediction_checkpoint(
    path: Path,
    *,
    experiment: int,
    config_hash: str,
    input_hash: str,
    completed_datasets: Iterable[str],
    rows: Iterable[PredictionRow],
) -> None:
    payload = {
        "schema": "exp226_235_prediction_checkpoint_v1",
        "experiment": int(experiment),
        "config_hash": str(config_hash),
        "input_hash": str(input_hash),
        "completed_datasets": sorted(set(str(name) for name in completed_datasets)),
        "rows": [_jsonable_prediction(row) for row in rows],
    }
    _atomic_write(
        Path(path),
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )


def load_prediction_checkpoint(
    path: Path, *, experiment: int, config_hash: str, input_hash: str
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    contract = (payload.get("experiment"), payload.get("config_hash"), payload.get("input_hash"))
    expected = (int(experiment), str(config_hash), str(input_hash))
    if contract != expected:
        raise ValueError(f"checkpoint contract mismatch: expected={expected} actual={contract}")
    if payload.get("schema") != "exp226_235_prediction_checkpoint_v1":
        raise ValueError("checkpoint schema mismatch")
    for row in payload.get("rows", []):
        if FORBIDDEN_PREDICTION_KEYS & set(row):
            raise ValueError("checkpoint contains evaluation fields")
    return payload


def _prediction_contract_hash(experiment: int, dataset_limit: int | None = None) -> str:
    payload = {
        "experiment": int(experiment),
        "seed": SEED,
        "confirmation_alpha": CONFIRMATION_ALPHA,
        "margin_levels": MARGIN_LEVELS,
        "weight_levels": WEIGHT_LEVELS,
        "fdr_levels": FDR_LEVELS,
        "dataset_limit": dataset_limit,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _prediction_input_hash() -> str:
    from experiments.exp226_235_hard_alert_quality.baseline_exp137 import EXP137_RESULTS
    from experiments.exp226_235_hard_alert_quality.score_cache import (
        CANONICAL_MANIFEST_SHA256,
        EXP153_MANIFEST_SHA256,
    )

    payload = {
        "database": sha256_file(DB_PATH),
        "exp137": sha256_file(EXP137_RESULTS),
        "full_score_manifest": EXP153_MANIFEST_SHA256,
        "canonical_manifest": CANONICAL_MANIFEST_SHA256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_pipeline_status(path: Path, status: dict[str, object]) -> None:
    payload = dict(status)
    payload.setdefault("schema", "exp226_235_pipeline_status_v1")
    _atomic_write(
        Path(path),
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )


def _source_pvalues(scores: ScoreBundle) -> dict[str, np.ndarray]:
    return {
        name: empirical_upper_pvalues(scores.train_oof[name], scores.test_scores[name])
        for name in sorted(scores.train_oof)
    }


def _prediction(
    experiment: int,
    baseline: BaselineLanes,
    variant: str,
    hard: Iterable[int],
    review: Iterable[int],
    evidence: dict[str, object],
) -> PredictionRow:
    return PredictionRow(
        experiment_id=f"experiment_{experiment}_hard_alert_quality",
        dataset=baseline.dataset,
        variant=variant,
        hard_indices=tuple(sorted(set(int(value) for value in hard))),
        review_indices=tuple(sorted(set(int(value) for value in review))),
        evidence=evidence,
        autonomous_alert=bool(set(hard)),
        contains_test_labels=False,
        contains_evaluation_metrics=False,
    )


def _train_drift(values: np.ndarray) -> float:
    scores = np.asarray(values, dtype=float)
    if len(scores) < 4:
        return float("inf")
    left, right = np.array_split(scores, 2)
    scale = 1.4826 * np.median(np.abs(scores - np.median(scores)))
    scale = max(float(scale), 1e-12)
    return abs(float(np.median(left) - np.median(right))) / scale


def generate_rows_for_dataset(
    experiment: int, baseline: BaselineLanes, scores: ScoreBundle
) -> list[PredictionRow]:
    number = int(experiment)
    if bool(scores.provenance.get("source_abstained", False)):
        return [_prediction(
            number,
            baseline,
            "calibration_abstention",
            (),
            baseline.hard,
            {
                "status": "calibration_abstention_n_train_lt_5",
                "n_train": int(scores.provenance.get("n_train", 0)),
                "source_abstained": True,
            },
        )]
    pvalues = _source_pvalues(scores)
    hard = set(baseline.hard)
    available = sorted(pvalues)
    temporal = set(np.flatnonzero(pvalues["temporal_random_kernel"] <= CONFIRMATION_ALPHA))
    canonical_available = "canonical_normal_state" in pvalues
    canonical = (
        set(np.flatnonzero(pvalues["canonical_normal_state"] <= CONFIRMATION_ALPHA))
        if canonical_available
        else set()
    )

    if number == 226:
        evidence = audit_frozen_train_risk(baseline.dataset)
        return [_prediction(226, baseline, "train_normal_conditional_risk", (), (), evidence)]
    if number == 227:
        if not canonical_available:
            return [_prediction(227, baseline, "canonical_unavailable", (), hard, {
                "status": "blocked_missing_canonical_source"
            })]
        tiers = confirmation_tiers(hard, temporal, canonical)
        return [
            _prediction(227, baseline, "dual_only", tiers.dual, hard - set(tiers.dual), {
                "confirmation_alpha": CONFIRMATION_ALPHA, "confirmation": "dual"
            }),
            _prediction(227, baseline, "single_or_dual", set(tiers.dual) | set(tiers.single), tiers.unconfirmed, {
                "confirmation_alpha": CONFIRMATION_ALPHA, "confirmation": "at_least_one"
            }),
        ]
    if number == 228:
        selected_sources = [name for name in ("temporal_random_kernel", "canonical_normal_state") if name in pvalues]
        margins = np.vstack([conformal_margin(pvalues[name], CONFIRMATION_ALPHA) for name in selected_sources])
        minimum = np.min(margins, axis=0)
        rows = []
        for level in MARGIN_LEVELS:
            accepted = hard & set(np.flatnonzero(minimum >= level))
            rows.append(_prediction(228, baseline, f"margin_{level:g}", accepted, hard - accepted, {
                "margin_level": level, "sources": selected_sources
            }))
        return rows
    if number == 229:
        matrix = spearman_matrix(scores.train_oof)
        weights = correlation_adjusted_weights(matrix)
        hit_weight = np.zeros(baseline.test_size, dtype=float)
        for name, weight in weights.items():
            hit_weight[pvalues[name] <= CONFIRMATION_ALPHA] += weight
        rows = []
        for level in WEIGHT_LEVELS:
            accepted = hard & set(np.flatnonzero(hit_weight >= level))
            rows.append(_prediction(229, baseline, f"weight_{level:g}", accepted, hard - accepted, {
                "weight_level": level, "weights": weights
            }))
        return rows
    if number == 230:
        return [_prediction(230, baseline, "seed_stability_unavailable", (), hard, {
            "status": "blocked_missing_frozen_seed_repetitions"
        })]
    if number == 231:
        names = sorted(scores.train_oof)
        train = np.column_stack([scores.train_oof[name] for name in names])
        test = np.column_stack([scores.test_scores[name] for name in names])
        center = np.median(train, axis=0)
        scale = 1.4826 * np.median(np.abs(train - center), axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        train_z, test_z = (train - center) / scale, (test - center) / scale
        rows = []
        for clusters in (1, 2, 4):
            if len(train_z) < clusters:
                continue
            modes = fit_normal_modes(train_z, clusters=clusters, seed=20260719)
            test_dist = np.sum((test_z[:, None, :] - modes.centers[None, :, :]) ** 2, axis=2)
            test_modes = np.argmin(test_dist, axis=1)
            train_score = np.sqrt(np.sum((train_z - modes.centers[modes.assignments]) ** 2, axis=1))
            test_score = np.sqrt(np.sum((test_z - modes.centers[test_modes]) ** 2, axis=1))
            local_p = local_empirical_pvalues(train_score, modes.assignments, test_score, test_modes)
            accepted = hard & set(np.flatnonzero(local_p <= CONFIRMATION_ALPHA))
            rows.append(_prediction(231, baseline, f"normal_modes_{clusters}", accepted, hard - accepted, {
                "clusters": clusters, "sources": names
            }))
        return rows
    if number == 232:
        source = scores.train_oof["temporal_random_kernel"]
        normal_p = leave_one_out_pvalues(source)
        false_count = int(np.count_nonzero(normal_p <= CONFIRMATION_ALPHA))
        stability = float(np.mean(np.abs(list(spearman_matrix(scores.train_oof)[available[0]].values()))))
        state = reliability_state(
            n_train=len(source),
            attainable_p=1.0 / (len(source) + 1.0),
            risk_upper=wilson_bound(false_count, len(source), side="upper"),
            source_stability=stability,
            train_drift=_train_drift(source),
        )
        accepted = hard if state == "eligible" else set()
        return [_prediction(232, baseline, state, accepted, hard - accepted, {
            "reliability_state": state, "normal_false_count": false_count
        })]
    if number == 233:
        rows = []
        for level in FDR_LEVELS:
            support: dict[int, int] = {}
            for values in pvalues.values():
                for index in bh_indices(values, q=level):
                    support[index] = support.get(index, 0) + 1
            accepted = hard & {index for index, count in support.items() if count >= 2}
            rows.append(_prediction(233, baseline, f"fdr_{level:g}", accepted, hard - accepted, {
                "fdr_level": level, "minimum_sources": 2
            }))
        return rows
    if number == 234:
        excluded = set(baseline.hard) | set(baseline.standard_review) | set(baseline.priority_review)
        hits = {name: set(np.flatnonzero(values <= CONFIRMATION_ALPHA)) for name, values in pvalues.items()}
        return [
            PredictionRow(
                experiment_id="experiment_234_hard_alert_quality",
                dataset=baseline.dataset,
                variant=f"minimum_sources_{minimum}",
                research_indices=tuple(sorted(research_rescue_candidates(
                    source_hits=hits, excluded_indices=excluded, minimum_sources=minimum
                ))),
                evidence={"minimum_sources": minimum, "available_sources": available},
                autonomous_alert=False,
                contains_test_labels=False,
                contains_evaluation_metrics=False,
            )
            for minimum in (2, 3, 4)
            if minimum <= len(hits)
        ]
    if number == 235:
        return [_prediction(235, baseline, "shadow_export_stage", (), (), {
            "status": "handled_after_prediction_freeze"
        })]
    raise ValueError(f"unsupported experiment: {number}")


def _write_runtime_status(
    *,
    experiment: int | None,
    status: str,
    done: int,
    expected: int,
    current_dataset: str | None,
    rows: int,
    error: str | None = None,
) -> None:
    payload = {
        "experiment_id": (
            f"experiment_{experiment}_hard_alert_quality" if experiment is not None else None
        ),
        "active_experiment": experiment,
        "active_label": (
            f"Experiment {experiment}: Hard alert quality" if experiment is not None else None
        ),
        "status": status,
        "done": int(done),
        "expected": int(expected),
        "rows": int(rows),
        "current_dataset": current_dataset,
        "pid": os.getpid(),
        "updated_unix": time.time(),
        "error": error,
    }
    write_pipeline_status(REPO_OUTPUT_ROOT / "pipeline_status.json", payload)
    write_pipeline_status(HEARTBEAT_PATH, payload)


def run_audit(output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    database = audit_database(DB_PATH)
    baseline = verify_exp137_exact_replay(load_exp137_baseline())
    repository = FrozenScoreRepository.from_frozen_manifests()
    report = {
        "database": dataclasses.asdict(database),
        "baseline": dataclasses.asdict(baseline),
        "full_source_datasets": len(repository.source_rows),
        "canonical_source_datasets": len(repository.canonical_rows),
        "retrospective": True,
        "prediction_uses_test_labels": False,
    }
    _atomic_write(
        Path(output_root) / "audit" / "audit_report.json",
        (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    return report


def run_prediction_stage(
    experiments: tuple[int, ...],
    *,
    dataset_limit: int | None,
    output_root: Path = OUTPUT_ROOT,
    resume: bool = False,
    reset: bool = False,
) -> dict[int, dict[str, object]]:
    baseline = load_exp137_baseline()
    names = sorted(baseline)
    if dataset_limit is not None:
        names = names[: int(dataset_limit)]
    repository = FrozenScoreRepository.from_frozen_manifests()
    input_hash = _prediction_input_hash()
    manifests: dict[int, dict[str, object]] = {}
    for experiment in experiments:
        if experiment == 235:
            manifests[experiment] = run_shadow_export_stage(
                names=names, output_root=Path(output_root)
            )
            continue
        rows: list[PredictionRow] = []
        completed: set[str] = set()
        checkpoint_path = Path(output_root) / "checkpoints" / f"experiment_{experiment}.json"
        config_hash = _prediction_contract_hash(experiment, dataset_limit)
        if reset:
            checkpoint_path.unlink(missing_ok=True)
        elif resume and checkpoint_path.exists():
            checkpoint = load_prediction_checkpoint(
                checkpoint_path,
                experiment=experiment,
                config_hash=config_hash,
                input_hash=input_hash,
            )
            completed = {str(name) for name in checkpoint["completed_datasets"]}
            rows = [
                PredictionRow(
                    **{
                        **raw,
                        "hard_indices": tuple(raw.get("hard_indices", ())),
                        "review_indices": tuple(raw.get("review_indices", ())),
                        "research_indices": tuple(raw.get("research_indices", ())),
                    }
                )
                for raw in checkpoint["rows"]
            ]
        _write_runtime_status(
            experiment=experiment, status="running", done=len(completed), expected=len(names),
            current_dataset=None, rows=0,
        )
        try:
            for done, dataset in enumerate(names, 1):
                if dataset in completed:
                    continue
                score_bundle = repository.load(dataset)
                rows.extend(generate_rows_for_dataset(experiment, baseline[dataset], score_bundle))
                completed.add(dataset)
                if done % 10 == 0 or done == len(names):
                    save_prediction_checkpoint(
                        checkpoint_path,
                        experiment=experiment,
                        config_hash=config_hash,
                        input_hash=input_hash,
                        completed_datasets=completed,
                        rows=rows,
                    )
                    _write_runtime_status(
                        experiment=experiment, status="running", done=len(completed),
                        expected=len(names), current_dataset=dataset, rows=len(rows),
                    )
                    print(
                        f"Exp{experiment}: [{done}/{len(names)}] rows={len(rows)} dataset={dataset}",
                        flush=True,
                    )
            prediction = Path(output_root) / "predictions" / f"experiment_{experiment}.jsonl"
            manifest = Path(output_root) / "manifest" / f"experiment_{experiment}_prediction.json"
            manifests[experiment] = freeze_prediction_rows(rows, prediction, manifest)
            save_prediction_checkpoint(
                checkpoint_path,
                experiment=experiment,
                config_hash=config_hash,
                input_hash=input_hash,
                completed_datasets=completed,
                rows=rows,
            )
            _write_runtime_status(
                experiment=experiment, status="completed", done=len(names), expected=len(names),
                current_dataset=names[-1] if names else None, rows=len(rows),
            )
        except Exception as error:
            _write_runtime_status(
                experiment=experiment, status="failed", done=0, expected=len(names),
                current_dataset=None, rows=len(rows), error=repr(error),
            )
            raise
    return manifests


def _prediction_row_from_mapping(raw: dict[str, object]) -> PredictionRow:
    return PredictionRow(
        experiment_id=str(raw["experiment_id"]),
        dataset=str(raw["dataset"]),
        variant=str(raw["variant"]),
        hard_indices=tuple(int(value) for value in raw.get("hard_indices", ())),
        review_indices=tuple(int(value) for value in raw.get("review_indices", ())),
        research_indices=tuple(int(value) for value in raw.get("research_indices", ())),
        evidence=dict(raw.get("evidence", {})),
        autonomous_alert=bool(raw.get("autonomous_alert", False)),
        contains_test_labels=False,
        contains_evaluation_metrics=False,
    )


def run_shadow_export_stage(*, names: list[str], output_root: Path) -> dict[str, object]:
    _write_runtime_status(
        experiment=235, status="running", done=0, expected=len(names),
        current_dataset=None, rows=0,
    )
    shadow_rows: list[dict[str, object]] = []
    counts = {name: 0 for name in names}
    source_hashes: dict[str, str] = {}
    for source_experiment in range(226, 235):
        prediction_path = output_root / "predictions" / f"experiment_{source_experiment}.jsonl"
        manifest_path = output_root / "manifest" / f"experiment_{source_experiment}_prediction.json"
        source_rows = load_verified_prediction_rows(prediction_path, manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hash = str(manifest["prediction_sha256"])
        source_hashes[str(source_experiment)] = source_hash
        exported = build_shadow_rows(
            (_prediction_row_from_mapping(row) for row in source_rows),
            prediction_sha256=source_hash,
            generated_at="frozen_prediction_export",
        )
        for row in exported:
            if row["dataset"] in counts:
                counts[str(row["dataset"])] += 1
                shadow_rows.append(row)
    shadow_rows.sort(
        key=lambda row: (
            str(row["source_experiment_id"]), str(row["dataset"]),
            str(row["variant"]), int(row["candidate_index"]),
        )
    )
    shadow_bytes = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in shadow_rows
    )
    shadow_path = output_root / "shadow" / "experiment_235_shadow.jsonl"
    _atomic_write(shadow_path, shadow_bytes)
    shadow_hash = hashlib.sha256(shadow_bytes).hexdigest()
    summary_rows = [
        PredictionRow(
            experiment_id="experiment_235_hard_alert_quality",
            dataset=name,
            variant="shadow_export",
            evidence={
                "shadow_candidate_rows": counts[name],
                "shadow_sha256": shadow_hash,
                "source_prediction_hashes": source_hashes,
                "scope": "prospective_shadow_input_only",
            },
        )
        for name in names
    ]
    prediction_path = output_root / "predictions" / "experiment_235.jsonl"
    manifest_path = output_root / "manifest" / "experiment_235_prediction.json"
    manifest = freeze_prediction_rows(summary_rows, prediction_path, manifest_path)
    manifest.update({
        "shadow_file": str(shadow_path),
        "shadow_rows": len(shadow_rows),
        "shadow_sha256": shadow_hash,
        "source_prediction_hashes": source_hashes,
    })
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    _write_runtime_status(
        experiment=235, status="completed", done=len(names), expected=len(names),
        current_dataset=names[-1] if names else None, rows=len(summary_rows),
    )
    return manifest


def run_evaluation_stage(
    experiments: tuple[int, ...], *, output_root: Path = OUTPUT_ROOT
) -> list[dict[str, object]]:
    all_summaries: list[dict[str, object]] = []
    for experiment in experiments:
        prediction = Path(output_root) / "predictions" / f"experiment_{experiment}.jsonl"
        manifest = Path(output_root) / "manifest" / f"experiment_{experiment}_prediction.json"
        rows = load_verified_prediction_rows(prediction, manifest)
        evaluated = evaluate_prediction_rows(
            rows, label_loader=load_test_labels_for_evaluation
        )
        summaries = summarize_evaluated_rows(evaluated)
        write_csv(
            Path(output_root) / "results" / f"experiment_{experiment}_results.csv",
            evaluated,
        )
        write_csv(
            Path(output_root) / "results" / f"experiment_{experiment}_summary.csv",
            summaries,
        )
        all_summaries.extend(summaries)
    return all_summaries


def run_report_stage(
    experiments: tuple[int, ...], *, output_root: Path = OUTPUT_ROOT
) -> Path:
    rows: list[dict[str, object]] = []
    dashboard_rows: list[dict[str, object]] = []
    for experiment in experiments:
        path = Path(output_root) / "results" / f"experiment_{experiment}_summary.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
        result_path = Path(output_root) / "results" / f"experiment_{experiment}_results.csv"
        with result_path.open(newline="", encoding="utf-8") as handle:
            dashboard_rows.extend(
                dashboard_policy_summaries(experiment, list(csv.DictReader(handle)))
            )
    if not rows:
        raise ValueError("no experiment summaries available for report")
    output = REPO_OUTPUT_ROOT / "experiment_comparison.csv"
    write_csv(output, rows)
    write_csv(Path(output_root) / "results" / "dashboard_policy_summary.csv", dashboard_rows)
    return output


def run_reaggregation_stage(
    experiments: tuple[int, ...], *, output_root: Path = OUTPUT_ROOT
) -> dict[str, object]:
    reports = [verify_experiment(Path(output_root), experiment) for experiment in experiments]
    report = {"passed": all(item["passed"] for item in reports), "experiments": reports}
    _atomic_write(
        Path(output_root) / "independent_reaggregation.json",
        (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    if not report["passed"]:
        raise ValueError("independent reaggregation mismatch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("audit", "cache", "baseline", "predict", "freeze", "evaluate", "reaggregate", "report", "all"),
        required=True,
    )
    parser.add_argument("--experiments", default="226-235")
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    experiments = parse_experiment_range(args.experiments)
    if args.stage in {"audit", "cache", "baseline", "all"}:
        report = run_audit(args.output_root)
        print(json.dumps(report, sort_keys=True), flush=True)
    if args.stage in {"predict", "freeze", "all"}:
        run_prediction_stage(
            experiments,
            dataset_limit=args.dataset_limit,
            output_root=args.output_root,
            resume=args.resume,
            reset=args.reset,
        )
    if args.stage in {"evaluate", "all"}:
        run_evaluation_stage(experiments, output_root=args.output_root)
    if args.stage in {"reaggregate", "all"}:
        print(json.dumps(run_reaggregation_stage(experiments, output_root=args.output_root)), flush=True)
    if args.stage in {"report", "all"}:
        print(run_report_stage(experiments, output_root=args.output_root), flush=True)


if __name__ == "__main__":
    main()
