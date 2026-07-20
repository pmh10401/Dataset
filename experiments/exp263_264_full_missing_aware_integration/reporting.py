from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from .config import DB_PATH, MIN_TRAIN_FOR_PROMOTION, REQUIRED_PROMOTION_AGREEMENT


LANES = (
    "baseline_hard",
    "autonomous_hard",
    "standard_review",
    "priority_review",
    "review",
)
SOURCE_NAMES = ("statistical", "structural", "change_point", "reconstruction")
SUMMARY_FIELDS = (
    "datasets",
    "alerts",
    "tp",
    "fp",
    "fn",
    "total_anomalies",
    "micro_precision",
    "micro_recall",
    "macro_mean_f1",
    "zero_f1_datasets",
)
STRUCTURED_FIELDS = (
    "baseline_hard_indices",
    "baseline_standard_indices",
    "baseline_priority_indices",
    "final_hard_indices",
    "final_standard_indices",
    "final_priority_indices",
    "promoted_indices",
    "quality_blocked_indices",
    "quality_flag_indices",
    "source_indices",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def read_results(path: Path) -> list[dict[str, object]]:
    """Read persisted evaluation rows without importing the evaluation module."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for field in STRUCTURED_FIELDS:
                try:
                    value = json.loads(str(row[field]))
                except (KeyError, json.JSONDecodeError) as exc:
                    raise ValueError(f"results row has invalid {field}") from exc
                if field == "source_indices":
                    if type(value) is not dict or set(value) != set(SOURCE_NAMES):
                        raise ValueError("results row has invalid source_indices")
                elif type(value) is not list:
                    raise ValueError(f"results row has invalid {field}")
                row[field] = value
            for field in ("n_train", "n_test", "true_anomaly_count"):
                try:
                    row[field] = int(str(row[field]))
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"results row has invalid {field}") from exc
            rows.append(row)
    if not rows:
        raise ValueError("results.csv has no rows")
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _lane_totals(rows: list[dict[str, object]], lane: str) -> dict[str, float | int]:
    tp = sum(int(row[f"{lane}_tp"]) for row in rows)
    fp = sum(int(row[f"{lane}_fp"]) for row in rows)
    alerts = sum(int(row[f"{lane}_alerts"]) for row in rows)
    return {
        "datasets": len(rows),
        "alerts": alerts,
        "tp": tp,
        "fp": fp,
        "micro_precision": tp / alerts if alerts else 0.0,
        "macro_mean_f1": _mean(float(row[f"{lane}_f1"]) for row in rows),
        "zero_f1_datasets": sum(float(row[f"{lane}_f1"]) == 0.0 for row in rows),
    }


def reaggregate_results(rows: list[dict[str, object]]) -> dict[str, object]:
    """Legacy CSV-column aggregation used only to construct trusted test fixtures."""

    return {
        "datasets": len(rows),
        **{lane: _lane_totals(rows, lane) for lane in LANES},
    }


def _lane_indices(row: Mapping[str, object], lane: str, dataset: str) -> set[int]:
    if lane == "review":
        return _integer_set(row, "final_standard_indices", dataset) | _integer_set(
            row, "final_priority_indices", dataset
        )
    field = {
        "baseline_hard": "baseline_hard_indices",
        "autonomous_hard": "final_hard_indices",
        "standard_review": "final_standard_indices",
        "priority_review": "final_priority_indices",
    }[lane]
    return _integer_set(row, field, dataset)


def _lane_metric(indices: set[int], truth: set[int]) -> dict[str, float | int]:
    tp = len(indices & truth)
    fp = len(indices - truth)
    fn = len(truth - indices)
    alerts = len(indices)
    precision = tp / alerts if alerts else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "alerts": alerts,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _require_persisted_metric_agreement(
    row: Mapping[str, object], dataset: str, lane: str, expected: Mapping[str, float | int]
) -> None:
    for field, expected_value in expected.items():
        key = f"{lane}_{field}"
        try:
            actual = int(str(row[key])) if type(expected_value) is int else float(str(row[key]))
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"persisted metric mismatch: dataset={dataset} lane={lane} field={field}"
            ) from exc
        if actual != expected_value:
            raise ValueError(
                f"persisted metric mismatch: dataset={dataset} lane={lane} field={field} "
                f"expected={expected_value!r} actual={actual!r}"
            )


def _truth_for_row(
    row: Mapping[str, object], truth_loader: Callable[[str], set[int]]
) -> set[int]:
    dataset = str(row["dataset"])
    n_test = row["n_test"]
    assert type(n_test) is int
    truth = set(truth_loader(dataset))
    if any(type(index) is not int or index < 0 or index >= n_test for index in truth):
        raise ValueError(f"truth index outside TEST range: dataset={dataset}")
    if int(row["true_anomaly_count"]) != len(truth):
        raise ValueError(f"truth count mismatch: dataset={dataset}")
    return truth


def reaggregate_results_from_truth(
    rows: list[dict[str, object]], *, truth_loader: Callable[[str], set[int]]
) -> dict[str, object]:
    """Recompute every lane metric from persisted indices and independently loaded truth."""

    preflight_routing_rows(rows)
    per_lane = {lane: [] for lane in LANES}
    total_anomalies = 0
    for row in rows:
        dataset = str(row["dataset"])
        truth = _truth_for_row(row, truth_loader)
        total_anomalies += len(truth)
        for lane in LANES:
            metric = _lane_metric(_lane_indices(row, lane, dataset), truth)
            _require_persisted_metric_agreement(row, dataset, lane, metric)
            per_lane[lane].append(metric)

    def totals(metrics: list[dict[str, float | int]]) -> dict[str, float | int]:
        tp = sum(int(metric["tp"]) for metric in metrics)
        fp = sum(int(metric["fp"]) for metric in metrics)
        fn = sum(int(metric["fn"]) for metric in metrics)
        alerts = sum(int(metric["alerts"]) for metric in metrics)
        return {
            "datasets": len(metrics),
            "alerts": alerts,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_anomalies": total_anomalies,
            "micro_precision": tp / alerts if alerts else 0.0,
            "micro_recall": tp / total_anomalies if total_anomalies else 0.0,
            "macro_mean_f1": _mean(float(metric["f1"]) for metric in metrics),
            "zero_f1_datasets": sum(float(metric["f1"]) == 0.0 for metric in metrics),
        }

    return {"datasets": len(rows), **{lane: totals(per_lane[lane]) for lane in LANES}}


def require_exact_summary_agreement(
    reaggregation: Mapping[str, object], summary: Mapping[str, object]
) -> None:
    """Fail closed when the persisted summary differs from CSV-derived totals."""

    if summary.get("datasets") != reaggregation.get("datasets"):
        raise ValueError(
            "summary mismatch: "
            f"top-level datasets expected={reaggregation.get('datasets')!r} "
            f"actual={summary.get('datasets')!r}"
        )
    for lane in LANES:
        recorded = summary.get(lane)
        if type(recorded) is not dict:
            raise ValueError(f"summary mismatch: missing lane={lane}")
        totals = reaggregation.get(lane)
        if type(totals) is not dict:
            raise ValueError(f"reaggregation is missing lane={lane}")
        for field in SUMMARY_FIELDS:
            expected = totals[field]
            actual = recorded.get(field)
            if actual != expected:
                raise ValueError(
                    "summary mismatch: "
                    f"lane={lane} field={field} expected={expected!r} actual={actual!r}"
                )


def _integer_set(row: Mapping[str, object], field: str, dataset: str) -> set[int]:
    values = row.get(field)
    if type(values) is not list or any(type(index) is not int for index in values):
        raise ValueError(f"routing evidence is invalid: dataset={dataset} field={field}")
    return set(values)


def require_persisted_routing_agreement(
    row: Mapping[str, object],
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Rebuild persisted promotion evidence before any retrospective scoring."""

    dataset = str(row.get("dataset", ""))
    n_test = row.get("n_test")
    n_train = row.get("n_train")
    if type(n_test) is not int or n_test < 0:
        raise ValueError(f"routing evidence is invalid: dataset={dataset} field=n_test")
    if type(n_train) is not int or n_train < 0:
        raise ValueError(f"routing evidence is invalid: dataset={dataset} field=n_train")
    baseline = {
        "hard": _integer_set(row, "baseline_hard_indices", dataset),
        "standard": _integer_set(row, "baseline_standard_indices", dataset),
        "priority": _integer_set(row, "baseline_priority_indices", dataset),
    }
    raw_sources = row.get("source_indices")
    if type(raw_sources) is not dict or set(raw_sources) != set(SOURCE_NAMES):
        raise ValueError(f"routing evidence is invalid: dataset={dataset} field=source_indices")
    sources = {
        source: _integer_set(raw_sources, source, dataset) for source in SOURCE_NAMES
    }
    quality_flags = _integer_set(row, "quality_flag_indices", dataset)
    actual_promoted = _integer_set(row, "promoted_indices", dataset)
    actual_quality_blocked = _integer_set(row, "quality_blocked_indices", dataset)
    final = {
        "hard": _integer_set(row, "final_hard_indices", dataset),
        "standard": _integer_set(row, "final_standard_indices", dataset),
        "priority": _integer_set(row, "final_priority_indices", dataset),
    }
    for name, indices in {
        **baseline,
        **sources,
        **final,
        "quality_flags": quality_flags,
        "promoted": actual_promoted,
        "quality_blocked": actual_quality_blocked,
    }.items():
        if any(index < 0 or index >= n_test for index in indices):
            raise ValueError(f"index outside TEST range: dataset={dataset} source={name}")

    support = {
        index: sum(index in indices for indices in sources.values())
        for index in baseline["standard"]
    }
    expected_promoted = {
        index
        for index, count in support.items()
        if n_train >= MIN_TRAIN_FOR_PROMOTION
        and count >= REQUIRED_PROMOTION_AGREEMENT
        and index not in quality_flags
    }
    expected_quality_blocked = {
        index
        for index, count in support.items()
        if count >= REQUIRED_PROMOTION_AGREEMENT and index in quality_flags
    }
    expected_priority_added = {
        index for index, count in support.items() if count >= 2 and index not in expected_promoted
    }
    expected_final = {
        "hard": baseline["hard"] | expected_promoted,
        "standard": baseline["standard"] - expected_promoted - expected_priority_added,
        "priority": baseline["priority"] | expected_priority_added,
    }
    for field, actual, expected in (
        ("promoted_indices", actual_promoted, expected_promoted),
        ("quality_blocked_indices", actual_quality_blocked, expected_quality_blocked),
        ("final_hard_indices", final["hard"], expected_final["hard"]),
        ("final_standard_indices", final["standard"], expected_final["standard"]),
        ("final_priority_indices", final["priority"], expected_final["priority"]),
    ):
        if actual != expected:
            raise ValueError(
                "routing mismatch: "
                f"dataset={dataset} field={field} expected={sorted(expected)} actual={sorted(actual)}"
            )
    return baseline, sources


def preflight_routing_rows(
    rows: list[dict[str, object]],
) -> list[tuple[dict[str, object], dict[str, set[int]], dict[str, set[int]]]]:
    """Validate label-free routing evidence for every row before scoring any row."""

    return [
        (row, *require_persisted_routing_agreement(row))
        for row in rows
    ]


def _metric(indices: set[int], truth: set[int]) -> dict[str, float | int]:
    tp = len(indices & truth)
    fp = len(indices - truth)
    count = len(indices)
    return {
        "count": count,
        "tp": tp,
        "fp": fp,
        "precision": tp / count if count else 0.0,
    }


def _load_truth(dataset: str, database_path: Path) -> set[int]:
    from experiments.exp226_235_hard_alert_quality.database import (
        load_test_labels_for_evaluation,
    )

    labels = load_test_labels_for_evaluation(dataset, path=database_path)
    return {index for index, label in enumerate(labels) if int(label) == 1}


def _merge_metrics(metrics: Iterable[dict[str, float | int]]) -> dict[str, float | int]:
    count = sum(int(metric["count"]) for metric in metrics)
    tp = sum(int(metric["tp"]) for metric in metrics)
    return {"count": count, "tp": tp, "fp": count - tp, "precision": tp / count if count else 0.0}


def source_diagnostics(
    rows: list[dict[str, object]], *, truth_loader: Callable[[str], set[int]]
) -> dict[str, object]:
    """Derive source and lane diagnostics from result-row evidence and TEST truth."""

    source_metrics: dict[str, list[dict[str, float | int]]] = {
        source: [] for source in SOURCE_NAMES
    }
    source_overlap = {
        source: {lane: 0 for lane in ("hard", "standard", "priority", "outside")}
        for source in SOURCE_NAMES
    }
    agreement = {str(count): 0 for count in range(len(SOURCE_NAMES) + 1)}
    promoted: list[dict[str, float | int]] = []
    quality_blocked: list[dict[str, float | int]] = []
    hard_confirmed: list[dict[str, float | int]] = []
    baseline_standard = baseline_priority = final_standard = final_priority = 0

    for row, baseline, sources in preflight_routing_rows(rows):
        dataset = str(row["dataset"])
        n_test = row["n_test"]
        assert type(n_test) is int
        truth = set(truth_loader(dataset))
        if any(index < 0 or index >= n_test for index in truth):
            raise ValueError(f"truth index outside TEST range: dataset={dataset}")
        if int(row["true_anomaly_count"]) != len(truth):
            raise ValueError(f"truth count mismatch: dataset={dataset}")

        baseline_union = set().union(*baseline.values())
        for source, indices in sources.items():
            source_metrics[source].append(_metric(indices, truth))
            for lane in baseline:
                source_overlap[source][lane] += len(indices & baseline[lane])
            source_overlap[source]["outside"] += len(indices - baseline_union)

        for index in baseline["standard"]:
            count = sum(index in indices for indices in sources.values())
            agreement[str(count)] += 1

        promoted.append(_metric(_integer_set(row, "promoted_indices", dataset), truth))
        quality_blocked.append(_metric(
            _integer_set(row, "quality_blocked_indices", dataset), truth
        ))
        hard_confirmed.append(_metric(set().union(*sources.values()) & baseline["hard"], truth))
        baseline_standard += len(baseline["standard"])
        baseline_priority += len(baseline["priority"])
        final_standard += len({int(index) for index in row["final_standard_indices"]})
        final_priority += len({int(index) for index in row["final_priority_indices"]})

    source_output = {}
    for source in SOURCE_NAMES:
        metric = _merge_metrics(source_metrics[source])
        source_output[source] = {
            "raw_candidates": metric["count"],
            "tp": metric["tp"],
            "fp": metric["fp"],
            "precision": metric["precision"],
            "overlap": source_overlap[source],
        }
    return {
        "agreement_scope": "aggregate_baseline_standard_candidates_across_all_result_rows",
        "agreement_rows": len(rows),
        "sources": source_output,
        "agreement_count_distribution": agreement,
        "promoted": _merge_metrics(promoted),
        "quality_blocked": _merge_metrics(quality_blocked),
        "hard_confidence_confirmed": _merge_metrics(hard_confirmed),
        "hard_confidence_confirmation_definition": (
            "union of raw source candidates overlapping preserved Exp253 baseline Hard"
        ),
        "workload_change": {
            "standard_review": {
                "baseline": baseline_standard,
                "final": final_standard,
                "delta": final_standard - baseline_standard,
            },
            "priority_review": {
                "baseline": baseline_priority,
                "final": final_priority,
                "delta": final_priority - baseline_priority,
            },
        },
    }


def _report_markdown(
    reaggregation: Mapping[str, object], diagnostics: Mapping[str, object]
) -> str:
    lines = [
        "# Exp264 Independent Reaggregation",
        "",
        "All metrics below are retrospective. They are independent reaggregations of "
        "`results.csv`; no policy, threshold, source, or lane was selected from these TEST results.",
        "",
        "## Exact Summary Agreement",
        "",
        "| Lane | Alerts | TP | FP | Precision | Mean F1 | Zero-F1 datasets |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lane in LANES:
        metric = reaggregation[lane]
        assert type(metric) is dict
        lines.append(
            f"| {lane} | {metric['alerts']} | {metric['tp']} | {metric['fp']} | "
            f"{float(metric['micro_precision']):.6f} | "
            f"{float(metric['macro_mean_f1']):.6f} | {metric['zero_f1_datasets']} |"
        )

    source_rows = diagnostics["sources"]
    assert type(source_rows) is dict
    lines.extend(
        [
            "",
            "## Source Diagnostics",
            "",
            "| Source | Raw candidates | TP | FP | Precision | Baseline Hard | Standard | Priority | Outside |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source in SOURCE_NAMES:
        metric = source_rows[source]
        assert type(metric) is dict
        overlap = metric["overlap"]
        assert type(overlap) is dict
        lines.append(
            f"| {source} | {metric['raw_candidates']} | {metric['tp']} | {metric['fp']} | "
            f"{float(metric['precision']):.6f} | {overlap['hard']} | {overlap['standard']} | "
            f"{overlap['priority']} | {overlap['outside']} |"
        )

    distribution = diagnostics["agreement_count_distribution"]
    assert type(distribution) is dict
    lines.extend(
        [
            "",
            "| Aggregate source agreement on baseline Standard candidates across all "
            f"{reaggregation['datasets']} result rows | 0 | 1 | 2 | 3 | 4 |",
            "|---|---:|---:|---:|---:|---:|",
            "| Candidates | " + " | ".join(str(distribution[str(count)]) for count in range(5)) + " |",
            "",
            "| Category | Count | TP | FP | Precision |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category in ("promoted", "quality_blocked", "hard_confidence_confirmed"):
        metric = diagnostics[category]
        assert type(metric) is dict
        lines.append(
            f"| {category} | {metric['count']} | {metric['tp']} | {metric['fp']} | "
            f"{float(metric['precision']):.6f} |"
        )
    workload = diagnostics["workload_change"]
    assert type(workload) is dict
    lines.extend(["", "| Review workload | Exp253 baseline | Final | Change |", "|---|---:|---:|---:|"])
    for lane in ("standard_review", "priority_review"):
        metric = workload[lane]
        assert type(metric) is dict
        lines.append(f"| {lane} | {metric['baseline']} | {metric['final']} | {metric['delta']:+d} |")
    lines.extend(
        [
            "",
            "## Provenance And Limits",
            "",
            "This is a retrospective integration validation, not prospective validation or "
            "end-to-end strict TRAIN-only operating evidence. Exp253 provenance remains "
            "`selected_from_prior_test_review=True` and `deployment_eligible=False`. "
            "Autonomous Hard and human-assisted review metrics remain separate.",
        ]
    )
    return "\n".join(lines) + "\n"


def _manifest(output_root: Path, database_path: Path) -> dict[str, object]:
    inputs = (
        "results.csv",
        "summary.json",
        "prediction_manifest.json",
        "lane_transitions.csv",
        "source_agreement_diagnostics.csv",
    )
    evidence = (
        "00_source_contract_audit.json",
        "01_source_coverage.csv",
        "evaluation_manifest.json",
        "status.json",
    )
    package_root = Path(__file__).parent
    experiments_root = package_root.parent
    repository_root = package_root.parents[1]
    code_paths = (
        Path(__file__),
        package_root / "config.py",
        package_root / "prediction.py",
        package_root / "routing.py",
        package_root / "evaluation.py",
        package_root / "inputs.py",
        package_root / "run_pipeline.py",
        experiments_root / "exp226_235_hard_alert_quality" / "database.py",
        repository_root / "scratch" / "run_exp263_264_full_missing_aware_integration.py",
        repository_root / "serve_rank_dashboard.py",
    )
    prediction = output_root / "frozen_predictions.jsonl"
    outputs = ("independent_reaggregation.json", "RESULT_REPORT.md")
    database_path = Path(database_path)
    return {
        "retrospective": True,
        "selected_from_prior_test_review": True,
        "deployment_eligible": False,
        "database": {
            "configured_path": str(database_path),
            "resolved_path": str(database_path.resolve()),
            "sha256": sha256_file(database_path),
        },
        "inputs": {name: sha256_file(output_root / name) for name in inputs},
        "evidence": {name: sha256_file(output_root / name) for name in evidence},
        "code": {str(path.resolve()): sha256_file(path) for path in code_paths},
        "git_commit_sha": _git_commit_sha(package_root.parents[1]),
        "prediction": {prediction.name: sha256_file(prediction)},
        "outputs": {name: sha256_file(output_root / name) for name in outputs},
        "manifest_excludes_self_hash": True,
        "manifest_self_hash_exclusion_reason": "MANIFEST.json is excluded to avoid recursive hashing",
        "status_capture": "status.json is hashed only after the report phase reaches completed",
    }


def _git_commit_sha(repository_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def generate_report(
    output_root: Path,
    *,
    database_path: Path = DB_PATH,
    truth_loader: Callable[[str], set[int]] | None = None,
) -> dict[str, object]:
    """Write independently reaggregated Task 6 artifacts after frozen-boundary checks."""

    output_root = Path(output_root)
    verify_frozen_prediction_boundary(output_root)
    rows = read_results(output_root / "results.csv")
    loader = truth_loader or (lambda dataset: _load_truth(dataset, Path(database_path)))
    reaggregation = reaggregate_results_from_truth(rows, truth_loader=loader)
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    if type(summary) is not dict:
        raise ValueError("summary.json is not an object")
    require_exact_summary_agreement(reaggregation, summary)
    if summary.get("retrospective") is not True:
        raise ValueError("summary provenance mismatch: retrospective")
    if summary.get("selected_from_prior_test_review") is not True:
        raise ValueError("summary provenance mismatch: Exp253 selection")
    if summary.get("deployment_eligible") is not False:
        raise ValueError("summary provenance mismatch: deployment eligibility")

    diagnostics = source_diagnostics(
        rows,
        truth_loader=loader,
    )
    payload: dict[str, object] = {
        "datasets": reaggregation["datasets"],
        "retrospective": True,
        "selected_from_prior_test_review": True,
        "deployment_eligible": False,
        "summary_agreement": {"exact": True, "compared_lanes": list(LANES), "fields": list(SUMMARY_FIELDS)},
        "reaggregation": reaggregation,
        "source_diagnostics": diagnostics,
    }
    _atomic_write(
        output_root / "independent_reaggregation.json",
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
    )
    _atomic_write(output_root / "RESULT_REPORT.md", _report_markdown(reaggregation, diagnostics))
    return payload


def verify_frozen_prediction_boundary(output_root: Path) -> str:
    """Verify the same frozen prediction digest used by evaluation before reporting reads truth."""

    output_root = Path(output_root)
    prediction = output_root / "frozen_predictions.jsonl"
    prediction_manifest = json.loads((output_root / "prediction_manifest.json").read_text())
    evaluation_manifest = json.loads((output_root / "evaluation_manifest.json").read_text())
    if type(prediction_manifest) is not dict or type(evaluation_manifest) is not dict:
        raise ValueError("reporting prediction manifest is not an object")
    actual = sha256_file(prediction)
    if prediction_manifest.get("prediction_sha256") != actual:
        raise ValueError("reporting prediction SHA256 mismatch: prediction manifest")
    if evaluation_manifest.get("prediction_sha256") != actual:
        raise ValueError("reporting prediction SHA256 mismatch: evaluation manifest")
    return actual


def finalize_manifest(output_root: Path, *, database_path: Path = DB_PATH) -> dict[str, object]:
    """Hash completed evidence after status.json has reached its terminal state."""

    output_root = Path(output_root)
    status = json.loads((output_root / "status.json").read_text(encoding="utf-8"))
    if type(status) is not dict or status.get("status") != "completed" or status.get("phase") != "report":
        raise ValueError("manifest requires completed report status")
    manifest = _manifest(output_root, database_path)
    _atomic_write(output_root / "MANIFEST.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest
