from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from .config import DB_PATH


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
    "micro_precision",
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
    """Independently recompute summary metrics from persisted result rows."""

    return {
        "datasets": len(rows),
        **{lane: _lane_totals(rows, lane) for lane in LANES},
    }


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

    for row in rows:
        dataset = str(row["dataset"])
        truth = set(truth_loader(dataset))
        n_test = int(row["n_test"])
        if any(index < 0 or index >= n_test for index in truth):
            raise ValueError(f"truth index outside TEST range: dataset={dataset}")
        if int(row["true_anomaly_count"]) != len(truth):
            raise ValueError(f"truth count mismatch: dataset={dataset}")

        baseline = {
            "hard": {int(index) for index in row["baseline_hard_indices"]},
            "standard": {int(index) for index in row["baseline_standard_indices"]},
            "priority": {int(index) for index in row["baseline_priority_indices"]},
        }
        baseline_union = set().union(*baseline.values())
        raw_sources = row["source_indices"]
        assert type(raw_sources) is dict
        sources = {
            source: {int(index) for index in raw_sources[source]}
            for source in SOURCE_NAMES
        }
        for name, indices in {**baseline, **sources}.items():
            if any(index < 0 or index >= n_test for index in indices):
                raise ValueError(f"index outside TEST range: dataset={dataset} source={name}")
        for source, indices in sources.items():
            source_metrics[source].append(_metric(indices, truth))
            for lane in baseline:
                source_overlap[source][lane] += len(indices & baseline[lane])
            source_overlap[source]["outside"] += len(indices - baseline_union)

        for index in baseline["standard"]:
            count = sum(index in indices for indices in sources.values())
            agreement[str(count)] += 1

        promoted.append(_metric({int(index) for index in row["promoted_indices"]}, truth))
        quality_blocked.append(
            _metric({int(index) for index in row["quality_blocked_indices"]}, truth)
        )
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
        "agreement_scope": "baseline_standard_indices",
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
            "| Source agreement on baseline Standard candidates | 0 | 1 | 2 | 3 | 4 |",
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
    code_paths = (Path(__file__),)
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
        "code": {str(path.resolve()): sha256_file(path) for path in code_paths},
        "prediction": {prediction.name: sha256_file(prediction)},
        "outputs": {name: sha256_file(output_root / name) for name in outputs},
        "manifest_excludes_self_hash": True,
    }


def generate_report(
    output_root: Path,
    *,
    database_path: Path = DB_PATH,
    truth_loader: Callable[[str], set[int]] | None = None,
) -> dict[str, object]:
    """Write Task 6 artifacts after strict CSV-to-summary consistency checks."""

    output_root = Path(output_root)
    rows = read_results(output_root / "results.csv")
    reaggregation = reaggregate_results(rows)
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
        truth_loader=truth_loader
        or (lambda dataset: _load_truth(dataset, Path(database_path))),
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
    _atomic_write(
        output_root / "MANIFEST.json",
        json.dumps(_manifest(output_root, database_path), sort_keys=True, indent=2) + "\n",
    )
    return payload
