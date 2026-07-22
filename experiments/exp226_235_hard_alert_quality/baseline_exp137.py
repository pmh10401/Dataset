from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from experiments.exp226_235_hard_alert_quality.config import DATA_ROOT


EXP137_RESULTS = DATA_ROOT / "experiment_137_operational_triage_results.csv"


def parse_indices(text: str | None) -> frozenset[int]:
    normalized = (text or "").replace(",", " ").strip()
    return frozenset(int(token) for token in normalized.split() if token)


@dataclass(frozen=True)
class BaselineLanes:
    dataset: str
    test_size: int
    hard: frozenset[int]
    standard_review: frozenset[int]
    priority_review: frozenset[int]
    hard_tp: int
    hard_fp: int


@dataclass(frozen=True)
class BaselineReplayReport:
    datasets: int
    hard_alerts: int
    hard_tp: int
    hard_fp: int
    lane_overlap_datasets: int
    index_error_datasets: int


def load_exp137_baseline(path: Path = EXP137_RESULTS) -> dict[str, BaselineLanes]:
    rows: dict[str, BaselineLanes] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            dataset = str(raw["dataset_name"])
            if dataset in rows:
                raise ValueError(f"duplicate Exp137 dataset: {dataset}")
            rows[dataset] = BaselineLanes(
                dataset=dataset,
                test_size=int(float(raw["test_size"])),
                hard=parse_indices(raw.get("hard_alert_indices")),
                standard_review=parse_indices(raw.get("standard_review_indices")),
                priority_review=parse_indices(raw.get("priority_review_indices")),
                hard_tp=int(float(raw["hard_tp"])),
                hard_fp=int(float(raw["hard_fp"])),
            )
    if not rows:
        raise ValueError("Exp137 baseline is empty")
    return rows


def verify_exp137_exact_replay(
    baseline: dict[str, BaselineLanes],
) -> BaselineReplayReport:
    overlap = 0
    index_errors = 0
    for row in baseline.values():
        if (
            row.hard & row.standard_review
            or row.hard & row.priority_review
            or row.standard_review & row.priority_review
        ):
            overlap += 1
        indices = row.hard | row.standard_review | row.priority_review
        if any(index < 0 or index >= row.test_size for index in indices):
            index_errors += 1
        if len(row.hard) != row.hard_tp + row.hard_fp:
            raise ValueError(f"Exp137 Hard metric mismatch: {row.dataset}")
    return BaselineReplayReport(
        datasets=len(baseline),
        hard_alerts=sum(len(row.hard) for row in baseline.values()),
        hard_tp=sum(row.hard_tp for row in baseline.values()),
        hard_fp=sum(row.hard_fp for row in baseline.values()),
        lane_overlap_datasets=overlap,
        index_error_datasets=index_errors,
    )
