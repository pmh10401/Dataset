from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .config import ALPHA, EVIDENCE_SOURCES, EXP236_PATH, EXP249_PATH, MINI_VARIANTS


@dataclass(frozen=True)
class FrozenDatasetInputs:
    dataset: str
    n_test: int
    hard: set[int]
    standard: set[int]
    priority: set[int]
    mini1024: set[int]
    mini4096: set[int]
    ordinal: set[int]
    complexity: set[int]
    fourier: set[int]
    masked: set[int]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_indices(dataset: str, n_test: int, values: set[int]) -> None:
    if any(index < 0 or index >= n_test for index in values):
        raise ValueError(f"candidate outside TEST range: dataset={dataset}")


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _select(path: Path, variant: str, alpha: float) -> dict[str, set[int]]:
    selected: dict[str, set[int]] = {}
    for row in _rows(path):
        if str(row.get("variant")) != variant or abs(float(row.get("alpha", -1)) - alpha) > 1e-12:
            continue
        dataset = str(row["dataset"])
        if dataset in selected:
            raise ValueError(f"duplicate source row: {path} dataset={dataset}")
        selected[dataset] = set(map(int, row["candidate_indices"]))
    return selected


def load_frozen_inputs() -> dict[str, FrozenDatasetInputs]:
    baseline_rows = _rows(EXP236_PATH)
    baseline = {str(row["dataset"]): row for row in baseline_rows}
    if len(baseline_rows) != 1117 or len(baseline) != 1117:
        raise ValueError("Exp236 must contain 1,117 unique datasets")
    mini = {name: _select(EXP249_PATH, variant, alpha) for name, (variant, alpha) in MINI_VARIANTS.items()}
    evidence = {name: _select(path, variant, ALPHA) for name, (path, variant) in EVIDENCE_SOURCES.items()}
    required = set(baseline)
    for name, values in {**mini, **evidence}.items():
        if set(values) != required:
            missing = sorted(required - set(values))[:5]
            extra = sorted(set(values) - required)[:5]
            raise ValueError(f"source coverage mismatch: {name} missing={missing} extra={extra}")
    output = {}
    for dataset, row in baseline.items():
        n_test = int(row["n_test"])
        values = {
            "hard": set(map(int, row["hard_indices"])),
            "standard": set(map(int, row["standard_review_indices"])),
            "priority": set(map(int, row["priority_review_indices"])),
            "mini1024": mini["mini1024"][dataset],
            "mini4096": mini["mini4096"][dataset],
            **{name: evidence[name][dataset] for name in EVIDENCE_SOURCES},
        }
        for indices in values.values():
            validate_indices(dataset, n_test, indices)
        output[dataset] = FrozenDatasetInputs(dataset=dataset, n_test=n_test, **values)
    return output


def audit_payload() -> dict[str, object]:
    paths = {"exp236": EXP236_PATH, "exp249": EXP249_PATH}
    paths.update({name: path for name, (path, _) in EVIDENCE_SOURCES.items()})
    inputs = load_frozen_inputs()
    return {
        "datasets": len(inputs),
        "alpha": ALPHA,
        "input_files": {name: str(path) for name, path in paths.items()},
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "contains_test_labels": False,
        "selected_from_prior_test_review": True,
        "retrospective": True,
        "deployment_eligible": False,
    }


def audit_inputs(output_root: Path) -> dict[str, object]:
    payload = audit_payload()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "audit.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
