from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp391_fn_reachability.fn_reachability_core import ALPHAS, sha256_file
from experiments.exp391_fn_reachability.run_experiment_391_fn_reachability_audit import (
    load_coverage,
    load_frozen_inputs,
)
from experiments.exp392_label_guided_discovery.discovery_core import (
    alpha_slug,
    validated_indices,
)


EXPERIMENT_ID = "experiment_392_label_guided_score_discovery_audit"
EXPECTED_DATASETS = 1117
DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "exp392_label_guided_score_discovery_audit"
PREPARED_MATRIX_NAME = "exp392_frozen_candidate_matrix.jsonl"
PREPARE_MANIFEST_NAME = "exp392_prepare_manifest.json"
SOURCES = (
    "delta",
    "delta_order_1",
    "delta_order_2",
    "ewma",
    "frequency",
    "normal_distance",
    "structural",
    "temporal",
    "topology",
)
SAFETY_FLAGS = {
    "label_guided_discovery": True,
    "eligible_for_deployment": False,
    "test_metrics_select_policy": False,
    "retrospective_counterfactual": True,
    "prospective_validated": False,
}


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def dataset_list_sha256(rows: Sequence[dict]) -> str:
    names = [str(row["dataset"]) for row in rows]
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("discovery_core.py"),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_status(
    output_root: Path,
    *,
    status: str,
    phase: str,
    done: int,
    total: int,
    errors: int = 0,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": EXPERIMENT_ID,
        "status": status,
        "phase": phase,
        "datasets_done": done,
        "datasets_total": total,
        "progress_percent": 100.0 * done / total if total else 0.0,
        "errors": errors,
    }
    (output_root / "exp392_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _lane_indices(row: dict, key: str, n_test: int) -> list[int]:
    return sorted(
        validated_indices(
            row.get(key, []),
            n_test=n_test,
            label=key,
        )
    )


def build_prepare_row(
    hard_row: dict,
    evidence_by_alpha: dict[float, dict[str, dict]],
    coverage_row: dict,
    *,
    alphas: Sequence[float] = ALPHAS,
    sources: Sequence[str] = SOURCES,
) -> dict:
    dataset = str(hard_row["dataset"])
    n_test = int(hard_row["n_test"])
    evidence_payload: dict[str, dict] = {}
    for alpha in alphas:
        evidence_row = evidence_by_alpha[alpha][dataset]
        evidence = evidence_row["evidence"]
        missing = set(sources) - set(evidence)
        if missing:
            raise ValueError(
                f"{dataset} Alpha {alpha} missing sources: {sorted(missing)}"
            )
        alpha_payload = {}
        for source in sorted(sources):
            payload = evidence[source]
            indices = sorted(
                validated_indices(
                    payload.get("indices", []),
                    n_test=n_test,
                    label=f"{dataset} {source} Alpha {alpha}",
                )
            )
            alpha_payload[source] = {
                "available": payload.get("available") is True,
                "indices": indices,
                "resolution_blocked": payload.get("resolution_blocked") is True,
            }
        evidence_payload[alpha_slug(alpha)] = alpha_payload
    return {
        "dataset": dataset,
        "n_train_normal": int(coverage_row["n_train_normal"]),
        "n_test": n_test,
        "baseline_hard_indices": _lane_indices(
            hard_row,
            "automatic_hard_indices",
            n_test,
        ),
        "standard_review_indices": _lane_indices(
            hard_row,
            "standard_review_indices",
            n_test,
        ),
        "priority_review_indices": _lane_indices(
            hard_row,
            "priority_review_indices",
            n_test,
        ),
        "automatic_normal_exact_indices": _lane_indices(
            hard_row,
            "automatic_normal_exact_indices",
            n_test,
        ),
        "quality_conflict_indices": _lane_indices(
            hard_row,
            "quality_conflict_indices",
            n_test,
        ),
        "evidence_by_alpha": evidence_payload,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }


def build_prepare_manifest(
    rows: Sequence[dict],
    prediction_sha256: str,
    input_records: Sequence[dict],
    code_sha256: str,
) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "phase": "prepare",
        "rows": len(rows),
        "datasets": len(rows),
        "dataset_list_sha256": dataset_list_sha256(rows),
        "prediction_file": PREPARED_MATRIX_NAME,
        "prediction_sha256": prediction_sha256,
        "input_records": list(input_records),
        "code_sha256": code_sha256,
        "alphas": list(ALPHAS),
        "sources": list(SOURCES),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        **SAFETY_FLAGS,
    }


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare(output_root: Path, dataset_limit: int | None = None) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    expected = dataset_limit or EXPECTED_DATASETS
    write_status(
        output_root,
        status="running",
        phase="prepare_verify_inputs",
        done=0,
        total=expected,
    )
    hard_rows, evidence_by_alpha, input_records = load_frozen_inputs(dataset_limit)
    coverage = load_coverage()
    rows: list[dict] = []
    for number, hard_row in enumerate(hard_rows, start=1):
        dataset = str(hard_row["dataset"])
        coverage_row = coverage.get(dataset)
        if coverage_row is None:
            raise ValueError(f"missing Exp381 coverage row: {dataset}")
        rows.append(
            build_prepare_row(
                hard_row,
                evidence_by_alpha,
                coverage_row,
            )
        )
        if number % 100 == 0 or number == len(hard_rows):
            write_status(
                output_root,
                status="running",
                phase="prepare",
                done=number,
                total=len(hard_rows),
            )
            print(
                f"Prepare: [{number}/{len(hard_rows)}] dataset={dataset} errors=0",
                flush=True,
            )
    rows.sort(key=lambda row: str(row["dataset"]))
    names = [str(row["dataset"]) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("prepare rows contain duplicate datasets")
    matrix_path = output_root / PREPARED_MATRIX_NAME
    write_jsonl(matrix_path, rows)
    manifest = build_prepare_manifest(
        rows,
        sha256_file(matrix_path),
        input_records,
        code_sha256(),
    )
    (output_root / PREPARE_MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_status(
        output_root,
        status="prepared",
        phase="prepare_completed",
        done=len(rows),
        total=len(rows),
    )
    return manifest


def load_prepared_rows(output_root: Path) -> tuple[list[dict], dict]:
    matrix_path = output_root / PREPARED_MATRIX_NAME
    manifest_path = output_root / PREPARE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(matrix_path) != manifest.get("prediction_sha256"):
        raise ValueError("prepared prediction SHA256 mismatch")
    if manifest.get("contains_test_labels") is not False:
        raise ValueError("prepared matrix contains TEST labels")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("prepared matrix contains evaluation metrics")
    rows = [
        json.loads(line)
        for line in matrix_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("prepared matrix row count mismatch")
    if dataset_list_sha256(rows) != manifest.get("dataset_list_sha256"):
        raise ValueError("prepared matrix dataset list SHA256 mismatch")
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("prepare", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.phase in {"prepare", "all"}:
        prepare(args.output_dir, args.dataset_limit)
    if args.phase in {"evaluate", "all"}:
        raise SystemExit("evaluate phase is not implemented yet")


if __name__ == "__main__":
    main()
