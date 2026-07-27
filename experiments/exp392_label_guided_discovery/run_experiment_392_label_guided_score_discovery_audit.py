from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import fisher_exact

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp391_fn_reachability.fn_reachability_core import ALPHAS, sha256_file
from experiments.exp391_fn_reachability.run_experiment_391_fn_reachability_audit import (
    classify_baseline,
    load_coverage,
    load_frozen_inputs,
    load_test_labels,
)
from experiments.exp392_label_guided_discovery.discovery_core import (
    assign_research_label,
    benjamini_hochberg,
    bootstrap_intervals,
    candidate_indices_for_rule,
    concentration_diagnostics,
    enumerate_combination_rules,
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


def build_observation_catalog() -> list[dict]:
    catalog: list[dict] = []
    for alpha in ALPHAS:
        for source in SOURCES:
            catalog.append(
                {
                    "observation_id": (
                        f"source__{source}__alpha_{alpha_slug(alpha)}"
                    ),
                    "observation_type": "source",
                    "alpha": alpha,
                    "operator": "source",
                    "sources": (source,),
                    "minimum_votes": 1,
                    "rule": None,
                }
            )
        for rule in enumerate_combination_rules(SOURCES, alpha=alpha):
            catalog.append(
                {
                    "observation_id": rule.rule_id,
                    "observation_type": "combination",
                    "alpha": alpha,
                    "operator": rule.operator,
                    "sources": rule.sources,
                    "minimum_votes": rule.minimum_votes,
                    "rule": rule,
                }
            )
    return sorted(catalog, key=lambda row: str(row["observation_id"]))


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


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate_metrics(
    candidates: set[int],
    *,
    truth: set[int],
    hard: set[int],
    review: set[int],
) -> tuple[int, int, int, int, int, int, int]:
    missed = truth - hard
    recovered = candidates & missed
    added_fp = candidates - truth
    review_overlap = candidates & review
    review_tp = recovered & review
    review_fp = review_overlap - truth
    novel_tp = recovered - review
    return (
        len(candidates),
        len(recovered),
        len(added_fp),
        len(review_overlap),
        len(review_tp),
        len(review_fp),
        len(novel_tp),
    )


def _direct_rule_candidates(
    operator: str,
    sources: Sequence[str],
    source_sets: dict[str, set[int]],
) -> set[int]:
    sets = [source_sets[source] for source in sources]
    if operator == "union":
        return set().union(*sets)
    if operator == "intersection":
        return set.intersection(*sets)
    if operator == "atleast2":
        votes: Counter[int] = Counter()
        for values in sets:
            votes.update(values)
        return {index for index, count in votes.items() if count >= 2}
    raise ValueError(f"unsupported operator: {operator}")


def _lineage(dataset: str) -> str:
    return dataset.rsplit("_normal_", 1)[0]


def _baseline_lane(
    index: int,
    *,
    hard: set[int],
    standard: set[int],
    priority: set[int],
    exact: set[int],
) -> str:
    if index in hard:
        return "hard"
    if index in priority:
        return "priority_review"
    if index in standard:
        return "standard_review"
    if index in exact:
        return "automatic_normal_exact"
    return "none"


def _summary_row(
    observation: dict,
    *,
    available: np.ndarray,
    baseline_fn: np.ndarray,
    metric_arrays: dict[str, np.ndarray],
) -> dict:
    index = int(observation["matrix_index"])
    candidate_count = int(metric_arrays["candidate_count"][index].sum())
    recovered_fn = int(metric_arrays["recovered_fn"][index].sum())
    added_fp = int(metric_arrays["added_fp"][index].sum())
    review_overlap_count = int(
        metric_arrays["review_overlap_count"][index].sum()
    )
    review_recovered_fn = int(
        metric_arrays["review_recovered_fn"][index].sum()
    )
    review_overlap_fp = int(
        metric_arrays["review_overlap_fp"][index].sum()
    )
    return {
        "observation_id": observation["observation_id"],
        "observation_type": observation["observation_type"],
        "alpha": observation["alpha"],
        "operator": observation["operator"],
        "sources": "|".join(observation["sources"]),
        "minimum_votes": observation["minimum_votes"],
        "datasets_total": int(available.shape[1]),
        "datasets_available": int(available[index].sum()),
        "baseline_fn": int(baseline_fn.sum()),
        "candidate_count": candidate_count,
        "recovered_fn": recovered_fn,
        "added_fp": added_fp,
        "candidate_precision": (
            recovered_fn / candidate_count if candidate_count else 0.0
        ),
        "fn_recovery_rate": (
            recovered_fn / int(baseline_fn.sum()) if baseline_fn.sum() else 0.0
        ),
        "review_overlap_count": review_overlap_count,
        "review_recovered_fn": review_recovered_fn,
        "review_overlap_fp": review_overlap_fp,
        "review_overlap_precision": (
            review_recovered_fn / review_overlap_count
            if review_overlap_count
            else 0.0
        ),
        "novel_recovered_fn": int(
            metric_arrays["novel_recovered_fn"][index].sum()
        ),
        "tp_contributing_datasets": int(
            np.count_nonzero(metric_arrays["recovered_fn"][index])
        ),
        "fp_contributing_datasets": int(
            np.count_nonzero(metric_arrays["added_fp"][index])
        ),
        **SAFETY_FLAGS,
    }


def _population_masks(dataset_metadata: Sequence[dict]) -> dict[str, np.ndarray]:
    n_test = np.array([int(row["n_test"]) for row in dataset_metadata])
    n_train = np.array(
        [int(row["n_train_normal"]) for row in dataset_metadata]
    )
    category = np.array(
        [str(row["baseline_category"]) for row in dataset_metadata]
    )
    return {
        "all": np.ones(len(dataset_metadata), dtype=bool),
        "test_lt_100": n_test < 100,
        "test_ge_100": n_test >= 100,
        "zero_f1": np.char.startswith(category.astype(str), "zero_"),
        "zero_no_hard": category == "zero_no_hard",
        "zero_wrong_hard": category == "zero_wrong_hard",
        "train_lt_5": n_train < 5,
        "train_5_19": (n_train >= 5) & (n_train < 20),
        "train_20_49": (n_train >= 20) & (n_train < 50),
        "train_ge_50": n_train >= 50,
    }


def build_stability_rows(
    catalog: Sequence[dict],
    dataset_metadata: Sequence[dict],
    available: np.ndarray,
    baseline_fn: np.ndarray,
    metric_arrays: dict[str, np.ndarray],
) -> list[dict]:
    rows: list[dict] = []
    for population, mask in _population_masks(dataset_metadata).items():
        for observation in catalog:
            index = int(observation["matrix_index"])
            candidates = int(metric_arrays["candidate_count"][index, mask].sum())
            recovered = int(metric_arrays["recovered_fn"][index, mask].sum())
            added_fp = int(metric_arrays["added_fp"][index, mask].sum())
            review_count = int(
                metric_arrays["review_overlap_count"][index, mask].sum()
            )
            review_tp = int(
                metric_arrays["review_recovered_fn"][index, mask].sum()
            )
            rows.append(
                {
                    "population": population,
                    "observation_id": observation["observation_id"],
                    "observation_type": observation["observation_type"],
                    "alpha": observation["alpha"],
                    "operator": observation["operator"],
                    "sources": "|".join(observation["sources"]),
                    "datasets": int(mask.sum()),
                    "datasets_available": int(available[index, mask].sum()),
                    "baseline_fn": int(baseline_fn[mask].sum()),
                    "candidate_count": candidates,
                    "recovered_fn": recovered,
                    "added_fp": added_fp,
                    "candidate_precision": (
                        recovered / candidates if candidates else 0.0
                    ),
                    "fn_recovery_rate": (
                        recovered / int(baseline_fn[mask].sum())
                        if baseline_fn[mask].sum()
                        else 0.0
                    ),
                    "review_overlap_count": review_count,
                    "review_recovered_fn": review_tp,
                    "review_overlap_precision": (
                        review_tp / review_count if review_count else 0.0
                    ),
                    "novel_recovered_fn": int(
                        metric_arrays["novel_recovered_fn"][index, mask].sum()
                    ),
                    **SAFETY_FLAGS,
                }
            )
    return rows


def render_report(
    source_rows: Sequence[dict],
    combination_rows: Sequence[dict],
    dataset_metadata: Sequence[dict],
) -> str:
    all_rows = list(source_rows) + list(combination_rows)
    labels = Counter(str(row["research_label"]) for row in all_rows)
    fixed_source_rows = sorted(
        (
            row
            for row in source_rows
            if abs(float(row["alpha"]) - 0.005) < 1e-12
        ),
        key=lambda row: str(row["sources"]),
    )
    preregistration = sorted(
        (
            row
            for row in all_rows
            if row["research_label"] == "preregistration_candidate"
        ),
        key=lambda row: str(row["observation_id"]),
    )
    total_fn = sum(int(row["baseline_fn"]) for row in dataset_metadata)
    large_fn = sum(
        int(row["baseline_fn"])
        for row in dataset_metadata
        if int(row["n_test"]) >= 100
    )
    lines = [
        "# Exp392 정답 기반 점수원 역 탐색 결과",
        "",
        "## 해석 경계",
        "",
        "- 이 실험은 TEST 정답을 사용하는 후향 연구 감사다.",
        "- 자동 경보, 일반 검토, 우선 검토, exact-normal lane은 변경하지 않았다.",
        "- TEST 결과로 Alpha, 점수원 또는 조합을 운영 정책으로 선택하지 않았다.",
        "- 아래 연구 분류는 별도 사전등록 필요성을 표시할 뿐 배포 자격이 아니다.",
        "",
        "## 분석 범위",
        "",
        f"- 데이터셋: {len(dataset_metadata):,}개",
        f"- 기준 FN: {total_fn:,}건",
        f"- TEST 100 이상 데이터셋 FN: {large_fn:,}건",
        f"- 단일 근거 관측: {len(source_rows):,}개",
        f"- 2·3개 근거 조합 관측: {len(combination_rows):,}개",
        "",
        "## 고정 연구 분류",
        "",
        f"- unsupported: {labels['unsupported']:,}개",
        f"- research_signal: {labels['research_signal']:,}개",
        (
            "- preregistration_candidate: "
            f"{labels['preregistration_candidate']:,}개"
        ),
        "",
        "## Alpha 0.005 단일 근거 진단",
        "",
        "| 근거 | 후보 | FN 회수 | 추가 FP | 정밀도 | TP 계보 | 연구 분류 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(
        (
            f"| {row['sources']} | {int(row['candidate_count']):,} | "
            f"{int(row['recovered_fn']):,} | {int(row['added_fp']):,} | "
            f"{100.0 * float(row['candidate_precision']):.2f}% | "
            f"{int(row['tp_contributing_lineages']):,} | "
            f"{row['research_label']} |"
        )
        for row in fixed_source_rows
    )
    lines.extend(
        [
            "",
            "## 사전등록 후보",
            "",
        ]
    )
    if preregistration:
        lines.extend(
            f"- `{row['observation_id']}`" for row in preregistration
        )
    else:
        lines.append("- 고정 기준을 모두 충족한 관측 없음")
    lines.extend(
        [
            "",
            "## 운영 결론",
            "",
            "- Exp392 결과만으로 자동 경보를 확대하지 않는다.",
            "- 특정 계보 집중 신호는 일반화 근거로 사용하지 않는다.",
            "- 후속 실험은 최대 한 개 가설만 새 번호로 사전등록한다.",
            "- 신규 잠금 데이터 또는 prospective Shadow 검증 전에는 배포하지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate(output_root: Path) -> dict:
    write_status(
        output_root,
        status="running",
        phase="evaluate_verify_frozen_prediction",
        done=0,
        total=EXPECTED_DATASETS,
    )
    prepared_rows, prepare_manifest = load_prepared_rows(output_root)

    # TEST labels become reachable only after the frozen prepare hash and
    # dataset-list hash have both passed in load_prepared_rows().
    labels_by_dataset = load_test_labels(
        {str(row["dataset"]) for row in prepared_rows}
    )
    catalog = build_observation_catalog()
    for index, observation in enumerate(catalog):
        observation["matrix_index"] = index
    observation_by_source = {
        (float(row["alpha"]), str(row["sources"][0])): row
        for row in catalog
        if row["observation_type"] == "source"
    }
    combinations_by_alpha: dict[float, list[dict]] = defaultdict(list)
    for row in catalog:
        if row["observation_type"] == "combination":
            combinations_by_alpha[float(row["alpha"])].append(row)

    n_observations = len(catalog)
    n_datasets = len(prepared_rows)
    available = np.zeros((n_observations, n_datasets), dtype=np.uint8)
    metric_names = (
        "candidate_count",
        "recovered_fn",
        "added_fp",
        "review_overlap_count",
        "review_recovered_fn",
        "review_overlap_fp",
        "novel_recovered_fn",
    )
    metric_arrays = {
        name: np.zeros((n_observations, n_datasets), dtype=np.int32)
        for name in metric_names
    }
    baseline_fn = np.zeros(n_datasets, dtype=np.int32)
    eligible_tp = np.zeros(n_datasets, dtype=np.int32)
    eligible_fp = np.zeros(n_datasets, dtype=np.int32)
    dataset_metadata: list[dict] = []
    complementarity: dict[tuple[float, str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    candidate_path = output_root / "exp392_candidate_level_evaluation.csv.gz"
    evidence_columns = [
        f"ev_{source}_alpha_{alpha_slug(alpha)}"
        for alpha in ALPHAS
        for source in SOURCES
    ]
    candidate_fields = [
        "dataset",
        "lineage",
        "index",
        "true_label",
        "baseline_lane",
        "evidence_count",
        *evidence_columns,
        "label_guided_discovery",
        "eligible_for_deployment",
    ]
    with gzip.open(candidate_path, "wt", newline="", encoding="utf-8") as handle:
        candidate_writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        candidate_writer.writeheader()
        for dataset_index, prepared in enumerate(prepared_rows):
            dataset = str(prepared["dataset"])
            n_test = int(prepared["n_test"])
            labels = labels_by_dataset[dataset]
            if len(labels) != n_test:
                raise ValueError(f"TEST label length mismatch: {dataset}")
            truth = {
                index for index, label in enumerate(labels) if str(label) == "1"
            }
            hard = validated_indices(
                prepared["baseline_hard_indices"],
                n_test=n_test,
                label=f"{dataset} Hard",
            )
            standard = validated_indices(
                prepared["standard_review_indices"],
                n_test=n_test,
                label=f"{dataset} Standard review",
            )
            priority = validated_indices(
                prepared["priority_review_indices"],
                n_test=n_test,
                label=f"{dataset} Priority review",
            )
            exact = validated_indices(
                prepared["automatic_normal_exact_indices"],
                n_test=n_test,
                label=f"{dataset} exact-normal",
            )
            review = standard | priority
            excluded = hard | exact
            baseline_fn[dataset_index] = len(truth - hard)
            eligible_truth = truth - excluded
            eligible_normal = set(range(n_test)) - truth - excluded
            eligible_tp[dataset_index] = len(eligible_truth)
            eligible_fp[dataset_index] = len(eligible_normal)
            dataset_metadata.append(
                {
                    "dataset": dataset,
                    "lineage": _lineage(dataset),
                    "n_train_normal": int(prepared["n_train_normal"]),
                    "n_test": n_test,
                    "baseline_category": classify_baseline(hard, labels),
                    "baseline_fn": int(baseline_fn[dataset_index]),
                    "eligible_tp": len(eligible_truth),
                    "eligible_fp": len(eligible_normal),
                }
            )

            source_sets_by_alpha: dict[float, dict[str, set[int]]] = {}
            source_available_by_alpha: dict[float, dict[str, bool]] = {}
            for alpha in ALPHAS:
                alpha_payload = prepared["evidence_by_alpha"][alpha_slug(alpha)]
                source_sets: dict[str, set[int]] = {}
                source_available: dict[str, bool] = {}
                for source in SOURCES:
                    payload = alpha_payload[source]
                    is_available = payload["available"] is True
                    source_available[source] = is_available
                    source_sets[source] = (
                        validated_indices(
                            payload["indices"],
                            n_test=n_test,
                            label=f"{dataset} {source} Alpha {alpha}",
                        )
                        - excluded
                        if is_available
                        else set()
                    )
                    observation = observation_by_source[(alpha, source)]
                    matrix_index = int(observation["matrix_index"])
                    available[matrix_index, dataset_index] = int(is_available)
                    metrics = _candidate_metrics(
                        source_sets[source],
                        truth=truth,
                        hard=hard,
                        review=review,
                    )
                    for metric_name, value in zip(metric_names, metrics):
                        metric_arrays[metric_name][matrix_index, dataset_index] = value
                source_sets_by_alpha[alpha] = source_sets
                source_available_by_alpha[alpha] = source_available

                for observation in combinations_by_alpha[alpha]:
                    matrix_index = int(observation["matrix_index"])
                    sources = tuple(observation["sources"])
                    is_available = all(source_available[source] for source in sources)
                    available[matrix_index, dataset_index] = int(is_available)
                    candidates = (
                        _direct_rule_candidates(
                            str(observation["operator"]),
                            sources,
                            source_sets,
                        )
                        if is_available
                        else set()
                    )
                    metrics = _candidate_metrics(
                        candidates,
                        truth=truth,
                        hard=hard,
                        review=review,
                    )
                    for metric_name, value in zip(metric_names, metrics):
                        metric_arrays[metric_name][matrix_index, dataset_index] = value

                for source_a, source_b in combinations(SOURCES, 2):
                    if not (
                        source_available[source_a] and source_available[source_b]
                    ):
                        continue
                    key = (alpha, source_a, source_b)
                    totals = complementarity[key]
                    values_a = source_sets[source_a]
                    values_b = source_sets[source_b]
                    tp_a = values_a & eligible_truth
                    tp_b = values_b & eligible_truth
                    fp_a = values_a - truth
                    fp_b = values_b - truth
                    totals["datasets_available"] += 1
                    totals["candidate_intersection"] += len(values_a & values_b)
                    totals["candidate_union"] += len(values_a | values_b)
                    totals["tp_intersection"] += len(tp_a & tp_b)
                    totals["tp_union"] += len(tp_a | tp_b)
                    totals["fp_intersection"] += len(fp_a & fp_b)
                    totals["fp_union"] += len(fp_a | fp_b)
                    totals["unique_tp_a"] += len(tp_a - tp_b)
                    totals["unique_tp_b"] += len(tp_b - tp_a)
                    totals["shared_tp"] += len(tp_a & tp_b)
                    totals["unique_fp_a"] += len(fp_a - fp_b)
                    totals["unique_fp_b"] += len(fp_b - fp_a)

            for index in range(n_test):
                evidence_values = {}
                evidence_count = 0
                for alpha in ALPHAS:
                    for source in SOURCES:
                        present = int(index in source_sets_by_alpha[alpha][source])
                        evidence_values[
                            f"ev_{source}_alpha_{alpha_slug(alpha)}"
                        ] = present
                        evidence_count += present
                candidate_writer.writerow(
                    {
                        "dataset": dataset,
                        "lineage": _lineage(dataset),
                        "index": index,
                        "true_label": labels[index],
                        "baseline_lane": _baseline_lane(
                            index,
                            hard=hard,
                            standard=standard,
                            priority=priority,
                            exact=exact,
                        ),
                        "evidence_count": evidence_count,
                        **evidence_values,
                        "label_guided_discovery": True,
                        "eligible_for_deployment": False,
                    }
                )
            if (
                (dataset_index + 1) % 25 == 0
                or dataset_index + 1 == n_datasets
            ):
                write_status(
                    output_root,
                    status="running",
                    phase="evaluate",
                    done=dataset_index + 1,
                    total=n_datasets,
                )
                print(
                    f"Evaluate: [{dataset_index + 1}/{n_datasets}] "
                    f"dataset={dataset} errors=0",
                    flush=True,
                )

    summaries = [
        _summary_row(
            observation,
            available=available,
            baseline_fn=baseline_fn,
            metric_arrays=metric_arrays,
        )
        for observation in catalog
    ]
    source_rows = [
        row for row in summaries if row["observation_type"] == "source"
    ]
    combination_rows = [
        row for row in summaries if row["observation_type"] == "combination"
    ]
    bootstrap_seed = 20260727
    bootstrap_iterations = 2000
    bootstrap = bootstrap_intervals(
        metric_arrays["recovered_fn"],
        metric_arrays["added_fp"],
        seed=bootstrap_seed,
        iterations=bootstrap_iterations,
    )
    concentration = concentration_diagnostics(
        metric_arrays["recovered_fn"],
        lineages=tuple(str(row["lineage"]) for row in dataset_metadata),
    )
    concentration_rows: list[dict] = []
    for index, row in enumerate(summaries):
        for field, values in bootstrap.items():
            row[field] = float(values[index])
        row["bootstrap_seed"] = bootstrap_seed
        row["bootstrap_iterations"] = bootstrap_iterations
        row["tp_contributing_lineages"] = int(
            concentration["tp_contributing_lineages"][index]
        )
        row["top_lineage_tp_share"] = float(
            concentration["top_lineage_tp_share"][index]
        )
        row["concentrated_signal"] = bool(
            concentration["concentrated_signal"][index]
        )
        row["leave_one_lineage_out_min_recovered_fn"] = int(
            concentration["leave_one_lineage_out_min_recovered_fn"][index]
        )
        row["leave_one_lineage_out_max_recovered_fn"] = int(
            concentration["leave_one_lineage_out_max_recovered_fn"][index]
        )
        row["research_label"] = assign_research_label(
            recovered_fn=int(row["recovered_fn"]),
            candidate_precision=float(row["candidate_precision"]),
            tp_contributing_lineages=int(row["tp_contributing_lineages"]),
            top_lineage_tp_share=float(row["top_lineage_tp_share"]),
            precision_ci_low=float(row["precision_ci_low"]),
        )
        row["research_label_selects_policy"] = False
        concentration_rows.append(
            {
                "observation_id": row["observation_id"],
                "observation_type": row["observation_type"],
                "alpha": row["alpha"],
                "sources": row["sources"],
                "recovered_fn": row["recovered_fn"],
                "tp_contributing_lineages": row["tp_contributing_lineages"],
                "top_lineage_tp_share": row["top_lineage_tp_share"],
                "concentrated_signal": row["concentrated_signal"],
                "leave_one_lineage_out_min_recovered_fn": row[
                    "leave_one_lineage_out_min_recovered_fn"
                ],
                "leave_one_lineage_out_max_recovered_fn": row[
                    "leave_one_lineage_out_max_recovered_fn"
                ],
                "research_label": row["research_label"],
                **SAFETY_FLAGS,
            }
        )
    stability_rows = build_stability_rows(
        catalog,
        dataset_metadata,
        available,
        baseline_fn,
        metric_arrays,
    )

    source_p_values: list[float] = []
    association_rows: list[dict] = []
    for row in source_rows:
        matrix_index = next(
            int(item["matrix_index"])
            for item in catalog
            if item["observation_id"] == row["observation_id"]
        )
        support = available[matrix_index].astype(bool)
        candidate_tp = int(metric_arrays["recovered_fn"][matrix_index].sum())
        candidate_fp = int(metric_arrays["added_fp"][matrix_index].sum())
        noncandidate_tp = int(eligible_tp[support].sum()) - candidate_tp
        noncandidate_fp = int(eligible_fp[support].sum()) - candidate_fp
        odds_ratio, p_value = fisher_exact(
            [
                [candidate_tp, candidate_fp],
                [noncandidate_tp, noncandidate_fp],
            ],
            alternative="greater",
        )
        source_p_values.append(float(p_value))
        association_rows.append(
            {
                "observation_id": row["observation_id"],
                "alpha": row["alpha"],
                "source": row["sources"],
                "candidate_tp": candidate_tp,
                "candidate_fp": candidate_fp,
                "noncandidate_tp": noncandidate_tp,
                "noncandidate_fp": noncandidate_fp,
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_value),
                **SAFETY_FLAGS,
            }
        )
    for row, adjusted in zip(
        association_rows,
        benjamini_hochberg(source_p_values),
    ):
        row["bh_adjusted_p_value"] = adjusted
        row["statistical_significance_selects_policy"] = False

    complementarity_rows = []
    for alpha in ALPHAS:
        for source_a, source_b in combinations(SOURCES, 2):
            totals = complementarity[(alpha, source_a, source_b)]
            complementarity_rows.append(
                {
                    "alpha": alpha,
                    "source_a": source_a,
                    "source_b": source_b,
                    "datasets_available": totals["datasets_available"],
                    "candidate_intersection": totals["candidate_intersection"],
                    "candidate_union": totals["candidate_union"],
                    "candidate_jaccard": (
                        totals["candidate_intersection"]
                        / totals["candidate_union"]
                        if totals["candidate_union"]
                        else 0.0
                    ),
                    "tp_intersection": totals["tp_intersection"],
                    "tp_union": totals["tp_union"],
                    "tp_jaccard": (
                        totals["tp_intersection"] / totals["tp_union"]
                        if totals["tp_union"]
                        else 0.0
                    ),
                    "fp_intersection": totals["fp_intersection"],
                    "fp_union": totals["fp_union"],
                    "fp_jaccard": (
                        totals["fp_intersection"] / totals["fp_union"]
                        if totals["fp_union"]
                        else 0.0
                    ),
                    "unique_tp_a": totals["unique_tp_a"],
                    "unique_tp_b": totals["unique_tp_b"],
                    "shared_tp": totals["shared_tp"],
                    "unique_fp_a": totals["unique_fp_a"],
                    "unique_fp_b": totals["unique_fp_b"],
                    **SAFETY_FLAGS,
                }
            )

    source_path = output_root / "exp392_individual_source_summary.csv"
    combination_path = output_root / "exp392_pair_triple_summary.csv"
    complementarity_path = output_root / "exp392_source_complementarity.csv"
    association_path = output_root / "exp392_source_association.csv"
    metadata_path = output_root / "exp392_dataset_metadata.csv"
    stability_path = output_root / "exp392_stability_summary.csv"
    concentration_path = output_root / "exp392_concentration_diagnostics.csv"
    high_fn_path = output_root / "exp392_high_fn_datasets.csv"
    report_path = output_root / "EXP392_RESULTS_KO.md"
    matrix_path = output_root / "exp392_dataset_observation_metrics.npz"
    write_csv(source_path, source_rows)
    write_csv(combination_path, combination_rows)
    write_csv(complementarity_path, complementarity_rows)
    write_csv(association_path, association_rows)
    write_csv(metadata_path, dataset_metadata)
    write_csv(stability_path, stability_rows)
    write_csv(concentration_path, concentration_rows)
    write_csv(
        high_fn_path,
        sorted(
            dataset_metadata,
            key=lambda row: (-int(row["baseline_fn"]), str(row["dataset"])),
        ),
    )
    report_path.write_text(
        render_report(source_rows, combination_rows, dataset_metadata),
        encoding="utf-8",
    )
    np.savez_compressed(
        matrix_path,
        observation_ids=np.array(
            [row["observation_id"] for row in catalog],
            dtype=str,
        ),
        dataset_names=np.array(
            [row["dataset"] for row in dataset_metadata],
            dtype=str,
        ),
        available=available,
        baseline_fn=baseline_fn,
        eligible_tp=eligible_tp,
        eligible_fp=eligible_fp,
        **metric_arrays,
    )
    output_paths = [
        source_path,
        combination_path,
        complementarity_path,
        association_path,
        metadata_path,
        stability_path,
        concentration_path,
        high_fn_path,
        report_path,
        matrix_path,
        candidate_path,
    ]
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "evaluate",
        "datasets": n_datasets,
        "observations": n_observations,
        "individual_sources": len(source_rows),
        "combination_rules": len(combination_rows),
        "stability_rows": len(stability_rows),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_iterations": bootstrap_iterations,
        "research_label_counts": dict(
            Counter(str(row["research_label"]) for row in summaries)
        ),
        "prepare_prediction_sha256": prepare_manifest["prediction_sha256"],
        "outputs": {
            path.name: sha256_file(path)
            for path in output_paths
        },
        "contains_test_labels": True,
        "contains_evaluation_metrics": True,
        "labels_used_after_frozen_prediction_verification": True,
        "automatic_hard_lane_changed": False,
        "review_lanes_changed": False,
        **SAFETY_FLAGS,
    }
    (output_root / "exp392_evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_status(
        output_root,
        status="completed",
        phase="evaluate_completed",
        done=n_datasets,
        total=n_datasets,
    )
    return manifest


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
        evaluate(args.output_dir)


if __name__ == "__main__":
    main()
