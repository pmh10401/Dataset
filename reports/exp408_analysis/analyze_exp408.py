"""Build a read-only Exp408 analysis artifact from frozen publication files."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(
    os.environ.get("TIMESERIES_DATA_ROOT", "/Users/mac/Documents/Dataset")
)
SOURCE_DIR = DATA_ROOT / "exp403_408_feature_ml_validation"
REPORT_DIR = Path(__file__).resolve().parent

VARIANT_LABELS = {
    "one_new_plus_existing_independent": "신규 1개 + 기존 독립",
    "synthetic_only": "Synthetic 단독",
    "tri_view_and_synthetic": "Tri-view ∩ Synthetic",
    "tri_view_only": "Tri-view 단독",
    "two_new_plus_existing_independent": "신규 2개 + 기존 독립",
}


def source(
    source_id: str,
    label: str,
    filename: str,
    definitions: list[str],
    sql: str,
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "query": {
            "language": "sql",
            "engine": "duckdb",
            "sql": sql,
            "description": f"{filename}의 동결 행을 읽어 dataset·variant·alpha 단위로 재집계",
            "tables_used": [filename],
            "filters": [
                "Exp408 frozen publication only",
                "No candidate promotion; retrospective review-only interpretation",
            ],
            "metric_definitions": definitions,
        },
    }


def main() -> None:
    metrics = pd.read_csv(SOURCE_DIR / "exp408_variant_metrics.csv")
    candidates = pd.read_csv(
        SOURCE_DIR / "exp408_variant_candidate_evaluation.csv"
    )
    consensus_summary = pd.read_csv(
        SOURCE_DIR / "exp408_consensus_summary.csv"
    )

    expected_grid = 1117 * 5 * 4
    checks = [
        {
            "check": "variant metric grid",
            "observed": len(metrics),
            "expected": expected_grid,
            "status": "PASS" if len(metrics) == expected_grid else "FAIL",
        },
        {
            "check": "metric grain duplicates",
            "observed": int(
                metrics.duplicated(["dataset", "variant", "alpha"]).sum()
            ),
            "expected": 0,
            "status": "PASS",
        },
        {
            "check": "metric null cells",
            "observed": int(metrics.isna().sum().sum()),
            "expected": 0,
            "status": "PASS",
        },
        {
            "check": "candidate grain duplicates",
            "observed": int(
                candidates.duplicated(
                    ["dataset", "variant", "alpha", "candidate_index"]
                ).sum()
            ),
            "expected": 0,
            "status": "PASS",
        },
        {
            "check": "consensus summary grid",
            "observed": len(consensus_summary),
            "expected": 1117 * 5,
            "status": "PASS",
        },
    ]

    invalid_arithmetic = (
        (metrics["candidate_count"] != metrics["tp"] + metrics["fp"])
        | (metrics["precision"] < 0)
        | (metrics["precision"] > 1)
    ).sum()
    checks.append(
        {
            "check": "candidate arithmetic violations",
            "observed": int(invalid_arithmetic),
            "expected": 0,
            "status": "PASS" if invalid_arithmetic == 0 else "FAIL",
        }
    )

    aggregate = (
        metrics.groupby(["variant", "alpha"], as_index=False)
        .agg(
            candidates=("candidate_count", "sum"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn_recovered=("fn_recovered", "sum"),
            zero_before=("zero_f1_before", "sum"),
            zero_after=("zero_f1_after", "sum"),
            standard_overlap=("standard_overlap", "sum"),
            priority_overlap=("priority_overlap", "sum"),
            outside=("review_outside_existing", "sum"),
            datasets_with_candidates=(
                "candidate_count",
                lambda values: int((values > 0).sum()),
            ),
        )
        .sort_values(["variant", "alpha"])
    )
    aggregate["precision"] = aggregate["tp"] / aggregate["candidates"]
    aggregate["zero_recovered"] = (
        aggregate["zero_before"] - aggregate["zero_after"]
    )
    aggregate["outside_share"] = aggregate["outside"] / aggregate["candidates"]
    aggregate["variant_label"] = aggregate["variant"].map(VARIANT_LABELS)
    aggregate["alpha_label"] = aggregate["alpha"].map(
        lambda value: f"{value:.3f}".rstrip("0")
    )

    candidate_aggregate = (
        candidates.groupby(["variant", "alpha"], as_index=False)
        .agg(candidate_rows=("candidate_index", "size"), candidate_tp=("is_true_anomaly", "sum"))
    )
    candidate_aggregate["candidate_fp"] = (
        candidate_aggregate["candidate_rows"]
        - candidate_aggregate["candidate_tp"]
    )
    reconciled = aggregate.merge(
        candidate_aggregate, on=["variant", "alpha"], how="left"
    )
    reconciliation_error = int(
        (
            (reconciled["candidates"] != reconciled["candidate_rows"])
            | (reconciled["tp"] != reconciled["candidate_tp"])
            | (reconciled["fp"] != reconciled["candidate_fp"])
        ).sum()
    )
    checks.append(
        {
            "check": "candidate-to-metric reconciliation",
            "observed": reconciliation_error,
            "expected": 0,
            "status": "PASS" if reconciliation_error == 0 else "FAIL",
        }
    )

    status_counts: dict[tuple[str, str, str], int] = {}
    outcomes: dict[str, int] = {}
    eligibility: dict[str, int] = {}
    consensus_rows = 0
    bad_source_count = 0
    with (SOURCE_DIR / "exp408_consensus_results.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            row = json.loads(line)
            consensus_rows += 1
            outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
            eligibility[row["eligibility"]] = (
                eligibility.get(row["eligibility"], 0) + 1
            )
            statuses = row["consensus"]["source_statuses"]
            bad_source_count += int(len(statuses) != 13)
            for item in statuses:
                key = (
                    item["source"],
                    item["status"],
                    item.get("reason", ""),
                )
                status_counts[key] = status_counts.get(key, 0) + 1

    reason_counts = {}
    for (_, status, reason), count in status_counts.items():
        if status == "abstained":
            reason_counts[reason] = reason_counts.get(reason, 0) + count
    source_quality = [
        {"metric": "consensus rows", "value": consensus_rows},
        {"metric": "completed rows", "value": outcomes.get("completed", 0)},
        {"metric": "datasets without 13 source statuses", "value": bad_source_count},
        {
            "metric": "available source statuses",
            "value": sum(
                count
                for (_, status, _), count in status_counts.items()
                if status == "available"
            ),
        },
        {
            "metric": "abstained: TRAIN below 50",
            "value": reason_counts.get("n_train_below_50", 0),
        },
        {
            "metric": "abstained: upstream source",
            "value": reason_counts.get("source_abstained", 0),
        },
        {
            "metric": "abstained: TRAIN count mismatch",
            "value": reason_counts.get("source_train_count_mismatch", 0),
        },
    ]

    alpha_001 = aggregate[aggregate["alpha"] == 0.01].copy()
    alpha_001 = alpha_001.sort_values("precision", ascending=False)

    strict = aggregate[
        aggregate["variant"] == "two_new_plus_existing_independent"
    ].copy()
    strict["added_candidates"] = strict["candidates"].diff().fillna(
        strict["candidates"]
    )
    strict["added_tp"] = strict["tp"].diff().fillna(strict["tp"])
    strict["marginal_precision"] = (
        strict["added_tp"] / strict["added_candidates"]
    )

    strict_dataset = metrics[
        (metrics["variant"] == "two_new_plus_existing_independent")
        & (metrics["alpha"] == 0.01)
    ].copy()
    top_strict = strict_dataset.sort_values(
        ["candidate_count", "tp"], ascending=False
    ).head(10)
    top10_share = (
        top_strict["candidate_count"].sum()
        / strict_dataset["candidate_count"].sum()
    )

    headline = [
        {
            "datasets": 1117,
            "strict_candidates": int(strict.loc[strict["alpha"] == 0.01, "candidates"].iloc[0]),
            "strict_precision": float(strict.loc[strict["alpha"] == 0.01, "precision"].iloc[0]),
            "strict_tp": int(strict.loc[strict["alpha"] == 0.01, "tp"].iloc[0]),
            "zero_f1_recovered": int(strict.loc[strict["alpha"] == 0.01, "zero_recovered"].iloc[0]),
            "available_source_rate": sum(
                count
                for (_, status, _), count in status_counts.items()
                if status == "available"
            )
            / sum(status_counts.values()),
        }
    ]

    all_rows = aggregate[
        [
            "variant_label",
            "alpha_label",
            "candidates",
            "tp",
            "fp",
            "precision",
            "zero_recovered",
            "datasets_with_candidates",
            "standard_overlap",
            "priority_overlap",
            "outside",
            "outside_share",
        ]
    ].to_dict("records")
    alpha_rows = alpha_001[
        [
            "variant_label",
            "candidates",
            "tp",
            "fp",
            "precision",
            "zero_recovered",
            "outside_share",
            "datasets_with_candidates",
        ]
    ].to_dict("records")
    marginal_rows = strict[
        [
            "alpha_label",
            "candidates",
            "tp",
            "fp",
            "precision",
            "added_candidates",
            "added_tp",
            "marginal_precision",
            "zero_recovered",
        ]
    ].to_dict("records")
    top_rows = top_strict[
        ["dataset", "train_size", "candidate_count", "tp", "fp", "precision"]
    ].to_dict("records")

    generated_at = datetime.now().astimezone().isoformat()
    metrics_source = source(
        "variant_metrics",
        "Exp408 variant metrics",
        "exp408_variant_metrics.csv",
        [
            "precision = sum(tp) / sum(candidate_count)",
            "zero_recovered = sum(zero_f1_before) - sum(zero_f1_after)",
            "outside_share = sum(review_outside_existing) / sum(candidate_count)",
        ],
        """
SELECT
  variant,
  alpha,
  SUM(candidate_count) AS candidates,
  SUM(tp) AS tp,
  SUM(fp) AS fp,
  SUM(tp) / NULLIF(SUM(candidate_count), 0) AS precision,
  SUM(zero_f1_before) - SUM(zero_f1_after) AS zero_recovered,
  SUM(review_outside_existing) / NULLIF(SUM(candidate_count), 0) AS outside_share
FROM read_csv_auto('exp408_variant_metrics.csv')
GROUP BY variant, alpha
ORDER BY variant, alpha
""".strip(),
    )
    candidates_source = source(
        "candidate_evaluation",
        "Exp408 candidate evaluation",
        "exp408_variant_candidate_evaluation.csv",
        [
            "One row per dataset, variant, alpha, candidate_index",
            "candidate TP = sum(is_true_anomaly)",
        ],
        """
SELECT
  dataset,
  variant,
  alpha,
  COUNT(*) AS candidate_rows,
  SUM(is_true_anomaly) AS tp,
  COUNT(*) - SUM(is_true_anomaly) AS fp
FROM read_csv_auto('exp408_variant_candidate_evaluation.csv')
GROUP BY dataset, variant, alpha
""".strip(),
    )
    consensus_source = source(
        "consensus_results",
        "Exp408 consensus results",
        "exp408_consensus_results.jsonl",
        [
            "One row per dataset",
            "Every completed row must expose exactly 13 source statuses",
        ],
        """
SELECT dataset, outcome, eligibility, train_size, consensus.source_statuses
FROM read_json_auto('exp408_consensus_results.jsonl')
ORDER BY dataset
""".strip(),
    )

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Exp408 동결 결과 분석",
            "description": "동결된 Exp408 publication의 review-only 효율, 범위 및 한계 분석",
            "generatedAt": generated_at,
            "sources": [
                metrics_source,
                candidates_source,
                consensus_source,
            ],
            "cards": [
                {
                    "id": "datasets",
                    "dataset": "headline",
                    "sourceId": "consensus_results",
                    "description": "오류 없이 완료된 Exp408 consensus dataset 수",
                    "metrics": [
                        {"label": "완료 데이터셋", "field": "datasets", "format": "number"}
                    ],
                },
                {
                    "id": "strict_precision",
                    "dataset": "headline",
                    "sourceId": "variant_metrics",
                    "description": "신규 2개와 기존 독립 신호가 합의한 alpha 0.01 review queue 정밀도",
                    "metrics": [
                        {"label": "엄격 합의 정밀도", "field": "strict_precision", "format": "percent"},
                        {"label": "후보", "field": "strict_candidates", "format": "number"},
                        {"label": "TP", "field": "strict_tp", "format": "number"},
                    ],
                },
                {
                    "id": "zero_recovery",
                    "dataset": "headline",
                    "sourceId": "variant_metrics",
                    "description": "alpha 0.01 엄격 합의가 zero-F1 상태에서 벗어나게 한 dataset 수",
                    "metrics": [
                        {"label": "Zero-F1 개선", "field": "zero_f1_recovered", "format": "number"}
                    ],
                },
                {
                    "id": "source_availability",
                    "dataset": "headline",
                    "sourceId": "consensus_results",
                    "description": "전체 14,521 source status 중 available 비율",
                    "metrics": [
                        {"label": "점수원 가용률", "field": "available_source_rate", "format": "percent"}
                    ],
                },
            ],
            "charts": [
                {
                    "id": "precision_at_001",
                    "title": "Variant별 review 정밀도",
                    "subtitle": "alpha 0.01, 후보 수·TP·FP와 함께 비교",
                    "type": "horizontalBar",
                    "intent": "comparison",
                    "dataset": "alpha_001",
                    "sourceId": "variant_metrics",
                    "encodings": {
                        "x": {"field": "variant_label", "type": "nominal", "label": "Variant"},
                        "y": {"field": "precision", "type": "quantitative", "format": "percent", "label": "정밀도"},
                        "tooltip": [
                            {"field": "candidates", "type": "quantitative", "label": "후보"},
                            {"field": "tp", "type": "quantitative", "label": "TP"},
                            {"field": "fp", "type": "quantitative", "label": "FP"},
                            {"field": "outside_share", "type": "quantitative", "format": "percent", "label": "기존 lane 밖 비율"},
                        ],
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                },
                {
                    "id": "strict_marginal",
                    "title": "엄격 합의의 alpha 증가분 효율",
                    "subtitle": "각 alpha 구간에서 새로 추가된 후보 중 TP 비율",
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "strict_marginal",
                    "sourceId": "variant_metrics",
                    "encodings": {
                        "x": {"field": "alpha_label", "type": "ordinal", "label": "Alpha"},
                        "y": {"field": "marginal_precision", "type": "quantitative", "format": "percent", "label": "추가 후보 정밀도"},
                        "tooltip": [
                            {"field": "added_candidates", "type": "quantitative", "label": "추가 후보"},
                            {"field": "added_tp", "type": "quantitative", "label": "추가 TP"},
                            {"field": "zero_recovered", "type": "quantitative", "label": "누적 Zero-F1 개선"},
                        ],
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "all_variants",
                    "title": "전체 variant·alpha 결과",
                    "subtitle": "1,117개 dataset 합계; 같은 후보는 alpha별로 반복 집계",
                    "dataset": "all_results",
                    "sourceId": "variant_metrics",
                    "defaultSort": {"field": "precision", "direction": "desc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "variant_label", "label": "Variant", "type": "text"},
                        {"field": "alpha_label", "label": "Alpha", "type": "text"},
                        {"field": "candidates", "label": "후보", "format": "number"},
                        {"field": "tp", "label": "TP", "format": "number"},
                        {"field": "fp", "label": "FP", "format": "number"},
                        {"field": "precision", "label": "정밀도", "format": "percent"},
                        {"field": "zero_recovered", "label": "Zero-F1 개선", "format": "number"},
                        {"field": "outside_share", "label": "기존 lane 밖", "format": "percent"},
                    ],
                },
                {
                    "id": "top_strict_datasets",
                    "title": "엄격 합의 후보가 많은 dataset",
                    "subtitle": f"alpha 0.01 상위 10개가 전체 후보의 {top10_share:.1%}",
                    "dataset": "top_strict",
                    "sourceId": "variant_metrics",
                    "defaultSort": {"field": "candidate_count", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "dataset", "label": "Dataset", "type": "text"},
                        {"field": "train_size", "label": "TRAIN", "format": "number"},
                        {"field": "candidate_count", "label": "후보", "format": "number"},
                        {"field": "tp", "label": "TP", "format": "number"},
                        {"field": "fp", "label": "FP", "format": "number"},
                        {"field": "precision", "label": "정밀도", "format": "percent"},
                    ],
                },
                {
                    "id": "quality_checks",
                    "title": "분석 데이터 품질 검사",
                    "subtitle": "완전성, 유일성, 산술 및 교차 파일 일치",
                    "dataset": "quality_checks",
                    "sourceId": "candidate_evaluation",
                    "defaultSort": {"field": "check", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "check", "label": "검사", "type": "text"},
                        {"field": "observed", "label": "관측", "format": "number"},
                        {"field": "expected", "label": "기대", "format": "number"},
                        {"field": "status", "label": "결과", "type": "text"},
                    ],
                },
                {
                    "id": "source_quality",
                    "title": "Consensus 점수원 상태",
                    "subtitle": "1,117개 dataset × 13개 source status",
                    "dataset": "source_quality",
                    "sourceId": "consensus_results",
                    "defaultSort": {"field": "value", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "metric", "label": "항목", "type": "text"},
                        {"field": "value", "label": "수량", "format": "number"},
                    ],
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Exp408 동결 결과 분석", "layout": "full"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## 기술 요약\n\n"
                        "**Exp408 결과는 존재하며 새 실행은 필요하지 않다.** 1,117개 dataset의 "
                        "5개 variant와 4개 alpha가 완전하게 평가됐다. 가장 엄격한 합의 "
                        "(`신규 2개 + 기존 독립`)는 alpha 0.01에서 174개 후보 중 77개 TP로 "
                        "44.3%의 review 정밀도를 보였다. 다만 zero-F1 dataset 개선은 356개 중 "
                        "2개뿐이며, 이는 기존 자동 Hard를 바꿀 근거가 아니라 사람이 검토할 "
                        "작은 queue를 만드는 retrospective 근거다.\n\n"
                        "alpha를 높이면 TP는 늘지만 추가 후보의 효율은 빠르게 낮아진다. 이 결과로 "
                        "alpha나 variant를 운영 선택하면 TEST 재선택이 되므로, 수치는 비교 설명에만 "
                        "사용하고 자동 경보 승격에는 사용하지 않는다."
                    ),
                    "layout": "full",
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["datasets", "strict_precision", "zero_recovery", "source_availability"], "layout": "full"},
                {
                    "id": "finding_precision",
                    "type": "markdown",
                    "body": (
                        "## 엄격한 독립 합의가 review 효율을 가장 높였다\n\n"
                        "alpha 0.01에서 `신규 2개 + 기존 독립`의 정밀도는 44.3%로 가장 높다. "
                        "`신규 1개 + 기존 독립`은 33.9%지만 190개 TP를 회수해 범위가 더 넓다. "
                        "따라서 전자는 작은 고밀도 검토함, 후자는 더 큰 탐색 검토함에 가깝다. "
                        "이 차이는 자동 경보 품질이 아니라 사람 검토 workload의 차이다."
                    ),
                    "sourceId": "variant_metrics",
                    "layout": "full",
                },
                {"id": "precision_chart", "type": "chart", "chartId": "precision_at_001", "layout": "full"},
                {
                    "id": "finding_alpha",
                    "type": "markdown",
                    "body": (
                        "## alpha 확대의 추가 효율은 0.02 이후 급격히 낮아졌다\n\n"
                        "엄격 합의에서 alpha 0.005→0.01 구간은 112개 후보를 더해 48개 TP를 "
                        "추가했다(추가 정밀도 42.9%). 0.01→0.02는 26.8%, "
                        "0.02→0.05는 10.1%로 하락한다. 동시에 zero-F1 개선은 alpha 0.01에서도 "
                        "2개에 불과하다. 높은 alpha는 더 많은 이상을 찾지만 false review 부담이 "
                        "훨씬 빠르게 증가한다."
                    ),
                    "sourceId": "variant_metrics",
                    "layout": "full",
                },
                {"id": "marginal_chart", "type": "chart", "chartId": "strict_marginal", "layout": "full"},
                {
                    "id": "scope_definition",
                    "type": "markdown",
                    "body": (
                        "## 범위와 지표 정의\n\n"
                        "- 분석 단위: dataset × variant × alpha\n"
                        "- 후보: 동결된 consensus가 review-only로 제안한 TEST index\n"
                        "- 정밀도: 전체 후보 중 TEST label로 확인된 실제 이상 비율\n"
                        "- FN 회수: 기존 Hard가 놓친 실제 이상 중 새 review 후보가 포착한 수\n"
                        "- Zero-F1 개선: 기존 결과가 F1=0인 dataset 중 새 후보로 0을 벗어난 수\n"
                        "- 기존 lane 밖: Standard 또는 Priority와 겹치지 않는 추가 review 후보\n\n"
                        "`exp408_consensus_summary.csv`의 candidate_count는 alpha별 후보 수의 합이다. "
                        "alpha 열이 없으므로 unique index 수로 해석하면 안 된다."
                    ),
                    "layout": "full",
                },
                {"id": "all_results_table", "type": "table", "tableId": "all_variants", "layout": "full"},
                {
                    "id": "concentration",
                    "type": "markdown",
                    "body": (
                        "## 엄격 합의의 근거는 일부 dataset에 집중되어 있다\n\n"
                        f"alpha 0.01 엄격 합의 후보 174개 중 상위 10개 dataset이 "
                        f"{top10_share:.1%}를 차지한다. 62개 dataset에만 후보가 있었고, "
                        "36개 dataset에서 TP가 확인된 반면 26개는 FP만 있었다. 평균 정밀도만으로 "
                        "모든 계열에 일반화됐다고 볼 수 없으며 lineage별 prospective 검증이 필요하다."
                    ),
                    "sourceId": "variant_metrics",
                    "layout": "full",
                },
                {"id": "top_dataset_table", "type": "table", "tableId": "top_strict_datasets", "layout": "full"},
                {
                    "id": "quality",
                    "type": "markdown",
                    "body": (
                        "## publication 데이터는 내부적으로 완전하고 일관된다\n\n"
                        "22,340개 metric 행은 기대 그리드를 모두 채우며 중복과 결측이 없다. "
                        "41,729개 candidate 행도 중복 없이 TP·FP 집계와 정확히 일치한다. "
                        "모든 1,117개 consensus 행이 completed이고 dataset마다 13개 점수원 상태를 "
                        "기록한다. 따라서 동결 결과의 기술적 재집계에는 사용할 수 있다."
                    ),
                    "layout": "full",
                },
                {"id": "quality_table", "type": "table", "tableId": "quality_checks", "layout": "full"},
                {"id": "source_table", "type": "table", "tableId": "source_quality", "layout": "full"},
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## 방법론\n\n"
                        "동결 CSV를 dataset·variant·alpha로 합산하고 candidate-level label 합계와 "
                        "교차 검증했다. consensus JSONL은 outcome, eligibility, 13개 점수원 상태의 "
                        "완전성을 검사했다. 후보 집합이 alpha 증가에 따라 단조 포함되는지도 확인했으며 "
                        "위반은 0이었다. 기존 publication의 paired-bootstrap 구간은 새로 선택하거나 "
                        "튜닝하지 않고 해석 보조 근거로만 사용했다."
                    ),
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 한계와 불확실성\n\n"
                        "- 동일 TEST benchmark를 사용한 retrospective 평가이므로 prospective 성능을 보장하지 않는다.\n"
                        "- 14,521개 점수원 상태 중 4,451개가 abstain했다. 주요 원인은 TRAIN<50 "
                        "(2,464), upstream abstention(1,426), TRAIN count mismatch(561)다.\n"
                        "- checkpoint 생성 코드 provenance는 현재 Git 이력으로 재현되지 않아 직접 resume할 수 없다.\n"
                        "- alpha·variant 간 성능 비교는 TEST label을 본 사후 결과다. 운영 설정 선택에 사용하면 안 된다.\n"
                        "- Exp408 후보는 Shadow/review-only이며 Priority, Standard, Hard로 자동 승격되지 않는다."
                    ),
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 권장 다음 단계\n\n"
                        "1. Exp408 결과는 현 상태로 동결하고 자동 Hard 정책을 유지한다.\n"
                        "2. 새 실험 번호로 review queue 목적, 고정 variant, alpha, review budget을 사전등록한다.\n"
                        "3. TEST label을 보지 않은 신규 설비 또는 신규 시계열에서 precision과 false-review workload를 검증한다.\n"
                        "4. checkpoint에는 합성 해시와 함께 코드·데이터·feature source·upstream 해시를 개별 저장한다.\n"
                        "5. TRAIN<50 및 source count mismatch 계층은 별도 coverage 실험으로 분리한다."
                    ),
                    "layout": "full",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 추가로 답해야 할 질문\n\n"
                        "- 실제 운영자가 한 회차에 검토할 수 있는 후보 budget은 얼마인가?\n"
                        "- 높은 정밀도가 특정 lineage와 dataset에 치우친 현상이 신규 데이터에서도 유지되는가?\n"
                        "- zero-F1 dataset을 목표로 하는 별도 TRAIN-only 후보 생성 방식이 필요한가?\n"
                        "- abstain 계층을 무리하게 채우지 않고 coverage를 높일 수 있는가?"
                    ),
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "alpha_001": alpha_rows,
                "strict_marginal": marginal_rows,
                "all_results": all_rows,
                "top_strict": top_rows,
                "quality_checks": checks,
                "source_quality": source_quality,
            },
        },
        "sources": [
            metrics_source,
            candidates_source,
            consensus_source,
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "analysis_summary.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "quality_checks": checks,
                "eligibility": eligibility,
                "source_abstention_reasons": reason_counts,
                "top10_strict_candidate_share": top10_share,
                "strict_alpha_0_01": headline[0],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
