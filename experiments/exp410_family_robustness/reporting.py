"""Deterministic CSV, JSON, and Korean report rendering for Exp410."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict
from io import StringIO

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .config import (
    FROZEN_CONFIG,
    Exp410FrozenConfig,
    require_frozen_research_parameters,
)
from .contracts import AuditResult


CONFIG_FILE = "exp410_frozen_config.json"
DATASET_FILE = "exp410_dataset_metrics.csv"
FAMILY_FILE = "exp410_family_metrics.csv"
WEIGHT_FILE = "exp410_weight_sensitivity.csv"
INFLUENCE_FILE = "exp410_family_influence.csv"
STRATIFIED_FILE = "exp410_stratified_metrics.csv"
SUMMARY_FILE = "exp410_summary.json"
REPORT_FILE = "EXP410_RESULT_REPORT_KO.md"
MANIFEST_FILE = "MANIFEST.json"

_CAP_ORDER = {"1": 0, "3": 1, "5": 2, "10": 3, "unlimited": 4}
_DIMENSION_ORDER = {
    "family_size": 0,
    "normal_train_count": 1,
    "exact_normal_overlap": 2,
    "test_anomaly_count": 3,
}
_BUCKET_ORDER = {
    "family_size": {"1": 0, "2-5": 1, "6-20": 2, "21+": 3},
    "normal_train_count": {"<20": 0, "20-49": 1, "50+": 2},
    "exact_normal_overlap": {"<50%": 0, "50-<90%": 1, ">=90%": 2},
    "test_anomaly_count": {"1": 0, "2-10": 1, "11+": 2},
}


def render_payloads(
    result: AuditResult,
    provenance: Mapping[str, object],
    config: Exp410FrozenConfig = FROZEN_CONFIG,
) -> Mapping[str, bytes]:
    """Render all eight non-manifest artifacts in canonical order."""
    require_frozen_research_parameters(config)
    _validate_summary(result.summary, config)
    _validate_provenance(provenance)

    dataset_rows = [
        asdict(row)
        for row in sorted(result.dataset_metrics, key=lambda row: row.dataset)
    ]
    family_rows = [
        asdict(row)
        for row in sorted(result.family_metrics, key=lambda row: row.family)
    ]
    weight_rows = sorted(
        (dict(row) for row in result.weight_sensitivity),
        key=lambda row: _CAP_ORDER[str(row["cap"])],
    )
    influence_rows = sorted(
        (dict(row) for row in result.family_influence),
        key=lambda row: (
            -abs(float(row["dataset_macro_f1_delta"])),
            str(row["family"]),
        ),
    )
    stratified_rows = sorted(
        (dict(row) for row in result.stratified_metrics),
        key=lambda row: (
            _DIMENSION_ORDER[str(row["dimension"])],
            _BUCKET_ORDER[str(row["dimension"])][str(row["bucket"])],
        ),
    )
    summary = {**dict(result.summary), "provenance": _plain(provenance)}

    payloads = {
        CONFIG_FILE: canonical_json_bytes(config.to_payload()) + b"\n",
        DATASET_FILE: _csv_bytes(
            dataset_rows,
            (
                "dataset",
                "family",
                "hard_count",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
            ),
        ),
        FAMILY_FILE: _csv_bytes(
            family_rows,
            (
                "family",
                "dataset_count",
                "mean_dataset_f1",
                "mean_dataset_fp",
                "pooled_tp",
                "pooled_fp",
                "pooled_fn",
                "pooled_precision",
                "pooled_recall",
                "pooled_f1",
            ),
        ),
        WEIGHT_FILE: _csv_bytes(
            weight_rows,
            ("cap", "mean_f1", "total_weight"),
        ),
        INFLUENCE_FILE: _csv_bytes(
            influence_rows,
            (
                "family",
                "dataset_count",
                "dataset_share",
                "fp_share",
                "removed_alert_count",
                "removed_tp",
                "removed_fp",
                "removed_fn",
                "dataset_macro_f1_without_family",
                "dataset_macro_f1_delta",
                "pooled_precision_without_family",
                "pooled_precision_delta",
            ),
        ),
        STRATIFIED_FILE: _csv_bytes(
            stratified_rows,
            (
                "dimension",
                "bucket",
                "dataset_count",
                "family_count",
                "family_balanced_mean_f1",
            ),
        ),
        SUMMARY_FILE: canonical_json_bytes(summary) + b"\n",
        REPORT_FILE: _korean_report(
            summary,
            weight_rows,
            influence_rows,
            stratified_rows,
        ).encode("utf-8"),
    }
    if set(payloads) != {
        CONFIG_FILE,
        DATASET_FILE,
        FAMILY_FILE,
        WEIGHT_FILE,
        INFLUENCE_FILE,
        STRATIFIED_FILE,
        SUMMARY_FILE,
        REPORT_FILE,
    }:
        raise ValueError("Exp410 rendered artifact coverage mismatch")
    return payloads


def _csv_bytes(
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                name: "" if row.get(name) is None else row.get(name)
                for name in fieldnames
            }
        )
    return buffer.getvalue().encode("utf-8")


def _korean_report(
    summary: Mapping[str, object],
    weights: Sequence[Mapping[str, object]],
    influence: Sequence[Mapping[str, object]],
    strata: Sequence[Mapping[str, object]],
) -> str:
    bootstrap = _mapping(summary["bootstrap"], "Exp410 bootstrap summary")
    interval = _mapping(
        bootstrap["delta_f1_interval_95"],
        "Exp410 bootstrap interval",
    )
    lines = [
        "# Exp410 동결 Hard family 견고성 감사 결과",
        "",
        "## 핵심 요약",
        "",
        f"- 데이터셋 수 / family 수: `{summary['dataset_count']}` / `{summary['family_count']}`",
        f"- 동결 Hard 수: `{summary['hard_count']}`",
        f"- TP / FP / FN: `{summary['tp']}` / `{summary['fp']}` / `{summary['fn']}`",
        f"- dataset-macro F1: `{_number(summary['dataset_macro_f1'])}`",
        f"- family-macro F1: `{_number(summary['family_macro_f1'])}`",
        f"- paired delta F1: `{_number(summary['delta_f1'])}`",
        f"- 기술적 분류: `{bootstrap['classification']}`",
        "",
        "dataset-macro는 파생 데이터셋마다 한 표를 주고, family-macro는",
        "먼저 family 안을 평균한 뒤 family마다 한 표를 준다.",
        "",
        "## 사전등록 family 가중치 cap",
        "",
        "| cap | mean F1 | total weight |",
        "|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['cap']} | {_number(row['mean_f1'])} | {row['total_weight']} |"
        for row in weights
    )
    lines.extend(
        (
            "",
            "## Family-cluster bootstrap",
            "",
            f"- 단위 / seed / 반복: `{bootstrap['unit']}` / `{bootstrap['seed']}` / `{bootstrap['iterations']}`",
            f"- delta F1 95% 구간: `[{_number(interval['low'])}, {_number(interval['high'])}]`",
            f"- robust 하한 margin: `{bootstrap['robustness_margin']}`",
            f"- severe 점추정 margin: `{bootstrap['severe_margin']}`",
            "",
            "## Family 영향도",
            "",
            "아래 delta는 해당 family 전체를 제외했을 때 dataset-macro F1의 변화다.",
            "",
            "| family | datasets | F1 delta | precision delta | removed FP |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for row in influence[:10]:
        lines.append(
            f"| {row['family']} | {row['dataset_count']} | "
            f"{_number(row['dataset_macro_f1_delta'])} | "
            f"{_number(row['pooled_precision_delta'])} | {row['removed_fp']} |"
        )
    lines.extend(
        (
            "",
            "## 사전등록 층화",
            "",
            "| 차원 | bucket | datasets | families | family-balanced F1 |",
            "|---|---|---:|---:|---:|",
        )
    )
    for row in strata:
        lines.append(
            f"| {row['dimension']} | {row['bucket']} | {row['dataset_count']} | "
            f"{row['family_count']} | {_number(row['family_balanced_mean_f1'])} |"
        )
    difficulty = _mapping(
        summary["difficulty_stratification"],
        "Exp410 difficulty status",
    )
    lines.extend(
        (
            "",
            "## 난이도 층화",
            "",
            f"- 상태: `{difficulty['status']}`",
            f"- 이유: `{difficulty['reason']}`",
            "",
            "원본 per-dataset 난이도 artifact가 없어 aggregate count로 소속을",
            "추정하지 않았다.",
            "",
            "## 주장 경계",
            "",
            f"- 운영 정책 변경: `{str(summary['operational_policy_changed']).lower()}`",
            f"- 배포 적격: `{str(summary['deployment_eligible']).lower()}`",
            f"- unseen family 검증: `{str(summary['unseen_family_validated']).lower()}`",
            f"- 전향 검증: `{str(summary['prospective_validated']).lower()}`",
            "- 이 결과는 동결 Hard의 읽기 전용 회고 감사다.",
            "- 새 반도체 설비 데이터, 배포 안전성, 자동 경보 향상을 입증하지 않는다.",
            "- Hard, Standard, Priority, threshold, source, alpha, budget을 변경하지 않는다.",
            "",
        )
    )
    return "\n".join(lines)


def _validate_summary(
    summary: Mapping[str, object],
    config: Exp410FrozenConfig,
) -> None:
    expected = {
        "evaluation_mode": "retrospective_family_robustness_audit",
        "operational_policy_changed": False,
        "deployment_eligible": False,
        "unseen_family_validated": False,
        "prospective_validated": False,
        "difficulty_stratification": {
            "status": "not_evaluated",
            "reason": "source_artifact_unavailable",
        },
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(f"Exp410 summary claim boundary mismatch: {name}")
    if (
        config.operational_policy_changed
        or config.deployment_eligible
        or config.unseen_family_validated
        or config.prospective_validated
    ):
        raise ValueError("Exp410 configuration claim flags changed")


def _validate_provenance(provenance: Mapping[str, object]) -> None:
    if not isinstance(provenance, Mapping) or not {
        "inputs",
        "code",
        "runtime",
    } <= set(provenance):
        raise ValueError("Exp410 provenance must include inputs, code, and runtime")
    for name in ("inputs", "code", "runtime"):
        if not isinstance(provenance[name], Mapping):
            raise ValueError(f"Exp410 provenance {name} must be an object")


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: object) -> str:
    if value is None:
        return "not_evaluated"
    return f"{float(value):.12g}"
