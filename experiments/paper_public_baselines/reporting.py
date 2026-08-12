"""Canonical paper tables and Korean Markdown from immutable evaluation leaves."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .artifacts import (
    canonical_json_bytes,
    read_strict_json,
    sha256_file,
    write_create_only_bytes,
)
from .evaluation import EvaluationBundle


METHOD_NAMES = {
    "knn_distance": "kNN distance",
    "lof_novelty": "LOF",
    "ocsvm_rbf": "RBF One-Class SVM",
    "isolation_forest": "Isolation Forest",
    "deep_svdd_1d": "Deep SVDD (1D adapter)",
    "drocc_1d": "DROCC (1D adapter)",
    "proposed_equipment_anomaly_detection": "제안기술",
}
SUMMARY_FIELDS = (
    "method",
    "is_external_baseline",
    "protocol",
    "alpha",
    "support",
    "dataset_count",
    "completed_datasets",
    "abstained_datasets",
    "coverage",
    "alerts",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "average_precision",
    "auroc",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_false_positive_rate",
    "macro_average_precision",
    "macro_auroc",
    "continuous_dataset_count",
)
DATASET_FIELDS = (
    "method",
    "is_external_baseline",
    "protocol",
    "alpha",
    "support",
    "status",
    "dataset",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "alert_count",
    "average_precision",
    "auroc",
)


def render_report(bundle: EvaluationBundle) -> str:
    """Render a descriptive report without field-validation or promotion claims."""
    dataset_count = int(bundle.metadata["dataset_count"])
    common_count = int(bundle.metadata["common_support_count"])
    deep_complete = bool(bundle.metadata.get("planned_deep_methods_complete", False))
    lines = [
        "# 공개 이상 탐지 기법 비교 결과",
        "",
        f"- 평가 데이터셋: {dataset_count:,}개",
        f"- 공통 지원 데이터셋: {common_count:,}개",
        "- 평가 상태: retrospective comparison, no automatic promotion",
        "- 해석 범위: 저장된 benchmark의 후향 비교이며 설비 현장 검증이 아님",
        f"- 심층 비교군 전체 coverage: {'완료' if deep_complete else '미완료(Deep SVDD/DROCC 공식 full run 제외)'}",
        "",
    ]
    if dataset_count != 1117:
        lines.extend(
            [
                "> 이 문서는 제한된 smoke 모집단 결과다. 논문 성능 수치로 사용하지 않는다.",
                "",
            ]
        )
    proposal_rows = [
        row
        for row in bundle.summaries
        if row["method"] == "proposed_equipment_anomaly_detection"
        and row["support"] == "full_population"
    ]
    lines.extend(["## 1. 제안기술 기준", ""])
    lines.extend(_summary_table(proposal_rows))
    lines.extend(["", "## 2. 공개 기법 비교", ""])
    public_rows = [
        row
        for row in bundle.summaries
        if row["is_external_baseline"] is True
        and row["support"] == "full_population"
    ]
    lines.extend(_summary_table(public_rows))
    lines.extend(
        [
            "",
            "연속 score의 macro AP/AUROC는 데이터셋 내부 원시 score 순위, "
            "micro AP/AUROC는 TRAIN-OOF upper-tail p-value의 1−p로 계산했다. "
            "matched-alert-budget는 제안기술의 데이터셋별 경보 수만 맞추며 좌표를 복사하지 않는다.",
            "",
            "## 3. 제안기술 내부 ablation",
            "",
            "| 논문 표기 | 내부 provenance | TP | FP | FN | TN | F1 | Macro F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bundle.author_ablation:
        lines.append(
            "| {paper_name} | {internal_provenance} | {tp} | {fp} | {fn} | {tn} | {f1:.4f} | {macro_f1:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "내부 provenance는 제안기술의 구성 추적용이며 공개 baseline이 아니다.",
            "",
            "## 4. Paired bootstrap",
            "",
            "점 추정치는 공개 baseline의 데이터셋 macro F1에서 제안기술 F1을 뺀 값이다.",
            "",
            "| 공개 방법 | Protocol | Alpha | Support | 차이 | 95% CI | Draws |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    primary_bootstrap = [
        row
        for row in bundle.bootstrap
        if row["protocol"] == "matched_alert_budget"
        and row["support"] == "full_population"
    ]
    for row in primary_bootstrap:
        lines.append(
            f"| {METHOD_NAMES.get(str(row['method']), row['method'])} | matched budget | — | "
            f"full | {_number(row['point_estimate'])} | "
            f"[{_number(row['lower'])}, {_number(row['upper'])}] | {row['draw_count']} |"
        )
    lines.extend(
        [
            "",
            "## 5. 해석 보류 항목",
            "",
            "- 수상 가능성 최종 판단: **[평가 완료 후 기입]**",
            "- 설비 현장 일반화와 운영 효과는 별도의 prospective/field validation 대상이다.",
            "- 본 결과만으로 자동 champion 승격이나 운영 적용을 결정하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def public_comparison_section(report: str) -> str:
    """Return only the public-comparison section for boundary tests/review."""
    start = report.index("## 2. 공개 기법 비교")
    end = report.index("## 3. 제안기술 내부 ablation")
    return report[start:end]


def load_evaluation_bundle(run_root: Path) -> EvaluationBundle:
    """Verify evaluation artifact inventory and reconstruct an immutable bundle."""
    root = Path(run_root) / "evaluation_public_baselines"
    expected_names = {
        "evaluation_summary.json",
        "evaluation_dataset_metrics.jsonl",
        "evaluation_bootstrap.json",
        "artifact_sha256.json",
    }
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("evaluation artifact inventory is not exact")
    artifact_map = read_strict_json(root / "artifact_sha256.json")
    expected_leaves = expected_names - {"artifact_sha256.json"}
    if set(artifact_map) != expected_leaves:
        raise ValueError("evaluation artifact map is invalid")
    for name in expected_leaves:
        if artifact_map[name] != sha256_file(root / name):
            raise ValueError("evaluation artifact SHA256 mismatch")
    summary = read_strict_json(root / "evaluation_summary.json")
    bootstrap_payload = read_strict_json(root / "evaluation_bootstrap.json")
    if set(summary) != {"metadata", "summaries", "author_ablation"}:
        raise ValueError("evaluation summary schema is invalid")
    if set(bootstrap_payload) != {"comparisons"}:
        raise ValueError("evaluation bootstrap schema is invalid")
    dataset_rows = _strict_jsonl(root / "evaluation_dataset_metrics.jsonl")
    return EvaluationBundle(
        metadata=MappingProxyType(dict(summary["metadata"])),
        summaries=tuple(MappingProxyType(dict(row)) for row in summary["summaries"]),
        dataset_metrics=tuple(MappingProxyType(dict(row)) for row in dataset_rows),
        bootstrap=tuple(
            MappingProxyType(dict(row)) for row in bootstrap_payload["comparisons"]
        ),
        author_ablation=tuple(
            MappingProxyType(dict(row)) for row in summary["author_ablation"]
        ),
    )


def write_report_artifacts(
    run_root: Path,
    bundle: EvaluationBundle,
    *,
    verify_only: bool = False,
) -> str:
    """Create or byte-verify CSV, JSON, and Markdown paper artifacts."""
    destination = Path(run_root) / "report_public_baselines"
    leaves = {
        "public_baseline_summary.csv": _csv_bytes(bundle.summaries, SUMMARY_FIELDS),
        "public_baseline_dataset_metrics.csv": _csv_bytes(
            bundle.dataset_metrics, DATASET_FIELDS
        ),
        "public_baseline_bootstrap.json": canonical_json_bytes(
            {"comparisons": [dict(row) for row in bundle.bootstrap]}
        ),
        "PAPER_PUBLIC_BASELINE_RESULT_KO.md": render_report(bundle).encode("utf-8"),
    }
    artifact_map = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in leaves.items()
    }
    map_bytes = canonical_json_bytes(artifact_map)
    if verify_only:
        if not destination.is_dir():
            raise ValueError("report artifact root is absent")
        expected = {**leaves, "artifact_sha256.json": map_bytes}
        actual_names = {path.name for path in destination.iterdir()}
        if actual_names != set(expected) or any(
            destination.joinpath(name).read_bytes() != payload
            for name, payload in expected.items()
        ):
            raise ValueError("report artifact bytes differ from deterministic replay")
    else:
        for name, payload in leaves.items():
            write_create_only_bytes(destination / name, payload)
        write_create_only_bytes(destination / "artifact_sha256.json", map_bytes)
    return hashlib.sha256(map_bytes).hexdigest()


def _summary_table(rows: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| 방법 | Protocol | Alpha | Coverage | Alerts | P | R | F1 | Macro F1 | AP | AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        alpha = "—" if row["alpha"] is None else str(row["alpha"])
        lines.append(
            f"| {METHOD_NAMES.get(str(row['method']), row['method'])} | {row['protocol']} | {alpha} | "
            f"{_number(row['coverage'])} | {row['alerts']} | {_number(row['precision'])} | "
            f"{_number(row['recall'])} | {_number(row['f1'])} | {_number(row['macro_f1'])} | "
            f"{_number(row['average_precision'])} | {_number(row['auroc'])} |"
        )
    return lines


def _csv_bytes(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(fields), extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise ValueError("report CSV row fields are not exact")
        writer.writerow(
            {
                field: ""
                if row[field] is None
                else str(row[field]).lower()
                if isinstance(row[field], bool)
                else row[field]
                for field in fields
            }
        )
    return stream.getvalue().encode("utf-8")


def _strict_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("evaluation JSONL is incomplete")
    rows: list[dict[str, object]] = []
    for line in payload.splitlines():
        try:
            value = json.loads(
                line.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("evaluation JSONL is invalid") from error
        if not isinstance(value, dict):
            raise ValueError("evaluation JSONL row is not an object")
        rows.append(value)
    return tuple(rows)


def _number(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}"


__all__ = [
    "load_evaluation_bundle",
    "public_comparison_section",
    "render_report",
    "write_report_artifacts",
]
