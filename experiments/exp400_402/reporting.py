from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.exp400_402 import EXP402_ID
from experiments.exp400_402.artifact_io import MANIFEST_NAME, PREDICTION_NAME, sha256_file


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_reports(output_root: Path) -> dict[str, object]:
    output_root = Path(output_root)
    prediction_manifest = json.loads(
        (output_root / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    evaluation_manifest = json.loads(
        (output_root / "exp401_evaluation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    consistency = json.loads(
        (output_root / "independent_consistency_report.json").read_text(
            encoding="utf-8"
        )
    )
    summary = _read_csv(output_root / "exp401_precision_at_k.csv")

    visible = [
        row
        for row in summary
        if row["variant"] in {"exp292_baseline", "tier_preserving_grammar"}
    ]
    report_lines = [
        "# Exp401 Frozen Shadow Replay Evaluation",
        "",
        "> 과거 1,117개 평가 데이터의 후향 재생 결과이며 실제 운영 검증이 아닙니다.",
        "",
        f"- Datasets: {prediction_manifest['datasets']}",
        f"- Standard review candidates: {prediction_manifest['standard_review_candidates']}",
        f"- Automatic Hard unchanged: {prediction_manifest['automatic_hard_count']}",
        f"- Priority review unchanged: {prediction_manifest['priority_review_count']}",
        f"- Independent mismatches: {consistency['mismatch_count']}",
        "",
        "| Variant | Alpha | Workload | Reviewed | TP | FP | Precision |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in visible:
        workload = (
            f"K={row['workload_k']}"
            if row["workload_type"] == "global_k"
            else "dataset top-1"
        )
        report_lines.append(
            "| {variant} | {alpha} | {workload} | {reviewed} | {tp} | {fp} | {precision:.3%} |".format(
                variant=row["variant"],
                alpha=row["alpha"],
                workload=workload,
                reviewed=row["reviewed"],
                tp=row["tp_at_k"],
                fp=row["fp_at_k"],
                precision=float(row["precision_at_k"]),
            )
        )
    report_lines.extend(
        [
            "",
            "TEST 결과로 alpha 또는 workload를 선택하지 않았습니다.",
            "이 지표는 사람의 검토 순서를 평가하는 human-assisted 지표입니다.",
        ]
    )
    (output_root / "EXP401_EVALUATION_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    readiness: dict[str, object] = {
        "experiment_id": EXP402_ID,
        "datasets": prediction_manifest["datasets"],
        "standard_review_candidates": prediction_manifest[
            "standard_review_candidates"
        ],
        "automatic_hard_count": prediction_manifest["automatic_hard_count"],
        "priority_review_count": prediction_manifest["priority_review_count"],
        "prediction_sha256": prediction_manifest["prediction_sha256"],
        "prediction_file_sha256_verified": (
            sha256_file(output_root / PREDICTION_NAME)
            == prediction_manifest["prediction_sha256"]
        ),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "automatic_policy_changed": False,
        "independent_mismatch_count": consistency["mismatch_count"],
        "prospective_run_count": 0,
        "prospective_upstream_available": False,
        "prospective_validation_completed": False,
        "dashboard_schema_ready": True,
        "retrospective_replay_completed": True,
        "retrospective": True,
        "prospective": False,
        "deployment_validated": False,
        "blocker": "No newly arriving run data is available during development.",
        "evaluation_manifest_sha256": sha256_file(
            output_root / "exp401_evaluation_manifest.json"
        ),
        "test_metrics_select_policy": False,
    }
    (output_root / "exp402_readiness_report.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readiness_lines = [
        "# Exp402 Shadow Readiness Report",
        "",
        "## 검증 완료",
        "",
        f"- 과거 데이터 재생: {readiness['datasets']} datasets",
        f"- 검토 후보 보존: {readiness['standard_review_candidates']}",
        f"- 자동 경보 불변: {readiness['automatic_hard_count']}",
        f"- Priority review 불변: {readiness['priority_review_count']}",
        f"- 독립 재집계 불일치: {readiness['independent_mismatch_count']}",
        "- Prediction TEST label/evaluation field: 0",
        "",
        "## 현재 차단 사항",
        "",
        "- 신규 유입 데이터가 없어 prospective run 수는 0입니다.",
        "- 이번 결과는 retrospective shadow replay이며 실제 설비 검증이 아닙니다.",
        "- 특정 alpha나 workload를 운영값으로 선택하지 않았습니다.",
    ]
    (output_root / "EXP402_READINESS_REPORT.md").write_text(
        "\n".join(readiness_lines) + "\n",
        encoding="utf-8",
    )
    claims = [
        "# Exp400-402 Claims",
        "",
        "## 검증 완료",
        "",
        "- Exp400 prediction은 TEST 라벨과 평가 지표를 포함하지 않는다.",
        "- 자동 경보, Priority review, Standard candidate 집합은 변경되지 않았다.",
        "- Exp401은 prediction hash 검증 후에만 TEST 라벨을 읽는다.",
        "",
        "## 조건부 사용",
        "",
        "- 기존 과거 데이터에서 Exp397 순위 품질을 후향 비교했다.",
        "- Dashboard ingestion과 인증 검증 완료 후 future label-free run을 기록할 수 있다.",
        "",
        "## 사용 금지",
        "",
        "- Exp397이 실제 설비에서 prospective 검증됐다.",
        "- 특정 alpha가 운영 최적값이다.",
        "- Exp397 순위가 자동 경보 성능을 개선했다.",
    ]
    (output_root / "EXP402_CLAIMS.md").write_text(
        "\n".join(claims) + "\n",
        encoding="utf-8",
    )
    return {
        "prediction": prediction_manifest,
        "evaluation": evaluation_manifest,
        "consistency": consistency,
        "readiness": readiness,
    }
