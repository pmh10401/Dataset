from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    rows = _read_csv(output_root / "exp398_precision_at_k.csv")
    _write_csv(output_root / "exp399_workload_summary.csv", rows)

    global_rows = [row for row in rows if row["workload_type"] == "global_k"]
    report_lines = [
        "# Exp398 fixed review workload evaluation",
        "",
        "Exp397 prediction SHA256를 검증한 뒤 TEST 라벨을 사후 평가에만 사용했다.",
        "어떤 variant, alpha, workload도 TEST 결과로 선택하지 않는다.",
        "",
        "| variant | alpha | K | TP | FP | precision | lift vs Exp292 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        report_lines.append(
            f"| {row['variant']} | {float(row['alpha']):.3f} | "
            f"{int(row['workload_k'])} | {int(row['tp_at_k'])} | "
            f"{int(row['fp_at_k'])} | {float(row['precision_at_k']):.4f} | "
            f"{float(row['precision_lift_vs_exp292']):+.4f} |"
        )
    (output_root / "exp398_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    claims = """# Exp399 presentation claims

## 검증 완료

- Exp292 일반 검토 후보 집합은 모든 정렬 variant에서 동일하다.
- 자동 경보, 기존 우선 검토, 일반 검토 lane은 변경하지 않는다.
- Exp397 prediction은 TEST 라벨과 평가 지표를 포함하지 않는다.
- Exp398은 동결 prediction SHA256 검증 후에만 TEST 라벨을 읽는다.

## 조건부 사용

- 문법 점수는 기존 일반 검토 후보의 순위 근거로 연구할 수 있다.
- 고정 workload Precision@K는 retrospective ranking 진단으로만 사용할 수 있다.

## 사용 금지

- TEST 결과가 가장 좋은 alpha나 variant를 운영 기본값이라고 표현하지 않는다.
- 이번 결과를 자동 경보 성능 개선으로 표현하지 않는다.
- 전체 파이프라인이 prospective 검증 또는 배포 검증됐다고 표현하지 않는다.
"""
    (output_root / "exp399_claims.md").write_text(claims, encoding="utf-8")
    return {
        "summary_rows": len(rows),
        "global_rows": len(global_rows),
        "automatic_hard_changed": False,
        "priority_review_changed": False,
        "standard_review_changed": False,
    }

