from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

from .config import EXPERIMENT_ID


def write_report(root: Path) -> Path:
    rows = list(csv.DictReader((root / "summary.csv").open(newline="", encoding="utf-8")))
    lines = [
        "# Exp255: 변화점 특징 결과",
        "",
        "Robust CUSUM/MOSUM과 PELT L1/L2/RBF를 독립 변화점 점수원으로 평가했습니다.",
        "이 실험은 자동 경보를 변경하지 않은 후향 점수원 검증입니다.",
        "",
        "| 특징 | Alpha | Exp253 Hard 재확인 | 정밀도 | 남은 Standard | TP | FP | 정밀도 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {float(row['alpha']):.3f} | "
            f"{row['exp253_hard_confirmation_count']} | "
            f"{100 * float(row['exp253_hard_confirmation_precision']):.2f}% | "
            f"{row['exp253_remaining_standard_priority_count']} | "
            f"{row['exp253_remaining_standard_priority_tp']} | "
            f"{row['exp253_remaining_standard_priority_fp']} | "
            f"{100 * float(row['exp253_remaining_standard_priority_precision']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "반복 관찰한 TEST 데이터에 대한 retrospective 결과입니다.",
            "TEST 결과로 특징, alpha 또는 자동 경보 승격 규칙을 선택하지 않습니다.",
            "신규 잠금 데이터 검증 전에는 deployment 또는 prospective 성능으로 표현하지 않습니다.",
        ]
    )
    path = root / "RESULT_REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_manifest(root: Path, command: str) -> Path:
    names = (
        "audit.json",
        "cache_manifest.csv",
        "frozen_predictions.jsonl",
        "prediction_manifest.json",
        "results.csv",
        "summary.csv",
        "evaluation_manifest.json",
        "independent_reaggregation.json",
        "RESULT_REPORT.md",
    )
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names
        if (root / name).exists()
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "files": files,
        "execution_command": command,
        "git_commit": commit,
        "retrospective": True,
        "deployment_eligible": False,
        "autonomous_policy_changed": False,
        "known_limitations": [
            "reused TEST benchmark",
            "no prospective validation",
            "no TEST-selected policy",
        ],
    }
    path = root / "MANIFEST.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path
