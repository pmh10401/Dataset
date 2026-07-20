from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

from .config import EXPERIMENT_ID


def _summary(output_root: Path) -> dict[str, str]:
    with (output_root / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def write_report(output_root: Path) -> Path:
    row = _summary(output_root)
    path = output_root / "RESULT_REPORT.md"
    lines = [
        "# Exp253: 제한적 자동 경보 확대",
        "",
        "## 쉬운 설명",
        "",
        "기존 자동 경보는 그대로 유지했습니다. 기존 **일반 검토** 후보 중에서",
        "MiniROCKET 1024 또는 4096이 같은 위치를 다시 지목한 항목만 자동 경보로 올렸습니다.",
        "경보·검토 범위 밖의 위치는 새로 추가하지 않았습니다.",
        "",
        "## 결과",
        "",
        "| 구분 | 후보 | TP | FP | Precision | Mean F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 기존 자동 경보 | {row['baseline_hard_count']} | {row['baseline_hard_tp']} | {row['baseline_hard_fp']} | {100*float(row['baseline_hard_precision']):.3f}% | {float(row['baseline_hard_mean_f1']):.6f} |",
        f"| 승격 후보 | {row['promoted_count']} | {row['promoted_tp']} | {row['promoted_fp']} | {100*float(row['promoted_precision']):.3f}% | {float(row['promoted_mean_f1']):.6f} |",
        f"| 확대 후 자동 경보 | {row['final_hard_count']} | {row['final_hard_tp']} | {row['final_hard_fp']} | {100*float(row['final_hard_precision']):.3f}% | {float(row['final_hard_mean_f1']):.6f} |",
        "",
        f"승격 후보의 정밀도 80% 진단 기준 통과: **{'예' if row['incremental_precision_gate_80pct'] == '1' else '아니오'}**",
        "",
        "## 신뢰도 표시",
        "",
        f"- S 등급: {row['grade_s_count']}개, 두 MiniROCKET과 다른 방식이 함께 동의",
        f"- A 등급: {row['grade_a_count']}개, 두 MiniROCKET이 동의",
        f"- B 등급: {row['grade_b_count']}개, 한 MiniROCKET이 동의",
        "",
        "## 중요한 제한",
        "",
        "이 결과는 이미 여러 번 관찰한 TEST 결과를 검토한 뒤 선택한 규칙의 **후향 재현**입니다.",
        "신규 설비에서 사전 고정해 검증한 결과가 아니므로 배포 가능한 운영 정책이나 prospective 검증으로 해석하면 안 됩니다.",
        "전체 파이프라인이 end-to-end TRAIN-only라고 주장하지 않습니다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_manifest(output_root: Path, command: str) -> Path:
    filenames = (
        "audit.json", "frozen_predictions.jsonl", "prediction_manifest.json",
        "results.csv", "summary.csv", "lane_transitions.csv",
        "confidence_grades.csv", "evaluation_manifest.json",
        "independent_reaggregation.json", "RESULT_REPORT.md",
    )
    files = {
        name: hashlib.sha256((output_root / name).read_bytes()).hexdigest()
        for name in filenames if (output_root / name).exists()
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
        "selected_from_prior_test_review": True,
        "deployment_eligible": False,
        "autonomous_scope": "baseline Hard plus guarded Standard promotions",
        "human_assisted_scope": "remaining Standard and unchanged Priority review",
        "known_limitations": [
            "repeatedly observed benchmark TEST data",
            "rule selected after prior TEST review",
            "no prospective equipment validation",
            "not end-to-end strict TRAIN-only",
        ],
    }
    path = output_root / "MANIFEST.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path
