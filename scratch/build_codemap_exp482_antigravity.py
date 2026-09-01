#!/usr/bin/env python3
"""Build and validate the minimal Exp482 caller/effect/test map."""

from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
MAP_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap"
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp482_antigravity.json"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp482_antigravity.html"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp482_antigravity.lock"
FINGERPRINT_PATHS_ANTIGRAVITY = (
    "experiments/exp481_strict_evidence_residual_correction_antigravity/policy_antigravity.py",
    "experiments/exp481_strict_evidence_residual_correction_antigravity/evaluation_antigravity.py",
    "experiments/exp481_strict_evidence_residual_correction_antigravity/results_antigravity.json",
    "experiments/exp482_gate_failure_decomposition_antigravity/diagnostic_antigravity.py",
    "experiments/exp482_gate_failure_decomposition_antigravity/tests/test_diagnostic_antigravity.py",
    "experiments/exp482_gate_failure_decomposition_antigravity/results_antigravity.json",
    "docs/EXP482_DASHBOARD_antigravity.html",
    "docs/EXP482_IMPLEMENTATION_REPORT_antigravity.html",
    "scratch/build_codemap_exp482_antigravity.py",
)


def _sha_antigravity(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_bytes_antigravity(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def build_map_antigravity() -> dict[str, object]:
    diagnostic = "experiments/exp482_gate_failure_decomposition_antigravity/diagnostic_antigravity.py"
    test = "experiments/exp482_gate_failure_decomposition_antigravity/tests/test_diagnostic_antigravity.py"
    return {
        "schema": "exp482_gate_failure_decomposition_codemap_v1",
        "baseline": {
            "path": "docs/codemap/codemap_exp481_antigravity.json",
            "sha256": _sha_antigravity(
                ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp481_antigravity.json"
            ),
        },
        "nodes": [
            {
                "id": "exp481_strict_policy",
                "path": "experiments/exp481_strict_evidence_residual_correction_antigravity/policy_antigravity.py",
                "caller": "Exp481 nested evaluation and the read-only Exp482 diagnostic wrapper",
                "effect": "Produces the frozen three-candidate local assessments without changing thresholds",
                "tests": [
                    "experiments/exp481_strict_evidence_residual_correction_antigravity/tests/test_policy_antigravity.py",
                    test,
                ],
            },
            {
                "id": "exp481_nested_evaluation",
                "path": "experiments/exp481_strict_evidence_residual_correction_antigravity/evaluation_antigravity.py",
                "caller": "Exp481 runner and Exp482 identity replay",
                "effect": "Recomputes the frozen 1,117-dataset decision payload used by the equality gate",
                "tests": [
                    "experiments/exp481_strict_evidence_residual_correction_antigravity/tests/test_evaluation_antigravity.py",
                    test,
                ],
            },
            {
                "id": "exp482_diagnostic",
                "path": diagnostic,
                "caller": "python -m experiments.exp482_gate_failure_decomposition_antigravity.diagnostic_antigravity",
                "effect": "Collects TRAIN-side clause failures and margins while rejecting any Exp481 evaluation change",
                "tests": [test],
            },
            {
                "id": "exp482_results",
                "path": "experiments/exp482_gate_failure_decomposition_antigravity/results_antigravity.json",
                "caller": "Exp482 HTML renderers and Codex result review",
                "effect": "Stores exactly five aggregate tables; raw candidate records are not persisted",
                "tests": [test],
            },
            {
                "id": "exp482_html",
                "path": "docs/EXP482_DASHBOARD_antigravity.html",
                "caller": "User progress and result review",
                "effect": "Shows completed tasks, identity status, and the easy-language conclusion",
                "tests": [test],
            },
        ],
        "edges": [
            {"from": "exp481_strict_policy", "to": "exp482_diagnostic", "type": "read_only_assessments"},
            {"from": "exp481_nested_evaluation", "to": "exp482_diagnostic", "type": "identity_replay"},
            {"from": "exp482_diagnostic", "to": "exp482_results", "type": "five_aggregate_tables"},
            {"from": "exp482_results", "to": "exp482_html", "type": "easy_korean_render"},
        ],
        "answers": {
            "callers": "The module CLI and focused test call Exp482; Exp482 calls the frozen Exp481 policy and nested evaluator.",
            "effects": "Only diagnostic JSON and HTML are written. Exp481 policy, thresholds, official result, and source evidence remain unchanged.",
            "tests": "The focused Exp482 test covers clause separation, complete inner collection, five-table aggregation, HTML completion state, and Exp481 mismatch rejection. The 46 Exp481 tests cover the reused policy/evaluator.",
        },
        "fingerprints": [
            {
                "path": relative,
                "sha256": _sha_antigravity(ROOT_ANTIGRAVITY / relative),
                "size": (ROOT_ANTIGRAVITY / relative).stat().st_size,
            }
            for relative in FINGERPRINT_PATHS_ANTIGRAVITY
        ],
        "unknown_relationships": [],
    }


def render_html_antigravity(payload: dict[str, object]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(node['id']))}</td>"
        f"<td><code>{escape(str(node['path']))}</code></td>"
        f"<td>{escape(str(node['caller']))}</td>"
        f"<td>{escape(str(node['effect']))}</td>"
        f"<td>{escape(', '.join(node['tests']))}</td>"
        "</tr>"
        for node in payload["nodes"]
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Exp482 코드맵</title><style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:30px auto;padding:0 20px;line-height:1.55}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3dc;padding:8px;text-align:left;vertical-align:top}}th{{background:#eef2f7}}</style></head><body><h1>Exp482 코드맵</h1><p>무엇이 호출하고, 무엇에 영향을 주며, 어떤 테스트가 확인하는지 기록합니다.</p><table><thead><tr><th>대상</th><th>파일</th><th>호출자</th><th>영향</th><th>테스트</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""


def write_antigravity() -> None:
    payload = build_map_antigravity()
    MAP_DIR_ANTIGRAVITY.mkdir(parents=True, exist_ok=True)
    JSON_PATH_ANTIGRAVITY.write_bytes(_json_bytes_antigravity(payload))
    HTML_PATH_ANTIGRAVITY.write_text(render_html_antigravity(payload), encoding="utf-8")
    LOCK_PATH_ANTIGRAVITY.write_bytes(
        _json_bytes_antigravity(
            {
                "schema": "exp482_gate_failure_decomposition_codemap_lock_v1",
                "status": "diagnostic_complete",
                "map_json_sha256": _sha_antigravity(JSON_PATH_ANTIGRAVITY),
                "map_html_sha256": _sha_antigravity(HTML_PATH_ANTIGRAVITY),
            }
        )
    )


def validate_antigravity() -> None:
    payload = json.loads(JSON_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    if payload != build_map_antigravity():
        raise RuntimeError("Exp482 code map is stale")
    if HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8") != render_html_antigravity(payload):
        raise RuntimeError("Exp482 HTML code map is stale")
    if lock["map_json_sha256"] != _sha_antigravity(JSON_PATH_ANTIGRAVITY):
        raise RuntimeError("Exp482 JSON map hash changed")
    if lock["map_html_sha256"] != _sha_antigravity(HTML_PATH_ANTIGRAVITY):
        raise RuntimeError("Exp482 HTML map hash changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.write == args.validate:
        parser.error("choose exactly one of --write or --validate")
    write_antigravity() if args.write else validate_antigravity()
