#!/usr/bin/env python3
"""Build and validate the minimal Exp483 caller/effect/test map."""

from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
MAP_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap"
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp483_antigravity.json"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp483_antigravity.html"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp483_antigravity.lock"
FINGERPRINT_PATHS_ANTIGRAVITY = (
    "experiments/exp481_strict_evidence_residual_correction_antigravity/evaluation_antigravity.py",
    "experiments/exp481_strict_evidence_residual_correction_antigravity/policy_antigravity.py",
    "experiments/exp482_gate_failure_decomposition_antigravity/results_antigravity.json",
    "experiments/exp483_train_opportunity_oracle_antigravity/diagnostic_antigravity.py",
    "experiments/exp483_train_opportunity_oracle_antigravity/tests/test_exp483_diagnostic_antigravity.py",
    "experiments/exp483_train_opportunity_oracle_antigravity/results_antigravity.json",
    "docs/EXP483_DASHBOARD_antigravity.html",
    "docs/EXP483_IMPLEMENTATION_REPORT_antigravity.html",
    "scratch/build_codemap_exp483_antigravity.py",
)


def _sha_antigravity(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_bytes_antigravity(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def build_map_antigravity() -> dict[str, object]:
    diagnostic = "experiments/exp483_train_opportunity_oracle_antigravity/diagnostic_antigravity.py"
    test = "experiments/exp483_train_opportunity_oracle_antigravity/tests/test_exp483_diagnostic_antigravity.py"
    return {
        "schema": "exp483_train_opportunity_oracle_codemap_v1",
        "baseline": {
            "path": "docs/codemap/codemap_exp482_antigravity.json",
            "sha256": _sha_antigravity(
                ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp482_antigravity.json"
            ),
        },
        "nodes": [
            {
                "id": "exp481_nested_row_policy",
                "path": "experiments/exp481_strict_evidence_residual_correction_antigravity/evaluation_antigravity.py",
                "caller": "Exp481 evaluation and the read-only Exp483 nested replay",
                "effect": "Selects the 130,642 TRAIN rows where row-KNN preserves MiniRocket",
                "tests": [
                    "experiments/exp481_strict_evidence_residual_correction_antigravity/tests/test_evaluation_antigravity.py",
                    test,
                ],
            },
            {
                "id": "exp481_strict_statistics",
                "path": "experiments/exp481_strict_evidence_residual_correction_antigravity/policy_antigravity.py",
                "caller": "Exp481 policy and Exp483 outcome-oracle assessment",
                "effect": "Applies the frozen wins, minimum-AUC, and q20-AUC clauses at K=31",
                "tests": [
                    "experiments/exp481_strict_evidence_residual_correction_antigravity/tests/test_policy_antigravity.py",
                    test,
                ],
            },
            {
                "id": "exp482_population_pin",
                "path": "experiments/exp482_gate_failure_decomposition_antigravity/results_antigravity.json",
                "caller": "Exp483 result identity gate",
                "effect": "Pins the nested population and refuses a changed replay",
                "tests": [test],
            },
            {
                "id": "exp483_diagnostic",
                "path": diagnostic,
                "caller": "python -m experiments.exp483_train_opportunity_oracle_antigravity.diagnostic_antigravity",
                "effect": "Measures F(m), the K=31 TRAIN-only upper bound, and the frozen next-step decision",
                "tests": [test],
            },
            {
                "id": "exp483_outputs",
                "path": "experiments/exp483_train_opportunity_oracle_antigravity/results_antigravity.json",
                "caller": "Exp483 HTML renderers and Codex verification",
                "effect": "Stores aggregate tables only and renders the checked dashboard and easy-language report",
                "tests": [test],
            },
        ],
        "edges": [
            {
                "from": "exp481_nested_row_policy",
                "to": "exp483_diagnostic",
                "type": "frozen_train_population",
            },
            {
                "from": "exp481_strict_statistics",
                "to": "exp483_diagnostic",
                "type": "frozen_three_clause_gate",
            },
            {
                "from": "exp482_population_pin",
                "to": "exp483_diagnostic",
                "type": "routing_identity_check",
            },
            {
                "from": "exp483_diagnostic",
                "to": "exp483_outputs",
                "type": "aggregate_json_and_html",
            },
        ],
        "answers": {
            "callers": "The module CLI and focused tests call Exp483; Exp483 calls the frozen Exp481 row policy and strict statistics.",
            "effects": "Only new Exp483 aggregate JSON and HTML are written. Exp481 and Exp482 artifacts and all router decisions remain unchanged.",
            "tests": "The focused Exp483 tests cover row opportunities, distinct-family selection, K support, aggregation, the three-way decision, source-population rejection, and checked HTML output.",
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
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Exp483 코드맵</title><style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:30px auto;padding:0 20px;line-height:1.55}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3dc;padding:8px;text-align:left;vertical-align:top}}th{{background:#eef2f7}}</style></head><body><h1>Exp483 코드맵</h1><p>무엇이 호출하고, 무엇에 영향을 주며, 어떤 테스트가 확인하는지 기록합니다.</p><table><thead><tr><th>대상</th><th>파일</th><th>호출자</th><th>영향</th><th>테스트</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""


def write_antigravity() -> None:
    payload = build_map_antigravity()
    MAP_DIR_ANTIGRAVITY.mkdir(parents=True, exist_ok=True)
    JSON_PATH_ANTIGRAVITY.write_bytes(_json_bytes_antigravity(payload))
    HTML_PATH_ANTIGRAVITY.write_text(
        render_html_antigravity(payload), encoding="utf-8"
    )
    LOCK_PATH_ANTIGRAVITY.write_bytes(
        _json_bytes_antigravity(
            {
                "schema": "exp483_train_opportunity_oracle_codemap_lock_v1",
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
        raise RuntimeError("Exp483 code map is stale")
    if HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8") != render_html_antigravity(
        payload
    ):
        raise RuntimeError("Exp483 HTML code map is stale")
    if lock["map_json_sha256"] != _sha_antigravity(JSON_PATH_ANTIGRAVITY):
        raise RuntimeError("Exp483 JSON map hash changed")
    if lock["map_html_sha256"] != _sha_antigravity(HTML_PATH_ANTIGRAVITY):
        raise RuntimeError("Exp483 HTML map hash changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.write == args.validate:
        parser.error("choose exactly one of --write or --validate")
    write_antigravity() if args.write else validate_antigravity()
