#!/usr/bin/env python3
"""Build and validate the Exp484 caller/effect/test map."""

from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
MAP_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap"
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp484_antigravity.json"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp484_antigravity.html"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp484_antigravity.lock"
RESULT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp484_distance_qualification_antigravity/results_antigravity.json"
)
FINGERPRINT_PATHS_ANTIGRAVITY = (
    "experiments/exp481_strict_evidence_residual_correction_antigravity/policy_antigravity.py",
    "experiments/exp481_strict_evidence_residual_correction_antigravity/kernels_numba_antigravity.py",
    "experiments/exp484_distance_qualification_antigravity/acceleration_antigravity.py",
    "experiments/exp484_distance_qualification_antigravity/run_antigravity.py",
    "experiments/exp484_distance_qualification_antigravity/tests/test_distance_variants_antigravity.py",
    "experiments/exp484_distance_qualification_antigravity/tests/test_run_antigravity.py",
    "experiments/exp484_distance_qualification_antigravity/results_antigravity.json",
    "experiments/exp484_distance_qualification_antigravity/run_manifest_antigravity.json",
    "scratch/exp484_acceleration_assessment_antigravity.json",
    "docs/EXP484_DASHBOARD_antigravity.html",
    "docs/EXP484_IMPLEMENTATION_REPORT_antigravity.html",
    "scratch/build_codemap_exp484_antigravity.py",
)


def _sha_antigravity(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_bytes_antigravity(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def build_map_antigravity() -> dict[str, object]:
    policy = "experiments/exp481_strict_evidence_residual_correction_antigravity/policy_antigravity.py"
    kernels = "experiments/exp481_strict_evidence_residual_correction_antigravity/kernels_numba_antigravity.py"
    acceleration = "experiments/exp484_distance_qualification_antigravity/acceleration_antigravity.py"
    runner = "experiments/exp484_distance_qualification_antigravity/run_antigravity.py"
    distance_test = "experiments/exp484_distance_qualification_antigravity/tests/test_distance_variants_antigravity.py"
    run_test = "experiments/exp484_distance_qualification_antigravity/tests/test_run_antigravity.py"
    return {
        "schema": "exp484_distance_qualification_codemap_v1",
        "baseline": {
            "path": "docs/codemap/codemap_exp483_antigravity.json",
            "sha256": _sha_antigravity(
                ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp483_antigravity.json"
            ),
        },
        "nodes": [
            {
                "id": "exp481_opt_in_distance_policy",
                "path": policy,
                "caller": "Frozen Exp481 evaluation by default; Exp484 runner with an explicit distance variant",
                "effect": "Keeps standard Euclidean unchanged and adds opt-in robust Manhattan and shrinkage Mahalanobis neighbourhoods",
                "tests": [
                    "experiments/exp481_strict_evidence_residual_correction_antigravity/tests/test_policy_antigravity.py",
                    distance_test,
                ],
            },
            {
                "id": "exp484_numba_distance_kernels",
                "path": kernels,
                "caller": "Exp484 policy distance stage and acceleration parity benchmark",
                "effect": "Computes only the two new distances; frozen neighbour selection and strict statistics remain on their selected existing backend",
                "tests": [distance_test],
            },
            {
                "id": "exp484_acceleration",
                "path": acceleration,
                "caller": "Exp484 runner before the 1,117-dataset evaluation",
                "effect": "Checks finite values, numerical/decision/candidate parity, cold-inclusive timing, and the MLX device gate",
                "tests": [distance_test],
            },
            {
                "id": "exp484_runner",
                "path": runner,
                "caller": "python -m experiments.exp484_distance_qualification_antigravity.run_antigravity",
                "effect": "Pins Exp480/481/483 sources, evaluates two distances on the same nested TRAIN population, and writes result-backed HTML",
                "tests": [run_test],
            },
            {
                "id": "exp484_outputs",
                "path": "experiments/exp484_distance_qualification_antigravity/results_antigravity.json",
                "caller": "Dashboard, report, manifest, code map, and later experiment design",
                "effect": "Records 1,117 datasets, 128 families, zero changes for all distances, and the decision to replace features",
                "tests": [run_test, "scratch/build_codemap_exp484_antigravity.py"],
            },
        ],
        "edges": [
            {"from": "exp481_opt_in_distance_policy", "to": "exp484_runner", "type": "explicit_distance_variant"},
            {"from": "exp484_numba_distance_kernels", "to": "exp484_acceleration", "type": "parity_and_timing"},
            {"from": "exp484_acceleration", "to": "exp484_runner", "type": "distance_backend_selection"},
            {"from": "exp484_runner", "to": "exp484_outputs", "type": "nested_train_result_and_html"},
        ],
        "answers": {
            "callers": "Exp481 continues to call the default standard policy. Only Exp484 passes the new distance_variant and distance_backend arguments; its CLI and focused tests call the new runner.",
            "effects": "The change affects TRAIN-only neighbourhood distance calculation and Exp484 artifacts. Candidate methods, K=31, strict gate clauses, dataset/family-name exclusion, Exp481 artifacts, and the official router are unchanged.",
            "tests": "The Exp481 suite protects the default behavior. Exp484 tests cover hand-checked neighbour orders, Numba parity and isolation, frozen source hashes, equal populations, result selection, publication, checked HTML, and verification state.",
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
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>EXP484 코드맵</title><style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:30px auto;padding:0 20px;line-height:1.55}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3dc;padding:8px;text-align:left;vertical-align:top}}th{{background:#eef2f7}}</style></head><body><h1>EXP484 코드맵</h1><p>호출자, 영향 범위, 검증 테스트를 한곳에 기록합니다.</p><table><thead><tr><th>대상</th><th>파일</th><th>호출자</th><th>영향</th><th>테스트</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""


def write_antigravity() -> None:
    payload = build_map_antigravity()
    MAP_DIR_ANTIGRAVITY.mkdir(parents=True, exist_ok=True)
    JSON_PATH_ANTIGRAVITY.write_bytes(_json_bytes_antigravity(payload))
    HTML_PATH_ANTIGRAVITY.write_text(render_html_antigravity(payload), encoding="utf-8")
    result = json.loads(RESULT_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    LOCK_PATH_ANTIGRAVITY.write_bytes(
        _json_bytes_antigravity(
            {
                "schema": "exp484_distance_qualification_codemap_lock_v1",
                "status": result["state"],
                "result_sha256": _sha_antigravity(RESULT_PATH_ANTIGRAVITY),
                "map_json_sha256": _sha_antigravity(JSON_PATH_ANTIGRAVITY),
                "map_html_sha256": _sha_antigravity(HTML_PATH_ANTIGRAVITY),
            }
        )
    )


def validate_antigravity() -> None:
    payload = json.loads(JSON_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    if payload != build_map_antigravity():
        raise RuntimeError("Exp484 code map is stale")
    if HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8") != render_html_antigravity(payload):
        raise RuntimeError("Exp484 HTML code map is stale")
    if lock["status"] != result["state"]:
        raise RuntimeError("Exp484 code map status is stale")
    if lock["result_sha256"] != _sha_antigravity(RESULT_PATH_ANTIGRAVITY):
        raise RuntimeError("Exp484 result hash changed")
    if lock["map_json_sha256"] != _sha_antigravity(JSON_PATH_ANTIGRAVITY):
        raise RuntimeError("Exp484 JSON map hash changed")
    if lock["map_html_sha256"] != _sha_antigravity(HTML_PATH_ANTIGRAVITY):
        raise RuntimeError("Exp484 HTML map hash changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.write == args.validate:
        parser.error("choose exactly one of --write or --validate")
    write_antigravity() if args.write else validate_antigravity()
