#!/usr/bin/env python3
"""Build and validate the scoped Exp481 code-map triplet."""

from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
from typing import Iterable


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
PACKAGE_ANTIGRAVITY = (
    "experiments/exp481_strict_evidence_residual_correction_antigravity"
)
TEST_DIR_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests"
EXP480_PACKAGE_ANTIGRAVITY = (
    "experiments/exp480_family_balanced_evidence_routing_antigravity"
)
TRIPLET_ANTIGRAVITY = {
    "html": "docs/codemap/codemap_exp481_antigravity.html",
    "json": "docs/codemap/codemap_exp481_antigravity.json",
    "lock": "docs/codemap/codemap_exp481_antigravity.lock",
}


def _package_antigravity(filename: str) -> str:
    return f"{PACKAGE_ANTIGRAVITY}/{filename}"


def _test_antigravity(filename: str) -> str:
    return f"{TEST_DIR_ANTIGRAVITY}/{filename}"


NODE_SPECS_ANTIGRAVITY = (
    (
        "exp480_evidence_database",
        f"{EXP480_PACKAGE_ANTIGRAVITY}/evidence_antigravity.sqlite3",
        "policy_antigravity.py and evaluation_antigravity.py",
        "Provides frozen Exp480 residual evidence without modification.",
        (_test_antigravity("test_policy_antigravity.py"),),
    ),
    (
        "exp480_official_result",
        f"{EXP480_PACKAGE_ANTIGRAVITY}/results_antigravity.json",
        "evaluation_antigravity.py",
        "Provides the frozen Exp480 official qualification result without modification.",
        (_test_antigravity("test_evaluation_antigravity.py"),),
    ),
    (
        "approved_spec",
        "docs/superpowers/specs/2026-09-01-exp481-strict-evidence-residual-correction-antigravity-design.html",
        "Task 1 generator and later implementation tasks",
        "Freezes the approved strict residual-correction design.",
        ("scratch/test_build_codemap_exp481_antigravity.py",),
    ),
    (
        "approved_plan",
        "docs/superpowers/plans/2026-09-01-exp481-strict-evidence-residual-correction-antigravity.html",
        "Task 1 generator and later implementation tasks",
        "Freezes the approved Task 1-6 implementation sequence.",
        ("scratch/test_build_codemap_exp481_antigravity.py",),
    ),
    (
        "config",
        _package_antigravity("config_antigravity.py"),
        "all Exp481 execution modules",
        "Freezes paths, residual thresholds, nested holdouts and qualification rules.",
        (_test_antigravity("test_policy_antigravity.py"),),
    ),
    (
        "contracts",
        _package_antigravity("contracts_antigravity.py"),
        "policy_antigravity.py and evaluation_antigravity.py",
        "Rejects malformed residual evidence and leaked family identifiers.",
        (_test_antigravity("test_policy_antigravity.py"),),
    ),
    (
        "strict_policy",
        _package_antigravity("policy_antigravity.py"),
        "evaluation_antigravity.py",
        "Preserves row-KNN unless strict residual evidence passes.",
        (_test_antigravity("test_policy_antigravity.py"),),
    ),
    (
        "nested_evaluation",
        _package_antigravity("evaluation_antigravity.py"),
        "run_antigravity.py",
        "Uses inner and outer family holdouts without score leakage.",
        (_test_antigravity("test_evaluation_antigravity.py"),),
    ),
    (
        "numba_kernels",
        _package_antigravity("kernels_numba_antigravity.py"),
        "policy_antigravity.py",
        "Provides optional Numba neighbour and strict statistics after parity and full-stage speed selection.",
        (_test_antigravity("test_numba_antigravity.py"),),
    ),
    (
        "acceleration",
        _package_antigravity("acceleration_antigravity.py"),
        "run_antigravity.py",
        "Selects Numba only after parity plus full-stage speed benefit; otherwise retains NumPy and records the fallback reason, then gates optional MLX.",
        (_test_antigravity("test_acceleration_antigravity.py"),),
    ),
    (
        "runner",
        _package_antigravity("run_antigravity.py"),
        "scratch CLI",
        "Writes deterministic results and verified HTML.",
        (_test_antigravity("test_runner_antigravity.py"),),
    ),
    (
        "cli",
        "scratch/run_exp481_strict_evidence_residual_correction_antigravity.py",
        "Codex after the applicable task gate",
        "Separates validate, smoke, official and verify-only modes.",
        (_test_antigravity("test_runner_antigravity.py"),),
    ),
    (
        "dashboard",
        "docs/EXP481_DASHBOARD_antigravity.html",
        "run_antigravity.py and the user",
        "Shows the Task 1 boundary map, later gates and the frozen row-KNN default.",
        (_test_antigravity("test_runner_antigravity.py"), "HTML parser validation"),
    ),
    (
        "acceleration_assessment",
        "scratch/exp481_acceleration_assessment_antigravity.json",
        "acceleration_antigravity.py and run_antigravity.py",
        "Records Numba/NumPy parity, timing, selected backend, and the optional MLX gate or CPU fallback reason.",
        (_test_antigravity("test_acceleration_antigravity.py"),),
    ),
    (
        "smoke_results",
        _package_antigravity("smoke_results_antigravity.json"),
        "run_antigravity.py and the user",
        "Preserves the non-official nested-holdout smoke evidence.",
        (_test_antigravity("test_runner_antigravity.py"),),
    ),
    (
        "official_results",
        _package_antigravity("results_antigravity.json"),
        "run_antigravity.py and independent verification",
        "Preserves the official 1117-row residual-correction result without changing row-KNN unless evidence passes.",
        (_test_antigravity("test_runner_antigravity.py"),),
    ),
    (
        "run_manifest",
        _package_antigravity("run_manifest_antigravity.json"),
        "run_official_antigravity() and verify_official_antigravity()",
        "Records execution timing, Numba/MLX evidence and the deterministic result SHA outside the result bytes.",
        (_test_antigravity("test_runner_antigravity.py"),),
    ),
    (
        "implementation_report",
        "docs/EXP481_IMPLEMENTATION_REPORT_antigravity.html",
        "verify_official_antigravity() after result bytes and manifest SHA match",
        "Explains the residual-correction decision and unchanged row-KNN fallback in easy Korean.",
        (_test_antigravity("test_runner_antigravity.py"), "HTML parser validation"),
    ),
)

EDGE_SPECS_ANTIGRAVITY = (
    ("exp480_evidence_database", "strict_policy", "read_only_residual_evidence"),
    ("exp480_official_result", "nested_evaluation", "read_only_baseline_result"),
    ("approved_spec", "config", "approved_design"),
    ("approved_plan", "cli", "task_sequence"),
    ("config", "strict_policy", "frozen_contract"),
    ("config", "nested_evaluation", "frozen_contract"),
    ("contracts", "strict_policy", "validated_rows"),
    ("contracts", "nested_evaluation", "validated_rows"),
    ("numba_kernels", "strict_policy", "exact_statistics"),
    ("numba_kernels", "acceleration", "parity_reference"),
    ("strict_policy", "nested_evaluation", "candidate_decisions"),
    ("nested_evaluation", "runner", "qualification_metrics"),
    ("acceleration", "runner", "backend_gate"),
    ("cli", "runner", "execution_mode"),
    ("runner", "dashboard", "atomic_interim_html"),
    ("acceleration", "acceleration_assessment", "parity_and_speed_evidence"),
    ("runner", "smoke_results", "non_official_smoke_json"),
    ("runner", "official_results", "deterministic_json"),
    ("runner", "run_manifest", "nondeterministic_execution_evidence"),
    ("runner", "implementation_report", "verified_easy_korean_report"),
)

STATIC_FINGERPRINT_PATHS_ANTIGRAVITY = (
    "docs/superpowers/specs/2026-09-01-exp481-strict-evidence-residual-correction-antigravity-design.html",
    "docs/superpowers/plans/2026-09-01-exp481-strict-evidence-residual-correction-antigravity.html",
    "docs/EXP481_DASHBOARD_antigravity.html",
    "docs/codemap/codemap_exp480_antigravity.json",
    "docs/codemap/codemap_exp480_antigravity.lock",
    f"{EXP480_PACKAGE_ANTIGRAVITY}/results_antigravity.json",
    f"{EXP480_PACKAGE_ANTIGRAVITY}/evidence_antigravity.sqlite3",
    "experiments/exp_routing_label_free_antigravity/official_router_antigravity.py",
    "scratch/build_codemap_exp481_antigravity.py",
    "scratch/test_build_codemap_exp481_antigravity.py",
)

CONSTRAINTS_ANTIGRAVITY = (
    "Dataset names and source families never enter the runtime feature vector.",
    "Exp480 evidence DB, Exp480 official result, original DB, and official row-KNN remain read-only.",
    "Numba is selected only after parity plus full-stage speed benefit; otherwise the simpler NumPy or official implementation is retained and the reason recorded.",
    "Task 6 official 1,117-dataset execution requires a separate explicit user approval.",
    "Every newly created file contains antigravity in its filename.",
)


def _node_antigravity(
    node_id: str,
    path: str,
    caller: str,
    effect: str,
    tests: Iterable[str],
) -> dict[str, object]:
    local_path = ROOT_ANTIGRAVITY / path if not path.startswith("/") else Path(path)
    return {
        "id": node_id,
        "path": path,
        "caller": caller,
        "effect": effect,
        "tests": list(tests),
        "planned": not local_path.exists(),
    }


def _fingerprint_paths_antigravity() -> tuple[str, ...]:
    dynamic = [
        path
        for _, path, _, _, _ in NODE_SPECS_ANTIGRAVITY
        if not path.startswith("/") and (ROOT_ANTIGRAVITY / path).is_file()
    ]
    return tuple(sorted(set(STATIC_FINGERPRINT_PATHS_ANTIGRAVITY + tuple(dynamic))))


def _fingerprints_antigravity() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in _fingerprint_paths_antigravity():
        payload = (ROOT_ANTIGRAVITY / relative).read_bytes()
        rows.append(
            {"path": relative, "sha256": sha256(payload).hexdigest(), "size": len(payload)}
        )
    return rows


def build_graph_antigravity() -> dict[str, object]:
    """Return the complete Exp481 dependency boundary."""
    return {
        "schema": "exp481_strict_evidence_residual_correction_codemap_v1",
        "status": "task_1_boundaries_mapped",
        "nodes": [
            _node_antigravity(node_id, path, caller, effect, tests)
            for node_id, path, caller, effect, tests in NODE_SPECS_ANTIGRAVITY
        ],
        "edges": [
            {"from": source, "to": target, "type": edge_type}
            for source, target, edge_type in EDGE_SPECS_ANTIGRAVITY
        ],
        "constraints": list(CONSTRAINTS_ANTIGRAVITY),
        "fingerprints": _fingerprints_antigravity(),
    }


def _json_bytes_antigravity(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _render_html_antigravity(graph: dict[str, object]) -> str:
    nodes = graph["nodes"]
    assert isinstance(nodes, list)
    rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(node['id']))}</code></td>"
        f"<td><code>{escape(str(node['path']))}</code></td>"
        f"<td>{escape(str(node['caller']))}</td>"
        f"<td>{escape(str(node['effect']))}</td>"
        f"<td>{'<br>'.join(escape(str(test)) for test in node['tests'])}</td>"
        f"<td>{'예정' if node['planned'] else '존재'}</td>"
        "</tr>"
        for node in nodes
        if isinstance(node, dict)
    )
    return f"""<!doctype html>
<html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Exp481 코드맵</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1250px;margin:30px auto;padding:0 20px 60px;line-height:1.55;color:#182230}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dee8;padding:8px;vertical-align:top;text-align:left}}th{{background:#eef2f7}}code{{background:#eef2f7;padding:2px 4px;border-radius:4px}}
</style></head><body><h1>Exp481 코드맵</h1>
<p>Task 1은 strict residual correction의 호출자, 영향, 테스트 경계를 고정했습니다. 예정 파일은 아직 없을 수 있으며 존재하는 파일은 SHA-256 지문으로 잠급니다. Exp480 근거 DB와 공식 결과는 읽기 전용입니다.</p>
<table><tr><th>ID</th><th>경로</th><th>호출자</th><th>영향</th><th>테스트</th><th>상태</th></tr>{rows}</table>
</body></html>"""


class _HTMLAuditAntigravity(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag == "table":
            self.tables += 1


def write_triplet_antigravity() -> None:
    """Write deterministic JSON, HTML and their lock hashes."""
    graph = build_graph_antigravity()
    json_bytes = _json_bytes_antigravity(graph)
    html_bytes = (_render_html_antigravity(graph) + "\n").encode("utf-8")
    json_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY["json"]
    html_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY["html"]
    lock_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY["lock"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(json_bytes)
    html_path.write_bytes(html_bytes)
    lock_path.write_bytes(
        _json_bytes_antigravity(
            {
                "schema": "exp481_strict_evidence_residual_correction_codemap_lock_v1",
                "map_json_sha256": sha256(json_bytes).hexdigest(),
                "map_html_sha256": sha256(html_bytes).hexdigest(),
                "status": "task_1_boundaries_mapped",
            }
        )
    )


def validate_triplet_antigravity() -> None:
    """Reject stale fingerprints, malformed HTML or mismatched lock hashes."""
    json_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY["json"]
    html_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY["html"]
    lock_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY["lock"]
    graph = json.loads(json_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    current_graph = build_graph_antigravity()
    if graph != current_graph:
        raise RuntimeError("Exp481 code-map JSON is stale")
    if lock.get("map_json_sha256") != sha256(json_path.read_bytes()).hexdigest():
        raise RuntimeError("Exp481 code-map JSON lock mismatch")
    if lock.get("map_html_sha256") != sha256(html_path.read_bytes()).hexdigest():
        raise RuntimeError("Exp481 code-map HTML lock mismatch")
    expected_html_bytes = (
        _render_html_antigravity(current_graph) + "\n"
    ).encode("utf-8")
    if html_path.read_bytes() != expected_html_bytes:
        raise RuntimeError("Exp481 code-map HTML is stale")
    parser = _HTMLAuditAntigravity()
    parser.feed(html_path.read_text(encoding="utf-8"))
    if parser.tables != 1:
        raise RuntimeError("Exp481 code-map HTML must contain exactly one table")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_triplet_antigravity()
    if args.validate:
        validate_triplet_antigravity()
    if not args.write and not args.validate:
        print(json.dumps(build_graph_antigravity(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_antigravity()
