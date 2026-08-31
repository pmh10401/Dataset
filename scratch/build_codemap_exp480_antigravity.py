#!/usr/bin/env python3
"""Build and validate the scoped Exp480 code-map triplet."""

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
    "experiments/exp480_family_balanced_evidence_routing_antigravity"
)
TEST_DIR_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests"
TRIPLET_ANTIGRAVITY = {
    "html": "docs/codemap/codemap_exp480_antigravity.html",
    "json": "docs/codemap/codemap_exp480_antigravity.json",
    "lock": "docs/codemap/codemap_exp480_antigravity.lock",
}


def _package_antigravity(filename: str) -> str:
    return f"{PACKAGE_ANTIGRAVITY}/{filename}"


def _test_antigravity(filename: str) -> str:
    return f"{TEST_DIR_ANTIGRAVITY}/{filename}"


NODE_SPECS_ANTIGRAVITY = (
    (
        "database",
        "/Users/mac/Documents/Dataset/univariate_ts.db",
        "official feature loader through a read-only SQLite connection",
        "Provides normal TRAIN values only; Exp480 never writes this database.",
        (_test_antigravity("test_evidence_antigravity.py"),),
    ),
    (
        "frozen_scores_962",
        "experiments/exp_routing_label_free_antigravity/checkpoints",
        "evidence_antigravity.py",
        "Provides the frozen four-method metrics for 962 datasets.",
        (_test_antigravity("test_evidence_antigravity.py"),),
    ),
    (
        "frozen_scores_155",
        "experiments/exp479_missing_score_recovery_antigravity/results_antigravity.json",
        "evidence_antigravity.py",
        "Provides the frozen four-method metrics for the remaining 155 datasets.",
        (_test_antigravity("test_evidence_antigravity.py"),),
    ),
    (
        "proposal_source",
        "experiments/exp478_full_1117_official_routing_antigravity/results_antigravity.json",
        "evidence_antigravity.py",
        "Provides the frozen proposal for the row-KNN control on all 1117 datasets.",
        (_test_antigravity("test_evidence_antigravity.py"),),
    ),
    (
        "config",
        _package_antigravity("config_antigravity.py"),
        "all Exp480 execution modules",
        "Freezes paths, method order, feature order and qualification thresholds.",
        (_test_antigravity("test_evidence_antigravity.py"),),
    ),
    (
        "contracts",
        _package_antigravity("contracts_antigravity.py"),
        "evidence_antigravity.py, policy_antigravity.py and evaluation_antigravity.py",
        "Rejects malformed or non-finite candidate metrics and evidence rows.",
        (_test_antigravity("test_evidence_antigravity.py"),),
    ),
    (
        "evidence",
        _package_antigravity("evidence_antigravity.py"),
        "run_antigravity.py",
        "Builds the read-only-derived 1117-row SQLite evidence database.",
        (_test_antigravity("test_evidence_antigravity.py"),),
    ),
    (
        "evidence_database",
        _package_antigravity("evidence_antigravity.sqlite3"),
        "policy_antigravity.py and evaluation_antigravity.py",
        "Stores the deterministic 1117-row derived evidence without changing sources.",
        (_test_antigravity("test_evidence_antigravity.py"),),
    ),
    (
        "family_policy",
        _package_antigravity("policy_antigravity.py"),
        "evaluation_antigravity.py",
        "Qualifies a candidate without modifying the official router.",
        (_test_antigravity("test_policy_antigravity.py"),),
    ),
    (
        "evaluation",
        _package_antigravity("evaluation_antigravity.py"),
        "run_antigravity.py",
        "Compares MiniRocket, row-KNN and Exp480 under identical out-of-family splits.",
        (_test_antigravity("test_evaluation_antigravity.py"),),
    ),
    (
        "acceleration",
        _package_antigravity("acceleration_antigravity.py"),
        "run_antigravity.py before smoke and official execution",
        "Selects MLX only after finite, index-parity, tolerance and 1.1x speed gates.",
        (_test_antigravity("test_acceleration_antigravity.py"),),
    ),
    (
        "runner",
        _package_antigravity("run_antigravity.py"),
        "scratch/run_exp480_family_balanced_evidence_routing_antigravity.py",
        "Writes only Exp480 evidence, deterministic results and HTML outputs.",
        (_test_antigravity("test_runner_antigravity.py"),),
    ),
    (
        "cli",
        "scratch/run_exp480_family_balanced_evidence_routing_antigravity.py",
        "Codex after the applicable task gate",
        "Separates validate, smoke, official and verify-only modes.",
        (_test_antigravity("test_runner_antigravity.py"),),
    ),
    (
        "dashboard",
        "docs/EXP480_DASHBOARD_antigravity.html",
        "run_antigravity.py and the user",
        "Shows only measured state and checks completed tasks immediately.",
        (_test_antigravity("test_runner_antigravity.py"), "HTML parser validation"),
    ),
    (
        "mlx_assessment",
        "scratch/exp480_mlx_assessment_antigravity.json",
        "acceleration_antigravity.py and run_antigravity.py",
        "Records the real-device parity and speed gate or the exact CPU fallback reason.",
        (_test_antigravity("test_acceleration_antigravity.py"),),
    ),
    (
        "smoke_results",
        _package_antigravity("smoke_results_antigravity.json"),
        "run_antigravity.py and the user",
        "Preserves the non-official eight-family end-to-end smoke evidence.",
        (_test_antigravity("test_runner_antigravity.py"),),
    ),
    (
        "official_results",
        _package_antigravity("results_antigravity.json"),
        "run_antigravity.py and independent verification",
        "Preserves deterministic 1117-row qualification evidence without changing the official router.",
        (_test_antigravity("test_runner_antigravity.py"),),
    ),
    (
        "implementation_report",
        "docs/EXP480_IMPLEMENTATION_REPORT_antigravity.html",
        "run_antigravity.py and the user",
        "Explains the preserved qualification result in easy Korean.",
        ("HTML parser validation",),
    ),
)

EDGE_SPECS_ANTIGRAVITY = (
    ("database", "evidence", "read_only_train_features"),
    ("frozen_scores_962", "evidence", "read_only_candidate_metrics"),
    ("frozen_scores_155", "evidence", "read_only_candidate_metrics"),
    ("proposal_source", "evidence", "read_only_control_proposals"),
    ("config", "evidence", "frozen_contract"),
    ("contracts", "evidence", "validated_rows"),
    ("evidence", "evidence_database", "deterministic_sqlite"),
    ("evidence_database", "family_policy", "validated_training_evidence"),
    ("family_policy", "evaluation", "candidate_decisions"),
    ("evaluation", "runner", "qualification_metrics"),
    ("acceleration", "runner", "backend_gate"),
    ("cli", "runner", "execution_mode"),
    ("runner", "dashboard", "atomic_html"),
    ("acceleration", "mlx_assessment", "parity_and_speed_evidence"),
    ("runner", "smoke_results", "non_official_smoke_json"),
    ("runner", "official_results", "deterministic_json"),
    ("official_results", "implementation_report", "easy_summary"),
)

STATIC_FINGERPRINT_PATHS_ANTIGRAVITY = (
    "docs/superpowers/specs/2026-08-31-exp480-family-balanced-evidence-routing-antigravity-design.html",
    "docs/superpowers/plans/2026-08-31-exp480-family-balanced-evidence-routing-antigravity.html",
    "docs/EXP480_DASHBOARD_antigravity.html",
    "docs/codemap/codemap_exp479_antigravity.json",
    "docs/codemap/codemap_exp479_antigravity.lock",
    "experiments/exp478_full_1117_official_routing_antigravity/results_antigravity.json",
    "experiments/exp479_missing_score_recovery_antigravity/results_antigravity.json",
    "experiments/exp_routing_label_free_antigravity/official_router_antigravity.py",
    "scratch/build_codemap_exp480_antigravity.py",
    "scratch/test_build_codemap_exp480_antigravity.py",
)

CONSTRAINTS_ANTIGRAVITY = (
    "Dataset names and source families never enter the runtime feature vector.",
    "A source family is exactly the prefix before the first _normal_ marker.",
    "The source database, frozen scores and official router remain read-only.",
    "Task 6 official execution requires a separate explicit user approval.",
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
    checkpoints = [
        str(path.relative_to(ROOT_ANTIGRAVITY))
        for path in sorted(
            (ROOT_ANTIGRAVITY / "experiments/exp_routing_label_free_antigravity/checkpoints").glob(
                "chunk_*_antigravity.json"
            )
        )
    ]
    return tuple(sorted(set(STATIC_FINGERPRINT_PATHS_ANTIGRAVITY + tuple(dynamic + checkpoints))))


def _fingerprints_antigravity() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in _fingerprint_paths_antigravity():
        payload = (ROOT_ANTIGRAVITY / relative).read_bytes()
        rows.append(
            {"path": relative, "sha256": sha256(payload).hexdigest(), "size": len(payload)}
        )
    return rows


def build_graph_antigravity() -> dict[str, object]:
    """Return the complete current/planned Exp480 dependency boundary."""
    return {
        "schema": "exp480_family_balanced_routing_codemap_v1",
        "status": "task_1_boundary_frozen",
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
<title>Exp480 코드맵</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1250px;margin:30px auto;padding:0 20px 60px;line-height:1.55;color:#182230}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dee8;padding:8px;vertical-align:top;text-align:left}}th{{background:#eef2f7}}code{{background:#eef2f7;padding:2px 4px;border-radius:4px}}
</style></head><body><h1>Exp480 코드맵</h1>
<p>Task 1에서 호출자, 영향 범위와 테스트 경계를 고정했습니다. 예정 파일은 구현 Task가 끝날 때마다 실제 파일과 해시로 갱신합니다.</p>
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
                "schema": "exp480_family_balanced_routing_codemap_lock_v1",
                "map_json_sha256": sha256(json_bytes).hexdigest(),
                "map_html_sha256": sha256(html_bytes).hexdigest(),
                "status": "task_1_boundary_frozen",
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
    if graph != build_graph_antigravity():
        raise RuntimeError("Exp480 code-map JSON is stale")
    if lock.get("map_json_sha256") != sha256(json_path.read_bytes()).hexdigest():
        raise RuntimeError("Exp480 code-map JSON lock mismatch")
    if lock.get("map_html_sha256") != sha256(html_path.read_bytes()).hexdigest():
        raise RuntimeError("Exp480 code-map HTML lock mismatch")
    parser = _HTMLAuditAntigravity()
    parser.feed(html_path.read_text(encoding="utf-8"))
    if parser.tables != 1:
        raise RuntimeError("Exp480 code-map HTML must contain exactly one table")


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
