"""Behavior tests for the Exp481 scoped code-map generator."""

from __future__ import annotations

from hashlib import sha256
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
BUILDER_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "scratch/build_codemap_exp481_antigravity.py"
)
PACKAGE_ANTIGRAVITY = (
    "experiments/exp481_strict_evidence_residual_correction_antigravity"
)
REQUIRED_NODE_IDS_ANTIGRAVITY = (
    "strict_policy",
    "nested_evaluation",
    "numba_kernels",
    "acceleration",
    "runner",
    "exp480_evidence_database",
    "exp480_official_result",
    "config",
    "contracts",
    "cli",
    "dashboard",
    "smoke_results",
    "official_results",
    "run_manifest",
    "implementation_report",
    "acceleration_assessment",
    "approved_spec",
    "approved_plan",
)
REQUIRED_EDGES_ANTIGRAVITY = (
    {
        "from": "exp480_evidence_database",
        "to": "strict_policy",
        "type": "read_only_residual_evidence",
    },
    {
        "from": "exp480_official_result",
        "to": "nested_evaluation",
        "type": "read_only_baseline_result",
    },
    {
        "from": "numba_kernels",
        "to": "strict_policy",
        "type": "exact_statistics",
    },
    {
        "from": "strict_policy",
        "to": "nested_evaluation",
        "type": "candidate_decisions",
    },
    {
        "from": "nested_evaluation",
        "to": "runner",
        "type": "qualification_metrics",
    },
    {
        "from": "acceleration",
        "to": "runner",
        "type": "backend_gate",
    },
    {
        "from": "cli",
        "to": "runner",
        "type": "execution_mode",
    },
    {
        "from": "runner",
        "to": "dashboard",
        "type": "atomic_interim_html",
    },
    {
        "from": "runner",
        "to": "smoke_results",
        "type": "non_official_smoke_json",
    },
    {
        "from": "runner",
        "to": "official_results",
        "type": "deterministic_json",
    },
    {
        "from": "runner",
        "to": "run_manifest",
        "type": "nondeterministic_execution_evidence",
    },
    {
        "from": "acceleration",
        "to": "acceleration_assessment",
        "type": "parity_and_speed_evidence",
    },
    {
        "from": "runner",
        "to": "implementation_report",
        "type": "verified_easy_korean_report",
    },
)
TRIPLET_FILES_ANTIGRAVITY = (
    "scratch/build_codemap_exp481_antigravity.py",
    "scratch/test_build_codemap_exp481_antigravity.py",
    "docs/codemap/codemap_exp481_antigravity.html",
    "docs/codemap/codemap_exp481_antigravity.json",
    "docs/codemap/codemap_exp481_antigravity.lock",
)


def _builder_antigravity():
    specification = importlib.util.spec_from_file_location(
        "build_codemap_exp481_antigravity", BUILDER_PATH_ANTIGRAVITY
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Exp481 code-map builder cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def build_graph_antigravity():
    return _builder_antigravity().build_graph_antigravity()


def test_exp481_map_answers_three_questions_antigravity():
    graph = build_graph_antigravity()
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes["strict_policy"]["caller"] == "evaluation_antigravity.py"
    assert "row-knn" in nodes["strict_policy"]["effect"].lower()
    assert nodes["strict_policy"]["tests"] == [
        "experiments/exp481_strict_evidence_residual_correction_antigravity/tests/test_policy_antigravity.py"
    ]


def test_exp481_map_contains_required_nodes_and_edges_antigravity() -> None:
    graph = build_graph_antigravity()
    nodes = {node["id"]: node for node in graph["nodes"]}
    for node_id in REQUIRED_NODE_IDS_ANTIGRAVITY:
        node = nodes[node_id]
        assert node["caller"]
        assert node["effect"]
        assert node["tests"]
        assert "planned" in node
    for edge in REQUIRED_EDGES_ANTIGRAVITY:
        assert edge in graph["edges"]
    assert nodes["strict_policy"]["path"] == (
        f"{PACKAGE_ANTIGRAVITY}/policy_antigravity.py"
    )
    assert nodes["nested_evaluation"]["caller"] == "run_antigravity.py"
    assert nodes["numba_kernels"]["caller"] == "policy_antigravity.py"
    assert nodes["acceleration"]["caller"] == "run_antigravity.py"
    assert nodes["runner"]["caller"] == "scratch CLI"
    assert nodes["exp480_evidence_database"]["planned"] is False
    assert nodes["exp480_official_result"]["planned"] is False
    assert nodes["dashboard"]["planned"] is False
    assert nodes["approved_spec"]["planned"] is False
    assert nodes["approved_plan"]["planned"] is False
    assert nodes["config"]["planned"] is False
    assert nodes["contracts"]["planned"] is False
    assert nodes["strict_policy"]["planned"] is False
    assert nodes["nested_evaluation"]["planned"] is True
    assert nodes["numba_kernels"]["planned"] is True


def test_exp481_graph_and_html_are_deterministic_antigravity() -> None:
    builder = _builder_antigravity()
    graph_a = builder.build_graph_antigravity()
    graph_b = builder.build_graph_antigravity()
    assert graph_a == graph_b
    assert builder._json_bytes_antigravity(
        graph_a
    ) == builder._json_bytes_antigravity(graph_b)
    assert builder._render_html_antigravity(
        graph_a
    ) == builder._render_html_antigravity(graph_b)


def test_exp481_json_html_and_lock_parse_and_hash_antigravity() -> None:
    builder = _builder_antigravity()
    builder.write_triplet_antigravity()
    json_path = builder.ROOT_ANTIGRAVITY / builder.TRIPLET_ANTIGRAVITY["json"]
    html_path = builder.ROOT_ANTIGRAVITY / builder.TRIPLET_ANTIGRAVITY["html"]
    lock_path = builder.ROOT_ANTIGRAVITY / builder.TRIPLET_ANTIGRAVITY["lock"]
    graph = json.loads(json_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    parser = builder._HTMLAuditAntigravity()
    parser.feed(html_path.read_text(encoding="utf-8"))
    assert parser.tables == 1
    assert graph["nodes"]
    assert graph["edges"]
    assert lock["map_json_sha256"] == sha256(json_path.read_bytes()).hexdigest()
    assert lock["map_html_sha256"] == sha256(html_path.read_bytes()).hexdigest()
    builder.validate_triplet_antigravity()


class _TableCounterAntigravity(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag == "table":
            self.tables += 1


def test_exp481_validate_rejects_stale_html_matching_lock_antigravity() -> None:
    builder = _builder_antigravity()
    builder.write_triplet_antigravity()
    html_path = builder.ROOT_ANTIGRAVITY / builder.TRIPLET_ANTIGRAVITY["html"]
    lock_path = builder.ROOT_ANTIGRAVITY / builder.TRIPLET_ANTIGRAVITY["lock"]
    stale_html = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>stale Exp481 code map</title></head>"
        "<body><table><tr><th>stale</th></tr></table></body></html>\n"
    ).encode("utf-8")
    try:
        html_path.write_bytes(stale_html)
        parser = _TableCounterAntigravity()
        parser.feed(stale_html.decode("utf-8"))
        assert parser.tables == 1
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["map_html_sha256"] = sha256(stale_html).hexdigest()
        lock_path.write_bytes(builder._json_bytes_antigravity(lock))
        try:
            builder.validate_triplet_antigravity()
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "expected stale HTML rejection against current graph render"
            )
    finally:
        builder.write_triplet_antigravity()


def test_exp481_validate_rejects_stale_path_and_fingerprint_antigravity() -> None:
    builder = _builder_antigravity()
    builder.write_triplet_antigravity()
    json_path = builder.ROOT_ANTIGRAVITY / builder.TRIPLET_ANTIGRAVITY["json"]
    lock_path = builder.ROOT_ANTIGRAVITY / builder.TRIPLET_ANTIGRAVITY["lock"]
    graph = json.loads(json_path.read_text(encoding="utf-8"))
    graph["nodes"][0]["path"] = "docs/stale_exp481_path_antigravity.py"
    json_path.write_bytes(builder._json_bytes_antigravity(graph))
    try:
        builder.validate_triplet_antigravity()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected stale path rejection")
    builder.write_triplet_antigravity()
    graph = json.loads(json_path.read_text(encoding="utf-8"))
    graph["fingerprints"][0]["sha256"] = "0" * 64
    json_path.write_bytes(builder._json_bytes_antigravity(graph))
    try:
        builder.validate_triplet_antigravity()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected fingerprint drift rejection")
    builder.write_triplet_antigravity()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["map_json_sha256"] = "0" * 64
    lock_path.write_bytes(builder._json_bytes_antigravity(lock))
    try:
        builder.validate_triplet_antigravity()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected lock hash rejection")
    builder.write_triplet_antigravity()
    builder.validate_triplet_antigravity()


def test_exp481_backend_selection_wording_antigravity() -> None:
    graph = build_graph_antigravity()
    nodes = {node["id"]: node for node in graph["nodes"]}
    joined = " ".join(
        [str(item) for item in graph["constraints"]]
        + [str(node["effect"]) for node in graph["nodes"]]
    )
    assert "parity plus full-stage speed benefit" in nodes["acceleration"]["effect"]
    assert "retains NumPy" in nodes["acceleration"]["effect"]
    assert "parity plus full-stage speed benefit" in str(graph["constraints"])
    assert "simpler NumPy or official implementation is retained" in str(
        graph["constraints"]
    )
    assert "Records Numba/NumPy parity" in nodes["acceleration_assessment"]["effect"]
    assert "after parity and full-stage speed selection" in nodes["numba_kernels"]["effect"]
    assert "Numba parity failure always stops execution" not in joined
    assert "Stops on Numba parity failure" not in joined
    assert "Numba is the future official execution path" not in joined
    plan_text = (
        ROOT_ANTIGRAVITY
        / "docs/superpowers/plans/2026-09-01-exp481-strict-evidence-residual-correction-antigravity.html"
    ).read_text(encoding="utf-8")
    assert "numpy_reference" in plan_text
    assert "numba_cpu" in plan_text
    assert "numba_plus_mlx" in plan_text
    assert "선택/대체 이유" in plan_text
    assert "완전한 backend 평가" in plan_text
    assert "Numba/NumPy parity·속도 선택과 MLX 조건 반영" in plan_text
    assert "Numba 공식 경로와 MLX 조건 반영" not in plan_text


def test_exp481_new_filenames_contain_antigravity() -> None:
    _builder_antigravity().write_triplet_antigravity()
    graph = build_graph_antigravity()
    for node in graph["nodes"]:
        name = Path(str(node["path"])).name
        assert "antigravity" in name
        for test_path in node["tests"]:
            test_name = Path(str(test_path)).name
            if test_name == "HTML parser validation":
                continue
            assert "antigravity" in test_name
    for relative in TRIPLET_FILES_ANTIGRAVITY:
        assert "antigravity" in Path(relative).name
        assert (ROOT_ANTIGRAVITY / relative).exists()
