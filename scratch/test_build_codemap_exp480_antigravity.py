"""Behavior tests for the Exp480 scoped code-map generator."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
BUILDER_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "scratch/build_codemap_exp480_antigravity.py"
)


def _builder_antigravity():
    specification = importlib.util.spec_from_file_location(
        "build_codemap_exp480_antigravity", BUILDER_PATH_ANTIGRAVITY
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Exp480 code-map builder cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_exp480_map_answers_calls_effects_and_tests_antigravity() -> None:
    """Removing an Exp480 boundary or its coverage must fail the map contract."""
    graph = _builder_antigravity().build_graph_antigravity()
    by_id = {node["id"]: node for node in graph["nodes"]}

    assert by_id["family_policy"]["caller"] == "evaluation_antigravity.py"
    assert "official router" in by_id["family_policy"]["effect"].lower()
    assert by_id["family_policy"]["tests"] == [
        "experiments/exp480_family_balanced_evidence_routing_antigravity/"
        "tests/test_policy_antigravity.py"
    ]


def test_exp480_map_tracks_the_derived_database_flow_antigravity() -> None:
    """Removing the derived DB node or its consumer edge must fail the map."""
    graph = _builder_antigravity().build_graph_antigravity()
    by_id = {node["id"]: node for node in graph["nodes"]}

    assert by_id["evidence_database"] == {
        "id": "evidence_database",
        "path": (
            "experiments/exp480_family_balanced_evidence_routing_antigravity/"
            "evidence_antigravity.sqlite3"
        ),
        "caller": "policy_antigravity.py and evaluation_antigravity.py",
        "effect": "Stores the deterministic 1117-row derived evidence without changing sources.",
        "tests": [
            "experiments/exp480_family_balanced_evidence_routing_antigravity/"
            "tests/test_evidence_antigravity.py"
        ],
        "planned": False,
    }
    assert {
        "from": "evidence_database",
        "to": "family_policy",
        "type": "validated_training_evidence",
    } in graph["edges"]


def test_exp480_map_tracks_smoke_and_mlx_assessment_artifacts_antigravity() -> None:
    """Task 5 evidence must remain visible in the current code map."""
    graph = _builder_antigravity().build_graph_antigravity()
    by_id = {node["id"]: node for node in graph["nodes"]}

    assert by_id["smoke_results"]["planned"] is False
    assert by_id["mlx_assessment"]["planned"] is False
    assert {
        "from": "acceleration",
        "to": "mlx_assessment",
        "type": "parity_and_speed_evidence",
    } in graph["edges"]
    assert {
        "from": "runner",
        "to": "smoke_results",
        "type": "non_official_smoke_json",
    } in graph["edges"]


def test_exp480_map_tracks_manifest_and_fail_closed_verification_flow_antigravity() -> None:
    """The map must not hide the separate manifest or post-byte-check report flow."""
    graph = _builder_antigravity().build_graph_antigravity()
    by_id = {node["id"]: node for node in graph["nodes"]}

    assert by_id["run_manifest"] == {
        "id": "run_manifest",
        "path": (
            "experiments/exp480_family_balanced_evidence_routing_antigravity/"
            "run_manifest_antigravity.json"
        ),
        "caller": "run_official_antigravity() and verify_official_antigravity()",
        "effect": (
            "Records execution timing, device, CPU/MLX parity and speed evidence, "
            "and the deterministic result SHA outside the result bytes."
        ),
        "tests": [
            "experiments/exp480_family_balanced_evidence_routing_antigravity/"
            "tests/test_runner_antigravity.py"
        ],
        "planned": False,
    }
    assert by_id["official_verification"]["caller"] == "CLI --verify-only"
    assert "exact result bytes" in by_id["official_verification"]["effect"]
    assert "exact manifest schema" in by_id["official_verification"]["effect"]
    assert "canonical evidence DB SHA" in by_id["evidence"]["effect"]
    assert by_id["implementation_report"]["caller"] == (
        "verify_official_antigravity() after result bytes and manifest SHA match"
    )
    assert {
        "from": "run_manifest",
        "to": "official_verification",
        "type": "result_sha_guard",
    } in graph["edges"]
    assert {
        "from": "official_verification",
        "to": "implementation_report",
        "type": "verified_easy_korean_report",
    } in graph["edges"]


def test_exp480_map_closes_the_verified_failed_result_in_present_tense_antigravity() -> None:
    """A preflight or future-artifact map must fail after verified qualification."""
    graph = _builder_antigravity().build_graph_antigravity()
    by_id = {node["id"]: node for node in graph["nodes"]}

    assert graph["status"] == "task_6_verified_failure_preserved"
    assert by_id["official_results"]["effect"] == (
        "Preserves the verified 1117-row failed qualification without changing "
        "the official router."
    )
    assert by_id["dashboard"]["effect"] == (
        "Shows the verified failed qualification, all gates, comparisons and "
        "fallback reasons."
    )
    assert by_id["run_manifest"]["effect"].startswith("Records execution timing")
    assert by_id["official_verification"]["effect"].startswith(
        "Independently recomputed NumPy CPU result bytes"
    )
    assert by_id["implementation_report"]["effect"].startswith(
        "Explains the verified failed qualification"
    )
    for artifact in (
        "dashboard",
        "official_results",
        "run_manifest",
        "implementation_report",
    ):
        assert by_id[artifact]["planned"] is False
    rendered = _builder_antigravity()._render_html_antigravity(graph)
    assert "자격 실패를 확인해 기존 라우터를 보존했습니다" in rendered
    assert "예정 파일은" not in rendered
