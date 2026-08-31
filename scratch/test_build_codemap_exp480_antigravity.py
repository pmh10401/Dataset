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

