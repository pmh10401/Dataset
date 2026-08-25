"""Final publication invariants for the synchronized Exp471 code map."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import unittest

from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import (
    EXP467_SOURCE_COMMIT_ANTIGRAVITY,
    EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
)
from scratch.generate_exp471_codemap_antigravity import (
    REPOSITORY_ROOT,
    main as codemap_main_antigravity,
)


EXP471_NAMESPACE_ANTIGRAVITY = (
    "experiments/exp471_exp467_evaluation_recovery_antigravity"
)
FROZEN_EXP467_COMMIT_ANTIGRAVITY = "1e616731a88da9bc0268823e1d2cef2a53c1054f"
FROZEN_EXP467_CONTEXT_SHA256_ANTIGRAVITY = (
    "31341d8e14cf58a486cf7418276c432ac2dbda1af139699ff2afbd85969442af"
)
PRODUCTION_PATHS_ANTIGRAVITY = frozenset(
    {
        "experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/numba_preprocessing_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/preprocessing_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/reporting_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py",
    }
)
TEST_PATHS_ANTIGRAVITY = frozenset(
    {
        "experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_cache_artifacts_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_codemap_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_config_contracts_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_evaluation_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_numba_preprocessing_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_preprocessing_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_score_cache_antigravity.py",
        "experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_source_authority_antigravity.py",
    }
)
SCRATCH_PATHS_ANTIGRAVITY = frozenset(
    {
        "scratch/generate_exp471_codemap_antigravity.py",
        "scratch/run_exp471_exp467_evaluation_recovery_antigravity.py",
    }
)
EXPECTED_PATHS_ANTIGRAVITY = (
    PRODUCTION_PATHS_ANTIGRAVITY | TEST_PATHS_ANTIGRAVITY | SCRATCH_PATHS_ANTIGRAVITY
)
EXPECTED_EFFECTS_ANTIGRAVITY = {
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/cache_artifacts_antigravity.py": (
        "Descriptor-bound create-only persistence for Exp471 score-cache artifacts."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/cli_antigravity.py": (
        "Closed Exp471 source-to-report command orchestration."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/config_antigravity.py": (
        "Frozen Exp471 source identity, roots, ordering, and backend policy."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/contracts_antigravity.py": (
        "Strict, deeply immutable generic artifact contracts for Exp471."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/evaluation_antigravity.py": (
        "Label-gated, create-only persistence for the frozen Exp471 evaluation."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/numba_preprocessing_antigravity.py": (
        "Candidate Numba ASPE kernels for the Exp471 preprocessing spike."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/preprocessing_antigravity.py": (
        "Result-preserving Exp471 preprocessing parallelism and live ETA evidence."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/reporting_antigravity.py": (
        "Verified, create-only Exp471 paper-report publication."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/score_cache_antigravity.py": (
        "Label-free, resumable preparation of deterministic Exp471 score shards."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/source_authority_antigravity.py": (
        "Descriptor-bound reconstruction and publication of frozen Exp467 authority."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/tests/test_cache_artifacts_antigravity.py": (
        "Lifecycle tests for durable create-only Exp471 score-cache artifacts."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/tests/test_codemap_antigravity.py": (
        "Final publication invariants for the synchronized Exp471 code map."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/tests/test_config_contracts_antigravity.py": (
        "Behavior tests for the frozen Exp471 configuration and strict contracts."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/tests/test_evaluation_antigravity.py": (
        "Public-literal tests for Exp471 durable continuous-score evaluation."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/tests/test_numba_preprocessing_antigravity.py": (
        "Parity tests for candidate Exp471 Numba preprocessing kernels."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/tests/test_preprocessing_antigravity.py": (
        "Deterministic preprocessing parallelism and live ETA evidence for Exp471."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/tests/test_reporting_cli_antigravity.py": (
        "Public-literal lifecycle tests for Exp471 reporting and its closed CLI."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/tests/test_score_cache_antigravity.py": (
        "Synthetic behavior tests for resumable Exp471 score preparation."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/tests/test_source_authority_antigravity.py": (
        "Synthetic Git and source-tree tests for the Exp467 authority boundary."
    ),
    "scratch/generate_exp471_codemap_antigravity.py": (
        "Generate and verify the factual, synchronized Exp471 code-map triplet."
    ),
    "scratch/run_exp471_exp467_evaluation_recovery_antigravity.py": (
        "Closed launcher for the verified Exp471 recovery pipeline."
    ),
}

EXPECTED_EDGES_ANTIGRAVITY = frozenset(
    {
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/numba_preprocessing_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/preprocessing_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/reporting_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/preprocessing_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/reporting_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/reporting_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/reporting_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/reporting_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_cache_artifacts_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_cache_artifacts_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_cache_artifacts_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_codemap_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_codemap_antigravity.py", "scratch/generate_exp471_codemap_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_config_contracts_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_config_contracts_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_config_contracts_antigravity.py", "scratch/generate_exp471_codemap_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_evaluation_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_evaluation_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_evaluation_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_evaluation_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_numba_preprocessing_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/numba_preprocessing_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_preprocessing_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/preprocessing_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_evaluation_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/reporting_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_evaluation_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_source_authority_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py", "scratch/run_exp471_exp467_evaluation_recovery_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_score_cache_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_score_cache_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_score_cache_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_score_cache_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_source_authority_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py"),
        ("experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_source_authority_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py"),
        ("scratch/run_exp471_exp467_evaluation_recovery_antigravity.py", "experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py"),
    }
)


def _tracked_python_paths_antigravity(root: Path) -> frozenset[str]:
    result = subprocess.run(
        ("git", "-C", str(root), "ls-files", "--cached", "-z"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paths: set[str] = set()
    for encoded in result.stdout.split(b"\0"):
        if not encoded:
            continue
        relative = encoded.decode("utf-8")
        candidate = root / relative
        if candidate.suffix == ".py" and candidate.is_file():
            paths.add(relative)
    return frozenset(paths)


def _module_name_antigravity(relative: str) -> str:
    path = PurePosixPath(relative).with_suffix("")
    parts = path.parts[:-1] if path.name == "__init__" else path.parts
    return ".".join(parts)


def _source_imports_antigravity(root: Path, relative: str) -> frozenset[str]:
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _module_name_antigravity(relative).split(".")[:-1]
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        prefix = (
            package[: max(0, len(package) - node.level + 1)]
            if node.level
            else []
        )
        if node.module:
            prefix.extend(node.module.split("."))
        if prefix:
            imported.add(".".join(prefix))
            imported.update(
                ".".join((*prefix, alias.name))
                for alias in node.names
                if alias.name != "*"
            )
    return frozenset(imported)


def _independent_ast_edges_antigravity(root: Path) -> frozenset[tuple[str, str]]:
    selected = {
        _module_name_antigravity(relative): relative
        for relative in EXPECTED_PATHS_ANTIGRAVITY
    }
    if len(selected) != len(EXPECTED_PATHS_ANTIGRAVITY):
        raise AssertionError("literal Exp471 closure has duplicate module names")
    edges: set[tuple[str, str]] = set()
    for caller in sorted(_tracked_python_paths_antigravity(root)):
        for imported in _source_imports_antigravity(root, caller):
            for module, callee in selected.items():
                if imported == module or imported.startswith(f"{module}."):
                    edges.add((caller, callee))
    return frozenset(edges)


def _source_effect_antigravity(root: Path, relative: str) -> str:
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return ast.get_docstring(tree) or "No module docstring is present."


def _assert_payload_answers_antigravity(
    payload: Mapping[str, object],
    expected_edges: frozenset[tuple[str, str]],
) -> None:
    if payload.get("schema") != "exp471-codemap.v1":
        raise AssertionError("code-map schema differs")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise AssertionError("code-map scope is not a mapping")
    expected_scope = {
        "node_count": 21,
        "production_node_count": 10,
        "test_node_count": 9,
        "external_closure_policy": "index_tracked_exp471_production_closure",
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            raise AssertionError(f"code-map scope {key} differs")

    encoded_edges = payload.get("edges")
    if not isinstance(encoded_edges, list):
        raise AssertionError("code-map edges are not a list")
    actual_edges: list[tuple[str, str]] = []
    for edge in encoded_edges:
        if (
            not isinstance(edge, Mapping)
            or set(edge) != {"caller", "callee", "kind"}
            or edge.get("kind") != "ast_import"
            or type(edge.get("caller")) is not str
            or type(edge.get("callee")) is not str
        ):
            raise AssertionError("code-map edge schema differs")
        actual_edges.append((edge["caller"], edge["callee"]))
    if len(actual_edges) != len(set(actual_edges)):
        raise AssertionError("code-map edges contain a duplicate")
    if frozenset(actual_edges) != expected_edges:
        raise AssertionError("code-map exact edge set differs")

    encoded_nodes = payload.get("nodes")
    if not isinstance(encoded_nodes, list):
        raise AssertionError("code-map nodes are not a list")
    nodes: dict[str, Mapping[str, object]] = {}
    for node in encoded_nodes:
        if (
            not isinstance(node, Mapping)
            or set(node) != {"path", "module", "callers", "effects", "tests"}
            or type(node.get("path")) is not str
        ):
            raise AssertionError("code-map node schema differs")
        path = node["path"]
        if path in nodes:
            raise AssertionError("code-map nodes contain a duplicate path")
        nodes[path] = node
    if set(nodes) != EXPECTED_PATHS_ANTIGRAVITY:
        raise AssertionError("code-map exact node closure differs")
    if set(EXPECTED_EFFECTS_ANTIGRAVITY) != EXPECTED_PATHS_ANTIGRAVITY:
        raise AssertionError("literal effect oracle is incomplete")

    for path in sorted(EXPECTED_PATHS_ANTIGRAVITY):
        node = nodes[path]
        expected_callers = sorted(
            caller for caller, callee in expected_edges if callee == path
        )
        expected_tests = [
            caller for caller in expected_callers if "/tests/" in caller
        ]
        if node.get("module") != _module_name_antigravity(path):
            raise AssertionError(f"code-map module answer differs for {path}")
        if node.get("callers") != expected_callers:
            raise AssertionError(f"code-map callers answer differs for {path}")
        if node.get("tests") != expected_tests:
            raise AssertionError(f"code-map tests answer differs for {path}")
        if node.get("effects") != EXPECTED_EFFECTS_ANTIGRAVITY[path]:
            raise AssertionError(f"code-map effects answer differs for {path}")


class Exp471CodeMapTestsAntigravity(unittest.TestCase):
    def test_independent_validator_rejects_edge_and_answer_mutations(self) -> None:
        validator = globals().get("_assert_payload_answers_antigravity")
        self.assertIsNotNone(
            validator, "independent code-map payload validator is missing"
        )
        self.assertTrue(callable(validator))
        payload_path = (
            REPOSITORY_ROOT / "docs/codemap/codemap_exp471_antigravity.json"
        )
        payload = json.loads(payload_path.read_bytes())

        missing_edge = deepcopy(payload)
        missing_edge["edges"].pop()

        extra_edge = deepcopy(payload)
        extra_edge["edges"].append(
            {
                "caller": "scratch/generate_exp471_codemap_antigravity.py",
                "callee": (
                    f"{EXP471_NAMESPACE_ANTIGRAVITY}/config_antigravity.py"
                ),
                "kind": "ast_import",
            }
        )

        wrong_callers = deepcopy(payload)
        config_node = next(
            node
            for node in wrong_callers["nodes"]
            if node["path"]
            == f"{EXP471_NAMESPACE_ANTIGRAVITY}/config_antigravity.py"
        )
        config_node["callers"].pop()

        wrong_tests = deepcopy(payload)
        config_node = next(
            node
            for node in wrong_tests["nodes"]
            if node["path"]
            == f"{EXP471_NAMESPACE_ANTIGRAVITY}/config_antigravity.py"
        )
        config_node["tests"].pop()

        for name, mutation in (
            ("missing_edge", missing_edge),
            ("extra_edge", extra_edge),
            ("wrong_callers", wrong_callers),
            ("wrong_tests", wrong_tests),
        ):
            with self.subTest(mutation=name):
                with self.assertRaises(AssertionError):
                    validator(mutation, EXPECTED_EDGES_ANTIGRAVITY)

    def test_exact_closure_edges_callers_effects_and_tests_are_independent(self) -> None:
        tracked = _tracked_python_paths_antigravity(REPOSITORY_ROOT)
        namespace_paths = {
            path
            for path in tracked
            if path.startswith(f"{EXP471_NAMESPACE_ANTIGRAVITY}/")
        }
        exp471_scratch_paths = {
            path
            for path in tracked
            if path.startswith("scratch/")
            and "exp471" in PurePosixPath(path).name
            and path.endswith("_antigravity.py")
        }
        self.assertEqual(
            namespace_paths | exp471_scratch_paths, EXPECTED_PATHS_ANTIGRAVITY
        )

        ast_edges = _independent_ast_edges_antigravity(REPOSITORY_ROOT)
        self.assertEqual(ast_edges, EXPECTED_EDGES_ANTIGRAVITY)
        for path, expected_effect in EXPECTED_EFFECTS_ANTIGRAVITY.items():
            with self.subTest(effect_path=path):
                self.assertEqual(
                    _source_effect_antigravity(REPOSITORY_ROOT, path),
                    expected_effect,
                )

        payload_path = (
            REPOSITORY_ROOT / "docs/codemap/codemap_exp471_antigravity.json"
        )
        payload = json.loads(payload_path.read_bytes())
        _assert_payload_answers_antigravity(payload, EXPECTED_EDGES_ANTIGRAVITY)

    def test_triplet_is_fresh_and_embeds_identical_json_bytes(self) -> None:
        self.assertEqual(codemap_main_antigravity(("--check",)), 0)
        json_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp471_antigravity.json"
        html_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp471_antigravity.html"
        json_bytes = json_path.read_bytes()
        html_bytes = html_path.read_bytes()
        marker = b'<script id="codemap-data" type="application/json">'
        embedded = html_bytes.split(marker, 1)[1].split(b"</script>", 1)[0]
        self.assertEqual(embedded, json_bytes)

    def test_lock_binds_all_modules_artifacts_and_fixed_source_identity(self) -> None:
        json_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp471_antigravity.json"
        html_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp471_antigravity.html"
        lock_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp471_antigravity.lock"
        payload = json.loads(json_path.read_bytes())
        lock = json.loads(lock_path.read_bytes())
        expected_fingerprints = {
            relative: sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest()
            for relative in sorted(EXPECTED_PATHS_ANTIGRAVITY)
        }
        self.assertEqual(lock["module_fingerprints"], expected_fingerprints)
        self.assertEqual(lock["scanned_scope"], sorted(EXPECTED_PATHS_ANTIGRAVITY))
        self.assertEqual(
            lock["artifact_sha256"],
            {
                "docs/codemap/codemap_exp471_antigravity.html": sha256(
                    html_path.read_bytes()
                ).hexdigest(),
                "docs/codemap/codemap_exp471_antigravity.json": sha256(
                    json_path.read_bytes()
                ).hexdigest(),
            },
        )
        expected_identity = {
            "frozen_exp467_source_commit": FROZEN_EXP467_COMMIT_ANTIGRAVITY,
            "frozen_exp467_source_context_sha256": (
                FROZEN_EXP467_CONTEXT_SHA256_ANTIGRAVITY
            ),
        }
        self.assertEqual(
            EXP467_SOURCE_COMMIT_ANTIGRAVITY, FROZEN_EXP467_COMMIT_ANTIGRAVITY
        )
        self.assertEqual(
            EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
            FROZEN_EXP467_CONTEXT_SHA256_ANTIGRAVITY,
        )
        for evidence in (
            payload["scope"]["governance_fingerprint_evidence"],
            lock["governance_fingerprint_evidence"],
        ):
            with self.subTest(evidence=evidence):
                self.assertEqual(
                    {key: evidence[key] for key in expected_identity}, expected_identity
                )

    def test_exp467_source_tree_is_unchanged_from_frozen_commit(self) -> None:
        result = subprocess.run(
            (
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "diff",
                "--name-only",
                FROZEN_EXP467_COMMIT_ANTIGRAVITY,
                "--",
                "experiments/exp467_numba_calibration_resume_antigravity",
            ),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
