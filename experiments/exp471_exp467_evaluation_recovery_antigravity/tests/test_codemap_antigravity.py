"""Final publication invariants for the synchronized Exp471 code map."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import unittest

from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import (
    EXP467_SOURCE_COMMIT_ANTIGRAVITY,
    EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
)
from scratch.generate_exp471_codemap_antigravity import (
    REPOSITORY_ROOT,
    ast_callers_antigravity,
    build_codemap_payload_antigravity,
    main as codemap_main_antigravity,
    tracked_existing_paths_antigravity,
)


EXP471_NAMESPACE_ANTIGRAVITY = (
    "experiments/exp471_exp467_evaluation_recovery_antigravity"
)
FROZEN_EXP467_COMMIT_ANTIGRAVITY = "1e616731a88da9bc0268823e1d2cef2a53c1054f"
FROZEN_EXP467_CONTEXT_SHA256_ANTIGRAVITY = (
    "31341d8e14cf58a486cf7418276c432ac2dbda1af139699ff2afbd85969442af"
)
PRODUCTION_PATHS_ANTIGRAVITY = frozenset(
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/{name}_antigravity.py"
    for name in (
        "cache_artifacts",
        "cli",
        "config",
        "contracts",
        "evaluation",
        "reporting",
        "score_cache",
        "source_authority",
    )
)
TEST_PATHS_ANTIGRAVITY = frozenset(
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/tests/{name}_antigravity.py"
    for name in (
        "test_cache_artifacts",
        "test_codemap",
        "test_config_contracts",
        "test_evaluation",
        "test_reporting_cli",
        "test_score_cache",
        "test_source_authority",
    )
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
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/reporting_antigravity.py": (
        "Verified, create-only Exp471 paper-report publication."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/score_cache_antigravity.py": (
        "Label-free, resumable preparation of deterministic Exp471 score shards."
    ),
    f"{EXP471_NAMESPACE_ANTIGRAVITY}/source_authority_antigravity.py": (
        "Descriptor-bound reconstruction and publication of frozen Exp467 authority."
    ),
}


def _relative_antigravity(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


class Exp471CodeMapTestsAntigravity(unittest.TestCase):
    def test_exact_final_closure_and_production_answers(self) -> None:
        paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
        self.assertEqual(
            {_relative_antigravity(path) for path in paths},
            EXPECTED_PATHS_ANTIGRAVITY,
        )
        payload = build_codemap_payload_antigravity(
            paths, ast_callers_antigravity(paths)
        )
        self.assertEqual(payload["schema"], "exp471-codemap.v1")
        scope = payload["scope"]
        self.assertEqual(scope["node_count"], 17)
        self.assertEqual(scope["production_node_count"], 8)
        self.assertEqual(scope["test_node_count"], 7)
        self.assertEqual(
            scope["external_closure_policy"],
            "index_tracked_exp471_production_closure",
        )
        nodes = {node["path"]: node for node in payload["nodes"]}
        self.assertEqual(set(nodes), EXPECTED_PATHS_ANTIGRAVITY)
        edges = {(edge["caller"], edge["callee"]) for edge in payload["edges"]}
        for path, node in nodes.items():
            with self.subTest(path=path):
                self.assertEqual(
                    set(node), {"path", "module", "callers", "effects", "tests"}
                )
                expected_callers = sorted(
                    caller for caller, callee in edges if callee == path
                )
                self.assertEqual(node["callers"], expected_callers)
                self.assertEqual(
                    node["tests"],
                    [caller for caller in expected_callers if "/tests/" in caller],
                )
        for path in PRODUCTION_PATHS_ANTIGRAVITY:
            with self.subTest(production_path=path):
                self.assertEqual(
                    nodes[path]["effects"], EXPECTED_EFFECTS_ANTIGRAVITY[path]
                )
                self.assertTrue(nodes[path]["callers"])
                self.assertTrue(nodes[path]["tests"])

    def test_required_direct_import_edges_are_present(self) -> None:
        paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
        payload = build_codemap_payload_antigravity(
            paths, ast_callers_antigravity(paths)
        )
        edges = {(edge["caller"], edge["callee"]) for edge in payload["edges"]}
        namespace = EXP471_NAMESPACE_ANTIGRAVITY
        def module(name: str) -> str:
            return f"{namespace}/{name}_antigravity.py"

        expected = {
            (module("cache_artifacts"), module("config")),
            (module("cache_artifacts"), module("contracts")),
            (module("cli"), module("cache_artifacts")),
            (module("cli"), module("config")),
            (module("cli"), module("contracts")),
            (module("cli"), module("evaluation")),
            (module("cli"), module("reporting")),
            (module("cli"), module("score_cache")),
            (module("cli"), module("source_authority")),
            (module("contracts"), module("config")),
            (module("evaluation"), module("cache_artifacts")),
            (module("evaluation"), module("config")),
            (module("evaluation"), module("contracts")),
            (module("evaluation"), module("score_cache")),
            (module("evaluation"), module("source_authority")),
            (module("reporting"), module("config")),
            (module("reporting"), module("contracts")),
            (module("reporting"), module("evaluation")),
            (module("reporting"), module("score_cache")),
            (module("score_cache"), module("cache_artifacts")),
            (module("score_cache"), module("config")),
            (module("score_cache"), module("contracts")),
            (module("score_cache"), module("source_authority")),
            (module("source_authority"), module("config")),
            (module("source_authority"), module("contracts")),
            (
                "scratch/run_exp471_exp467_evaluation_recovery_antigravity.py",
                module("cli"),
            ),
            (
                f"{namespace}/tests/test_codemap_antigravity.py",
                "scratch/generate_exp471_codemap_antigravity.py",
            ),
        }
        self.assertTrue(expected <= edges, sorted(expected - edges))

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
