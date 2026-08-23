"""Behavioral contracts for the synchronized Exp467 code-map triplet."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

from scratch.generate_exp467_codemap_antigravity import (
    FROZEN_EXP450_LOCK_SHA256_ANTIGRAVITY,
    REPOSITORY_ROOT,
    ast_callers_antigravity,
    build_codemap_payload_antigravity,
    frozen_exp450_fingerprint_evidence_antigravity,
    main as codemap_main_antigravity,
    tracked_existing_paths_antigravity,
)
from scratch.run_exp467_numba_calibration_resume_antigravity import (
    _code_closure_antigravity,
    runtime_dependency_relative_paths_antigravity,
)


EXP467_NAMESPACE_ANTIGRAVITY = (
    "experiments/exp467_numba_calibration_resume_antigravity"
)
PRODUCTION_PATHS_ANTIGRAVITY = frozenset(
    f"{EXP467_NAMESPACE_ANTIGRAVITY}/{name}_antigravity.py"
    for name in (
        "artifacts",
        "backend",
        "config",
        "contracts",
        "evaluation",
        "imaging",
        "hard_route_adapter",
        "hard_sources_adapter",
        "migration",
        "minirocket_adapter",
        "reporting",
        "runner",
        "science_artifacts",
        "scoring",
        "transforms",
        "verifier",
    )
)
TEST_PATHS_ANTIGRAVITY = frozenset(
    f"{EXP467_NAMESPACE_ANTIGRAVITY}/tests/{name}_antigravity.py"
    for name in (
        "test_codemap",
        "test_contracts_config",
        "test_migration_artifacts",
        "test_runner_cli",
        "test_scoring_imaging",
        "test_transforms_backend",
        "test_verification_reporting",
    )
)
EXTERNAL_RUNTIME_PATHS_ANTIGRAVITY = frozenset(
    relative
    for relative in runtime_dependency_relative_paths_antigravity()
    if not relative.startswith(f"{EXP467_NAMESPACE_ANTIGRAVITY}/")
    and relative != "scratch/run_exp467_numba_calibration_resume_antigravity.py"
)
SCRATCH_PATHS_ANTIGRAVITY = frozenset(
    {
        "scratch/generate_exp467_codemap_antigravity.py",
        "scratch/run_exp467_numba_calibration_resume_antigravity.py",
    }
)
EXPECTED_PATHS_ANTIGRAVITY = (
    PRODUCTION_PATHS_ANTIGRAVITY
    | TEST_PATHS_ANTIGRAVITY
    | EXTERNAL_RUNTIME_PATHS_ANTIGRAVITY
    | SCRATCH_PATHS_ANTIGRAVITY
)


def _relative_antigravity(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


class Exp467CodeMapTestsAntigravity(unittest.TestCase):
    def test_exact_factual_closure_and_production_answers(self) -> None:
        paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
        self.assertEqual({_relative_antigravity(path) for path in paths}, EXPECTED_PATHS_ANTIGRAVITY)
        self.assertEqual(len(paths), 77)

        payload = build_codemap_payload_antigravity(paths, ast_callers_antigravity(paths))
        self.assertEqual(
            payload["scope"]["external_transitive_node_count"],
            len(EXTERNAL_RUNTIME_PATHS_ANTIGRAVITY),
        )
        nodes = {node["path"]: node for node in payload["nodes"]}
        self.assertEqual(set(nodes), EXPECTED_PATHS_ANTIGRAVITY)
        for node in nodes.values():
            self.assertEqual(set(node), {"path", "module", "callers", "effects", "tests"})
        for path in PRODUCTION_PATHS_ANTIGRAVITY:
            with self.subTest(path=path):
                self.assertTrue(nodes[path]["callers"])
                self.assertTrue(nodes[path]["effects"])
                self.assertTrue(nodes[path]["tests"])

    def test_critical_direct_import_edges_are_present(self) -> None:
        paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
        payload = build_codemap_payload_antigravity(paths, ast_callers_antigravity(paths))
        edges = {(edge["caller"], edge["callee"]) for edge in payload["edges"]}
        namespace = EXP467_NAMESPACE_ANTIGRAVITY
        cli = "scratch/run_exp467_numba_calibration_resume_antigravity.py"
        generator = "scratch/generate_exp467_codemap_antigravity.py"
        test_codemap = f"{namespace}/tests/test_codemap_antigravity.py"
        expected = {
            (f"{namespace}/config_antigravity.py", "experiments/exp449_reproducible_rebaseline_antigravity/config_antigravity.py"),
            (f"{namespace}/evaluation_antigravity.py", "experiments/exp450_resampling_image_ablation_antigravity/evaluation_antigravity.py"),
            (f"{namespace}/reporting_antigravity.py", "experiments/exp450_resampling_image_ablation_antigravity/reporting_antigravity.py"),
            (f"{namespace}/verifier_antigravity.py", "experiments/exp450_resampling_image_ablation_antigravity/contracts_antigravity.py"),
            (f"{namespace}/verifier_antigravity.py", "experiments/exp450_resampling_image_ablation_antigravity/runner_antigravity.py"),
            (f"{namespace}/verifier_antigravity.py", f"{namespace}/migration_antigravity.py"),
            (f"{namespace}/verifier_antigravity.py", f"{namespace}/science_artifacts_antigravity.py"),
            (f"{namespace}/scoring_antigravity.py", "paper/equipment_anomaly_detection/numba_calibration_antigravity.py"),
            (f"{namespace}/scoring_antigravity.py", f"{namespace}/hard_route_adapter_antigravity.py"),
            (f"{namespace}/scoring_antigravity.py", f"{namespace}/hard_sources_adapter_antigravity.py"),
            (f"{namespace}/scoring_antigravity.py", f"{namespace}/minirocket_adapter_antigravity.py"),
            (f"{namespace}/hard_route_adapter_antigravity.py", "paper/equipment_anomaly_detection/numba_calibration_antigravity.py"),
            (f"{namespace}/hard_sources_adapter_antigravity.py", "paper/equipment_anomaly_detection/numba_calibration_antigravity.py"),
            (f"{namespace}/minirocket_adapter_antigravity.py", "paper/equipment_anomaly_detection/numba_calibration_antigravity.py"),
            (f"{namespace}/science_artifacts_antigravity.py", "paper/equipment_anomaly_detection/numba_calibration_antigravity.py"),
            (f"{namespace}/scoring_antigravity.py", "paper/equipment_anomaly_detection/sources_paper/multirocket_hydra_paper.py"),
            (f"{namespace}/scoring_antigravity.py", "paper/equipment_anomaly_detection/sources_paper/rocket_paper.py"),
            (cli, "paper/equipment_anomaly_detection/catalog_paper.py"),
            (cli, "paper/equipment_anomaly_detection/data_paper.py"),
            (cli, f"{namespace}/backend_antigravity.py"),
            (cli, f"{namespace}/config_antigravity.py"),
            (cli, f"{namespace}/contracts_antigravity.py"),
            (cli, f"{namespace}/evaluation_antigravity.py"),
            (cli, f"{namespace}/migration_antigravity.py"),
            (cli, f"{namespace}/reporting_antigravity.py"),
            (cli, f"{namespace}/runner_antigravity.py"),
            (cli, f"{namespace}/verifier_antigravity.py"),
            (test_codemap, generator),
            (test_codemap, cli),
        }
        self.assertTrue(expected <= edges, sorted(expected - edges))

    def test_cli_code_closure_binds_all_exp467_production_modules(self) -> None:
        closure = _code_closure_antigravity()
        self.assertTrue(PRODUCTION_PATHS_ANTIGRAVITY <= set(closure))
        self.assertTrue(
            {
                "paper/equipment_anomaly_detection/data_paper.py",
                "paper/equipment_anomaly_detection/lineage_paper.py",
                "paper/equipment_anomaly_detection/sources_paper/canonical_paper.py",
                "paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py",
                "experiments/exp450_resampling_image_ablation_antigravity/artifacts_antigravity.py",
                "experiments/exp450_resampling_image_ablation_antigravity/evaluation_antigravity.py",
                "experiments/exp450_resampling_image_ablation_antigravity/reporting_antigravity.py",
                "experiments/exp450_resampling_image_ablation_antigravity/runner_antigravity.py",
            } <= set(closure)
        )
        self.assertEqual(tuple(closure), tuple(sorted(closure)))

    def test_runtime_closure_binds_package_initializers_and_lineage_resource(self) -> None:
        """Import-time package code and non-Python lineage authority are runtime inputs."""
        closure = _code_closure_antigravity()
        self.assertTrue({
            "experiments/paper_public_baselines/__init__.py",
            "experiments/paper_public_baselines/methods/__init__.py",
            "paper/equipment_anomaly_detection/lineage_lock_paper.json",
        } <= set(closure))

    def test_historical_serializer_worktree_mismatch_fails_closed(self) -> None:
        """The worktree copy may not silently stand in for fixed-Git executed bytes."""
        from experiments.exp467_numba_calibration_resume_antigravity import migration_antigravity
        with patch.object(migration_antigravity, "_historical_git_bytes_antigravity", return_value=b"changed"):
            with self.assertRaisesRegex(ValueError, "serializer"):
                migration_antigravity.validate_historical_serializer_worktree_antigravity(REPOSITORY_ROOT)

    def test_frozen_exp450_lock_fingerprints_are_unchanged(self) -> None:
        evidence = frozen_exp450_fingerprint_evidence_antigravity(REPOSITORY_ROOT)
        self.assertEqual(
            evidence,
            {
                "lock_path": "docs/codemap/codemap_exp450_antigravity.lock",
                "lock_sha256": FROZEN_EXP450_LOCK_SHA256_ANTIGRAVITY,
                "module_fingerprint_count": 36,
                "module_fingerprints_sha256": "29660a700804ac75bfa825e4ea49d8fd514d47526b7f752bc438d385cf0db402",
                "caller_edge_policy": "frozen_triplet_not_regenerated_after_exp467_callers",
            },
        )

    def test_lock_binds_the_standalone_and_embedded_artifacts(self) -> None:
        json_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp467_antigravity.json"
        html_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp467_antigravity.html"
        lock_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp467_antigravity.lock"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(
            lock.get("artifact_sha256"),
            {
                "docs/codemap/codemap_exp467_antigravity.html": sha256(
                    html_path.read_bytes()
                ).hexdigest(),
                "docs/codemap/codemap_exp467_antigravity.json": sha256(
                    json_path.read_bytes()
                ).hexdigest(),
            },
        )

    def test_triplet_is_fresh_and_embeds_the_same_json(self) -> None:
        self.assertEqual(codemap_main_antigravity(("--check",)), 0)
        json_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp467_antigravity.json"
        html_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp467_antigravity.html"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        html = html_path.read_text(encoding="utf-8")
        marker = '<script id="codemap-data" type="application/json">'
        embedded = html.split(marker, 1)[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(embedded), payload)

    def test_unknown_argument_is_rejected_without_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "usage"):
            codemap_main_antigravity(("--chec",))


if __name__ == "__main__":
    unittest.main()
