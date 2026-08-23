"""Behavioral contracts for the synchronized Exp451 code-map triplet."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import unittest

from scratch.generate_exp451_codemap_antigravity import (
    FROZEN_EXP450_LOCK_SHA256_ANTIGRAVITY,
    REPOSITORY_ROOT,
    ast_callers_antigravity,
    build_codemap_payload_antigravity,
    frozen_exp450_fingerprint_evidence_antigravity,
    main as codemap_main_antigravity,
    tracked_existing_paths_antigravity,
)
from scratch.run_exp451_verified_accelerated_resume_antigravity import (
    _code_closure_antigravity,
)


EXP451_NAMESPACE_ANTIGRAVITY = (
    "experiments/exp451_verified_accelerated_resume_antigravity"
)
PRODUCTION_PATHS_ANTIGRAVITY = frozenset(
    f"{EXP451_NAMESPACE_ANTIGRAVITY}/{name}_antigravity.py"
    for name in (
        "artifacts",
        "backend",
        "config",
        "contracts",
        "evaluation",
        "imaging",
        "migration",
        "reporting",
        "runner",
        "scoring",
        "transforms",
        "verifier",
    )
)
TEST_PATHS_ANTIGRAVITY = frozenset(
    f"{EXP451_NAMESPACE_ANTIGRAVITY}/tests/{name}_antigravity.py"
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
EXTERNAL_DIRECT_PATHS_ANTIGRAVITY = frozenset(
    {
        "experiments/exp449_reproducible_rebaseline_antigravity/config_antigravity.py",
        "experiments/exp450_resampling_image_ablation_antigravity/artifacts_antigravity.py",
        "experiments/exp450_resampling_image_ablation_antigravity/contracts_antigravity.py",
        "experiments/exp450_resampling_image_ablation_antigravity/evaluation_antigravity.py",
        "experiments/exp450_resampling_image_ablation_antigravity/reporting_antigravity.py",
        "experiments/exp450_resampling_image_ablation_antigravity/runner_antigravity.py",
        "paper/equipment_anomaly_detection/calibration_paper.py",
        "paper/equipment_anomaly_detection/catalog_paper.py",
        "paper/equipment_anomaly_detection/contracts_paper.py",
        "paper/equipment_anomaly_detection/data_paper.py",
        "paper/equipment_anomaly_detection/hard_route_paper.py",
        "paper/equipment_anomaly_detection/routing_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/imaging_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/knn_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/minirocket_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/multirocket_hydra_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/preprocessing_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/rocket_paper.py",
        "paper/equipment_anomaly_detection/sources_paper/stability_paper.py",
    }
)
SCRATCH_PATHS_ANTIGRAVITY = frozenset(
    {
        "scratch/generate_exp451_codemap_antigravity.py",
        "scratch/run_exp451_verified_accelerated_resume_antigravity.py",
    }
)
EXPECTED_PATHS_ANTIGRAVITY = (
    PRODUCTION_PATHS_ANTIGRAVITY
    | TEST_PATHS_ANTIGRAVITY
    | EXTERNAL_DIRECT_PATHS_ANTIGRAVITY
    | SCRATCH_PATHS_ANTIGRAVITY
)


def _relative_antigravity(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


class Exp451CodeMapTestsAntigravity(unittest.TestCase):
    def test_exact_factual_closure_and_production_answers(self) -> None:
        paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
        self.assertEqual({_relative_antigravity(path) for path in paths}, EXPECTED_PATHS_ANTIGRAVITY)
        self.assertEqual(len(paths), 42)

        payload = build_codemap_payload_antigravity(paths, ast_callers_antigravity(paths))
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
        namespace = EXP451_NAMESPACE_ANTIGRAVITY
        cli = "scratch/run_exp451_verified_accelerated_resume_antigravity.py"
        generator = "scratch/generate_exp451_codemap_antigravity.py"
        test_codemap = f"{namespace}/tests/test_codemap_antigravity.py"
        expected = {
            (f"{namespace}/config_antigravity.py", "experiments/exp449_reproducible_rebaseline_antigravity/config_antigravity.py"),
            (f"{namespace}/evaluation_antigravity.py", "experiments/exp450_resampling_image_ablation_antigravity/evaluation_antigravity.py"),
            (f"{namespace}/reporting_antigravity.py", "experiments/exp450_resampling_image_ablation_antigravity/reporting_antigravity.py"),
            (f"{namespace}/verifier_antigravity.py", "experiments/exp450_resampling_image_ablation_antigravity/artifacts_antigravity.py"),
            (f"{namespace}/verifier_antigravity.py", "experiments/exp450_resampling_image_ablation_antigravity/contracts_antigravity.py"),
            (f"{namespace}/verifier_antigravity.py", "experiments/exp450_resampling_image_ablation_antigravity/runner_antigravity.py"),
            (f"{namespace}/verifier_antigravity.py", f"{namespace}/migration_antigravity.py"),
            (f"{namespace}/scoring_antigravity.py", "paper/equipment_anomaly_detection/calibration_paper.py"),
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

    def test_cli_code_closure_binds_all_exp451_production_modules(self) -> None:
        closure = _code_closure_antigravity()
        self.assertTrue(PRODUCTION_PATHS_ANTIGRAVITY <= set(closure))
        self.assertIn("paper/equipment_anomaly_detection/data_paper.py", closure)

    def test_frozen_exp450_lock_fingerprints_are_unchanged(self) -> None:
        evidence = frozen_exp450_fingerprint_evidence_antigravity(REPOSITORY_ROOT)
        self.assertEqual(
            evidence,
            {
                "lock_path": "docs/codemap/codemap_exp450_antigravity.lock",
                "lock_sha256": FROZEN_EXP450_LOCK_SHA256_ANTIGRAVITY,
                "module_fingerprint_count": 36,
                "module_fingerprints_sha256": "29660a700804ac75bfa825e4ea49d8fd514d47526b7f752bc438d385cf0db402",
                "caller_edge_policy": "frozen_triplet_not_regenerated_after_exp451_callers",
            },
        )

    def test_lock_binds_the_standalone_and_embedded_artifacts(self) -> None:
        json_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp451_antigravity.json"
        html_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp451_antigravity.html"
        lock_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp451_antigravity.lock"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(
            lock.get("artifact_sha256"),
            {
                "docs/codemap/codemap_exp451_antigravity.html": sha256(
                    html_path.read_bytes()
                ).hexdigest(),
                "docs/codemap/codemap_exp451_antigravity.json": sha256(
                    json_path.read_bytes()
                ).hexdigest(),
            },
        )

    def test_triplet_is_fresh_and_embeds_the_same_json(self) -> None:
        self.assertEqual(codemap_main_antigravity(("--check",)), 0)
        json_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp451_antigravity.json"
        html_path = REPOSITORY_ROOT / "docs/codemap/codemap_exp451_antigravity.html"
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
