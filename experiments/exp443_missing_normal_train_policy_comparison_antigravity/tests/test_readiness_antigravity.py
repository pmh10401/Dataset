"""Behavior-first tests for the Exp443 Task 8 readiness boundary."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import FROZEN_CONFIG


def _source_fixture() -> dict[str, object]:
    datasets = tuple(f"family_normal_{index:04d}" for index in range(1117))
    affected = datasets[:19]
    hashes = {
        name: "a" * 64
        for name in (
            "database", "exp249_prediction", "exp249_prediction_manifest",
            "exp249_artifact_manifest", "exp249_cache_manifest", "exp249_audit",
            "exp291_routing", "exp291_manifest", "exp414_repair_audit",
            "exp414_manifest", "exp404_hard", "exp404_manifest",
        )
    }
    hashes["database"] = FROZEN_CONFIG.database_sha256
    return {
        "datasets": datasets,
        "affected_datasets": affected,
        "affected_families": ("family_0", "family_1", "family_2", "family_3"),
        "affected_rows": 164,
        "nonfinite_scalars": 983,
        "all_missing_rows": 0,
        "automatic_hard_count": 2218,
        "exact_normal_count": 35,
        "standard_count": 806,
        "priority_count": 31,
        "routing_algebra": (2116, 137, 35, 2218),
        "source_hashes": hashes,
        "label_access_count": 0,
    }


class ReadinessAntigravityTests(unittest.TestCase):
    def test_fixture_readiness_writes_exact_label_free_identity_and_no_score_import(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.readiness_antigravity import (
            ReadinessDependenciesAntigravity,
            create_readiness_antigravity,
        )

        source = _source_fixture()
        commit = "a" * 40
        with tempfile.TemporaryDirectory(prefix="exp443_task8_readiness_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            dependencies = ReadinessDependenciesAntigravity(
                clean_head=lambda: commit,
                verify_configuration=lambda: {"sha256": "b" * 64, "approved": True},
                verify_sources=lambda: source,
                verify_environment=lambda: {"logical_cpu_count": 12, "outer_workers": 11, "inner_threads": 1},
            )
            payload = create_readiness_antigravity(root, reviewed_commit=commit, dependencies=dependencies)
            self.assertEqual(payload["schema"], "exp443.readiness.v1")
            self.assertEqual(payload["reviewed_commit"], commit)
            self.assertEqual(payload["label_access_count"], 0)
            self.assertEqual(payload["dataset_count"], 1117)
            self.assertEqual(payload["affected_dataset_count"], 19)
            self.assertEqual(payload["workers"], 11)
            self.assertFalse(payload["contains_test_labels"])
            self.assertFalse(payload["contains_evaluation_metrics"])
            self.assertTrue((root / "readiness_antigravity.json").is_file())
            self.assertEqual(json.loads((root / "readiness_antigravity.json").read_text()), payload)

    def test_readiness_rejects_dirty_head_and_existing_output_root(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.readiness_antigravity import (
            ReadinessDependenciesAntigravity,
            create_readiness_antigravity,
        )

        source = _source_fixture()
        with tempfile.TemporaryDirectory(prefix="exp443_task8_readiness_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            root.mkdir()
            dependencies = ReadinessDependenciesAntigravity(
                clean_head=lambda: "a" * 40,
                verify_configuration=lambda: {"sha256": "b" * 64, "approved": True},
                verify_sources=lambda: source,
                verify_environment=lambda: {"logical_cpu_count": 12, "outer_workers": 11, "inner_threads": 1},
            )
            with self.assertRaisesRegex(ValueError, "output root"):
                create_readiness_antigravity(root, reviewed_commit="a" * 40, dependencies=dependencies)

            dirty = ReadinessDependenciesAntigravity(
                clean_head=lambda: "working-tree-dirty",
                verify_configuration=lambda: {"sha256": "b" * 64, "approved": True},
                verify_sources=lambda: source,
                verify_environment=lambda: {"logical_cpu_count": 12, "outer_workers": 11, "inner_threads": 1},
            )
            with self.assertRaisesRegex(ValueError, "clean"):
                create_readiness_antigravity(Path(directory) / "other_antigravity", reviewed_commit="a" * 40, dependencies=dirty)

    def test_readiness_rejects_labels_and_wrong_topology(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.readiness_antigravity import (
            ReadinessDependenciesAntigravity,
            create_readiness_antigravity,
        )

        for mutation, expected in (
            ({"label_access_count": 1}, "label"),
            ({"contains_test_labels": True}, "label"),
        ):
            source = _source_fixture()
            source.update(mutation)
            with tempfile.TemporaryDirectory(prefix="exp443_task8_readiness_antigravity_") as directory:
                dependencies = ReadinessDependenciesAntigravity(
                    clean_head=lambda: "a" * 40,
                    verify_configuration=lambda: {"sha256": "b" * 64, "approved": True},
                    verify_sources=lambda: source,
                    verify_environment=lambda: {"logical_cpu_count": 11, "outer_workers": 10, "inner_threads": 1},
                )
                with self.subTest(mutation=mutation):
                    with self.assertRaisesRegex(ValueError, expected):
                        create_readiness_antigravity(Path(directory) / "run_antigravity", reviewed_commit="a" * 40, dependencies=dependencies)


if __name__ == "__main__":
    unittest.main()
