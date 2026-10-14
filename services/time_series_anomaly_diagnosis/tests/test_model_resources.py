from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from services.time_series_anomaly_diagnosis.dataset_repository import open_source_database
from services.time_series_anomaly_diagnosis.model_resources import load_model_resources
from services.time_series_anomaly_diagnosis.tools.build_service_resources import (
    build_service_resources,
)

RESOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "resources"
RESOURCE_BUILDER = Path(__file__).resolve().parents[1] / "tools" / "build_service_resources.py"


class ModelResourcesTest(unittest.TestCase):
    def test_resource_builder_help_runs_directly(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [sys.executable, str(RESOURCE_BUILDER), "--help"],
                cwd=temporary_directory,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--review-sha256", completed.stdout)

    def test_loads_only_from_service_resource_directory(self) -> None:
        resources = load_model_resources(RESOURCE_DIRECTORY)
        self.assertEqual(len(resources.route_by_dataset), 1117)
        self.assertEqual(len(resources.recovery_datasets), 155)
        self.assertEqual(len(resources.review_evidence), 1117)

    def test_rejects_changed_service_resource(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            resource_directory = Path(temporary_directory)
            for source in RESOURCE_DIRECTORY.iterdir():
                (resource_directory / source.name).write_bytes(source.read_bytes())
            (resource_directory / "routing_catalog.json").write_text("{}")

            with self.assertRaisesRegex(ValueError, "changed"):
                load_model_resources(resource_directory)

    def test_source_database_path_is_required(self) -> None:
        with self.assertRaises(TypeError):
            open_source_database()

    def test_builder_is_idempotent_and_excludes_its_output(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            source = repository_root / "source"
            output = repository_root / "service" / "resources"
            source.mkdir()
            payloads = {
                "route.json": b'{"route": true}',
                "recovery.json": b'{"recovery": true}',
                "review.jsonl": (
                    b'{"dataset":"Sample","review_only":true,'
                    b'"contains_test_labels":false,"contains_evaluation_metrics":false,'
                    b'"review_priority_1_indices":[7],"review_priority_2_indices":[],'
                    b'"review_priority_3_indices":[9],'
                    b'"review_support_by_index":{"7":["structural"],"9":[]}}\n'
                ),
            }
            for name, payload in payloads.items():
                (source / name).write_bytes(payload)
            route_sha256 = sha256(payloads["route.json"]).hexdigest()
            recovery_sha256 = sha256(payloads["recovery.json"]).hexdigest()
            review_sha256 = sha256(payloads["review.jsonl"]).hexdigest()
            review_manifest = {
                "contains_evaluation_metrics": False,
                "contains_test_labels": False,
                "prediction_sha256": review_sha256,
                "review_only": True,
                "review_tier_policy_frozen_before_evaluation": True,
            }
            review_manifest_bytes = json.dumps(review_manifest, sort_keys=True).encode("utf-8")
            (source / "review_manifest.json").write_bytes(review_manifest_bytes)
            review_manifest_sha256 = sha256(review_manifest_bytes).hexdigest()

            for _ in range(2):
                build_service_resources(
                    repository_root,
                    output,
                    route_sha256,
                    recovery_sha256,
                    review_sha256,
                    review_manifest_sha256,
                )

            manifest = json.loads((output / "model_manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["schema"], "service_model_resources_v2")
            self.assertEqual(
                set(manifest["files"]),
                {"recovery_catalog.json", "review_evidence.json", "routing_catalog.json"},
            )
            review_resource = json.loads((output / "review_evidence.json").read_text("utf-8"))
            self.assertEqual(
                review_resource,
                {
                    "datasets": [
                        {
                            "dataset": "Sample",
                            "priority_1": [7],
                            "priority_2": [],
                            "priority_3": [9],
                            "support": {"7": ["structural"], "9": []},
                        }
                    ],
                    "schema": "review_evidence_v1",
                },
            )
            serialized = (output / "review_evidence.json").read_text("utf-8").lower()
            for forbidden in ("test_label", "evaluation", "experiment", "/users/"):
                self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
