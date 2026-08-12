"""Adversarial Task 8 lifecycle tests against an actual Task 2 SQLite fixture."""
from __future__ import annotations

from hashlib import sha256
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _blob(values: tuple[float, ...]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE datasets(id INTEGER PRIMARY KEY,name TEXT UNIQUE,series_length INTEGER);"
        "CREATE TABLE instances(id INTEGER PRIMARY KEY,dataset_id INTEGER,split TEXT,"
        "instance_index INTEGER,label TEXT,values_blob BLOB);"
    )
    connection.executemany("INSERT INTO datasets VALUES(?,?,?)", ((1, "alpha", 2), (2, "beta", 2)))
    rows = []
    row_id = 1
    for dataset_id in (1, 2):
        for split, label in (("TRAIN", "0"), ("TEST", "1")):
            for index in range(2):
                rows.append((row_id, dataset_id, split, index, label, _blob((dataset_id + index, dataset_id + index + .5))))
                row_id += 1
        rows.append((row_id, dataset_id, "TRAIN", 9, "1", _blob((9, 9)))); row_id += 1
    connection.executemany("INSERT INTO instances VALUES(?,?,?,?,?,?)", rows)
    connection.commit(); connection.close()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"


class Task8LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        # Use the physical /private path: raw /var is deliberately rejected
        # by Task 8's no-symlink-root authority on macOS.
        self.base = Path(self.temporary.name).resolve()
        self.db = self.base / "fixture.sqlite"
        _database(self.db)
        from paper.equipment_anomaly_detection.catalog_paper import build_catalog
        self.catalog = build_catalog(self.db, mode="smoke", smoke_datasets=("alpha", "beta"))
        self.oracle = self.base / "oracle_antigravity.json"
        policy_coordinates = ((("alpha", 0),), (("alpha", 1),), (("beta", 0),), (("beta", 1),))
        oracle_value = {"schema": "paper-task8-coordinate-oracle.v1", "policies": [
            {"policy_id": name, "coordinates": [[dataset, index] for dataset, index in coordinates]}
            for name, coordinates in zip(("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable"), policy_coordinates, strict=True)
        ]}
        self.oracle.write_bytes(_canonical(oracle_value))
        self.acceptance = self.base / "acceptance_antigravity.json"
        self._write_acceptance(policy_coordinates)
        self.root = self.base / "run_antigravity"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_acceptance(self, policies: tuple[tuple[tuple[str, int], ...], ...]) -> None:
        from paper.equipment_anomaly_detection.catalog_paper import coordinate_digest
        records = [
            {"policy_id": name, "order": order, "coordinates": [[dataset, index] for dataset, index in coordinates],
             "coordinate_count": len(coordinates), "coordinate_sha256": coordinate_digest(coordinates)}
            for order, (name, coordinates) in enumerate(zip(("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable"), policies, strict=True))
        ]
        bare = {"schema": "paper-task8-acceptance.v1", "catalog_sha256": self.catalog["catalog_sha256"],
                "oracle": {"identifier": "fixture-oracle", "version": "v1", "relative_path": self.oracle.name,
                           "sha256": sha256(self.oracle.read_bytes()).hexdigest()}, "policies": records}
        self.acceptance.write_bytes(_canonical({**bare, "acceptance_sha256": sha256(_canonical(bare)).hexdigest()}))

    def config(self):
        from paper.equipment_anomaly_detection.runner_paper import PipelineConfig
        return PipelineConfig(self.db, self.root, "smoke", ("alpha", "beta"), self.acceptance)

    def test_run_completes_only_real_preflight_and_prepare_then_fails_acceleration(self) -> None:
        """Replacing the unavailable-stage fail-closed branch must make this test fail."""
        from paper.equipment_anomaly_detection.runner_paper import PipelineNotImplementedError, run_pipeline, verify_completed
        with self.assertRaisesRegex(PipelineNotImplementedError, "stage_not_implemented"):
            run_pipeline(self.config())
        state = verify_completed(self.config())
        self.assertEqual(state, {"status": "failed", "completed_stages": ["preflight", "prepare"], "failed_stage": "acceleration", "next_stage": "acceleration"})

    def test_arbitrary_handler_cannot_complete_unregistered_acceleration(self) -> None:
        """Reintroducing a generic builder injection must make this authority check fail."""
        from paper.equipment_anomaly_detection.runner_paper import PipelineNotImplementedError, run_pipeline
        with self.assertRaises(TypeError):
            run_pipeline(self.config(), test_stage_builders={"acceleration": lambda *_: None})
        with self.assertRaises(PipelineNotImplementedError):
            run_pipeline(self.config())

    def test_catalog_recomputation_rejects_fabricated_and_gapped_identities(self) -> None:
        """Trusting supplied catalog digests or positions would make these checks pass."""
        from paper.equipment_anomaly_detection.catalog_paper import verify_catalog
        forged = json.loads(_canonical(self.catalog)); forged["datasets"][0]["test"]["row_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "catalog"): verify_catalog(forged, self.db, "smoke", ("alpha", "beta"))
        connection = sqlite3.connect(self.db); connection.execute("UPDATE instances SET instance_index=3 WHERE dataset_id=1 AND split='TEST' AND instance_index=1"); connection.commit(); connection.close()
        with self.assertRaisesRegex(ValueError, "contiguous"): verify_catalog(self.catalog, self.db, "smoke", ("alpha", "beta"))

    def test_verify_rejects_modified_database_and_internal_namespace_pollution(self) -> None:
        """Dropping runtime recomputation or namespace checks would make these pass."""
        from paper.equipment_anomaly_detection.runner_paper import PipelineNotImplementedError, PipelineStateError, run_pipeline, verify_completed
        with self.assertRaises(PipelineNotImplementedError): run_pipeline(self.config())
        connection = sqlite3.connect(self.db); connection.execute("UPDATE instances SET values_blob=? WHERE id=1", (_blob((99, 99)),)); connection.commit(); connection.close()
        with self.assertRaises(PipelineStateError): verify_completed(self.config())
        self.root = self.base / "polluted_antigravity"
        self.catalog = __import__("paper.equipment_anomaly_detection.catalog_paper", fromlist=["build_catalog"]).build_catalog(self.db, mode="smoke", smoke_datasets=("alpha", "beta"))
        self._write_acceptance(((("alpha", 0),), (("alpha", 1),), (("beta", 0),), (("beta", 1),)))
        with self.assertRaises(PipelineNotImplementedError): run_pipeline(self.config())
        (self.root / "stage_artifacts_paper" / "prepare" / "junk_paper.json").write_text("{}\n")
        with self.assertRaisesRegex(PipelineStateError, "pollution"): verify_completed(self.config())

    def test_symlink_and_legacy_roots_fail_closed(self) -> None:
        """Allowing a redirected root or old journal would make this test pass."""
        from paper.equipment_anomaly_detection.runner_paper import PipelineStateError, run_pipeline
        link = self.base / "link_antigravity"; link.symlink_to(self.base, target_is_directory=True)
        self.root = link / "escape"
        with self.assertRaisesRegex(ValueError, "symlink"): run_pipeline(self.config())
        self.root = self.base / "legacy_antigravity"; self.root.mkdir(); (self.root / "pipeline_state_paper.json").write_text("{}")
        with self.assertRaisesRegex(PipelineStateError, "legacy_root_requires_new_root"): run_pipeline(self.config())

    def test_empty_policy_is_allowed_only_with_empty_digest_and_parity_reopens_oracle(self) -> None:
        """Accepting false count/digest/oracle evidence would make this test pass."""
        from paper.equipment_anomaly_detection.artifacts_paper import verify_parity
        from paper.equipment_anomaly_detection.catalog_paper import coordinate_digest
        self._write_acceptance(((), (("alpha", 1),), (("beta", 0),), (("beta", 1),)))
        acceptance = __import__("paper.equipment_anomaly_detection.catalog_paper", fromlist=["load_acceptance"]).load_acceptance(self.acceptance, self.catalog)
        self.assertEqual(acceptance["policies"][0]["coordinate_sha256"], coordinate_digest(()))
        self.root.mkdir()
        with self.assertRaises(ValueError): verify_parity(self.root, self.catalog, acceptance)


if __name__ == "__main__":
    unittest.main()
