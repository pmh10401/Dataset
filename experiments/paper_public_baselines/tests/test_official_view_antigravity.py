from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.paper_public_baselines.official_view_antigravity import (
    freeze_single_method_view_antigravity,
)
from experiments.paper_public_baselines.runner import ScoreRunConfig, run_methods
from experiments.paper_public_baselines.tests.test_runner import (
    CountingScorer,
    _build_database,
)


class OfficialSingleMethodViewTests(unittest.TestCase):
    def _source(self, directory: str) -> Path:
        base = Path(directory)
        database = base / "toy.sqlite"
        _build_database(database)
        config = ScoreRunConfig(
            db_path=database,
            run_root=base / "source",
            datasets=("toy",),
            methods=("deep_svdd_1d", "drocc_1d"),
            workers=1,
            database_sha256=hashlib.sha256(database.read_bytes()).hexdigest(),
            epochs=50,
        )
        run_methods(config, scorer=CountingScorer())
        return config.run_root

    def test_single_method_view_preserves_shard_bytes_and_freezes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            destination = Path(directory) / "destination"
            source_shard = next((source / "shards/deep_svdd_1d").glob("*.json"))

            manifest = freeze_single_method_view_antigravity(
                source, destination, "deep_svdd_1d"
            )

            destination_shard = (
                destination / "shards/deep_svdd_1d" / source_shard.name
            )
            provenance = json.loads(
                destination.joinpath(
                    "derived_view_provenance_antigravity.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest.pair_count, 1)
            self.assertEqual(destination_shard.read_bytes(), source_shard.read_bytes())
            self.assertEqual(provenance["method"], "deep_svdd_1d")
            self.assertEqual(provenance["dataset_count"], 1)
            self.assertFalse((destination / "shards/drocc_1d").exists())

    def test_single_method_view_rejects_tampered_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            destination = Path(directory) / "destination"
            metadata = next((source / "shards/deep_svdd_1d").glob("*.meta"))
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            payload["shard_sha256"] = "0" * 64
            metadata.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source shard SHA256"):
                freeze_single_method_view_antigravity(
                    source, destination, "deep_svdd_1d"
                )


if __name__ == "__main__":
    unittest.main()
