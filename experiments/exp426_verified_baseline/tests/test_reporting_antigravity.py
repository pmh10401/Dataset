"""Frozen, label-free publication tests for Exp426."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp426_verified_baseline.contracts_antigravity import (
    Exp426DatasetRecord,
)
from experiments.exp426_verified_baseline.reporting_antigravity import (
    _PublicationExpectations,
    publish_exp426,
    verify_exp426_publication,
)
from experiments.exp426_verified_baseline.source_antigravity import (
    Exp424SourceClassification,
)
from experiments.exp426_verified_baseline.tests import (
    test_runtime_antigravity as runtime_fixtures,
)
from experiments.exp426_verified_baseline.runtime_antigravity import run_exp426_new


class Exp426ReportingTests(unittest.TestCase):
    expectations = _PublicationExpectations(3, 2, 1)

    @staticmethod
    def _fixture(parent: Path):
        datasets = ("dataset-00", "dataset-01", "dataset-02")
        records = (
            Exp426DatasetRecord(
                datasets[0], "exp424_import", "scored", ((datasets[0], 1),),
                "b" * 64, "c" * 64, None, None,
                "verified_exp424_nonterminal",
            ),
            Exp426DatasetRecord(
                datasets[1], "exp424_import", "scored", ((datasets[1], 2),),
                "d" * 64, "e" * 64, None, None,
                "verified_exp424_nonterminal",
            ),
        )
        classifications = (
            Exp424SourceClassification(
                datasets[0], "b" * 64, "c" * 64, records[0], None,
                3, (64,), 4, (64,),
            ),
            Exp424SourceClassification(
                datasets[1], "d" * 64, "e" * 64, records[1], None,
                3, (64,), 4, (64,),
            ),
            Exp424SourceClassification(
                datasets[2], "f" * 64, "0" * 64, None,
                "oof_identity_insufficient", 0, (), 4, (64,),
            ),
        )
        provenance = runtime_fixtures.Exp426RuntimeTests.provenance(datasets)
        context = runtime_fixtures.Exp426RuntimeTests.checkpoint_context(datasets)
        output, checkpoint = parent / "output", parent / "checkpoint"
        runtime = runtime_fixtures._FakeRuntime(classifications)
        database = runtime_fixtures._FakeDatabase(
            runtime_fixtures.Exp426RuntimeTests.trains((datasets[2],))
        )
        run_exp426_new(
            output, checkpoint, provenance, context, object(), database, object(),
            _dependencies=runtime.dependencies(),
        )
        return datasets, provenance, context, output, checkpoint

    @staticmethod
    def _tree_state(root: Path) -> dict[str, tuple[bytes, int, int]]:
        return {
            str(path.relative_to(root)): (
                path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns,
            )
            for path in sorted(root.rglob("*")) if path.is_file()
        }

    @staticmethod
    def _json_values(publication: Path):
        for path in sorted(publication.iterdir()):
            if path.suffix == ".json":
                yield json.loads(path.read_text())
            elif path.suffix == ".jsonl":
                for line in path.read_text().splitlines():
                    yield json.loads(line)

    def test_publish_exact_population_governance_hashes_and_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            datasets, provenance, context, output, checkpoint = self._fixture(parent)
            result = publish_exp426(
                output, checkpoint, provenance, context,
                _expectations=self.expectations,
            )
            publication = output / "publication_antigravity"
            self.assertEqual(result.publication, publication)
            manifest = result.manifest
            self.assertEqual(manifest["completed_datasets"], 3)
            self.assertEqual(manifest["imported_datasets"], 2)
            self.assertEqual(manifest["verified_train_abstentions"], 1)
            self.assertEqual(manifest["failed_datasets"], 0)
            self.assertTrue(manifest["review_only"])
            self.assertFalse(manifest["operational_alert_eligible"])
            self.assertFalse(manifest["contains_test_labels"])
            self.assertFalse(manifest["test_labels_opened"])
            self.assertEqual(manifest["workers"], 12)

            predictions = [
                json.loads(line) for line in
                (publication / "prediction_antigravity.jsonl").read_text().splitlines()
            ]
            self.assertEqual(predictions, [
                {"dataset": datasets[0], "instance_index": 1},
                {"dataset": datasets[1], "instance_index": 2},
            ])
            self.assertEqual(predictions, sorted(
                predictions, key=lambda item: (item["dataset"], item["instance_index"]),
            ))

            artifact_map = json.loads(
                (publication / "artifact_sha256_antigravity.json").read_text()
            )["artifact_sha256"]
            for filename, digest in artifact_map.items():
                self.assertEqual(
                    sha256((publication / filename).read_bytes()).hexdigest(), digest,
                )
            for filename, digest in manifest["artifact_sha256"].items():
                self.assertEqual(
                    sha256((publication / filename).read_bytes()).hexdigest(), digest,
                )

            checkpoint_progress = json.loads(
                (checkpoint / "progress.json").read_text()
            )
            for dataset, digest in checkpoint_progress["record_digests"].items():
                filename = sha256(dataset.encode()).hexdigest() + ".json"
                self.assertEqual(
                    sha256((checkpoint / "datasets" / filename).read_bytes()).hexdigest(),
                    digest,
                )

            allowed = {"contains_test_labels", "test_labels_opened"}
            forbidden = (
                "label", "target", "groundtruth", "confusion", "precision",
                "recall", "f1", "anomalylocation", "familytest",
            )

            def scan(value: object) -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        normalized = "".join(character for character in key.lower() if character.isalnum())
                        if key not in allowed:
                            self.assertFalse(any(token in normalized for token in forbidden), key)
                        scan(child)
                elif isinstance(value, list):
                    for child in value:
                        scan(child)

            for value in self._json_values(publication):
                scan(value)

            before = self._tree_state(parent)
            verified = verify_exp426_publication(
                output, checkpoint, provenance, context,
                _expectations=self.expectations,
            )
            again = publish_exp426(
                output, checkpoint, provenance, context,
                _expectations=self.expectations,
            )
            after = self._tree_state(parent)
            self.assertEqual(verified.manifest, manifest)
            self.assertEqual(again.manifest, manifest)
            self.assertEqual(before, after)

    def test_population_mismatch_writes_no_publication(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            _datasets, provenance, context, output, checkpoint = self._fixture(parent)
            with self.assertRaisesRegex(ValueError, "population"):
                publish_exp426(
                    output, checkpoint, provenance, context,
                    _expectations=_PublicationExpectations(4, 3, 1),
                )
            self.assertFalse((output / "publication_antigravity").exists())

    def test_tampered_prediction_fails_independent_verification(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            _datasets, provenance, context, output, checkpoint = self._fixture(parent)
            publish_exp426(
                output, checkpoint, provenance, context,
                _expectations=self.expectations,
            )
            prediction = output / "publication_antigravity" / "prediction_antigravity.jsonl"
            prediction.write_bytes(prediction.read_bytes() + b'{}\n')
            with self.assertRaisesRegex(ValueError, "SHA256|prediction"):
                verify_exp426_publication(
                    output, checkpoint, provenance, context,
                    _expectations=self.expectations,
                )


if __name__ == "__main__":
    unittest.main()
