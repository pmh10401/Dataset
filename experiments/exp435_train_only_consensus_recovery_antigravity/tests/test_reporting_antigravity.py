"""TDD coverage for Exp435 review-only publication replay."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.exp435_train_only_consensus_recovery_antigravity.checkpoint_antigravity import (
    Exp435CheckpointStore,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.contracts_antigravity import (
    Exp435Candidate,
    Exp435DatasetResult,
    InstanceScoreEvidence,
    BackendDecision,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.progress_antigravity import (
    ProgressWriter,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp435_train_only_consensus_recovery_antigravity.reporting_antigravity import (
    CandidateBudgetExceeded,
    FrozenPublication,
    _PublicationExpectations,
    _exclusive_rename,
    inspect_publication,
    publish_exp435,
)


def _context() -> dict[str, object]:
    return {
        "schema": "fixture.context.v1",
        "run_identity": FROZEN_CONFIG.run_identity,
        "datasets": ["Demo"],
        "test_sizes": {"Demo": 101},
        "database_sha256": "d" * 64,
        "code_commit": "f" * 40,
        "runtime_versions": {"python": "test"},
        "backend_decision": BackendDecision.cpu_fallback("fixture").to_dict(),
    }


def _result(dataset: str = "Demo", *, candidates: int = 0) -> Exp435DatasetResult:
    evidence = []
    candidate_rows = []
    for index in range(candidates):
        item = InstanceScoreEvidence.fixture(
            dataset=dataset, instance_index=index, length=16,
            prototype_abnormal=True, gradient_abnormal=True,
        )
        evidence.append(item)
        candidate_rows.append(Exp435Candidate(dataset, index, item.length, item.evidence_digest))
    return Exp435DatasetResult(
        dataset=dataset,
        status="scored",
        test_size=101,
        candidates=tuple(candidate_rows),
        evidence=tuple(evidence),
    )


def _runtime(root: Path, checkpoint: Path, *, completed: int = 1, candidates: int = 0) -> Exp435CheckpointStore:
    store = Exp435CheckpointStore.create(checkpoint, _context())
    digest = store.commit(_result(candidates=candidates))
    writer = ProgressWriter.create(
        root,
        run_identity=store.run_identity,
        checkpoint=checkpoint,
        total_datasets=1,
        backend="cpu",
        context_fingerprint=store.context_fingerprint,
        heartbeat_seconds=None,
    )
    writer.update(
        stage="idle", current_dataset=None, dataset_metadata=None,
        completed=1, success=1, abstention=0, error=0,
        candidate_count=candidates,
    )
    writer.append_event({
        "kind": "dataset_complete", "dataset": "Demo", "record_sha256": digest,
    })
    writer.mark_complete(candidate_count=candidates)
    writer.close()
    return store


class _BudgetStore:
    """A deliberately over-budget read-only checkpoint double."""

    def __init__(self, context: dict[str, object], result: Exp435DatasetResult):
        self.root = Path(context["_root"]) if "_root" in context else Path(".")
        self.context = context
        self.context_fingerprint = "0" * 64
        self.run_identity = FROZEN_CONFIG.run_identity
        self.datasets = (result.dataset,)
        self._result = result
        self.record_set_sha256 = "1" * 64

    def inspect(self):
        return SimpleNamespace(total=1, completed=1)

    def completed_digests(self):
        return {self._result.dataset: "2" * 64}

    def load_record(self, _dataset):
        return self._result


def _runtime_empty(root: Path, checkpoint: Path, store: _BudgetStore) -> None:
    checkpoint.mkdir()
    store.root = checkpoint
    writer = ProgressWriter.create(
        root,
        run_identity=store.run_identity,
        checkpoint=checkpoint,
        total_datasets=1,
        backend="cpu",
        context_fingerprint=store.context_fingerprint,
        heartbeat_seconds=None,
    )
    writer.update(
        stage="idle", current_dataset=None, dataset_metadata=None,
        completed=1, success=1, abstention=0, error=0,
        candidate_count=store._result.candidate_count,
    )
    writer.append_event({
        "kind": "dataset_complete", "dataset": "Demo", "record_sha256": "2" * 64,
    })
    writer.mark_complete(candidate_count=store._result.candidate_count)
    writer.close()


class Exp435ReportingTests(unittest.TestCase):
    expectations = _PublicationExpectations(total=1)

    def test_publication_adapters_reject_nonfrozen_store_identity(self) -> None:
        """Publication must not mint or replay an artifact for another run."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _BudgetStore(_context(), _result())
            store.run_identity = "exp435-fixture"
            _runtime_empty(root / "output", root / "checkpoint", store)
            with self.assertRaisesRegex(ValueError, "run identity"):
                publish_exp435(store, root / "output", _expectations=self.expectations)
            self.assertFalse((root / "output" / "publication_antigravity").exists())

    def test_publication_replays_checkpoint_without_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _runtime(root / "output", root / "checkpoint")
            publication = publish_exp435(
                store,
                root / "output",
                _expectations=self.expectations,
            )
            replay = inspect_publication(
                root / "output",
                store,
                _expectations=self.expectations,
            )
            self.assertIsInstance(publication, FrozenPublication)
            self.assertEqual(publication.prediction_sha256, replay.prediction_sha256)
            self.assertFalse(publication.manifest["test_labels_opened"])
            self.assertTrue(publication.manifest["review_only"])

    def test_budget_failure_never_creates_publication(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _BudgetStore(_context(), _result(candidates=101))
            _runtime_empty(root / "output", root / "checkpoint", store)
            with self.assertRaises(CandidateBudgetExceeded):
                publish_exp435(store, root / "output", _expectations=self.expectations)
            self.assertFalse((root / "output" / "publication_antigravity").exists())

    def test_hash_map_excludes_itself_and_is_canonical(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _runtime(root / "output", root / "checkpoint")
            publication = publish_exp435(store, root / "output", _expectations=self.expectations)
            mapping = json.loads((publication.publication / "artifact_sha256_antigravity.json").read_bytes())
            self.assertEqual(mapping["self_excluded"], "artifact_sha256_antigravity.json")
            self.assertNotIn("artifact_sha256_antigravity.json", mapping["artifact_sha256"])
            for name, digest in mapping["artifact_sha256"].items():
                self.assertEqual(sha256((publication.publication / name).read_bytes()).hexdigest(), digest)

    def test_publication_requires_checkpoint_complete_progress_state(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _runtime(root / "output", root / "checkpoint")
            progress = root / "output" / "progress_antigravity.json"
            payload = json.loads(progress.read_bytes())
            payload["state"] = "complete_review_only"
            from experiments.exp403_408.artifact_io import canonical_json_bytes

            progress.write_bytes(canonical_json_bytes(payload))
            with self.assertRaises(ValueError):
                publish_exp435(store, root / "output", _expectations=self.expectations)

    def test_publication_requires_checkpoint_counter_conservation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _runtime(root / "output", root / "checkpoint")
            progress = root / "output" / "progress_antigravity.json"
            payload = json.loads(progress.read_bytes())
            payload["success"] = 0
            payload["abstention"] = 1
            payload["completed"] = 1
            payload["completed_datasets"] = 1
            from experiments.exp403_408.artifact_io import canonical_json_bytes

            progress.write_bytes(canonical_json_bytes(payload))
            with self.assertRaises(ValueError):
                publish_exp435(store, root / "output", _expectations=self.expectations)

    def test_publication_does_not_replace_destination_that_appears_during_publish(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _runtime(root / "output", root / "checkpoint")
            destination = root / "output" / "publication_antigravity"
            original = _exclusive_rename

            def race(staging: Path, target: Path) -> None:
                target.mkdir()
                original(staging, target)

            with patch(
                "experiments.exp435_train_only_consensus_recovery_antigravity.reporting_antigravity._exclusive_rename",
                race,
            ):
                with self.assertRaises((OSError, ValueError, RuntimeError)):
                    publish_exp435(store, root / "output", _expectations=self.expectations)
            self.assertTrue(destination.is_dir())
            self.assertEqual(tuple(destination.iterdir()), ())
            self.assertEqual(tuple(root.joinpath("output").glob(".publication_antigravity.*.tmp")), ())

    def test_manifest_status_and_hash_map_bind_all_terminal_runtime_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _runtime(root / "output", root / "checkpoint")
            publication = publish_exp435(store, root / "output", _expectations=self.expectations)
            runtime_names = {
                "progress_antigravity.json",
                "events_antigravity.jsonl",
                "run_antigravity.pid",
                "parent_antigravity.log",
            }
            expected = {
                name: sha256((root / "output" / name).read_bytes()).hexdigest()
                for name in runtime_names
            }
            manifest = json.loads((publication.publication / "manifest_antigravity.json").read_bytes())
            status = json.loads((publication.publication / "status_antigravity.json").read_bytes())
            artifact_map = json.loads((publication.publication / "artifact_sha256_antigravity.json").read_bytes())
            self.assertEqual(manifest["runtime_sha256"], expected)
            self.assertEqual(status["runtime_sha256"], expected)
            self.assertEqual(artifact_map["runtime_sha256"], expected)

    def test_mutating_any_terminal_runtime_file_invalidates_replay(self) -> None:
        runtime_names = (
            "progress_antigravity.json",
            "events_antigravity.jsonl",
            "run_antigravity.pid",
            "parent_antigravity.log",
        )
        for name in runtime_names:
            with self.subTest(name=name), TemporaryDirectory() as directory:
                root = Path(directory)
                store = _runtime(root / "output", root / "checkpoint")
                publish_exp435(store, root / "output", _expectations=self.expectations)
                path = root / "output" / name
                if name == "events_antigravity.jsonl":
                    first = path.read_bytes().splitlines()[0]
                    path.write_bytes(path.read_bytes() + first + b"\n")
                elif name == "parent_antigravity.log":
                    with path.open("ab") as handle:
                        handle.write(b"2026-08-06T00:00:00Z valid post-publication line\n")
                else:
                    from experiments.exp403_408.artifact_io import canonical_json_bytes

                    payload = json.loads(path.read_bytes())
                    payload["updated_at"] = "2026-08-06T00:00:00Z"
                    path.write_bytes(canonical_json_bytes(payload))
                with self.assertRaises(ValueError):
                    inspect_publication(root / "output", store, _expectations=self.expectations)


if __name__ == "__main__":
    unittest.main()
