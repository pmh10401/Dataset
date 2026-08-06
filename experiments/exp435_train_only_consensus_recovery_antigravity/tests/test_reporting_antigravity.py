"""TDD coverage for Exp435 review-only publication replay."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

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
from experiments.exp435_train_only_consensus_recovery_antigravity.reporting_antigravity import (
    CandidateBudgetExceeded,
    FrozenPublication,
    _PublicationExpectations,
    inspect_publication,
    publish_exp435,
)


def _context() -> dict[str, object]:
    return {
        "schema": "fixture.context.v1",
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
        stage="scoring", current_dataset="Demo", dataset_metadata=None,
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
        self.run_identity = "exp435-fixture"
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
        stage="scoring", current_dataset="Demo", dataset_metadata=None,
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


if __name__ == "__main__":
    unittest.main()
