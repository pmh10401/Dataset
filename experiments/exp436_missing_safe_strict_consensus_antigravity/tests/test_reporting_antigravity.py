"""Real-artifact TDD coverage for the Exp436 label-free publication boundary."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp436_missing_safe_strict_consensus_antigravity.checkpoint_antigravity import (
    Exp436CheckpointStore, Exp436RunContext, _name, _result_payload,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import (
    ExcludedTrainRowAudit, Exp436Candidate, Exp436DatasetResult, InstanceScoreEvidence,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.progress_antigravity import ProgressWriter
from experiments.exp436_missing_safe_strict_consensus_antigravity.reporting_antigravity import (
    ABSTENTION_FILE, BACKEND_FILE, EVIDENCE_FILE, HASH_FILE, MANIFEST_FILE,
    PREDICTION_FILE, PUBLICATION_DIRECTORY, STATUS_FILE, SUMMARY_FILE, TRAIN_AUDIT_FILE,
    inspect_publication, publish_exp436,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_checkpoint_antigravity import _context


_FILES = frozenset({PREDICTION_FILE, SUMMARY_FILE, EVIDENCE_FILE, ABSTENTION_FILE,
                    TRAIN_AUDIT_FILE, BACKEND_FILE, MANIFEST_FILE, STATUS_FILE, HASH_FILE})
_METADATA = {"train_count": 4, "test_count": 1, "train_length_histogram": {"16": 4},
             "test_length_histogram": {"16": 1}}


def _frozen_context() -> Exp436RunContext:
    payload = _context().to_payload()
    payload["database_sha256"] = FROZEN_CONFIG.database_sha256
    for name, digest in {
        "exp291_prediction": FROZEN_CONFIG.exp291_prediction_sha256,
        "exp291_manifest": FROZEN_CONFIG.exp291_manifest_sha256,
        "exp292_prediction": FROZEN_CONFIG.exp292_prediction_sha256,
        "exp292_manifest": FROZEN_CONFIG.exp292_manifest_sha256,
    }.items():
        payload["lane_artifacts"][name]["sha256"] = digest
    payload["lane_snapshot_sha256"] = sha256(canonical_json_bytes({
        "artifacts": payload["lane_artifacts"], "mappings": payload["lane_mappings"],
        "datasets": payload["datasets"],
    })).hexdigest()
    payload["context_sha256"] = sha256(canonical_json_bytes(
        {key: value for key, value in payload.items() if key != "context_sha256"}
    )).hexdigest()
    return Exp436RunContext.from_payload(payload)


def _row(dataset: str, *, candidate: bool = False, audit: bool = False) -> Exp436DatasetResult:
    evidence = InstanceScoreEvidence.fixture(dataset=dataset, instance_index=2, length=16,
                                             prototype_abnormal=candidate,
                                             gradient_abnormal=candidate,
                                             acf_abnormal=candidate)
    candidates = (Exp436Candidate(dataset, 2, 16, evidence.evidence_digest),) if candidate else ()
    audits = (ExcludedTrainRowAudit(dataset, 1, 16, 1, "a" * 64),) if audit else ()
    return Exp436DatasetResult(dataset, "scored", (2,), candidates, (evidence,), audits)


class ReportingAntigravityTests(unittest.TestCase):
    """Every test uses the real v1 checkpoint and terminal progress artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = TemporaryDirectory()
        root = Path(cls._temporary.name)
        cls.checkpoint = root / "checkpoint_antigravity"
        cls.runtime = root / "runtime_antigravity"
        seed_store = Exp436CheckpointStore.create(cls.checkpoint, _frozen_context())
        # Construct the canonical ordered record prefix directly once.  Calling
        # ``commit`` 1,117 times deliberately revalidates every predecessor and
        # turns this production-shaped replay fixture into O(n²) test work.
        records: dict[str, str] = {}
        results: list[Exp436DatasetResult] = []
        for sequence, dataset in enumerate(seed_store.datasets):
            result = _row(dataset, candidate=dataset == "D0000", audit=dataset == "D0000")
            raw = canonical_json_bytes(_result_payload(result, sequence, _METADATA))
            (cls.checkpoint / "records_antigravity" / _name(dataset)).write_bytes(raw)
            records[dataset] = sha256(raw).hexdigest()
            results.append(result)
        record_set = Exp436CheckpointStore._set_payload_static(
            seed_store.context_fingerprint, seed_store.datasets, records, tuple(results),
        )
        (cls.checkpoint / "record_set_antigravity.json").write_bytes(canonical_json_bytes(record_set))
        cls.store = Exp436CheckpointStore.open_read_only(cls.checkpoint)
        writer = ProgressWriter.create(cls.runtime, run_identity=cls.store.run_identity,
                                       checkpoint=cls.checkpoint, total_datasets=1117,
                                       backend="cpu", context_fingerprint=cls.store.context_fingerprint,
                                       heartbeat_seconds=None)
        candidates = 0
        for sequence, dataset in enumerate(cls.store.datasets):
            result = results[sequence]
            digest = records[dataset]
            candidates += result.candidate_count
            writer.record_dataset_complete(sequence_index=sequence, dataset=dataset, record_sha256=digest,
                                           status="scored", dataset_metadata=_METADATA,
                                           completed_datasets=sequence + 1, success=sequence + 1,
                                           abstention=0, error=0, excluded_train_rows=1 if sequence else 1,
                                           nonfinite_test_rows=0, candidate_count=candidates)
        writer.mark_checkpoint_complete()
        writer.close()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _runtime_copy(self) -> Path:
        destination = Path(self._temporary.name) / f"case_{self.id().rsplit('.', 1)[-1]}_antigravity"
        shutil.copytree(self.runtime, destination)
        return destination

    def _publication(self) -> tuple[Path, object]:
        output = self._runtime_copy()
        return output, publish_exp436(self.store, output)

    def test_replays_a_real_1117_record_checkpoint_into_exact_nine_file_inventory(self) -> None:
        output, publication = self._publication()
        self.assertEqual({item.name for item in publication.publication.iterdir()}, _FILES)
        self.assertEqual(publication.manifest["completed_datasets"], 1117)
        self.assertEqual(inspect_publication(output, self.store).manifest_sha256, publication.manifest_sha256)

    def test_prediction_is_globally_sorted_unique_and_within_frozen_budget(self) -> None:
        _, publication = self._publication()
        rows = [json.loads(line) for line in (publication.publication / PREDICTION_FILE).read_bytes().splitlines()]
        self.assertEqual([(row["dataset"], row["instance_index"]) for row in rows], [("D0000", 2)])
        self.assertLessEqual(len(rows), 100)

    def test_every_test_identity_has_exactly_one_evidence_row(self) -> None:
        _, publication = self._publication()
        rows = [json.loads(line) for line in (publication.publication / EVIDENCE_FILE).read_bytes().splitlines()]
        self.assertEqual(len(rows), 1117)
        self.assertEqual([(rows[0]["dataset"], rows[0]["instance_index"]), (rows[-1]["dataset"], rows[-1]["instance_index"])], [("D0000", 2), ("D1116", 2)])

    def test_audit_ledger_is_ordered_and_binds_its_checkpoint_record_digest(self) -> None:
        _, publication = self._publication()
        row = json.loads((publication.publication / TRAIN_AUDIT_FILE).read_bytes().strip())
        self.assertEqual((row["dataset"], row["length"], row["instance_index"]), ("D0000", 16, 1))
        self.assertEqual(row["record_sha256"], self.store.completed_digests()["D0000"])

    def test_manifest_and_hash_map_bind_each_semantic_artifact_and_runtime_file(self) -> None:
        _, publication = self._publication()
        manifest = json.loads((publication.publication / MANIFEST_FILE).read_bytes())
        hashes = json.loads((publication.publication / HASH_FILE).read_bytes())
        self.assertEqual(set(manifest["artifact_sha256"]), {PREDICTION_FILE, SUMMARY_FILE, EVIDENCE_FILE, ABSTENTION_FILE, TRAIN_AUDIT_FILE, BACKEND_FILE})
        self.assertEqual(set(hashes["artifact_sha256"]), _FILES - {HASH_FILE})
        self.assertNotIn(HASH_FILE, hashes["artifact_sha256"])
        self.assertEqual(set(manifest["runtime_sha256"]), {"progress_antigravity.json", "events_antigravity.jsonl", "parent_antigravity.log", "run_antigravity.pid"})

    def test_status_is_review_only_and_matches_the_manifest_and_prediction(self) -> None:
        _, publication = self._publication()
        status = json.loads((publication.publication / STATUS_FILE).read_bytes())
        self.assertEqual(status["state"], "complete_review_only")
        self.assertTrue(status["review_only"])
        self.assertFalse(status["operational_alert_eligible"])
        self.assertEqual(status["manifest_sha256"], sha256((publication.publication / MANIFEST_FILE).read_bytes()).hexdigest())

    def test_missing_runtime_authority_rejects_publication_without_staging(self) -> None:
        output = self._runtime_copy()
        (output / "events_antigravity.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "inventory"):
            publish_exp436(self.store, output)
        self.assertEqual(tuple(output.glob(".publication_antigravity.*.tmp")), ())

    def test_nonterminal_runtime_state_rejects_publication(self) -> None:
        output = self._runtime_copy()
        progress = output / "progress_antigravity.json"
        payload = json.loads(progress.read_bytes()); payload["state"] = "running"
        payload["progress_sha256"] = sha256(canonical_json_bytes({key: value for key, value in payload.items() if key != "progress_sha256"})).hexdigest()
        progress.write_bytes(canonical_json_bytes(payload))
        with self.assertRaises(ValueError):
            publish_exp436(self.store, output)

    def test_active_pid_rejects_terminal_publication(self) -> None:
        output = self._runtime_copy()
        pid = output / "run_antigravity.pid"
        payload = json.loads(pid.read_bytes()); payload["pid_active"] = True
        payload["pid_sha256"] = sha256(canonical_json_bytes({key: value for key, value in payload.items() if key != "pid_sha256"})).hexdigest()
        pid.write_bytes(canonical_json_bytes(payload))
        with self.assertRaises(ValueError):
            publish_exp436(self.store, output)

    def test_unknown_runtime_inventory_rejects_publication(self) -> None:
        output = self._runtime_copy(); (output / "unexpected_antigravity.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "inventory"):
            publish_exp436(self.store, output)

    def test_replay_detects_tampering_of_every_publication_artifact(self) -> None:
        output, publication = self._publication()
        for name in sorted(_FILES):
            with self.subTest(name=name):
                path = publication.publication / name
                original = path.read_bytes(); path.write_bytes(original + b" ")
                with self.assertRaises(ValueError): inspect_publication(output, self.store)
                path.write_bytes(original)

    def test_replay_rejects_a_symlinked_publication_artifact(self) -> None:
        output, publication = self._publication()
        target = publication.publication / PREDICTION_FILE
        original = target.read_bytes(); target.unlink(); target.symlink_to("../" + STATUS_FILE)
        with self.assertRaises(ValueError): inspect_publication(output, self.store)
        target.unlink(); target.write_bytes(original)

    def test_replay_rejects_nonregular_publication_artifact(self) -> None:
        output, publication = self._publication()
        target = publication.publication / PREDICTION_FILE
        target.unlink(); target.mkdir()
        with self.assertRaises(ValueError): inspect_publication(output, self.store)

    def test_rejects_recursive_fpr_key_smuggling(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import reporting_antigravity
        with self.assertRaisesRegex(ValueError, "forbidden"):
            reporting_antigravity._reject_forbidden_keys({"nested": {"fpr": 2}}, artifact=PREDICTION_FILE)

    def test_existing_destination_on_entry_is_only_verified_by_replay(self) -> None:
        output, first = self._publication()
        before = {path.name: path.read_bytes() for path in first.publication.iterdir()}
        replay = publish_exp436(self.store, output)
        self.assertEqual(replay.manifest_sha256, first.manifest_sha256)
        self.assertEqual({path.name: path.read_bytes() for path in replay.publication.iterdir()}, before)

    def test_production_cli_inspect_replays_published_tree_without_progress_mutation(self) -> None:
        """A publication directory is not an invalid progress inventory entry."""
        from experiments.exp436_missing_safe_strict_consensus_antigravity import cli_antigravity
        output, publication = self._publication()
        readiness = output.parent / "readiness_antigravity.json"; readiness.write_bytes(b"{}")
        before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
        with patch.object(cli_antigravity, "RUN_PARENT", output.parent), \
                patch.object(cli_antigravity, "FROZEN_OUTPUT", output), \
                patch.object(cli_antigravity, "FROZEN_CHECKPOINT", self.checkpoint), \
                patch.object(cli_antigravity, "FROZEN_READINESS", readiness), \
                patch.object(cli_antigravity, "FROZEN_EVALUATION", output.parent / "evaluation_antigravity"), \
                patch.object(cli_antigravity, "_clean_head", return_value="b" * 40):
            inspected = cli_antigravity.run_inspect()
        self.assertEqual(inspected["state"], "complete_review_only")
        self.assertEqual(before, {path: path.read_bytes() for path in output.rglob("*") if path.is_file()})

    def test_real_1117_record_publication_passes_evaluator_prelabel_commit_handoff(self) -> None:
        """Publisher and evaluator must agree before any TEST-label route exists."""
        from experiments.exp436_missing_safe_strict_consensus_antigravity import evaluation_antigravity
        output, publication = self._publication()
        manifest, summaries, predictions = evaluation_antigravity._validate_verified_publication(
            publication, self.store, expected_total=FROZEN_CONFIG.expected_datasets, config=FROZEN_CONFIG,
        )
        self.assertEqual(manifest["reviewed_commit"], self.store.context["reviewed_commit"])
        self.assertEqual(len(summaries), FROZEN_CONFIG.expected_datasets)
        self.assertEqual(len(predictions), 1)

    def test_public_evaluate_replays_an_existing_bound_evaluation_without_labels(self) -> None:
        """The production evaluate route reuses only a fully bound immutable result."""
        from experiments.exp436_missing_safe_strict_consensus_antigravity import cli_antigravity
        from experiments.exp436_missing_safe_strict_consensus_antigravity import evaluation_antigravity
        output, publication = self._publication()
        evaluation_root = output.parent / "evaluation_antigravity"
        rows = (
            evaluation_antigravity.DatasetComparison(
                "D0000", 1,
                evaluation_antigravity.ConfusionCounts(0, 0, 0, 1),
                evaluation_antigravity.ConfusionCounts(0, 0, 0, 1),
            ),
        )
        bootstrap = evaluation_antigravity._paired_dataset_bootstrap_for_test(rows, repetitions=8, seed=123)
        existing = evaluation_antigravity.Exp436Evaluation(
            dataset_count=1, candidate_count=0, added_fp=0, dataset_rows=rows,
            baseline_confusion=rows[0].baseline, combined_confusion=rows[0].combined,
            micro_f1_delta=0.0, macro_f1_delta=0.0,
            micro_f1_ci=bootstrap.micro_f1_ci, macro_f1_ci=bootstrap.macro_f1_ci,
            rescued_datasets=(), regressed_datasets=(), strong_baseline_regressions=(),
            prediction_sha256=publication.prediction_sha256,
            manifest_sha256=publication.manifest_sha256,
            database_sha256=FROZEN_CONFIG.database_sha256,
            code_commit=str(publication.manifest["reviewed_commit"]), bootstrap=bootstrap,
        )
        evaluation_antigravity._write_evaluation_artifacts(evaluation_root, existing)
        before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
        with patch.object(cli_antigravity, "RUN_PARENT", output.parent), \
                patch.object(cli_antigravity, "FROZEN_OUTPUT", output), \
                patch.object(cli_antigravity, "FROZEN_CHECKPOINT", self.checkpoint), \
                patch.object(cli_antigravity, "FROZEN_EVALUATION", evaluation_root):
            replayed = evaluation_antigravity.evaluate_exp436(output, evaluation_root)
        self.assertEqual(replayed, existing)
        self.assertEqual(before, {path: path.read_bytes() for path in output.rglob("*") if path.is_file()})

    def test_destination_appearing_during_staging_fails_without_clobbering_it(self) -> None:
        output = self._runtime_copy()
        from experiments.exp436_missing_safe_strict_consensus_antigravity import reporting_antigravity
        original = reporting_antigravity._exclusive_rename
        destination = output / PUBLICATION_DIRECTORY
        def race(staging: Path, target: Path) -> None:
            target.mkdir(); original(staging, target)
        with patch.object(reporting_antigravity, "_exclusive_rename", side_effect=race):
            with self.assertRaises((OSError, RuntimeError, ValueError)):
                publish_exp436(self.store, output)
        self.assertTrue(destination.is_dir())
        self.assertEqual(tuple(destination.iterdir()), ())
        self.assertEqual(tuple(output.glob(".publication_antigravity.*.tmp")), ())

    def test_source_mutation_after_publication_invalidates_replay(self) -> None:
        output, _ = self._publication()
        event = output / "events_antigravity.jsonl"; event.write_bytes(event.read_bytes() + b"\n")
        with self.assertRaises(ValueError): inspect_publication(output, self.store)

    def test_source_mutation_during_staging_blocks_exclusive_rename(self) -> None:
        """The source authorities are replayed again after staged-byte verification."""
        output = self._runtime_copy()
        from experiments.exp436_missing_safe_strict_consensus_antigravity import reporting_antigravity
        original = reporting_antigravity._write_new
        writes = {"count": 0}
        def mutate(path: Path, payload: bytes) -> None:
            original(path, payload); writes["count"] += 1
            if writes["count"] == 1:
                event = output / "events_antigravity.jsonl"
                event.write_bytes(event.read_bytes() + b"\n")
        with patch.object(reporting_antigravity, "_write_new", side_effect=mutate), \
                patch.object(reporting_antigravity, "_exclusive_rename", wraps=reporting_antigravity._exclusive_rename) as renamed:
            with self.assertRaises(ValueError):
                publish_exp436(self.store, output)
        self.assertEqual(renamed.call_count, 0)
        self.assertFalse((output / PUBLICATION_DIRECTORY).exists())

    def test_staging_files_are_create_only_and_private(self) -> None:
        output = self._runtime_copy()
        from experiments.exp436_missing_safe_strict_consensus_antigravity import reporting_antigravity
        original = reporting_antigravity._write_new
        seen: list[int] = []
        def observe(path: Path, payload: bytes) -> None:
            original(path, payload); seen.append(stat.S_IMODE(path.stat().st_mode))
        with patch.object(reporting_antigravity, "_write_new", side_effect=observe):
            publish_exp436(self.store, output)
        self.assertEqual(len(seen), 9)
        self.assertEqual(set(seen), {0o600})


if __name__ == "__main__":
    unittest.main()
