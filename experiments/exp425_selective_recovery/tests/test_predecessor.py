"""Descriptor-bound Exp424 predecessor import tests."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import json
import pickle
import unittest
from unittest.mock import patch

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp425_selective_recovery.predecessor import (
    Exp424CompatibilityContract,
    import_exp424_nonterminal,
    open_validated_exp424_source,
)
from experiments.exp417_causal_multihorizon_shadow.checkpoints import Exp417CheckpointStore
from experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints import (
    _context as exp417_context,
    _provenance as exp417_provenance,
    _result as exp417_result,
    _write as exp417_write,
)
from experiments.exp425_selective_recovery.checkpoints import Exp425CheckpointStore
from experiments.exp425_selective_recovery.tests.test_checkpoints import _provenance as exp425_provenance


class Exp424PredecessorTests(unittest.TestCase):
    """Breaks if validation accepts a record-set digest from the wrong preimage."""

    def test_validation_requires_sorted_record_digest_items_preimage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            (root / "datasets").mkdir(parents=True)
            record = {"schema_version": 3, "dataset": "Eligible_normal_1"}
            raw = canonical_json_bytes(record) + b"\n"
            digest = sha256(raw).hexdigest()
            (root / "datasets" / "record.json").write_bytes(raw)
            progress = {
                "expected_datasets": ["Eligible_normal_1"],
                "completed_datasets": 1,
                "completed_record_digests": {"Eligible_normal_1": digest},
                "current_dataset": None,
            }
            provenance = {"schema_version": 3, "provenance": {}, "expected_datasets": ["Eligible_normal_1"]}
            (root / "progress.json").write_bytes(canonical_json_bytes(progress) + b"\n")
            (root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(provenance) + b"\n")
            contract = Exp424CompatibilityContract.for_fixture(
                root,
                root_provenance_sha256=sha256((root / "checkpoint_provenance.json").read_bytes()).hexdigest(),
                progress_sha256=sha256((root / "progress.json").read_bytes()).hexdigest(),
                record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest(),
            )
            with open_validated_exp424_source(root, contract) as source:
                self.assertEqual(source.datasets, ("Eligible_normal_1",))
            wrong = Exp424CompatibilityContract.for_fixture(
                root,
                root_provenance_sha256=contract.root_provenance_sha256,
                progress_sha256=contract.progress_sha256,
                record_set_sha256=sha256(canonical_json_bytes(progress["completed_record_digests"])).hexdigest(),
            )
            with self.assertRaises(ValueError):
                with open_validated_exp424_source(root, wrong):
                    pass

    def test_live_source_is_noncopyable_and_revalidation_detects_mutation(self) -> None:
        """Removing the live-authority or byte-integrity guard must fail this test."""
        with TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            (root / "datasets").mkdir(parents=True)
            raw = canonical_json_bytes({"schema_version": 3, "dataset": "Eligible_normal_1"}) + b"\n"
            digest = sha256(raw).hexdigest()
            (root / "datasets" / "record.json").write_bytes(raw)
            progress = {"expected_datasets": ["Eligible_normal_1"], "completed_datasets": 1, "completed_record_digests": {"Eligible_normal_1": digest}, "current_dataset": None}
            provenance = {"schema_version": 3, "provenance": {}, "expected_datasets": ["Eligible_normal_1"]}
            (root / "progress.json").write_bytes(canonical_json_bytes(progress) + b"\n")
            (root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(provenance) + b"\n")
            contract = Exp424CompatibilityContract.for_fixture(root, root_provenance_sha256=sha256((root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            with open_validated_exp424_source(root, contract) as source:
                with self.assertRaises(TypeError): copy.copy(source)
                with self.assertRaises(TypeError): pickle.dumps(source)
                (root / "datasets" / "record.json").write_bytes(raw + b" ")
                with self.assertRaises(ValueError): source._revalidate()
            with self.assertRaises(TypeError):
                source.__class__(object())

    def test_import_stages_only_a_verified_nonterminal_result(self) -> None:
        """Changing import to recompute or publish terminal evidence must fail this test."""
        context = exp417_context()
        dataset = context.datasets[0]
        with TemporaryDirectory() as directory:
            source_root = Path(directory) / "exp424"
            source_store = Exp417CheckpointStore(source_root, exp417_provenance(), context.datasets)
            exp417_write(source_store, exp417_result(context, dataset))
            record_path = source_root / "datasets" / (sha256(dataset.encode()).hexdigest() + ".json")
            record_digest = json.loads(record_path.read_text())["payload_sha256"]
            provenance = {"schema_version": 3, "provenance": {}, "expected_datasets": [dataset]}
            progress = {"expected_datasets": [dataset], "completed_datasets": 1, "completed_record_digests": {dataset: record_digest}, "current_dataset": None}
            (source_root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(provenance) + b"\n")
            (source_root / "progress.json").write_bytes(canonical_json_bytes(progress) + b"\n")
            contract = Exp424CompatibilityContract.for_fixture(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            target = Exp425CheckpointStore.create(Path(directory) / "exp425", exp425_provenance(context), context)
            with open_validated_exp424_source(source_root, contract) as source:
                report = import_exp424_nonterminal(source, target, context)
            self.assertEqual(report.imported, (dataset,))
            self.assertEqual(report.rejected_terminal, ())
            self.assertEqual(report.rejected_incompatible, ())
            self.assertEqual(target.load_record(dataset).origin, "exp424_import")

    def test_crashed_import_requires_a_new_source_authority_before_promotion(self) -> None:
        """Removing source revalidation after an import crash would publish stale authority work."""
        context = exp417_context()
        dataset = context.datasets[0]
        with TemporaryDirectory() as directory:
            source_root = Path(directory) / "exp424"
            source_store = Exp417CheckpointStore(source_root, exp417_provenance(), context.datasets)
            exp417_write(source_store, exp417_result(context, dataset))
            record_path = source_root / "datasets" / (sha256(dataset.encode()).hexdigest() + ".json")
            record_digest = json.loads(record_path.read_text())["payload_sha256"]
            provenance_payload = {"schema_version": 3, "provenance": {}, "expected_datasets": [dataset]}
            progress = {"expected_datasets": [dataset], "completed_datasets": 1, "completed_record_digests": {dataset: record_digest}, "current_dataset": None}
            (source_root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(provenance_payload) + b"\n")
            (source_root / "progress.json").write_bytes(canonical_json_bytes(progress) + b"\n")
            contract = Exp424CompatibilityContract.for_fixture(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            provenance = exp425_provenance(context)
            target_root = Path(directory) / "exp425"
            target = Exp425CheckpointStore.create(target_root, provenance, context)
            with open_validated_exp424_source(source_root, contract) as old_source:
                old_nonce = old_source.nonce
                with patch("experiments.exp425_selective_recovery.checkpoints._after_import_phase", side_effect=lambda phase: (_ for _ in ()).throw(RuntimeError("crash")) if phase == "intent_durable" else None):
                    with self.assertRaisesRegex(RuntimeError, "crash"):
                        import_exp424_nonterminal(old_source, target, context)
            reopened = Exp425CheckpointStore.open(target_root, provenance, context)
            self.assertTrue(reopened.import_revalidation_required)
            with open_validated_exp424_source(source_root, contract) as new_source:
                self.assertNotEqual(new_source.nonce, old_nonce)
                report = import_exp424_nonterminal(new_source, reopened, context)
            self.assertEqual(report.imported, (dataset,))
            self.assertEqual(reopened.completed_record_digests(), {dataset: reopened.completed_record_digests()[dataset]})

    def test_source_mutation_after_staging_aborts_then_fresh_authority_resumes(self) -> None:
        """Publishing after a staged-source mutation would import evidence not fully revalidated."""
        context = exp417_context()
        dataset = context.datasets[0]
        with TemporaryDirectory() as directory:
            source_root = Path(directory) / "exp424"
            source_store = Exp417CheckpointStore(source_root, exp417_provenance(), context.datasets)
            exp417_write(source_store, exp417_result(context, dataset))
            record_path = source_root / "datasets" / (sha256(dataset.encode()).hexdigest() + ".json")
            record_bytes = record_path.read_bytes()
            record_digest = json.loads(record_bytes)["payload_sha256"]
            provenance_payload = {"schema_version": 3, "provenance": {}, "expected_datasets": [dataset]}
            progress = {"expected_datasets": [dataset], "completed_datasets": 1, "completed_record_digests": {dataset: record_digest}, "current_dataset": None}
            (source_root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(provenance_payload) + b"\n")
            (source_root / "progress.json").write_bytes(canonical_json_bytes(progress) + b"\n")
            contract = Exp424CompatibilityContract.for_fixture(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            target = Exp425CheckpointStore.create(Path(directory) / "exp425", exp425_provenance(context), context)
            real_stage = target._stage_imported
            def stage_then_mutate(*args):
                real_stage(*args)
                record_path.write_bytes(record_bytes + b" ")
            with open_validated_exp424_source(source_root, contract) as old_source:
                old_nonce = old_source.nonce
                with patch.object(target, "_stage_imported", side_effect=stage_then_mutate):
                    with self.assertRaises(ValueError):
                        import_exp424_nonterminal(old_source, target, context)
            self.assertEqual(target.completed_record_digests(), {})
            record_path.write_bytes(record_bytes)
            with open_validated_exp424_source(source_root, contract) as new_source:
                self.assertNotEqual(new_source.nonce, old_nonce)
                report = import_exp424_nonterminal(new_source, target, context)
            self.assertEqual(report.imported, (dataset,))
            self.assertEqual(target.load_record(dataset).origin, "exp424_import")
