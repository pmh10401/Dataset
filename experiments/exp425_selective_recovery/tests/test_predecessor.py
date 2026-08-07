"""Descriptor-bound Exp424 predecessor import tests."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import json
import os
import pickle
import unittest
from unittest.mock import patch
from types import MappingProxyType

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp425_selective_recovery.predecessor import (
    Exp424CompatibilityContract,
    EXP424_SCORE_CACHE_CONTRACT,
    EXP424_THRESHOLDS,
    _batch_identity,
    _durable_attestation,
    _fixture_contract,
    import_exp424_nonterminal,
    open_validated_exp424_source,
)
from experiments.exp417_causal_multihorizon_shadow.checkpoints import Exp417CheckpointStore
from experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints import (
    _context as exp417_context,
    _mixed_result as exp417_mixed_result,
    _provenance as exp417_provenance,
    _result as exp417_result,
    _write as exp417_write,
)
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import BackendAudit
from experiments.exp417_causal_multihorizon_shadow.contracts import DatasetResult, DatasetTerminalEvidence
from experiments.exp425_selective_recovery.checkpoints import Exp425CheckpointStore
from experiments.exp425_selective_recovery import checkpoints as exp425_checkpoints
from experiments.exp425_selective_recovery.tests.test_checkpoints import _provenance as exp425_provenance
from dataclasses import replace


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
            contract = _fixture_contract(
                root,
                root_provenance_sha256=sha256((root / "checkpoint_provenance.json").read_bytes()).hexdigest(),
                progress_sha256=sha256((root / "progress.json").read_bytes()).hexdigest(),
                record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest(),
            )
            with open_validated_exp424_source(root, contract) as source:
                self.assertEqual(source.datasets, ("Eligible_normal_1",))
            (root / "unrecognized-entry").write_bytes(b"foreign")
            with self.assertRaises(ValueError):
                with open_validated_exp424_source(root, contract):
                    pass
            (root / "unrecognized-entry").unlink()
            alias = Path(directory) / "source-alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "contract path"):
                with open_validated_exp424_source(alias, contract):
                    pass
            wrong = _fixture_contract(
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
            contract = _fixture_contract(root, root_provenance_sha256=sha256((root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            with open_validated_exp424_source(root, contract) as source:
                with self.assertRaises(TypeError): copy.copy(source)
                with self.assertRaises(TypeError): pickle.dumps(source)
                (root / "datasets" / "record.json").write_bytes(raw + b" ")
                with self.assertRaises(ValueError): source._revalidate()
            with self.assertRaises(TypeError):
                source.__class__(object())

    def test_revalidation_rejects_identical_byte_progress_replacement(self) -> None:
        """Dropping metadata inode binding would accept an ABA progress replacement."""
        with TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            (root / "datasets").mkdir(parents=True)
            raw = canonical_json_bytes({"schema_version": 3, "dataset": "Eligible_normal_1"}) + b"\n"
            digest = sha256(raw).hexdigest()
            (root / "datasets" / "record.json").write_bytes(raw)
            progress = {"expected_datasets": ["Eligible_normal_1"], "completed_datasets": 1, "completed_record_digests": {"Eligible_normal_1": digest}, "current_dataset": None}
            provenance = {"schema_version": 3, "provenance": {}, "expected_datasets": ["Eligible_normal_1"]}
            progress_bytes = canonical_json_bytes(progress) + b"\n"
            (root / "progress.json").write_bytes(progress_bytes)
            (root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(provenance) + b"\n")
            contract = _fixture_contract(root, root_provenance_sha256=sha256((root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256(progress_bytes).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            with open_validated_exp424_source(root, contract) as source:
                replacement = root / "replacement-progress.json"
                replacement.write_bytes(progress_bytes)
                os.replace(replacement, root / "progress.json")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    source._revalidate()

    def test_production_contract_rejects_every_frozen_field_mutation(self) -> None:
        """A caller must not create a self-consistent but altered production contract."""
        contract = Exp424CompatibilityContract.production()
        mutations = {
            "database_sha256": "0" * 64, "lane_hashes": {}, "config_sha256": "0" * 64,
            "preprocessing_contract_sha256": "0" * 64,
            "fold_contract_sha256": "0" * 64, "model_contract_sha256": "0" * 64,
            "backend_contract_sha256": "0" * 64, "model_seeds": (1,),
            "perturbation_seeds": (1,), "runtime_versions": {"python": "x"},
            "thresholds_payload": {}, "alpha_values": (0.5,),
            "candidate_formula_sha256": "0" * 64, "routing_contract_sha256": "0" * 64,
            "device_transfer_contract_sha256": "0" * 64, "dataset_universe_sha256": "0" * 64,
            "test_identity_manifest_sha256": "0" * 64, "entrypoint_sha256": "0" * 64,
            "entrypoint_git_blob": "0" * 40, "source_git_commit": "0" * 40,
            "device": "cpu", "mps_device_guard": "other", "batch_size": 1,
            "query_chunk_size": 1, "score_formula": "other", "score_cache_contract_sha256": "0" * 64,
            "percentile_method": "nearest", "score_dtype": "float32",
            "synchronization_contract_sha256": "0" * 64,
            "score_cache_contract_payload": {}, "synchronization_contract_payload": {},
            "device_transfer_contract_payload": {},
        }
        for field, value in mutations.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(contract, **{field: value})

    def test_compatibility_payloads_are_recursively_detached_and_rechecked_for_attestation(self) -> None:
        """Shallow mapping proxies would let a caller alter nested compatibility evidence after construction."""
        supplied_score_payload = copy.deepcopy(EXP424_SCORE_CACHE_CONTRACT)
        supplied_thresholds = copy.deepcopy(EXP424_THRESHOLDS)
        contract = replace(Exp424CompatibilityContract.production(), score_cache_contract_payload=supplied_score_payload, thresholds_payload=supplied_thresholds)
        before = contract.attestation_sha256()
        supplied_score_payload["components"].append("caller mutation")
        supplied_thresholds["candidate_alpha"] = 0.5
        self.assertEqual(contract.score_cache_contract_payload["components"], tuple(EXP424_SCORE_CACHE_CONTRACT["components"]))
        self.assertEqual(contract.thresholds_payload["candidate_alpha"], EXP424_THRESHOLDS["candidate_alpha"])
        self.assertEqual(contract.attestation_sha256(), before)
        object.__setattr__(contract, "score_cache_contract_payload", MappingProxyType({"schema": "tampered"}))
        with self.assertRaisesRegex(ValueError, "payload digest"):
            contract.attestation_sha256()

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
            contract = _fixture_contract(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            target = Exp425CheckpointStore.create(Path(directory) / "exp425", replace(exp425_provenance(context), predecessor_compatibility_sha256=contract.attestation_sha256()), context)
            with open_validated_exp424_source(source_root, contract) as source:
                report = import_exp424_nonterminal(source, target, context)
            self.assertEqual(report.imported, (dataset,))
            self.assertEqual(report.rejected_terminal, ())
            self.assertEqual(report.rejected_incompatible, ())
            self.assertEqual(target.load_record(dataset).origin, "exp424_import")

    def test_import_rejects_target_with_a_different_predecessor_attestation(self) -> None:
        """Deleting the target-attestation gate would make unrelated source bytes importable."""
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
            contract = _fixture_contract(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            target = Exp425CheckpointStore.create(Path(directory) / "exp425", exp425_provenance(context), context)
            with open_validated_exp424_source(source_root, contract) as source:
                with self.assertRaisesRegex(ValueError, "predecessor compatibility"):
                    import_exp424_nonterminal(source, target, context)
            self.assertEqual(target.completed_record_digests(), {})

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
            contract = _fixture_contract(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            provenance = replace(exp425_provenance(context), predecessor_compatibility_sha256=contract.attestation_sha256())
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
            self.assertEqual([path for path in (target_root / "import_batches").iterdir() if path.name.startswith(".quarantined-")], [])

    def test_importer_value_errors_after_promotion_remain_resumable_without_quarantine(self) -> None:
        """The importer wrapper must not recast post-promotion recovery as a pre-promotion abort."""
        context = exp417_context()
        dataset = context.datasets[0]
        phases = (
            ("promotion", "intent_durable"), ("promotion", "record_durable"),
            ("promotion", "progress_durable"), ("promotion", "intent_removed"),
            ("cleanup", "cleanup_tombstone_durable"),
            ("cleanup", "cleanup_after_staged_isolation_before_phase_update"),
            ("cleanup", "cleanup_staged_unlinked"),
            ("cleanup", "cleanup_after_metadata_isolation_before_phase_update"),
            ("cleanup", "cleanup_metadata_removed"),
            ("cleanup", "cleanup_after_directory_isolation_before_phase_update"),
            ("cleanup", "cleanup_directory_removed"), ("cleanup", "cleanup_directory_fsynced"),
        )
        for hook, phase in phases:
            with self.subTest(hook=hook, phase=phase), TemporaryDirectory() as directory:
                source_root = Path(directory) / "source"
                source_store = Exp417CheckpointStore(source_root, exp417_provenance(), context.datasets)
                exp417_write(source_store, exp417_result(context, dataset))
                record_path = source_root / "datasets" / (sha256(dataset.encode()).hexdigest() + ".json")
                digest = json.loads(record_path.read_text())["payload_sha256"]
                provenance_payload = {"schema_version": 3, "provenance": {}, "expected_datasets": [dataset]}
                progress = {"expected_datasets": [dataset], "completed_datasets": 1, "completed_record_digests": {dataset: digest}, "current_dataset": None}
                (source_root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(provenance_payload) + b"\n")
                (source_root / "progress.json").write_bytes(canonical_json_bytes(progress) + b"\n")
                contract = _fixture_contract(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
                target_root = Path(directory) / "target"
                provenance = replace(exp425_provenance(context), predecessor_compatibility_sha256=contract.attestation_sha256())
                target = Exp425CheckpointStore.create(target_root, provenance, context)
                checkpoint_hook = "_after_import_phase" if hook == "promotion" else "_after_import_cleanup_phase"
                with open_validated_exp424_source(source_root, contract) as old_source:
                    old_nonce = old_source.nonce
                    with patch.object(exp425_checkpoints, checkpoint_hook, side_effect=lambda reached: (_ for _ in ()).throw(ValueError("injected promotion failure")) if reached == phase else None):
                        with self.assertRaisesRegex(ValueError, "injected promotion failure"):
                            import_exp424_nonterminal(old_source, target, context)
                self.assertEqual([path for path in (target_root / "import_batches").iterdir() if path.name.startswith(".quarantined-")], [])
                reopened = Exp425CheckpointStore.open(target_root, provenance, context)
                self.assertEqual(len(reopened.import_revalidation_required), 1)
                with open_validated_exp424_source(source_root, contract) as new_source:
                    self.assertNotEqual(new_source.nonce, old_nonce)
                    report = import_exp424_nonterminal(new_source, reopened, context)
                self.assertEqual(report.imported, (dataset,))
                self.assertEqual(len(reopened.completed_record_digests()), 1)
                self.assertEqual(list((target_root / "import_batches").iterdir()), [])

    def test_mint_time_promotion_revalidation_failure_quarantines_unpublished_batch(self) -> None:
        """Setting the promotion flag before minting would strand a pre-promotion source failure."""
        context = exp417_context()
        dataset = context.datasets[0]
        with TemporaryDirectory() as directory:
            source_root = Path(directory) / "source"
            source_store = Exp417CheckpointStore(source_root, exp417_provenance(), context.datasets)
            exp417_write(source_store, exp417_result(context, dataset))
            record_path = source_root / "datasets" / (sha256(dataset.encode()).hexdigest() + ".json")
            digest = json.loads(record_path.read_text())["payload_sha256"]
            source_provenance = {"schema_version": 3, "provenance": {}, "expected_datasets": [dataset]}
            progress = {"expected_datasets": [dataset], "completed_datasets": 1, "completed_record_digests": {dataset: digest}, "current_dataset": None}
            (source_root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(source_provenance) + b"\n")
            (source_root / "progress.json").write_bytes(canonical_json_bytes(progress) + b"\n")
            contract = _fixture_contract(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            target_root = Path(directory) / "target"
            provenance = replace(exp425_provenance(context), predecessor_compatibility_sha256=contract.attestation_sha256())
            target = Exp425CheckpointStore.create(target_root, provenance, context)
            with open_validated_exp424_source(source_root, contract) as source:
                real_revalidate = type(source)._revalidate
                calls = 0

                def fail_only_at_promotion_mint(current: object) -> None:
                    nonlocal calls
                    calls += 1
                    if current is source and calls == 2:
                        raise ValueError("mint-time source revalidation failure")
                    real_revalidate(current)

                with patch.object(type(source), "_revalidate", autospec=True, side_effect=fail_only_at_promotion_mint):
                    with self.assertRaisesRegex(ValueError, "mint-time source revalidation failure"):
                        import_exp424_nonterminal(source, target, context)
            self.assertEqual(target.completed_record_digests(), {})
            self.assertEqual(list((target_root / "transactions").iterdir()), [])
            self.assertEqual(list((target_root / "datasets").iterdir()), [])
            self.assertEqual(len([path for path in (target_root / "import_batches").iterdir() if path.name.startswith(".quarantined-")]), 1)
            reopened = Exp425CheckpointStore.open(target_root, provenance, context)
            self.assertEqual(reopened.import_revalidation_required, ())
            self.assertEqual(len(list((target_root / "import_batches").iterdir())), 1)

    def test_import_preserves_candidate_and_diagnostic_group_evidence_per_length(self) -> None:
        """OOF/calibration evidence is present for diagnostics, while model evidence is qualified-only."""
        base_context = exp417_context()
        result = exp417_mixed_result(base_context)
        context = result.context
        dataset = result.dataset
        with TemporaryDirectory() as directory:
            source_root = Path(directory) / "exp424"
            source_store = Exp417CheckpointStore(source_root, exp417_provenance(), context.datasets)
            source_store.write_dataset(
                result,
                raw_train_identity_sha256="a" * 64,
                repaired_train_identity_sha256="b" * 64,
                length_group_identity_sha256={64: "c" * 64, 100: "c" * 64},
                canonical_model_hashes={100: {seed: "d" * 64 for seed in FROZEN_CONFIG.model_seeds}},
                score_calibration_hashes={64: "e" * 64, 100: "e" * 64},
                backend_audit=BackendAudit("cpu", True, 0.0, 1.0, 1.0, False),
                elapsed_seconds=0.125,
            )
            record_path = source_root / "datasets" / (sha256(dataset.encode()).hexdigest() + ".json")
            record_digest = json.loads(record_path.read_text())["payload_sha256"]
            provenance = {"schema_version": 3, "provenance": {}, "expected_datasets": [dataset]}
            progress = {"expected_datasets": [dataset], "completed_datasets": 1, "completed_record_digests": {dataset: record_digest}, "current_dataset": None}
            (source_root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(provenance) + b"\n")
            (source_root / "progress.json").write_bytes(canonical_json_bytes(progress) + b"\n")
            contract = _fixture_contract(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            target = Exp425CheckpointStore.create(Path(directory) / "exp425", replace(exp425_provenance(context), predecessor_compatibility_sha256=contract.attestation_sha256()), context)
            with open_validated_exp424_source(source_root, contract) as source:
                report = import_exp424_nonterminal(source, target, context)
            self.assertEqual(report.imported, (dataset,))
            groups = target.load_record(dataset).group_evidence.source_group_evidence["groups"]
            self.assertEqual([dict(group) for group in groups], [
                {"length": 64, "qualification_status": "diagnostic", "oof_evidence_present": True, "final_model_evidence_present": False, "calibration_evidence_present": True, "split_counts": "unavailable"},
                {"length": 100, "qualification_status": "qualified", "oof_evidence_present": True, "final_model_evidence_present": True, "calibration_evidence_present": True, "split_counts": "unavailable"},
            ])

    def test_durable_attestation_excludes_ephemeral_source_nonce(self) -> None:
        """A renewed source authority must bind the same durable import bytes, not its nonce."""
        context = exp417_context()
        dataset = context.datasets[0]
        with TemporaryDirectory() as directory:
            source_root = Path(directory) / "source"
            source_store = Exp417CheckpointStore(source_root, exp417_provenance(), context.datasets)
            exp417_write(source_store, exp417_result(context, dataset))
            record_path = source_root / "datasets" / (sha256(dataset.encode()).hexdigest() + ".json")
            digest = json.loads(record_path.read_text())["payload_sha256"]
            provenance = {"schema_version": 3, "provenance": {}, "expected_datasets": [dataset]}
            progress = {"expected_datasets": [dataset], "completed_datasets": 1, "completed_record_digests": {dataset: digest}, "current_dataset": None}
            (source_root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(provenance) + b"\n")
            (source_root / "progress.json").write_bytes(canonical_json_bytes(progress) + b"\n")
            contract = _fixture_contract(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            target = Exp425CheckpointStore.create(Path(directory) / "target", replace(exp425_provenance(context), predecessor_compatibility_sha256=contract.attestation_sha256()), context)
            with open_validated_exp424_source(source_root, contract) as first, open_validated_exp424_source(source_root, contract) as second:
                self.assertNotEqual(first.nonce, second.nonce)
                batch = _batch_identity(first, target)
                self.assertEqual(batch, _batch_identity(second, target))
                self.assertEqual(_durable_attestation(first, target, batch, {dataset: digest}), _durable_attestation(second, target, batch, {dataset: digest}))

    def test_terminal_source_record_is_reported_without_importing(self) -> None:
        """Terminal Exp424 evidence remains inspectable but can never become an imported result."""
        context = exp417_context()
        dataset = context.datasets[0]
        terminal = DatasetResult(dataset, context.test_sizes[dataset], 0, (), (), DatasetTerminalEvidence(dataset, context, "oof", "synthetic terminal"), context.lanes, context, "failed")
        with TemporaryDirectory() as directory:
            source_root = Path(directory) / "source"
            source_store = Exp417CheckpointStore(source_root, exp417_provenance(), context.datasets)
            source_store.write_dataset(terminal, raw_train_identity_sha256=None, repaired_train_identity_sha256=None, length_group_identity_sha256={}, canonical_model_hashes={}, score_calibration_hashes={}, backend_audit=None, elapsed_seconds=0.0)
            record_path = source_root / "datasets" / (sha256(dataset.encode()).hexdigest() + ".json")
            digest = json.loads(record_path.read_text())["payload_sha256"]
            provenance = {"schema_version": 3, "provenance": {}, "expected_datasets": [dataset]}
            progress = {"expected_datasets": [dataset], "completed_datasets": 1, "completed_record_digests": {dataset: digest}, "current_dataset": None}
            (source_root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(provenance) + b"\n")
            (source_root / "progress.json").write_bytes(canonical_json_bytes(progress) + b"\n")
            contract = _fixture_contract(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            target = Exp425CheckpointStore.create(Path(directory) / "target", replace(exp425_provenance(context), predecessor_compatibility_sha256=contract.attestation_sha256()), context)
            with open_validated_exp424_source(source_root, contract) as source:
                report = import_exp424_nonterminal(source, target, context)
            self.assertEqual(report.imported, ())
            self.assertEqual(report.rejected_terminal, (dataset,))
            self.assertEqual(target.completed_record_digests(), {})

    def test_context_incompatible_source_record_is_reported_without_importing(self) -> None:
        """A verified source record with a different target TEST identity manifest is not portable."""
        context = exp417_context()
        dataset = context.datasets[0]
        target_manifest = dict(context.test_identity_manifest)
        target_manifest[dataset] = ((0, 100),)
        target_context = replace(context, test_identity_manifest=target_manifest)
        with TemporaryDirectory() as directory:
            source_root = Path(directory) / "source"
            source_store = Exp417CheckpointStore(source_root, exp417_provenance(), context.datasets)
            exp417_write(source_store, exp417_result(context, dataset))
            record_path = source_root / "datasets" / (sha256(dataset.encode()).hexdigest() + ".json")
            digest = json.loads(record_path.read_text())["payload_sha256"]
            provenance = {"schema_version": 3, "provenance": {}, "expected_datasets": [dataset]}
            progress = {"expected_datasets": [dataset], "completed_datasets": 1, "completed_record_digests": {dataset: digest}, "current_dataset": None}
            (source_root / "checkpoint_provenance.json").write_bytes(canonical_json_bytes(provenance) + b"\n")
            (source_root / "progress.json").write_bytes(canonical_json_bytes(progress) + b"\n")
            contract = _fixture_contract(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            target = Exp425CheckpointStore.create(Path(directory) / "target", replace(exp425_provenance(target_context), predecessor_compatibility_sha256=contract.attestation_sha256()), target_context)
            with open_validated_exp424_source(source_root, contract) as source:
                report = import_exp424_nonterminal(source, target, target_context)
            self.assertEqual(report.imported, ())
            self.assertEqual(report.rejected_incompatible, (dataset,))
            self.assertEqual(target.completed_record_digests(), {})

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
            contract = _fixture_contract(source_root, root_provenance_sha256=sha256((source_root / "checkpoint_provenance.json").read_bytes()).hexdigest(), progress_sha256=sha256((source_root / "progress.json").read_bytes()).hexdigest(), record_set_sha256=sha256(canonical_json_bytes(sorted(progress["completed_record_digests"].items()))).hexdigest())
            target = Exp425CheckpointStore.create(Path(directory) / "exp425", replace(exp425_provenance(context), predecessor_compatibility_sha256=contract.attestation_sha256()), context)
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
            self.assertEqual(len([path for path in (target.root / "import_batches").iterdir() if path.name.startswith(".quarantined-")]), 1)
            record_path.write_bytes(record_bytes)
            with open_validated_exp424_source(source_root, contract) as new_source:
                self.assertNotEqual(new_source.nonce, old_nonce)
                report = import_exp424_nonterminal(new_source, target, context)
            self.assertEqual(report.imported, (dataset,))
            self.assertEqual(target.load_record(dataset).origin, "exp424_import")
