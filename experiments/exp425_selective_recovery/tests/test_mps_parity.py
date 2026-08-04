"""Focused fail-closed contract tests for the Exp425 Task 5 MPS smoke."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import sleep
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class MPSParityContractTests(unittest.TestCase):
    @staticmethod
    def _script_module() -> object:
        script = Path(__file__).resolve().parents[3] / "scratch/run_exp425_task5_mps_parity.py"
        spec = importlib.util.spec_from_file_location("exp425_task5_mps_parity_test", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_smoke_binds_guarded_train_and_test_loaders_before_session_construction(self) -> None:
        """Moving the loader patch below session construction would expose TEST values."""
        module = self._script_module()
        snapshot = object.__new__(module.Exp425DatabaseSnapshot)
        context = SimpleNamespace()
        counters = {"train": 0, "test_values": 0, "test_labels": 0}
        train = object()

        def fake_train(_snapshot, dataset, frozen):
            self.assertEqual((dataset, frozen), (module.FIXTURE, context))
            return train

        with patch.object(module.Exp425DatabaseSnapshot, "load_train", fake_train):
            with module._guarded_smoke_loaders(snapshot, counters) as guards:
                session = module._build_smoke_session_with_guards(
                    lambda: SimpleNamespace(runner=SimpleNamespace(
                        load_train=snapshot.load_train, load_test=snapshot.load_test,
                    )), guards,
                )
                self.assertIs(session.runner.load_test.__func__, guards.forbidden_test)
                self.assertIs(session.runner.load_train.__func__, guards.counted_train)
                self.assertIs(session.runner.load_train(module.FIXTURE, context), train)
                with self.assertRaisesRegex(ValueError, "TEST value loader"):
                    session.runner.load_test(module.FIXTURE, context)
        self.assertEqual(counters, {"train": 1, "test_values": 1, "test_labels": 0})

    def test_smoke_writer_heartbeats_during_blocked_compute_and_stops_before_terminal(self) -> None:
        """Removing the heartbeat thread leaves the 94-second MPS phase invisible."""
        from experiments.exp425_selective_recovery.mps_parity import SmokeEvidenceWriter, SmokeHeartbeat

        with TemporaryDirectory() as directory:
            writer = SmokeEvidenceWriter.create(Path(directory) / "smoke", "run", "ECG200_normal_1")
            detail = {
                "dataset": "ECG200_normal_1", "train_instance_count": 2,
                "test_instance_count": 3, "train_length_groups": [{"length": 4, "series_count": 2}],
                "train_value_count": 8, "train_value_bytes": 64,
                "current_phase": "native_train_qualification", "backend": "mps", "device": "mps",
                "completed": 0, "total": 1, "elapsed_seconds": 0.0,
                "throughput_per_second": None, "eta_seconds": None,
                "checkpoint_path": "/tmp/checkpoint", "loader_counters": {"train": 1, "test_values": 0, "test_labels": 0}, "loader_counts": {"train": 1, "test_values": 0, "test_labels": 0},
                "latest_error": None,
            }
            started = Event()
            heartbeat = SmokeHeartbeat(writer, lambda: {"stage": "native_train_qualification", "current_dataset": "ECG200_normal_1", **detail}, interval_seconds=0.01)
            def fake_compute() -> None:
                started.set(); sleep(0.04)
            heartbeat.start(); fake_compute(); heartbeat.stop()
            self.assertTrue(started.is_set())
            writer.terminal_success(**(detail | {"elapsed_seconds": 0.04}))
            events = [json.loads(line) for line in writer.events.read_text().splitlines()]
            sequences = [event["heartbeat_sequence"] for event in events if event["event"] == "heartbeat"]
            self.assertGreaterEqual(len(sequences), 2)
            self.assertEqual(sequences, list(range(1, len(sequences) + 1)))
            self.assertFalse(heartbeat.is_alive())
            progress = json.loads((writer.root / "progress.json").read_bytes())
            for key in ("dataset", "train_instance_count", "test_instance_count", "train_length_groups", "train_value_count", "train_value_bytes", "current_phase", "backend", "device", "checkpoint_path", "loader_counters", "loader_counts", "heartbeat_sequence", "latest_error"):
                self.assertIn(key, progress)
            self.assertIsNone(progress["current_dataset"])
            self.assertEqual((progress["throughput_per_second"], progress["eta_seconds"]), (0, 0))
            parent_log = writer.parent_log.read_text()
            self.assertIn("heartbeat", parent_log)
            self.assertIn("backend=mps", parent_log)
            self.assertIn("checkpoint=/tmp/checkpoint", parent_log)

    def test_train_only_loader_surface_rejects_label_and_evaluation_paths(self) -> None:
        module = self._script_module()
        module._assert_train_only_loader_surface({"load_raw_train_values": object()})
        with self.assertRaisesRegex(ValueError, "label/evaluation"):
            module._assert_train_only_loader_surface({"load_raw_test_labels": object()})

    def test_smoke_writer_failure_preserves_last_operational_context(self) -> None:
        """A failure after MPS starts must not erase its checkpoint and elapsed evidence."""
        from experiments.exp425_selective_recovery.mps_parity import SmokeEvidenceWriter

        with TemporaryDirectory() as directory:
            writer = SmokeEvidenceWriter.create(Path(directory) / "smoke", "run", "ECG200_normal_1")
            writer.progress(
                stage="native_train_qualification", current_dataset="ECG200_normal_1", success=None,
                checkpoint_path="/tmp/checkpoint", elapsed_seconds=1.25,
                throughput_per_second=None, eta_seconds=None, latest_error=None,
            )
            writer.failure(ValueError("blocked"))
            terminal = json.loads((writer.root / "progress.json").read_bytes())
            self.assertEqual(terminal["checkpoint_path"], "/tmp/checkpoint")
            self.assertEqual(terminal["elapsed_seconds"], 1.25)
            self.assertEqual(terminal["current_dataset"], None)
            self.assertEqual((terminal["throughput_per_second"], terminal["eta_seconds"]), (0, 0))

    def test_smoke_writer_early_failure_records_zero_elapsed_terminal_fields(self) -> None:
        from experiments.exp425_selective_recovery.mps_parity import SmokeEvidenceWriter

        with TemporaryDirectory() as directory:
            writer = SmokeEvidenceWriter.create(Path(directory) / "smoke", "run", "ECG200_normal_1")
            writer.failure(ValueError("preflight"))
            terminal = json.loads((writer.root / "progress.json").read_bytes())
            self.assertEqual((terminal["elapsed_seconds"], terminal["throughput_per_second"], terminal["eta_seconds"]), (0, 0, 0))

    def test_smoke_native_commit_claim_binds_both_mps_evidence_digests(self) -> None:
        """The smoke's direct native commit must satisfy the v2 capability seam."""
        module = self._script_module()
        session = SimpleNamespace(process_nonce="nonce")
        store = SimpleNamespace(
            provenance=SimpleNamespace(run_identity="run"), run_context_sha256="a" * 64,
        )
        native = SimpleNamespace(
            trace_sha256="b" * 64, record_payload_sha256="c" * 64,
            backend_audit_sha256="d" * 64,
            mps_adoption=SimpleNamespace(sha256="e" * 64),
            mps_execution=SimpleNamespace(sha256="f" * 64),
        )
        with patch.object(module.capabilities, "_mint", return_value="capability") as mint:
            self.assertEqual(module._native_commit_capability(session, store, native), "capability")
        self.assertEqual(mint.call_args.kwargs, {
            "kind": "native", "run_identity": "run", "dataset": module.FIXTURE,
            "context_sha256": "a" * 64, "trace_sha256": "b" * 64,
            "record_sha256": "c" * 64, "backend_sha256": "d" * 64,
            "mps_adoption_sha256": "e" * 64,
            "mps_execution_sha256": "f" * 64, "process_nonce": "nonce",
        })

    def test_exact_golden_record_decodes_to_an_abstained_dataset_result(self) -> None:
        from experiments.exp425_selective_recovery.mps_parity import validate_golden_reference

        source = Path("/Users/mac/Documents/Dataset/exp424_mps_score_cache_20260802-153831.checkpoint")
        golden = source / "datasets/0a19fece1acc72bd1fd0e2845ee962e54988940a96da37ba5c231fa52e551ce8.json"
        reference = validate_golden_reference(Path.cwd(), source, golden)
        result = reference.record["qualification"]
        self.assertEqual(result.dataset, "ECG200_normal_1")
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.candidates, ())
    def test_source_hashes_reject_a_mutated_frozen_runner(self) -> None:
        from experiments.exp425_selective_recovery.mps_parity import verify_source_hashes

        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "runner.py"
            path.write_bytes(b"frozen\n")
            expected = sha256(b"frozen\n").hexdigest()
            self.assertEqual(verify_source_hashes(root, {"runner.py": expected}), {"runner.py": expected})
            path.write_bytes(b"mutated\n")
            with self.assertRaisesRegex(ValueError, "source digest"):
                verify_source_hashes(root, {"runner.py": expected})

    def test_normalized_digest_has_a_closed_scientific_allowlist(self) -> None:
        from experiments.exp425_selective_recovery.mps_parity import (
            NORMALIZED_SHARED_KEYS,
            normalized_digest,
        )

        payload = {key: "x" for key in NORMALIZED_SHARED_KEYS}
        digest = normalized_digest(payload)
        self.assertEqual(len(digest), 64)
        with self.assertRaisesRegex(ValueError, "allowlist"):
            normalized_digest(payload | {"timing": 1})
        with self.assertRaisesRegex(ValueError, "allowlist"):
            normalized_digest({key: value for key, value in payload.items() if key != "fallback"})

    def test_comparison_rejects_model_digest_loader_and_qualification_mutations(self) -> None:
        from experiments.exp425_selective_recovery.mps_parity import compare_shared_science

        common = {
            "dataset": "ECG200_normal_1", "database_sha256": "a" * 64,
            "raw_train_sha256": "b" * 64, "repaired_train_sha256": "c" * 64,
            "length_identity_sha256": {"96": "d" * 64},
            "config": {"alpha": 0.01, "model_seeds": [41701, 41702, 41703], "perturbation_seeds": [41711, 41712, 41713], "batch_size": 32, "query_chunk": 256},
            "series_count": 106, "lengths": [96],
            "final_model_hashes": {"96": {"41701": "e" * 64}},
            "legacy_oof_digest": {"96": "f" * 64},
            "qualification": [{"length": 96, "status": "failed", "finite": True, "scalar": 0.5}],
            "status": "abstained", "candidate_coordinates": [], "finiteness": True, "fallback": False,
        }
        self.assertTrue(compare_shared_science(common, common, tolerance=0.0001)["passed"])
        changed = json.loads(json.dumps(common)); changed["final_model_hashes"]["96"]["41701"] = "0" * 64
        self.assertFalse(compare_shared_science(common, changed, tolerance=0.0001)["passed"])
        changed = json.loads(json.dumps(common)); changed["qualification"][0]["scalar"] += 0.001
        self.assertFalse(compare_shared_science(common, changed, tolerance=0.0001)["passed"])

    def test_atomic_writer_is_no_clobber_and_clears_terminal_current_dataset(self) -> None:
        from experiments.exp425_selective_recovery.mps_parity import SmokeEvidenceWriter

        with TemporaryDirectory() as directory:
            root = Path(directory) / "smoke"
            writer = SmokeEvidenceWriter.create(root, "run", "ECG200_normal_1")
            writer.progress(stage="train", current_dataset="ECG200_normal_1", success=None)
            writer.failure(ValueError("boom"))
            terminal = json.loads((root / "progress.json").read_bytes())
            self.assertEqual(terminal["current_dataset"], None)
            self.assertFalse(terminal["success"])
            self.assertTrue((root / "events.jsonl").read_bytes().endswith(b"\n"))
            with self.assertRaisesRegex(ValueError, "no-clobber"):
                SmokeEvidenceWriter.create(root, "other", "ECG200_normal_1")

    def test_review_only_assessment_preserves_the_authenticated_non_adoption_speed(self) -> None:
        from experiments.exp425_selective_recovery.mps_parity import authenticated_cpu_audit, review_only_mps_assessment

        audit, assessment = authenticated_cpu_audit(), review_only_mps_assessment()
        self.assertEqual(audit.backend, "cpu")
        self.assertFalse(audit.fallback); self.assertFalse(assessment.adopted)
        self.assertEqual(assessment.observed_speedup, 0.6709759390840965)


if __name__ == "__main__":
    unittest.main()
