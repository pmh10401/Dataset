"""Focused fail-closed contract tests for the Exp425 Task 5 MPS smoke."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class MPSParityContractTests(unittest.TestCase):
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
