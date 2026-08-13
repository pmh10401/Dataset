import json
import os
import time
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np

from experiments.exp451_rdst_local_gap_qualification_antigravity.backend_gate_antigravity import BackendDecisionAntigravity, BackendProbeAntigravity, decide_backend_antigravity
from experiments.exp451_rdst_local_gap_qualification_antigravity.contracts_antigravity import CandidateScoresAntigravity, DatasetBundleAntigravity
from experiments.exp451_rdst_local_gap_qualification_antigravity.prediction_antigravity import (
    PredictionDependenciesAntigravity, _run_prediction_with_dependencies_antigravity,
    _validate_frozen_prediction_root_antigravity,
)


SHA = "a" * 40

def decision():
    probe = BackendProbeAntigravity("mlx", False, False, False, 0.0, 1.0, 1.0, (("mlx", "none"),), "test", "no", False)
    return decide_backend_antigravity(probe)

def bundles():
    return tuple(DatasetBundleAntigravity(i, "Family", f"Dataset_{i}_antigravity", np.arange(20, dtype=float).reshape(5,4) + i*100, np.arange(12, dtype=float).reshape(3,4) + i*100, 3) for i in range(48))

def scorer(train, test, seed, backend):
    time.sleep((48 - int(train[0, 0]) // 100) * 0.0005)
    return CandidateScoresAntigravity(np.arange(len(train), dtype=float), np.arange(len(test), dtype=float), len(train), len(test), 3, 1, seed)

def dying_scorer(train, test, seed, backend):
    if int(train[0, 0]) // 100 == 12: os._exit(17)
    return scorer(train, test, seed, backend)

def nonfinite_scorer(train, test, seed, backend):
    return CandidateScoresAntigravity(np.arange(len(train), dtype=float), np.full(len(test), np.nan), len(train), len(test), 3, 1, seed)


class PredictionTestsAntigravity(unittest.TestCase):
    def dependencies(self, root, fault=None):
        return PredictionDependenciesAntigravity(fault, lambda: None, decision(), root=root, bundles=bundles(), scorer=scorer, logical_cpus=12)

    def test_synthetic_48_freeze_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            result = _run_prediction_with_dependencies_antigravity(SHA, self.dependencies(root))
            self.assertEqual((result.progress.completed, result.progress.reused, result.progress.recomputed), (48, 0, 48))
            self.assertEqual(len(result.authority.dataset_identities), 48)
            self.assertEqual(_validate_frozen_prediction_root_antigravity(root, result.authority), result.authority)
            self.assertEqual(len(tuple((root / "leaves_antigravity").iterdir())), 48)
            self.assertFalse(json.loads((root / "pid_antigravity.json").read_text())["active"])
            self.assertEqual(json.loads((root / "checkpoint_antigravity.json").read_text())["next_sequence"], 48)
            manifest = json.loads((root / "manifest_antigravity.json").read_text())
            self.assertIn("probes", manifest["backend_decision"])
            self.assertEqual(len(tuple((root / "receipts_antigravity").iterdir())), 48)
            progress = json.loads((root / "progress_antigravity.json").read_text())
            self.assertIsNone(progress["seed"]); self.assertIsNone(progress["fold"])

    def test_interruption_resumes_prefix_but_terminal_failure_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            def interrupt(operation, sequence):
                if operation == "after_leaf" and sequence == 11: raise KeyboardInterrupt("stop")
            with self.assertRaises(KeyboardInterrupt): _run_prediction_with_dependencies_antigravity(SHA, self.dependencies(root, interrupt))
            result = _run_prediction_with_dependencies_antigravity(SHA, self.dependencies(root))
            self.assertEqual((result.progress.reused, result.progress.recomputed), (12, 36))
            with self.assertRaises(ValueError): _run_prediction_with_dependencies_antigravity(SHA, self.dependencies(root))

    def test_nonfinite_and_descriptor_attack_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            deps = replace(self.dependencies(root), scorer=nonfinite_scorer)
            with self.assertRaises(ValueError): _run_prediction_with_dependencies_antigravity(SHA, deps)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"; result = _run_prediction_with_dependencies_antigravity(SHA, self.dependencies(root))
            leaf = root / "leaves_antigravity" / "0000_antigravity.json"
            leaf.unlink(); leaf.symlink_to(root / "manifest_antigravity.json")
            with self.assertRaises(ValueError): _validate_frozen_prediction_root_antigravity(root, result.authority)

    def test_real_process_pool_finishes_out_of_order_but_commits_ordered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"; completions=[]
            deps = replace(self.dependencies(root), logical_cpus=4, completion_observer=lambda sequence, pid: completions.append((sequence,pid)))
            _run_prediction_with_dependencies_antigravity(SHA, deps)
            self.assertGreater(len({pid for _, pid in completions}), 1)
            self.assertNotEqual([sequence for sequence,_ in completions], list(range(48)))
            self.assertEqual([json.loads(path.read_text())["sequence"] for path in sorted((root/"leaves_antigravity").iterdir())], list(range(48)))

    def test_worker_death_is_terminal_with_inactive_pid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            with self.assertRaises(Exception): _run_prediction_with_dependencies_antigravity(SHA, replace(self.dependencies(root), scorer=dying_scorer, logical_cpus=3))
            self.assertTrue((root/"failure_antigravity.json").exists())
            self.assertFalse(json.loads((root/"pid_antigravity.json").read_text())["active"])
            with self.assertRaises(ValueError): _run_prediction_with_dependencies_antigravity(SHA, self.dependencies(root))

    def test_recovery_rejects_exact_key_forgery_and_extra_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"
            def interrupt(operation, sequence):
                if operation == "after_leaf" and sequence == 11: raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt): _run_prediction_with_dependencies_antigravity(SHA, self.dependencies(root, interrupt))
            leaf = root/"leaves_antigravity"/"0000_antigravity.json"; value=json.loads(leaf.read_text()); value["alpha"]=0.01; leaf.write_text(json.dumps(value,sort_keys=True,separators=(",", ":"))+"\n")
            with self.assertRaises(ValueError): _run_prediction_with_dependencies_antigravity(SHA, self.dependencies(root))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run_antigravity"; result=_run_prediction_with_dependencies_antigravity(SHA,self.dependencies(root)); (root/"extra_antigravity.json").write_text("{}\n")
            with self.assertRaises(ValueError): _validate_frozen_prediction_root_antigravity(root,result.authority)

    def test_receipt_chain_independently_authenticates_leaf(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"run_antigravity"; result=_run_prediction_with_dependencies_antigravity(SHA,self.dependencies(root))
            receipt=root/"receipts_antigravity"/"0012_antigravity.json"; value=json.loads(receipt.read_text()); value["previous_chain_sha256"]="0"*64; receipt.write_text(json.dumps(value,sort_keys=True,separators=(",", ":"))+"\n")
            with self.assertRaises(ValueError): _validate_frozen_prediction_root_antigravity(root,result.authority)

    def test_manifest_is_publication_last(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"run_antigravity"
            def fault(operation, sequence):
                if operation == "publish_manifest": raise OSError("publication fault")
            with self.assertRaises(OSError): _run_prediction_with_dependencies_antigravity(SHA,self.dependencies(root,fault))
            self.assertFalse((root/"manifest_antigravity.json").exists())
            self.assertTrue((root/"failure_antigravity.json").exists())
            self.assertFalse(json.loads((root/"pid_antigravity.json").read_text())["active"])


if __name__ == "__main__": unittest.main()
