from __future__ import annotations
import json, os, tempfile, unittest
import hashlib
from pathlib import Path
from unittest.mock import patch
from .test_evaluation_antigravity import SHA, fixture_antigravity

class ReplayAntigravityTest(unittest.TestCase):
    def _rewrite_and_rehash_antigravity(self, root, name, mutate):
        path=root/name; value=json.loads(path.read_bytes()); mutate(value)
        encode=lambda item: json.dumps(item,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()+b"\n"
        path.write_bytes(encode(value))
        artifact_path=root/"artifact_map_antigravity.json"; artifact=json.loads(artifact_path.read_bytes()); artifact["leaf_sha256"][name]=hashlib.sha256(path.read_bytes()).hexdigest(); artifact_path.write_bytes(encode(artifact))
        manifest_path=root/"manifest_antigravity.json"; manifest=json.loads(manifest_path.read_bytes()); manifest["artifact_map_sha256"]=hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if name=="result_antigravity.json": manifest["result_sha256"]=artifact["leaf_sha256"][name]
        if name=="dashboard_antigravity.json": manifest["dashboard_sha256"]=artifact["leaf_sha256"][name]
        manifest_path.write_bytes(encode(manifest))

    def test_source_free_replay_equals_evaluation_and_rejects_mutation_inventory_links_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"evaluation_antigravity"; authority,deps,_ = fixture_antigravity(root)
            result = _evaluate_with_dependencies_antigravity(authority, SHA, deps)
            with patch("experiments.exp451_rdst_local_gap_qualification_antigravity.sources_antigravity.open_ordered_label_session_antigravity", side_effect=AssertionError), patch("experiments.exp451_rdst_local_gap_qualification_antigravity.predecessor_antigravity.verify_exp448_predecessor_antigravity", side_effect=AssertionError):
                self.assertEqual(inspect_replay_antigravity(root, expected_prediction_authority=authority), result)
            forged = type(authority)("b" * 40, authority.dataset_identities, authority.target_lengths, authority.candidate_indices)
            with self.assertRaises(ValueError): inspect_replay_antigravity(root, expected_prediction_authority=forged)
            (root/"extra_antigravity.json").write_text("{}")
            with self.assertRaises(ValueError): inspect_replay_antigravity(root, expected_prediction_authority=authority)
            (root/"extra_antigravity.json").unlink()
            leaf=root/"datasets_antigravity.json"; payload=leaf.read_bytes(); leaf.write_bytes(payload.replace(b'"tp":1', b'"tp":2', 1))
            with self.assertRaises(ValueError): inspect_replay_antigravity(root, expected_prediction_authority=authority)

    def test_rejects_symlink_and_hardlink_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            manifest=root/"manifest_antigravity.json"; backup=root/"manifest_copy_antigravity.json"; os.link(manifest,backup)
            with self.assertRaises(ValueError): inspect_replay_antigravity(root, expected_prediction_authority=authority)

    def test_pre_promotion_write_failure_leaves_final_root_absent_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_antigravity as evaluation
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root)
            original=evaluation.write_create_only_antigravity
            calls=0
            def fail_second(path, payload):
                nonlocal calls; calls += 1
                if calls == 2: raise OSError("injected write failure")
                return original(path,payload)
            with patch.object(evaluation,"write_create_only_antigravity",side_effect=fail_second), self.assertRaises(OSError):
                _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            self.assertFalse(root.exists()); self.assertFalse(root.is_symlink())

    def test_rehashed_result_predecessor_authority_drift_is_rejected_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            self._rewrite_and_rehash_antigravity(root,"result_antigravity.json",lambda value:value.__setitem__("predecessor_hashes",["9"*64,*value["predecessor_hashes"][1:]]))
            with self.assertRaises(ValueError): inspect_replay_antigravity(root,expected_prediction_authority=authority)

if __name__ == "__main__": unittest.main()
