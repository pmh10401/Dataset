from __future__ import annotations
import hashlib, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from .test_evaluation_antigravity import SHA, fixture_antigravity

class ReplayAntigravityTest(unittest.TestCase):
    def rewrite(self, root, name, mutate):
        canonical=lambda x:json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()+b"\n"
        path=root/name; value=json.loads(path.read_bytes()); mutate(value); path.write_bytes(canonical(value))
        artifact_path=root/"artifact_map_antigravity.json"; artifact=json.loads(artifact_path.read_bytes()); artifact["leaf_sha256"][name]=hashlib.sha256(path.read_bytes()).hexdigest(); artifact_path.write_bytes(canonical(artifact))
        manifest_path=root/"manifest_antigravity.json"; manifest=json.loads(manifest_path.read_bytes()); manifest["artifact_map_sha256"]=hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if name=="result_antigravity.json": manifest["result_sha256"]=artifact["leaf_sha256"][name]
        if name=="dashboard_antigravity.json": manifest["dashboard_sha256"]=artifact["leaf_sha256"][name]
        manifest_path.write_bytes(canonical(manifest))

    def test_source_free_replay_and_inventory_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); result=_evaluate_with_dependencies_antigravity(authority,SHA,deps)
            with patch("experiments.exp451_rdst_local_gap_qualification_antigravity.sources_antigravity.open_ordered_label_session_antigravity",side_effect=AssertionError), patch("experiments.exp451_rdst_local_gap_qualification_antigravity.predecessor_antigravity.verify_exp448_predecessor_antigravity",side_effect=AssertionError):
                self.assertEqual(inspect_replay_antigravity(root,expected_prediction_authority=authority,expected_predecessor_authority=deps.predecessor_loader()),result)
            (root/"extra_antigravity.json").write_text("{}")
            with self.assertRaises(ValueError): inspect_replay_antigravity(root,expected_prediction_authority=authority,expected_predecessor_authority=deps.predecessor_loader())

    def test_rehashed_scientific_and_exact_type_mutations_reject_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        mutations=(("datasets_antigravity.json",lambda x:x["rows"][0].__setitem__("new_detections",2)),("result_antigravity.json",lambda x:x.__setitem__("label_row_count",True)),("result_antigravity.json",lambda x:x.__setitem__("target_baseline_fn",954.0)))
        for name,mutation in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps); self.rewrite(root,name,mutation)
                with self.assertRaises(ValueError): inspect_replay_antigravity(root,expected_prediction_authority=authority,expected_predecessor_authority=deps.predecessor_loader())

    def test_expected_authorities_are_independent_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            forged=type(authority)("b"*40,authority.dataset_identities,authority.target_lengths,authority.candidate_indices)
            with self.assertRaises(ValueError): inspect_replay_antigravity(root,expected_prediction_authority=forged,expected_predecessor_authority=deps.predecessor_loader())

if __name__=="__main__": unittest.main()
