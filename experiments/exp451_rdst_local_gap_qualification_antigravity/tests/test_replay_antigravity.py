from __future__ import annotations
import json, os, tempfile, unittest
import hashlib
import stat
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
                self.assertEqual(inspect_replay_antigravity(root, expected_prediction_authority=authority, expected_predecessor_authority=deps.predecessor_loader()), result)
            forged = type(authority)("b" * 40, authority.dataset_identities, authority.target_lengths, authority.candidate_indices)
            with self.assertRaises(ValueError): inspect_replay_antigravity(root, expected_prediction_authority=forged, expected_predecessor_authority=deps.predecessor_loader())
            (root/"extra_antigravity.json").write_text("{}")
            with self.assertRaises(ValueError): inspect_replay_antigravity(root, expected_prediction_authority=authority, expected_predecessor_authority=deps.predecessor_loader())
            (root/"extra_antigravity.json").unlink()
            leaf=root/"datasets_antigravity.json"; payload=leaf.read_bytes(); leaf.write_bytes(payload.replace(b'"tp":1', b'"tp":2', 1))
            with self.assertRaises(ValueError): inspect_replay_antigravity(root, expected_prediction_authority=authority, expected_predecessor_authority=deps.predecessor_loader())

    def test_rejects_symlink_and_hardlink_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            manifest=root/"manifest_antigravity.json"; backup=root/"manifest_copy_antigravity.json"; os.link(manifest,backup)
            with self.assertRaises(ValueError): inspect_replay_antigravity(root, expected_prediction_authority=authority, expected_predecessor_authority=deps.predecessor_loader())

    def test_pre_promotion_write_failure_leaves_final_root_absent_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_antigravity as evaluation
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root)
            original=evaluation._write_leaf_at_antigravity
            calls=0
            def fail_second(directory_fd, name, payload):
                nonlocal calls; calls += 1
                if calls == 2: raise OSError("injected write failure")
                return original(directory_fd,name,payload)
            with patch.object(evaluation,"_write_leaf_at_antigravity",side_effect=fail_second), self.assertRaises(OSError):
                _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            self.assertFalse(root.exists()); self.assertFalse(root.is_symlink())

    def test_rehashed_result_predecessor_authority_drift_is_rejected_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            self._rewrite_and_rehash_antigravity(root,"result_antigravity.json",lambda value:value.__setitem__("predecessor_hashes",["9"*64,*value["predecessor_hashes"][1:]]))
            with self.assertRaises(ValueError): inspect_replay_antigravity(root,expected_prediction_authority=authority,expected_predecessor_authority=deps.predecessor_loader())

    def test_actual_symlink_leaf_and_rehashed_transition_forgery_are_rejected_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            dataset=root/"datasets_antigravity.json"; outside=Path(directory)/"outside_antigravity.json"; outside.write_bytes(dataset.read_bytes()); dataset.unlink(); dataset.symlink_to(outside)
            with self.assertRaises((OSError,ValueError)): inspect_replay_antigravity(root,expected_prediction_authority=authority,expected_predecessor_authority=deps.predecessor_loader())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            self._rewrite_and_rehash_antigravity(root,"datasets_antigravity.json",lambda value:value["rows"][0].__setitem__("new_detections",2))
            with self.assertRaises(ValueError): inspect_replay_antigravity(root,expected_prediction_authority=authority,expected_predecessor_authority=deps.predecessor_loader())

    def test_post_promotion_parent_fsync_failure_moves_exact_root_to_forensic_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_antigravity as evaluation
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); original=evaluation.os.fsync
            def fail_public_parent(fd):
                if root.exists() and stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError("injected parent fsync failure")
                return original(fd)
            with patch.object(evaluation.os,"fsync",side_effect=fail_public_parent), self.assertRaises(OSError): _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            self.assertFalse(root.exists()); self.assertEqual(len(tuple(Path(directory).glob(".*forensic.*_antigravity"))),1)

    def test_forensic_atomic_collision_retries_and_preserves_primary_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_antigravity as evaluation
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); original_fsync=evaluation.os.fsync; original_rename=evaluation._rename_no_replace_antigravity; renames=0
            def fail_public_parent(fd):
                if root.exists() and stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError("primary parent fsync")
                return original_fsync(fd)
            def collide_once(parent_fd,source,destination):
                nonlocal renames; renames+=1
                if renames==2: raise FileExistsError("atomic collision")
                return original_rename(parent_fd,source,destination)
            with patch.object(evaluation.os,"fsync",side_effect=fail_public_parent), patch.object(evaluation,"_rename_no_replace_antigravity",side_effect=collide_once), self.assertRaisesRegex(OSError,"primary parent fsync"):
                _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            self.assertGreaterEqual(renames,3); self.assertFalse(root.exists()); self.assertEqual(len(tuple(Path(directory).glob(".*forensic.*_antigravity"))),1)

    def test_pre_promotion_fault_table_restores_fds_and_leaves_final_absent_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_antigravity as evaluation
        from experiments.exp451_rdst_local_gap_qualification_antigravity import replay_antigravity as replay
        from experiments.exp451_rdst_local_gap_qualification_antigravity import dashboard_export_antigravity as dashboard
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        cases=("mkdir","staging_open","dashboard","leaf","private_replay","promotion")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); baseline=len(os.listdir("/dev/fd")); stack=[]
                if case=="mkdir": stack.append(patch.object(evaluation.os,"mkdir",side_effect=OSError("mkdir fault")))
                elif case=="staging_open":
                    original=evaluation.os.open
                    def open_fault(path,flags,*args,**kwargs):
                        if isinstance(path,str) and "staging_antigravity" in path and flags & os.O_DIRECTORY: raise OSError("staging open fault")
                        return original(path,flags,*args,**kwargs)
                    stack.append(patch.object(evaluation.os,"open",side_effect=open_fault))
                elif case=="dashboard": stack.append(patch.object(dashboard,"public_result_antigravity",side_effect=ValueError("dashboard fault")))
                elif case=="leaf": stack.append(patch.object(evaluation,"_write_leaf_at_antigravity",side_effect=OSError("leaf fault")))
                elif case=="private_replay": stack.append(patch.object(replay,"inspect_replay_antigravity",side_effect=ValueError("private replay fault")))
                else: stack.append(patch.object(evaluation,"_rename_no_replace_antigravity",side_effect=OSError("promotion fault")))
                for item in stack: item.start()
                try:
                    with self.assertRaises(Exception): _evaluate_with_dependencies_antigravity(authority,SHA,deps)
                finally:
                    for item in reversed(stack): item.stop()
                self.assertFalse(root.exists()); self.assertFalse(root.is_symlink()); self.assertEqual(len(os.listdir("/dev/fd")),baseline)

    def test_replay_schema_authority_failure_table_has_no_fd_leak_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        cases=("manifest","artifact","context","datasets","result","dashboard","authority")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps); baseline=len(os.listdir("/dev/fd")); expected=authority
                if case=="manifest":
                    value=json.loads((root/"manifest_antigravity.json").read_bytes()); value["extra"]=1; (root/"manifest_antigravity.json").write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n")
                elif case=="artifact":
                    value=json.loads((root/"artifact_map_antigravity.json").read_bytes()); value["leaf_sha256"].pop("context_antigravity.json"); (root/"artifact_map_antigravity.json").write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n")
                elif case in {"context","datasets","result","dashboard"}: self._rewrite_and_rehash_antigravity(root,f"{case}_antigravity.json",lambda value:value.__setitem__("extra",1))
                else: expected=type(authority)("b"*40,authority.dataset_identities,authority.target_lengths,authority.candidate_indices)
                with self.assertRaises((ValueError,OSError)): inspect_replay_antigravity(root,expected_prediction_authority=expected,expected_predecessor_authority=deps.predecessor_loader())
                self.assertEqual(len(os.listdir("/dev/fd")),baseline)

    def test_public_replay_failure_forensics_and_unrecoverable_rename_classification_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_antigravity as evaluation
        from experiments.exp451_rdst_local_gap_qualification_antigravity import replay_antigravity as replay
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity, PublicationInvariantErrorAntigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); original=replay.inspect_replay_antigravity; calls=0
            def fail_public(*args,**kwargs):
                nonlocal calls; calls+=1
                if calls==2: raise ValueError("public replay fault")
                return original(*args,**kwargs)
            with patch.object(replay,"inspect_replay_antigravity",side_effect=fail_public), self.assertRaisesRegex(ValueError,"public replay fault"): _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            self.assertFalse(root.exists()); self.assertEqual(len(tuple(Path(directory).glob(".*forensic.*_antigravity"))),1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); original_fsync=evaluation.os.fsync; original_rename=evaluation._rename_no_replace_antigravity; calls=0
            def fail_parent(fd):
                if root.exists() and stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError("primary fsync")
                return original_fsync(fd)
            def fail_forensic(parent_fd,source,destination):
                nonlocal calls; calls+=1
                if calls>=2: raise OSError("forensic rename fault")
                return original_rename(parent_fd,source,destination)
            with patch.object(evaluation.os,"fsync",side_effect=fail_parent), patch.object(evaluation,"_rename_no_replace_antigravity",side_effect=fail_forensic), self.assertRaises(PublicationInvariantErrorAntigravity): _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            self.assertTrue(root.exists())

    def test_each_leaf_and_staging_directory_fsync_failure_is_atomic_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_antigravity as evaluation
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        names=("context_antigravity.json","datasets_antigravity.json","result_antigravity.json","dashboard_antigravity.json","artifact_map_antigravity.json","manifest_antigravity.json")
        for failed_name in names:
            with self.subTest(leaf=failed_name), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); original=evaluation._write_leaf_at_antigravity
                def fail_leaf(fd,name,payload):
                    if name==failed_name: raise OSError(f"{name} write fault")
                    return original(fd,name,payload)
                with patch.object(evaluation,"_write_leaf_at_antigravity",side_effect=fail_leaf), self.assertRaises(OSError): _evaluate_with_dependencies_antigravity(authority,SHA,deps)
                self.assertFalse(root.exists())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); original=evaluation.os.fsync
            def fail_directory(fd):
                if stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError("staging directory fsync fault")
                return original(fd)
            with patch.object(evaluation.os,"fsync",side_effect=fail_directory), self.assertRaises(OSError): _evaluate_with_dependencies_antigravity(authority,SHA,deps)
            self.assertFalse(root.exists())

    def test_replay_descriptor_close_failure_is_observable_without_leak_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import replay_antigravity as replay
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps); baseline=len(os.listdir("/dev/fd")); original=replay.os.close; injected=False
            def close_fault(fd):
                nonlocal injected; original(fd)
                if not injected: injected=True; raise OSError("injected replay close failure")
            with patch.object(replay.os,"close",side_effect=close_fault), self.assertRaises((OSError,ExceptionGroup)):
                replay.inspect_replay_antigravity(root,expected_prediction_authority=authority,expected_predecessor_authority=deps.predecessor_loader())
            self.assertEqual(len(os.listdir("/dev/fd")),baseline)

    def test_verified_publication_success_close_failure_is_nonfatal_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_antigravity as evaluation
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); original=evaluation._close_publication_fd_antigravity; injected=False
            def close_after(fd):
                nonlocal injected; original(fd)
                if root.exists() and not injected: injected=True; raise OSError("verified-success close")
            with patch.object(evaluation,"_close_publication_fd_antigravity",side_effect=close_after): result=_evaluate_with_dependencies_antigravity(authority,SHA,deps)
            self.assertEqual(result.global_dataset_count,1117); self.assertTrue(root.is_dir()); self.assertTrue(injected)

    def test_result_bool_as_int_and_int_as_float_rehash_are_rejected_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.replay_antigravity import inspect_replay_antigravity
        for field,bad in (("label_row_count",True),("target_baseline_fn",954.0)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/"evaluation_antigravity"; authority,deps,_=fixture_antigravity(root); _evaluate_with_dependencies_antigravity(authority,SHA,deps)
                self._rewrite_and_rehash_antigravity(root,"result_antigravity.json",lambda value:value.__setitem__(field,bad))
                with self.assertRaises(ValueError): inspect_replay_antigravity(root,expected_prediction_authority=authority,expected_predecessor_authority=deps.predecessor_loader())

if __name__ == "__main__": unittest.main()
