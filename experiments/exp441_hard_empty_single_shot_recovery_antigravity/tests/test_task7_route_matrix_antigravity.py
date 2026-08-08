"""Production-shaped Task 7 route and malformed-readiness matrix."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.config_antigravity import FROZEN_CONFIG


def _hashes_antigravity() -> dict[str, str]:
    return {"context":FROZEN_CONFIG.checkpoint_context_sha256,"state":FROZEN_CONFIG.checkpoint_state_sha256,"retained_inventory":FROZEN_CONFIG.checkpoint_retained_inventory_sha256,"publication_prediction":FROZEN_CONFIG.publication_prediction_sha256,"publication_manifest":FROZEN_CONFIG.publication_manifest_sha256,"publication_artifact_map":FROZEN_CONFIG.publication_artifact_map_sha256,"hard_prediction":FROZEN_CONFIG.hard_prediction_sha256,"hard_manifest":FROZEN_CONFIG.hard_manifest_sha256,"hard_coordinate_identity":FROZEN_CONFIG.hard_coordinate_identity_sha256}


class Task7RouteMatrixAntigravityTests(unittest.TestCase):
    def _backend(self, readiness):
        return readiness.BackendAssessmentAntigravity("cpu",11,False,"not_run_inapplicable","inapplicable_no_tensor_stage","inapplicable_no_tensor_stage","inapplicable_no_tensor_stage",{"numpy_version":"2.4.6"})

    def _payload(self, readiness, commit="a"*40):
        source={"context_dataset_count":1117,"record_digest_count":1117,"hard_coordinate_count":2218,"provisional_candidate_count":840,"label_access_count":0,"source_hashes":_hashes_antigravity()}
        selection={"provisional_candidate_count":840,"outside_hard_candidate_count":394,"eligible_candidate_count":73,"winner_count":49,"selected_candidate_count":30}
        return readiness._readiness_payload_antigravity(commit,source,selection,self._backend(readiness)),source,selection

    def _write_root(self, root, readiness, payload):
        root.mkdir(mode=0o700); path=root/"readiness_antigravity.json"; path.write_text(json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n");path.chmod(0o600)

    def test_validate_rejects_initial_namespace_and_byte_boundary_matrix(self):
        """Symlinks, FIFO, mode, inventory, and encoding cannot become a readiness binding."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as r
        payload,source,selection=self._payload(r)
        mutations=("root_symlink","file_symlink","fifo","root_mode","file_mode","extra","noncanonical")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/"run_antigravity"
                if mutation=="root_symlink":
                    root.symlink_to(Path(tmp)/"missing")
                else:
                    self._write_root(root,r,payload); file=root/"readiness_antigravity.json"
                    if mutation=="file_symlink": file.unlink(); file.symlink_to(Path(tmp)/"other")
                    elif mutation=="fifo": file.unlink(); __import__("os").mkfifo(file)
                    elif mutation=="root_mode": root.chmod(0o755)
                    elif mutation=="file_mode": file.chmod(0o644)
                    elif mutation=="extra": (root/"publication_antigravity").symlink_to(Path(tmp)/"absent")
                    elif mutation=="noncanonical": file.write_text(json.dumps(payload,indent=2)+"\n")
                with ExitStack() as stack:
                    stack.enter_context(patch.object(r,"_RUN_ROOT_ANTIGRAVITY",root));stack.enter_context(patch.object(r,"_current_clean_head_antigravity",return_value="a"*40));stack.enter_context(patch.object(r,"verify_frozen_sources_antigravity",return_value=object()));stack.enter_context(patch.object(r,"_source_payload_antigravity",return_value=source));stack.enter_context(patch.object(r,"_selection_payload_antigravity",return_value=selection));stack.enter_context(patch.object(r,"_backend_antigravity",return_value=self._backend(r)))
                    with self.assertRaises((ValueError,OSError)): r.validate_readiness_for_run_antigravity("a"*40)

    def test_payload_subtests_reject_all_governance_source_and_backend_mutations(self):
        """Every frozen governing value has a key/type/value rejection path."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as r
        payload,_,_=self._payload(r)
        for key,value in list(r.governance_payload_antigravity().items()):
            for kind,replacement in (("missing",None),("type",{"x":value}),("value",(not value) if type(value)is bool else value+"x")):
                with self.subTest(key=key,kind=kind):
                    changed=dict(payload)
                    if kind=="missing":del changed[key]
                    else:changed[key]=replacement
                    with self.assertRaises(ValueError):r._validate_readiness_payload_antigravity(changed)
        for key in ("source_snapshot_sha256","database_sha256","database_path"):
            with self.subTest(key=key):
                changed=dict(payload);changed[key]=0
                with self.assertRaises(ValueError):r._validate_readiness_payload_antigravity(changed)
        for key in ("workers","canonical_backend","tensor_eligible"):
            with self.subTest(backend=key):
                changed=dict(payload);backend=dict(payload["backend"]);backend[key]=12 if key=="workers" else ("numba" if key=="canonical_backend" else True);changed["backend"]=backend
                with self.assertRaises(ValueError):r._validate_readiness_payload_antigravity(changed)

    def test_cli_mocked_new_inspect_evaluate_have_only_canonical_stdout_and_zero(self):
        """Every successful public route returns int zero and leaks no repr/log output."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.readiness_antigravity as readiness
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.reporting_antigravity as reporting
        binding=types.SimpleNamespace(reviewed_commit="a"*40)
        evalname="experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity"; fake=types.ModuleType(evalname);fake.evaluate_exp441_antigravity=lambda:{"evaluation":"ok"}
        with patch.object(cli,"current_clean_head_antigravity",return_value="a"*40),patch.object(readiness,"validate_readiness_for_run_antigravity",return_value=binding),patch.object(runtime,"run_exp441_new_antigravity",return_value={"attempt":1}),patch.object(reporting,"publish_exp441_antigravity",return_value={"publish":"ok"}),patch.object(reporting,"inspect_exp441_publication_antigravity",return_value={"inspect":"ok"}),patch.dict(sys.modules,{evalname:fake}):
            for route,expected in (("new",{"publish":"ok"}),("inspect",{"inspect":"ok"}),("evaluate",{"evaluation":"ok"})):
                with self.subTest(route=route):
                    out=io.StringIO()
                    with redirect_stdout(out):self.assertEqual(cli.main_antigravity([route]),0)
                    self.assertEqual(json.loads(out.getvalue()),{"command":route,"result":expected})
                    self.assertEqual(out.getvalue().encode(),json.dumps(json.loads(out.getvalue()),sort_keys=True,separators=(",",":")).encode()+b"\n")

    def test_fresh_interpreter_prediction_routes_firewall_and_outside_cwd_help(self):
        """Route-level import sentinels prove help/readiness/new/inspect never touch label paths."""
        root=Path(__file__).resolve().parents[3]; wrapper=root/"scratch"/"run_exp441_hard_empty_single_shot_recovery_antigravity.py"
        code='''import importlib.abc,sqlite3,sys,types\nsqlite3.connect=lambda *a,**k:(_ for _ in()).throw(AssertionError("sqlite"))\nblocked={"experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity","experiments.exp440_train_only_low_frequency_precision_gate_antigravity.evaluation_antigravity","experiments.exp417_causal_multihorizon_shadow.inputs"}\nclass F(importlib.abc.MetaPathFinder):\n def find_spec(self,n,p=None,t=None):\n  if n in blocked: raise AssertionError(n)\nsys.meta_path.insert(0,F());b="experiments.exp441_hard_empty_single_shot_recovery_antigravity."\nr=types.ModuleType(b+"readiness_antigravity");r.current_clean_head_antigravity=lambda:"a"*40;r.run_readiness_antigravity=lambda x:{};r.validate_readiness_for_run_antigravity=lambda x:types.SimpleNamespace(reviewed_commit=x)\nt=types.ModuleType(b+"runtime_antigravity");t.run_exp441_new_antigravity=lambda **k:{}\np=types.ModuleType(b+"reporting_antigravity");p.publish_exp441_antigravity=lambda x:{};p.inspect_exp441_publication_antigravity=lambda:{}\nsys.modules.update({b+"readiness_antigravity":r,b+"runtime_antigravity":t,b+"reporting_antigravity":p})\nfrom experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as c\nfor x in("readiness","new","inspect"):assert c.main_antigravity([x])==0\n'''
        subprocess.run([sys.executable,"-c",code],cwd=root,check=True,capture_output=True,text=True)
        result=subprocess.run([sys.executable,str(wrapper),"--help"],cwd="/tmp",check=True,capture_output=True,text=True)
        self.assertIn("{readiness,new,inspect,evaluate}",result.stdout)
