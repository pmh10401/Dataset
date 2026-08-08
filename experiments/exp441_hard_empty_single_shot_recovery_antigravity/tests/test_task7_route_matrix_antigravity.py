"""Production-shaped Task 7 route and malformed-readiness matrix."""
from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack, redirect_stdout
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import stat
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
        code='''import importlib.abc,sqlite3,sys,types\nsqlite3.connect=lambda *a,**k:(_ for _ in()).throw(AssertionError("sqlite"))\nblocked={"experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity","experiments.exp440_train_only_low_frequency_precision_gate_antigravity.evaluation_antigravity","experiments.exp417_causal_multihorizon_shadow.inputs"}\nclass F(importlib.abc.MetaPathFinder):\n def find_spec(self,n,p=None,t=None):\n  if n in blocked: raise AssertionError(n)\nsys.meta_path.insert(0,F())\nfrom experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as c,readiness_antigravity as r,runtime_antigravity as t,reporting_antigravity as p\nr.current_clean_head_antigravity=lambda:"a"*40;c.current_clean_head_antigravity=r.current_clean_head_antigravity;r.run_readiness_antigravity=lambda x:{};r.validate_readiness_for_run_antigravity=lambda x:types.SimpleNamespace(reviewed_commit=x)\nt.run_exp441_new_antigravity=lambda **k:{}\np.publish_exp441_antigravity=lambda x:{};p.inspect_exp441_publication_antigravity=lambda:{}\nfor x in("readiness","new","inspect"):assert c.main_antigravity([x])==0\n'''
        subprocess.run([sys.executable,"-c",code],cwd=root,check=True,capture_output=True,text=True)
        result=subprocess.run([sys.executable,str(wrapper),"--help"],cwd="/tmp",check=True,capture_output=True,text=True)
        self.assertIn("{readiness,new,inspect,evaluate}",result.stdout)

    def test_staging_write_and_directory_fsync_failures_keep_primary_error_visible(self):
        """Initial readiness artifact failures must never create the fixed parent."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as r
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_readiness_review_fix_antigravity import ReadinessReviewFixAntigravityTests
        helper=ReadinessReviewFixAntigravityTests(); commit="a"*40
        for failure in ("write","fsync"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/"run_antigravity"
                with ExitStack() as stack:
                    helper._patch_successful_replay_antigravity(stack,r,root,commit)
                    if failure=="write": stack.enter_context(patch.object(r,"write_create_only_antigravity",side_effect=OSError("staging write")))
                    else:
                        stack.enter_context(patch.object(r,"fsync_directory_antigravity",side_effect=OSError("staging fsync")))
                    with self.assertRaisesRegex(OSError,"staging") as caught:r.run_readiness_antigravity(commit)
                self.assertFalse(root.exists());self.assertIsNotNone(caught.exception.__cause__)

    def test_rollback_rename_and_failed_parent_fsync_chain_from_original_parent_fsync(self):
        """Rollback races and failed-staging durability errors chain from, never mask, parent fsync."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as r
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_readiness_review_fix_antigravity import ReadinessReviewFixAntigravityTests
        helper=ReadinessReviewFixAntigravityTests();commit="a"*40
        for failure in ("rollback_rename","failed_parent_fsync"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/"run_antigravity";original_rename=r.exclusive_rename_antigravity;seen={"rename":0}
                def rename(source,destination):
                    seen["rename"]+=1
                    if failure=="rollback_rename" and seen["rename"]==2:raise OSError("rollback rename")
                    return original_rename(source,destination)
                def fsync(path):
                    if path==root.parent and root.exists():raise OSError("parent fsync primary")
                    if failure=="failed_parent_fsync" and path==root.parent:raise OSError("failed parent fsync")
                    return r.fsync_directory_antigravity(path)
                original_fsync=r.fsync_directory_antigravity
                def fsync_safe(path):
                    if path==root.parent and root.exists():raise OSError("parent fsync primary")
                    if failure=="failed_parent_fsync" and path==root.parent:raise OSError("failed parent fsync")
                    return original_fsync(path)
                with ExitStack() as stack:
                    helper._patch_successful_replay_antigravity(stack,r,root,commit)
                    stack.enter_context(patch.object(r,"exclusive_rename_antigravity",side_effect=rename));stack.enter_context(patch.object(r,"fsync_directory_antigravity",side_effect=fsync_safe))
                    with self.assertRaisesRegex(OSError,"parent fsync primary") as caught:r.run_readiness_antigravity(commit)
                self.assertIsNotNone(caught.exception.__cause__);self.assertIn("preservation also failed","\n".join(caught.exception.__notes__))

    def test_cli_new_rejects_moving_head_after_readiness_before_runtime(self):
        """A second HEAD read must stop `new` before the source-replay runtime boundary."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.readiness_antigravity as readiness
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        binding=types.SimpleNamespace(reviewed_commit="a"*40)
        with patch.object(cli,"current_clean_head_antigravity",side_effect=["a"*40,"b"*40]),patch.object(readiness,"validate_readiness_for_run_antigravity",return_value=binding),patch.object(runtime,"run_exp441_new_antigravity") as run:
            with self.assertRaises(ValueError):cli.main_antigravity(["new"])
        run.assert_not_called()

    def test_real_readiness_payload_rejects_every_environment_backend_and_source_leaf_mutation(self):
        """Every discovered nested readiness leaf has missing, wrong-type, and wrong-value rejection."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        backend = readiness._backend_antigravity()
        payload, source, selection = self._payload(readiness, commit)
        payload["backend"] = backend.to_payload()

        def leaf_paths(value: object, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
            if isinstance(value, dict):
                discovered: set[tuple[str, ...]] = set()
                for key, item in value.items():
                    discovered.update(leaf_paths(item, prefix + (key,)))
                return discovered
            return {prefix}

        environment_paths = leaf_paths(payload["backend"]["environment"], ("backend", "environment"))
        backend_core_paths = {("backend", key) for key in payload["backend"] if key != "environment"}
        source_paths = leaf_paths(payload["source"], ("source",))
        self.assertEqual(environment_paths, {
            ("backend", "environment", "python_version"),
            ("backend", "environment", "numpy_version"),
            ("backend", "environment", "pytorch_version"),
            ("backend", "environment", "mps_built"),
            ("backend", "environment", "mps_available"),
            ("backend", "environment", "mlx_version"),
            ("backend", "environment", "mlx_metal_version"),
            ("backend", "environment", "mlx_default_device"),
            ("backend", "environment", "acceleration_reason"),
        })
        self.assertEqual(backend_core_paths, {
            ("backend", "canonical_backend"), ("backend", "workers"),
            ("backend", "tensor_eligible"), ("backend", "acceleration_benchmark_status"),
            ("backend", "pytorch"), ("backend", "mlx"), ("backend", "mps"),
        })
        self.assertEqual(source_paths, {
            ("source", "context_dataset_count"), ("source", "record_digest_count"),
            ("source", "hard_coordinate_count"), ("source", "provisional_candidate_count"),
            ("source", "label_access_count"), ("source", "source_hashes", "context"),
            ("source", "source_hashes", "state"), ("source", "source_hashes", "retained_inventory"),
            ("source", "source_hashes", "publication_prediction"),
            ("source", "source_hashes", "publication_manifest"),
            ("source", "source_hashes", "publication_artifact_map"),
            ("source", "source_hashes", "hard_prediction"),
            ("source", "source_hashes", "hard_manifest"),
            ("source", "source_hashes", "hard_coordinate_identity"),
        })

        def parent_and_key(value: dict[str, object], path: tuple[str, ...]) -> tuple[dict[str, object], str]:
            parent: dict[str, object] = value
            for part in path[:-1]:
                parent = parent[part]  # type: ignore[assignment]
            return parent, path[-1]

        def wrong_value(value: object) -> object:
            if type(value) is bool:
                return not value
            if type(value) is int:
                return value + 1
            if type(value) is float:
                return value + 1.0
            if isinstance(value, str):
                return value + "x"
            if value is None:
                return "unexpected"
            raise AssertionError(f"unhandled readiness leaf: {value!r}")

        all_paths = environment_paths | backend_core_paths | source_paths
        self.assertEqual(len(all_paths), 30)
        for path in sorted(all_paths):
            for mutation in ("missing", "type", "value"):
                with self.subTest(path=".".join(path), mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                    changed = deepcopy(payload)
                    parent, key = parent_and_key(changed, path)
                    original = parent[key]
                    if mutation == "missing":
                        del parent[key]
                    elif mutation == "type":
                        parent[key] = {"wrong_type": original}
                    else:
                        parent[key] = wrong_value(original)
                    if path[0] == "source":
                        changed["source_snapshot_sha256"] = sha256(
                            readiness.canonical_json_bytes_antigravity(changed["source"])
                        ).hexdigest()
                    root = Path(temporary) / "run_antigravity"
                    self._write_root(root, readiness, changed)
                    with ExitStack() as stack:
                        stack.enter_context(patch.object(readiness, "_RUN_ROOT_ANTIGRAVITY", root))
                        stack.enter_context(patch.object(readiness, "_current_clean_head_antigravity", return_value=commit))
                        stack.enter_context(patch.object(readiness, "verify_frozen_sources_antigravity", return_value=object()))
                        stack.enter_context(patch.object(readiness, "_source_payload_antigravity", return_value=source))
                        stack.enter_context(patch.object(readiness, "_selection_payload_antigravity", return_value=selection))
                        stack.enter_context(patch.object(readiness, "_backend_antigravity", return_value=backend))
                        with self.assertRaises(ValueError):
                            readiness.validate_readiness_for_run_antigravity(commit)

    def test_post_rollback_evidence_write_and_failed_directory_fsync_preserve_primary_error(self):
        """Post-rollback evidence faults stay secondary while the owned inode remains recoverable."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_readiness_review_fix_antigravity import ReadinessReviewFixAntigravityTests

        helper = ReadinessReviewFixAntigravityTests()
        commit = "a" * 40
        for failure in ("failure_evidence_write", "failed_directory_fsync"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "run_antigravity"
                primary = OSError("primary readiness parent fsync")
                secondary_text = "canonical failure evidence write" if failure == "failure_evidence_write" else "failed directory fsync"
                original_write = readiness.write_create_only_antigravity
                original_fsync = readiness.fsync_directory_antigravity
                original_rename = readiness.exclusive_rename_antigravity
                moved_identities: list[tuple[int, int]] = []

                def write(path: Path, payload_bytes: bytes) -> None:
                    if failure == "failure_evidence_write" and path.name == "failure_antigravity.json":
                        raise OSError(secondary_text)
                    original_write(path, payload_bytes)

                def fsync(path: Path) -> None:
                    path = Path(path)
                    if path == root.parent and root.exists():
                        raise primary
                    if failure == "failed_directory_fsync" and path.parent == root.parent and "_failed_" in path.name:
                        raise OSError(secondary_text)
                    original_fsync(path)

                def rename(source_path: Path, destination_path: Path) -> None:
                    before = source_path.lstat()
                    original_rename(source_path, destination_path)
                    after = destination_path.lstat()
                    self.assertEqual((before.st_dev, before.st_ino), (after.st_dev, after.st_ino))
                    moved_identities.append((after.st_dev, after.st_ino))

                with ExitStack() as stack:
                    helper._patch_successful_replay_antigravity(stack, readiness, root, commit)
                    stack.enter_context(patch.object(readiness, "write_create_only_antigravity", side_effect=write))
                    stack.enter_context(patch.object(readiness, "fsync_directory_antigravity", side_effect=fsync))
                    stack.enter_context(patch.object(readiness, "exclusive_rename_antigravity", side_effect=rename))
                    with self.assertRaisesRegex(OSError, "primary readiness parent fsync") as caught:
                        readiness.run_readiness_antigravity(commit)
                self.assertIs(caught.exception, primary)
                self.assertIsInstance(caught.exception.__cause__, OSError)
                self.assertIn(secondary_text, str(caught.exception.__cause__))
                self.assertIn("preservation also failed", "\n".join(caught.exception.__notes__))
                self.assertFalse(root.exists())
                self.assertEqual(list(root.parent.glob(".run_antigravity_staging_*_antigravity")), [])
                failed = list(root.parent.glob(".run_antigravity_failed_*_antigravity"))
                self.assertEqual(len(failed), 1)
                self.assertEqual(len(moved_identities), 2)
                self.assertEqual(moved_identities[0], moved_identities[1])
                self.assertEqual(stat.S_IMODE(failed[0].lstat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE((failed[0] / "readiness_antigravity.json").lstat().st_mode), 0o600)
                evidence = failed[0] / "failure_antigravity.json"
                if failure == "failure_evidence_write":
                    self.assertFalse(evidence.exists())
                else:
                    encoded = evidence.read_bytes()
                    self.assertEqual(
                        encoded,
                        json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":")).encode() + b"\n",
                    )

    def test_fresh_interpreters_use_real_modules_for_all_four_non_evaluate_routes(self):
        """Help/readiness/new/inspect import real production modules behind SQLite and evaluator bombs."""
        root = Path(__file__).resolve().parents[3]
        code = r'''import contextlib,importlib.abc,io,json,pathlib,sqlite3,sys
from unittest.mock import patch
def sqlite_bomb(*args,**kwargs): raise AssertionError("sqlite.connect")
sqlite3.connect=sqlite_bomb
blocked=("experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity","experiments.exp440_train_only_low_frequency_precision_gate_antigravity.evaluation_antigravity","experiments.exp417_causal_multihorizon_shadow.inputs")
class FirewallAntigravity(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==name or fullname.startswith(name+".") for name in blocked): raise AssertionError(fullname)
sys.meta_path.insert(0,FirewallAntigravity())
from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli
from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness
from experiments.exp441_hard_empty_single_shot_recovery_antigravity import runtime_antigravity as runtime
from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity as reporting
expected_root=(pathlib.Path.cwd()/"experiments"/"exp441_hard_empty_single_shot_recovery_antigravity").resolve()
for module in (cli,readiness,runtime,reporting):
 assert pathlib.Path(module.__file__).resolve().is_relative_to(expected_root)
route=sys.argv[1]
if route=="--help":
 try: cli.main_antigravity([route])
 except SystemExit as error: assert error.code==0
 else: raise AssertionError("help did not exit")
else:
 class BindingAntigravity: reviewed_commit="a"*40
 with contextlib.ExitStack() as stack:
  stack.enter_context(patch.object(cli,"current_clean_head_antigravity",return_value="a"*40))
  stack.enter_context(patch.object(readiness,"run_readiness_antigravity",return_value={}))
  stack.enter_context(patch.object(readiness,"validate_readiness_for_run_antigravity",return_value=BindingAntigravity()))
  stack.enter_context(patch.object(runtime,"run_exp441_new_antigravity",return_value={}))
  stack.enter_context(patch.object(reporting,"publish_exp441_antigravity",return_value={}))
  stack.enter_context(patch.object(reporting,"inspect_exp441_publication_antigravity",return_value={}))
  result=cli.main_antigravity([route])
  assert type(result) is int and result==0
assert all(name not in sys.modules for name in blocked)
'''
        for route in ("--help", "readiness", "new", "inspect"):
            with self.subTest(route=route):
                result = subprocess.run(
                    [sys.executable, "-c", code, route], cwd=root, check=True,
                    capture_output=True, text=True,
                )
                self.assertEqual(result.stderr, "")
                if route == "--help":
                    self.assertIn("{readiness,new,inspect,evaluate}", result.stdout)
                else:
                    self.assertEqual(result.stdout, f'{{"command":"{route}","result":{{}}}}\n')

    def test_real_cli_inspect_is_canonical_and_byte_tree_read_only(self):
        """Actual CLI inspect preserves the exact isolated six-file publication tree and bytes."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import reporting_antigravity as reporting
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_reporting_antigravity import ReportingContractAntigravityTests

        helper = ReportingContractAntigravityTests()
        snapshot = helper._fixture_snapshot_antigravity()

        def tree_snapshot(root: Path) -> dict[str, tuple[str, int, bytes | None, tuple[int, int]]]:
            result: dict[str, tuple[str, int, bytes | None, tuple[int, int]]] = {}
            for path in (root, *sorted(root.rglob("*"))):
                status = path.lstat()
                relative = "." if path == root else path.relative_to(root).as_posix()
                if stat.S_ISDIR(status.st_mode):
                    kind, payload_bytes = "directory", None
                elif stat.S_ISREG(status.st_mode):
                    kind, payload_bytes = "file", path.read_bytes()
                else:
                    kind, payload_bytes = "other", None
                result[relative] = (kind, stat.S_IMODE(status.st_mode), payload_bytes, (status.st_dev, status.st_ino))
            return result

        expected_paths = {
            ".", "attempts_antigravity", "attempts_antigravity/attempt_0000_antigravity",
            "attempts_antigravity/attempt_0000_antigravity/events_antigravity.jsonl",
            "attempts_antigravity/attempt_0000_antigravity/pid_antigravity",
            "attempts_antigravity/attempt_0000_antigravity/progress_antigravity.json",
            "attempts_antigravity/attempt_0000_antigravity/selection_antigravity.json",
            "attempts_antigravity/attempt_0000_antigravity/source_snapshot_antigravity.json",
            "attempts_antigravity/attempt_0000_antigravity/text_antigravity.log",
            "publication_antigravity", "publication_antigravity/prediction_antigravity.jsonl",
            "publication_antigravity/candidate_ranking_antigravity.jsonl",
            "publication_antigravity/source_audit_antigravity.json",
            "publication_antigravity/manifest_antigravity.json",
            "publication_antigravity/status_antigravity.json",
            "publication_antigravity/artifact_sha256_antigravity.json",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = helper._write_terminal_attempt_antigravity(root, snapshot)
            with patch.object(reporting, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(reporting, "verify_frozen_sources_antigravity", return_value=snapshot), \
                 patch.object(reporting, "selected_coordinate_identity_sha256_antigravity", return_value=reporting.FROZEN_CONFIG.selected_coordinate_identity_sha256), \
                 patch.object(reporting, "exclusive_rename_antigravity", side_effect=os.rename):
                reporting.publish_exp441_antigravity(attempt)
                before = tree_snapshot(root)
                self.assertEqual(set(before), expected_paths)
                self.assertEqual(
                    {path for path, evidence in before.items() if path.startswith("publication_antigravity/") and evidence[0] == "file"},
                    {f"publication_antigravity/{name}" for name in reporting._PUBLICATION_NAMES_ANTIGRAVITY},
                )
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    result = cli.main_antigravity(["inspect"])
                after = tree_snapshot(root)
            self.assertIs(type(result), int)
            self.assertEqual(result, 0)
            self.assertEqual(after, before)
            parsed = json.loads(stdout.getvalue())
            self.assertEqual(stdout.getvalue().encode(), json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n")
            self.assertEqual(parsed["command"], "inspect")
            self.assertEqual(set(parsed["result"]), {"publication_sha256", "prediction_sha256", "ranking_sha256", "source_audit_sha256", "selection"})
            publication = root / "publication_antigravity"
            self.assertEqual(parsed["result"]["publication_sha256"], sha256((publication / "artifact_sha256_antigravity.json").read_bytes()).hexdigest())
            self.assertEqual(parsed["result"]["prediction_sha256"], sha256((publication / "prediction_antigravity.jsonl").read_bytes()).hexdigest())
            self.assertEqual(parsed["result"]["ranking_sha256"], sha256((publication / "candidate_ranking_antigravity.jsonl").read_bytes()).hexdigest())
            self.assertEqual(parsed["result"]["source_audit_sha256"], sha256((publication / "source_audit_antigravity.json").read_bytes()).hexdigest())
            self.assertEqual(
                {name: len(parsed["result"]["selection"][name]) for name in ("provisional_candidates", "outside_hard_candidates", "eligible_candidates", "winners", "selected")},
                {"provisional_candidates": 49, "outside_hard_candidates": 49, "eligible_candidates": 49, "winners": 49, "selected": 30},
            )
