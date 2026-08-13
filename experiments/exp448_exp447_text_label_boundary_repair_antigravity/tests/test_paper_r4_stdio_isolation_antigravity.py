"""Behavioral RED/GREEN contracts for official-r4 launcher stdio isolation."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_paper_r4_operations_antigravity import (  # noqa: E402
    FORBIDDEN_LABEL_MODULE,
    LAUNCHER_PATH,
)


def _ops_helpers() -> dict[str, object]:
    from test_paper_r4_operations_antigravity import PaperR4OperationsTests

    return {
        "setUp": PaperR4OperationsTests.setUp,
        "tearDown": PaperR4OperationsTests.tearDown,
        "load_launcher": PaperR4OperationsTests.load_launcher,
        "_git_repo": PaperR4OperationsTests._git_repo,
        "_prepare_fixture": PaperR4OperationsTests._prepare_fixture,
        "_install_venv_interpreter": PaperR4OperationsTests._install_venv_interpreter,
        "_bind_official_runtime": PaperR4OperationsTests._bind_official_runtime,
        "_write_smoke_result": PaperR4OperationsTests._write_smoke_result,
        "_publish_launch_authority": PaperR4OperationsTests._publish_launch_authority,
    }


_OPS_HELPERS = _ops_helpers()


FAILED_OFFICIAL_OPS_ROOT = Path(
    "/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813_ops_antigravity"
)
VERIFIED_V2_SMOKE_ROOT = Path(
    "/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_v2_antigravity_20260813"
)
VERIFIED_V2_SMOKE_OPS_ROOT = Path(str(VERIFIED_V2_SMOKE_ROOT) + "_ops_antigravity")
CURRENT_OPS_ROOT = Path(
    "/Users/mac/Documents/Dataset/paper_proposal_full_r4_v2_antigravity_20260814_ops_antigravity"
)
CURRENT_SMOKE_ROOT = Path(
    "/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_v3_antigravity_20260814"
)
CURRENT_SMOKE_OPS_ROOT = Path(str(CURRENT_SMOKE_ROOT) + "_ops_antigravity")
OFFICIAL_R4_ROOT = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813")

EXPECTED_FAILED_OFFICIAL_OPS_TREE_SHA256 = "61bee0c09a314c85058269b61fcb444077007ad74bfcc975626bcd40f261a8d6"
EXPECTED_FAILED_OFFICIAL_OPS_AUTHORITY_SHA256 = "dc2b97b2a1118f120551a63ed0e4071ca5f8e00bc808975adf9f7611e3f75860"
EXPECTED_FAILED_OFFICIAL_OPS_STATE_SHA256 = "db3eb188ac7fadcc0a5fb754f095f02bb646bafcffb252385eb82619968094b9"
EXPECTED_FAILED_OFFICIAL_OPS_PROGRESS_LOG_SHA256 = "cfb65a61ffa6599dff9ef110226b1ae6fc11d71e7cb6a5ce8dfcc47e5d88e019"
EXPECTED_FAILED_OFFICIAL_OPS_PROGRESS_EVENTS_SHA256 = "4f106be4f75ddb004f847fcb80ee456848de802e03c0000d056cd37a1eb24db7"
EXPECTED_VERIFIED_V2_SMOKE_TREE_SHA256 = "a31b588cd6be3c90575032c830822c79d333e34e2dadd20c19d0b2c7fdd2a3f8"
EXPECTED_VERIFIED_V2_SMOKE_OPS_TREE_SHA256 = "66d9fa0b79cb76aa3d76c40decd3fac57c1220bc22f59ed07b062b3c5d956d46"
EXPECTED_VERIFIED_V2_SMOKE_RESULT_SHA256 = "ffce73aaa0d93770f3f86452c6f22b4703fc1f1cc82f339ede1b6b86b3a30953"
EXPECTED_VERIFIED_V2_SMOKE_OPS_AUTHORITY_SHA256 = "d276164b00672b8b92e749673ebef1b89edda4cba13315819789bbd7d87d78a8"


class PaperR4StdioIsolationTests(unittest.TestCase):
    setUp = _OPS_HELPERS["setUp"]
    tearDown = _OPS_HELPERS["tearDown"]
    load_launcher = _OPS_HELPERS["load_launcher"]
    _git_repo = _OPS_HELPERS["_git_repo"]
    _prepare_fixture = _OPS_HELPERS["_prepare_fixture"]
    _install_venv_interpreter = _OPS_HELPERS["_install_venv_interpreter"]
    _bind_official_runtime = _OPS_HELPERS["_bind_official_runtime"]
    _write_smoke_result = _OPS_HELPERS["_write_smoke_result"]
    _publish_launch_authority = _OPS_HELPERS["_publish_launch_authority"]

    def test_spawn_worker_passes_stdin_devnull_and_keeps_stdio_routing_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        authority = module.check(commit)
        digest = self._write_smoke_result(module, authority)
        launch_authority = module.build_authority(commit, smoke_result_sha256=digest)
        paths = module.ops_paths()
        paths.root.mkdir(parents=True)
        encoded = module.canonical_json_bytes(launch_authority)
        paths.authority.write_bytes(encoded)
        paths.authority_sha256.write_text(sha256(encoded).hexdigest() + "\n", encoding="ascii")
        captured: dict[str, object] = {}

        def fake_spawn(*_args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(pid=424242)

        lock_stream = paths.worker_lock.open("a+b")
        self.addCleanup(lock_stream.close)
        module._spawn_worker(commit, sha256(encoded).hexdigest(), "full", lock_stream, spawn=fake_spawn)
        self.assertIs(captured["stdin"], subprocess.DEVNULL)
        self.assertEqual(captured["stderr"], subprocess.STDOUT)
        self.assertIsNot(captured["stdout"], subprocess.DEVNULL)
        self.assertIsNot(captured["stdout"], subprocess.PIPE)
        self.assertIsNotNone(captured["stdout"])

    def _pipeline_popen_capture(self, module, captured: dict[str, object], child):
        real_popen = module.subprocess.Popen

        def fake_popen(command, **kwargs):
            if isinstance(command, (list, tuple)) and command and command[0] == "git":
                return real_popen(command, **kwargs)
            captured.update(kwargs)
            captured["command"] = command
            return child(command, **kwargs) if callable(child) else child

        return fake_popen

    def test_run_pipeline_child_passes_stdin_devnull_and_keeps_pipe_stdout_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, _commit, _dataset = self._prepare_fixture(module)
        paths = module.ops_paths()
        paths.root.mkdir(parents=True)
        captured: dict[str, object] = {}

        class FakeChild:
            pid = 777001

            def __init__(self, *_args, **_kwargs) -> None:
                self.stdout = self

            def __iter__(self):
                return iter(("child-line\n",))

            def close(self) -> None:
                return None

            def wait(self) -> int:
                return 0

        with mock.patch.object(module.subprocess, "Popen", self._pipeline_popen_capture(module, captured, FakeChild)):
            code = module._run_pipeline_child("full", module.OFFICIAL_R4_ROOT, paths)
        self.assertEqual(code, 0)
        self.assertIs(captured["stdin"], subprocess.DEVNULL)
        self.assertIs(captured["stdout"], subprocess.PIPE)
        self.assertEqual(captured["stderr"], subprocess.STDOUT)

    def test_worker_pipeline_child_popen_uses_stdin_devnull_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        _authority, digest, paths, _smoke = self._publish_launch_authority(module, commit)
        captured: dict[str, object] = {}

        class FakeChild:
            pid = 888002

            def __init__(self, *_args, **_kwargs) -> None:
                self.stdout = self

            def __iter__(self):
                return iter(("worker-child-line\n",))

            def close(self) -> None:
                return None

            def wait(self) -> int:
                return 0

            def poll(self):
                return 0

        lock_stream = paths.worker_lock.open("a+b")
        self.addCleanup(lock_stream.close)
        worker_fd = os.dup(lock_stream.fileno())
        with mock.patch.object(
            module,
            "pipeline_command",
            return_value=[sys.executable, "-c", "print('unused')"],
        ), mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}), mock.patch.object(
            module.subprocess, "Popen", self._pipeline_popen_capture(module, captured, FakeChild)
        ):
            code = module._worker(commit, digest, worker_fd, "full")
        self.assertEqual(code, 0)
        self.assertIs(captured["stdin"], subprocess.DEVNULL)
        self.assertIs(captured["stdout"], subprocess.PIPE)
        self.assertEqual(captured["stderr"], subprocess.STDOUT)

    def test_worker_emits_child_start_only_after_popen_with_positive_pid_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        _authority, digest, paths, _smoke = self._publish_launch_authority(module, commit)
        popen_calls: list[str] = []

        class FakeChild:
            pid = 999003

            def __init__(self, *_args, **_kwargs) -> None:
                popen_calls.append("started")
                self.stdout = self

            def __iter__(self):
                return iter(("ok\n",))

            def close(self) -> None:
                return None

            def wait(self) -> int:
                return 0

            def poll(self):
                return 0

        lock_stream = paths.worker_lock.open("a+b")
        self.addCleanup(lock_stream.close)
        worker_fd = os.dup(lock_stream.fileno())
        captured: dict[str, object] = {}
        with mock.patch.object(
            module,
            "pipeline_command",
            return_value=[sys.executable, "-c", "print('ok')"],
        ), mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}), mock.patch.object(
            module.subprocess, "Popen", self._pipeline_popen_capture(module, captured, FakeChild)
        ):
            code = module._worker(commit, digest, worker_fd, "full")
        self.assertEqual(code, 0)
        events = [json.loads(line) for line in paths.progress_events.read_text(encoding="utf-8").splitlines()]
        starts = [event for event in events if event.get("line") == "child_start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(popen_calls, ["started"])
        self.assertEqual(starts[0]["child_pid"], 999003)
        self.assertIs(type(starts[0]["child_pid"]), int)
        self.assertGreater(starts[0]["child_pid"], 0)

    def test_worker_omits_child_start_when_popen_fails_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        _authority, digest, paths, _smoke = self._publish_launch_authority(module, commit)

        real_popen = module.subprocess.Popen

        def boom(command, **kwargs):
            if isinstance(command, (list, tuple)) and command and command[0] == "git":
                return real_popen(command, **kwargs)
            raise OSError("popen failed")

        lock_stream = paths.worker_lock.open("a+b")
        self.addCleanup(lock_stream.close)
        worker_fd = os.dup(lock_stream.fileno())
        with mock.patch.object(
            module,
            "pipeline_command",
            return_value=[sys.executable, "-c", "print('unused')"],
        ), mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}), mock.patch.object(
            module.subprocess, "Popen", boom
        ):
            code = module._worker(commit, digest, worker_fd, "full")
        self.assertEqual(code, 1)
        if paths.progress_events.is_file():
            events = [json.loads(line) for line in paths.progress_events.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event for event in events if event.get("line") == "child_start"], [])

    def test_acquire_lock_fd_is_at_least_3_when_stdio_fds_unavailable_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, dataset = self._prepare_fixture(module)
        module.check(commit)
        result_path = dataset / "lock_fd_result_antigravity.json"
        driver = dataset / "lock_fd_driver_antigravity.py"
        driver.write_text(
            "\n".join(
                (
                    "import json, os, importlib.util, sys",
                    "from pathlib import Path",
                    f"result = Path({str(result_path)!r})",
                    f"launcher = Path({str(LAUNCHER_PATH)!r})",
                    f"ops_root = Path({str(module.OFFICIAL_OPS_ROOT)!r})",
                    "for fd in (0, 1, 2):",
                    "    try:",
                    "        os.close(fd)",
                    "    except OSError:",
                    "        pass",
                    "spec = importlib.util.spec_from_file_location('paper_r4_lock_fd_antigravity', launcher)",
                    "loaded = importlib.util.module_from_spec(spec)",
                    "sys.modules[spec.name] = loaded",
                    "spec.loader.exec_module(loaded)",
                    "loaded.OFFICIAL_OPS_ROOT = ops_root",
                    "paths = loaded.ops_paths()",
                    "lock = loaded._acquire_lock(paths)",
                    "payload = {'fileno': lock.fileno(), 'lock_path': str(paths.worker_lock)}",
                    "descriptor = os.open(result, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)",
                    "os.write(descriptor, json.dumps(payload).encode('utf-8'))",
                    "os.close(descriptor)",
                    "ready = Path(str(result) + '.ready_antigravity')",
                    "os.close(os.open(ready, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))",
                    "stop = Path(str(result) + '.stop_antigravity')",
                    "while not stop.exists():",
                    "    pass",
                    "lock.close()",
                    "",
                )
            ),
            encoding="utf-8",
        )
        child = subprocess.Popen([sys.executable, "-B", "-u", str(driver)], cwd=str(module.REPOSITORY_ROOT))
        self.addCleanup(lambda: child.poll() is not None or child.kill() or child.wait())
        ready = Path(str(result_path) + ".ready_antigravity")
        for _ in range(2000):
            if ready.exists() or child.poll() is not None:
                break
            child.wait(timeout=0.01) if False else None
            import time

            time.sleep(0.01)
        self.assertTrue(ready.exists(), msg=f"lock driver exited {child.poll()} before ready")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(payload["fileno"], 3)
        lock_path = Path(payload["lock_path"])
        with lock_path.open("a+b") as contender:
            with self.assertRaises(BlockingIOError):
                import fcntl

                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        Path(str(result_path) + ".stop_antigravity").write_text("stop\n", encoding="utf-8")
        self.assertEqual(child.wait(timeout=5), 0)

    def test_two_hop_worker_child_survives_closed_parent_stdin_antigravity(self) -> None:
        module = self.load_launcher()
        repo, commit, dataset = self._prepare_fixture(module)
        authority = module.check(commit)
        smoke_digest = self._write_smoke_result(module, authority)
        launch_authority = module.build_authority(commit, smoke_result_sha256=smoke_digest)
        encoded = module.canonical_json_bytes(launch_authority)
        digest = sha256(encoded).hexdigest()
        paths = module.ops_paths()
        paths.root.mkdir(parents=True)
        paths.authority.write_bytes(encoded)
        paths.authority_sha256.write_text(digest + "\n", encoding="ascii")
        result_path = dataset / "two_hop_result_antigravity.json"
        worker_driver = dataset / "two_hop_worker_driver_antigravity.py"
        parent_driver = dataset / "two_hop_parent_driver_antigravity.py"
        remap = {
            "launcher": str(LAUNCHER_PATH),
            "repo": str(repo),
            "commit": commit,
            "digest": digest,
            "smoke": smoke_digest,
            "db_sha": module.EXPECTED_DATABASE_SHA256,
            "paths": {
                "OFFICIAL_INTERPRETER": str(module.OFFICIAL_INTERPRETER),
                "OFFICIAL_DATABASE": str(module.OFFICIAL_DATABASE),
                "OFFICIAL_R3_ROOT": str(module.OFFICIAL_R3_ROOT),
                "OFFICIAL_R4_ROOT": str(module.OFFICIAL_R4_ROOT),
                "OFFICIAL_OPS_ROOT": str(module.OFFICIAL_OPS_ROOT),
                "OFFICIAL_SMOKE_ROOT": str(module.OFFICIAL_SMOKE_ROOT),
            },
        }
        worker_driver.write_text(
            "\n".join(
                (
                    "from hashlib import sha256",
                    "import importlib.util, json, os, sys",
                    "from pathlib import Path",
                    "try:",
                    "    os.close(0)",
                    "except OSError:",
                    "    pass",
                    f"remap = json.loads({json.dumps(json.dumps(remap))})",
                    "spec = importlib.util.spec_from_file_location('paper_r4_two_hop_worker_antigravity', remap['launcher'])",
                    "loaded = importlib.util.module_from_spec(spec)",
                    "sys.modules[spec.name] = loaded",
                    "spec.loader.exec_module(loaded)",
                    "loaded.REPOSITORY_ROOT = Path(remap['repo'])",
                    "loaded.EXPECTED_DATABASE_SHA256 = remap['db_sha']",
                    "for name, value in remap['paths'].items():",
                    "    setattr(loaded, name, Path(value))",
                    "loaded._require_official_runtime = lambda **_k: {",
                    "    'executable': str(loaded.OFFICIAL_INTERPRETER),",
                    "    'executable_realpath': str(Path(loaded.OFFICIAL_INTERPRETER).resolve()),",
                    "    'python_prefix': str(Path(loaded.OFFICIAL_INTERPRETER).parent.parent),",
                    "    'python_base_prefix': sys.base_prefix,",
                    "    'python_identity_sha256': 'a' * 64,",
                    "}",
                    "stored = loaded._load_authority(loaded.ops_paths())",
                    "loaded.build_authority = lambda *_a, **_k: stored",
                    "loaded.verify_smoke_authorization = lambda _commit: {'result_sha256': remap['smoke']}",
                    "loaded._pipeline_status = lambda *_a, **_k: {'status': 'completed'}",
                    "loaded.pipeline_command = lambda mode, run_root: [",
                    "    sys.executable, '-c',",
                    "    'import sys; print(\"stdio-isolation-child-ok-antigravity\"); raise SystemExit(0)',",
                    "]",
                    "lock_fd = int(sys.argv[1])",
                    "raise SystemExit(loaded._worker(remap['commit'], remap['digest'], lock_fd, 'full'))",
                    "",
                )
            ),
            encoding="utf-8",
        )
        parent_driver.write_text(
            "\n".join(
                (
                    "import importlib.util, json, os, subprocess, sys",
                    "from pathlib import Path",
                    "try:",
                    "    os.close(0)",
                    "except OSError:",
                    "    pass",
                    f"launcher = Path({str(LAUNCHER_PATH)!r})",
                    f"remap = json.loads({json.dumps(json.dumps(remap))})",
                    f"worker_driver = Path({str(worker_driver)!r})",
                    f"result_path = Path({str(result_path)!r})",
                    f"digest = {digest!r}",
                    f"commit = {commit!r}",
                    "spec = importlib.util.spec_from_file_location('paper_r4_two_hop_parent_antigravity', launcher)",
                    "loaded = importlib.util.module_from_spec(spec)",
                    "sys.modules[spec.name] = loaded",
                    "spec.loader.exec_module(loaded)",
                    "loaded.REPOSITORY_ROOT = Path(remap['repo'])",
                    "loaded.EXPECTED_DATABASE_SHA256 = remap['db_sha']",
                    "for name, value in remap['paths'].items():",
                    "    setattr(loaded, name, Path(value))",
                    "paths = loaded.ops_paths()",
                    "lock_stream = loaded._acquire_lock(paths)",
                    "lock_fd = lock_stream.fileno()",
                    "held = {}",
                    "def spawn(command, **kwargs):",
                    "    rewritten = [sys.executable, '-B', '-u', str(worker_driver), str(lock_fd)]",
                    "    process = subprocess.Popen(rewritten, **kwargs)",
                    "    held['process'] = process",
                    "    held['stdin_is_devnull'] = kwargs.get('stdin') is subprocess.DEVNULL",
                    "    held['pass_fds'] = list(kwargs.get('pass_fds') or ())",
                    "    held['stderr'] = kwargs.get('stderr')",
                    "    return process",
                    "spawned = loaded._spawn_worker(commit, digest, 'full', lock_stream, spawn=spawn)",
                    "process = held['process']",
                    "worker_code = process.wait(timeout=30)",
                    "lock_stream.close()",
                    "events = []",
                    "if paths.progress_events.is_file():",
                    "    events = [json.loads(line) for line in paths.progress_events.read_text(encoding='utf-8').splitlines()]",
                    "log = paths.progress_log.read_text(encoding='utf-8') if paths.progress_log.is_file() else ''",
                    "state = {}",
                    "if paths.state.is_file():",
                    "    state = json.loads(paths.state.read_text(encoding='utf-8'))",
                    "result_path.write_text(json.dumps({",
                    "    'lock_fd': lock_fd,",
                    "    'stdin_is_devnull': held['stdin_is_devnull'],",
                    "    'pass_fds': held['pass_fds'],",
                    "    'worker_code': worker_code,",
                    "    'spawned_pid': spawned.get('pid'),",
                    "    'state': state,",
                    "    'events': events,",
                    "    'log': log,",
                    "}), encoding='utf-8')",
                    "",
                )
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-u", str(parent_driver)],
            cwd=str(repo),
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr + completed.stdout)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(payload["lock_fd"], 3)
        self.assertTrue(payload["stdin_is_devnull"])
        self.assertEqual(payload["worker_code"], 0)
        self.assertIs(type(payload["spawned_pid"]), int)
        self.assertGreater(payload["spawned_pid"], 0)
        self.assertNotIn("init_sys_streams", payload["log"])
        self.assertIn("stdio-isolation-child-ok-antigravity", payload["log"])
        starts = [event for event in payload["events"] if event.get("line") == "child_start"]
        self.assertEqual(len(starts), 1)
        self.assertIs(type(starts[0]["child_pid"]), int)
        self.assertGreater(starts[0]["child_pid"], 0)
        self.assertEqual(payload["state"].get("child_returncode"), 0)
        self.assertNotIn(FORBIDDEN_LABEL_MODULE, sys.modules)
        self.assertFalse(OFFICIAL_R4_ROOT.exists())
        self.assertFalse(CURRENT_SMOKE_ROOT.exists())
        self.assertFalse(CURRENT_OPS_ROOT.exists())

    def test_current_paths_bind_immutable_evidence_and_require_v3_smoke_antigravity(self) -> None:
        module = self.load_launcher()
        self.assertEqual(module.OFFICIAL_R4_ROOT, OFFICIAL_R4_ROOT)
        self.assertEqual(module.OFFICIAL_OPS_ROOT, CURRENT_OPS_ROOT)
        self.assertEqual(module.OFFICIAL_SMOKE_ROOT, CURRENT_SMOKE_ROOT)
        self.assertEqual(module.FAILED_OFFICIAL_OPS_ROOT, FAILED_OFFICIAL_OPS_ROOT)
        self.assertEqual(module.VERIFIED_V2_SMOKE_ROOT, VERIFIED_V2_SMOKE_ROOT)
        self.assertEqual(module.VERIFIED_V2_SMOKE_OPS_ROOT, VERIFIED_V2_SMOKE_OPS_ROOT)
        self.assertEqual(module.FAILED_OFFICIAL_OPS_TREE_SHA256, EXPECTED_FAILED_OFFICIAL_OPS_TREE_SHA256)
        self.assertEqual(module.FAILED_OFFICIAL_OPS_AUTHORITY_SHA256, EXPECTED_FAILED_OFFICIAL_OPS_AUTHORITY_SHA256)
        self.assertEqual(module.FAILED_OFFICIAL_OPS_STATE_SHA256, EXPECTED_FAILED_OFFICIAL_OPS_STATE_SHA256)
        self.assertEqual(module.FAILED_OFFICIAL_OPS_PROGRESS_LOG_SHA256, EXPECTED_FAILED_OFFICIAL_OPS_PROGRESS_LOG_SHA256)
        self.assertEqual(
            module.FAILED_OFFICIAL_OPS_PROGRESS_EVENTS_SHA256,
            EXPECTED_FAILED_OFFICIAL_OPS_PROGRESS_EVENTS_SHA256,
        )
        self.assertEqual(module.VERIFIED_V2_SMOKE_TREE_SHA256, EXPECTED_VERIFIED_V2_SMOKE_TREE_SHA256)
        self.assertEqual(module.VERIFIED_V2_SMOKE_OPS_TREE_SHA256, EXPECTED_VERIFIED_V2_SMOKE_OPS_TREE_SHA256)
        self.assertEqual(module.VERIFIED_V2_SMOKE_RESULT_SHA256, EXPECTED_VERIFIED_V2_SMOKE_RESULT_SHA256)
        self.assertEqual(
            module.VERIFIED_V2_SMOKE_OPS_AUTHORITY_SHA256,
            EXPECTED_VERIFIED_V2_SMOKE_OPS_AUTHORITY_SHA256,
        )
        self.assertEqual(
            module._tree_digest(FAILED_OFFICIAL_OPS_ROOT, "failed official ops"),
            EXPECTED_FAILED_OFFICIAL_OPS_TREE_SHA256,
        )
        self.assertEqual(
            module.sha256_file(FAILED_OFFICIAL_OPS_ROOT / "authority_antigravity.json"),
            EXPECTED_FAILED_OFFICIAL_OPS_AUTHORITY_SHA256,
        )
        self.assertEqual(
            module.sha256_file(FAILED_OFFICIAL_OPS_ROOT / "state_antigravity.json"),
            EXPECTED_FAILED_OFFICIAL_OPS_STATE_SHA256,
        )
        self.assertEqual(
            module.sha256_file(FAILED_OFFICIAL_OPS_ROOT / "progress_antigravity.log"),
            EXPECTED_FAILED_OFFICIAL_OPS_PROGRESS_LOG_SHA256,
        )
        self.assertEqual(
            module.sha256_file(FAILED_OFFICIAL_OPS_ROOT / "progress_events_antigravity.jsonl"),
            EXPECTED_FAILED_OFFICIAL_OPS_PROGRESS_EVENTS_SHA256,
        )
        self.assertEqual(
            module._tree_digest(VERIFIED_V2_SMOKE_ROOT, "verified v2 smoke"),
            EXPECTED_VERIFIED_V2_SMOKE_TREE_SHA256,
        )
        self.assertEqual(
            module._tree_digest(VERIFIED_V2_SMOKE_OPS_ROOT, "verified v2 smoke-ops"),
            EXPECTED_VERIFIED_V2_SMOKE_OPS_TREE_SHA256,
        )
        smoke_result = json.loads(
            (VERIFIED_V2_SMOKE_OPS_ROOT / "smoke_result_antigravity.json").read_text(encoding="utf-8")
        )
        self.assertEqual(smoke_result["result_sha256"], EXPECTED_VERIFIED_V2_SMOKE_RESULT_SHA256)
        self.assertEqual(
            (VERIFIED_V2_SMOKE_OPS_ROOT / "authority_sha256_antigravity.txt").read_text(encoding="ascii").strip(),
            EXPECTED_VERIFIED_V2_SMOKE_OPS_AUTHORITY_SHA256,
        )
        self.assertTrue(FAILED_OFFICIAL_OPS_ROOT.is_dir())
        self.assertTrue(VERIFIED_V2_SMOKE_ROOT.is_dir())
        self.assertTrue(VERIFIED_V2_SMOKE_OPS_ROOT.is_dir())
        self.assertFalse(CURRENT_OPS_ROOT.exists())
        self.assertFalse(CURRENT_SMOKE_ROOT.exists())
        self.assertFalse(CURRENT_SMOKE_OPS_ROOT.exists())
        self.assertFalse(OFFICIAL_R4_ROOT.exists())
        self.assertFalse(FAILED_OFFICIAL_OPS_ROOT.is_symlink())
        self.assertFalse(VERIFIED_V2_SMOKE_ROOT.is_symlink())

    def test_launch_rejects_bound_v2_smoke_result_as_authorization_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        authority = module.check(commit)
        with self.assertRaisesRegex(ValueError, "v2|smoke"):
            module.build_authority(commit, smoke_result_sha256=EXPECTED_VERIFIED_V2_SMOKE_RESULT_SHA256)
        paths = module.smoke_ops_paths()
        paths.root.mkdir(parents=True, exist_ok=True)
        bare = {
            "schema": module.SMOKE_RESULT_SCHEMA,
            "authority_sha256": sha256(module.canonical_json_bytes(authority)).hexdigest(),
            "pipeline": {"status": "completed"},
            "smoke_root": str(VERIFIED_V2_SMOKE_ROOT),
            "label_module_loaded": False,
        }
        digest = sha256(module.canonical_json_bytes(bare)).hexdigest()
        paths.smoke_result.write_bytes(
            module.canonical_json_bytes({**bare, "result_sha256": EXPECTED_VERIFIED_V2_SMOKE_RESULT_SHA256})
        )
        with mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}):
            with self.assertRaisesRegex(ValueError, "v2|smoke|root"):
                module.verify_smoke_authorization(commit)
        self.assertNotEqual(digest, "")
        self.assertFalse(module.OFFICIAL_R4_ROOT.exists())

    def test_cpu_worker_policy_remains_logical_minus_one_antigravity(self) -> None:
        module = self.load_launcher()
        logical, workers = module._logical_workers()
        self.assertEqual(logical, os.cpu_count())
        self.assertEqual(workers, max(1, int(os.cpu_count() or 1) - 1))
        if logical == 12:
            self.assertEqual(workers, 11)
        self.assertNotIn(FORBIDDEN_LABEL_MODULE, sys.modules)
        self.assertFalse(any(name.startswith("paper.equipment_anomaly_detection.label") for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
