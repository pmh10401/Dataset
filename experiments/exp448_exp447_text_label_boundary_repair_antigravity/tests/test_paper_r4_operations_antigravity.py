"""Boundary tests for the fixed-authority paper r4 operations launcher."""
from __future__ import annotations

from hashlib import sha256
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER_PATH = REPOSITORY_ROOT / "scratch/manage_paper_r4_proposal_antigravity.py"
FORBIDDEN_LABEL_MODULE = "paper.equipment_anomaly_detection.label_access_paper"
OFFICIAL_PATHS = {
    "OFFICIAL_INTERPRETER": Path(
        "/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python"
    ),
    "OFFICIAL_DATABASE": Path("/Users/mac/Documents/Dataset/univariate_ts.db"),
    "OFFICIAL_R3_ROOT": Path(
        "/Users/mac/Documents/Dataset/paper_proposal_full_r3_antigravity_20260813"
    ),
    "OFFICIAL_R4_ROOT": Path(
        "/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813"
    ),
    "OFFICIAL_OPS_ROOT": Path(
        "/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813_ops_antigravity"
    ),
    "OFFICIAL_SMOKE_ROOT": Path(
        "/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_antigravity_20260813"
    ),
}
EXPECTED_SMOKE_DATASETS = (
    "ChlorineConcentration_normal_3",
    "DodgerLoopDay_normal_1",
    "DodgerLoopDay_normal_2",
    "DodgerLoopDay_normal_4",
    "DodgerLoopDay_normal_5",
    "DodgerLoopDay_normal_6",
    "DodgerLoopDay_normal_7",
    "DodgerLoopGame_normal_1",
    "DodgerLoopGame_normal_2",
    "DodgerLoopWeekend_normal_1",
    "DodgerLoopWeekend_normal_2",
    "MelbournePedestrian_normal_1",
    "MelbournePedestrian_normal_2",
    "MelbournePedestrian_normal_3",
    "MelbournePedestrian_normal_4",
    "MelbournePedestrian_normal_5",
    "MelbournePedestrian_normal_6",
    "MelbournePedestrian_normal_7",
    "MelbournePedestrian_normal_8",
    "MelbournePedestrian_normal_9",
    "Phoneme_normal_39",
    "PigAirwayPressure_normal_1",
)


class PaperR4OperationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self._loaded_module_names: list[str] = []

    def tearDown(self) -> None:
        for name in self._loaded_module_names:
            sys.modules.pop(name, None)
        self.temporary.cleanup()

    def load_launcher(self):
        self.assertTrue(LAUNCHER_PATH.is_file(), "r4 operations launcher is absent")
        name = f"_paper_r4_operations_antigravity_{id(self)}_{len(self._loaded_module_names)}"
        specification = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        self._loaded_module_names.append(name)
        specification.loader.exec_module(module)
        return module

    def _git_repo(self) -> tuple[Path, str]:
        repo = self.root / f"repo_{len(list(self.root.iterdir()))}_antigravity"
        repo.mkdir()
        subprocess.check_call(["git", "init"], cwd=repo, stdout=subprocess.DEVNULL)
        subprocess.check_call(["git", "config", "user.email", "ops@example.com"], cwd=repo)
        subprocess.check_call(["git", "config", "user.name", "ops"], cwd=repo)
        subprocess.check_call(["git", "config", "commit.gpgsign", "false"], cwd=repo)
        (repo / "README").write_text("fixture\n", encoding="utf-8")
        subprocess.check_call(["git", "add", "README"], cwd=repo)
        subprocess.check_call(["git", "commit", "-m", "init"], cwd=repo, stdout=subprocess.DEVNULL)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        return repo, commit

    def _prepare_fixture(self, module, *, create_r4: bool = False, symlink_r3: bool = False):
        repo, commit = self._git_repo()
        dataset = self.root / f"dataset_{len(list(self.root.iterdir()))}_antigravity"
        dataset.mkdir()
        database = dataset / "univariate_ts.db"
        database.write_bytes(b"fixture-db")
        r3 = dataset / "paper_proposal_full_r3_antigravity_20260813"
        if symlink_r3:
            target = dataset / "r3_target_antigravity"
            target.mkdir()
            r3.symlink_to(target, target_is_directory=True)
        else:
            identity = r3 / "pipeline_journal_paper"
            identity.mkdir(parents=True)
            (identity / "000000_context_paper.json").write_text('{"schema":"fixture"}\n', encoding="utf-8")
        r4 = dataset / "paper_proposal_full_r4_antigravity_20260813"
        if create_r4:
            r4.mkdir()
        smoke = dataset / "paper_proposal_r4_prelaunch_smoke_antigravity_20260813"
        ops = dataset / "paper_proposal_full_r4_antigravity_20260813_ops_antigravity"
        for relative in module.SCORING_CLOSURE_PATHS:
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(f"fixture {relative}\n", encoding="utf-8")
        launcher_copy = repo / "scratch/manage_paper_r4_proposal_antigravity.py"
        launcher_copy.parent.mkdir(parents=True, exist_ok=True)
        if not launcher_copy.exists():
            launcher_copy.write_bytes(LAUNCHER_PATH.read_bytes())
        subprocess.check_call(["git", "add", "-A"], cwd=repo)
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True
        )
        if status.strip():
            subprocess.check_call(["git", "commit", "-m", "closure"], cwd=repo, stdout=subprocess.DEVNULL)
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        module.REPOSITORY_ROOT = repo
        module.OFFICIAL_INTERPRETER = self._install_venv_interpreter(dataset)
        module.OFFICIAL_DATABASE = database
        module.OFFICIAL_R3_ROOT = r3
        module.OFFICIAL_R4_ROOT = r4
        module.OFFICIAL_OPS_ROOT = ops
        module.OFFICIAL_SMOKE_ROOT = smoke
        module.EXPECTED_DATABASE_SHA256 = sha256(database.read_bytes()).hexdigest()
        self._bind_official_runtime(module)
        return repo, commit, dataset

    def _install_venv_interpreter(self, dataset: Path) -> Path:
        venv = dataset / "venv_antigravity"
        bindir = venv / "bin"
        bindir.mkdir(parents=True)
        target = Path(sys.executable).resolve()
        interpreter = bindir / "python"
        interpreter.symlink_to(target)
        (venv / "pyvenv.cfg").write_text(
            "\n".join(
                (
                    f"home = {target.parent}",
                    "include-system-site-packages = false",
                    f"version = {platform.python_version()}",
                    f"executable = {target}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        self.assertTrue(interpreter.is_symlink())
        self.assertTrue(interpreter.resolve().is_file())
        self.assertFalse(interpreter.resolve().is_symlink())
        return interpreter

    def _bind_official_runtime(self, module) -> None:
        official = Path(module.OFFICIAL_INTERPRETER)
        executable = mock.patch.object(module.sys, "executable", str(official))
        prefix = mock.patch.object(module.sys, "prefix", str(official.parent.parent))
        executable.start()
        prefix.start()
        self.addCleanup(executable.stop)
        self.addCleanup(prefix.stop)

    def _authority_digest(self, module, authority) -> str:
        return sha256(module.canonical_json_bytes(authority)).hexdigest()

    def test_launcher_file_exists_antigravity(self) -> None:
        self.assertTrue(LAUNCHER_PATH.is_file())

    def test_official_paths_and_smoke_names_are_fixed_antigravity(self) -> None:
        module = self.load_launcher()
        for name, path in OFFICIAL_PATHS.items():
            self.assertEqual(getattr(module, name), path)
        self.assertEqual(module.SMOKE_DATASETS, EXPECTED_SMOKE_DATASETS)
        self.assertEqual(module.SMOKE_DATASETS, tuple(sorted(module.SMOKE_DATASETS)))
        self.assertEqual(len(module.SMOKE_DATASETS), 22)
        self.assertEqual(module.EXPECTED_DATASET_COUNT, 1117)
        self.assertEqual(module.INNER_NUMERICAL_THREADS, 1)

    def test_check_rejects_non_lowercase_or_dirty_commit_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        with self.assertRaisesRegex(ValueError, "reviewed commit"):
            module.check(commit.upper())
        with self.assertRaisesRegex(ValueError, "reviewed commit"):
            module.check("not-a-commit")
        with self.assertRaisesRegex(ValueError, "reviewed commit"):
            module.check("0" * 40)
        dirty = module.REPOSITORY_ROOT / "dirty_antigravity.txt"
        dirty.write_text("dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "clean"):
            module.check(commit)

    def test_check_records_authority_without_writes_or_labels_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, dataset = self._prepare_fixture(module)
        before = {path: path.stat().st_mtime_ns for path in dataset.rglob("*")}
        before_names = {path.relative_to(dataset) for path in dataset.rglob("*")}
        sys.modules.pop(FORBIDDEN_LABEL_MODULE, None)
        result = module.check(commit)
        after_names = {path.relative_to(dataset) for path in dataset.rglob("*")}
        self.assertEqual(after_names, before_names)
        self.assertEqual(
            {path: path.stat().st_mtime_ns for path in dataset.rglob("*")},
            before,
        )
        self.assertEqual(result["schema"], "paper-r4-launch-authority-antigravity.v1")
        self.assertEqual(result["reviewed_commit"], commit)
        self.assertIs(result["git_clean"], True)
        self.assertEqual(result["database_sha256"], module.EXPECTED_DATABASE_SHA256)
        self.assertEqual(result["executable"], str(module.OFFICIAL_INTERPRETER))
        self.assertTrue(module.OFFICIAL_INTERPRETER.is_symlink())
        self.assertEqual(result["executable_realpath"], str(module.OFFICIAL_INTERPRETER.resolve()))
        self.assertNotEqual(result["executable"], result["executable_realpath"])
        self.assertEqual(result["python_prefix"], str(module.OFFICIAL_INTERPRETER.parent.parent))
        self.assertEqual(result["python_base_prefix"], module.sys.base_prefix)
        self.assertRegex(result["python_identity_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(result["r4_absent"], True)
        self.assertEqual(result["dataset_workers"], max(1, os.cpu_count() - 1))
        self.assertEqual(result["inner_numerical_threads"], 1)
        self.assertEqual(result["logical_cpu_count"], os.cpu_count())
        self.assertEqual(set(result["scoring_closure_sha256"]), set(module.SCORING_CLOSURE_PATHS))
        self.assertRegex(result["r3_digest"], r"^[0-9a-f]{64}$")
        self.assertRegex(result["launcher_sha256"], r"^[0-9a-f]{64}$")
        backend = result["backend"]
        self.assertEqual(backend["selected_backend"], "cpu")
        self.assertIn("pytorch_version", backend)
        self.assertIn("mlx_version", backend)
        self.assertIn("mps_available", backend)
        self.assertNotIn(FORBIDDEN_LABEL_MODULE, sys.modules)
        self.assertFalse(module.OFFICIAL_R4_ROOT.exists())
        self.assertFalse(module.OFFICIAL_OPS_ROOT.exists())
        self.assertFalse(module.OFFICIAL_SMOKE_ROOT.exists())

    def test_check_rejects_symlink_r3_and_existing_r4_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module, symlink_r3=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            module.check(commit)
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module, create_r4=True)
        with self.assertRaisesRegex(ValueError, "r4"):
            module.check(commit)

    def test_authority_loader_rejects_unknown_fields_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        authority = module.check(commit)
        authority["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown|schema"):
            module.validate_authority(authority)

    def _write_smoke_result(self, module, authority, pipeline=None) -> str:
        paths = module.smoke_ops_paths()
        paths.root.mkdir(parents=True, exist_ok=True)
        bare = {
            "schema": module.SMOKE_RESULT_SCHEMA,
            "authority_sha256": sha256(module.canonical_json_bytes(authority)).hexdigest(),
            "pipeline": pipeline or {"status": "completed"},
            "smoke_root": str(module.OFFICIAL_SMOKE_ROOT),
            "label_module_loaded": False,
        }
        digest = sha256(module.canonical_json_bytes(bare)).hexdigest()
        paths.smoke_result.write_bytes(module.canonical_json_bytes({**bare, "result_sha256": digest}))
        return digest

    def test_launch_requires_smoke_and_does_not_create_r4_before_authorization_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("spawn must not run before smoke authorization")

        with self.assertRaisesRegex(ValueError, "smoke"):
            module.launch(commit, spawn=forbidden)
        self.assertFalse(module.OFFICIAL_R4_ROOT.exists())
        self.assertFalse(module.OFFICIAL_OPS_ROOT.exists())

    def test_launch_uses_exclusive_flock_and_ignores_stale_pid_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        authority = module.check(commit)
        self._write_smoke_result(module, authority)
        paths = module.ops_paths()
        paths.root.mkdir(parents=True)
        paths.pid.write_text(f"{os.getpid()}\n", encoding="ascii")
        captured: dict[str, object] = {}

        def fake_spawn(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return SimpleNamespace(pid=424242)

        with mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}):
            with paths.worker_lock.open("a+b") as lock_stream:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, "already running"):
                    module.launch(commit, spawn=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("spawn")))
                self.assertFalse(module.OFFICIAL_R4_ROOT.exists())
            result = module.launch(commit, spawn=fake_spawn)
        self.assertEqual(result["status"], "launched")
        self.assertFalse(module.OFFICIAL_R4_ROOT.exists())
        command = captured["command"]
        kwargs = captured["kwargs"]
        self.assertEqual(command[0], str(module.OFFICIAL_INTERPRETER))
        self.assertNotEqual(command[0], str(Path(module.OFFICIAL_INTERPRETER).resolve()))
        self.assertEqual(
            command,
            [
                str(module.OFFICIAL_INTERPRETER),
                "-B",
                "-u",
                str(Path(module.__file__).resolve()),
                "_worker",
                "--reviewed-commit",
                commit,
                "--authority-sha256",
                result["authority_sha256"],
                "--lock-fd",
                str(kwargs["pass_fds"][0]),
                "--mode",
                "full",
            ],
        )
        self.assertEqual(command[command.index("--lock-fd") + 1], str(kwargs["pass_fds"][0]))
        self.assertEqual(command[command.index("--mode") + 1], "full")
        for name in module.NUMERICAL_THREAD_VARIABLES:
            self.assertEqual(kwargs["env"][name], "1")
        child = module.pipeline_command("full", module.OFFICIAL_R4_ROOT)
        self.assertEqual(child[:6], [str(module.OFFICIAL_INTERPRETER), "-B", "-u", "-m", module.PIPELINE_MODULE, "run"])
        self.assertNotEqual(child[0], str(Path(module.OFFICIAL_INTERPRETER).resolve()))
        self.assertNotIn("--dataset-workers", child)

    def test_stop_does_not_signal_unrelated_live_pid_when_lock_is_free_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        paths = module.ops_paths()
        paths.root.mkdir(parents=True)
        paths.pid.write_text(f"{os.getpid()}\n", encoding="ascii")
        result = module.stop(commit)
        self.assertEqual(result["status"], "already_stopped")
        self.assertFalse(result["signaled"])
        os.kill(os.getpid(), 0)

    def test_worker_writes_pid_after_lock_and_relays_signals_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        authority = module.check(commit)
        digest = self._write_smoke_result(module, authority)
        launch_authority = module.build_authority(commit, smoke_result_sha256=digest)
        paths = module.ops_paths()
        paths.root.mkdir(parents=True)
        received: list[int] = []
        lock_stream = paths.worker_lock.open("a+b")
        self.addCleanup(lock_stream.close)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker_fd = os.dup(lock_stream.fileno())
        with mock.patch.object(
            module,
            "pipeline_command",
            return_value=[sys.executable, "-c", "print('progress line')"],
        ), mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}):
            code = module._worker(
                commit,
                sha256(module.canonical_json_bytes(launch_authority)).hexdigest(),
                worker_fd,
                "full",
            )
        self.assertEqual(code, 0)
        self.assertEqual(int(paths.pid.read_text(encoding="ascii").strip()), os.getpid())
        events = paths.progress_events.read_text(encoding="utf-8")
        self.assertIn("paper-r4-progress-event-antigravity.v1", events)
        self.assertIn("progress line", events)

        class FakeChild:
            def poll(self):
                return None

            def send_signal(self, signum):
                received.append(signum)

        restore = module._install_child_signal_handlers(lambda: FakeChild())
        try:
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)
        finally:
            restore()
        self.assertEqual(received, [signal.SIGTERM])

    def test_status_reports_empty_running_incomplete_tampered_and_completed_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        empty = module.inspect_status(commit)
        self.assertEqual(empty["status"], "not_started")
        self.assertIsNone(empty["pid"])
        self.assertFalse(empty["lock_held"])
        authority = module.check(commit)
        digest = self._write_smoke_result(module, authority)
        launch_authority = {**authority, "smoke_result_sha256": digest}
        paths = module.ops_paths()
        encoded = module.canonical_json_bytes(launch_authority)
        paths.root.mkdir(parents=True)
        paths.authority.write_bytes(encoded)
        paths.authority_sha256.write_text(sha256(encoded).hexdigest() + "\n", encoding="ascii")
        module.OFFICIAL_R4_ROOT.mkdir()
        with mock.patch.object(module, "_pipeline_status", return_value={"status": "proposal_score_incomplete", "completed": 4, "failed": 1, "pending": 1112}):
            incomplete = module.inspect_status(commit)
        self.assertEqual(incomplete["status"], "incomplete")
        self.assertEqual(incomplete["completed"], 4)
        self.assertEqual(incomplete["failed"], 1)
        with paths.worker_lock.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            paths.pid.write_text("99999\n", encoding="ascii")
            with mock.patch.object(module, "_pipeline_status", return_value={"status": "proposal_score_incomplete", "completed": 4, "failed": 1, "pending": 1112}):
                running = module.inspect_status(commit)
        self.assertEqual(running["status"], "running")
        self.assertTrue(running["lock_held"])
        self.assertEqual(running["pid"], 99999)
        with mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}):
            completed = module.inspect_status(commit)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["completed"], 1117)
        with mock.patch.object(module, "_pipeline_status", side_effect=ValueError("namespace pollution")):
            tampered = module.inspect_status(commit)
        self.assertEqual(tampered["status"], "tampered")

    def test_resume_requires_identical_authority_and_incomplete_root_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            module.resume(commit, spawn=lambda *_a, **_k: SimpleNamespace(pid=1))
        authority = module.check(commit)
        digest = self._write_smoke_result(module, authority)
        with mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}):
            module.launch(commit, spawn=lambda *_a, **_k: SimpleNamespace(pid=1))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                module.resume(commit, spawn=lambda *_a, **_k: SimpleNamespace(pid=1))
        module.OFFICIAL_R4_ROOT.mkdir()
        with mock.patch.object(
            module,
            "verify_smoke_authorization",
            return_value={"result_sha256": digest},
        ), mock.patch.object(module, "_pipeline_status", return_value={"status": "proposal_score_incomplete", "pending": 1}):
            result = module.resume(commit, spawn=lambda *_a, **_k: SimpleNamespace(pid=2))
        self.assertEqual(result["status"], "resumed")
        forged = json.loads(module.ops_paths().authority.read_text(encoding="utf-8"))
        forged["python_version"] = "forged"
        module.ops_paths().authority.write_bytes(module.canonical_json_bytes(forged))
        with mock.patch.object(module, "_pipeline_status", return_value={"status": "proposal_score_incomplete"}):
            with self.assertRaisesRegex(ValueError, "authority|SHA256|schema"):
                module.resume(commit, spawn=lambda *_a, **_k: SimpleNamespace(pid=3))

    def test_smoke_selection_and_resume_equivalence_antigravity(self) -> None:
        from paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper import Task4Task5FailureBatchingTests
        from paper.equipment_anomaly_detection import runner_paper as runner
        from paper.equipment_anomaly_detection import proposal_score_paper as score
        from paper.equipment_anomaly_detection.contracts_paper import LowFrequencyResult
        from paper.equipment_anomaly_detection.routing_paper import PolicyBundle, PolicyRoute
        import numpy as np

        fixture = Task4Task5FailureBatchingTests("run")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        module = self.load_launcher()
        _repo, commit, dataset = self._prepare_fixture(module)
        module.OFFICIAL_DATABASE = fixture.fixture.fixture.db
        module.EXPECTED_DATABASE_SHA256 = sha256(module.OFFICIAL_DATABASE.read_bytes()).hexdigest()
        module.SMOKE_DATASETS = ("alpha", "beta")
        module.OFFICIAL_SMOKE_ROOT = dataset / "smoke_root_antigravity"
        official_r4 = module.OFFICIAL_R4_ROOT
        alpha = fixture.fixture._task4_task5_values()
        beta_raw = fixture.fixture.fixture._typed("beta")[0]
        raw = {"alpha": alpha[0], "beta": beta_raw}
        values = {"alpha": (alpha[1], alpha[2]), "beta": (alpha[1], alpha[2])}
        low = LowFrequencyResult(
            ("alpha", "beta"),
            fixture.fixture.fixture._typed("alpha")[3].rows + fixture.fixture.fixture._typed("beta")[3].rows,
            (),
            (),
            0,
            {"alpha": frozenset(), "beta": frozenset()},
        )
        policies = PolicyBundle(
            tuple(
                PolicyRoute(name, {dataset_name: frozenset() for dataset_name in raw}, {dataset_name: np.zeros(2) for dataset_name in raw}, {dataset_name: np.zeros(2) for dataset_name in raw})
                for name in ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")
            )
        )
        original_acceleration = runner.build_acceleration_manifest

        def run_child(mode, root, _paths, fail_beta_once=False, failed=None):
            if failed is None:
                failed = [fail_beta_once]

            def calculate(raw_value):
                if raw_value.dataset == "beta" and failed[0]:
                    failed[0] = False
                    raise RuntimeError("fixture beta failure")
                return values[raw_value.dataset]

            def acceleration(context_sha256):
                return original_acceleration(context_sha256, logical_cpu_count=1)

            config = runner.PipelineConfig(module.OFFICIAL_DATABASE, root, "smoke", module.SMOKE_DATASETS)
            with mock.patch.object(runner, "_closure", return_value={"runner_paper.py": "0" * 64}), mock.patch.object(
                runner, "build_catalog", return_value=fixture.fixture.catalog
            ), mock.patch.object(
                runner, "load_dataset_values", side_effect=lambda _path, name: raw[name]
            ), mock.patch.object(runner, "build_acceleration_manifest", side_effect=acceleration), mock.patch.object(
                score, "_score_dataset_task4_task5", side_effect=calculate
            ), mock.patch.object(score, "score_low_frequency", return_value=low), mock.patch.object(
                score, "classify_stability", return_value=alpha[4]
            ), mock.patch.object(score, "build_hard_evidence_surface", return_value=object()), mock.patch.object(
                score, "build_policies", return_value=policies
            ):
                first = runner.run_pipeline(config)
                if first["status"] != "completed":
                    return 2
            return 0

        sys.modules.pop(FORBIDDEN_LABEL_MODULE, None)
        with mock.patch.object(runner, "_closure", return_value={"runner_paper.py": "0" * 64}), mock.patch.object(
            runner, "build_catalog", return_value=fixture.fixture.catalog
        ), mock.patch.object(runner, "load_dataset_values", side_effect=lambda _path, name: raw[name]):
            first = module.smoke(commit, run_child=lambda mode, root, paths: run_child(mode, root, paths, fail_beta_once=True))
            self.assertEqual(first["pipeline"]["status"], "proposal_score_incomplete")
            self.assertEqual(first["pipeline"]["pending_datasets"], ["beta"])
            self.assertFalse(module.smoke_ops_paths().smoke_result.exists())
            self.assertNotIn("result_sha256", first)
            self.assertNotIn(FORBIDDEN_LABEL_MODULE, sys.modules)
            self.assertFalse(official_r4.exists())
            second = module.smoke(commit, run_child=lambda mode, root, paths: run_child(mode, root, paths, fail_beta_once=False))
            self.assertEqual(second["pipeline"]["status"], "completed")
            freeze = module.OFFICIAL_SMOKE_ROOT / "stage_artifacts_paper" / "proposal_freeze"
            resumed = {path.name: path.read_bytes() for path in sorted((freeze / "shards").iterdir())}
            resumed["proposal_manifest_paper.json"] = (freeze / "proposal_manifest_paper.json").read_bytes()
            shutil.rmtree(module.OFFICIAL_SMOKE_ROOT)
            shutil.rmtree(module.smoke_ops_paths().root)
            third = module.smoke(commit, run_child=lambda mode, root, paths: run_child(mode, root, paths, fail_beta_once=False))
            self.assertEqual(third["pipeline"]["status"], "completed")
        baseline_freeze = module.OFFICIAL_SMOKE_ROOT / "stage_artifacts_paper" / "proposal_freeze"
        baseline = {path.name: path.read_bytes() for path in sorted((baseline_freeze / "shards").iterdir())}
        baseline["proposal_manifest_paper.json"] = (baseline_freeze / "proposal_manifest_paper.json").read_bytes()
        self.assertEqual(resumed, baseline)
        self.assertFalse(official_r4.exists())
        self.assertNotIn(FORBIDDEN_LABEL_MODULE, sys.modules)
        self.assertTrue(module.smoke_ops_paths().smoke_result.is_file())

    def test_launch_acquires_lock_once_and_does_not_leak_parent_lock_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        authority = module.check(commit)
        self._write_smoke_result(module, authority)
        acquired: list[object] = []
        original = module._acquire_lock

        def counted(paths):
            acquired.append(paths)
            return original(paths)

        with mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}), mock.patch.object(
            module, "_acquire_lock", side_effect=counted
        ):
            module.launch(commit, spawn=lambda *_a, **_k: SimpleNamespace(pid=7))
        self.assertEqual(len(acquired), 1)
        self.assertFalse(module._worker_lock_is_held(module.ops_paths().worker_lock))

    def test_incomplete_smoke_does_not_publish_authorization_or_launch_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        module.OFFICIAL_SMOKE_ROOT.mkdir()
        with mock.patch.object(
            module, "_pipeline_status", return_value={"status": "proposal_score_incomplete", "pending": 1}
        ):
            result = module.smoke(commit, run_child=lambda *_a, **_k: 2)
        self.assertEqual(result["pipeline"]["status"], "proposal_score_incomplete")
        self.assertFalse(module.smoke_ops_paths().smoke_result.exists())
        self.assertNotIn("result_sha256", result)
        with self.assertRaisesRegex(ValueError, "smoke"):
            module.launch(commit, spawn=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("spawn")))
        self.assertFalse(module.OFFICIAL_R4_ROOT.exists())

    def test_smoke_authorization_rejects_forged_and_incomplete_results_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        authority = module.check(commit)
        digest = self._write_smoke_result(module, authority)
        module.OFFICIAL_SMOKE_ROOT.mkdir()
        with mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}):
            loaded = module.verify_smoke_authorization(commit)
        self.assertEqual(loaded["result_sha256"], digest)

        def reject(payload, expected="smoke", *, rehash=True):
            forged = dict(payload)
            if rehash and "result_sha256" in forged:
                forged["result_sha256"] = sha256(
                    module.canonical_json_bytes({key: value for key, value in forged.items() if key != "result_sha256"})
                ).hexdigest()
            module.smoke_ops_paths().smoke_result.write_bytes(module.canonical_json_bytes(forged))
            with mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}):
                with self.assertRaisesRegex(ValueError, expected):
                    module.verify_smoke_authorization(commit)

        original = json.loads(module.smoke_ops_paths().smoke_result.read_text(encoding="utf-8"))
        reject({**original, "schema": "forged-schema"}, "schema")
        reject({**original, "unexpected": True}, "schema|unknown|fields")
        missing = dict(original)
        missing.pop("smoke_root")
        reject(missing, "schema|fields")
        reject({**original, "authority_sha256": "a" * 64}, "authority")
        reject({**original, "smoke_root": "/tmp/forged_smoke_antigravity"}, "root")
        reject({**original, "label_module_loaded": True}, "label")
        incomplete = {**original, "pipeline": {"status": "proposal_score_incomplete"}}
        incomplete["result_sha256"] = sha256(
            module.canonical_json_bytes({key: value for key, value in incomplete.items() if key != "result_sha256"})
        ).hexdigest()
        reject(incomplete, "completed|incomplete")
        reject({**original, "result_sha256": "b" * 64}, "hash", rehash=False)
        module.smoke_ops_paths().smoke_result.write_bytes(module.canonical_json_bytes(original))
        with mock.patch.object(module, "_pipeline_status", return_value={"status": "proposal_score_incomplete"}):
            with self.assertRaisesRegex(ValueError, "semantic|completed"):
                module.verify_smoke_authorization(commit)

    def test_check_rejects_wrong_runtime_interpreter_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, dataset = self._prepare_fixture(module)
        wrong = dataset / "wrong_python_antigravity"
        wrong.write_bytes(b"not-the-official-interpreter")
        module.OFFICIAL_INTERPRETER = wrong
        with self.assertRaisesRegex(ValueError, "interpreter"):
            module.check(commit)

    def test_stop_does_not_signal_unrelated_live_pid_when_lock_is_held_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        module.STOP_SIGINT_SECONDS = 0.2
        module.STOP_SIGTERM_SECONDS = 0.2
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: child.kill() or child.wait())
        paths = module.ops_paths()
        paths.root.mkdir(parents=True)
        paths.pid.write_text(f"{child.pid}\n", encoding="ascii")
        with paths.worker_lock.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = module.stop(commit)
        self.assertFalse(result["signaled"])
        self.assertTrue(module._pid_exists(child.pid))

    def test_stop_signals_only_matching_worker_identity_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        module.STOP_SIGINT_SECONDS = 0.2
        module.STOP_SIGTERM_SECONDS = 0.2
        authority = module.check(commit)
        paths = module.ops_paths()
        paths.root.mkdir(parents=True)
        encoded = module.canonical_json_bytes(authority)
        paths.authority.write_bytes(encoded)
        paths.authority_sha256.write_text(sha256(encoded).hexdigest() + "\n", encoding="ascii")
        child = subprocess.Popen(
            [sys.executable, "-c", "import time,signal; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(30)"]
        )
        paths.pid.write_text(f"{child.pid}\n", encoding="ascii")
        paths.state.write_bytes(
            module.canonical_json_bytes(
                {
                    "schema": module.STATE_SCHEMA,
                    "status": "running",
                    "mode": "full",
                    "started_epoch": 1.0,
                    "heartbeat_epoch": 2.0,
                    "pid": child.pid,
                    "child_returncode": None,
                    "last_error": None,
                    "reviewed_commit": commit,
                }
            )
        )
        identity = self._canonical_worker_argv(module, commit, self._authority_digest(module, authority))
        with paths.worker_lock.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(module, "_process_command_identity", return_value=identity):
                result = module.stop(commit)
        child.wait(timeout=5)
        self.assertIn(result["signaled"], {"SIGINT", "SIGTERM"})
        self.assertFalse(module._pid_exists(child.pid))

    def test_launch_spawn_failure_leaves_r4_absent_and_retryable_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        authority = module.check(commit)
        self._write_smoke_result(module, authority)

        def boom(*_args, **_kwargs):
            raise RuntimeError("spawn failed")

        with mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}):
            with self.assertRaisesRegex(RuntimeError, "spawn failed"):
                module.launch(commit, spawn=boom)
            self.assertFalse(module.OFFICIAL_R4_ROOT.exists())
            self.assertFalse(module._worker_lock_is_held(module.ops_paths().worker_lock))
            published = module.ops_paths().authority.read_bytes()
            retry = module.launch(commit, spawn=lambda *_a, **_k: SimpleNamespace(pid=11))
        self.assertEqual(retry["status"], "launched")
        self.assertFalse(module.OFFICIAL_R4_ROOT.exists())
        self.assertEqual(module.ops_paths().authority.read_bytes(), published)
        self.assertFalse(module._worker_lock_is_held(module.ops_paths().worker_lock))

    def test_smoke_child_failure_leaves_root_absent_and_retryable_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)

        def boom(*_args, **_kwargs):
            raise RuntimeError("child spawn failed")

        with self.assertRaisesRegex(RuntimeError, "child spawn failed"):
            module.smoke(commit, run_child=boom)
        self.assertFalse(module.OFFICIAL_SMOKE_ROOT.exists())
        self.assertFalse(module._worker_lock_is_held(module.smoke_ops_paths().worker_lock))
        with mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}):
            retry = module.smoke(commit, run_child=lambda *_a, **_k: 0)
        self.assertEqual(retry["status"], "smoke_verified")
        self.assertFalse(module.OFFICIAL_SMOKE_ROOT.exists())
        self.assertFalse(module._worker_lock_is_held(module.smoke_ops_paths().worker_lock))

    def test_stop_identity_preserves_spaces_in_real_command_line_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, dataset = self._prepare_fixture(module)
        spaced = dataset / "Time Series"
        spaced.mkdir()
        script = spaced / "manage_paper_r4_proposal_antigravity.py"
        script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        interpreter = str(module.OFFICIAL_INTERPRETER)
        digest = "a" * 64
        worker = subprocess.Popen(
            [
                interpreter,
                "-B",
                "-u",
                str(script.resolve()),
                "_worker",
                "--reviewed-commit",
                commit,
                "--authority-sha256",
                digest,
                "--lock-fd",
                "3",
                "--mode",
                "full",
            ]
        )
        unrelated = subprocess.Popen([interpreter, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: worker.poll() is not None or worker.kill() or worker.wait())
        self.addCleanup(lambda: unrelated.poll() is not None or unrelated.kill() or unrelated.wait())
        identity = ()
        for _attempt in range(40):
            try:
                identity = module._process_command_identity(worker.pid)
            except ValueError:
                identity = ()
            if identity:
                break
            time.sleep(0.05)
        other: object = ()
        for _attempt in range(40):
            try:
                other = module._process_command_identity(unrelated.pid)
                break
            except ValueError:
                time.sleep(0.05)
        self.assertIsInstance(identity, tuple)
        self.assertTrue(all(type(token) is str for token in identity))
        self.assertIn(str(script.resolve()), identity)
        self.assertTrue(any("Time Series" in token for token in identity))
        self.assertNotIn(" ".join(identity), identity)
        self.assertIn(identity[0], module._official_interpreter_tokens())
        with mock.patch.object(module, "__file__", str(script.resolve())):
            self.assertTrue(module._command_is_this_worker(identity, commit, digest), msg=repr(identity))
            self.assertFalse(module._command_is_this_worker(other, commit, digest))
            self.assertFalse(module._command_is_this_worker(identity, "0" * 40, digest))
            self.assertFalse(module._command_is_this_worker(" ".join(identity), commit, digest))
            self.assertFalse(module._command_is_this_worker(identity, commit, "0" * 64))

    def _canonical_worker_argv(self, module, commit: str, authority_sha256: str, *, lock_fd: str = "3") -> tuple[str, ...]:
        return (
            str(module.OFFICIAL_INTERPRETER),
            "-B",
            "-u",
            str(Path(module.__file__).resolve()),
            "_worker",
            "--reviewed-commit",
            commit,
            "--authority-sha256",
            authority_sha256,
            "--lock-fd",
            lock_fd,
            "--mode",
            "full",
        )

    def test_process_command_identity_returns_exact_tokens_and_fails_closed_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, dataset = self._prepare_fixture(module)
        spaced = dataset / "Time Series"
        spaced.mkdir()
        script = spaced / "manage_paper_r4_proposal_antigravity.py"
        script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        interpreter = str(module.OFFICIAL_INTERPRETER)
        worker = subprocess.Popen(
            [interpreter, str(script.resolve()), "_worker", "--reviewed-commit", commit]
        )
        self.addCleanup(lambda: worker.poll() is not None or worker.kill() or worker.wait())
        identity = ()
        for _attempt in range(40):
            try:
                identity = module._process_command_identity(worker.pid)
            except ValueError:
                identity = ()
            if identity:
                break
            time.sleep(0.05)
        self.assertIsInstance(identity, tuple)
        self.assertIn(identity[0], module._official_interpreter_tokens())
        self.assertEqual(
            identity[1:5],
            (str(script.resolve()), "_worker", "--reviewed-commit", commit),
        )
        with mock.patch.object(module, "_read_process_argv", return_value=()):
            with mock.patch.object(module.subprocess, "check_output") as check_output:
                with self.assertRaisesRegex(ValueError, "exact OS argv"):
                    module._process_command_identity(worker.pid)
                check_output.assert_not_called()
        source = Path(module.__file__).read_text(encoding="utf-8")
        identity_fn = source.split("def _process_command_identity", 1)[1].split("\ndef ", 1)[0]
        self.assertNotIn("ps", identity_fn)
        self.assertNotIn("join", identity_fn)

    def test_command_is_this_worker_rejects_adversarial_near_matches_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        interpreter = str(module.OFFICIAL_INTERPRETER)
        launcher = str(Path(module.__file__).resolve())
        digest = "c" * 64
        canonical = self._canonical_worker_argv(module, commit, digest)
        self.assertTrue(module._command_is_this_worker(canonical, commit, digest))
        near_matches = (
            " ".join(canonical),
            (interpreter, launcher + " _worker --reviewed-commit " + commit),
            (interpreter, launcher, "_worker_helper", "--reviewed-commit", commit),
            (interpreter, launcher, "_worker", "--reviewed-commit " + commit),
            (interpreter, launcher, "_worker", "--reviewed-commit=" + commit),
            (interpreter, launcher, "_worker", "--reviewed-commit", commit + "0"),
            (interpreter, launcher, "_worker", commit, "--reviewed-commit"),
            (interpreter, launcher, "--reviewed-commit", commit, "_worker"),
            (interpreter, launcher, "_worker", "--reviewed-commit", commit, "--reviewed-commit", commit),
            (interpreter, launcher, "_worker", "--reviewed-commit", "mode", commit),
            (interpreter, launcher, "_worker", "--reviewed-commit"),
            (interpreter, launcher, "--reviewed-commit", commit),
            (launcher, "_worker", "--reviewed-commit", commit),
            (interpreter, "_worker", "--reviewed-commit", commit),
            (interpreter, launcher, "_worker", "--reviewed-commit", "0" * 40),
            canonical[:-2],
            canonical + ("--extra",),
            (*canonical[:5], "--reviewed-commit", commit, "--mode", "full"),
            (),
            None,
        )
        for candidate in near_matches:
            self.assertFalse(
                module._command_is_this_worker(candidate, commit, digest),
                msg=repr(candidate),
            )

    def test_smoke_rejects_concurrent_invocation_and_does_not_call_child_while_lock_held_antigravity(self) -> None:
        import threading

        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        child_calls: list[str] = []
        started = threading.Event()
        release = threading.Event()

        def holder(_mode, root, _paths):
            child_calls.append("holder")
            started.set()
            self.assertTrue(module._worker_lock_is_held(module.smoke_ops_paths().worker_lock))
            self.assertFalse(root.exists())
            self.assertTrue(release.wait(timeout=5))
            return 0

        def forbidden(*_args, **_kwargs):
            child_calls.append("forbidden")
            raise AssertionError("second smoke must not start a child")

        with mock.patch.object(module, "_pipeline_status", return_value={"status": "completed"}):
            first_error: list[BaseException] = []

            def run_first():
                try:
                    module.smoke(commit, run_child=holder)
                except BaseException as error:
                    first_error.append(error)

            thread = threading.Thread(target=run_first)
            thread.start()
            self.assertTrue(started.wait(timeout=5))
            with self.assertRaisesRegex(ValueError, "already running"):
                module.smoke(commit, run_child=forbidden)
            self.assertFalse(module.OFFICIAL_SMOKE_ROOT.exists())
            self.assertNotIn("forbidden", child_calls)
            release.set()
            thread.join(timeout=5)
        self.assertEqual(first_error, [])
        self.assertEqual(child_calls, ["holder"])
        self.assertFalse(module._worker_lock_is_held(module.smoke_ops_paths().worker_lock))

    def test_smoke_holds_lock_for_child_verification_and_is_separate_from_r4_lock_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        held: dict[str, bool] = {}

        def run_child(_mode, root, _paths):
            held["child"] = module._worker_lock_is_held(module.smoke_ops_paths().worker_lock)
            held["r4_during_smoke"] = module._worker_lock_is_held(module.ops_paths().worker_lock)
            self.assertFalse(root.exists())
            return 0

        def status(*_args, **_kwargs):
            held["status"] = module._worker_lock_is_held(module.smoke_ops_paths().worker_lock)
            return {"status": "completed"}

        original_publish = module._publish_smoke_result

        def publish(*args, **kwargs):
            held["publish"] = module._worker_lock_is_held(module.smoke_ops_paths().worker_lock)
            return original_publish(*args, **kwargs)

        r4_paths = module.ops_paths()
        r4_paths.root.mkdir(parents=True)
        with r4_paths.worker_lock.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(module, "_pipeline_status", side_effect=status), mock.patch.object(
                module, "_publish_smoke_result", side_effect=publish
            ):
                result = module.smoke(commit, run_child=run_child)
        self.assertEqual(result["status"], "smoke_verified")
        self.assertEqual(held["child"], True)
        self.assertEqual(held["status"], True)
        self.assertEqual(held["publish"], True)
        self.assertEqual(held["r4_during_smoke"], True)
        self.assertFalse(module._worker_lock_is_held(module.smoke_ops_paths().worker_lock))
        self.assertFalse(module.OFFICIAL_SMOKE_ROOT.exists())

    def test_r3_digest_covers_full_tree_and_rejects_symlinks_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        first = module._identity_digest(module.OFFICIAL_R3_ROOT)
        extra = module.OFFICIAL_R3_ROOT / "extra_identity_antigravity.json"
        extra.write_text('{"extra":true}\n', encoding="utf-8")
        second = module._identity_digest(module.OFFICIAL_R3_ROOT)
        self.assertNotEqual(first, second)
        linked = module.OFFICIAL_R3_ROOT / "link_antigravity.json"
        linked.symlink_to(extra)
        with self.assertRaisesRegex(ValueError, "symlink"):
            module._identity_digest(module.OFFICIAL_R3_ROOT)

    def test_require_official_runtime_accepts_only_lexical_venv_symlink_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, dataset = self._prepare_fixture(module)
        official = Path(module.OFFICIAL_INTERPRETER)
        self.assertTrue(official.is_symlink())
        identity = module._require_official_runtime()
        self.assertEqual(identity["executable"], str(official))
        self.assertEqual(identity["executable_realpath"], str(official.resolve()))
        self.assertEqual(identity["python_prefix"], str(official.parent.parent))
        self.assertEqual(identity["python_base_prefix"], module.sys.base_prefix)
        self.assertRegex(identity["python_identity_sha256"], r"^[0-9a-f]{64}$")
        with mock.patch.object(module.sys, "executable", str(official.resolve())):
            with self.assertRaisesRegex(ValueError, "interpreter"):
                module._require_official_runtime()
        with mock.patch.object(module.sys, "prefix", str(official.resolve().parent.parent)):
            with self.assertRaisesRegex(ValueError, "prefix|venv"):
                module._require_official_runtime()
        regular = dataset / "regular_python_antigravity"
        regular.write_bytes(official.resolve().read_bytes())
        regular.chmod(0o755)
        module.OFFICIAL_INTERPRETER = regular
        with mock.patch.object(module.sys, "executable", str(regular)):
            with mock.patch.object(module.sys, "prefix", str(regular.parent)):
                with self.assertRaisesRegex(ValueError, "symlink|venv|interpreter"):
                    module._require_official_runtime()

    def test_pinned_venv_check_reaches_authority_without_official_writes_antigravity(self) -> None:
        official = Path(OFFICIAL_PATHS["OFFICIAL_INTERPRETER"])
        self.assertTrue(official.is_file())
        self.assertTrue(official.is_symlink())
        module = self.load_launcher()
        repo, commit, dataset = self._prepare_fixture(module)
        seam = dataset / "production_check_seam_antigravity.py"
        seam.write_text(
            "\n".join(
                (
                    "from hashlib import sha256",
                    "import importlib.util",
                    "import json",
                    "from pathlib import Path",
                    "import sys",
                    f"launcher = Path({str(LAUNCHER_PATH)!r})",
                    "spec = importlib.util.spec_from_file_location('paper_r4_ops_prod_antigravity', launcher)",
                    "loaded = importlib.util.module_from_spec(spec)",
                    "sys.modules[spec.name] = loaded",
                    "spec.loader.exec_module(loaded)",
                    f"loaded.REPOSITORY_ROOT = Path({str(repo)!r})",
                    f"loaded.OFFICIAL_DATABASE = Path({str(module.OFFICIAL_DATABASE)!r})",
                    f"loaded.OFFICIAL_R3_ROOT = Path({str(module.OFFICIAL_R3_ROOT)!r})",
                    f"loaded.OFFICIAL_R4_ROOT = Path({str(module.OFFICIAL_R4_ROOT)!r})",
                    f"loaded.OFFICIAL_OPS_ROOT = Path({str(module.OFFICIAL_OPS_ROOT)!r})",
                    f"loaded.OFFICIAL_SMOKE_ROOT = Path({str(module.OFFICIAL_SMOKE_ROOT)!r})",
                    "loaded.EXPECTED_DATABASE_SHA256 = sha256(loaded.OFFICIAL_DATABASE.read_bytes()).hexdigest()",
                    f"result = loaded.check({commit!r})",
                    "print(json.dumps(result, sort_keys=True, separators=(',', ':')))",
                    "",
                )
            ),
            encoding="utf-8",
        )
        official_roots = (
            OFFICIAL_PATHS["OFFICIAL_R4_ROOT"],
            OFFICIAL_PATHS["OFFICIAL_OPS_ROOT"],
            OFFICIAL_PATHS["OFFICIAL_SMOKE_ROOT"],
            Path(str(OFFICIAL_PATHS["OFFICIAL_SMOKE_ROOT"]) + "_ops_antigravity"),
        )
        before = {path: (path.exists(), path.is_symlink()) for path in official_roots}
        completed = subprocess.run(
            [str(official), "-B", "-u", str(seam)],
            cwd=str(REPOSITORY_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema"], "paper-r4-launch-authority-antigravity.v1")
        self.assertEqual(payload["executable"], str(official))
        self.assertEqual(payload["python_prefix"], str(official.parent.parent))
        self.assertEqual(payload["python_base_prefix"], "/Library/Frameworks/Python.framework/Versions/3.14")
        self.assertEqual(payload["executable_realpath"], str(official.resolve()))
        self.assertNotEqual(payload["executable"], payload["executable_realpath"])
        self.assertRegex(payload["python_identity_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(payload["reviewed_commit"], commit)
        self.assertFalse(module.OFFICIAL_R4_ROOT.exists())
        self.assertFalse(module.OFFICIAL_OPS_ROOT.exists())
        self.assertFalse(module.OFFICIAL_SMOKE_ROOT.exists())
        self.assertEqual({path: (path.exists(), path.is_symlink()) for path in official_roots}, before)
        for path in official_roots:
            self.assertFalse(path.exists())
            self.assertFalse(path.is_symlink())

    def test_command_is_this_worker_rejects_unrelated_c_prefix_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        digest = "d" * 64
        canonical = self._canonical_worker_argv(module, commit, digest)
        prefixed = (canonical[0], "-c", "raise SystemExit(0)") + canonical[1:]
        self.assertTrue(module._command_is_this_worker(canonical, commit, digest))
        self.assertFalse(module._command_is_this_worker(prefixed, commit, digest))
        self.assertFalse(module._command_is_this_worker((canonical[0], "-c") + canonical[1:], commit, digest))
        self.assertEqual(len(canonical), 13)

    def test_stop_rereads_argv_before_sigterm_and_refuses_mutation_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        module.STOP_SIGINT_SECONDS = 0.05
        module.STOP_SIGTERM_SECONDS = 0.05
        authority = module.check(commit)
        digest = self._authority_digest(module, authority)
        paths = module.ops_paths()
        paths.root.mkdir(parents=True)
        encoded = module.canonical_json_bytes(authority)
        paths.authority.write_bytes(encoded)
        paths.authority_sha256.write_text(digest + "\n", encoding="ascii")
        identity = self._canonical_worker_argv(module, commit, digest)
        mutated = identity[:1] + ("-c", "import time; time.sleep(30)") + identity[1:]
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: child.poll() is not None or child.kill() or child.wait())
        paths.pid.write_text(f"{child.pid}\n", encoding="ascii")
        paths.state.write_bytes(
            module.canonical_json_bytes(
                {
                    "schema": module.STATE_SCHEMA,
                    "status": "running",
                    "mode": "full",
                    "started_epoch": 1.0,
                    "heartbeat_epoch": 2.0,
                    "pid": child.pid,
                    "child_returncode": None,
                    "last_error": None,
                    "reviewed_commit": commit,
                }
            )
        )
        identities = iter((identity, mutated))
        sent: list[int] = []

        def fake_kill(pid, signum):
            if pid != child.pid:
                raise AssertionError("stop signaled an unexpected pid")
            sent.append(signum)
            if signum == signal.SIGTERM:
                raise AssertionError("SIGTERM must not be sent after identity mutation")

        with paths.worker_lock.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(module, "_process_command_identity", side_effect=lambda _pid: next(identities)):
                with mock.patch.object(module, "_pid_exists", return_value=True):
                    with mock.patch.object(module.os, "kill", side_effect=fake_kill):
                        result = module.stop(commit)
        self.assertEqual(result["status"], "stop_refused")
        self.assertNotIn(signal.SIGTERM, sent)
        self.assertEqual(sent, [signal.SIGINT])
        self.assertTrue(module._pid_exists(child.pid))

    def test_publish_smoke_result_is_create_only_and_byte_identical_reuse_antigravity(self) -> None:
        module = self.load_launcher()
        _repo, commit, _dataset = self._prepare_fixture(module)
        authority = module.check(commit)
        pipeline = {"status": "completed", "completed": 22, "failed": 0, "pending": 0}
        first = module._publish_smoke_result(authority, pipeline)
        paths = module.smoke_ops_paths()
        original = paths.smoke_result.read_bytes()
        inode = paths.smoke_result.stat().st_ino
        replaced: list[object] = []
        original_replace = module._atomic_replace

        def tracked_replace(path, payload):
            replaced.append((path, payload))
            return original_replace(path, payload)

        with mock.patch.object(module, "_atomic_replace", side_effect=tracked_replace):
            second = module._publish_smoke_result(authority, pipeline)
        self.assertEqual(paths.smoke_result.read_bytes(), original)
        self.assertEqual(paths.smoke_result.stat().st_ino, inode)
        self.assertEqual(second, first)
        self.assertEqual(replaced, [])
        different = {"status": "completed", "completed": 21, "failed": 1, "pending": 0}
        with self.assertRaisesRegex(ValueError, "existing smoke result"):
            module._publish_smoke_result(authority, different)
        self.assertEqual(paths.smoke_result.read_bytes(), original)
        self.assertEqual(paths.smoke_result.stat().st_ino, inode)
        self.assertEqual(replaced, [])


if __name__ == "__main__":
    unittest.main()
