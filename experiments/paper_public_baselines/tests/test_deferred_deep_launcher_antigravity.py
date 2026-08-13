from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import fcntl
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from experiments.paper_public_baselines.artifacts import canonical_json_bytes
from experiments.paper_public_baselines.config import (
    ALPHAS,
    EXPECTED_DATASETS,
    EXPECTED_DB_SHA256,
    SEED,
)
from experiments.paper_public_baselines.runner import ScoreRunConfig, run_methods
from experiments.paper_public_baselines.tests.test_runner import (
    CountingScorer,
    _build_database,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER_PATH = (
    REPOSITORY_ROOT / "scratch/manage_deferred_deep_baselines_antigravity.py"
)
SCORING_CLOSURE_PATHS = (
    "scratch/run_paper_public_baselines.py",
    "experiments/paper_public_baselines/acceleration.py",
    "experiments/paper_public_baselines/artifacts.py",
    "experiments/paper_public_baselines/calibration.py",
    "experiments/paper_public_baselines/cli.py",
    "experiments/paper_public_baselines/config.py",
    "experiments/paper_public_baselines/contracts.py",
    "experiments/paper_public_baselines/data.py",
    "experiments/paper_public_baselines/methods/__init__.py",
    "experiments/paper_public_baselines/methods/classical.py",
    "experiments/paper_public_baselines/methods/deep_common.py",
    "experiments/paper_public_baselines/methods/deep_svdd.py",
    "experiments/paper_public_baselines/methods/drocc.py",
    "experiments/paper_public_baselines/preprocessing.py",
    "experiments/paper_public_baselines/runner.py",
)
RUN_CONFIG_SOURCE_PATHS = (
    "acceleration.py",
    "artifacts.py",
    "calibration.py",
    "config.py",
    "contracts.py",
    "data.py",
    "methods/__init__.py",
    "methods/classical.py",
    "methods/deep_common.py",
    "methods/deep_svdd.py",
    "methods/drocc.py",
    "preprocessing.py",
    "runner.py",
)


class DeferredDeepLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self._loaded_module_names: list[str] = []

    def tearDown(self) -> None:
        for name in self._loaded_module_names:
            sys.modules.pop(name, None)
        self.temporary.cleanup()

    def load_launcher(self):
        name = f"_deferred_deep_launcher_antigravity_{id(self)}_{len(self._loaded_module_names)}"
        specification = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        self._loaded_module_names.append(name)
        specification.loader.exec_module(module)
        return module

    @staticmethod
    def _ops_root(run_root: Path) -> Path:
        return run_root.parent / f"{run_root.name}_ops_antigravity"

    def _write_authority(
        self,
        module,
        run_root: Path,
        *,
        method: str = "deep_svdd_1d",
        **overrides,
    ) -> Path:
        ops_root = self._ops_root(run_root)
        ops_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "paper-public-deep-launch-authority-antigravity.v1",
            "method": method,
            "database_path": str((self.root / "official.sqlite").resolve()),
            "run_root": str(run_root.resolve()),
            "repository_root": str(REPOSITORY_ROOT.resolve()),
            "launcher_path": str(LAUNCHER_PATH.resolve()),
            "public_baseline_script_path": str(
                (REPOSITORY_ROOT / "scratch/run_paper_public_baselines.py").resolve()
            ),
            "database_sha256": EXPECTED_DB_SHA256,
            "expected_dataset_count": EXPECTED_DATASETS,
            "expected_pair_count": EXPECTED_DATASETS,
            "epochs": 50,
            "seed": SEED,
            "workers": 1,
            "alphas": list(ALPHAS),
            "executable": str(Path(sys.executable).resolve()),
            "python_version": "3.fixture",
            "git_commit": "1" * 40,
            "git_clean": True,
            "wrapper_sha256": "2" * 64,
            "scoring_closure_sha256": {
                path: format(index + 3, "064x")
                for index, path in enumerate(SCORING_CLOSURE_PATHS)
            },
            "logical_cpu_count": 8,
            "logical_cpu_minus_one_worker_count": 7,
            "actual_worker_count": 1,
            "cpu_worker_policy_assessment": (
                "logical-CPU-minus-one is inapplicable to the frozen sequential "
                "deep stage; official actual worker count remains one"
            ),
            "backend": {
                "selected_backend": "cpu",
                "pytorch_version": "fixture",
                "mlx_version": "absent",
                "mlx_metal_version": "absent",
                "mps_built": False,
                "mps_available": False,
                "tensor_eligible": True,
                "exact_accelerated_path_available": False,
                "transfer_compile_overhead_benchmarked": False,
                "cpu_fallback": "frozen deterministic PyTorch CPU implementation",
                "fallback_reason": (
                    "No exact MLX/MPS path preserves the frozen PyTorch "
                    "initialization, optimizer, autograd, score, and routing."
                ),
                "canonical_gate": {
                    "same_input_and_seed": True,
                    "finite_outputs": True,
                    "identical_candidates_all_alphas": True,
                    "score_absolute_tolerance": 1e-5,
                    "minimum_end_to_end_speedup": 1.1,
                    "include_transfer_and_synchronization": True,
                },
            },
        }
        payload.update(overrides)
        encoded = canonical_json_bytes(payload)
        (ops_root / "authority_antigravity.json").write_bytes(encoded)
        (ops_root / "authority_sha256_antigravity.txt").write_text(
            hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
        )
        return ops_root

    def _write_run_config(
        self,
        run_root: Path,
        authority: dict[str, object],
        **overrides,
    ) -> str:
        payload = {
            "schema": "paper-public-baselines.score-run.v1",
            "database_path": authority["database_path"],
            "database_sha256": authority["database_sha256"],
            "datasets": [f"dataset_{index:04d}" for index in range(EXPECTED_DATASETS)],
            "methods": [authority["method"]],
            "alphas": list(ALPHAS),
            "seed": SEED,
            "epochs": 50,
            "worker_policy": "classical_up_to_7_deep_exactly_1",
            "versions": {
                "python": authority["python_version"],
                "numpy": "fixture",
                "scipy": "fixture",
                "scikit_learn": "fixture",
                "torch": authority["backend"]["pytorch_version"],
            },
            "source_sha256": {
                relative: authority["scoring_closure_sha256"][
                    f"experiments/paper_public_baselines/{relative}"
                ]
                for relative in RUN_CONFIG_SOURCE_PATHS
            },
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
        payload.update(overrides)
        encoded = canonical_json_bytes(payload)
        digest = hashlib.sha256(encoded).hexdigest()
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "run_config.json").write_bytes(encoded)
        (run_root / "run_config.sha256").write_text(digest + "\n", encoding="ascii")
        return digest

    @staticmethod
    def _read_authority(ops_root: Path) -> dict[str, object]:
        import json

        return json.loads((ops_root / "authority_antigravity.json").read_text())

    @staticmethod
    def _write_state(
        ops_root: Path,
        *,
        method: str = "deep_svdd_1d",
        status: str = "running",
        started_epoch: float = 900.0,
        heartbeat_epoch: float = 999.0,
    ) -> None:
        payload = {
            "schema": "paper-public-deep-launch-state-antigravity.v1",
            "method": method,
            "status": status,
            "started_epoch": started_epoch,
            "heartbeat_epoch": heartbeat_epoch,
            "last_error": None,
            "failure_kind": None,
            "child_returncode": None,
        }
        (ops_root / "state_antigravity.json").write_bytes(
            canonical_json_bytes(payload)
        )

    def test_import_does_not_create_process_or_files(self) -> None:
        before = set(self.root.iterdir())
        self.load_launcher()
        self.assertEqual(set(self.root.iterdir()), before)

    def test_ops_paths_are_sibling_and_antigravity_named(self) -> None:
        module = self.load_launcher()

        paths = module.ops_paths(Path("/x/run"))

        self.assertEqual(paths.root, Path("/x/run_ops_antigravity"))
        self.assertEqual(
            (
                paths.authority.name,
                paths.authority_sha256.name,
                paths.pid.name,
                paths.state.name,
                paths.progress_log.name,
                paths.worker_lock.name,
            ),
            (
                "authority_antigravity.json",
                "authority_sha256_antigravity.txt",
                "pid_antigravity.txt",
                "state_antigravity.json",
                "progress_antigravity.log",
                "worker_lock_antigravity.lock",
            ),
        )

    def test_parser_accepts_only_two_deep_methods(self) -> None:
        module = self.load_launcher()
        self.assertEqual(module.parse_method("deep_svdd_1d"), "deep_svdd_1d")
        self.assertEqual(module.parse_method("drocc_1d"), "drocc_1d")
        with self.assertRaisesRegex(ValueError, "deep method"):
            module.parse_method("knn_distance")

    def test_stage_commands_freeze_official_order_and_configuration(self) -> None:
        module = self.load_launcher()
        commands = module.stage_commands(
            "deep_svdd_1d", Path("/db.sqlite"), Path("/run"), Path("/python")
        )
        self.assertEqual(
            [stage for stage, _ in commands],
            ["prepare", "assess_mlx", "score", "freeze"],
        )
        self.assertIn("--epochs", commands[2][1])
        self.assertEqual(
            commands[2][1][commands[2][1].index("--epochs") + 1], "50"
        )
        self.assertEqual(
            commands[2][1][commands[2][1].index("--workers") + 1], "1"
        )

    def test_status_counts_only_verified_complete_shard_meta_pairs(self) -> None:
        module = self.load_launcher()
        database = self.root / "toy.sqlite"
        _build_database(database)
        database_sha256 = hashlib.sha256(database.read_bytes()).hexdigest()
        run_root = self.root / "run"
        config = ScoreRunConfig(
            db_path=database,
            run_root=run_root,
            datasets=("toy",),
            methods=("deep_svdd_1d",),
            workers=1,
            database_sha256=database_sha256,
        )
        run_methods(config, scorer=CountingScorer())
        authority = {
            "method": "deep_svdd_1d",
            "database_sha256": database_sha256,
            "expected_pair_count": 1,
        }
        complete = module._scan_verified_pairs(run_root, authority)

        self.assertEqual(complete.completed, 1)
        self.assertEqual(complete.abstained, 0)
        metadata = next(run_root.glob("shards/**/*.meta"))
        metadata.unlink()

        with self.assertRaises(module.PartialShardPairError):
            module._scan_verified_pairs(
                run_root,
                authority,
                sleep=lambda _: None,
                half_pair_grace_seconds=0.001,
            )

    def test_scanner_retries_one_transient_half_pair_after_injected_grace(self) -> None:
        module = self.load_launcher()
        database = self.root / "toy.sqlite"
        _build_database(database)
        database_sha256 = hashlib.sha256(database.read_bytes()).hexdigest()
        run_root = self.root / "transient"
        config = ScoreRunConfig(
            db_path=database,
            run_root=run_root,
            datasets=("toy",),
            methods=("deep_svdd_1d",),
            workers=1,
            database_sha256=database_sha256,
        )
        run_methods(config, scorer=CountingScorer())
        metadata = next(run_root.glob("shards/**/*.meta"))
        metadata_bytes = metadata.read_bytes()
        metadata.unlink()
        sleeps: list[float] = []

        def finish_pair(delay: float) -> None:
            sleeps.append(delay)
            metadata.write_bytes(metadata_bytes)

        result = module._scan_verified_pairs(
            run_root,
            {
                "method": "deep_svdd_1d",
                "database_sha256": database_sha256,
                "expected_pair_count": 1,
            },
            sleep=finish_pair,
            half_pair_grace_seconds=0.0125,
        )

        self.assertEqual(result.completed, 1)
        self.assertEqual(sleeps, [0.0125])

    def test_held_flock_remains_running_with_stale_heartbeat(self) -> None:
        module = self.load_launcher()
        run_root = self.root / "stale"
        ops_root = self._write_authority(module, run_root)
        self._write_state(
            ops_root,
            status="running",
            started_epoch=700.0,
            heartbeat_epoch=819.0,
        )
        (ops_root / "pid_antigravity.txt").write_text(
            f"{os.getpid()}\n", encoding="ascii"
        )
        with (ops_root / "worker_lock_antigravity.lock").open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = module.inspect_status(run_root, now_epoch=1_000.0)

        self.assertEqual(result["status"], "running")
        self.assertIsNone(result["failure_kind"])

    def test_status_uses_flock_not_an_unrelated_live_pid(self) -> None:
        module = self.load_launcher()
        run_root = self.root / "status-lock"
        ops_root = self._write_authority(module, run_root)
        self._write_state(ops_root, heartbeat_epoch=999.0)
        (ops_root / "pid_antigravity.txt").write_text(
            f"{os.getpid()}\n", encoding="ascii"
        )

        unlocked = module.inspect_status(run_root, now_epoch=1_000.0)
        with (ops_root / "worker_lock_antigravity.lock").open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = module.inspect_status(run_root, now_epoch=1_000.0)

        self.assertEqual(unlocked["status"], "failed")
        self.assertEqual(unlocked["failure_kind"], "interrupted")
        self.assertEqual(locked["status"], "running")

    def test_status_lock_remains_authoritative_with_invalid_pid_metadata(self) -> None:
        module = self.load_launcher()
        run_root = self.root / "invalid-pid"
        ops_root = self._write_authority(module, run_root)
        self._write_state(ops_root, heartbeat_epoch=999.0)
        (ops_root / "pid_antigravity.txt").write_text("not-a-pid\n", encoding="ascii")

        with (ops_root / "worker_lock_antigravity.lock").open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = module.inspect_status(run_root, now_epoch=1_000.0)

        self.assertEqual(result["status"], "running")
        self.assertIsNone(result["pid"])

    def test_launch_ignores_stale_unrelated_live_pid_when_flock_is_free(self) -> None:
        module = self.load_launcher()
        database = self.root / "fixture.sqlite"
        database.write_bytes(b"fixture")
        run_root = self.root / "launch"
        paths = module.ops_paths(run_root)
        paths.root.mkdir(parents=True)
        paths.pid.write_text(f"{os.getpid()}\n", encoding="ascii")
        captured: dict[str, object] = {}

        def fake_spawn(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return SimpleNamespace(pid=424242)

        with mock.patch.object(
            module, "_build_authority", return_value={"fixture": True}
        ):
            result = module.launch(
                "deep_svdd_1d",
                database,
                run_root,
                executable=Path(sys.executable),
                spawn=fake_spawn,
            )

        self.assertEqual(result["status"], "launched")
        self.assertEqual(paths.pid.read_text(encoding="ascii"), "424242\n")
        command = captured["command"]
        kwargs = captured["kwargs"]
        self.assertEqual(len(kwargs["pass_fds"]), 1)
        self.assertEqual(command[command.index("--lock-fd") + 1], str(kwargs["pass_fds"][0]))

    def test_launch_refuses_held_canonical_flock(self) -> None:
        module = self.load_launcher()
        database = self.root / "fixture.sqlite"
        database.write_bytes(b"fixture")
        run_root = self.root / "duplicate"
        paths = module.ops_paths(run_root)
        paths.root.mkdir(parents=True)

        def forbidden_spawn(*args, **kwargs):
            raise AssertionError("spawn must not be called while flock is held")

        with paths.worker_lock.open("a+b") as lock_stream, mock.patch.object(
            module, "_build_authority", return_value={"fixture": True}
        ):
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "already running"):
                module.launch(
                    "deep_svdd_1d",
                    database,
                    run_root,
                    executable=Path(sys.executable),
                    spawn=forbidden_spawn,
                )

    def test_launch_rejects_symlink_run_root_before_ops_path_derivation(self) -> None:
        module = self.load_launcher()
        database = self.root / "fixture.sqlite"
        database.write_bytes(b"fixture")
        target = self.root / "target"
        target.mkdir()
        run_root = self.root / "run-link"
        run_root.symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "run-root.*symlink"):
            module.launch(
                "deep_svdd_1d",
                database,
                run_root,
                executable=Path(sys.executable),
                spawn=lambda *args, **kwargs: SimpleNamespace(pid=1),
            )

        self.assertFalse(self._ops_root(run_root).exists())
        self.assertFalse(self._ops_root(target).exists())

    def test_authority_loader_requires_exact_schema_and_official_constants(self) -> None:
        module = self.load_launcher()
        run_root = self.root / "authority"
        ops_root = self._write_authority(module, run_root)

        loaded = module._load_authority_core(module.ops_paths(run_root), run_root)
        self.assertEqual(loaded["expected_pair_count"], EXPECTED_DATASETS)

        for mutation, expected_error in (
            ({"epochs": 49}, "official"),
            ({"workers": 2}, "official"),
            ({"database_sha256": "f" * 64}, "database"),
            ({"unexpected": True}, "schema"),
        ):
            with self.subTest(mutation=mutation):
                self._write_authority(module, run_root, **mutation)
                with self.assertRaisesRegex(ValueError, expected_error):
                    module._load_authority_core(module.ops_paths(run_root), run_root)
        payload = self._read_authority(ops_root)
        payload.pop("backend")
        encoded = canonical_json_bytes(payload)
        (ops_root / "authority_antigravity.json").write_bytes(encoded)
        (ops_root / "authority_sha256_antigravity.txt").write_text(
            hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
        )
        with self.assertRaisesRegex(ValueError, "schema"):
            module._load_authority_core(module.ops_paths(run_root), run_root)
        encoded = canonical_json_bytes([])
        (ops_root / "authority_antigravity.json").write_bytes(encoded)
        (ops_root / "authority_sha256_antigravity.txt").write_text(
            hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
        )
        with self.assertRaisesRegex(ValueError, "schema"):
            module._load_authority_core(module.ops_paths(run_root), run_root)

    def test_authority_loader_rejects_non_boolean_backend_gate(self) -> None:
        module = self.load_launcher()
        run_root = self.root / "backend-gate"
        ops_root = self._write_authority(module, run_root)
        payload = self._read_authority(ops_root)
        payload["backend"]["canonical_gate"]["same_input_and_seed"] = 1
        encoded = canonical_json_bytes(payload)
        (ops_root / "authority_antigravity.json").write_bytes(encoded)
        (ops_root / "authority_sha256_antigravity.txt").write_text(
            hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
        )

        with self.assertRaisesRegex(ValueError, "backend"):
            module._load_authority_core(module.ops_paths(run_root), run_root)

    def test_verify_binds_frozen_and_verified_run_config_identity(self) -> None:
        module = self.load_launcher()
        run_root = self.root / "verify"
        ops_root = self._write_authority(module, run_root)
        authority = self._read_authority(ops_root)
        run_config_sha = self._write_run_config(run_root, authority)
        shards = tuple(
            {
                "dataset": f"dataset_{index:04d}",
                "method": "deep_svdd_1d",
                "path": f"shards/deep_svdd_1d/{index:04d}.json",
                "sha256": "a" * 64,
                "status": "complete",
            }
            for index in range(EXPECTED_DATASETS)
        )
        frozen = SimpleNamespace(
            run_root=run_root,
            run_config_sha256=run_config_sha,
            database_sha256=EXPECTED_DB_SHA256,
            pair_count=EXPECTED_DATASETS,
            shards=shards,
        )
        (run_root / "frozen_score_manifest.json").write_bytes(b"{}")

        with mock.patch.object(module, "verify_score_root", return_value=frozen):
            verified = module.verify_frozen(run_root)
        self.assertEqual(verified["run_config_sha256"], run_config_sha)

        mutations = (
            ({"database_sha256": "f" * 64}, "database"),
            ({"pair_count": EXPECTED_DATASETS - 1}, "pair count"),
            (
                {
                    "shards": (
                        {**shards[0], "method": "drocc_1d"},
                        *shards[1:],
                    )
                },
                "method",
            ),
        )
        for changes, expected_error in mutations:
            with self.subTest(changes=changes):
                altered = SimpleNamespace(**{**vars(frozen), **changes})
                with mock.patch.object(
                    module, "verify_score_root", return_value=altered
                ), self.assertRaisesRegex(ValueError, expected_error):
                    module.verify_frozen(run_root)

    def test_verify_rejects_run_config_source_identity_drift(self) -> None:
        module = self.load_launcher()
        run_root = self.root / "source-drift"
        ops_root = self._write_authority(module, run_root)
        authority = self._read_authority(ops_root)
        source_hashes = {
            relative: authority["scoring_closure_sha256"][
                f"experiments/paper_public_baselines/{relative}"
            ]
            for relative in RUN_CONFIG_SOURCE_PATHS
        }
        source_hashes["runner.py"] = "f" * 64
        run_config_sha = self._write_run_config(
            run_root, authority, source_sha256=source_hashes
        )
        frozen = SimpleNamespace(
            run_root=run_root,
            run_config_sha256=run_config_sha,
            database_sha256=EXPECTED_DB_SHA256,
            pair_count=EXPECTED_DATASETS,
            shards=tuple(
                {"method": "deep_svdd_1d"} for _ in range(EXPECTED_DATASETS)
            ),
        )
        (run_root / "frozen_score_manifest.json").write_bytes(b"{}")

        with mock.patch.object(
            module, "verify_score_root", return_value=frozen
        ), self.assertRaisesRegex(ValueError, "source"):
            module.verify_frozen(run_root)

        versions = {
            "python": "different",
            "numpy": "fixture",
            "scipy": "fixture",
            "scikit_learn": "fixture",
            "torch": authority["backend"]["pytorch_version"],
        }
        version_sha = self._write_run_config(run_root, authority, versions=versions)
        version_frozen = SimpleNamespace(**{**vars(frozen), "run_config_sha256": version_sha})
        with mock.patch.object(
            module, "verify_score_root", return_value=version_frozen
        ), self.assertRaisesRegex(ValueError, "versions"):
            module.verify_frozen(run_root)

    def test_fake_signal_handlers_forward_wait_and_kill_active_child(self) -> None:
        module = self.load_launcher()
        handlers: dict[int, object] = {}
        registrations: list[tuple[int, object]] = []
        events: list[object] = []

        class FakeChild:
            def poll(self):
                return None

            def send_signal(self, signum):
                events.append(("signal", signum))

            def wait(self, timeout=None):
                events.append(("wait", timeout))
                if timeout is not None:
                    raise subprocess.TimeoutExpired("fake", timeout)
                return -signal.SIGKILL

            def kill(self):
                events.append("kill")

        def fake_register(signum, handler):
            registrations.append((signum, handler))
            previous = handlers.get(signum, f"previous-{signum}")
            handlers[signum] = handler
            return previous

        child = FakeChild()
        restore = module._install_child_signal_handlers(
            lambda: child,
            register=fake_register,
            wait_timeout=0.25,
        )
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        handlers[signal.SIGINT](signal.SIGINT, None)
        restore()

        self.assertEqual(
            events,
            [
                ("signal", signal.SIGTERM),
                ("wait", 0.25),
                "kill",
                ("wait", None),
                ("signal", signal.SIGINT),
                ("wait", 0.25),
                "kill",
                ("wait", None),
            ],
        )
        self.assertEqual(
            [signum for signum, _ in registrations],
            [signal.SIGTERM, signal.SIGINT, signal.SIGTERM, signal.SIGINT],
        )

    def test_verify_never_imports_label_access(self) -> None:
        sys.modules.pop("experiments.paper_public_baselines.label_access", None)
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.load_launcher().verify_frozen(self.root / "incomplete")
        self.assertNotIn(
            "experiments.paper_public_baselines.label_access", sys.modules
        )


if __name__ == "__main__":
    unittest.main()
