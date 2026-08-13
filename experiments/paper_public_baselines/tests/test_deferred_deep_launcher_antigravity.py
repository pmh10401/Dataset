from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

from experiments.paper_public_baselines.artifacts import canonical_json_bytes
from experiments.paper_public_baselines.runner import ScoreRunConfig, run_methods
from experiments.paper_public_baselines.tests.test_runner import (
    CountingScorer,
    _build_database,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER_PATH = (
    REPOSITORY_ROOT / "scratch/manage_deferred_deep_baselines_antigravity.py"
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
        run_root: Path,
        *,
        method: str = "deep_svdd_1d",
        expected_pairs: int = 1,
        database_sha256: str = "a" * 64,
    ) -> Path:
        ops_root = self._ops_root(run_root)
        ops_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "paper-public-deep-launch-authority-antigravity.v1",
            "method": method,
            "run_root": str(run_root.resolve()),
            "database_sha256": database_sha256,
            "expected_dataset_count": expected_pairs,
            "expected_pair_count": expected_pairs,
        }
        encoded = canonical_json_bytes(payload)
        (ops_root / "authority_antigravity.json").write_bytes(encoded)
        (ops_root / "authority_sha256_antigravity.txt").write_text(
            hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
        )
        return ops_root

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
        ops_root = self._write_authority(
            run_root, database_sha256=database_sha256
        )
        self._write_state(ops_root, status="score_complete")

        complete = module.inspect_status(run_root, now_epoch=1_000.0)

        self.assertEqual(complete["completed_pairs"], 1)
        self.assertEqual(complete["expected_pairs"], 1)
        self.assertEqual(complete["abstained_pairs"], 0)
        metadata = next(run_root.glob("shards/**/*.meta"))
        metadata.unlink()

        partial = module.inspect_status(run_root, now_epoch=1_000.0)

        self.assertEqual(partial["status"], "failed")
        self.assertEqual(partial["failure_kind"], "partial_shard_pair")

    def test_live_pid_with_stale_heartbeat_is_not_running(self) -> None:
        module = self.load_launcher()
        run_root = self.root / "stale"
        ops_root = self._write_authority(run_root)
        self._write_state(
            ops_root,
            status="running",
            started_epoch=700.0,
            heartbeat_epoch=819.0,
        )
        (ops_root / "pid_antigravity.txt").write_text(
            f"{os.getpid()}\n", encoding="ascii"
        )

        result = module.inspect_status(run_root, now_epoch=1_000.0)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_kind"], "stale_heartbeat")

    def test_duplicate_launch_refuses_live_worker(self) -> None:
        module = self.load_launcher()
        run_root = self.root / "duplicate"
        ops_root = self._write_authority(run_root)
        (ops_root / "pid_antigravity.txt").write_text(
            f"{os.getpid()}\n", encoding="ascii"
        )

        def forbidden_spawn(*args, **kwargs):
            raise AssertionError("spawn must not be called for a live worker")

        with self.assertRaisesRegex(ValueError, "already running"):
            module.launch(
                "deep_svdd_1d",
                self.root / "missing.sqlite",
                run_root,
                executable=Path(sys.executable),
                spawn=forbidden_spawn,
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
