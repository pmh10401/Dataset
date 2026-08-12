from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.paper_public_baselines.contracts import ScoreResult
from experiments.paper_public_baselines.runner import (
    ScoreRunConfig,
    freeze_score_root,
    run_methods,
    verify_score_root,
)


def _blob(values: tuple[float, ...]) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _build_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
            CREATE TABLE instances (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER NOT NULL,
                split TEXT NOT NULL,
                instance_index INTEGER NOT NULL,
                label TEXT NOT NULL,
                values_blob BLOB NOT NULL
            );
            """
        )
        connection.execute("INSERT INTO datasets(id, name) VALUES(1, 'toy')")
        rows = (
            ("TRAIN", 0, "0", _blob((0, 1, 2, 3, 4, 5, 6, 7))),
            ("TRAIN", 1, "0", _blob((1, 2, 3, 4, 5, 6, 7, 8))),
            ("TRAIN", 2, "0", _blob((2, 3, 4, 5, 6, 7, 8, 9))),
            ("TEST", 0, "0", _blob((0, 1, 2, 3, 4, 5, 6, 7))),
            ("TEST", 1, "1", _blob((9, 9, 9, 9, 9, 9, 9, 9))),
        )
        connection.executemany(
            "INSERT INTO instances(dataset_id, split, instance_index, label, values_blob) "
            "VALUES(1, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


class CountingScorer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, method_id: str, train: np.ndarray, test: np.ndarray, seed: int
    ) -> ScoreResult:
        self.calls += 1
        return ScoreResult(
            method_id=method_id,
            status="complete",
            reason=None,
            oof_scores=np.linspace(0.1, 0.3, len(train)),
            test_scores=np.linspace(0.4, 0.5, len(test)),
        )


class FailSecondScorer(CountingScorer):
    def __call__(
        self, method_id: str, train: np.ndarray, test: np.ndarray, seed: int
    ) -> ScoreResult:
        if self.calls == 1:
            raise RuntimeError("intentional second-pair failure")
        return super().__call__(method_id, train, test, seed)


class RunnerTests(unittest.TestCase):
    def _config(self, directory: str) -> ScoreRunConfig:
        root = Path(directory)
        database = root / "toy.sqlite"
        _build_database(database)
        digest = hashlib.sha256(database.read_bytes()).hexdigest()
        return ScoreRunConfig(
            db_path=database,
            run_root=root / "run",
            datasets=("toy",),
            methods=("knn_distance",),
            workers=1,
            database_sha256=digest,
        )

    def test_verified_complete_shard_is_reused_without_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            scorer = CountingScorer()

            first = run_methods(config, scorer=scorer)
            second = run_methods(config, scorer=scorer)

        self.assertEqual(first.completed, 1)
        self.assertEqual(second.completed, 1)
        self.assertEqual(scorer.calls, 1)
        self.assertEqual(second.reused, 1)

    def test_run_config_binds_every_scoring_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            run_methods(config, scorer=CountingScorer())
            run_config = config.run_root.joinpath("run_config.json").read_text(
                encoding="utf-8"
            )

        for relative in (
            "artifacts.py",
            "acceleration.py",
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
        ):
            with self.subTest(relative=relative):
                self.assertIn(f'"{relative}":', run_config)

    def test_run_config_rejects_nonpositive_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            with self.assertRaisesRegex(ValueError, "epochs"):
                ScoreRunConfig(
                    db_path=config.db_path,
                    run_root=config.run_root,
                    datasets=config.datasets,
                    methods=config.methods,
                    workers=1,
                    database_sha256=config.database_sha256,
                    epochs=0,
                )

    def test_tampered_shard_blocks_frozen_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            run_methods(config, scorer=CountingScorer())
            freeze_score_root(config.run_root)
            shard = next(config.run_root.glob("shards/**/*.json"))
            shard.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_score_root(config.run_root)

    def test_freeze_rejects_partial_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            run_methods(config, scorer=CountingScorer())
            config.run_root.joinpath("run_config.json").write_bytes(
                config.run_root.joinpath("run_config.json").read_bytes().replace(
                    b'["knn_distance"]', b'["knn_distance","lof_novelty"]'
                )
            )

            with self.assertRaisesRegex(ValueError, "run config|coverage"):
                freeze_score_root(config.run_root)

    def test_freeze_rejects_unregistered_extra_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            run_methods(config, scorer=CountingScorer())
            extra = config.run_root / "shards" / "knn_distance" / "extra.json"
            extra.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inventory"):
                freeze_score_root(config.run_root)

    def test_file_entrypoint_supports_two_process_workers(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.paper_public_baselines.tests.parallel_smoke",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("parallel-smoke-ok", completed.stdout)

    def test_disjoint_method_commands_share_one_frozen_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = self._config(directory)
            first = ScoreRunConfig(
                db_path=base.db_path,
                run_root=base.run_root,
                datasets=base.datasets,
                methods=("knn_distance", "lof_novelty"),
                workers=2,
                database_sha256=base.database_sha256,
            )
            second = ScoreRunConfig(
                db_path=base.db_path,
                run_root=base.run_root,
                datasets=base.datasets,
                methods=("knn_distance", "lof_novelty"),
                workers=1,
                database_sha256=base.database_sha256,
            )
            scorer = CountingScorer()

            run_methods(first, selected_methods=("knn_distance",), scorer=scorer)
            run_methods(second, selected_methods=("lof_novelty",), scorer=scorer)
            manifest = freeze_score_root(base.run_root)

        self.assertEqual(manifest.pair_count, 2)
        self.assertEqual(scorer.calls, 2)

    def test_each_completed_shard_is_published_before_later_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = self._config(directory)
            config = ScoreRunConfig(
                db_path=base.db_path,
                run_root=base.run_root,
                datasets=base.datasets,
                methods=("knn_distance", "lof_novelty"),
                workers=1,
                database_sha256=base.database_sha256,
            )

            with self.assertRaisesRegex(RuntimeError, "second-pair"):
                run_methods(config, scorer=FailSecondScorer())

            published = tuple(config.run_root.glob("shards/**/*.json"))

        self.assertEqual(len(published), 1)


if __name__ == "__main__":
    unittest.main()
