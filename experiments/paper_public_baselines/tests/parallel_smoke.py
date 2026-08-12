"""File-backed multiprocessing smoke used by test_runner on spawn platforms."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from .test_runner import _build_database
from ..runner import ScoreRunConfig, freeze_score_root, run_methods


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        database = root / "toy.sqlite"
        _build_database(database)
        config = ScoreRunConfig(
            db_path=database,
            run_root=root / "run",
            datasets=("toy",),
            methods=("knn_distance", "lof_novelty"),
            workers=2,
            database_sha256=hashlib.sha256(database.read_bytes()).hexdigest(),
        )
        summary = run_methods(config)
        manifest = freeze_score_root(config.run_root)
        if summary.completed != 2 or manifest.pair_count != 2:
            raise RuntimeError("parallel scoring coverage mismatch")
    print("parallel-smoke-ok")


if __name__ == "__main__":
    main()
