from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.paper_public_baselines.artifacts import (
    canonical_json_bytes,
    write_create_only_bytes,
)
from experiments.paper_public_baselines.cli import main as cli_main, parse_method_list
from experiments.paper_public_baselines.config import ALPHAS, METHOD_IDS, SEED
from experiments.paper_public_baselines.tests.test_runner import _build_database


def _write_toy_prepared(root: Path) -> Path:
    database = root / "toy.sqlite"
    _build_database(database)
    payload = {
        "schema": "paper-public-baselines.prepared.v1",
        "database_path": str(database.resolve()),
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "population_authority": "synthetic-test",
        "population_sha256": hashlib.sha256(b'["toy"]\n').hexdigest(),
        "full_dataset_count": 1,
        "datasets": ["toy"],
        "limited_for_smoke": True,
        "methods": list(METHOD_IDS),
        "alphas": list(ALPHAS),
        "seed": SEED,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    run_root = root / "run"
    encoded = canonical_json_bytes(payload)
    write_create_only_bytes(run_root / "prepared.json", encoded)
    write_create_only_bytes(
        run_root / "prepared.sha256",
        (hashlib.sha256(encoded).hexdigest() + "\n").encode(),
    )
    return run_root


class CliTests(unittest.TestCase):
    def test_score_cli_imports_no_label_module(self) -> None:
        code = (
            "import sys; import experiments.paper_public_baselines.cli; "
            "assert 'experiments.paper_public_baselines.label_access' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_method_list_is_exact_and_rejects_unknown_or_duplicate(self) -> None:
        self.assertEqual(
            parse_method_list("knn_distance,lof_novelty"),
            ("knn_distance", "lof_novelty"),
        )
        for invalid in ("", "invented", "knn_distance,knn_distance"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "method"):
                    parse_method_list(invalid)

    def test_assess_mlx_materializes_all_six_cpu_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = _write_toy_prepared(Path(directory))

            exit_code = cli_main(["assess-mlx", "--run-root", str(run_root)])
            payload = json.loads(
                run_root.joinpath("acceleration.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(tuple(record["method_id"] for record in payload["records"]), METHOD_IDS)
        self.assertTrue(all(record["selected_backend"] == "cpu" for record in payload["records"]))

    def test_score_refuses_to_start_before_mlx_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = _write_toy_prepared(Path(directory))

            with self.assertRaisesRegex(ValueError, "MLX assessment"):
                cli_main(
                    [
                        "score",
                        "--run-root",
                        str(run_root),
                        "--methods",
                        "knn_distance",
                        "--workers",
                        "1",
                    ]
                )

    def test_deep_score_rejects_more_than_one_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = _write_toy_prepared(Path(directory))
            cli_main(["assess-mlx", "--run-root", str(run_root)])

            with self.assertRaisesRegex(ValueError, "one worker"):
                cli_main(
                    [
                        "score",
                        "--run-root",
                        str(run_root),
                        "--methods",
                        "deep_svdd_1d",
                        "--workers",
                        "2",
                    ]
                )

    def test_evaluate_dispatch_verifies_manifest_before_label_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sys.modules.pop("experiments.paper_public_baselines.label_access", None)

            with self.assertRaisesRegex(ValueError, "manifest"):
                cli_main(
                    [
                        "evaluate",
                        "--run-root",
                        directory,
                        "--exp448-root",
                        str(Path(directory) / "exp448"),
                    ]
                )

        self.assertNotIn(
            "experiments.paper_public_baselines.label_access", sys.modules
        )


if __name__ == "__main__":
    unittest.main()
