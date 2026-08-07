from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = ROOT / "scratch" / "run_exp421_mps_batched_acceleration.py"
PREREGISTRATION_PATH = ROOT / "docs" / "EXP421_PREREGISTRATION_KO.md"
RUNBOOK_PATH = ROOT / "docs" / "EXP421_RUNBOOK_KO.md"


def _load_cli():
    spec = importlib.util.spec_from_file_location("exp421_bounded_cli", CLI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load Exp421 CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()


class Exp421CliTests(unittest.TestCase):
    def test_help_exposes_only_bounded_runner_arguments(self):
        help_text = CLI.build_parser().format_help()
        for option in ("--database", "--output-root", "--dataset", "--repeats"):
            self.assertIn(option, help_text)
        self.assertNotIn("--full-exp419", help_text)

    def test_database_and_output_root_are_required(self):
        with self.assertRaises(SystemExit):
            CLI.main([])

    def test_official_exp419_root_and_forbidden_dataset_tokens_fail_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "train.db"
            database.touch()
            for arguments in (
                ["--database", str(database), "--output-root", str(Path(temp) / "exp419_deterministic_cpu_parallel")],
                ["--database", str(database), "--output-root", str(Path(temp) / "attempt"), "--dataset", "TEST_labels"],
                ["--database", str(database), "--output-root", str(Path(temp) / "attempt-2"), "--dataset", "sensor_label"],
                ["--database", str(database), "--output-root", str(Path(temp) / "attempt-3"), "--full-exp419"],
            ):
                with self.subTest(arguments=arguments), patch.object(CLI, "run_train_only_benchmark") as runner:
                    with self.assertRaises(SystemExit):
                        CLI.main(arguments)
                    runner.assert_not_called()

    def test_dispatch_passes_only_frozen_runner_inputs_and_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "train.db"
            database.touch()
            output_root = Path(temp) / "attempt"
            manifest = SimpleNamespace(
                state="failed",
                attempt_root=str(output_root),
                policy_bundle=SimpleNamespace(reasons={1: "mps_unavailable", 4: "mps_unavailable"}),
            )
            stdout = io.StringIO()
            with patch.object(CLI, "run_train_only_benchmark", return_value=manifest) as runner, patch.object(
                CLI.torch, "set_num_threads"
            ) as set_threads, redirect_stdout(stdout):
                result = CLI.main(
                    [
                        "--database", str(database),
                        "--output-root", str(output_root),
                        "--dataset", "B_train",
                        "--dataset", "A_train",
                        "--repeats", "3",
                    ]
                )
            self.assertEqual(result, 0)
            set_threads.assert_called_once_with(1)
            runner.assert_called_once_with(
                database=database.resolve(),
                output_root=output_root.resolve(),
                config=CLI.FROZEN_CONFIG,
                dataset_names=("B_train", "A_train"),
                repeats=3,
            )
            self.assertIn('"state": "failed"', stdout.getvalue())
            self.assertIn("manifest.json", stdout.getvalue())

    def test_cli_source_has_no_process_pool_or_official_full_run_path(self):
        source = CLI_PATH.read_text(encoding="utf-8")
        self.assertNotIn("multiprocessing", source)
        self.assertNotIn("ProcessPoolExecutor", source)
        self.assertIn("set_num_threads", source)
        self.assertIn("run_train_only_benchmark", source)

    def test_korean_preregistration_and_runbook_state_operating_boundaries(self):
        preregistration = PREREGISTRATION_PATH.read_text(encoding="utf-8")
        runbook = RUNBOOK_PATH.read_text(encoding="utf-8")
        combined = preregistration + "\n" + runbook
        for phrase in ("Exp421", "TRAIN-only", "진단", "TEST", "Exp419", "MLX", "1,117"):
            self.assertIn(phrase, combined)
        self.assertIn("quick_check", runbook)
        self.assertIn("취소", runbook)
        self.assertIn("SHA-256", runbook)


if __name__ == "__main__":
    unittest.main()
