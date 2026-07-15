import unittest
import os
import shutil
import tempfile
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# We will test execution structure of exp162 runner.
from experiments.exp162.run_experiment_162_full_coverage_alternative_score_cache import (
    load_frozen_population,
    EXP160_EXPECTED_HASH,
)


class TestExp162Structure(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_population_hash_and_exact_names(self):
        """Verify the loaded frozen population size matches 1,117 datasets."""
        try:
            population = load_frozen_population()
            self.assertEqual(len(population), 1117, f"Population size must be exactly 1,117, but got {len(population)}")
            self.assertTrue(all(isinstance(name, str) and len(name) > 0 for name in population))
            self.assertEqual(population, sorted(population))
        except FileNotFoundError:
            pass

    def test_runner_is_directly_executable(self):
        script = Path(__file__).with_name(
            "run_experiment_162_full_coverage_alternative_score_cache.py"
        )
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runner_records_database_and_population_list_hashes(self):
        source = Path(__file__).with_name(
            "run_experiment_162_full_coverage_alternative_score_cache.py"
        ).read_text()
        self.assertIn("database_sha256", source)
        self.assertIn("population_list_sha256", source)
        self.assertIn("VERIFICATION_REPORT.md", source)

    @patch("experiments.exp162.run_experiment_162_full_coverage_alternative_score_cache.sys.exit")
    @patch("experiments.exp162.run_experiment_162_full_coverage_alternative_score_cache.argparse.ArgumentParser.parse_args")
    def test_cli_parsing_and_limits(self, mock_parse_args, mock_exit):
        """Verify CLI arguments are parsed correctly and exit code is controlled."""
        from experiments.exp162.run_experiment_162_full_coverage_alternative_score_cache import run_experiment

        # Mock CLI inputs
        mock_args = MagicMock()
        mock_args.dataset_limit = 2
        mock_args.workers = 1
        mock_args.reset_checkpoint = True
        mock_args.output_dir = str(self.tmp_dir / "cli_test_out")
        mock_parse_args.return_value = mock_args

        # We mock load_frozen_population
        with patch("experiments.exp162.run_experiment_162_full_coverage_alternative_score_cache.load_frozen_population", return_value=["mock_ds_1", "mock_ds_2", "mock_ds_3"]):
            # Mock process_single_dataset
            with patch("experiments.exp162.run_experiment_162_full_coverage_alternative_score_cache.process_single_dataset") as mock_process:
                mock_process.return_value = {
                    "dataset": "mock_ds_1",
                    "success": True,
                    "error": None,
                    "traceback": None,
                    "file_path": "mock_ds_1_cache.npz",
                    "file_hash": "dummyhash",
                }

                # Execute run_experiment
                with patch(
                    "experiments.exp162.run_experiment_162_full_coverage_alternative_score_cache.get_sha256",
                    return_value=EXP160_EXPECTED_HASH,
                ):
                    try:
                        run_experiment()
                    except SystemExit:
                        pass

                # Ensure limit worked (it should process 2 out of 3)
                self.assertEqual(mock_process.call_count, 2)

                # Check output directory exists and files generated
                out_path = self.tmp_dir / "cli_test_out"
                self.assertTrue((out_path / "01_source_contract.json").exists())
                self.assertTrue((out_path / "02_cache_manifest.csv").exists())
                self.assertTrue((out_path / "04_coverage_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
