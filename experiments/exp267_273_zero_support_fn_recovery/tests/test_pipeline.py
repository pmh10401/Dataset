from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import subprocess
import sys
import numpy as np
import inspect

from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp267_273_zero_support_fn_recovery.run_pipeline import STAGES
from experiments.exp267_273_zero_support_fn_recovery.run_pipeline import run_stage
from experiments.exp267_273_zero_support_fn_recovery.run_pipeline import _normalize_prediction_result


class PipelineStatusTests(unittest.TestCase):
    def test_status_has_dashboard_contract_fields(self) -> None:
        row = build_status(
            "experiment_268_multiscale_scattering_deviation", "running", "268",
            3, 1117, "dataset-a", errors=[], timestamps=((1.0, 1), (61.0, 3)),
            stdout_log=Path("full.log"), started_unix=1.0, now_unix=61.0,
        )
        required = {
            "id", "status", "phase", "done", "expected", "progress_percent",
            "current_dataset", "errors", "updated_at", "stdout_log",
            "recent_datasets_per_minute", "eta_seconds",
        }
        self.assertTrue(required <= set(row))
        self.assertEqual(row["recent_datasets_per_minute"], 2.0)

    def test_publish_writes_all_dashboard_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            status = build_status(
                "x", "queued", "audit", 0, 10, None, errors=[], timestamps=(),
                stdout_log=root / "full.log", started_unix=1.0, now_unix=1.0,
            )
            paths = publish_status(status, output_root=root, data_dir=root)
            self.assertTrue(all(path.exists() for path in paths))

    def test_cli_stage_contract_is_complete(self) -> None:
        self.assertEqual(
            STAGES,
            ("audit", "267", "268", "269", "270a", "270b", "270c", "271", "272", "273", "evaluate", "all", "device-benchmark"),
        )

    def test_direct_scratch_wrapper_can_import_package(self) -> None:
        root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            [sys.executable, root / "scratch/run_exp267_273_zero_support_fn_recovery.py", "--help"],
            cwd=Path(tempfile.gettempdir()), capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--stage", result.stdout)

    def test_promotion_stage_runs_frozen_evaluation_prerequisite(self) -> None:
        source = inspect.getsource(run_stage)
        promotion_branch = source.split('elif stage == "273":', 1)[1].split('elif stage == "all":', 1)[0]
        self.assertIn("_evaluate_stage(paths, datasets, runtime, root)", promotion_branch)

    def test_prediction_boundary_normalizes_numpy_indices(self) -> None:
        result = _normalize_prediction_result(((np.int64(3), np.int64(1)), "hash", False, 0.2))
        self.assertEqual(result, ((3, 1), "hash", False, 0.2))
        self.assertTrue(all(type(index) is int for index in result[0]))


if __name__ == "__main__":
    unittest.main()
