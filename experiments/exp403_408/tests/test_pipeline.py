from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


class PipelineTests(unittest.TestCase):
    def test_resume_skips_completed_dataset_hashes(self) -> None:
        from experiments.exp403_408.pipeline import run_smoke

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            first = run_smoke(output, limit=3, stop_after=2)
            second = run_smoke(output, limit=3, resume=True)

        self.assertEqual(first.completed, 2)
        self.assertEqual(second.recomputed_datasets, 1)
        self.assertEqual(second.completed, 3)

    def test_status_has_dashboard_fields(self) -> None:
        from experiments.exp403_408.pipeline import read_status, run_smoke

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_smoke(output, limit=1)
            status = read_status(output)

        self.assertTrue(
            {
                "stage",
                "current_dataset",
                "completed",
                "total",
                "progress_percent",
                "eta_seconds",
                "throughput_per_minute",
                "errors",
                "device",
                "checkpoint_sha256",
            }
            <= set(status)
        )
        self.assertEqual(status["status"], "completed")

    def test_failed_dataset_is_checkpointed_as_explicit_abstention(self) -> None:
        from experiments.exp403_408.pipeline import run_stage

        def failing_processor(_dataset: str) -> dict[str, object]:
            raise RuntimeError("fixture failure")

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = run_stage(
                "audit-features",
                ["small", "large"],
                output,
                processor=failing_processor,
            )
            checkpoint_path = next(
                (output / "checkpoints" / "audit-features").glob("small-*.json")
            )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        self.assertEqual(result.errors, 2)
        self.assertEqual(checkpoint["outcome"], "abstained")
        self.assertEqual(checkpoint["abstention_reason"], "stage_error")
        self.assertIn("fixture failure", checkpoint["error"])

    def test_stage_preserves_small_to_large_input_order(self) -> None:
        from experiments.exp403_408.pipeline import run_stage

        seen: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            run_stage(
                "report",
                ["small", "medium", "large"],
                Path(tmp),
                processor=lambda dataset: seen.append(dataset) or {"dataset": dataset},
                workers=1,
            )

        self.assertEqual(seen, ["small", "medium", "large"])

    def test_cpu_stages_cap_outer_workers_and_learned_stages_are_serial(self) -> None:
        from experiments.exp403_408.pipeline import stage_worker_count

        self.assertEqual(stage_worker_count("classical-predict", 99), 7)
        self.assertEqual(stage_worker_count("classical-predict", 1), 7)
        self.assertEqual(stage_worker_count("ssl-predict", 7), 1)


if __name__ == "__main__":
    unittest.main()
