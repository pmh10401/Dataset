"""Resume inventory, durable native checkpoint, and CLI progress tests."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp451_verified_accelerated_resume_antigravity.contracts_antigravity import (
    Exp451MigrationEntry,
    Exp451MigrationManifest,
)


def _catalog_antigravity() -> dict[str, object]:
    datasets = [f"dataset_{index:04d}_antigravity" for index in range(1117)]
    return {
        "catalog_sha256": "c" * 64,
        "datasets": [
            {
                "dataset": dataset,
                "train": {"count": 0},
                "test": {"count": 0},
            }
            for dataset in datasets
        ],
    }


def _manifest_antigravity(catalog: dict[str, object]) -> Exp451MigrationManifest:
    entries = []
    for condition, count in (("A", 1117), ("B", 1117), ("C", 1000), ("D", 1000)):
        for item in catalog["datasets"][:count]:
            dataset = item["dataset"]
            digest = sha256(f"{dataset}:{condition}".encode()).hexdigest()
            entries.append(
                Exp451MigrationEntry(
                    dataset=dataset,
                    condition_id=condition,
                    source_relative_path=f"source_antigravity/{condition}/{digest}.json",
                    source_envelope_sha256=digest,
                    payload_sha256=digest,
                    target_relative_path=f"target_antigravity/{condition}/{digest}.json",
                    target_envelope_sha256=digest,
                )
            )
    return Exp451MigrationManifest("a" * 64, tuple(entries), "b" * 64)


class ResumeInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = _catalog_antigravity()
        self.manifest = _manifest_antigravity(self.catalog)

    def test_verified_migration_has_exact_remaining_coordinate_schedule(self) -> None:
        """Runner counts imported successes and schedules only the 234 C/D gaps."""
        from experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity import (
            build_resume_inventory_antigravity,
        )

        inventory = build_resume_inventory_antigravity(self.manifest, self.catalog)

        self.assertEqual(inventory.completed_count, 4234)
        self.assertEqual(inventory.remaining_count, 234)
        self.assertEqual(
            inventory.by_condition, {"A": 1117, "B": 1117, "C": 1000, "D": 1000}
        )
        self.assertEqual(
            inventory.pending[:2],
            (
                ("dataset_1000_antigravity", "C"),
                ("dataset_1001_antigravity", "C"),
            ),
        )
        self.assertEqual(inventory.pending[-1], ("dataset_1116_antigravity", "D"))

    def test_progress_events_have_complete_live_status_fields(self) -> None:
        """ETA uses only measured native attempts, never the 4,234 imports."""
        from experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity import (
            progress_event_antigravity,
        )

        for status in ("success", "failure"):
            event = progress_event_antigravity(
                dataset="dataset_1000_antigravity",
                condition="C",
                status=status,
                completed=4234,
                failed=1,
                total=4468,
                elapsed_seconds=12.5,
                native_attempts=1,
                native_elapsed_seconds=12.5,
            )
            self.assertEqual(
                set(event),
                {
                    "experiment", "dataset", "condition", "status", "completed",
                    "failed", "remaining", "total", "elapsed_seconds", "eta_seconds",
                    "native_attempts", "native_elapsed_seconds",
                },
            )
            self.assertEqual(event["remaining"], 234)
            self.assertEqual(event["eta_seconds"], 12.5 * 234)

    def test_failure_ledger_is_unique_and_never_allows_wave_one_after_wave_two(self) -> None:
        """A retry is appended only after all first-wave work was recorded."""
        from experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity import (
            FailureRecordAntigravity,
            validate_failure_records_antigravity,
        )

        first = FailureRecordAntigravity("dataset_1000_antigravity", "C", 1, "ValueError", "first")
        second = FailureRecordAntigravity("dataset_1000_antigravity", "C", 2, "ValueError", "second")
        validate_failure_records_antigravity((first, second))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_failure_records_antigravity((first, first))
        later_first = FailureRecordAntigravity("dataset_1001_antigravity", "D", 1, "ValueError", "later")
        with self.assertRaisesRegex(ValueError, "wave one after wave two"):
            validate_failure_records_antigravity((first, second, later_first))

    def test_native_success_is_published_before_its_progress_event(self) -> None:
        """The schedule callback observes an already durable native checkpoint."""
        from experiments.exp451_verified_accelerated_resume_antigravity import runner_antigravity

        with TemporaryDirectory(prefix="exp451_runner_fixture_antigravity_") as temporary:
            root = Path(temporary).resolve()
            observed: list[tuple[str, bool]] = []

            class Result:
                dataset = "dataset_1000_antigravity"
                condition_id = "C"

            def scorer(dataset: str, condition: str):
                return Result()

            def persist(dataset: str, condition: str, result: object) -> None:
                del result
                (root / f"{dataset}_{condition}_antigravity.done").write_text("done", encoding="utf-8")

            def progress(event: dict[str, object]) -> None:
                marker = root / f"{event['dataset']}_{event['condition']}_antigravity.done"
                observed.append((str(event["status"]), marker.is_file()))

            runner_antigravity.execute_resume_coordinates_antigravity(
                (("dataset_1000_antigravity", "C"),),
                scorer=scorer,
                persist_success=persist,
                progress=progress,
                completed=4234,
                failed=0,
                total=4468,
                wave=1,
                append_failure=lambda *_: None,
                native_attempts=0,
                native_elapsed_seconds=0.0,
            )
            self.assertEqual(observed, [("success", True)])

    def test_wave_two_waits_for_full_wave_one_and_skips_successes(self) -> None:
        """The retry queue contains only first-wave failures after all other gaps."""
        from experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity import (
            FailureRecordAntigravity,
            build_resume_inventory_antigravity,
            wave_one_coordinates_antigravity,
        )

        failed = FailureRecordAntigravity("dataset_1000_antigravity", "C", 1, "RuntimeError", "mps")
        inventory = build_resume_inventory_antigravity(
            self.manifest, self.catalog, failures=(failed,)
        )
        first = wave_one_coordinates_antigravity(inventory)

        self.assertNotIn(("dataset_1000_antigravity", "C"), first)
        self.assertEqual(first[0], ("dataset_1001_antigravity", "C"))
        self.assertEqual(inventory.wave_two, (("dataset_1000_antigravity", "C"),))
        self.assertNotIn(("dataset_0000_antigravity", "A"), first)

    def test_scorer_coordinate_mismatch_is_failure_and_is_not_persisted(self) -> None:
        """A scorer cannot place a mismatched result in the scheduled checkpoint."""
        from experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity import (
            execute_resume_coordinates_antigravity,
        )

        class WrongResult:
            dataset = "other_antigravity"
            condition_id = "D"

        persisted: list[object] = []
        failures = []
        events = []
        execute_resume_coordinates_antigravity(
            (("dataset_1000_antigravity", "C"),),
            scorer=lambda *_: WrongResult(),
            persist_success=lambda *args: persisted.append(args),
            progress=events.append,
            completed=4234,
            failed=0,
            total=4468,
            wave=1,
            append_failure=failures.append,
            native_attempts=0,
            native_elapsed_seconds=0.0,
        )
        self.assertEqual(persisted, [])
        self.assertEqual(len(failures), 1)
        self.assertEqual(events[0]["status"], "failure")

    def test_exclusive_run_lock_rejects_a_second_process_and_symlink(self) -> None:
        """One contained lock prevents concurrent MPS runs and fails closed on links."""
        from experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity import (
            exclusive_run_lock_antigravity,
        )

        with TemporaryDirectory(prefix="exp451_lock_fixture_antigravity_") as temporary:
            root = Path(temporary).resolve() / "root_antigravity"
            with exclusive_run_lock_antigravity(root):
                with self.assertRaisesRegex(ValueError, "lock"):
                    with exclusive_run_lock_antigravity(root):
                        pass
                child = subprocess.run(
                    [
                        sys.executable, "-c",
                        "from pathlib import Path\n"
                        "from experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity "
                        "import exclusive_run_lock_antigravity\n"
                        f"with exclusive_run_lock_antigravity(Path({str(root)!r})):\n    pass\n",
                    ],
                    cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True,
                )
                self.assertNotEqual(child.returncode, 0)
                self.assertIn("run lock exists", child.stderr)
            root.mkdir(exist_ok=True)
            (root / "run_lock_antigravity.json").symlink_to(root / "outside_antigravity")
            with self.assertRaisesRegex(ValueError, "lock"):
                with exclusive_run_lock_antigravity(root):
                    pass

    def test_progress_path_rejects_symlink_and_status_uses_latest_event(self) -> None:
        """Status reports durable counters while JSONL links are never followed."""
        from experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity import (
            append_progress_event_antigravity,
            progress_event_antigravity,
            status_snapshot_antigravity,
        )

        with TemporaryDirectory(prefix="exp451_progress_fixture_antigravity_") as temporary:
            root = Path(temporary).resolve() / "root_antigravity"
            root.mkdir()
            target = root / "outside_antigravity.jsonl"
            target.write_text("outside", encoding="utf-8")
            (root / "progress_antigravity.jsonl").symlink_to(target)
            event = progress_event_antigravity(
                dataset="dataset_1000_antigravity", condition="C", status="success",
                completed=4235, failed=0, total=4468, elapsed_seconds=5.0,
                native_attempts=1, native_elapsed_seconds=5.0,
            )
            with self.assertRaisesRegex(ValueError, "progress"):
                append_progress_event_antigravity(root, event)
            (root / "progress_antigravity.jsonl").unlink()
            with patch(
                "experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity.os.fsync"
            ) as fsync:
                append_progress_event_antigravity(root, event)
            self.assertTrue(fsync.called)
            status = status_snapshot_antigravity(root, imported_completed=4234)
            self.assertEqual(status["completed"], 4235)
            self.assertEqual(status["latest_coordinate"], ["dataset_1000_antigravity", "C"])
            self.assertEqual(status["eta_seconds"], 5.0 * 233)

    def test_cli_accepts_mode_full_and_rejects_other_modes(self) -> None:
        """All planned Task8 commands receive --mode full without changing roots."""
        script_path = Path(__file__).resolve().parents[3] / "scratch" / "run_exp451_verified_accelerated_resume_antigravity.py"
        specification = importlib.util.spec_from_file_location("exp451_cli_antigravity", script_path)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        stdout = io.StringIO()
        with patch.object(module, "_status_antigravity", return_value={"experiment": "Exp451", "command": "status"}), patch("sys.stdout", stdout):
            self.assertEqual(module.main(["status", "--mode", "full"]), 0)
        self.assertIn('"command":"status"', stdout.getvalue())
        with patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                module.main(["status", "--mode", "smoke"])
