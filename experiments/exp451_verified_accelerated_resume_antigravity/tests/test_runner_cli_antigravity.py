"""Resume inventory, durable native checkpoint, and CLI progress tests."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

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
        """Every durable outcome carries counters, elapsed time, and ETA."""
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
            )
            self.assertEqual(
                set(event),
                {
                    "experiment", "dataset", "condition", "status", "completed",
                    "failed", "remaining", "total", "elapsed_seconds", "eta_seconds",
                },
            )
            self.assertEqual(event["remaining"], 234)
            self.assertGreaterEqual(event["eta_seconds"], 0.0)

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

            def scorer(dataset: str, condition: str):
                return {"dataset": dataset, "condition_id": condition, "payload": {}}

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
            )
            self.assertEqual(observed, [("success", True)])
