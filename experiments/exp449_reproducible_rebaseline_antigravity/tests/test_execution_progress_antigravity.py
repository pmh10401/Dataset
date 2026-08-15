"""Fail-safe Smoke A operations metadata tests using private temporary seams."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp449_reproducible_rebaseline_antigravity import (
    operations_antigravity as operations,
)


def _event_antigravity() -> dict[str, object]:
    """Hand-written, result-neutral event fixture; no scientific payload exists here."""
    return {
        "schema": "exp449-operations-progress-antigravity.v1",
        "timestamp": "2026-08-15T00:00:00Z",
        "event": "wave_started",
        "wave": 1,
        "dataset": None,
        "completed": 0,
        "failed": 0,
        "pending": 23,
        "elapsed_seconds": 0.0,
        "throughput_per_second": 0.0,
        "eta_seconds": 0.0,
        "latest_error": None,
        "worker_pid": 449,
    }


class Exp449ExecutionProgressTests(unittest.TestCase):
    """Only an active lease may create durable metadata in a temporary namespace."""

    def _lease(self, root: Path) -> object:
        # The writer's authorization check is separately covered by the claim
        # suite.  This seam isolates real fd durability behavior from authority
        # document preparation.
        return SimpleNamespace(operations_root=root)

    def test_operations_metadata_names_and_required_result_neutral_event(self) -> None:
        """Renaming a durable record or dropping a field would make recovery ambiguous."""
        self.assertEqual(operations._EXECUTION_AUTHORITY_CONSUMED_NAME, "execution_authority_consumed_antigravity.json")
        self.assertEqual(operations._PROGRESS_NAME, "progress_antigravity.jsonl")
        self.assertEqual(operations._STATUS_NAME, "status_antigravity.json")
        self.assertEqual(operations._WORKER_STDOUT_NAME, "worker_stdout_antigravity.log")
        self.assertEqual(operations._WORKER_STDERR_NAME, "worker_stderr_antigravity.log")
        self.assertEqual({operations._TERMINAL_COMPLETE_NAME, operations._TERMINAL_INCOMPLETE_NAME}, {
            "terminal_complete_antigravity.json", "terminal_incomplete_antigravity.json",
        })
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary) / "operations_fixture_antigravity"
            root.mkdir()
            with patch.object(operations, "_authorized_operations_root_for_lease_antigravity", return_value=root):
                operations.append_authorized_progress_antigravity(self._lease(root), _event_antigravity())
            self.assertEqual(json.loads((root / "progress_antigravity.jsonl").read_text()), _event_antigravity())
            for forbidden in ("labels", "raw_values", "scores", "p_values", "traceback"):
                with self.subTest(forbidden=forbidden), patch.object(operations, "_authorized_operations_root_for_lease_antigravity", return_value=root):
                    with self.assertRaisesRegex(ValueError, "forbidden"):
                        operations.append_authorized_progress_antigravity(self._lease(root), {**_event_antigravity(), forbidden: "forbidden"})

    def test_progress_retries_eintr_and_short_writes_before_fsync(self) -> None:
        """Replacing the fd loop would permit a torn authoritative progress line."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary) / "operations_fixture_antigravity"
            root.mkdir()
            original_write = os.write
            calls = 0

            def write(descriptor: int, data: object) -> int:
                nonlocal calls
                calls += 1
                if calls == 1:
                    view = memoryview(data)  # type: ignore[arg-type]
                    return original_write(descriptor, view[:7])
                if calls == 2:
                    raise InterruptedError(errno.EINTR, "interrupted")
                return original_write(descriptor, data)  # type: ignore[arg-type]

            with patch.object(operations, "_authorized_operations_root_for_lease_antigravity", return_value=root), patch.object(operations.os, "write", side_effect=write):
                operations.append_authorized_progress_antigravity(self._lease(root), _event_antigravity())
            self.assertGreaterEqual(calls, 3)
            self.assertEqual(json.loads((root / "progress_antigravity.jsonl").read_text()), _event_antigravity())

    def test_progress_fsync_failure_is_reported_not_downgraded_to_cache_failure(self) -> None:
        """Swallowing fsync failure could schedule work whose progress is not durable."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary) / "operations_fixture_antigravity"
            root.mkdir()
            fsync_calls = 0
            def fsync(descriptor: int) -> None:
                nonlocal fsync_calls
                fsync_calls += 1
                raise OSError(errno.EIO, "fsync failed")
            with patch.object(operations, "_authorized_operations_root_for_lease_antigravity", return_value=root), patch.object(operations.os, "fsync", side_effect=fsync):
                with self.assertRaises(OSError):
                    operations.append_authorized_progress_antigravity(self._lease(root), _event_antigravity())
            self.assertGreaterEqual(fsync_calls, 1)

    def test_append_failure_leaves_no_following_progress_event(self) -> None:
        """Continuing after a failed append could replenish a queue without durable recovery evidence."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary) / "operations_fixture_antigravity"
            root.mkdir()
            with patch.object(operations, "_authorized_operations_root_for_lease_antigravity", return_value=root), patch.object(operations.os, "write", side_effect=OSError(errno.EIO, "append failed")):
                with self.assertRaises(OSError):
                    operations.append_authorized_progress_antigravity(self._lease(root), _event_antigravity())
            self.assertEqual((root / "progress_antigravity.jsonl").read_bytes(), b"")

    def test_progress_survives_a_crash_before_derived_cache_refresh(self) -> None:
        """Using the cache as recovery authority would lose a completed append on a parent crash."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary) / "operations_fixture_antigravity"
            root.mkdir()
            with patch.object(operations, "_authorized_operations_root_for_lease_antigravity", return_value=root):
                operations.append_authorized_progress_antigravity(self._lease(root), _event_antigravity())
            self.assertFalse((root / "status_antigravity.json").exists())
            self.assertEqual(operations._status_cache_payload_antigravity(root)["progress_tail"], _event_antigravity())

    def test_status_cache_replace_failure_preserves_authoritative_files(self) -> None:
        """Treating a cache write as authority would mutate recovery state on a harmless failure."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary) / "operations_fixture_antigravity"
            root.mkdir()
            receipt = root / "execution_authority_consumed_antigravity.json"
            receipt.write_text('{"authority":"fixed"}\n', encoding="utf-8")
            before = receipt.read_bytes()
            with patch.object(operations, "_authorized_operations_root_for_lease_antigravity", return_value=root), patch.object(operations.os, "replace", side_effect=OSError(errno.EIO, "replace failed")):
                with self.assertRaises(OSError):
                    operations.refresh_derived_status_antigravity(self._lease(root))
            self.assertEqual(receipt.read_bytes(), before)
            self.assertFalse((root / "status_antigravity.json").exists())

    def test_cache_failure_does_not_block_authoritative_complete_terminal(self) -> None:
        """Letting a derived-cache error suppress terminal publication would make recovery depend on a cache."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary) / "operations_fixture_antigravity"
            root.mkdir()
            names = operations.FROZEN_EXP449_CONFIG.smoke_datasets
            context = object.__new__(operations.Exp449Context)
            for name, value in {
                "experiment_id": "Exp449", "mode": "smoke", "run_root": Path(temporary) / "scientific_fixture_antigravity",
                "database_sha256": "a" * 64, "catalog_sha256": "b" * 64,
                "code_closure": {"fixture_antigravity.py": "c" * 64}, "lineage_sha256": "d" * 64,
                "codemap_sha256": "e" * 64, "interpreter": "fixture-python", "environment": {"fixture": True},
                "backend_record": {"selected_backend": "cpu"}, "seeds_and_constants": {"seed": 449},
                "historical_parity_status": "not_used",
            }.items():
                object.__setattr__(context, name, value)
            config = operations.Exp449WaveConfig(
                context=context,
                catalog={"catalog_sha256": "b" * 64, "datasets": [{"dataset": name} for name in names]},
                raw_datasets={name: object() for name in names}, acceleration_manifest={"fixture": True},
            )
            snapshot = operations._RecoverySnapshot(names, (), (), (), 1)
            with patch.object(operations, "append_authorized_progress_antigravity"), patch.object(operations, "refresh_derived_status_antigravity", side_effect=OSError(errno.EIO, "cache failed")), patch.object(operations, "_operations_root_for_context_antigravity", return_value=root):
                operations._emit_progress_antigravity(config, wave=1, snapshot=snapshot, started=0.0, _authorized_lease=object())
                status = operations._publish_terminal_complete_antigravity(config, snapshot)
            self.assertEqual(status.status, "dataset_local_complete")
            self.assertTrue((root / "terminal_complete_antigravity.json").is_file())

    def test_read_only_status_does_not_import_scorer_or_create_fixed_roots(self) -> None:
        """Importing the scorer or mkdir from status would breach its inspection-only contract."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            roots = tuple(base / f"root_{index}_antigravity" for index in range(6))
            with patch.object(operations, "_fixed_roots_antigravity", return_value=roots):
                result = operations.status_exp449_read_only_antigravity()
            self.assertEqual(result["status"], "not_started")
            self.assertTrue(all(not root.exists() for root in roots))

    def test_exp449_lock_records_task7_document_boundary_test_and_state_deltas(self) -> None:
        """Task 7 changes documented, boundary-test, and state provenance—not runtime closure."""
        lock_path = Path("docs/codemap/codemap_exp449_antigravity.lock")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertNotIn("paper", lock["changed_modules_since_prior_lock"])
        self.assertEqual(set(lock["changed_modules_since_prior_lock"]), {"docs", "experiments", "root"})


if __name__ == "__main__":
    unittest.main()
