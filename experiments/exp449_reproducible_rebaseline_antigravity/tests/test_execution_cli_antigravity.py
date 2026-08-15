"""Task 6 command-boundary tests using a temporary Git authority fixture."""

from __future__ import annotations

from dataclasses import fields
import os
import fcntl
import signal
import select
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import patch
from unittest.mock import Mock

from experiments.exp449_reproducible_rebaseline_antigravity import operations_antigravity as operations
from experiments.exp449_reproducible_rebaseline_antigravity import execution_authority_antigravity as authority
from experiments.exp449_reproducible_rebaseline_antigravity.tests.test_execution_claim_antigravity import (
    _context_antigravity,
    _verified_antigravity,
)
from paper.equipment_anomaly_detection.artifacts_paper import canonical_json_bytes


class Exp449ExecutionCliTests(unittest.TestCase):
    """Tests name the unsafe command behavior each one prevents."""

    def test_prepare_smoke_a_grant_creates_only_two_untracked_authority_json_files(self) -> None:
        """Skipping create-only publication or the Git-shape gate could prepare executable state."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            repository = Path(temporary)
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            roots = tuple(repository / f"root_{index}_antigravity" for index in range(6))
            claim = repository / "claim_antigravity.lock"
            authority_dir = repository / "docs" / "authorities"
            grant_path = authority_dir / "smoke_a_execution_grant_antigravity.json"
            digest_path = authority_dir / "smoke_a_execution_grant_digest_antigravity.json"
            frozen = SimpleNamespace(**{
                field.name: getattr(operations.FROZEN_EXP449_CONFIG, field.name)
                for field in fields(operations.FROZEN_EXP449_CONFIG)
            })
            frozen.smoke_a_scientific_root, frozen.smoke_a_operations_root = roots[:2]
            frozen.smoke_b_scientific_root, frozen.smoke_b_operations_root = roots[2:4]
            frozen.official_scientific_root, frozen.official_operations_root = roots[4:]
            facts = {
                "database": {"path": "/fixture/database.sqlite", "observed_size": 17, "sha256": "a" * 64},
                "contexts": {"smoke_a": {"catalog_sha256": "b" * 64, "context_sha256": "c" * 64, "acceleration_assessment_sha256": "d" * 64}},
                "code_closure": {"fixture_antigravity.py": "e" * 64},
                "lineage_lock_paper_sha256": "f" * 64,
                "codemap_lock_sha256": "0" * 64,
                "package_inventory": {"fixture": "1"},
                "seeds_and_constants": {"fixture": 449},
                "acceleration": {"selected_backend": "cpu"},
                "workers": {"logical_cpu_count": 12, "dataset_workers": 11},
            }
            with patch.object(operations, "FROZEN_EXP449_CONFIG", frozen), \
                 patch.object(authority, "FROZEN_EXP449_CONFIG", frozen), \
                 patch.object(operations, "SMOKE_A_CLAIM_PATH_ANTIGRAVITY", claim), \
                 patch.object(authority, "SMOKE_A_CLAIM_PATH_ANTIGRAVITY", claim), \
                 patch.object(operations, "_REPOSITORY_ROOT_ANTIGRAVITY", repository), \
                 patch.object(operations, "_fixed_roots_antigravity", return_value=roots), \
                 patch.object(operations, "readiness_exp449_antigravity", return_value=facts), \
                 patch.object(operations, "_sha256_file_antigravity", return_value="f" * 64), \
                 patch.object(operations, "SMOKE_A_GRANT_PATH_ANTIGRAVITY", grant_path, create=True), \
                 patch.object(operations, "SMOKE_A_DIGEST_PATH_ANTIGRAVITY", digest_path, create=True), \
                 patch.object(authority, "SMOKE_A_GRANT_PATH_ANTIGRAVITY", grant_path), \
                 patch.object(authority, "SMOKE_A_DIGEST_PATH_ANTIGRAVITY", digest_path):
                paths = operations.prepare_smoke_a_grant_antigravity("1" * 40)
            self.assertEqual(paths, (grant_path, digest_path))
            self.assertTrue(grant_path.is_file())
            self.assertTrue(digest_path.is_file())
            self.assertEqual(
                subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository, check=True, capture_output=True, text=True).stdout.splitlines(),
                [
                    "?? docs/authorities/smoke_a_execution_grant_antigravity.json",
                    "?? docs/authorities/smoke_a_execution_grant_digest_antigravity.json",
                ],
            )
            self.assertTrue(all(not path.exists() for path in (*roots, claim)))

    def test_worker_rejects_an_unlocked_canonical_claim_descriptor(self) -> None:
        """Dropping the live-lock probe would let a direct private worker mint authority."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            scientific, operational, claim_path = root / "science_antigravity", root / "operations_antigravity", root / "claim_antigravity.lock"
            context = _context_antigravity(scientific)
            verified = _verified_antigravity(scientific, operational, claim_path, context)
            descriptor = os.open(claim_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor, canonical_json_bytes(operations._claim_payload_antigravity(verified, context)))
                with self.assertRaisesRegex(ValueError, "not live"):
                    operations._authenticate_inherited_smoke_a_claim_fd_antigravity(descriptor, verified, context)
            finally:
                os.close(descriptor)

    def test_worker_receipt_validator_rejects_a_claim_identity_mismatch(self) -> None:
        """Omitting receipt semantics would let a worker reuse another consumed claim."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            scientific, operational, claim_path = root / "science_antigravity", root / "operations_antigravity", root / "claim_antigravity.lock"
            context = _context_antigravity(scientific)
            verified = _verified_antigravity(scientific, operational, claim_path, context)
            operational.mkdir()
            descriptor = os.open(claim_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor, canonical_json_bytes(operations._claim_payload_antigravity(verified, context)))
                receipt = operations._receipt_payload_antigravity(verified, context, claim_path, os.fstat(descriptor))
                receipt["claim_inode"] = int(receipt["claim_inode"]) + 1
                with self.assertRaisesRegex(ValueError, "receipt"):
                    operations._validate_smoke_a_consumption_receipt_antigravity(
                        receipt, verified, context, claim_path, os.fstat(descriptor),
                    )
            finally:
                os.close(descriptor)

    def test_worker_rejects_replaced_or_separately_contended_claim_before_mint(self) -> None:
        """Accepting a replacement or separate lock holder would bypass inherited-FD authority."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary); scientific, operational, claim = root / "science_antigravity", root / "ops_antigravity", root / "claim_antigravity.lock"
            context = _context_antigravity(scientific); verified = _verified_antigravity(scientific, operational, claim, context)
            operational.mkdir(); descriptor = os.open(claim, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor, canonical_json_bytes(operations._claim_payload_antigravity(verified, context)))
                receipt = operational / "execution_authority_consumed_antigravity.json"
                receipt.write_bytes(canonical_json_bytes(operations._receipt_payload_antigravity(verified, context, claim, os.fstat(descriptor))))
                replacement = root / "replacement_antigravity.lock"; replacement.write_bytes(claim.read_bytes()); os.replace(replacement, claim)
                with self.assertRaisesRegex(ValueError, "differs"):
                    operations._authenticate_inherited_smoke_a_claim_fd_antigravity(descriptor, verified, context)
            finally:
                os.close(descriptor)
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary); scientific, operational, claim = root / "science_antigravity", root / "ops_antigravity", root / "claim_antigravity.lock"
            context = _context_antigravity(scientific); verified = _verified_antigravity(scientific, operational, claim, context)
            operational.mkdir(); descriptor = os.open(claim, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            holder = -1
            try:
                os.write(descriptor, canonical_json_bytes(operations._claim_payload_antigravity(verified, context)))
                (operational / "execution_authority_consumed_antigravity.json").write_bytes(canonical_json_bytes(operations._receipt_payload_antigravity(verified, context, claim, os.fstat(descriptor))))
                holder = os.open(claim, os.O_RDWR); fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, "contended"):
                    operations._authenticate_inherited_smoke_a_claim_fd_antigravity(descriptor, verified, context)
            finally:
                if holder >= 0: os.close(holder)
                os.close(descriptor)

    def test_fork_only_worker_command_rejects_before_authority_verification(self) -> None:
        """A direct hidden worker command is unavailable before authority verification."""
        with patch.object(operations, "verify_smoke_a_authorization_commit_antigravity", side_effect=AssertionError("authority")):
            with self.assertRaises(SystemExit):
                operations.main_antigravity(["_worker"])

    def test_fork_child_watchdog_kills_its_group_after_parent_death_before_lifecycle(self) -> None:
        """A reparented real child must target only its own process group before any lifecycle write."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary); operations_root = root / "ops_antigravity"; operations_root.mkdir()
            sentinel = operations_root / "sentinel_ops_antigravity.json"; sentinel.write_bytes(b"unchanged")
            ready_read, ready_write = os.pipe(); result_read, result_write = os.pipe()
            supervisor = os.fork()
            if supervisor == 0:
                os.close(result_read)
                worker = os.fork()
                if worker == 0:
                    os.close(ready_read)
                    try:
                        os.setsid()
                        expected_parent = os.getppid()

                        def record_group(group: int, received: int) -> None:
                            os.write(result_write, f"{os.getpgrp()}:{group}:{received}".encode())

                        with patch.object(operations.os, "killpg", side_effect=record_group):
                            stop, watchdog = operations._start_smoke_a_parent_watchdog_antigravity(expected_parent)
                            os.write(ready_write, b"ready")
                            watchdog.join(timeout=1.0)
                            stop.set()
                            if watchdog.is_alive():
                                os._exit(3)
                        os._exit(0)
                    except BaseException:
                        os._exit(4)
                os.close(ready_write)
                os.read(ready_read, 5)
                os._exit(0)
            os.close(ready_read); os.close(ready_write); os.close(result_write)
            try:
                self.assertEqual(os.waitpid(supervisor, 0)[1], 0)
                readable, _writable, _exceptional = select.select([result_read], [], [], 2.0)
                self.assertEqual(readable, [result_read], "watchdog did not issue its group termination")
                payload = os.read(result_read, 128)
                own_group, targeted_group, received = map(int, payload.decode().split(":"))
                self.assertEqual(targeted_group, own_group)
                self.assertEqual(received, signal.SIGTERM)
                self.assertEqual(sentinel.read_bytes(), b"unchanged")
            finally:
                os.close(result_read)

    def test_actual_launcher_handshake_queues_before_ready_and_rejects_captured_parent_mismatch(self) -> None:
        """The fork launcher gates relays on readiness and kills a mismatched child before lifecycle."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary); scientific, operations_root, claim = root / "science_antigravity", root / "ops_antigravity", root / "claim_antigravity.lock"
            operations_root.mkdir(); sentinel = operations_root / "sentinel_ops_antigravity.json"; sentinel.write_bytes(b"unchanged")
            context = _context_antigravity(scientific); verified = _verified_antigravity(scientific, operations_root, claim, context)
            descriptor = os.open(claim, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600); lease = SimpleNamespace(claim_fd=descriptor, close=Mock())
            group_read, group_write = os.pipe(); proof_read, proof_write = os.pipe()

            def record_group(group: int, received: int) -> None:
                os.write(group_write, f"{group}:{received}".encode())

            def delayed_worker(_verified, _digest, _claim_fd, _mode, _parent_pid, ready_fd: int) -> int:
                os.kill(os.getppid(), signal.SIGTERM)
                time.sleep(0.1)
                early = bool(select.select([group_read], [], [], 0)[0])
                os.write(proof_write, b"early" if early else b"clean")
                os.write(ready_fd, b"R"); os.close(ready_fd)
                time.sleep(0.1)
                return 0

            try:
                with patch.object(operations, "verify_smoke_a_authorization_commit_antigravity", return_value=verified), \
                     patch.object(operations, "_authorized_smoke_a_wave_config_antigravity", return_value=SimpleNamespace(context=context)), \
                     patch.object(operations, "acquire_smoke_a_execution_lease_antigravity", return_value=lease), \
                     patch.object(operations, "_run_smoke_a_worker_antigravity", side_effect=delayed_worker), \
                     patch.object(operations.os, "killpg", side_effect=record_group):
                    launched = operations._launch_smoke_a_worker_antigravity("a" * 40, resume=False)
                self.assertEqual(os.read(proof_read, 16), b"clean")
                ready, _writable, _exceptional = select.select([group_read], [], [], 1.0)
                self.assertEqual(ready, [group_read])
                self.assertEqual(os.read(group_read, 64), f"{launched['worker_pid']}:{signal.SIGTERM}".encode())
                self.assertEqual(sentinel.read_bytes(), b"unchanged")

                parent_pid = os.getpid(); mismatch_read, mismatch_write = os.pipe()
                def record_mismatch(group: int, received: int) -> None:
                    os.write(mismatch_write, f"{os.getpgrp()}:{group}:{received}".encode())
                with patch.object(operations, "verify_smoke_a_authorization_commit_antigravity", return_value=verified), \
                     patch.object(operations, "_authorized_smoke_a_wave_config_antigravity", return_value=SimpleNamespace(context=context)), \
                     patch.object(operations, "_acquire_smoke_a_resume_lease_antigravity", return_value=lease), \
                     patch.object(operations.os, "getpid", return_value=parent_pid + 1), \
                     patch.object(operations.os, "killpg", side_effect=record_mismatch):
                    with self.assertRaisesRegex(ValueError, "group readiness"):
                        operations._launch_smoke_a_worker_antigravity("a" * 40, resume=True)
                readable, _writable, _exceptional = select.select([mismatch_read], [], [], 1.0)
                self.assertEqual(readable, [mismatch_read])
                own_group, targeted_group, received = map(int, os.read(mismatch_read, 64).decode().split(":"))
                self.assertEqual(targeted_group, own_group); self.assertEqual(received, signal.SIGTERM)
                self.assertEqual(sentinel.read_bytes(), b"unchanged")
                os.close(mismatch_read); os.close(mismatch_write)
            finally:
                os.close(group_read); os.close(group_write); os.close(proof_read); os.close(proof_write); os.close(descriptor)

    def test_resume_logs_append_and_parent_installs_group_relay_before_spawn(self) -> None:
        """Resume must preserve prior logs and an early signal must target the worker group."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary); operations_root = root / "ops_antigravity"; operations_root.mkdir()
            stdout, stderr = operations_root / operations._WORKER_STDOUT_NAME, operations_root / operations._WORKER_STDERR_NAME
            stdout.write_bytes(b"old-out\n"); stderr.write_bytes(b"old-err\n")
            claim = root / "claim_antigravity.lock"; descriptor = os.open(claim, os.O_RDWR | os.O_CREAT, 0o600)
            lease = SimpleNamespace(claim_fd=descriptor, close=Mock())
            verified = SimpleNamespace(approved_authority_commit="a" * 40, grant_sha256="b" * 64, grant=SimpleNamespace(operations_root=operations_root))
            registered: dict[int, object] = {}; killed: list[tuple[int, int]] = []
            def register(signum, handler):
                previous = registered.get(signum, signal.SIG_DFL); registered[signum] = handler; return previous
            def fork():
                self.assertIn(signal.SIGINT, registered); self.assertIn(signal.SIGTERM, registered)
                registered[signal.SIGINT](signal.SIGINT, None)
                registered[signal.SIGTERM](signal.SIGTERM, None)
                return 777
            real_close, real_read = os.close, os.read
            def close(descriptor: int) -> None:
                if descriptor not in {901, 902}:
                    real_close(descriptor)
            def read(descriptor: int, size: int) -> bytes:
                return b"R" if descriptor == 901 else real_read(descriptor, size)
            with patch.object(operations, "verify_smoke_a_authorization_commit_antigravity", return_value=verified), \
                 patch.object(operations, "_authorized_smoke_a_wave_config_antigravity", return_value=SimpleNamespace(context=object())), \
                 patch.object(operations, "_acquire_smoke_a_resume_lease_antigravity", return_value=lease), \
                 patch.object(operations.os, "fork", side_effect=fork), \
                 patch.object(operations.os, "waitpid", return_value=(777, 0)), \
                 patch.object(operations.os, "WIFEXITED", return_value=True), \
                 patch.object(operations.os, "WEXITSTATUS", return_value=0), \
                 patch.object(operations.signal, "signal", side_effect=register), \
                 patch.object(operations.os, "pipe", return_value=(901, 902)), \
                 patch.object(operations.os, "set_inheritable", return_value=None), \
                 patch.object(operations.select, "select", return_value=([901], [], [])), \
                 patch.object(operations.os, "read", side_effect=read), \
                 patch.object(operations.os, "close", side_effect=close), \
                 patch.object(operations.os, "killpg", side_effect=lambda pid, sig: killed.append((pid, sig))):
                operations._launch_smoke_a_worker_antigravity("a" * 40, resume=True)
            self.assertEqual(stdout.read_bytes(), b"old-out\n"); self.assertEqual(stderr.read_bytes(), b"old-err\n")
            self.assertEqual(killed, [(777, signal.SIGINT), (777, signal.SIGTERM)])
            os.close(descriptor)
