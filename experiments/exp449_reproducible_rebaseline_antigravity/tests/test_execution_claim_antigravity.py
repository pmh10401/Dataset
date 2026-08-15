"""Focused Exp449 Smoke A execution-capability and claim tests.

Every filesystem target in this module is a temporary private seam.  It never
creates an authority document or touches one of the six configured roots.
"""

from __future__ import annotations

import copy
import errno
import fcntl
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp449_reproducible_rebaseline_antigravity import (
    execution_authority_antigravity as authority,
)
from experiments.exp449_reproducible_rebaseline_antigravity import (
    operations_antigravity as operations,
)
from experiments.exp449_reproducible_rebaseline_antigravity.contracts_antigravity import (
    Exp449Context,
)
from experiments.exp449_reproducible_rebaseline_antigravity.execution_authority_antigravity import (
    VerifiedExp449SmokeAAuthorityAntigravity,
)
from experiments.exp449_reproducible_rebaseline_antigravity.artifacts_antigravity import (
    context_sha256_antigravity,
)


def _context_antigravity(scientific_root: Path) -> Exp449Context:
    """Create a typed fixture identity without widening the production contract."""
    context = object.__new__(Exp449Context)
    for name, value in {
        "experiment_id": "Exp449", "mode": "smoke", "run_root": scientific_root,
        "database_sha256": "a" * 64, "catalog_sha256": "b" * 64,
        "code_closure": {"fixture_antigravity.py": "c" * 64},
        "lineage_sha256": "d" * 64, "codemap_sha256": "e" * 64,
        "interpreter": "fixture-python", "environment": {"fixture": "deterministic"},
        "backend_record": {"selected_backend": "cpu"}, "seeds_and_constants": {"seed": 449},
        "historical_parity_status": "not_used",
    }.items():
        object.__setattr__(context, name, value)
    return context


def _verified_antigravity(scientific_root: Path, operations_root: Path, claim_path: Path, context: Exp449Context):
    """Use retained in-memory facts only; no authority JSON is prepared."""
    verified = object.__new__(VerifiedExp449SmokeAAuthorityAntigravity)
    grant = SimpleNamespace(
        target="smoke-a", scientific_root=scientific_root, operations_root=operations_root,
        claim_path=claim_path, context_sha256=context_sha256_antigravity(context),
    )
    for name, value in {
        "approved_authority_commit": "a" * 40, "implementation_commit": "b" * 40,
        "grant": grant, "grant_sha256": "c" * 64, "grant_bytes": b"fixture grant",
        "digest_bytes": b"fixture digest", "grant_stat_identity": (1, 2, 3, 4),
        "digest_stat_identity": (5, 6, 7, 8),
    }.items():
        object.__setattr__(verified, name, value)
    return verified


class Exp449ExecutionClaimTests(unittest.TestCase):
    """A capability bug must fail before any scientific or operations mutation."""

    def test_execution_surface_exposes_private_capability_and_lease(self) -> None:
        """Removing either the one-shot lease or exact capability gate is a mutation-authority bug."""
        self.assertTrue(hasattr(authority, "_require_execution_capability_antigravity"))
        self.assertTrue(hasattr(operations, "acquire_smoke_a_execution_lease_antigravity"))
        self.assertTrue(hasattr(operations, "freeze_authorized_exp449_proposal_antigravity"))

    def test_lease_mints_noncopyable_nonpicklable_capability_and_consumes_once(self) -> None:
        """A copied or replayed token would let a second process mutate the same root."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            scientific, operational, claim = (base / "scientific_antigravity", base / "operations_antigravity", base / "claim_antigravity.lock")
            context = _context_antigravity(scientific)
            verified = _verified_antigravity(scientific, operational, claim, context)
            with patch.object(operations, "readiness_for_authorized_implementation_antigravity", return_value={"fixture": True}):
                lease = operations.acquire_smoke_a_execution_lease_antigravity(
                    verified, context, _test_paths_antigravity=(scientific, operational, claim),
                )
                receipt = operational / "execution_authority_consumed_antigravity.json"
                self.assertTrue(claim.is_file())
                self.assertTrue(receipt.is_file())
                with self.assertRaises(TypeError):
                    copy.copy(lease.capability)
                with self.assertRaises(TypeError):
                    copy.deepcopy(lease.capability)
                with self.assertRaises(TypeError):
                    pickle.dumps(lease.capability)
                authority._require_execution_capability_antigravity(
                    lease.capability, target="smoke-a", context=context, claim_fd=lease.claim_fd,
                )
                with self.assertRaises(AttributeError):
                    getattr(lease.capability, "grant_sha256")
                with self.assertRaisesRegex(ValueError, "binding"):
                    authority._require_execution_capability_antigravity(
                        lease.capability, target="official", context=context, claim_fd=lease.claim_fd,
                    )
                receipt.write_text("tampered")
                with self.assertRaisesRegex(ValueError, "receipt"):
                    authority._require_execution_capability_antigravity(
                        lease.capability, target="smoke-a", context=context, claim_fd=lease.claim_fd,
                    )
                with self.assertRaisesRegex(ValueError, "already|claim|absent"):
                    operations.acquire_smoke_a_execution_lease_antigravity(
                        verified, context, _test_paths_antigravity=(scientific, operational, claim),
                    )
                lease.close()

    def test_existing_or_symlinked_claim_or_root_is_rejected_before_receipt(self) -> None:
        """Accepting a pre-existing component would make the fixed-path claim replaceable."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            scientific, operational, claim = (base / "scientific_antigravity", base / "operations_antigravity", base / "claim_antigravity.lock")
            context = _context_antigravity(scientific)
            verified = _verified_antigravity(scientific, operational, claim, context)
            claim.write_text("attacker")
            with patch.object(operations, "readiness_for_authorized_implementation_antigravity", return_value={"fixture": True}), \
                 self.assertRaisesRegex(ValueError, "absent|claim"):
                operations.acquire_smoke_a_execution_lease_antigravity(
                    verified, context, _test_paths_antigravity=(scientific, operational, claim),
                )
            self.assertFalse(operational.exists())
            claim.unlink()
            scientific.symlink_to(base / "outside_antigravity")
            with patch.object(operations, "readiness_for_authorized_implementation_antigravity", return_value={"fixture": True}), \
                 self.assertRaisesRegex(ValueError, "unsafe|absent"):
                operations.acquire_smoke_a_execution_lease_antigravity(
                    verified, context, _test_paths_antigravity=(scientific, operational, claim),
                )
            self.assertFalse(operational.exists())

    def test_forked_pid_cannot_reuse_the_inherited_capability(self) -> None:
        """Changing the owner-PID comparison would let a forked child publish to the same claim."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            scientific, operational, claim = (base / "scientific_antigravity", base / "operations_antigravity", base / "claim_antigravity.lock")
            context = _context_antigravity(scientific)
            verified = _verified_antigravity(scientific, operational, claim, context)
            with patch.object(operations, "readiness_for_authorized_implementation_antigravity", return_value={"fixture": True}):
                lease = operations.acquire_smoke_a_execution_lease_antigravity(
                    verified, context, _test_paths_antigravity=(scientific, operational, claim),
                )
                child = os.fork()
                if child == 0:
                    try:
                        authority._require_execution_capability_antigravity(
                            lease.capability, target="smoke-a", context=context, claim_fd=lease.claim_fd,
                        )
                    except ValueError:
                        os._exit(0)
                    os._exit(1)
                _pid, status = os.waitpid(child, 0)
                self.assertEqual(status, 0)
                lease.close()

    def test_unregistered_constructed_capability_cannot_reconstruct_authority(self) -> None:
        """A class instance with copied visible facts is not an issued capability."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            scientific, operational, claim = (base / "scientific_antigravity", base / "operations_antigravity", base / "claim_antigravity.lock")
            context = _context_antigravity(scientific)
            forged = object.__new__(authority._Exp449ExecutionCapabilityAntigravity)
            with self.assertRaisesRegex(ValueError, "capability"):
                authority._require_execution_capability_antigravity(
                    forged, target="smoke-a", context=context, claim_fd=-1,
                )
            self.assertEqual(authority._execution_capability_claim_fd_antigravity(forged), -1)

    def test_released_or_reopened_claim_fd_cannot_reacquire_authority(self) -> None:
        """A validating flock call that reacquires an unlocked FD would mask lock loss."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            scientific, operational, claim = (base / "scientific_antigravity", base / "operations_antigravity", base / "claim_antigravity.lock")
            context = _context_antigravity(scientific)
            verified = _verified_antigravity(scientific, operational, claim, context)
            with patch.object(operations, "readiness_for_authorized_implementation_antigravity", return_value={"fixture": True}):
                lease = operations.acquire_smoke_a_execution_lease_antigravity(
                    verified, context, _test_paths_antigravity=(scientific, operational, claim),
                )
                fcntl.flock(lease.claim_fd, fcntl.LOCK_UN)
                with self.assertRaisesRegex(ValueError, "lock"):
                    authority._require_execution_capability_antigravity(
                        lease.capability, target="smoke-a", context=context, claim_fd=lease.claim_fd,
                    )
                os.close(lease.claim_fd)
                reopened = os.open(claim, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                try:
                    self.assertEqual(reopened, lease.claim_fd)
                    with self.assertRaisesRegex(ValueError, "closed|lock"):
                        authority._require_execution_capability_antigravity(
                            lease.capability, target="smoke-a", context=context, claim_fd=reopened,
                        )
                finally:
                    os.close(reopened)
                    lease.claim_fd = -1

    def test_unexpected_contention_probe_oserror_is_not_a_lock_conflict(self) -> None:
        """A probe error is not evidence that the registered owner still locks the claim."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            claim = Path(temporary) / "claim_antigravity.lock"
            claim.write_text("fixture")
            with patch.object(authority.fcntl, "flock", side_effect=OSError(errno.EINTR, "interrupted")):
                details = claim.stat()
                self.assertFalse(authority._claim_lock_contended_antigravity(claim, expected_device=details.st_dev, expected_inode=details.st_ino))

    def test_only_flock_conflict_counts_as_contention(self) -> None:
        """Probe setup errors must not be mistaken for a live claim lock."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            claim = Path(temporary) / "claim_antigravity.lock"
            claim.write_text("fixture")
            details = claim.stat()
            arguments = {"expected_device": details.st_dev, "expected_inode": details.st_ino}
            with patch.object(authority.os, "fstat", side_effect=OSError(errno.EACCES, "denied")):
                self.assertFalse(authority._claim_lock_contended_antigravity(claim, **arguments))
            with patch.object(authority.Path, "lstat", side_effect=OSError(errno.EAGAIN, "again")):
                self.assertFalse(authority._claim_lock_contended_antigravity(claim, **arguments))
            with patch.object(authority.fcntl, "flock", side_effect=OSError(errno.EAGAIN, "busy")):
                self.assertTrue(authority._claim_lock_contended_antigravity(claim, **arguments))

    def test_distinct_context_claim_replacement_and_traversal_fail_closed(self) -> None:
        """Context drift, inode swaps, and lexical traversal cannot reuse the issued lease."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            scientific, operational, claim = (base / "scientific_antigravity", base / "operations_antigravity", base / "claim_antigravity.lock")
            context = _context_antigravity(scientific)
            verified = _verified_antigravity(scientific, operational, claim, context)
            with patch.object(operations, "readiness_for_authorized_implementation_antigravity", return_value={"fixture": True}):
                lease = operations.acquire_smoke_a_execution_lease_antigravity(verified, context, _test_paths_antigravity=(scientific, operational, claim))
                other = _context_antigravity(scientific)
                object.__setattr__(other, "catalog_sha256", "f" * 64)
                with self.assertRaisesRegex(ValueError, "binding"):
                    authority._require_execution_capability_antigravity(lease.capability, target="smoke-a", context=other, claim_fd=lease.claim_fd)
                claim.unlink()
                claim.write_text("replacement")
                with self.assertRaisesRegex(ValueError, "claim"):
                    authority._require_execution_capability_antigravity(lease.capability, target="smoke-a", context=context, claim_fd=lease.claim_fd)
                lease.close()
            traversal = base / "nested_antigravity" / ".." / "traversal_claim_antigravity.lock"
            with self.assertRaisesRegex(ValueError, "unsafe"):
                operations._lstat_absent_path_antigravity(traversal, name="claim")

    def test_concurrent_acquisition_has_exactly_one_owner(self) -> None:
        """Replacing O_EXCL or the nonblocking lock would permit two mutation owners."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            base = Path(temporary)
            scientific, operational, claim = (base / "scientific_antigravity", base / "operations_antigravity", base / "claim_antigravity.lock")
            context = _context_antigravity(scientific)
            verified = _verified_antigravity(scientific, operational, claim, context)
            original_readiness = operations.readiness_for_authorized_implementation_antigravity
            def acquire():
                try:
                    return operations.acquire_smoke_a_execution_lease_antigravity(
                        verified, context, _test_paths_antigravity=(scientific, operational, claim),
                    )
                except ValueError:
                    return None
            with patch.object(operations, "readiness_for_authorized_implementation_antigravity", return_value={"fixture": True}):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    leases = list(pool.map(lambda _item: acquire(), range(2)))
            owners = [lease for lease in leases if lease is not None]
            self.assertEqual(len(owners), 1)
            owners[0].close()
            self.assertIs(operations.readiness_for_authorized_implementation_antigravity, original_readiness)

    def test_direct_official_mutation_without_capability_fails_before_progress(self) -> None:
        """Dropping the gate would let a public recovery call create a progress log."""
        context = _context_antigravity(operations.FROZEN_EXP449_CONFIG.smoke_a_scientific_root)
        config = operations.Exp449WaveConfig(
            context=context,
            catalog={"catalog_sha256": "b" * 64, "datasets": [{"dataset": name} for name in operations.FROZEN_EXP449_CONFIG.smoke_datasets]},
            raw_datasets={name: object() for name in operations.FROZEN_EXP449_CONFIG.smoke_datasets},
            acceleration_manifest={"fixture": True},
        )
        with self.assertRaisesRegex(ValueError, "capability"):
            operations.run_exp449_wave_antigravity(config)
        self.assertFalse(operations.FROZEN_EXP449_CONFIG.smoke_a_operations_root.exists())


if __name__ == "__main__":
    unittest.main()
