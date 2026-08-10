"""TDD coverage for descriptor-pinned Exp444 provenance authorities."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import os
import tempfile
import unittest

from experiments.exp444_train_only_stability_precision_gate_antigravity import provenance_antigravity as provenance


class Exp444ProvenanceAntigravityTests(unittest.TestCase):
    def test_pinned_reader_rejects_hardlinked_authority_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_provenance_antigravity_") as directory:
            root = Path(directory)
            authority = root / "authority_antigravity.json"
            authority.write_bytes(b"pinned\n")
            os.link(authority, root / "replacement_antigravity.json")
            with self.assertRaisesRegex(ValueError, "single-linked"):
                provenance._read_pinned_antigravity(root, authority.name, sha256(b"pinned\n").hexdigest())

    def test_real_exp440_and_task5_authorities_verify_antigravity(self) -> None:
        checked = provenance.verify_exp444_provenance_antigravity()
        self.assertEqual((len(checked.dataset_summary_bytes.splitlines()), checked.backend_decision.backend, checked.backend_decision.workers), (1117, "cpu", 11))


if __name__ == "__main__":
    unittest.main()
