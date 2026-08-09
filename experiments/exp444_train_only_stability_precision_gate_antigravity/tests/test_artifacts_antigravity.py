"""RED/GREEN tests for the narrow Exp441 artifact wrappers."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.exp444_train_only_stability_precision_gate_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    canonical_jsonl_bytes_antigravity,
    parse_canonical_json_bytes_antigravity,
    read_pinned_regular_bytes_antigravity,
    write_create_only_antigravity,
)


class Exp444ArtifactsAntigravityTests(unittest.TestCase):
    def test_canonical_json_and_jsonl_wrappers_round_trip(self) -> None:
        payload = {"b": 2, "a": [1, True]}
        encoded = canonical_json_bytes_antigravity(payload)
        self.assertEqual(parse_canonical_json_bytes_antigravity(encoded), payload)
        self.assertEqual(canonical_jsonl_bytes_antigravity((payload, {"a": 1})), encoded + canonical_json_bytes_antigravity({"a": 1}))
        with self.assertRaises(ValueError):
            parse_canonical_json_bytes_antigravity(b'{"a":1}')

    def test_create_only_write_refuses_replacement_and_pinned_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_artifacts_antigravity_") as root_name:
            root = Path(root_name)
            target = root / "artifact_antigravity.json"
            write_create_only_antigravity(target, b"hello")
            self.assertEqual(read_pinned_regular_bytes_antigravity(target)[0], b"hello")
            with self.assertRaises(FileExistsError):
                write_create_only_antigravity(target, b"replacement")


if __name__ == "__main__":
    unittest.main()
