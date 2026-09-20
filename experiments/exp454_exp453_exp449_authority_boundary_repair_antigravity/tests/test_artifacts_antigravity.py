from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.artifacts_antigravity as artifacts_module
from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.artifacts_antigravity import (
    artifact_map_bytes_antigravity,
    canonical_json_bytes_antigravity,
    canonical_jsonl_bytes_antigravity,
    strict_json_bytes_antigravity,
    write_create_only_antigravity,
)


class ArtifactsAntigravityTest(unittest.TestCase):
    def test_canonical_json_and_jsonl_are_exact_antigravity(self) -> None:
        """Break caught: accepting alternate bytes, duplicate keys, or NaN."""
        self.assertEqual(canonical_json_bytes_antigravity({"b": 1, "a": "x"}), b'{"a":"x","b":1}\n')
        self.assertEqual(canonical_jsonl_bytes_antigravity(({"b": 1}, {"a": 2})), b'{"b":1}\n{"a":2}\n')
        self.assertEqual(strict_json_bytes_antigravity(b'{"a":1}\n'), {"a": 1})
        with self.assertRaisesRegex(ValueError, "duplicate JSON key: a"):
            strict_json_bytes_antigravity(b'{"a":1,"a":2}\n')
        with self.assertRaisesRegex(ValueError, "non-canonical JSON bytes"):
            strict_json_bytes_antigravity(b'{"a": 1}\n')
        with self.assertRaisesRegex(ValueError, "value is not canonical JSON"):
            canonical_json_bytes_antigravity({"value": float("nan")})
        with self.assertRaisesRegex(ValueError, "iterable of JSON values"):
            canonical_jsonl_bytes_antigravity("not-records")

    def test_create_only_rejects_existing_and_symlink_destinations_antigravity(self) -> None:
        """Break caught: overwriting an authority leaf or following a symlink."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target_antigravity.json"
            write_create_only_antigravity(target, b"x")
            self.assertEqual(target.read_bytes(), b"x")
            with self.assertRaisesRegex(FileExistsError, "destination already exists"):
                write_create_only_antigravity(target, b"replacement")
            link = root / "link_antigravity.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                write_create_only_antigravity(link, b"x")

    def test_artifact_map_rejects_extra_leaf_and_is_deterministic_antigravity(self) -> None:
        """Break caught: hashing a partial or namespace-dependent inventory."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a_antigravity.json").write_bytes(b"a")
            expected = {"a_antigravity.json": hashlib.sha256(b"a").hexdigest()}
            payload = artifact_map_bytes_antigravity(root, ("a_antigravity.json",))
            self.assertEqual(strict_json_bytes_antigravity(payload), expected)
            (root / "extra_antigravity.json").write_bytes(b"b")
            with self.assertRaisesRegex(ValueError, "unexpected artifact leaf"):
                artifact_map_bytes_antigravity(root, ("a_antigravity.json",))

    def test_artifact_map_rejects_root_replacement_after_inventory_antigravity(self) -> None:
        """Break caught: reading a replacement root after validating another inode."""
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "root_antigravity"
            root.mkdir()
            (root / "a_antigravity.json").write_bytes(b"a")

            def replace_root() -> None:
                root.rename(parent / "moved_antigravity")
                root.mkdir()
                (root / "a_antigravity.json").write_bytes(b"replacement")

            with patch.object(artifacts_module, "_after_inventory_antigravity", replace_root, create=True):
                with self.assertRaisesRegex(ValueError, "root namespace changed"):
                    artifact_map_bytes_antigravity(root, ("a_antigravity.json",))

    def test_artifact_map_rejects_symlink_swap_after_inventory_antigravity(self) -> None:
        """Break caught: following a leaf rebound to content outside the root."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root_antigravity"
            root.mkdir()
            leaf = root / "a_antigravity.json"
            leaf.write_bytes(b"a")
            outside = Path(temporary) / "outside_antigravity.json"
            outside.write_bytes(b"outside")

            def swap_leaf() -> None:
                leaf.unlink()
                leaf.symlink_to(outside)

            with patch.object(artifacts_module, "_after_inventory_antigravity", swap_leaf, create=True):
                with self.assertRaisesRegex(ValueError, "artifact namespace changed"):
                    artifact_map_bytes_antigravity(root, ("a_antigravity.json",))

    def test_artifact_map_rejects_depth_two_directory_rebind_during_read_antigravity(self) -> None:
        """Break caught: trusting a nested directory name after its descriptor was pinned."""
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "root_antigravity"
            nested = root / "first_antigravity" / "second_antigravity"
            nested.mkdir(parents=True)
            leaf = nested / "a_antigravity.json"
            leaf.write_bytes(b"pinned")

            def rebind_nested(relative: str) -> None:
                self.assertEqual(relative, "first_antigravity/second_antigravity/a_antigravity.json")
                nested.rename(parent / "detached_antigravity")
                nested.mkdir()
                (nested / "a_antigravity.json").write_bytes(b"replacement")

            with patch.object(artifacts_module, "_before_leaf_read_antigravity", rebind_nested, create=True):
                with self.assertRaisesRegex(ValueError, "artifact namespace changed"):
                    artifact_map_bytes_antigravity(root, ("first_antigravity/second_antigravity/a_antigravity.json",))


if __name__ == "__main__":
    unittest.main()
