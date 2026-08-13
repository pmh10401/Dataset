import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from experiments.exp451_rdst_local_gap_qualification_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity, read_canonical_file_antigravity,
    write_atomic_antigravity, write_create_only_antigravity,
)


class ArtifactsTestsAntigravity(unittest.TestCase):
    def test_canonical_and_create_only_reject_overwrite_and_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); destination = root / "leaf_antigravity.json"
            payload = canonical_json_bytes_antigravity({"x": 1})
            write_create_only_antigravity(destination, payload)
            self.assertEqual(read_canonical_file_antigravity(destination), {"x": 1})
            with self.assertRaises(FileExistsError): write_create_only_antigravity(destination, payload)
            os.link(destination, root / "hard_antigravity.json")
            with self.assertRaises(ValueError): read_canonical_file_antigravity(destination)

    def test_atomic_replace_leaves_canonical_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "progress_antigravity.json"
            write_atomic_antigravity(destination, canonical_json_bytes_antigravity({"completed": 1}))
            write_atomic_antigravity(destination, canonical_json_bytes_antigravity({"completed": 2}))
            self.assertEqual(read_canonical_file_antigravity(destination), {"completed": 2})

    def test_short_write_loops_preserve_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "leaf_antigravity.json"
            payload = canonical_json_bytes_antigravity({"payload": "x" * 200})
            real_write = os.write
            def short_write(fd, value):
                return real_write(fd, bytes(value[:max(1, len(value) // 3)]))
            with mock.patch("experiments.exp451_rdst_local_gap_qualification_antigravity.artifacts_antigravity.os.write", side_effect=short_write):
                write_create_only_antigravity(destination, payload)
            self.assertEqual(destination.read_bytes(), payload)

    def test_zero_write_and_directory_fsync_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "leaf_antigravity.json"
            payload = canonical_json_bytes_antigravity({"x": 1})
            with mock.patch("experiments.exp451_rdst_local_gap_qualification_antigravity.artifacts_antigravity.os.write", return_value=0):
                with self.assertRaises(OSError): write_create_only_antigravity(destination, payload)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "leaf_antigravity.json"
            payload = canonical_json_bytes_antigravity({"x": 1})
            real_fsync = os.fsync
            def fail_directory(fd):
                if os.path.isdir(f"/dev/fd/{fd}"):
                    raise OSError("directory fsync")
                return real_fsync(fd)
            with mock.patch("experiments.exp451_rdst_local_gap_qualification_antigravity.artifacts_antigravity.os.fsync", side_effect=fail_directory):
                with self.assertRaises(OSError): write_create_only_antigravity(destination, payload)


if __name__ == "__main__": unittest.main()
