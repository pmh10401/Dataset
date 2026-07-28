from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp414_missing_aware_near_repeat.publication import (
    publish_payloads,
)


def _payloads() -> dict[str, bytes]:
    return {
        f"artifact_{index}.bin": f"payload-{index}\n".encode("utf-8")
        for index in range(8)
    }


def _provenance() -> dict[str, object]:
    return {
        "mode": "fixture",
        "git_commit": "a" * 40,
        "contracts": {"frozen_config_sha256": "b" * 64},
    }


class PublicationTests(unittest.TestCase):
    def test_requires_exactly_eight_nonmanifest_payloads(self) -> None:
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            for name, payloads in (
                ("seven", dict(list(_payloads().items())[:7])),
                (
                    "nine",
                    {**_payloads(), "artifact_8.bin": b"payload-8\n"},
                ),
                (
                    "manifest",
                    {**dict(list(_payloads().items())[:7]), "MANIFEST.json": b"{}"},
                ),
            ):
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError,
                    "eight non-manifest",
                ):
                    publish_payloads(
                        payloads,
                        parent / name,
                        _provenance(),
                    )
                self.assertFalse((parent / name).exists())
            self.assertEqual(tuple(parent.iterdir()), ())

    def test_existing_output_is_rejected_without_modification(self) -> None:
        with TemporaryDirectory() as raw:
            output = Path(raw) / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "must not exist"):
                publish_payloads(_payloads(), output, _provenance())

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_staged_hash_mismatch_leaves_no_output_or_staging(self) -> None:
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            output = parent / "publication"
            with patch(
                "experiments.exp414_missing_aware_near_repeat.publication."
                "sha256_file",
                return_value="0" * 64,
            ):
                with self.assertRaisesRegex(ValueError, "verification failed"):
                    publish_payloads(_payloads(), output, _provenance())

            self.assertFalse(output.exists())
            self.assertEqual(tuple(parent.iterdir()), ())

    def test_pre_publish_failure_leaves_no_output_or_staging(self) -> None:
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            output = parent / "publication"

            def reject_changed_state() -> None:
                raise ValueError("execution state changed")

            with self.assertRaisesRegex(ValueError, "execution state changed"):
                publish_payloads(
                    _payloads(),
                    output,
                    _provenance(),
                    pre_publish_verify=reject_changed_state,
                )

            self.assertFalse(output.exists())
            self.assertEqual(tuple(parent.iterdir()), ())

    def test_success_publishes_nine_files_and_manifest_hashes_payloads(
        self,
    ) -> None:
        with TemporaryDirectory() as raw:
            output = Path(raw) / "publication"
            payloads = _payloads()

            hashes = publish_payloads(payloads, output, _provenance())

            self.assertEqual(
                {path.name for path in output.iterdir()},
                set(payloads) | {"MANIFEST.json"},
            )
            self.assertEqual(
                hashes,
                {
                    name: sha256(payload).hexdigest()
                    for name, payload in sorted(payloads.items())
                },
            )
            manifest = json.loads(
                (output / "MANIFEST.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["experiment_id"], "Exp414")
            self.assertEqual(manifest["artifact_sha256"], hashes)
            self.assertEqual(manifest["provenance"], _provenance())


if __name__ == "__main__":
    unittest.main()
