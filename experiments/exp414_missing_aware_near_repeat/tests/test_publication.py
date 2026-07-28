from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp414_missing_aware_near_repeat import publication
from experiments.exp414_missing_aware_near_repeat.publication import (
    publish_payloads,
)


def _payloads() -> dict[str, bytes]:
    return {
        "exp414_frozen_config.json": b'{"experiment_id":"Exp414"}\n',
        "exp414_missing_repair_audit.csv": b"dataset,instance_index\n",
        "exp414_dataset_qualification.csv": b"dataset,execution_source\n",
        "exp414_quantile_summary.csv": b"subgroup,quantile\n",
        "exp414_backend_assessment.json": b'{"backend":"fixture"}\n',
        "exp414_selected_rule.json": b'{"status":"qualified"}\n',
        "exp414_summary.json": b'{"test_values_used":false}\n',
        "EXP414_RESULT_REPORT_KO.md": b"# Exp414 fixture\n",
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
            items = tuple(_payloads().items())
            for name, payloads in (
                ("seven", dict(items[:7])),
                (
                    "nine",
                    {**_payloads(), "foreign.bin": b"foreign\n"},
                ),
                (
                    "manifest",
                    {**dict(items[:7]), "MANIFEST.json": b"{}"},
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

    def test_rejects_same_count_payloads_with_a_wrong_artifact_name(
        self,
    ) -> None:
        payloads = _payloads()
        payloads.pop("exp414_summary.json")
        payloads["foreign_summary.json"] = b"{}\n"
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            output = parent / "publication"

            with self.assertRaisesRegex(ValueError, "exact eight"):
                publish_payloads(payloads, output, _provenance())

            self.assertFalse(output.exists())
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

    def test_callback_staging_mutation_is_rejected_before_rename(self) -> None:
        for mutation in ("payload", "manifest", "inventory"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as raw:
                parent = Path(raw)
                output = parent / "publication"

                def mutate_staging() -> None:
                    staging = next(parent.glob(".*.staging"))
                    if mutation == "payload":
                        (
                            staging / "exp414_summary.json"
                        ).write_bytes(b'{"tampered":true}\n')
                    elif mutation == "manifest":
                        (
                            staging / "MANIFEST.json"
                        ).write_bytes(b'{"tampered":true}\n')
                    else:
                        (staging / "foreign.bin").write_bytes(b"foreign\n")

                with self.assertRaisesRegex(
                    ValueError,
                    "verification|coverage",
                ):
                    publish_payloads(
                        _payloads(),
                        output,
                        _provenance(),
                        pre_publish_verify=mutate_staging,
                    )

                self.assertFalse(output.exists())
                self.assertEqual(tuple(parent.iterdir()), ())

    def test_parent_fsync_failure_rolls_back_published_output(self) -> None:
        real_fsync_directory = publication._fsync_directory
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            output = parent / "publication"

            def fail_parent_fsync(path: Path) -> None:
                if path == parent:
                    raise OSError("parent fsync failed")
                real_fsync_directory(path)

            with patch(
                "experiments.exp414_missing_aware_near_repeat.publication."
                "_fsync_directory",
                side_effect=fail_parent_fsync,
            ):
                with self.assertRaisesRegex(OSError, "parent fsync failed"):
                    publish_payloads(_payloads(), output, _provenance())

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
