from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp416_exp404_hard_test_confusion_audit import publication
from experiments.exp416_exp404_hard_test_confusion_audit.publication import (
    publish_payloads,
)
from experiments.exp416_exp404_hard_test_confusion_audit.tests.test_reporting import (
    fixture_config_validation,
    hand_config,
    valid_payloads,
)


def valid_provenance() -> dict[str, object]:
    return {
        "mode": "synthetic_fixture",
        "git_commit": "a" * 40,
        "input_sha256": {"database_reference": "4" * 64},
    }


class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = hand_config()

    def publish(
        self,
        output: Path,
        *,
        payloads: dict[str, bytes] | None = None,
        pre_publish_verify: object = None,
    ) -> dict[str, str]:
        selected = valid_payloads(self.config) if payloads is None else payloads
        with fixture_config_validation():
            return dict(
                publish_payloads(
                    selected,
                    output,
                    valid_provenance(),
                    pre_publish_verify=pre_publish_verify,
                    config=self.config,
                )
            )

    def test_callback_mutation_leaves_no_output(self) -> None:
        output = self.root / "publication"

        def mutate() -> None:
            staging = next(self.root.glob(".publication.*.staging"))
            (staging / "exp416_confusion_matrix.json").write_bytes(
                b"changed"
            )

        with self.assertRaises(ValueError):
            self.publish(output, pre_publish_verify=mutate)

        self.assertFalse(output.exists())
        self.assertEqual(tuple(self.root.glob(".publication.*.staging")), ())

    def test_callback_extra_file_leaves_no_output(self) -> None:
        output = self.root / "publication"

        def mutate() -> None:
            staging = next(self.root.glob(".publication.*.staging"))
            (staging / "foreign.bin").write_bytes(b"foreign\n")

        with self.assertRaisesRegex(ValueError, "coverage"):
            self.publish(output, pre_publish_verify=mutate)

        self.assertFalse(output.exists())
        self.assertEqual(tuple(self.root.iterdir()), ())

    def test_callback_staging_file_symlink_leaves_external_target_untouched(
        self,
    ) -> None:
        output = self.root / "publication"
        external = self.root / "external.json"
        external.write_bytes(b"external\n")

        def mutate() -> None:
            staging = next(self.root.glob(".publication.*.staging"))
            target = staging / "exp416_confusion_matrix.json"
            target.unlink()
            target.symlink_to(external)

        with self.assertRaisesRegex(ValueError, "symlink|verification"):
            self.publish(output, pre_publish_verify=mutate)

        self.assertFalse(output.exists())
        self.assertEqual(external.read_bytes(), b"external\n")
        self.assertEqual(tuple(self.root.glob(".publication.*.staging")), ())

    def test_callback_staging_root_symlink_substitution_preserves_external_tree(
        self,
    ) -> None:
        output = self.root / "publication"
        external_tree = self.root / "external_staged_tree"

        def mutate() -> None:
            staging = next(self.root.glob(".publication.*.staging"))
            staging.rename(external_tree)
            staging.symlink_to(external_tree, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "staging root"):
            self.publish(output, pre_publish_verify=mutate)

        self.assertFalse(output.exists())
        self.assertFalse(output.is_symlink())
        self.assertTrue(external_tree.is_dir())
        self.assertTrue(
            (external_tree / "exp416_confusion_matrix.json").is_file()
        )
        self.assertEqual(tuple(self.root.glob(".publication.*.staging")), ())

    def test_output_appearing_during_staging_is_not_clobbered(self) -> None:
        output = self.root / "publication"

        def create_competing_output() -> None:
            output.mkdir()
            (output / "owner.txt").write_text("other", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "appeared"):
            self.publish(
                output,
                pre_publish_verify=create_competing_output,
            )

        self.assertEqual(
            (output / "owner.txt").read_text(encoding="utf-8"),
            "other",
        )
        self.assertEqual(tuple(self.root.glob(".publication.*.staging")), ())

    def test_callback_moving_owned_staging_to_output_is_rolled_back(
        self,
    ) -> None:
        output = self.root / "publication"

        def move_owned_staging() -> None:
            staging = next(self.root.glob(".publication.*.staging"))
            os.replace(staging, output)

        with self.assertRaisesRegex(FileExistsError, "appeared"):
            self.publish(output, pre_publish_verify=move_owned_staging)

        self.assertFalse(output.exists())
        self.assertEqual(tuple(self.root.iterdir()), ())

    def test_foreign_staging_replacement_survives_owned_output_cleanup(
        self,
    ) -> None:
        output = self.root / "publication"
        foreign_staging: list[Path] = []

        def replace_both_publication_paths() -> None:
            staging = next(self.root.glob(".publication.*.staging"))
            os.replace(staging, output)
            staging.mkdir()
            (staging / "foreign.txt").write_text(
                "foreign",
                encoding="utf-8",
            )
            foreign_staging.append(staging)

        with self.assertRaises(FileExistsError):
            self.publish(
                output,
                pre_publish_verify=replace_both_publication_paths,
            )

        self.assertFalse(output.exists())
        self.assertEqual(len(foreign_staging), 1)
        self.assertEqual(
            (foreign_staging[0] / "foreign.txt").read_text(
                encoding="utf-8"
            ),
            "foreign",
        )

    def test_atomic_final_rename_does_not_clobber_postcheck_competitor(
        self,
    ) -> None:
        output = self.root / "publication"
        external = self.root / "foreign_output"
        external.mkdir()
        (external / "foreign.txt").write_text(
            "foreign",
            encoding="utf-8",
        )
        output_checks = 0

        def inject_competitor_after_final_check(path: Path) -> bool:
            nonlocal output_checks
            if path == output:
                output_checks += 1
                if output_checks == 3:
                    output.symlink_to(
                        external,
                        target_is_directory=True,
                    )
                    return False
            return os.path.lexists(path)

        with patch.object(
            publication,
            "_lexists",
            side_effect=inject_competitor_after_final_check,
        ):
            with self.assertRaises(FileExistsError):
                self.publish(output)

        self.assertTrue(output.is_symlink())
        self.assertEqual(output.readlink(), external)
        self.assertEqual(
            (external / "foreign.txt").read_text(encoding="utf-8"),
            "foreign",
        )
        self.assertEqual(tuple(self.root.glob(".publication.*.staging")), ())

    def test_parent_fsync_failure_rolls_back_output(self) -> None:
        real_fsync_directory = publication._fsync_directory
        output = self.root / "publication"

        def fail_parent_fsync(path: Path) -> None:
            if path == self.root:
                raise OSError("parent fsync failed")
            real_fsync_directory(path)

        with patch.object(
            publication,
            "_fsync_directory",
            side_effect=fail_parent_fsync,
        ):
            with self.assertRaisesRegex(OSError, "parent fsync failed"):
                self.publish(output)

        self.assertFalse(output.exists())
        self.assertEqual(tuple(self.root.iterdir()), ())

    def test_existing_output_is_no_clobber(self) -> None:
        output = self.root / "publication"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "must not exist"):
            self.publish(output)

        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertEqual(
            {path.name for path in self.root.iterdir()},
            {"publication"},
        )

    def test_existing_output_symlink_is_no_clobber(self) -> None:
        real_output = self.root / "real"
        real_output.mkdir()
        output = self.root / "publication"
        output.symlink_to(real_output, target_is_directory=True)

        with self.assertRaisesRegex(FileExistsError, "must not exist"):
            self.publish(output)

        self.assertTrue(output.is_symlink())
        self.assertEqual(tuple(real_output.iterdir()), ())

    def test_requires_exact_three_payloads(self) -> None:
        payloads = valid_payloads(self.config)
        for name, invalid in (
            (
                "missing",
                {
                    key: value
                    for key, value in payloads.items()
                    if key != "EXP416_RESULT_REPORT_KO.md"
                },
            ),
            ("extra", {**payloads, "foreign.bin": b"foreign\n"}),
            (
                "manifest",
                {
                    **payloads,
                    "MANIFEST.json": b"{}\n",
                },
            ),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                "three non-manifest|exact",
            ):
                self.publish(self.root / name, payloads=invalid)
            self.assertFalse((self.root / name).exists())

    def test_success_publishes_exact_four_files_and_manifest_hashes(
        self,
    ) -> None:
        output = self.root / "publication"
        payloads = valid_payloads(self.config)

        hashes = self.publish(output, payloads=payloads)

        self.assertEqual(
            {path.name for path in output.iterdir()},
            set(payloads) | {"MANIFEST.json"},
        )
        self.assertTrue(
            all(not path.is_symlink() for path in output.iterdir())
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
        self.assertEqual(
            set(manifest),
            {"experiment_id", "artifact_sha256", "provenance"},
        )
        self.assertEqual(manifest["experiment_id"], "Exp416")
        self.assertEqual(manifest["artifact_sha256"], hashes)
        self.assertEqual(manifest["provenance"], valid_provenance())
        self.assertEqual(
            os.stat(output, follow_symlinks=False).st_nlink >= 1,
            True,
        )


if __name__ == "__main__":
    unittest.main()
