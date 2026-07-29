"""Filesystem attack tests for Exp417's atomic eight-file publication."""

from __future__ import annotations

import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp417_causal_multihorizon_shadow.tests.test_reporting import (
    fixture_trusted_checkpoint,
    rendered_payloads,
)


class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payloads = rendered_payloads()

    def publish(self, output: Path, callback=None) -> dict[str, str]:
        from experiments.exp417_causal_multihorizon_shadow.publication import (
            publish_exp417,
        )
        from experiments.exp417_causal_multihorizon_shadow.reporting import (
            TrustedCheckpointProvenance,
        )

        checkpoint_store = object()
        with patch.object(
            TrustedCheckpointProvenance,
            "from_store",
            return_value=fixture_trusted_checkpoint(),
        ):
            return dict(
                publish_exp417(
                    self.payloads,
                    output,
                    checkpoints=checkpoint_store,
                    pre_publish_verify=(lambda: None) if callback is None else callback,
                )
            )

    def test_success_publishes_exact_eight_files_and_bound_manifest(self) -> None:
        output = self.root / "publication"

        hashes = self.publish(output)

        self.assertEqual(
            {path.name for path in output.iterdir()},
            set(self.payloads) | {"MANIFEST.json"},
        )
        self.assertEqual(
            hashes,
            {
                name: sha256(payload).hexdigest()
                for name, payload in sorted(self.payloads.items())
            },
        )
        manifest = json.loads((output / "MANIFEST.json").read_bytes())
        self.assertEqual(manifest["experiment_id"], "Exp417")
        self.assertEqual(manifest["artifact_sha256"], hashes)
        self.assertEqual(
            manifest["frozen_config_sha256"],
            json.loads(self.payloads["exp417_frozen_config.json"])[
                "frozen_config_sha256"
            ],
        )
        self.assertEqual(
            manifest["report_sha256"],
            sha256(self.payloads["EXP417_RESULT_REPORT_KO.md"]).hexdigest(),
        )
        self.assertTrue(all(not path.is_symlink() for path in output.iterdir()))

    def test_filesystem_without_atomic_exchange_fails_before_staging(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow import publication

        output = self.root / "publication"
        callback_called = False

        def callback() -> None:
            nonlocal callback_called
            callback_called = True

        unsupported = OSError(
            errno.ENOTSUP, "filesystem does not support atomic exchange"
        )
        with patch.object(
            publication, "_atomic_exchange", side_effect=unsupported
        ):
            with self.assertRaises(OSError) as caught:
                self.publish(output, callback)

        self.assertEqual(caught.exception.errno, errno.ENOTSUP)
        self.assertFalse(callback_called)
        self.assertFalse(os.path.lexists(output))
        capsules = tuple(
            self.root.glob(".exp417.exchange-probe.retained.*")
        )
        self.assertEqual(len(capsules), 1)
        capsule = capsules[0]
        self.assertEqual(set(self.root.iterdir()), {capsule})
        self.assertEqual(stat.S_IMODE(capsule.lstat().st_mode), 0o700)
        self.assertEqual(
            {path.name for path in capsule.iterdir()},
            {"a", "b", "PROBE_PROVENANCE.json"},
        )
        provenance = json.loads(
            (capsule / "PROBE_PROVENANCE.json").read_bytes()
        )
        self.assertFalse(provenance["atomic_exchange_supported"])
        self.assertEqual(provenance["operation_errno"], errno.ENOTSUP)
        self.assertFalse(provenance["automatic_cleanup"])
        capsule_metadata = capsule.lstat()
        self.assertEqual(
            tuple(provenance["capsule_identity"]),
            (capsule_metadata.st_dev, capsule_metadata.st_ino),
        )
        self.assertEqual(tuple(self.root.glob(".publication.*.staging")), ())
        self.assertEqual(tuple(self.root.glob(".publication.*.capture")), ())

    def test_capability_probe_swaps_and_restores_real_inodes(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow import publication

        real_exchange = publication._atomic_exchange
        real_fsync = publication._fsync_directory
        parent_fsyncs = 0
        exchanges: list[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
        ] = []

        def observe(first: Path, second: Path) -> None:
            before_first = first.lstat()
            before_second = second.lstat()
            real_exchange(first, second)
            after_first = first.lstat()
            after_second = second.lstat()
            exchanges.append(
                (
                    (before_first.st_dev, before_first.st_ino),
                    (before_second.st_dev, before_second.st_ino),
                    (after_first.st_dev, after_first.st_ino),
                    (after_second.st_dev, after_second.st_ino),
                )
            )

        def observe_fsync(path: Path) -> None:
            nonlocal parent_fsyncs
            if path == self.root:
                parent_fsyncs += 1
            real_fsync(path)

        with (
            patch.object(publication, "_atomic_exchange", side_effect=observe),
            patch.object(
                publication, "_fsync_directory", side_effect=observe_fsync
            ),
        ):
            publication._require_atomic_exchange(self.root)

        self.assertEqual(len(exchanges), 2)
        first_before_a, first_before_b, first_after_a, first_after_b = exchanges[0]
        self.assertEqual(first_after_a, first_before_b)
        self.assertEqual(first_after_b, first_before_a)
        second_before_a, second_before_b, second_after_a, second_after_b = exchanges[1]
        self.assertEqual(second_before_a, first_after_a)
        self.assertEqual(second_before_b, first_after_b)
        self.assertEqual(second_after_a, first_before_a)
        self.assertEqual(second_after_b, first_before_b)
        self.assertGreaterEqual(parent_fsyncs, 2)
        capsules = tuple(
            self.root.glob(".exp417.exchange-probe.retained.*")
        )
        self.assertEqual(len(capsules), 1)
        self.assertEqual(set(self.root.iterdir()), {capsules[0]})
        provenance = json.loads(
            (capsules[0] / "PROBE_PROVENANCE.json").read_bytes()
        )
        self.assertTrue(provenance["atomic_exchange_supported"])
        self.assertIsNone(provenance["operation_errno"])
        self.assertFalse(provenance["automatic_cleanup"])
        self.assertEqual(tuple(self.root.glob(".publication.*.staging")), ())
        self.assertEqual(tuple(self.root.glob(".publication.*.capture")), ())

    def test_probe_identity_race_never_deletes_foreign_or_owned_entry(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow import publication

        real_identity = publication._path_identity
        real_rmdir = publication.os.rmdir
        real_unlink = publication.os.unlink
        a_identity_checks = 0
        displaced_owned: Path | None = None
        probe_deletions: list[tuple[str, object]] = []

        def inject_after_identity_check(path: Path) -> tuple[int, int]:
            nonlocal a_identity_checks, displaced_owned
            identity = real_identity(path)
            if path.name == "a" and path.parent.name.startswith(".exp417"):
                a_identity_checks += 1
                if a_identity_checks == 3:
                    displaced_owned = path.with_name("a.owned-displaced")
                    path.rename(displaced_owned)
                    path.mkdir(mode=0o700)
                    (path / "foreign.txt").write_bytes(b"foreign a\n")
            return identity

        def observe_rmdir(
            path: object, *args: object, **kwargs: object
        ) -> None:
            if path in {"a", "b"} and type(kwargs.get("dir_fd")) is int:
                probe_deletions.append(("rmdir", path))
            real_rmdir(path, *args, **kwargs)

        def observe_unlink(
            path: object, *args: object, **kwargs: object
        ) -> None:
            if ".exp417" in os.fspath(path):
                probe_deletions.append(("unlink", path))
            real_unlink(path, *args, **kwargs)

        probe_error: Exception | None = None
        with (
            patch.object(
                publication,
                "_path_identity",
                side_effect=inject_after_identity_check,
            ),
            patch.object(publication.os, "rmdir", side_effect=observe_rmdir),
            patch.object(publication.os, "unlink", side_effect=observe_unlink),
        ):
            try:
                publication._require_atomic_exchange(self.root)
            except Exception as error:
                probe_error = error

        self.assertIsNone(probe_error)
        self.assertEqual(probe_deletions, [])
        self.assertIsNotNone(displaced_owned)
        assert displaced_owned is not None
        self.assertTrue(displaced_owned.is_dir())
        self.assertEqual(
            (displaced_owned.parent / "a" / "foreign.txt").read_bytes(),
            b"foreign a\n",
        )
        self.assertTrue((displaced_owned.parent / "b").is_dir())
        self.assertTrue(
            (displaced_owned.parent / "PROBE_PROVENANCE.json").is_file()
        )
        provenance = json.loads(
            (displaced_owned.parent / "PROBE_PROVENANCE.json").read_bytes()
        )
        displaced_metadata = displaced_owned.lstat()
        self.assertEqual(
            tuple(provenance["probe_inode_provenance"]["a"]),
            (displaced_metadata.st_dev, displaced_metadata.st_ino),
        )
        self.assertFalse(provenance["automatic_cleanup"])
        self.assertEqual(tuple(self.root.glob(".publication.*.staging")), ())
        self.assertEqual(tuple(self.root.glob(".publication.*.capture")), ())

    def test_mutation_extra_file_and_symlink_are_fail_closed(self) -> None:
        attacks = {}

        def mutation(staging: Path) -> None:
            (staging / "exp417_backend_benchmark.json").write_bytes(b"changed\n")

        attacks["mutation"] = mutation

        def extra(staging: Path) -> None:
            (staging / "foreign.bin").write_bytes(b"foreign\n")

        attacks["extra"] = extra

        external = self.root / "external.json"
        external.write_bytes(b"external\n")

        def symlink(staging: Path) -> None:
            target = staging / "exp417_backend_benchmark.json"
            target.unlink()
            target.symlink_to(external)

        attacks["symlink"] = symlink

        for name, attack in attacks.items():
            output = self.root / f"publication-{name}"

            def callback(attack=attack, output=output) -> None:
                staging = next(self.root.glob(f".{output.name}.*.staging"))
                attack(staging)

            with self.subTest(name=name), self.assertRaises(ValueError):
                self.publish(output, callback)
            self.assertFalse(os.path.lexists(output))
        self.assertEqual(external.read_bytes(), b"external\n")

    def test_root_substitution_preserves_foreign_tree(self) -> None:
        output = self.root / "publication"
        displaced = self.root / "displaced-owned"

        def substitute() -> None:
            staging = next(self.root.glob(".publication.*.staging"))
            staging.rename(displaced)
            staging.mkdir()
            (staging / "foreign.txt").write_text("foreign", encoding="utf-8")

        with self.assertRaises(ExceptionGroup):
            self.publish(output, substitute)

        self.assertFalse(output.exists())
        self.assertEqual(
            next(self.root.glob(".publication.*.staging/foreign.txt")).read_text(
                encoding="utf-8"
            ),
            "foreign",
        )
        self.assertTrue(
            (displaced / "exp417_backend_benchmark.json").is_file()
        )

    def test_staging_root_symlink_substitution_leaves_link_and_owned_tree_untouched(self) -> None:
        output = self.root / "publication"
        owned_tree = self.root / "owned-tree"

        def substitute() -> None:
            staging = next(self.root.glob(".publication.*.staging"))
            staging.rename(owned_tree)
            staging.symlink_to(owned_tree, target_is_directory=True)

        with self.assertRaises(ExceptionGroup):
            self.publish(output, substitute)

        self.assertFalse(output.exists())
        links = tuple(self.root.glob(".publication.*.staging"))
        self.assertEqual(len(links), 1)
        self.assertTrue(links[0].is_symlink())
        self.assertEqual(links[0].resolve(), owned_tree.resolve())
        self.assertTrue((owned_tree / "exp417_backend_benchmark.json").is_file())

    def test_output_race_and_existing_output_are_never_replaced(self) -> None:
        for name, create_during_staging in (("existing", False), ("race", True)):
            output = self.root / name
            if not create_during_staging:
                output.mkdir()
                (output / "owner.txt").write_text("owner", encoding="utf-8")

            def callback(output=output) -> None:
                output.mkdir()
                (output / "owner.txt").write_text("owner", encoding="utf-8")

            if create_during_staging:
                expected = self.assertRaises(ExceptionGroup)
            else:
                expected = self.assertRaises(FileExistsError)
            with self.subTest(name=name), expected:
                self.publish(output, callback if create_during_staging else None)
            self.assertEqual(
                (output / "owner.txt").read_text(encoding="utf-8"),
                "owner",
            )

    def test_post_rename_content_and_inode_changes_are_rolled_back(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow import publication

        for name, replace_inode in (("content", False), ("inode", True)):
            output = self.root / name
            real_rename = publication._atomic_rename_noreplace

            def mutate_after_rename(source: Path, destination: Path) -> None:
                real_rename(source, destination)
                if destination != output:
                    return
                target = destination / "exp417_backend_benchmark.json"
                if replace_inode:
                    target.unlink()
                    target.write_bytes(self.payloads[target.name])
                else:
                    target.write_bytes(b"changed\n")

            with patch.object(
                publication,
                "_atomic_rename_noreplace",
                side_effect=mutate_after_rename,
            ):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.publish(output)
            self.assertFalse(os.path.lexists(output))

    def test_atomic_no_replace_preserves_postcheck_competitor(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow import publication

        output = self.root / "publication"
        real_rename = publication._atomic_rename_noreplace
        injected = False

        def race(source: Path, destination: Path) -> None:
            nonlocal injected
            if destination == output and not injected:
                injected = True
                destination.mkdir()
                (destination / "owner.txt").write_text("foreign", encoding="utf-8")
            real_rename(source, destination)

        with patch.object(
            publication, "_atomic_rename_noreplace", side_effect=race
        ):
            with self.assertRaises(ExceptionGroup):
                self.publish(output)

        self.assertTrue(injected)
        self.assertEqual(
            (output / "owner.txt").read_text(encoding="utf-8"), "foreign"
        )
        self.assertEqual(tuple(self.root.glob(".publication.*.staging")), ())

    def test_descriptor_preopen_substitution_is_detected(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow import publication

        output = self.root / "publication"
        real_stat = os.stat
        substitutions = 0

        def substitute(path: object, *args: object, **kwargs: object):
            nonlocal substitutions
            metadata = real_stat(path, *args, **kwargs)
            if (
                path == "exp417_backend_benchmark.json"
                and type(kwargs.get("dir_fd")) is int
                and kwargs.get("follow_symlinks") is False
                and substitutions == 0
            ):
                substitutions += 1
                staging = next(self.root.glob(".publication.*.staging"))
                target = staging / path
                target.unlink()
                target.write_bytes(self.payloads[path])
            return metadata

        with patch.object(publication.os, "stat", side_effect=substitute):
            with self.assertRaisesRegex(ValueError, "inode|identity"):
                self.publish(output)
        self.assertEqual(substitutions, 1)
        self.assertFalse(output.exists())

    def test_parent_fsync_failure_rolls_back_and_cleanup_failure_is_surfaced(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow import publication

        output = self.root / "publication"
        real_remove = publication._remove_owned_path

        def cleanup_denied(path: Path, identity):
            removed = real_remove(path, identity)
            if removed:
                raise PermissionError("cleanup denied")
            return removed

        def fail() -> None:
            raise RuntimeError("pre-publish verification failed")

        with patch.object(
            publication, "_remove_owned_path", side_effect=cleanup_denied
        ):
            with self.assertRaises(ExceptionGroup) as caught:
                self.publish(output, fail)

        messages = tuple(str(error) for error in caught.exception.exceptions)
        self.assertTrue(any("pre-publish verification failed" in text for text in messages))
        self.assertTrue(any("cleanup denied" in text for text in messages))
        self.assertFalse(output.exists())

    def test_cleanup_never_unlinks_foreign_replacement_after_quarantine_race(self) -> None:
        """A file swapped after root quarantine must survive cleanup unchanged."""
        from experiments.exp417_causal_multihorizon_shadow import publication

        output = self.root / "publication"
        real_rename = publication._atomic_rename_noreplace
        injected_paths: list[Path] = []

        def inject_before_placeholder_release(
            source: Path, destination: Path
        ) -> None:
            if destination.name.endswith(".released") and not injected_paths:
                displaced = source.with_name(f"{source.name}.placeholder-away")
                source.rename(displaced)
                source.mkdir()
                target = source / "foreign.txt"
                target.write_bytes(b"foreign replacement\n")
                injected_paths.append(target)
            real_rename(source, destination)

        def fail() -> None:
            staging = next(self.root.glob(".publication.*.staging"))
            (staging / "exp417_backend_benchmark.json").write_bytes(b"trigger\n")

        with patch.object(
            publication,
            "_atomic_rename_noreplace",
            side_effect=inject_before_placeholder_release,
        ):
            with self.assertRaises(Exception):
                self.publish(output, fail)

        self.assertEqual(len(injected_paths), 1)
        self.assertTrue(injected_paths[0].exists())
        self.assertEqual(injected_paths[0].read_bytes(), b"foreign replacement\n")

    def test_cleanup_atomic_capture_restores_foreign_a_and_preserves_racing_b(
        self,
    ) -> None:
        """Foreign A is restored and racing foreign B is retained in capture."""
        from experiments.exp417_causal_multihorizon_shadow import publication

        output = self.root / "publication"
        owned = self.root / "owned-displaced"
        foreign_b = self.root / "foreign-b"
        foreign_b.mkdir()
        (foreign_b / "owner.txt").write_bytes(b"foreign B\n")
        real_exchange = publication._atomic_exchange
        output_exchanges = 0

        def observe(first: Path, second: Path) -> None:
            nonlocal output_exchanges
            if first == output:
                output_exchanges += 1
            if first == output and output_exchanges == 2:
                # After atomic capture of foreign A, replace the owned
                # placeholder with foreign B immediately before restoration.
                placeholder_away = self.root / "placeholder-away"
                output.rename(placeholder_away)
                foreign_b.rename(output)
            real_exchange(first, second)

        def substitute_with_foreign() -> None:
            staging = next(self.root.glob(".publication.*.staging"))
            staging.rename(owned)
            output.mkdir()
            (output / "owner.txt").write_bytes(b"foreign A\n")

        with patch.object(
            publication, "_atomic_exchange", side_effect=observe
        ):
            with self.assertRaises(ExceptionGroup):
                self.publish(output, substitute_with_foreign)

        self.assertEqual(output_exchanges, 2)
        self.assertEqual((output / "owner.txt").read_bytes(), b"foreign A\n")
        captures = tuple(self.root.glob(".publication.*.capture/owner.txt"))
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0].read_bytes(), b"foreign B\n")
        self.assertTrue((owned / "exp417_backend_benchmark.json").is_file())

    def test_parent_is_fsynced_again_after_rollback_mutates_it(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow import publication

        output = self.root / "publication"
        parent_calls = 0
        failure_injected = False
        real_fsync = publication._fsync_directory

        def fail_once(path: Path) -> None:
            nonlocal failure_injected, parent_calls
            if path == self.root and (failure_injected or output.exists()):
                parent_calls += 1
                if not failure_injected:
                    failure_injected = True
                    raise OSError("first parent fsync failed")
            real_fsync(path)

        with patch.object(publication, "_fsync_directory", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "first parent fsync failed"):
                self.publish(output)

        self.assertEqual(parent_calls, 2)
        self.assertFalse(output.exists())

    def test_rollback_parent_fsync_failure_is_grouped_with_original(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow import publication

        output = self.root / "publication"
        parent_calls = 0
        failure_injected = False
        real_fsync = publication._fsync_directory

        def fail_twice(path: Path) -> None:
            nonlocal failure_injected, parent_calls
            if path == self.root and (failure_injected or output.exists()):
                failure_injected = True
                parent_calls += 1
                raise OSError(f"parent fsync failed {parent_calls}")
            real_fsync(path)

        with patch.object(publication, "_fsync_directory", side_effect=fail_twice):
            with self.assertRaises(ExceptionGroup) as caught:
                self.publish(output)

        messages = tuple(str(error) for error in caught.exception.exceptions)
        self.assertTrue(any("parent fsync failed 1" in message for message in messages))
        self.assertTrue(any("parent fsync failed 2" in message for message in messages))
        self.assertEqual(parent_calls, 2)

    def test_missing_parent_and_invalid_payload_inventory_fail_before_staging(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow.publication import (
            publish_exp417,
        )

        with self.assertRaises(FileNotFoundError):
            publish_exp417(
                self.payloads,
                self.root / "missing" / "output",
                checkpoints=object(),
                pre_publish_verify=lambda: None,
            )
        invalid = dict(self.payloads)
        invalid["foreign.bin"] = b"foreign\n"
        with self.assertRaisesRegex(ValueError, "seven|inventory|payload"):
            publish_exp417(
                invalid,
                self.root / "invalid",
                checkpoints=object(),
                pre_publish_verify=lambda: None,
            )

    def test_publication_rejects_caller_asserted_nonstore_checkpoint_root(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow.publication import (
            publish_exp417,
        )

        output = self.root / "untrusted"
        with self.assertRaisesRegex(ValueError, "Exp417CheckpointStore"):
            publish_exp417(
                self.payloads,
                output,
                checkpoints=object(),
                pre_publish_verify=lambda: None,
            )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
