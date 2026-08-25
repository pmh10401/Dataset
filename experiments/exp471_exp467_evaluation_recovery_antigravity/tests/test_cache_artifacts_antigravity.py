"""Lifecycle tests for durable create-only Exp471 score-cache artifacts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import unittest
from unittest import mock

from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    cache_artifacts_antigravity as cache_artifacts,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import (
    METHOD_ORDER_ANTIGRAVITY,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.contracts_antigravity import (
    Exp471ScoreCacheManifestAntigravity,
    Exp471ScoreShardAntigravity,
)


SOURCE_AUTHORITY_SHA256_ANTIGRAVITY = "a" * 64
CACHE_CONTEXT_SHA256_ANTIGRAVITY = "b" * 64
SOURCE_MANIFEST_SHA256_ANTIGRAVITY = "c" * 64
MANIFEST_NAME_ANTIGRAVITY = "score_cache_manifest_antigravity.json"
SCORE_SHARD_SCHEMA_ANTIGRAVITY = "exp471-score-cache-shard-antigravity.v2"
CACHE_MANIFEST_SCHEMA_ANTIGRAVITY = "exp471-score-cache-manifest-antigravity.v2"


def _canonical_bytes_antigravity(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _shard_record_antigravity(
    *,
    method: str = "A",
    dataset: str = "dataset_1",
    status: str = "scored",
    test_indices: tuple[int, ...] = (0, 1),
    raw_scores: tuple[float, ...] = (0.25, 0.75),
    micro_scores: tuple[float, ...] = (0.5, 1.0),
    abstention_reason: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SCORE_SHARD_SCHEMA_ANTIGRAVITY,
        "source_authority_sha256": SOURCE_AUTHORITY_SHA256_ANTIGRAVITY,
        "cache_context_sha256": CACHE_CONTEXT_SHA256_ANTIGRAVITY,
        "method": method,
        "dataset": dataset,
        "status": status,
        "test_count": len(test_indices),
        "test_indices": list(test_indices),
        "raw_scores": list(raw_scores),
        "micro_scores": list(micro_scores),
        "abstention_reason": abstention_reason,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256_ANTIGRAVITY,
    }
    payload["payload_sha256"] = sha256(
        _canonical_bytes_antigravity(payload)
    ).hexdigest()
    envelope = dict(payload)
    envelope["envelope_sha256"] = sha256(
        _canonical_bytes_antigravity(envelope)
    ).hexdigest()
    return envelope


def _score_shard_antigravity(
    *,
    method: str = "A",
    dataset: str = "dataset_1",
    status: str = "scored",
    test_indices: tuple[int, ...] = (0, 1),
    raw_scores: tuple[float, ...] = (0.25, 0.75),
    micro_scores: tuple[float, ...] = (0.5, 1.0),
    abstention_reason: str | None = None,
) -> Exp471ScoreShardAntigravity:
    record = _shard_record_antigravity(
        method=method,
        dataset=dataset,
        status=status,
        test_indices=test_indices,
        raw_scores=raw_scores,
        micro_scores=micro_scores,
        abstention_reason=abstention_reason,
    )
    record.pop("schema")
    return Exp471ScoreShardAntigravity(**record)


def _shard_record_from_contract_antigravity(
    shard: Exp471ScoreShardAntigravity,
) -> dict[str, object]:
    return {"schema": SCORE_SHARD_SCHEMA_ANTIGRAVITY, **asdict(shard)}


def _manifest_antigravity(
    shards: dict[tuple[str, str], Exp471ScoreShardAntigravity],
    datasets: tuple[str, ...],
) -> Exp471ScoreCacheManifestAntigravity:
    record: dict[str, object] = {
        "schema": CACHE_MANIFEST_SCHEMA_ANTIGRAVITY,
        "source_authority_sha256": SOURCE_AUTHORITY_SHA256_ANTIGRAVITY,
        "cache_context_sha256": CACHE_CONTEXT_SHA256_ANTIGRAVITY,
        "method_order": list(METHOD_ORDER_ANTIGRAVITY),
        "dataset_order": list(datasets),
        "shard_sha256": {
            f"{method}:{dataset}": shards[(method, dataset)].envelope_sha256
            for method in METHOD_ORDER_ANTIGRAVITY
            for dataset in datasets
        },
        "total_shards": len(METHOD_ORDER_ANTIGRAVITY) * len(datasets),
        "scored_shards": sum(
            shard.status == "scored" for shard in shards.values()
        ),
        "abstained_shards": sum(
            shard.status == "abstained" for shard in shards.values()
        ),
        "failed_shards": 0,
    }
    record["manifest_sha256"] = sha256(
        _canonical_bytes_antigravity(record)
    ).hexdigest()
    record.pop("schema")
    return Exp471ScoreCacheManifestAntigravity(**record)


def _manifest_record_from_contract_antigravity(
    manifest: Exp471ScoreCacheManifestAntigravity,
) -> dict[str, object]:
    return {
        "schema": CACHE_MANIFEST_SCHEMA_ANTIGRAVITY,
        "source_authority_sha256": manifest.source_authority_sha256,
        "cache_context_sha256": manifest.cache_context_sha256,
        "method_order": list(manifest.method_order),
        "dataset_order": list(manifest.dataset_order),
        "shard_sha256": dict(manifest.shard_sha256),
        "total_shards": manifest.total_shards,
        "scored_shards": manifest.scored_shards,
        "abstained_shards": manifest.abstained_shards,
        "failed_shards": manifest.failed_shards,
        "manifest_sha256": manifest.manifest_sha256,
    }


def _publish_complete_shards_antigravity(
    root: Path, datasets: tuple[str, ...] = ("dataset_1",)
) -> dict[tuple[str, str], Exp471ScoreShardAntigravity]:
    shards: dict[tuple[str, str], Exp471ScoreShardAntigravity] = {}
    for method in METHOD_ORDER_ANTIGRAVITY:
        for dataset in datasets:
            shard = _score_shard_antigravity(method=method, dataset=dataset)
            cache_artifacts.publish_score_shard_antigravity(root, shard)
            shards[(method, dataset)] = shard
    return shards


class Exp471ScoreShardArtifactTestsAntigravity(unittest.TestCase):
    def test_manifest_quarantine_no_replace_is_unavailable_off_darwin(self) -> None:
        """Quarantine must fail closed when Darwin's no-replace syscall is absent."""
        with mock.patch.object(cache_artifacts.sys, "platform", "linux"), self.assertRaisesRegex(
            RuntimeError, "no-replace move is unavailable"
        ):
            cache_artifacts._rename_no_replace_at_antigravity(
                -1,
                "source_antigravity.json",
                -1,
                "destination_antigravity.json",
            )

    def test_manifest_quarantine_no_replace_fails_when_darwin_symbol_is_missing(
        self,
    ) -> None:
        """A missing Darwin primitive must not select an overwrite-capable fallback."""
        with mock.patch.object(
            cache_artifacts.ctypes, "CDLL", return_value=object()
        ), self.assertRaisesRegex(RuntimeError, "no-replace move is unavailable"):
            cache_artifacts._rename_no_replace_at_antigravity(
                -1,
                "source_antigravity.json",
                -1,
                "destination_antigravity.json",
            )

    def test_digest_path_is_literal_deterministic_contained_and_antigravity_named(self) -> None:
        """Raw identities or escaping components must never become shard pathnames."""
        expected = Path(
            "score_shards_antigravity/"
            "8f6a4b67c1e326eab7aea91a9cecd249d14d100e29d5f2e12720023d57a5980d_"
            "antigravity.json"
        )
        first = cache_artifacts.score_shard_relative_path_antigravity(
            "A", "dataset/../alpha"
        )
        second = cache_artifacts.score_shard_relative_path_antigravity(
            "A", "dataset/../alpha"
        )
        self.assertEqual((first, second), (expected, expected))
        self.assertFalse(first.is_absolute())
        self.assertNotIn("..", first.parts)
        self.assertNotIn("dataset", first.as_posix())
        self.assertTrue(all("antigravity" in part for part in first.parts))
        for method, dataset in (("unknown", "x"), ("A", ""), ("A", 3)):
            with self.subTest(method=method, dataset=dataset):
                with self.assertRaises(ValueError):
                    cache_artifacts.score_shard_relative_path_antigravity(
                        method, dataset  # type: ignore[arg-type]
                    )

    def test_both_arms_publish_reopen_and_reuse_exact_bytes(self) -> None:
        """A restart must reuse exact scored and abstained v2 bytes."""
        arms = (
            _score_shard_antigravity(),
            _score_shard_antigravity(
                status="abstained",
                test_indices=(4, 9),
                raw_scores=(),
                micro_scores=(),
                abstention_reason="insufficient_train_rows",
            ),
        )
        for shard in arms:
            with self.subTest(status=shard.status), tempfile.TemporaryDirectory(
                prefix=f"exp471_shard_{shard.status}_reuse_antigravity_"
            ) as temporary:
                root = Path(temporary).resolve() / "cache_root_antigravity"
                path = cache_artifacts.publish_score_shard_antigravity(root, shard)
                before = path.stat()
                reopened = cache_artifacts.read_score_shard_antigravity(
                    root, shard.method, shard.dataset
                )
                reused = cache_artifacts.publish_score_shard_antigravity(root, shard)
                after = reused.stat()
                expected = _shard_record_from_contract_antigravity(shard)
                self.assertEqual(reopened, shard)
                self.assertEqual(
                    path.read_bytes(), _canonical_bytes_antigravity(expected)
                )
                self.assertEqual(
                    (before.st_dev, before.st_ino),
                    (after.st_dev, after.st_ino),
                )
                self.assertTrue(stat.S_ISREG(after.st_mode))
                self.assertEqual(stat.S_IMODE(after.st_mode), 0o444)
                self.assertEqual(
                    tuple(path.parent.glob(".*temporary_antigravity*")), ()
                )

        with tempfile.TemporaryDirectory(
            prefix="exp471_shard_union_conflict_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            cache_artifacts.publish_score_shard_antigravity(root, arms[0])
            with self.assertRaisesRegex(
                ValueError, "existing score_shard bytes differ"
            ):
                cache_artifacts.publish_score_shard_antigravity(root, arms[1])

    def test_v1_and_changed_union_discriminants_fail_closed(self) -> None:
        """Status, reason, indices, and the v2 schema must remain digest-bound."""
        valid = _shard_record_antigravity()
        cases: list[dict[str, object]] = []
        for key, value in (
            ("status", "abstained"),
            ("abstention_reason", "insufficient_train_rows"),
            ("test_indices", [1, 0]),
            ("schema", "exp471-score-cache-shard-antigravity.v1"),
        ):
            changed = dict(valid)
            changed[key] = value
            cases.append(changed)
        for index, record in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory(
                prefix="exp471_shard_digest_bound_antigravity_"
            ) as temporary:
                root = Path(temporary).resolve() / "cache_root_antigravity"
                path = root / cache_artifacts.score_shard_relative_path_antigravity(
                    "A", "dataset_1"
                )
                path.parent.mkdir(parents=True)
                path.write_bytes(_canonical_bytes_antigravity(record))
                with self.assertRaises(ValueError):
                    cache_artifacts.read_score_shard_antigravity(
                        root, "A", "dataset_1"
                    )

    def test_changed_truncated_and_noncanonical_existing_bytes_fail_without_rewrite(self) -> None:
        """Foreign or malformed durable bytes must never be repaired in place."""
        shard = _score_shard_antigravity()
        replacements = (
            b'{"foreign_antigravity":true}\n',
            _canonical_bytes_antigravity(
                _shard_record_from_contract_antigravity(shard)
            )[:-7],
            _canonical_bytes_antigravity(
                _shard_record_from_contract_antigravity(shard)
            ).replace(b'"dataset_1"', b'"dataset_1" '),
        )
        for replacement in replacements:
            with self.subTest(replacement=replacement[:24]), tempfile.TemporaryDirectory(
                prefix="exp471_shard_foreign_antigravity_"
            ) as temporary:
                root = Path(temporary).resolve() / "cache_root_antigravity"
                path = root / cache_artifacts.score_shard_relative_path_antigravity(
                    shard.method, shard.dataset
                )
                path.parent.mkdir(parents=True)
                path.write_bytes(replacement)
                before = path.lstat()
                with self.assertRaises(ValueError):
                    cache_artifacts.publish_score_shard_antigravity(root, shard)
                with self.assertRaises(ValueError):
                    cache_artifacts.read_score_shard_antigravity(
                        root, shard.method, shard.dataset
                    )
                after = path.lstat()
                self.assertEqual(path.read_bytes(), replacement)
                self.assertEqual(
                    (before.st_dev, before.st_ino), (after.st_dev, after.st_ino)
                )

    def test_leaf_parent_and_root_symlinks_are_rejected_without_reclamation(self) -> None:
        """A symlink at any artifact boundary must not be followed or removed."""
        shard = _score_shard_antigravity()
        with tempfile.TemporaryDirectory(
            prefix="exp471_shard_symlink_antigravity_"
        ) as temporary:
            base = Path(temporary).resolve()
            root = base / "cache_root_antigravity"
            destination = root / cache_artifacts.score_shard_relative_path_antigravity(
                shard.method, shard.dataset
            )
            destination.parent.mkdir(parents=True)
            target = base / "foreign_target_antigravity.json"
            target_bytes = _canonical_bytes_antigravity(
                _shard_record_from_contract_antigravity(shard)
            )
            target.write_bytes(target_bytes)
            destination.symlink_to(target)
            with self.assertRaises(ValueError):
                cache_artifacts.publish_score_shard_antigravity(root, shard)
            with self.assertRaises(ValueError):
                cache_artifacts.read_score_shard_antigravity(
                    root, shard.method, shard.dataset
                )
            self.assertTrue(destination.is_symlink())
            self.assertEqual(target.read_bytes(), target_bytes)

            linked_root = base / "linked_root_antigravity"
            linked_root.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                cache_artifacts.read_score_shard_antigravity(
                    linked_root, shard.method, shard.dataset
                )

            real_parent = base / "real_shards_antigravity"
            real_parent.mkdir()
            destination.unlink()
            destination.parent.rmdir()
            destination.parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                cache_artifacts.publish_score_shard_antigravity(root, shard)
            self.assertTrue(destination.parent.is_symlink())

    def test_crash_before_link_leaves_no_destination_or_temporary_leaf(self) -> None:
        """A failed publication link must leave only an absent shard, ready to retry."""
        shard = _score_shard_antigravity()
        with tempfile.TemporaryDirectory(
            prefix="exp471_shard_crash_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            destination = root / cache_artifacts.score_shard_relative_path_antigravity(
                shard.method, shard.dataset
            )
            with mock.patch.object(
                cache_artifacts.os, "link", side_effect=OSError("synthetic crash")
            ):
                with self.assertRaises(OSError):
                    cache_artifacts.publish_score_shard_antigravity(root, shard)
            self.assertFalse(destination.exists())
            self.assertEqual(tuple(destination.parent.iterdir()), ())

    def test_temporary_path_swap_cannot_publish_foreign_regular_or_symlink(self) -> None:
        """Replacing the temporary pathname must not redirect the authoritative link."""
        shard = _score_shard_antigravity()
        original_link = os.link
        for replacement_kind in ("regular", "symlink"):
            with self.subTest(replacement_kind=replacement_kind), tempfile.TemporaryDirectory(
                prefix=f"exp471_temp_swap_{replacement_kind}_antigravity_"
            ) as temporary:
                base = Path(temporary).resolve()
                root = base / "cache_root_antigravity"
                destination = (
                    root
                    / cache_artifacts.score_shard_relative_path_antigravity(
                        shard.method, shard.dataset
                    )
                )
                foreign_target = base / "foreign_target_antigravity.json"
                foreign_bytes = b'{"foreign_antigravity":true}\n'
                foreign_target.write_bytes(foreign_bytes)
                observed: dict[str, str] = {}

                def swap_then_link(
                    source: str, final: str, **kwargs: object
                ) -> None:
                    source_directory = kwargs["src_dir_fd"]
                    self.assertIsInstance(source_directory, int)
                    saved = f"{source}_saved_antigravity"
                    os.rename(
                        source,
                        saved,
                        src_dir_fd=source_directory,
                        dst_dir_fd=source_directory,
                    )
                    if replacement_kind == "regular":
                        descriptor = os.open(
                            source,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o600,
                            dir_fd=source_directory,
                        )
                        try:
                            os.write(descriptor, foreign_bytes)
                        finally:
                            os.close(descriptor)
                    else:
                        os.symlink(
                            foreign_target,
                            source,
                            dir_fd=source_directory,
                        )
                    observed["source"] = source
                    observed["saved"] = saved
                    original_link(source, final, **kwargs)  # type: ignore[arg-type]

                with mock.patch.object(
                    cache_artifacts.os, "link", side_effect=swap_then_link
                ):
                    with self.assertRaises(ValueError):
                        cache_artifacts.publish_score_shard_antigravity(root, shard)

                replacement = destination.parent / observed["source"]
                saved = destination.parent / observed["saved"]
                self.assertTrue(saved.is_file())
                if replacement_kind == "regular":
                    self.assertTrue(replacement.is_file())
                    self.assertEqual(replacement.read_bytes(), foreign_bytes)
                    self.assertTrue(destination.is_file())
                    self.assertEqual(destination.read_bytes(), foreign_bytes)
                else:
                    self.assertTrue(replacement.is_symlink())
                    self.assertTrue(destination.is_symlink())
                    self.assertEqual(foreign_target.read_bytes(), foreign_bytes)

    def test_link_failure_cleanup_preserves_foreign_temporary_replacement(self) -> None:
        """Failure cleanup must unlink only the temporary inode created by the publisher."""
        shard = _score_shard_antigravity()
        for replacement_kind in ("regular", "symlink"):
            with self.subTest(replacement_kind=replacement_kind), tempfile.TemporaryDirectory(
                prefix=f"exp471_temp_cleanup_{replacement_kind}_antigravity_"
            ) as temporary:
                base = Path(temporary).resolve()
                root = base / "cache_root_antigravity"
                destination = (
                    root
                    / cache_artifacts.score_shard_relative_path_antigravity(
                        shard.method, shard.dataset
                    )
                )
                foreign_target = base / "cleanup_target_antigravity.json"
                foreign_bytes = b'{"foreign_cleanup_antigravity":true}\n'
                foreign_target.write_bytes(foreign_bytes)
                observed: dict[str, str] = {}

                def swap_then_fail(
                    source: str, final: str, **kwargs: object
                ) -> None:
                    source_directory = kwargs["src_dir_fd"]
                    self.assertIsInstance(source_directory, int)
                    saved = f"{source}_saved_antigravity"
                    os.rename(
                        source,
                        saved,
                        src_dir_fd=source_directory,
                        dst_dir_fd=source_directory,
                    )
                    if replacement_kind == "regular":
                        descriptor = os.open(
                            source,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o600,
                            dir_fd=source_directory,
                        )
                        try:
                            os.write(descriptor, foreign_bytes)
                        finally:
                            os.close(descriptor)
                    else:
                        os.symlink(
                            foreign_target,
                            source,
                            dir_fd=source_directory,
                        )
                    observed["source"] = source
                    observed["saved"] = saved
                    raise OSError("synthetic link failure antigravity")

                with mock.patch.object(
                    cache_artifacts.os, "link", side_effect=swap_then_fail
                ):
                    with self.assertRaises(OSError):
                        cache_artifacts.publish_score_shard_antigravity(
                            root, shard
                        )

                replacement = destination.parent / observed["source"]
                saved = destination.parent / observed["saved"]
                self.assertFalse(destination.exists())
                self.assertTrue(saved.is_file())
                if replacement_kind == "regular":
                    self.assertTrue(replacement.is_file())
                    self.assertEqual(replacement.read_bytes(), foreign_bytes)
                else:
                    self.assertTrue(replacement.is_symlink())
                    self.assertEqual(foreign_target.read_bytes(), foreign_bytes)

    def test_partial_write_fsync_and_same_directory_hard_link_are_durable(self) -> None:
        """Short writes or omitted file/directory fsync must not publish partial data."""
        shard = _score_shard_antigravity()
        original_write = os.write
        original_fsync = os.fsync
        original_link = os.link
        fsync_modes: list[int] = []
        link_arguments: list[tuple[str, str, int | None, int | None]] = []

        def partial_write(descriptor: int, payload: bytes) -> int:
            return original_write(descriptor, payload[:7])

        def recording_fsync(descriptor: int) -> None:
            fsync_modes.append(os.fstat(descriptor).st_mode)
            original_fsync(descriptor)

        def recording_link(source: str, destination: str, **kwargs: object) -> None:
            link_arguments.append(
                (
                    source,
                    destination,
                    kwargs.get("src_dir_fd"),  # type: ignore[arg-type]
                    kwargs.get("dst_dir_fd"),  # type: ignore[arg-type]
                )
            )
            original_link(source, destination, **kwargs)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory(
            prefix="exp471_shard_partial_antigravity_"
        ) as temporary, mock.patch.object(
            cache_artifacts.os, "write", side_effect=partial_write
        ), mock.patch.object(
            cache_artifacts.os, "fsync", side_effect=recording_fsync
        ), mock.patch.object(
            cache_artifacts.os, "link", side_effect=recording_link
        ):
            root = Path(temporary).resolve() / "cache_root_antigravity"
            path = cache_artifacts.publish_score_shard_antigravity(root, shard)
            self.assertEqual(
                path.read_bytes(),
                _canonical_bytes_antigravity(
                    _shard_record_from_contract_antigravity(shard)
                ),
            )
        self.assertGreater(len(fsync_modes), 1)
        self.assertTrue(any(stat.S_ISREG(mode) for mode in fsync_modes))
        self.assertTrue(any(stat.S_ISDIR(mode) for mode in fsync_modes))
        self.assertEqual(len(link_arguments), 1)
        source, destination, source_directory, destination_directory = link_arguments[0]
        self.assertNotIn("/", source)
        self.assertNotIn("/", destination)
        self.assertEqual(source_directory, destination_directory)

    def test_identical_and_competing_concurrent_writers_never_clobber(self) -> None:
        """Concurrent publication must converge on exact bytes or reject the loser."""
        shard = _score_shard_antigravity()
        with tempfile.TemporaryDirectory(
            prefix="exp471_shard_concurrent_same_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            with ThreadPoolExecutor(max_workers=2) as executor:
                paths = tuple(
                    executor.map(
                        lambda _: cache_artifacts.publish_score_shard_antigravity(
                            root, shard
                        ),
                        range(2),
                    )
                )
            self.assertEqual(paths[0], paths[1])
            self.assertEqual(
                paths[0].read_bytes(),
                _canonical_bytes_antigravity(
                    _shard_record_from_contract_antigravity(shard)
                ),
            )
            self.assertEqual(tuple(paths[0].parent.glob(".*temporary_antigravity*")), ())

        first = _score_shard_antigravity(raw_scores=(1.0, 2.0))
        second = _score_shard_antigravity(raw_scores=(3.0, 4.0))
        with tempfile.TemporaryDirectory(
            prefix="exp471_shard_concurrent_diff_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            barrier = threading.Barrier(2)
            original_link = os.link

            def synchronized_link(*args: object, **kwargs: object) -> None:
                barrier.wait(timeout=5)
                original_link(*args, **kwargs)  # type: ignore[arg-type]

            def publish(candidate: Exp471ScoreShardAntigravity) -> object:
                try:
                    return cache_artifacts.publish_score_shard_antigravity(
                        root, candidate
                    )
                except ValueError as error:
                    return error

            with mock.patch.object(
                cache_artifacts.os, "link", side_effect=synchronized_link
            ), ThreadPoolExecutor(max_workers=2) as executor:
                results = tuple(executor.map(publish, (first, second)))
            self.assertEqual(sum(isinstance(result, Path) for result in results), 1)
            self.assertEqual(sum(isinstance(result, ValueError) for result in results), 1)
            persisted = cache_artifacts.read_score_shard_antigravity(
                root, "A", "dataset_1"
            )
            self.assertIn(persisted, (first, second))

    def test_read_rejects_nonfinite_wrong_count_and_wrong_identity(self) -> None:
        """Invalid vectors or a shard stored under another identity must fail closed."""
        shard = _score_shard_antigravity()
        cases: list[tuple[str, bytes]] = []

        nonfinite = _shard_record_antigravity()
        nonfinite["raw_scores"] = [float("nan"), 0.75]
        cases.append(
            (
                "nonfinite",
                json.dumps(nonfinite, allow_nan=True, separators=(",", ":")).encode()
                + b"\n",
            )
        )

        wrong_count = _shard_record_antigravity()
        wrong_count["test_count"] = 3
        payload = {key: value for key, value in wrong_count.items() if key not in {"payload_sha256", "envelope_sha256"}}
        wrong_count["payload_sha256"] = sha256(_canonical_bytes_antigravity(payload)).hexdigest()
        envelope = {key: value for key, value in wrong_count.items() if key != "envelope_sha256"}
        wrong_count["envelope_sha256"] = sha256(_canonical_bytes_antigravity(envelope)).hexdigest()
        cases.append(("wrong_count", _canonical_bytes_antigravity(wrong_count)))

        wrong_identity = _score_shard_antigravity(method="B")
        cases.append(
            (
                "wrong_identity",
                _canonical_bytes_antigravity(
                    _shard_record_from_contract_antigravity(wrong_identity)
                ),
            )
        )

        for name, raw in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                prefix=f"exp471_shard_{name}_antigravity_"
            ) as temporary:
                root = Path(temporary).resolve() / "cache_root_antigravity"
                path = root / cache_artifacts.score_shard_relative_path_antigravity(
                    shard.method, shard.dataset
                )
                path.parent.mkdir(parents=True)
                path.write_bytes(raw)
                expected_error = (
                    "identity differs from its path"
                    if name == "wrong_identity"
                    else ".*"
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    cache_artifacts.read_score_shard_antigravity(
                        root, shard.method, shard.dataset
                    )

    def test_descriptor_bound_leaf_swap_is_rejected(self) -> None:
        """Replacing a shard pathname after open must invalidate the read."""
        shard = _score_shard_antigravity()
        with tempfile.TemporaryDirectory(
            prefix="exp471_shard_swap_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            path = cache_artifacts.publish_score_shard_antigravity(root, shard)
            replacement = (
                Path(temporary).resolve() / "replacement_shard_antigravity.json"
            )
            replacement.write_bytes(path.read_bytes())
            fired = False

            def swap(event: str, relative: str) -> None:
                nonlocal fired
                if not fired and event == "cache:after_leaf_open":
                    fired = True
                    path.rename(path.with_suffix(".saved_antigravity"))
                    path.symlink_to(replacement)

            with mock.patch.object(
                cache_artifacts, "_race_hook_antigravity", side_effect=swap
            ):
                with self.assertRaises(ValueError):
                    cache_artifacts.read_score_shard_antigravity(
                        root, shard.method, shard.dataset
                    )
            self.assertTrue(fired)


class Exp471CacheManifestArtifactTestsAntigravity(unittest.TestCase):
    def test_manifest_scans_mixed_statuses_and_rejects_false_counts(self) -> None:
        """Terminal counts must equal durable scored/abstained shard statuses."""
        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_mixed_status_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            shards: dict[tuple[str, str], Exp471ScoreShardAntigravity] = {}
            for method in METHOD_ORDER_ANTIGRAVITY:
                shard = (
                    _score_shard_antigravity(
                        method=method,
                        status="abstained",
                        raw_scores=(),
                        micro_scores=(),
                        abstention_reason="insufficient_train_rows",
                    )
                    if method == "A"
                    else _score_shard_antigravity(method=method)
                )
                cache_artifacts.publish_score_shard_antigravity(root, shard)
                shards[(method, "dataset_1")] = shard

            manifest = _manifest_antigravity(shards, ("dataset_1",))
            self.assertEqual(
                (
                    manifest.scored_shards,
                    manifest.abstained_shards,
                    manifest.failed_shards,
                ),
                (9, 1, 0),
            )
            cache_artifacts.publish_cache_manifest_antigravity(root, manifest)
            self.assertEqual(
                cache_artifacts.read_cache_manifest_antigravity(root), manifest
            )

        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_false_status_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            shards = _publish_complete_shards_antigravity(root)
            valid = _manifest_antigravity(shards, ("dataset_1",))
            false_record = _manifest_record_from_contract_antigravity(valid)
            false_record.update(scored_shards=9, failed_shards=1)
            false_record.pop("manifest_sha256")
            false_record["manifest_sha256"] = sha256(
                _canonical_bytes_antigravity(false_record)
            ).hexdigest()
            constructor_record = dict(false_record)
            constructor_record.pop("schema")
            false_manifest = Exp471ScoreCacheManifestAntigravity(
                **constructor_record
            )
            with self.assertRaisesRegex(ValueError, "status counts differ"):
                cache_artifacts.publish_cache_manifest_antigravity(
                    root, false_manifest
                )

    def test_complete_manifest_publishes_reopens_and_reuses_exact_bytes(self) -> None:
        """A complete shard matrix must publish one durable terminal authority."""
        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_complete_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            shards = _publish_complete_shards_antigravity(root)
            manifest = _manifest_antigravity(shards, ("dataset_1",))
            path = cache_artifacts.publish_cache_manifest_antigravity(root, manifest)
            before = path.stat()
            reopened = cache_artifacts.read_cache_manifest_antigravity(root)
            reused = cache_artifacts.publish_cache_manifest_antigravity(root, manifest)
            after = reused.stat()
            self.assertEqual(reopened, manifest)
            self.assertEqual(path.name, MANIFEST_NAME_ANTIGRAVITY)
            self.assertEqual(
                path.read_bytes(),
                _canonical_bytes_antigravity(
                    _manifest_record_from_contract_antigravity(manifest)
                ),
            )
            self.assertEqual((before.st_dev, before.st_ino), (after.st_dev, after.st_ino))
            self.assertEqual(stat.S_IMODE(after.st_mode), 0o444)

    def test_manifest_before_complete_coverage_is_rejected_without_publication(self) -> None:
        """A terminal manifest must not exist while even one declared shard is absent."""
        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_incomplete_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            shards = {
                (method, "dataset_1"): _score_shard_antigravity(
                    method=method, dataset="dataset_1"
                )
                for method in METHOD_ORDER_ANTIGRAVITY
            }
            for index, shard in enumerate(shards.values()):
                if index:
                    cache_artifacts.publish_score_shard_antigravity(root, shard)
            manifest = _manifest_antigravity(shards, ("dataset_1",))
            with self.assertRaises(ValueError):
                cache_artifacts.publish_cache_manifest_antigravity(root, manifest)
            self.assertFalse((root / MANIFEST_NAME_ANTIGRAVITY).exists())

    def test_coverage_directory_swap_rolls_back_new_terminal_manifest(self) -> None:
        """Coverage replaced after verification must not leave terminal authority."""
        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_coverage_swap_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            shards = _publish_complete_shards_antigravity(root)
            manifest = _manifest_antigravity(shards, ("dataset_1",))
            original_link = os.link
            swapped = False

            def swap_coverage_then_link(
                source: str, destination: str, **kwargs: object
            ) -> None:
                nonlocal swapped
                destination_directory = kwargs["dst_dir_fd"]
                self.assertIsInstance(destination_directory, int)
                if destination == MANIFEST_NAME_ANTIGRAVITY:
                    swapped = True
                    os.rename(
                        "score_shards_antigravity",
                        "score_shards_saved_antigravity",
                        src_dir_fd=destination_directory,
                        dst_dir_fd=destination_directory,
                    )
                    os.mkdir(
                        "score_shards_antigravity",
                        0o755,
                        dir_fd=destination_directory,
                    )
                original_link(source, destination, **kwargs)  # type: ignore[arg-type]

            with mock.patch.object(
                cache_artifacts.os, "link", side_effect=swap_coverage_then_link
            ):
                with self.assertRaises(ValueError):
                    cache_artifacts.publish_cache_manifest_antigravity(root, manifest)

            self.assertTrue(swapped)
            self.assertFalse((root / MANIFEST_NAME_ANTIGRAVITY).exists())
            self.assertTrue((root / "score_shards_antigravity").is_dir())
            self.assertEqual(tuple((root / "score_shards_antigravity").iterdir()), ())
            self.assertTrue((root / "score_shards_saved_antigravity").is_dir())

    def test_identical_and_different_concurrent_manifest_writers_are_create_only(self) -> None:
        """Concurrent terminal writers must converge exactly or reject the changed loser."""
        datasets = ("dataset_1", "dataset_2")
        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_concurrent_same_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            shards = _publish_complete_shards_antigravity(root, datasets)
            manifest = _manifest_antigravity(shards, datasets)
            with ThreadPoolExecutor(max_workers=2) as executor:
                paths = tuple(
                    executor.map(
                        lambda _: cache_artifacts.publish_cache_manifest_antigravity(
                            root, manifest
                        ),
                        range(2),
                    )
                )
            expected_bytes = _canonical_bytes_antigravity(
                _manifest_record_from_contract_antigravity(manifest)
            )
            self.assertEqual(paths[0], paths[1])
            self.assertEqual(paths[0].read_bytes(), expected_bytes)
            self.assertEqual(
                tuple(root.glob(".*score_cache_manifest_temporary_antigravity*")),
                (),
            )

        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_concurrent_diff_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            shards = _publish_complete_shards_antigravity(root, datasets)
            first = _manifest_antigravity(shards, datasets)
            second = _manifest_antigravity(shards, tuple(reversed(datasets)))
            barrier = threading.Barrier(2)
            original_link = os.link

            def synchronized_manifest_link(
                source: str, destination: str, **kwargs: object
            ) -> None:
                if destination == MANIFEST_NAME_ANTIGRAVITY:
                    barrier.wait(timeout=5)
                original_link(source, destination, **kwargs)  # type: ignore[arg-type]

            def publish(
                candidate: Exp471ScoreCacheManifestAntigravity,
            ) -> object:
                try:
                    return cache_artifacts.publish_cache_manifest_antigravity(
                        root, candidate
                    )
                except ValueError as error:
                    return error

            with mock.patch.object(
                cache_artifacts.os,
                "link",
                side_effect=synchronized_manifest_link,
            ), ThreadPoolExecutor(max_workers=2) as executor:
                results = tuple(executor.map(publish, (first, second)))

            self.assertEqual(sum(isinstance(result, Path) for result in results), 1)
            self.assertEqual(
                sum(isinstance(result, ValueError) for result in results), 1
            )
            destination = root / MANIFEST_NAME_ANTIGRAVITY
            winner = cache_artifacts.read_cache_manifest_antigravity(root)
            self.assertIn(winner, (first, second))
            self.assertEqual(
                destination.read_bytes(),
                _canonical_bytes_antigravity(
                    _manifest_record_from_contract_antigravity(winner)
                ),
            )
            self.assertEqual(
                tuple(root.glob(".*score_cache_manifest_temporary_antigravity*")),
                (),
            )

    def test_manifest_rejects_wrong_shard_digest_identity_and_existing_bytes(self) -> None:
        """Coverage metadata must bind every exact method/dataset envelope."""
        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_mismatch_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            shards = _publish_complete_shards_antigravity(root)
            valid = _manifest_antigravity(shards, ("dataset_1",))
            record = _manifest_record_from_contract_antigravity(valid)
            record["shard_sha256"]["A:dataset_1"] = "f" * 64
            bare = {key: value for key, value in record.items() if key != "manifest_sha256"}
            record["manifest_sha256"] = sha256(_canonical_bytes_antigravity(bare)).hexdigest()
            constructor_record = dict(record)
            constructor_record.pop("schema")
            mismatched = Exp471ScoreCacheManifestAntigravity(**constructor_record)
            with self.assertRaises(ValueError):
                cache_artifacts.publish_cache_manifest_antigravity(root, mismatched)

            destination = root / MANIFEST_NAME_ANTIGRAVITY
            foreign = b'{"foreign_manifest_antigravity":true}\n'
            destination.write_bytes(foreign)
            before = destination.lstat()
            with self.assertRaises(ValueError):
                cache_artifacts.publish_cache_manifest_antigravity(root, valid)
            with self.assertRaises(ValueError):
                cache_artifacts.read_cache_manifest_antigravity(root)
            after = destination.lstat()
            self.assertEqual(destination.read_bytes(), foreign)
            self.assertEqual((before.st_dev, before.st_ino), (after.st_dev, after.st_ino))

    def test_manifest_read_rejects_wrong_order_count_truncation_and_symlink(self) -> None:
        """Malformed terminal schemas and linked destinations must never authorize evaluation."""
        valid_record = _manifest_record_from_contract_antigravity(
            _manifest_antigravity(
                {
                    (method, "dataset_1"): _score_shard_antigravity(
                        method=method, dataset="dataset_1"
                    )
                    for method in METHOD_ORDER_ANTIGRAVITY
                },
                ("dataset_1",),
            )
        )
        malformed_records: list[bytes] = []
        wrong_order = dict(valid_record)
        wrong_order["method_order"] = list(reversed(METHOD_ORDER_ANTIGRAVITY))
        malformed_records.append(_canonical_bytes_antigravity(wrong_order))
        wrong_dataset_order = dict(valid_record)
        wrong_dataset_order["dataset_order"] = ["dataset_1", "dataset_1"]
        malformed_records.append(_canonical_bytes_antigravity(wrong_dataset_order))
        wrong_count = dict(valid_record)
        wrong_count["total_shards"] = 9
        malformed_records.append(_canonical_bytes_antigravity(wrong_count))
        v1 = dict(valid_record)
        v1["schema"] = "exp471-score-cache-manifest-antigravity.v1"
        malformed_records.append(_canonical_bytes_antigravity(v1))
        malformed_records.append(_canonical_bytes_antigravity(valid_record)[:-5])

        for index, raw in enumerate(malformed_records):
            with self.subTest(index=index), tempfile.TemporaryDirectory(
                prefix="exp471_manifest_malformed_antigravity_"
            ) as temporary:
                root = Path(temporary).resolve() / "cache_root_antigravity"
                root.mkdir()
                (root / MANIFEST_NAME_ANTIGRAVITY).write_bytes(raw)
                with self.assertRaises(ValueError):
                    cache_artifacts.read_cache_manifest_antigravity(root)

        with tempfile.TemporaryDirectory(
            prefix="exp471_manifest_symlink_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve() / "cache_root_antigravity"
            root.mkdir()
            target = (
                Path(temporary).resolve() / "manifest_target_antigravity.json"
            )
            target.write_bytes(_canonical_bytes_antigravity(valid_record))
            linked = root / MANIFEST_NAME_ANTIGRAVITY
            linked.symlink_to(target)
            with self.assertRaises(ValueError):
                cache_artifacts.read_cache_manifest_antigravity(root)
            self.assertTrue(linked.is_symlink())


if __name__ == "__main__":
    unittest.main()
