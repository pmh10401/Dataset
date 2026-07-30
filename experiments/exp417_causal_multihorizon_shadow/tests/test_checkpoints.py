"""Synthetic-only tests for Exp417 local checkpoint safety."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    Exp417CheckpointProvenance,
    Exp417CheckpointStore,
)
from experiments.exp417_causal_multihorizon_shadow import checkpoints
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    DatasetDecisionEvidence,
    DatasetResult,
    FrozenLaneSnapshot,
    FrozenRunContext,
    FrozenTrainContract,
    GroupQualification,
    HorizonScore,
    SeedDecision,
    TestInstanceDecision,
)


def _lanes() -> FrozenLaneSnapshot:
    return FrozenLaneSnapshot(
        hard={"lane-hard": tuple(range(2218))},
        standard={"lane-standard": tuple(range(806))},
        priority={"lane-priority": tuple(range(31))},
        automatic_normal={"lane-automatic": tuple(range(35))},
    )


def _context() -> FrozenRunContext:
    datasets = tuple(f"synthetic-{number:04d}" for number in range(1117))
    return FrozenRunContext(
        database_sha256=FROZEN_CONFIG.database_sha256,
        datasets=datasets,
        lanes=_lanes(),
        test_identity_manifest={name: ((0, 100), (1, 100)) for name in datasets},
        code_commit="c" * 40,
        runtime_versions={"numpy": "synthetic", "python": "synthetic", "torch": "synthetic"},
    )


def _import_context() -> FrozenRunContext:
    datasets = ("a", "b") + tuple(
        f"synthetic-{number:04d}" for number in range(1115)
    )
    return FrozenRunContext(
        database_sha256=FROZEN_CONFIG.database_sha256,
        datasets=datasets,
        lanes=_lanes(),
        test_identity_manifest={
            name: ((0, 100), (1, 100)) for name in datasets
        },
        code_commit="c" * 40,
        runtime_versions={
            "numpy": "synthetic",
            "python": "synthetic",
            "torch": "synthetic",
        },
    )


def _group() -> GroupQualification:
    return GroupQualification(
        length=100,
        series_count=100,
        status="qualified",
        oof_candidate_rate=0.01,
        wilson_upper=0.05,
        pooled_synthetic_sensitivity=0.70,
        per_type_synthetic_sensitivity={
            "level_shift": 0.50,
            "gradual_drift": 0.50,
            "amplitude_scaling": 0.50,
            "period_change": 0.50,
        },
        seed_agreement=True,
    )


def _diagnostic_group() -> GroupQualification:
    return GroupQualification(
        length=64,
        series_count=64,
        status="diagnostic",
        oof_candidate_rate=0.01,
        wilson_upper=0.05,
        pooled_synthetic_sensitivity=None,
        per_type_synthetic_sensitivity={
            "level_shift": None,
            "gradual_drift": None,
            "amplitude_scaling": None,
            "period_change": None,
        },
        seed_agreement=True,
    )


def _decision(dataset: str, index: int) -> TestInstanceDecision:
    scores = {
        1: HorizonScore(1, 0.0, 0.5, 0.0),
        2: HorizonScore(2, 0.0, 0.5, 0.0),
        10: HorizonScore(10, 0.0, 0.5, 0.0),
    }
    return TestInstanceDecision(
        dataset, index, 100,
        tuple(SeedDecision(seed, scores, False) for seed in FROZEN_CONFIG.model_seeds),
    )


def _result(context: FrozenRunContext, dataset: str = "synthetic-0000") -> DatasetResult:
    train = FrozenTrainContract(
        dataset, (_group(),), BackendAudit("cpu", True, 0.0, 1.0, 1.0, False), "a" * 64
    )
    evidence = DatasetDecisionEvidence(
        dataset, context, train, context.database_sha256, train.train_identity_sha256,
        (_decision(dataset, 0), _decision(dataset, 1)),
    )
    return DatasetResult(dataset, 2, 1, (_group(),), (), evidence, context.lanes, context, "qualified")


def _mixed_result(context: FrozenRunContext) -> DatasetResult:
    dataset = "synthetic-0000"
    manifest = dict(context.test_identity_manifest)
    manifest[dataset] = ((0, 64), (1, 100))
    mixed_context = replace(context, test_identity_manifest=manifest)
    groups = (_diagnostic_group(), _group())
    train = FrozenTrainContract(
        dataset,
        groups,
        BackendAudit("cpu", True, 0.0, 1.0, 1.0, False),
        "a" * 64,
    )
    evidence = DatasetDecisionEvidence(
        dataset,
        mixed_context,
        train,
        mixed_context.database_sha256,
        train.train_identity_sha256,
        (_decision(dataset, 1),),
    )
    return DatasetResult(
        dataset,
        2,
        2,
        groups,
        (),
        evidence,
        mixed_context.lanes,
        mixed_context,
        "qualified",
    )


def _provenance() -> Exp417CheckpointProvenance:
    return Exp417CheckpointProvenance(
        git_commit="c" * 40,
        database_sha256=FROZEN_CONFIG.database_sha256,
        exp291_prediction_sha256=FROZEN_CONFIG.exp291_prediction_sha256,
        exp291_manifest_sha256=FROZEN_CONFIG.exp291_manifest_sha256,
        exp292_prediction_sha256=FROZEN_CONFIG.exp292_prediction_sha256,
        exp292_manifest_sha256=FROZEN_CONFIG.exp292_manifest_sha256,
        frozen_config_sha256="5" * 64,
        preprocessing_contract_sha256="6" * 64,
        fold_contract_sha256="7" * 64,
        model_contract_sha256="8" * 64,
        model_seeds=FROZEN_CONFIG.model_seeds,
        perturbation_seeds=FROZEN_CONFIG.perturbation_seeds,
        runtime_versions={"numpy": "synthetic", "python": "synthetic", "torch": "synthetic"},
        backend_contract_sha256="9" * 64,
    )


def _write(store: Exp417CheckpointStore, result: DatasetResult) -> None:
    store.write_dataset(
        result,
        raw_train_identity_sha256="a" * 64,
        repaired_train_identity_sha256="b" * 64,
        length_group_identity_sha256={100: "c" * 64},
        canonical_model_hashes={100: {seed: "d" * 64 for seed in FROZEN_CONFIG.model_seeds}},
        score_calibration_hashes={100: "e" * 64},
        backend_audit=BackendAudit("cpu", True, 0.0, 1.0, 1.0, False),
        elapsed_seconds=0.125,
    )


def _checkpoint_tree_snapshot(
    root: Path,
) -> dict[str, tuple[int, int, int, bytes | None]]:
    snapshot: dict[str, tuple[int, int, int, bytes | None]] = {}
    for path in (root, *sorted(root.rglob("*"))):
        metadata = os.lstat(path)
        payload = path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None
        snapshot[str(path.relative_to(root))] = (
            stat.S_IFMT(metadata.st_mode),
            metadata.st_dev,
            metadata.st_ino,
            payload,
        )
    return snapshot


class Exp417CheckpointTests(unittest.TestCase):
    def test_final_fix_invalidates_pre_semantic_checkpoint_schema(self) -> None:
        """Old records cannot resume after diagnostic and trust-root semantics changed."""
        from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
            _SCHEMA_VERSION,
        )

        self.assertEqual(_SCHEMA_VERSION, 3)

    def test_root_initialization_never_overwrites_a_competing_entry(self) -> None:
        """Root provenance publication is descriptor-bound and no-replace."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            provenance = root / "checkpoint_provenance.json"
            foreign_bytes = b"foreign-provenance-during-initialization"
            original_exists = checkpoints._path_exists_lstat
            original_rename = checkpoints._atomic_rename_noreplace_at
            injected = False

            def inject_after_path_absence(path: Path) -> bool:
                nonlocal injected
                existed = original_exists(path)
                if path == provenance and not existed and not injected:
                    provenance.write_bytes(foreign_bytes)
                    injected = True
                return existed

            def inject_before_descriptor_publish(
                source_descriptor: int,
                source_name: str,
                destination_descriptor: int,
                destination_name: str,
            ) -> None:
                nonlocal injected
                if (
                    destination_name == "checkpoint_provenance.json"
                    and not injected
                ):
                    descriptor = os.open(
                        destination_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=destination_descriptor,
                    )
                    try:
                        os.write(descriptor, foreign_bytes)
                    finally:
                        os.close(descriptor)
                    injected = True
                original_rename(
                    source_descriptor,
                    source_name,
                    destination_descriptor,
                    destination_name,
                )

            with (
                patch.object(
                    checkpoints,
                    "_path_exists_lstat",
                    side_effect=inject_after_path_absence,
                ),
                patch.object(
                    checkpoints,
                    "_atomic_rename_noreplace_at",
                    side_effect=inject_before_descriptor_publish,
                ),
                self.assertRaises(BaseException),
            ):
                Exp417CheckpointStore(
                    root, _provenance(), context.datasets
                )

            self.assertTrue(injected)
            self.assertTrue(any(
                path.is_file() and path.read_bytes() == foreign_bytes
                for path in parent.rglob("*")
            ))

    def test_root_initialization_populates_only_its_staging_inode(self) -> None:
        """A mkdir-to-open pathname swap cannot populate a foreign root."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            foreign = parent / "foreign-root"
            foreign.mkdir()
            foreign_identity = (
                foreign.stat().st_dev,
                foreign.stat().st_ino,
            )
            owned_saved = parent / "owned-root-saved"
            original_stat = os.stat
            swapped = False

            def substitute_staging_before_first_stat(
                path, *args, **kwargs
            ):
                nonlocal swapped
                if (
                    not swapped
                    and isinstance(path, str)
                    and path.startswith(".exp417-init-")
                    and kwargs.get("dir_fd") is not None
                ):
                    staging = parent / path
                    os.rename(staging, owned_saved)
                    os.rename(foreign, staging)
                    swapped = True
                return original_stat(path, *args, **kwargs)

            with (
                patch.object(
                    checkpoints.os,
                    "stat",
                    side_effect=substitute_staging_before_first_stat,
                ),
                self.assertRaises(ValueError),
            ):
                Exp417CheckpointStore(root, _provenance(), context.datasets)

            self.assertTrue(swapped)
            self.assertFalse(root.exists())
            foreign_locations = tuple(
                path
                for path in parent.iterdir()
                if path.is_dir()
                and (path.stat().st_dev, path.stat().st_ino)
                == foreign_identity
            )
            self.assertEqual(len(foreign_locations), 1)
            self.assertEqual(tuple(foreign_locations[0].iterdir()), ())
            self.assertTrue(owned_saved.is_dir())
            self.assertEqual(tuple(owned_saved.iterdir()), ())

    def test_root_publication_rolls_back_a_substituted_staging_basename(
        self,
    ) -> None:
        """A final source swap cannot poison the canonical checkpoint name."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            foreign = parent / "foreign-root"
            foreign.mkdir()
            foreign_payload = b"foreign-root-payload"
            (foreign / "foreign.bin").write_bytes(foreign_payload)
            foreign_identity = (
                foreign.stat().st_dev,
                foreign.stat().st_ino,
            )
            owned_saved = parent / "owned-complete-staging"
            original_rename = checkpoints._atomic_rename_noreplace_at
            swapped = False

            def substitute_staging_before_final_publish(
                source_descriptor: int,
                source_name: str,
                destination_descriptor: int,
                destination_name: str,
            ) -> None:
                nonlocal swapped
                if (
                    not swapped
                    and source_name.startswith(".exp417-init-")
                    and destination_name == "checkpoint"
                ):
                    staging = parent / source_name
                    os.rename(staging, owned_saved)
                    os.rename(foreign, staging)
                    swapped = True
                original_rename(
                    source_descriptor,
                    source_name,
                    destination_descriptor,
                    destination_name,
                )

            with (
                patch.object(
                    checkpoints,
                    "_atomic_rename_noreplace_at",
                    side_effect=substitute_staging_before_final_publish,
                ),
                self.assertRaises(BaseException),
            ):
                Exp417CheckpointStore(
                    root, _provenance(), context.datasets
                )

            self.assertTrue(swapped)
            self.assertFalse(root.exists())
            foreign_locations = tuple(
                path
                for path in parent.iterdir()
                if path.is_dir()
                and (path.stat().st_dev, path.stat().st_ino)
                == foreign_identity
            )
            self.assertEqual(len(foreign_locations), 1)
            self.assertEqual(
                (foreign_locations[0] / "foreign.bin").read_bytes(),
                foreign_payload,
            )
            self.assertTrue(owned_saved.is_dir())
            self.assertTrue(
                (owned_saved / "checkpoint_provenance.json").is_file()
            )

    def test_root_publication_revalidates_staging_payload_after_rename(
        self,
    ) -> None:
        """A post-preflight mutation cannot become an official checkpoint."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            corrupt_progress = b"corrupt-after-preflight"
            original_rename = checkpoints._atomic_rename_noreplace_at
            injected = False

            def corrupt_staging_before_final_publish(
                source_descriptor: int,
                source_name: str,
                destination_descriptor: int,
                destination_name: str,
            ) -> None:
                nonlocal injected
                if (
                    not injected
                    and source_name.startswith(".exp417-init-")
                    and destination_name == "checkpoint"
                ):
                    staging_descriptor = os.open(
                        source_name,
                        checkpoints._DIRECTORY_OPEN_FLAGS,
                        dir_fd=source_descriptor,
                    )
                    try:
                        descriptor = os.open(
                            "progress.json",
                            os.O_WRONLY | os.O_TRUNC,
                            dir_fd=staging_descriptor,
                        )
                        try:
                            os.write(descriptor, corrupt_progress)
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
                    finally:
                        os.close(staging_descriptor)
                    injected = True
                original_rename(
                    source_descriptor,
                    source_name,
                    destination_descriptor,
                    destination_name,
                )

            with (
                patch.object(
                    checkpoints,
                    "_atomic_rename_noreplace_at",
                    side_effect=corrupt_staging_before_final_publish,
                ),
                self.assertRaises(BaseException),
            ):
                Exp417CheckpointStore(
                    root, _provenance(), context.datasets
                )

            self.assertTrue(injected)
            self.assertFalse(root.exists())
            recovered = tuple(
                path
                for path in parent.iterdir()
                if path.is_dir()
                and (path / "progress.json").is_file()
                and (path / "progress.json").read_bytes()
                == corrupt_progress
            )
            self.assertEqual(len(recovered), 1)

    def test_complete_synthetic_checkpoint_round_trips_and_accounts_for_progress(self) -> None:
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            _write(store, _result(context))
            loaded = store.load_completed()
            self.assertEqual(tuple(loaded), ("synthetic-0000",))
            self.assertEqual(loaded["synthetic-0000"], _result(context))
            self.assertEqual(store.progress()["completed_datasets"], 1)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                store.require_complete()

    def test_imports_one_validated_shard_record(self) -> None:
        """Re-encoding a shard could silently change its canonical checkpoint bytes."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a",)
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )
            _write(source, _result(context, "a"))
            source_record = next(
                (source.root / "datasets").glob("*.json")
            ).read_bytes()

            digest = target.import_validated_dataset(source, "a")

            self.assertEqual(
                digest, target.completed_record_digests()["a"]
            )
            self.assertEqual(
                target.load_completed()["a"], _result(context, "a")
            )
            self.assertEqual(
                next((target.root / "datasets").glob("*.json")).read_bytes(),
                source_record,
            )

    def test_import_rejects_provenance_mismatch(self) -> None:
        """A shard from a different frozen contract cannot enter the target run."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a",)
            )
            target = Exp417CheckpointStore(
                parent / "target",
                replace(_provenance(), model_contract_sha256="f" * 64),
                ("a", "b"),
            )
            _write(source, _result(context, "a"))

            with self.assertRaisesRegex(ValueError, "provenance"):
                target.import_validated_dataset(source, "a")

            self.assertEqual(dict(target.load_completed()), {})

    def test_import_rejects_wrong_source_shard_inventory(self) -> None:
        """An unaccounted source entry must not be ignored during shard import."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a",)
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )
            _write(source, _result(context, "a"))
            (source.root / "datasets" / "unexpected.json").write_bytes(b"{}")

            with self.assertRaisesRegex(ValueError, "inventory"):
                target.import_validated_dataset(source, "a")

            self.assertEqual(dict(target.load_completed()), {})

    def test_import_rejects_missing_source_record(self) -> None:
        """A source progress file cannot stand in for the requested record."""
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a",)
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )

            with self.assertRaisesRegex(ValueError, "missing"):
                target.import_validated_dataset(source, "a")

            self.assertEqual(dict(target.load_completed()), {})

    def test_import_rejects_duplicate_target_record(self) -> None:
        """A transfer must never replace a checkpoint already owned by the target."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a",)
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )
            result = _result(context, "a")
            _write(source, result)
            _write(target, result)
            before = dict(target.completed_record_digests())

            with self.assertRaisesRegex(ValueError, "already exists"):
                target.import_validated_dataset(source, "a")

            self.assertEqual(dict(target.completed_record_digests()), before)

    def test_import_rejects_corrupt_source_json(self) -> None:
        """Malformed source bytes cannot be copied before full reconstruction."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a",)
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )
            _write(source, _result(context, "a"))
            next((source.root / "datasets").glob("*.json")).write_bytes(b"{")

            with self.assertRaises(ValueError):
                target.import_validated_dataset(source, "a")

            self.assertEqual(dict(target.load_completed()), {})

    def test_import_rejects_nonregular_source_record(self) -> None:
        """A directory or device node cannot masquerade as canonical shard bytes."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a",)
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )
            _write(source, _result(context, "a"))
            record = next((source.root / "datasets").glob("*.json"))
            record.unlink()
            record.mkdir()

            with self.assertRaises(ValueError):
                target.import_validated_dataset(source, "a")

            self.assertEqual(dict(target.load_completed()), {})

    def test_import_rejects_source_target_path_alias(self) -> None:
        """Locking one root twice could deadlock or turn an import into replacement."""
        with TemporaryDirectory() as temporary:
            target = Exp417CheckpointStore(
                Path(temporary) / "target",
                _provenance(),
                ("a", "b"),
            )

            with self.assertRaisesRegex(ValueError, "alias"):
                target.import_validated_dataset(target, "a")

            self.assertEqual(dict(target.load_completed()), {})

    def test_import_rolls_back_exactly_after_record_publication_failure(
        self,
    ) -> None:
        """A committed target record without progress must be removed exactly once."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a",)
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )
            _write(source, _result(context, "a"))
            _write(target, _result(context, "b"))
            source_before = _checkpoint_tree_snapshot(source.root)
            progress_before = (target.root / "progress.json").read_bytes()
            digests_before = dict(target.completed_record_digests())
            target_record_before = target._dataset_path("b").read_bytes()
            original_publish = checkpoints._atomic_write_exclusive_at
            injected = False

            def publish_then_fail(
                directory_descriptor: int,
                name: str,
                data: bytes,
                *,
                recovery_root_descriptor: int,
            ) -> tuple[int, int]:
                nonlocal injected
                identity = original_publish(
                    directory_descriptor,
                    name,
                    data,
                    recovery_root_descriptor=recovery_root_descriptor,
                )
                if name == target._dataset_path("a").name and not injected:
                    injected = True
                    raise checkpoints._CommittedWriteError(
                        OSError("injected post-record publication failure"),
                        identity,
                    )
                return identity

            with (
                patch.object(
                    checkpoints,
                    "_atomic_write_exclusive_at",
                    side_effect=publish_then_fail,
                ),
                self.assertRaisesRegex(OSError, "post-record"),
            ):
                target.import_validated_dataset(source, "a")

            self.assertTrue(injected)
            self.assertEqual(
                _checkpoint_tree_snapshot(source.root), source_before
            )
            self.assertEqual(
                dict(target.completed_record_digests()), digests_before
            )
            self.assertEqual(
                (target.root / "progress.json").read_bytes(),
                progress_before,
            )
            self.assertEqual(
                target._dataset_path("b").read_bytes(),
                target_record_before,
            )
            self.assertFalse(target._dataset_path("a").exists())

    def test_import_rolls_back_exactly_after_progress_replacement_failure(
        self,
    ) -> None:
        """A committed progress replacement must restore prior progress and records."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a",)
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )
            _write(source, _result(context, "a"))
            _write(target, _result(context, "b"))
            source_before = _checkpoint_tree_snapshot(source.root)
            progress_before = (target.root / "progress.json").read_bytes()
            digests_before = dict(target.completed_record_digests())
            target_record_before = target._dataset_path("b").read_bytes()
            replace_progress = target._replace_owned_progress_at
            injected = False

            def replace_then_fail(
                root_descriptor: int,
                payload: bytes,
                expected_identity: tuple[int, int],
            ) -> tuple[int, int]:
                nonlocal injected
                identity = replace_progress(
                    root_descriptor, payload, expected_identity
                )
                injected = True
                raise checkpoints._CommittedWriteError(
                    OSError("injected post-progress replacement failure"),
                    identity,
                )

            with (
                patch.object(
                    target,
                    "_replace_owned_progress_at",
                    side_effect=replace_then_fail,
                ),
                self.assertRaisesRegex(OSError, "post-progress"),
            ):
                target.import_validated_dataset(source, "a")

            self.assertTrue(injected)
            self.assertEqual(
                _checkpoint_tree_snapshot(source.root), source_before
            )
            self.assertEqual(
                dict(target.completed_record_digests()), digests_before
            )
            self.assertEqual(
                (target.root / "progress.json").read_bytes(),
                progress_before,
            )
            self.assertEqual(
                target._dataset_path("b").read_bytes(),
                target_record_before,
            )
            self.assertFalse(target._dataset_path("a").exists())

    def test_opposing_imports_finish_without_lock_order_deadlock(self) -> None:
        """Opposite source/target roles must acquire the same inode lock order."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            first = Exp417CheckpointStore(
                parent / "first", _provenance(), ("a", "b")
            )
            second = Exp417CheckpointStore(
                parent / "second", _provenance(), ("a", "b")
            )
            _write(first, _result(context, "a"))
            _write(second, _result(context, "b"))
            first_before = _checkpoint_tree_snapshot(first.root)
            second_before = _checkpoint_tree_snapshot(second.root)
            start = threading.Barrier(3)
            outcomes: dict[str, object] = {}

            def transfer(
                name: str,
                target: Exp417CheckpointStore,
                source: Exp417CheckpointStore,
                dataset: str,
            ) -> None:
                start.wait()
                try:
                    outcomes[name] = target.import_validated_dataset(
                        source, dataset
                    )
                except BaseException as error:
                    outcomes[name] = error

            threads = (
                threading.Thread(
                    target=transfer,
                    args=("first", first, second, "b"),
                    daemon=True,
                ),
                threading.Thread(
                    target=transfer,
                    args=("second", second, first, "a"),
                    daemon=True,
                ),
            )
            for thread in threads:
                thread.start()
            start.wait()
            for thread in threads:
                thread.join(timeout=5)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertTrue(
                all(isinstance(value, str) for value in outcomes.values()),
                outcomes,
            )
            self.assertEqual(tuple(first.load_completed()), ("a", "b"))
            self.assertEqual(tuple(second.load_completed()), ("a", "b"))
            self.assertEqual(
                first._dataset_path("a").read_bytes(),
                first_before[
                    str(first._dataset_path("a").relative_to(first.root))
                ][3],
            )
            self.assertEqual(
                second._dataset_path("b").read_bytes(),
                second_before[
                    str(second._dataset_path("b").relative_to(second.root))
                ][3],
            )

    def test_concurrent_imports_publish_one_duplicate_target(self) -> None:
        """Two valid sources racing for one target may commit exactly one record."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            sources = (
                Exp417CheckpointStore(
                    parent / "source-a", _provenance(), ("a",)
                ),
                Exp417CheckpointStore(
                    parent / "source-b", _provenance(), ("a",)
                ),
            )
            for source in sources:
                _write(source, _result(context, "a"))
            source_before = tuple(
                _checkpoint_tree_snapshot(source.root)
                for source in sources
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )
            start = threading.Barrier(3)
            outcomes: list[object] = []
            outcome_lock = threading.Lock()

            def transfer(source: Exp417CheckpointStore) -> None:
                start.wait()
                try:
                    outcome: object = target.import_validated_dataset(
                        source, "a"
                    )
                except BaseException as error:
                    outcome = error
                with outcome_lock:
                    outcomes.append(outcome)

            threads = tuple(
                threading.Thread(
                    target=transfer, args=(source,), daemon=True
                )
                for source in sources
            )
            for thread in threads:
                thread.start()
            start.wait()
            for thread in threads:
                thread.join(timeout=5)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(
                sum(isinstance(outcome, str) for outcome in outcomes), 1
            )
            failures = tuple(
                outcome
                for outcome in outcomes
                if isinstance(outcome, BaseException)
            )
            self.assertEqual(len(failures), 1)
            self.assertRegex(str(failures[0]), "already exists")
            self.assertEqual(tuple(target.load_completed()), ("a",))
            self.assertEqual(
                tuple(
                    _checkpoint_tree_snapshot(source.root)
                    for source in sources
                ),
                source_before,
            )

    def test_import_rejects_source_root_and_datasets_path_substitution(
        self,
    ) -> None:
        """A substituted source pathname cannot validate a pinned source inode."""
        context = _import_context()
        for substituted_entry in ("root", "datasets"):
            with self.subTest(
                substituted_entry=substituted_entry
            ), TemporaryDirectory() as temporary:
                parent = Path(temporary)
                source = Exp417CheckpointStore(
                    parent / "source", _provenance(), ("a",)
                )
                foreign = Exp417CheckpointStore(
                    parent / "foreign", _provenance(), ("a",)
                )
                target = Exp417CheckpointStore(
                    parent / "target", _provenance(), ("a", "b")
                )
                _write(source, _result(context, "a"))
                _write(foreign, _result(context, "a"))
                source_before = _checkpoint_tree_snapshot(source.root)
                foreign_before = _checkpoint_tree_snapshot(foreign.root)
                publish = checkpoints._atomic_write_exclusive_at
                swapped = False

                def substitute_then_publish(
                    directory_descriptor: int,
                    name: str,
                    data: bytes,
                    *,
                    recovery_root_descriptor: int,
                ) -> tuple[int, int]:
                    nonlocal swapped
                    if name == target._dataset_path("a").name and not swapped:
                        if substituted_entry == "root":
                            checkpoints._atomic_exchange(
                                source.root, foreign.root
                            )
                        else:
                            checkpoints._atomic_exchange(
                                source.root / "datasets",
                                foreign.root / "datasets",
                            )
                        swapped = True
                    return publish(
                        directory_descriptor,
                        name,
                        data,
                        recovery_root_descriptor=recovery_root_descriptor,
                    )

                try:
                    with (
                        patch.object(
                            checkpoints,
                            "_atomic_write_exclusive_at",
                            side_effect=substitute_then_publish,
                        ),
                        self.assertRaises(
                            (ValueError, BaseExceptionGroup)
                        ),
                    ):
                        target.import_validated_dataset(source, "a")
                finally:
                    if swapped:
                        if substituted_entry == "root":
                            checkpoints._atomic_exchange(
                                source.root, foreign.root
                            )
                        else:
                            checkpoints._atomic_exchange(
                                source.root / "datasets",
                                foreign.root / "datasets",
                            )

                self.assertTrue(swapped)
                self.assertEqual(dict(target.load_completed()), {})
                self.assertEqual(
                    _checkpoint_tree_snapshot(source.root), source_before
                )
                self.assertEqual(
                    _checkpoint_tree_snapshot(foreign.root), foreign_before
                )

    def test_import_revalidates_complete_source_snapshot_before_success(
        self,
    ) -> None:
        """Stable corruption outside the requested record must roll back the target."""
        context = _import_context()
        mutations = (
            "nonrequested_corruption",
            "nonrequested_identity",
            "root_inventory",
            "dataset_inventory",
            "provenance",
        )
        for mutation in mutations:
            with self.subTest(
                mutation=mutation
            ), TemporaryDirectory() as temporary:
                parent = Path(temporary)
                source = Exp417CheckpointStore(
                    parent / "source", _provenance(), ("a", "b")
                )
                target = Exp417CheckpointStore(
                    parent / "target", _provenance(), ("a", "b")
                )
                _write(source, _result(context, "a"))
                _write(source, _result(context, "b"))
                publish = checkpoints._atomic_write_exclusive_at
                mutated = False
                mutated_snapshot: dict[
                    str, tuple[int, int, int, bytes | None]
                ] | None = None

                def mutate_then_publish(
                    directory_descriptor: int,
                    name: str,
                    data: bytes,
                    *,
                    recovery_root_descriptor: int,
                ) -> tuple[int, int]:
                    nonlocal mutated, mutated_snapshot
                    if name == target._dataset_path("a").name and not mutated:
                        nonrequested = source._dataset_path("b")
                        if mutation == "nonrequested_corruption":
                            nonrequested.write_bytes(b"{")
                        elif mutation == "nonrequested_identity":
                            replacement = parent / "replacement.json"
                            replacement.write_bytes(nonrequested.read_bytes())
                            os.replace(replacement, nonrequested)
                        elif mutation == "root_inventory":
                            (source.root / "unexpected.bin").write_bytes(b"x")
                        elif mutation == "dataset_inventory":
                            (
                                source.root / "datasets" / "unexpected.json"
                            ).write_bytes(b"{}")
                        else:
                            replacement = parent / "provenance.json"
                            replacement.write_bytes(b'{"foreign":true}\n')
                            os.replace(
                                replacement,
                                source.root / "checkpoint_provenance.json",
                            )
                        mutated = True
                        mutated_snapshot = _checkpoint_tree_snapshot(
                            source.root
                        )
                    return publish(
                        directory_descriptor,
                        name,
                        data,
                        recovery_root_descriptor=recovery_root_descriptor,
                    )

                with (
                    patch.object(
                        checkpoints,
                        "_atomic_write_exclusive_at",
                        side_effect=mutate_then_publish,
                    ),
                    self.assertRaises(
                        (ValueError, BaseExceptionGroup)
                    ),
                ):
                    target.import_validated_dataset(source, "a")

                self.assertTrue(mutated)
                self.assertIsNotNone(mutated_snapshot)
                self.assertEqual(dict(target.load_completed()), {})
                self.assertEqual(
                    _checkpoint_tree_snapshot(source.root),
                    mutated_snapshot,
                    "import mutated the source while rejecting corruption",
                )

    def test_import_snapshot_rejects_root_inventory_hidden_from_validator(
        self,
    ) -> None:
        """A stable unexpected root entry cannot be hidden only during decode."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a",)
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )
            _write(source, _result(context, "a"))
            unexpected = source.root / "unexpected.bin"
            hidden = parent / "unexpected.hidden"
            unexpected.write_bytes(b"stable-unexpected-root-entry")
            source_before = _checkpoint_tree_snapshot(source.root)
            validate = source._load_records_from_descriptor

            def hide_inventory_during_validation(
                root_descriptor: int,
                datasets_descriptor: int | None = None,
            ):
                os.rename(unexpected, hidden)
                try:
                    return validate(root_descriptor, datasets_descriptor)
                finally:
                    os.rename(hidden, unexpected)

            with (
                patch.object(
                    source,
                    "_load_records_from_descriptor",
                    side_effect=hide_inventory_during_validation,
                ),
                self.assertRaises(ValueError),
            ):
                target.import_validated_dataset(source, "a")

            self.assertEqual(dict(target.load_completed()), {})
            self.assertEqual(
                _checkpoint_tree_snapshot(source.root), source_before
            )

    def test_import_snapshot_rejects_provenance_hidden_from_validator(
        self,
    ) -> None:
        """Canonical bytes shown only to the decoder cannot bless wrong provenance."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a",)
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )
            _write(source, _result(context, "a"))
            provenance = source.root / "checkpoint_provenance.json"
            canonical = provenance.read_bytes()
            wrong = b'{"wrong_provenance":true}\n'
            provenance.write_bytes(wrong)
            source_before = _checkpoint_tree_snapshot(source.root)
            validate = source._load_records_from_descriptor

            def show_canonical_during_validation(
                root_descriptor: int,
                datasets_descriptor: int | None = None,
            ):
                provenance.write_bytes(canonical)
                try:
                    return validate(root_descriptor, datasets_descriptor)
                finally:
                    provenance.write_bytes(wrong)

            with (
                patch.object(
                    source,
                    "_load_records_from_descriptor",
                    side_effect=show_canonical_during_validation,
                ),
                self.assertRaises(ValueError),
            ):
                target.import_validated_dataset(source, "a")

            self.assertEqual(dict(target.load_completed()), {})
            self.assertEqual(
                _checkpoint_tree_snapshot(source.root), source_before
            )

    def test_successful_import_never_mutates_source_tree(self) -> None:
        """A validated import is a read-only operation on every source entry."""
        context = _import_context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = Exp417CheckpointStore(
                parent / "source", _provenance(), ("a", "b")
            )
            target = Exp417CheckpointStore(
                parent / "target", _provenance(), ("a", "b")
            )
            _write(source, _result(context, "a"))
            _write(source, _result(context, "b"))
            source_before = _checkpoint_tree_snapshot(source.root)

            target.import_validated_dataset(source, "a")

            self.assertEqual(
                _checkpoint_tree_snapshot(source.root), source_before
            )

    def test_complete_snapshot_returns_one_sorted_read_only_atomic_pair(self) -> None:
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            _write(store, _result(context))

            snapshot = getattr(store, "require_complete_snapshot", None)
            self.assertIsNotNone(snapshot)
            completed, digests = snapshot()
            self.assertEqual(tuple(completed), ("synthetic-0000",))
            self.assertEqual(tuple(digests), ("synthetic-0000",))
            self.assertEqual(completed["synthetic-0000"], _result(context))
            self.assertEqual(
                digests, store.completed_record_digests()
            )
            with self.assertRaises(TypeError):
                completed["synthetic-0000"] = _result(context)
            with self.assertRaises(TypeError):
                digests["synthetic-0000"] = "f" * 64

    def test_complete_snapshot_binds_reads_to_locked_root_across_path_aba(self) -> None:
        """A pathname exchange-and-restore cannot substitute another complete root."""
        context = _context()
        original = _result(context)
        changed_group = replace(
            original.groups[0], pooled_synthetic_sensitivity=0.71
        )
        changed_contract = replace(
            original.decision_evidence.frozen_train_contract,
            groups=(changed_group,),
        )
        changed_evidence = replace(
            original.decision_evidence,
            frozen_train_contract=changed_contract,
        )
        changed = replace(
            original,
            groups=(changed_group,),
            decision_evidence=changed_evidence,
        )
        with TemporaryDirectory() as temporary:
            root_a = Path(temporary) / "checkpoint-a"
            root_b = Path(temporary) / "checkpoint-b"
            store_a = Exp417CheckpointStore(
                root_a, _provenance(), ("synthetic-0000",)
            )
            store_b = Exp417CheckpointStore(
                root_b, _provenance(), ("synthetic-0000",)
            )
            _write(store_a, original)
            _write(store_b, changed)
            expected_results = {"synthetic-0000": original}
            expected_digests = dict(store_a.completed_record_digests())
            foreign_digests = dict(store_b.completed_record_digests())
            self.assertNotEqual(expected_digests, foreign_digests)

            load_records = store_a._load_records_from_descriptor
            read_progress = store_a._read_progress_from_descriptor
            swapped = False
            restored = False

            def load_while_path_is_substituted(
                root_descriptor, datasets_descriptor
            ):
                nonlocal swapped
                checkpoints._atomic_exchange(root_a, root_b)
                swapped = True
                return load_records(root_descriptor, datasets_descriptor)

            def read_progress_then_restore(root_descriptor):
                nonlocal restored
                progress = read_progress(root_descriptor)
                checkpoints._atomic_exchange(root_a, root_b)
                restored = True
                return progress

            with (
                patch.object(
                    store_a,
                    "_load_records_from_descriptor",
                    side_effect=load_while_path_is_substituted,
                ),
                patch.object(
                    store_a,
                    "_read_progress_from_descriptor",
                    side_effect=read_progress_then_restore,
                ),
            ):
                completed, digests = store_a.require_complete_snapshot()

            self.assertTrue(swapped)
            self.assertTrue(restored)
            self.assertEqual(dict(completed), expected_results)
            self.assertEqual(dict(digests), expected_digests)
            self.assertNotEqual(dict(digests), foreign_digests)

    def test_constructor_validation_cannot_quarantine_from_path_aba(self) -> None:
        """Only errors read through the locked root descriptor may quarantine."""
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root_a = parent / "checkpoint-a"
            root_b = parent / "checkpoint-b"
            expected = ("synthetic-0000",)
            Exp417CheckpointStore(root_a, _provenance(), expected)
            Exp417CheckpointStore(root_b, _provenance(), expected)
            foreign_artifact = root_b / "foreign-crash-artifact"
            foreign_bytes = b"foreign-root-artifact"
            foreign_artifact.write_bytes(foreign_bytes)
            load_records = Exp417CheckpointStore._load_records
            swapped = False

            def load_from_substituted_path(store):
                nonlocal swapped
                if store.root == root_a:
                    checkpoints._atomic_exchange(root_a, root_b)
                    swapped = True
                    try:
                        return load_records(store)
                    finally:
                        checkpoints._atomic_exchange(root_a, root_b)
                return load_records(store)

            with patch.object(
                Exp417CheckpointStore,
                "_load_records",
                side_effect=load_from_substituted_path,
                autospec=True,
            ):
                reopened = Exp417CheckpointStore(
                    root_a, _provenance(), expected
                )

            self.assertFalse(
                swapped,
                "constructor validation still read checkpoint state by pathname",
            )
            self.assertTrue(root_a.is_dir())
            self.assertEqual(dict(reopened.load_completed()), {})
            self.assertEqual(foreign_artifact.read_bytes(), foreign_bytes)
            self.assertEqual(
                tuple(parent.glob(".checkpoint-a.quarantine.*")),
                (),
            )

    def test_constructor_rejects_final_root_basename_substitution(self) -> None:
        """Successful descriptor reads still rebind the official root pathname."""
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root_a = parent / "checkpoint-a"
            root_b = parent / "checkpoint-b"
            expected = ("synthetic-0000",)
            Exp417CheckpointStore(root_a, _provenance(), expected)
            Exp417CheckpointStore(
                root_b,
                replace(_provenance(), git_commit="d" * 40),
                expected,
            )
            read_progress = (
                Exp417CheckpointStore._read_progress_from_descriptor
            )
            swapped = False

            def read_then_substitute_root(
                store, root_descriptor: int
            ):
                nonlocal swapped
                progress = read_progress(store, root_descriptor)
                if store.root == root_a and not swapped:
                    checkpoints._atomic_exchange(root_a, root_b)
                    swapped = True
                return progress

            with (
                patch.object(
                    Exp417CheckpointStore,
                    "_read_progress_from_descriptor",
                    side_effect=read_then_substitute_root,
                    autospec=True,
                ),
                self.assertRaisesRegex(
                    ValueError, "root identity changed"
                ),
            ):
                Exp417CheckpointStore(
                    root_a, _provenance(), expected
                )

            self.assertTrue(swapped)
            self.assertTrue(root_a.is_dir())
            self.assertTrue(root_b.is_dir())
            self.assertEqual(
                tuple(parent.glob(".checkpoint-a.quarantine.*")),
                (),
            )

    def test_every_provenance_field_rejects_resume_before_dataset_load(self) -> None:
        context = _context()
        fields = {
            "git_commit": "d" * 40,
            "database_sha256": "f" * 64,
            "exp291_prediction_sha256": "f" * 64,
            "exp291_manifest_sha256": "f" * 64,
            "exp292_prediction_sha256": "f" * 64,
            "exp292_manifest_sha256": "f" * 64,
            "frozen_config_sha256": "f" * 64,
            "preprocessing_contract_sha256": "f" * 64,
            "fold_contract_sha256": "f" * 64,
            "model_contract_sha256": "f" * 64,
            "model_seeds": (1, 2, 3),
            "perturbation_seeds": (4, 5, 6),
            "runtime_versions": {"python": "other"},
            "backend_contract_sha256": "f" * 64,
        }
        for field, changed in fields.items():
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                root = Path(temporary) / "checkpoint"
                store = Exp417CheckpointStore(root, _provenance(), context.datasets)
                _write(store, _result(context))
                with patch.object(Exp417CheckpointStore, "_load_records", side_effect=AssertionError("records loaded")):
                    with self.assertRaisesRegex(ValueError, "provenance"):
                        Exp417CheckpointStore(root, replace(_provenance(), **{field: changed}), context.datasets)
                self.assertTrue(root.is_dir())
                self.assertEqual(len(store.load_completed()), 1)

    def test_rejects_duplicate_outside_universe_and_nonfinite_write_without_record(self) -> None:
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            result = _result(context)
            _write(store, result)
            with self.assertRaisesRegex(ValueError, "already exists"):
                _write(store, result)
            outside_store = Exp417CheckpointStore(
                Path(temporary) / "outside", _provenance(), ("synthetic-0001",)
            )
            with self.assertRaisesRegex(ValueError, "outside"):
                outside_store.write_dataset(result, **_write_arguments())
            with self.assertRaisesRegex(ValueError, "elapsed"):
                store.write_dataset(result, **_write_arguments(elapsed_seconds=float("nan")))
            self.assertEqual(len(store.load_completed()), 1)

    def test_corrupt_record_schema_digest_candidate_lane_and_inventory_fail_closed(self) -> None:
        context = _context()
        mutations = ("schema", "digest", "candidate", "extra")
        for mutation in mutations:
            with self.subTest(mutation=mutation), TemporaryDirectory() as temporary:
                root = Path(temporary) / "checkpoint"
                store = Exp417CheckpointStore(root, _provenance(), context.datasets)
                _write(store, _result(context))
                path = next((root / "datasets").glob("*.json"))
                if mutation == "extra":
                    (root / "datasets" / "unexpected.txt").write_text("x", encoding="utf-8")
                else:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if mutation == "schema":
                        payload["schema_version"] = True
                    elif mutation == "digest":
                        payload["payload_sha256"] = "0" * 64
                    else:
                        payload["candidate_coordinates"] = [["lane-hard", 0]]
                        payload["payload_sha256"] = _digest(payload)
                    path.write_bytes(canonical_json_bytes(payload) + b"\n")
                with self.assertRaises(ValueError):
                    store.load_completed()

    def test_symlink_truncation_and_write_failure_never_yield_a_completed_record(self) -> None:
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            record = _result(context)
            with patch("experiments.exp417_causal_multihorizon_shadow.checkpoints.os.fsync", side_effect=OSError("injected")):
                with self.assertRaises((OSError, BaseExceptionGroup)):
                    _write(store, record)
            self.assertEqual(tuple((root / "datasets").iterdir()), ())
            _write(store, record)
            path = next((root / "datasets").glob("*.json"))
            path.write_bytes(b"{")
            with self.assertRaises(ValueError):
                store.load_completed()

    def test_every_publish_boundary_failure_rolls_back_record_and_progress(self) -> None:
        context = _context()
        # Record write, datasets directory, progress capture source/parent,
        # progress write, and final root fsync all have rollback under the
        # directory transaction lock.
        for fail_at in range(1, 7):
            with self.subTest(fail_at=fail_at), TemporaryDirectory() as temporary:
                root = Path(temporary) / "checkpoint"
                store = Exp417CheckpointStore(root, _provenance(), context.datasets)
                calls = 0
                original_fsync = os.fsync

                def fail_once(descriptor: int) -> None:
                    nonlocal calls
                    calls += 1
                    if calls == fail_at:
                        raise OSError(f"injected fsync {fail_at}")
                    original_fsync(descriptor)

                with patch.object(checkpoints.os, "fsync", side_effect=fail_once):
                    with self.assertRaises(OSError):
                        _write(store, _result(context))
                self.assertEqual(tuple((root / "datasets").iterdir()), ())
                self.assertEqual(dict(store.load_completed()), {})
                self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_rollback_continues_after_an_existence_probe_failure(self) -> None:
        """One cleanup probe error cannot strand a published record."""
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            datasets_identity = (
                (root / "datasets").stat().st_dev,
                (root / "datasets").stat().st_ino,
            )
            original_fsync = os.fsync
            original_exists = checkpoints._entry_exists_at
            publish_failed = False
            probe_failed = False

            def fail_after_record_publish(descriptor: int) -> None:
                nonlocal publish_failed
                metadata = os.fstat(descriptor)
                if (
                    not publish_failed
                    and (metadata.st_dev, metadata.st_ino)
                    == datasets_identity
                    and tuple((root / "datasets").iterdir())
                ):
                    publish_failed = True
                    raise OSError("injected datasets publish fsync failure")
                original_fsync(descriptor)

            def fail_first_rollback_probe(
                descriptor: int, name: str
            ) -> bool:
                nonlocal probe_failed
                if publish_failed and not probe_failed:
                    probe_failed = True
                    raise OSError("injected rollback existence probe failure")
                return original_exists(descriptor, name)

            with (
                patch.object(
                    checkpoints.os,
                    "fsync",
                    side_effect=fail_after_record_publish,
                ),
                patch.object(
                    checkpoints,
                    "_entry_exists_at",
                    side_effect=fail_first_rollback_probe,
                ),
                self.assertRaises(BaseExceptionGroup),
            ):
                _write(store, _result(context))

            self.assertTrue(publish_failed)
            self.assertTrue(probe_failed)
            self.assertEqual(tuple((root / "datasets").iterdir()), ())
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_progress_replace_failure_stale_transaction_or_orphan_record_fail_closed(self) -> None:
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            original_rename = checkpoints._atomic_rename_noreplace_at
            failed = False

            def reject_progress(
                source_descriptor: int,
                source_name: str,
                destination_descriptor: int,
                destination_name: str,
            ) -> None:
                nonlocal failed
                if destination_name == "progress.json" and not failed:
                    failed = True
                    raise OSError("injected progress replace")
                original_rename(
                    source_descriptor,
                    source_name,
                    destination_descriptor,
                    destination_name,
                )

            with patch.object(
                checkpoints,
                "_atomic_rename_noreplace_at",
                side_effect=reject_progress,
            ):
                with self.assertRaises(OSError):
                    _write(store, _result(context))
            self.assertEqual(tuple((root / "datasets").iterdir()), ())
            self.assertEqual(store.progress()["completed_datasets"], 0)
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            original_write = checkpoints._atomic_write_at
            failed = False

            def reject_progress_write(
                descriptor: int,
                name: str,
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal failed
                if name == "progress.json" and not failed:
                    failed = True
                    raise OSError("injected progress temp/write")
                return original_write(descriptor, name, *args, **kwargs)

            with patch.object(
                checkpoints,
                "_atomic_write_at",
                side_effect=reject_progress_write,
            ):
                with self.assertRaises(OSError):
                    _write(store, _result(context))
            self.assertEqual(tuple((root / "datasets").iterdir()), ())
            self.assertEqual(store.progress()["completed_datasets"], 0)
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            _write(store, _result(context))
            progress = root / "progress.json"
            payload = json.loads(progress.read_text(encoding="utf-8"))
            payload["completed_datasets"] = 0
            payload["completed_record_digests"] = {}
            progress.write_bytes(canonical_json_bytes(payload) + b"\n")
            with self.assertRaises(ValueError):
                store.load_completed()
            (root / ".checkpoint-transaction.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                Exp417CheckpointStore(root, _provenance(), context.datasets)

    def test_writer_lock_precedes_progress_snapshot_and_prevents_stale_rollback(self) -> None:
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            first = Exp417CheckpointStore(root, _provenance(), context.datasets)
            second = Exp417CheckpointStore(root, _provenance(), context.datasets)
            stale_snapshot_captured = threading.Event()
            release_first = threading.Event()
            second_finished = threading.Event()
            original_read = checkpoints._read_regular_entry_at
            original_publish = checkpoints._atomic_write_exclusive_at
            progress_reads = 0
            outcomes: dict[str, object] = {}

            def interleave_progress(
                descriptor: int, name: str, context_name: str
            ) -> tuple[bytes, tuple[int, int]]:
                nonlocal progress_reads
                raw = original_read(descriptor, name, context_name)
                if (
                    threading.current_thread().name == "checkpoint-writer-a"
                    and name == "progress.json"
                ):
                    progress_reads += 1
                    if progress_reads == 1:
                        stale_snapshot_captured.set()
                        if not release_first.wait(timeout=5):
                            raise RuntimeError("test did not release first writer")
                return raw

            def fail_first_publish(
                descriptor: int,
                name: str,
                data: bytes,
                *,
                recovery_root_descriptor: int,
            ) -> tuple[int, int]:
                if threading.current_thread().name == "checkpoint-writer-a":
                    raise OSError("injected first-writer publication failure")
                return original_publish(
                    descriptor,
                    name,
                    data,
                    recovery_root_descriptor=recovery_root_descriptor,
                )

            def writer_a() -> None:
                try:
                    _write(first, _result(context, "synthetic-0000"))
                except BaseException as error:
                    outcomes["a"] = error
                else:
                    outcomes["a"] = "ok"

            def writer_b() -> None:
                try:
                    _write(second, _result(context, "synthetic-0001"))
                except BaseException as error:
                    outcomes["b"] = error
                else:
                    outcomes["b"] = "ok"
                finally:
                    second_finished.set()

            with (
                patch.object(
                    checkpoints,
                    "_read_regular_entry_at",
                    side_effect=interleave_progress,
                ),
                patch.object(
                    checkpoints,
                    "_atomic_write_exclusive_at",
                    side_effect=fail_first_publish,
                ),
            ):
                thread_a = threading.Thread(target=writer_a, name="checkpoint-writer-a")
                thread_a.start()
                self.assertTrue(stale_snapshot_captured.wait(timeout=5))
                thread_b = threading.Thread(target=writer_b, name="checkpoint-writer-b")
                thread_b.start()
                second_was_serialized = not second_finished.wait(timeout=0.25)
                release_first.set()
                thread_a.join(timeout=5)
                thread_b.join(timeout=5)

            self.assertFalse(thread_a.is_alive())
            self.assertFalse(thread_b.is_alive())
            self.assertTrue(
                second_was_serialized,
                "second writer completed while the first writer held a stale progress snapshot",
            )
            self.assertIsInstance(outcomes["a"], OSError)
            self.assertEqual(outcomes["b"], "ok")
            self.assertEqual(tuple(first.load_completed()), ("synthetic-0001",))
            self.assertEqual(first.progress()["completed_datasets"], 1)

    def test_writer_transaction_binds_mutations_to_locked_root_across_path_aba(self) -> None:
        """A pathname exchange-and-restore cannot split record and progress writes."""
        context = _context()
        with TemporaryDirectory() as temporary:
            root_a = Path(temporary) / "checkpoint-a"
            root_b = Path(temporary) / "checkpoint-b"
            store_a = Exp417CheckpointStore(
                root_a, _provenance(), ("synthetic-0000",)
            )
            store_b = Exp417CheckpointStore(
                root_b, _provenance(), ("synthetic-0000",)
            )
            publish = checkpoints._atomic_write_exclusive_at
            swapped = False
            restored = False

            def publish_while_path_is_substituted(
                directory_descriptor: int,
                name: str,
                data: bytes,
                *,
                recovery_root_descriptor: int,
            ):
                nonlocal swapped, restored
                checkpoints._atomic_exchange(root_a, root_b)
                swapped = True
                try:
                    return publish(
                        directory_descriptor,
                        name,
                        data,
                        recovery_root_descriptor=recovery_root_descriptor,
                    )
                finally:
                    checkpoints._atomic_exchange(root_a, root_b)
                    restored = True

            with patch.object(
                checkpoints,
                "_atomic_write_exclusive_at",
                side_effect=publish_while_path_is_substituted,
            ):
                _write(store_a, _result(context))

            self.assertTrue(swapped)
            self.assertTrue(restored)
            self.assertEqual(
                tuple(store_a.load_completed()), ("synthetic-0000",)
            )
            self.assertEqual(dict(store_b.load_completed()), {})

    def test_writer_rejects_substituted_datasets_directory_descriptor(self) -> None:
        """A child-directory ABA cannot bind records outside the locked root."""
        context = _context()
        with TemporaryDirectory() as temporary:
            root_a = Path(temporary) / "checkpoint-a"
            root_b = Path(temporary) / "checkpoint-b"
            store_a = Exp417CheckpointStore(
                root_a, _provenance(), ("synthetic-0000",)
            )
            store_b = Exp417CheckpointStore(
                root_b, _provenance(), ("synthetic-0000",)
            )
            original_open = checkpoints.os.open
            swaps = 0

            def open_with_substituted_child(
                path, flags, *args, **kwargs
            ):
                nonlocal swaps
                if path == "datasets" and kwargs.get("dir_fd") is not None:
                    checkpoints._atomic_exchange(
                        root_a / "datasets", root_b / "datasets"
                    )
                    swaps += 1
                    try:
                        return original_open(path, flags, *args, **kwargs)
                    finally:
                        checkpoints._atomic_exchange(
                            root_a / "datasets", root_b / "datasets"
                        )
                return original_open(path, flags, *args, **kwargs)

            with (
                patch.object(
                    checkpoints.os,
                    "open",
                    side_effect=open_with_substituted_child,
                ),
                self.assertRaisesRegex(ValueError, "datasets.*(identity|changed)"),
            ):
                _write(store_a, _result(context))

            self.assertGreater(swaps, 0)
            self.assertEqual(dict(store_a.load_completed()), {})
            self.assertEqual(dict(store_b.load_completed()), {})

    def test_rollback_preserves_a_foreign_replacement_of_owned_record(self) -> None:
        """Rollback must never unlink a record basename after its inode changes."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            record_name = sha256(
                b"synthetic-0000"
            ).hexdigest() + ".json"
            target = root / "datasets" / record_name
            owned_saved = parent / "owned-saved.json"
            foreign = parent / "foreign.bin"
            foreign_bytes = b"foreign-checkpoint-replacement"
            foreign.write_bytes(foreign_bytes)
            replace_progress = checkpoints._atomic_write_at
            substituted = False

            def substitute_before_progress(
                descriptor: int,
                name: str,
                data: bytes,
                *,
                replace: bool,
                recovery_root_descriptor: int,
            ) -> None:
                nonlocal substituted
                if name == "progress.json" and not substituted:
                    substituted = True
                    os.rename(target, owned_saved)
                    os.rename(foreign, target)
                    raise OSError("injected progress failure after substitution")
                replace_progress(
                    descriptor,
                    name,
                    data,
                    replace=replace,
                    recovery_root_descriptor=recovery_root_descriptor,
                )

            with (
                patch.object(
                    checkpoints,
                    "_atomic_write_at",
                    side_effect=substitute_before_progress,
                ),
                self.assertRaises((OSError, BaseExceptionGroup)),
            ):
                _write(store, _result(context))

            self.assertTrue(substituted)
            self.assertTrue(owned_saved.is_file())
            preserved = tuple(
                path
                for path in parent.rglob("*")
                if path.is_file() and path.read_bytes() == foreign_bytes
            )
            self.assertTrue(
                preserved,
                "rollback deleted the foreign replacement instead of preserving it",
            )
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_rollback_preserves_a_foreign_replacement_of_progress(self) -> None:
        """Progress rollback must capture, never overwrite, a foreign replacement."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            progress = root / "progress.json"
            owned_saved = parent / "owned-progress-saved.json"
            foreign = parent / "foreign-progress.bin"
            foreign_bytes = b"foreign-progress-replacement"
            foreign.write_bytes(foreign_bytes)
            original_fsync = os.fsync
            substituted = False
            root_identity = (
                root.stat().st_dev,
                root.stat().st_ino,
            )

            def substitute_at_root_fsync(descriptor: int) -> None:
                nonlocal substituted
                metadata = os.fstat(descriptor)
                is_root = (
                    metadata.st_dev,
                    metadata.st_ino,
                ) == root_identity
                if (
                    not substituted
                    and is_root
                    and progress.is_file()
                    and json.loads(progress.read_text(encoding="utf-8"))[
                        "completed_datasets"
                    ] == 1
                ):
                    os.rename(progress, owned_saved)
                    os.rename(foreign, progress)
                    substituted = True
                    raise OSError("injected final root fsync failure")
                original_fsync(descriptor)

            with (
                patch.object(
                    checkpoints.os,
                    "fsync",
                    side_effect=substitute_at_root_fsync,
                ),
                self.assertRaises((OSError, BaseExceptionGroup)),
            ):
                _write(store, _result(context))

            self.assertTrue(substituted)
            self.assertTrue(owned_saved.is_file())
            preserved = tuple(
                path
                for path in parent.rglob("*")
                if path.is_file() and path.read_bytes() == foreign_bytes
            )
            self.assertTrue(
                preserved,
                "rollback overwrote the foreign progress replacement",
            )
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_record_publish_has_no_post_link_target_unlink_window(self) -> None:
        """A successful no-replace publish must need no unsafe target cleanup."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            record_name = sha256(
                b"synthetic-0000"
            ).hexdigest() + ".json"
            target = root / "datasets" / record_name
            owned_saved = parent / "owned-publish-saved.json"
            foreign = parent / "foreign-publish.bin"
            foreign_bytes = b"foreign-publish-replacement"
            foreign.write_bytes(foreign_bytes)
            original_unlink = os.unlink
            injected = False

            def reject_post_link_temp_unlink(
                path, *args, **kwargs
            ) -> None:
                nonlocal injected
                if (
                    not injected
                    and isinstance(path, str)
                    and path.startswith(".checkpoint-")
                    and target.exists()
                ):
                    os.rename(target, owned_saved)
                    os.rename(foreign, target)
                    injected = True
                    raise OSError("injected post-link temp unlink failure")
                original_unlink(path, *args, **kwargs)

            with patch.object(
                checkpoints.os,
                "unlink",
                side_effect=reject_post_link_temp_unlink,
            ):
                _write(store, _result(context))

            self.assertFalse(
                injected,
                "record publication exposed a post-link cleanup window",
            )
            self.assertEqual(foreign.read_bytes(), foreign_bytes)
            self.assertEqual(
                tuple(store.load_completed()), ("synthetic-0000",)
            )

    def test_temp_write_failure_preserves_a_substituted_foreign_inode(self) -> None:
        """Temporary cleanup must not unlink an unowned replacement basename."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            owned_saved = parent / "owned-temporary-saved.bin"
            foreign = parent / "foreign-temporary.bin"
            foreign_bytes = b"foreign-temporary-replacement"
            foreign.write_bytes(foreign_bytes)
            original_open = os.open
            original_fsync = os.fsync
            temporary_entry: tuple[int, str, int] | None = None
            substituted = False

            def observe_temporary_open(
                path, flags, *args, **kwargs
            ) -> int:
                nonlocal temporary_entry
                descriptor = original_open(path, flags, *args, **kwargs)
                if (
                    temporary_entry is None
                    and isinstance(path, str)
                    and path.startswith(".checkpoint-")
                    and kwargs.get("dir_fd") is not None
                ):
                    temporary_entry = (
                        kwargs["dir_fd"],
                        path,
                        descriptor,
                    )
                return descriptor

            def substitute_before_temp_fsync(descriptor: int) -> None:
                nonlocal substituted
                if (
                    not substituted
                    and temporary_entry is not None
                    and descriptor == temporary_entry[2]
                ):
                    directory_descriptor, name, _ = temporary_entry
                    os.rename(
                        name,
                        owned_saved,
                        src_dir_fd=directory_descriptor,
                    )
                    os.rename(
                        foreign,
                        name,
                        dst_dir_fd=directory_descriptor,
                    )
                    substituted = True
                    raise OSError("injected temporary fsync failure")
                original_fsync(descriptor)

            with (
                patch.object(
                    checkpoints.os,
                    "open",
                    side_effect=observe_temporary_open,
                ),
                patch.object(
                    checkpoints.os,
                    "fsync",
                    side_effect=substitute_before_temp_fsync,
                ),
                self.assertRaises((OSError, BaseExceptionGroup)),
            ):
                _write(store, _result(context))

            self.assertTrue(substituted)
            self.assertTrue(owned_saved.is_file())
            self.assertTrue(
                any(
                    path.is_file() and path.read_bytes() == foreign_bytes
                    for path in parent.rglob("*")
                ),
                "temporary cleanup deleted the foreign replacement",
            )
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_temp_close_failure_preserves_the_owned_entry(self) -> None:
        """A post-close error cannot leave an orphan inside the checkpoint."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            original_open = os.open
            original_close = os.close
            temporary_descriptor: int | None = None
            injected = False

            def observe_temporary_open(
                path, flags, *args, **kwargs
            ) -> int:
                nonlocal temporary_descriptor
                descriptor = original_open(path, flags, *args, **kwargs)
                if (
                    temporary_descriptor is None
                    and isinstance(path, str)
                    and path.startswith(".checkpoint-")
                    and kwargs.get("dir_fd") is not None
                ):
                    temporary_descriptor = descriptor
                return descriptor

            def close_then_fail(descriptor: int) -> None:
                nonlocal injected
                if descriptor == temporary_descriptor and not injected:
                    original_close(descriptor)
                    injected = True
                    raise OSError("injected after real temporary close")
                original_close(descriptor)

            with (
                patch.object(
                    checkpoints.os,
                    "open",
                    side_effect=observe_temporary_open,
                ),
                patch.object(
                    checkpoints.os,
                    "close",
                    side_effect=close_then_fail,
                ),
                self.assertRaises((OSError, BaseExceptionGroup)),
            ):
                _write(store, _result(context))

            self.assertTrue(injected)
            self.assertEqual(tuple((root / "datasets").iterdir()), ())
            self.assertTrue(
                tuple(parent.glob(
                    ".exp417-checkpoint.rollback-temporary.*.bin"
                ))
            )
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_publish_cleanup_continues_after_temp_probe_failure(self) -> None:
        """A failed temp probe cannot strand an uncommitted entry."""
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            original_rename = checkpoints._atomic_rename_noreplace_at
            original_exists = checkpoints._entry_exists_at
            publish_failed = False
            probe_failed = False

            def fail_record_publish(
                source_descriptor: int,
                source_name: str,
                destination_descriptor: int,
                destination_name: str,
            ) -> None:
                nonlocal publish_failed
                if (
                    not publish_failed
                    and destination_name.endswith(".json")
                    and destination_name
                    not in {
                        "checkpoint_provenance.json",
                        "progress.json",
                    }
                ):
                    publish_failed = True
                    raise OSError("injected pre-commit record rename failure")
                original_rename(
                    source_descriptor,
                    source_name,
                    destination_descriptor,
                    destination_name,
                )

            def fail_source_probe(
                descriptor: int, name: str
            ) -> bool:
                nonlocal probe_failed
                if (
                    publish_failed
                    and not probe_failed
                    and name.startswith(".checkpoint-")
                ):
                    probe_failed = True
                    raise OSError("injected temporary existence probe failure")
                return original_exists(descriptor, name)

            with (
                patch.object(
                    checkpoints,
                    "_atomic_rename_noreplace_at",
                    side_effect=fail_record_publish,
                ),
                patch.object(
                    checkpoints,
                    "_entry_exists_at",
                    side_effect=fail_source_probe,
                ),
                self.assertRaises(BaseExceptionGroup),
            ):
                _write(store, _result(context))

            self.assertTrue(publish_failed)
            self.assertTrue(probe_failed)
            self.assertEqual(tuple((root / "datasets").iterdir()), ())
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_publish_cleanup_handles_missing_temp_and_foreign_target(self) -> None:
        """Combined target/source races preserve the target and restore state."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            owned_saved = parent / "owned-temp-removed-before-probe.bin"
            foreign = parent / "foreign-canonical-target.bin"
            foreign_bytes = b"foreign-canonical-target"
            foreign.write_bytes(foreign_bytes)
            original_rename = checkpoints._atomic_rename_noreplace_at
            original_exists = checkpoints._entry_exists_at
            substituted = False
            probe_failed = False

            def substitute_foreign_target(
                source_descriptor: int,
                source_name: str,
                destination_descriptor: int,
                destination_name: str,
            ) -> None:
                nonlocal substituted
                if (
                    not substituted
                    and destination_name.endswith(".json")
                    and destination_name
                    not in {
                        "checkpoint_provenance.json",
                        "progress.json",
                    }
                ):
                    os.rename(
                        source_name,
                        owned_saved,
                        src_dir_fd=source_descriptor,
                    )
                    os.rename(
                        foreign,
                        destination_name,
                        dst_dir_fd=destination_descriptor,
                    )
                    substituted = True
                    raise OSError("injected combined publish race")
                original_rename(
                    source_descriptor,
                    source_name,
                    destination_descriptor,
                    destination_name,
                )

            def fail_missing_source_probe(
                descriptor: int, name: str
            ) -> bool:
                nonlocal probe_failed
                if (
                    substituted
                    and not probe_failed
                    and name.startswith(".checkpoint-")
                ):
                    probe_failed = True
                    raise OSError("injected missing-source probe failure")
                return original_exists(descriptor, name)

            with (
                patch.object(
                    checkpoints,
                    "_atomic_rename_noreplace_at",
                    side_effect=substitute_foreign_target,
                ),
                patch.object(
                    checkpoints,
                    "_entry_exists_at",
                    side_effect=fail_missing_source_probe,
                ),
                self.assertRaises(BaseExceptionGroup),
            ):
                _write(store, _result(context))

            self.assertTrue(substituted)
            self.assertTrue(probe_failed)
            self.assertTrue(owned_saved.is_file())
            self.assertTrue(any(
                path.is_file() and path.read_bytes() == foreign_bytes
                for path in parent.rglob("*")
            ))
            self.assertEqual(tuple((root / "datasets").iterdir()), ())
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_publish_error_captures_a_competing_canonical_target(self) -> None:
        """A no-replace competitor is preserved without poisoning resume."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            foreign = parent / "foreign-precommit-target.bin"
            foreign_bytes = b"foreign-precommit-canonical-target"
            foreign.write_bytes(foreign_bytes)
            original_rename = checkpoints._atomic_rename_noreplace_at
            injected = False

            def insert_competing_target(
                source_descriptor: int,
                source_name: str,
                destination_descriptor: int,
                destination_name: str,
            ) -> None:
                nonlocal injected
                if (
                    not injected
                    and destination_name.endswith(".json")
                    and destination_name
                    not in {
                        "checkpoint_provenance.json",
                        "progress.json",
                    }
                ):
                    os.rename(
                        foreign,
                        destination_name,
                        dst_dir_fd=destination_descriptor,
                    )
                    injected = True
                original_rename(
                    source_descriptor,
                    source_name,
                    destination_descriptor,
                    destination_name,
                )

            with (
                patch.object(
                    checkpoints,
                    "_atomic_rename_noreplace_at",
                    side_effect=insert_competing_target,
                ),
                self.assertRaises(BaseException),
            ):
                _write(store, _result(context))

            self.assertTrue(injected)
            self.assertTrue(any(
                path.is_file() and path.read_bytes() == foreign_bytes
                for path in parent.rglob("*")
            ))
            self.assertEqual(tuple((root / "datasets").iterdir()), ())
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_publish_cleanup_survives_target_stat_failure_after_commit(self) -> None:
        """A target-stat error cannot strand an actually committed record."""
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            original_rename = checkpoints._atomic_rename_noreplace_at
            original_identity = checkpoints._regular_identity_at
            renamed = False
            stat_failed = False

            def rename_then_fail(
                source_descriptor: int,
                source_name: str,
                destination_descriptor: int,
                destination_name: str,
            ) -> None:
                nonlocal renamed
                original_rename(
                    source_descriptor,
                    source_name,
                    destination_descriptor,
                    destination_name,
                )
                if (
                    not renamed
                    and destination_name.endswith(".json")
                    and destination_name
                    not in {
                        "checkpoint_provenance.json",
                        "progress.json",
                    }
                ):
                    renamed = True
                    raise OSError("injected after committed record rename")

            def fail_published_target_stat(
                descriptor: int, name: str, context_name: str
            ) -> tuple[int, int]:
                nonlocal stat_failed
                if (
                    renamed
                    and not stat_failed
                    and context_name
                    == "Exp417 checkpoint published record"
                ):
                    stat_failed = True
                    raise OSError("injected published-target stat failure")
                return original_identity(descriptor, name, context_name)

            with (
                patch.object(
                    checkpoints,
                    "_atomic_rename_noreplace_at",
                    side_effect=rename_then_fail,
                ),
                patch.object(
                    checkpoints,
                    "_regular_identity_at",
                    side_effect=fail_published_target_stat,
                ),
                self.assertRaises(BaseExceptionGroup),
            ):
                _write(store, _result(context))

            self.assertTrue(renamed)
            self.assertTrue(stat_failed)
            self.assertEqual(tuple((root / "datasets").iterdir()), ())
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_post_rename_baseexception_rolls_back_committed_record(self) -> None:
        """A BaseException after atomic rename must carry committed ownership."""
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            rename = checkpoints._atomic_rename_noreplace_at
            injected = False

            def rename_then_interrupt(*args, **kwargs) -> None:
                nonlocal injected
                rename(*args, **kwargs)
                if not injected:
                    injected = True
                    raise KeyboardInterrupt(
                        "injected after committed record rename"
                    )

            with (
                patch.object(
                    checkpoints,
                    "_atomic_rename_noreplace_at",
                    side_effect=rename_then_interrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                _write(store, _result(context))

            self.assertTrue(injected)
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_post_rename_baseexception_rolls_back_committed_progress(self) -> None:
        """Committed progress ownership also survives a post-rename interruption."""
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            rename = checkpoints._atomic_rename_noreplace_at
            injected = False

            def rename_then_interrupt(
                source_descriptor: int,
                source_name: str,
                destination_descriptor: int,
                destination_name: str,
            ) -> None:
                nonlocal injected
                rename(
                    source_descriptor,
                    source_name,
                    destination_descriptor,
                    destination_name,
                )
                if destination_name == "progress.json" and not injected:
                    injected = True
                    raise KeyboardInterrupt(
                        "injected after committed progress rename"
                    )

            with (
                patch.object(
                    checkpoints,
                    "_atomic_rename_noreplace_at",
                    side_effect=rename_then_interrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                _write(store, _result(context))

            self.assertTrue(injected)
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_progress_commit_preserves_a_preexisting_foreign_replacement(self) -> None:
        """Success-path progress publication must not overwrite an unowned inode."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            progress = root / "progress.json"
            owned_saved = parent / "owned-progress-before-commit.json"
            foreign = parent / "foreign-progress-before-commit.bin"
            foreign_bytes = b"foreign-before-progress-commit"
            foreign.write_bytes(foreign_bytes)
            progress_bytes = store._progress_bytes
            substituted = False

            def substitute_before_commit(*args, **kwargs) -> bytes:
                nonlocal substituted
                payload = progress_bytes(*args, **kwargs)
                if not substituted:
                    os.rename(progress, owned_saved)
                    os.rename(foreign, progress)
                    substituted = True
                return payload

            with (
                patch.object(
                    store,
                    "_progress_bytes",
                    side_effect=substitute_before_commit,
                ),
                self.assertRaises((ValueError, BaseExceptionGroup)),
            ):
                _write(store, _result(context))

            self.assertTrue(substituted)
            self.assertTrue(owned_saved.is_file())
            preserved = tuple(
                path
                for path in parent.rglob("*")
                if path.is_file() and path.read_bytes() == foreign_bytes
            )
            self.assertTrue(
                preserved,
                "success-path progress publication deleted foreign bytes",
            )
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_writer_rejects_record_substitution_before_success_return(self) -> None:
        """A final success check must bind the published record inode and bytes."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            record_name = sha256(
                b"synthetic-0000"
            ).hexdigest() + ".json"
            record = root / "datasets" / record_name
            progress = root / "progress.json"
            owned_saved = parent / "owned-record-before-return.json"
            foreign = parent / "foreign-record-before-return.bin"
            foreign_bytes = b"foreign-record-before-success-return"
            foreign.write_bytes(foreign_bytes)
            original_fsync = os.fsync
            root_identity = (root.stat().st_dev, root.stat().st_ino)
            substituted = False

            def substitute_at_final_root_fsync(descriptor: int) -> None:
                nonlocal substituted
                metadata = os.fstat(descriptor)
                if (
                    not substituted
                    and (metadata.st_dev, metadata.st_ino) == root_identity
                    and progress.is_file()
                    and json.loads(progress.read_text(encoding="utf-8"))[
                        "completed_datasets"
                    ] == 1
                ):
                    os.rename(record, owned_saved)
                    os.rename(foreign, record)
                    substituted = True
                original_fsync(descriptor)

            with (
                patch.object(
                    checkpoints.os,
                    "fsync",
                    side_effect=substitute_at_final_root_fsync,
                ),
                self.assertRaises((ValueError, BaseExceptionGroup)),
            ):
                _write(store, _result(context))

            self.assertTrue(substituted)
            self.assertTrue(owned_saved.is_file())
            self.assertTrue(any(
                path.is_file() and path.read_bytes() == foreign_bytes
                for path in parent.rglob("*")
            ))
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_writer_rejects_progress_substitution_before_success_return(self) -> None:
        """A final success check must bind the published progress inode and bytes."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            progress = root / "progress.json"
            owned_saved = parent / "owned-progress-before-return.json"
            foreign = parent / "foreign-progress-before-return.bin"
            foreign_bytes = b"foreign-progress-before-success-return"
            foreign.write_bytes(foreign_bytes)
            original_fsync = os.fsync
            root_identity = (root.stat().st_dev, root.stat().st_ino)
            substituted = False

            def substitute_at_final_root_fsync(descriptor: int) -> None:
                nonlocal substituted
                metadata = os.fstat(descriptor)
                if (
                    not substituted
                    and (metadata.st_dev, metadata.st_ino) == root_identity
                    and progress.is_file()
                    and json.loads(progress.read_text(encoding="utf-8"))[
                        "completed_datasets"
                    ] == 1
                ):
                    os.rename(progress, owned_saved)
                    os.rename(foreign, progress)
                    substituted = True
                original_fsync(descriptor)

            with (
                patch.object(
                    checkpoints.os,
                    "fsync",
                    side_effect=substitute_at_final_root_fsync,
                ),
                self.assertRaises((ValueError, BaseExceptionGroup)),
            ):
                _write(store, _result(context))

            self.assertTrue(substituted)
            self.assertTrue(owned_saved.is_file())
            self.assertTrue(any(
                path.is_file() and path.read_bytes() == foreign_bytes
                for path in parent.rglob("*")
            ))
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_writer_rejects_provenance_substitution_before_success_return(self) -> None:
        """A successful append must rebind immutable provenance and root inventory."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            provenance = root / "checkpoint_provenance.json"
            progress = root / "progress.json"
            owned_saved = parent / "owned-provenance-before-return.json"
            foreign = parent / "foreign-provenance-before-return.bin"
            foreign_bytes = b"foreign-provenance-before-success-return"
            foreign.write_bytes(foreign_bytes)
            original_fsync = os.fsync
            root_identity = (root.stat().st_dev, root.stat().st_ino)
            substituted = False

            def substitute_at_final_root_fsync(descriptor: int) -> None:
                nonlocal substituted
                metadata = os.fstat(descriptor)
                if (
                    not substituted
                    and (metadata.st_dev, metadata.st_ino) == root_identity
                    and progress.is_file()
                    and json.loads(progress.read_text(encoding="utf-8"))[
                        "completed_datasets"
                    ] == 1
                ):
                    os.rename(provenance, owned_saved)
                    os.rename(foreign, provenance)
                    substituted = True
                original_fsync(descriptor)

            with (
                patch.object(
                    checkpoints.os,
                    "fsync",
                    side_effect=substitute_at_final_root_fsync,
                ),
                self.assertRaises((ValueError, BaseExceptionGroup)),
            ):
                _write(store, _result(context))

            self.assertTrue(substituted)
            self.assertTrue(owned_saved.is_file())
            self.assertTrue(any(
                path.is_file() and path.read_bytes() == foreign_bytes
                for path in parent.rglob("*")
            ))
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_writer_rejects_record_name_swap_after_final_fstat(self) -> None:
        """The opened inode cannot validate a basename swapped after its read."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            record_name = sha256(
                b"synthetic-0000"
            ).hexdigest() + ".json"
            record = root / "datasets" / record_name
            owned_saved = parent / "owned-record-during-read.json"
            foreign = parent / "foreign-record-during-read.bin"
            foreign_bytes = b"foreign-record-during-final-read"
            foreign.write_bytes(foreign_bytes)
            original_fstat = os.fstat
            substituted = False
            record_fstats = 0

            def substitute_basename_after_final_fstat(
                descriptor: int,
            ) -> os.stat_result:
                nonlocal substituted, record_fstats
                metadata = original_fstat(descriptor)
                if not substituted and record.is_file():
                    named = os.stat(record, follow_symlinks=False)
                    if (
                        metadata.st_dev,
                        metadata.st_ino,
                    ) == (
                        named.st_dev,
                        named.st_ino,
                    ):
                        record_fstats += 1
                        if record_fstats == 2:
                            os.rename(record, owned_saved)
                            os.rename(foreign, record)
                            substituted = True
                return metadata

            with (
                patch.object(
                    checkpoints.os,
                    "fstat",
                    side_effect=substitute_basename_after_final_fstat,
                ),
                self.assertRaises((ValueError, BaseExceptionGroup)),
            ):
                _write(store, _result(context))

            self.assertTrue(substituted)
            self.assertTrue(owned_saved.is_file())
            self.assertTrue(any(
                path.is_file() and path.read_bytes() == foreign_bytes
                for path in parent.rglob("*")
            ))
            self.assertEqual(dict(store.load_completed()), {})
            self.assertEqual(store.progress()["completed_datasets"], 0)

    def test_reader_waits_for_exclusive_writer_and_observes_one_coherent_commit(self) -> None:
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            writer_store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            reader_store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            writer_entered_publish = threading.Event()
            release_writer = threading.Event()
            reader_finished = threading.Event()
            original_publish = checkpoints._atomic_write_exclusive_at
            outcomes: dict[str, object] = {}

            def pause_publish(
                descriptor: int,
                name: str,
                data: bytes,
                *,
                recovery_root_descriptor: int,
            ) -> tuple[int, int]:
                writer_entered_publish.set()
                if not release_writer.wait(timeout=5):
                    raise RuntimeError("test did not release writer")
                return original_publish(
                    descriptor,
                    name,
                    data,
                    recovery_root_descriptor=recovery_root_descriptor,
                )

            def writer() -> None:
                try:
                    _write(writer_store, _result(context))
                except BaseException as error:
                    outcomes["writer"] = error
                else:
                    outcomes["writer"] = "ok"

            def reader() -> None:
                try:
                    outcomes["reader"] = tuple(reader_store.load_completed())
                except BaseException as error:
                    outcomes["reader"] = error
                finally:
                    reader_finished.set()

            with patch.object(
                checkpoints,
                "_atomic_write_exclusive_at",
                side_effect=pause_publish,
            ):
                writer_thread = threading.Thread(target=writer)
                writer_thread.start()
                self.assertTrue(writer_entered_publish.wait(timeout=5))
                reader_thread = threading.Thread(target=reader)
                reader_thread.start()
                reader_was_serialized = not reader_finished.wait(timeout=0.25)
                release_writer.set()
                writer_thread.join(timeout=5)
                reader_thread.join(timeout=5)

            self.assertFalse(writer_thread.is_alive())
            self.assertFalse(reader_thread.is_alive())
            self.assertTrue(
                reader_was_serialized,
                "reader observed the checkpoint while its transaction was in flight",
            )
            self.assertEqual(outcomes["writer"], "ok")
            self.assertEqual(outcomes["reader"], ("synthetic-0000",))

    def test_reopen_quarantines_owned_crash_artifacts_and_preserves_every_byte(self) -> None:
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            Exp417CheckpointStore(root, _provenance(), context.datasets)
            (root / ".checkpoint-transaction.json").write_bytes(b'{"interrupted":true}\n')
            (root / ".checkpoint-orphan.tmp").write_bytes(b"partial-record")
            before = {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

            with self.assertRaisesRegex(ValueError, "quarantined"):
                Exp417CheckpointStore(root, _provenance(), context.datasets)

            self.assertFalse(root.exists())
            quarantines = tuple(parent.glob(".checkpoint.quarantine.*"))
            self.assertEqual(len(quarantines), 1)
            after = {
                str(path.relative_to(quarantines[0])): path.read_bytes()
                for path in quarantines[0].rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_quarantine_never_chmods_a_substituted_foreign_path(self) -> None:
        """Quarantine permissions cannot follow a transient foreign symlink."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            Exp417CheckpointStore(root, _provenance(), context.datasets)
            (root / "foreign-crash-artifact").write_bytes(b"corrupt")
            victim = parent / "foreign-victim.bin"
            victim_bytes = b"foreign-victim"
            victim.write_bytes(victim_bytes)
            os.chmod(victim, 0o644)
            original_chmod = os.chmod
            substituted = False

            def substitute_quarantine_for_chmod(
                path, mode, *args, **kwargs
            ) -> None:
                nonlocal substituted
                candidate = Path(path)
                if (
                    not substituted
                    and candidate.name.startswith(
                        ".checkpoint.quarantine."
                    )
                ):
                    saved = parent / "owned-quarantine-placeholder"
                    os.rename(candidate, saved)
                    os.symlink(victim, candidate)
                    try:
                        original_chmod(
                            candidate, mode, *args, **kwargs
                        )
                    finally:
                        os.unlink(candidate)
                        os.rename(saved, candidate)
                    substituted = True
                    return
                original_chmod(path, mode, *args, **kwargs)

            with (
                patch.object(
                    checkpoints.os,
                    "chmod",
                    side_effect=substitute_quarantine_for_chmod,
                ),
                self.assertRaisesRegex(ValueError, "quarantined"),
            ):
                Exp417CheckpointStore(
                    root, _provenance(), context.datasets
                )

            self.assertEqual(victim.read_bytes(), victim_bytes)
            self.assertEqual(victim.stat().st_mode & 0o777, 0o644)

    def test_reopen_quarantines_progress_record_mismatch_and_preserves_record(self) -> None:
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            _write(store, _result(context))
            record = next((root / "datasets").glob("*.json"))
            expected_record = record.read_bytes()
            progress = root / "progress.json"
            payload = json.loads(progress.read_text(encoding="utf-8"))
            payload["completed_datasets"] = 0
            payload["completed_record_digests"] = {}
            progress.write_bytes(canonical_json_bytes(payload) + b"\n")

            with self.assertRaisesRegex(ValueError, "quarantined"):
                Exp417CheckpointStore(root, _provenance(), context.datasets)

            self.assertFalse(root.exists())
            quarantines = tuple(parent.glob(".checkpoint.quarantine.*"))
            self.assertEqual(len(quarantines), 1)
            quarantined_records = tuple((quarantines[0] / "datasets").glob("*.json"))
            self.assertEqual(len(quarantined_records), 1)
            self.assertEqual(quarantined_records[0].read_bytes(), expected_record)

    def test_reopen_quarantines_an_owned_root_missing_datasets(self) -> None:
        """Pinned provenance makes a missing datasets directory owned corruption."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            os.rmdir(root / "datasets")

            with self.assertRaisesRegex(ValueError, "quarantined"):
                Exp417CheckpointStore(
                    root, _provenance(), ("synthetic-0000",)
                )

            self.assertFalse(root.exists())
            self.assertEqual(
                len(tuple(parent.glob(".checkpoint.quarantine.*"))),
                1,
            )

    def test_preopened_writer_quarantines_a_missing_datasets_directory(self) -> None:
        """A surviving writer applies the same owned-corruption boundary."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            survivor = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            os.rmdir(root / "datasets")

            with self.assertRaisesRegex(ValueError, "quarantined"):
                _write(survivor, _result(context))

            self.assertFalse(root.exists())
            self.assertEqual(
                len(tuple(parent.glob(".checkpoint.quarantine.*"))),
                1,
            )

    def test_preopened_writer_reconciles_and_quarantines_before_next_write(self) -> None:
        """A writer surviving a peer crash must not append to mismatched progress."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            first = Exp417CheckpointStore(root, _provenance(), context.datasets)
            survivor = Exp417CheckpointStore(root, _provenance(), context.datasets)
            _write(first, _result(context, "synthetic-0000"))
            record = next((root / "datasets").glob("*.json"))
            expected_record = record.read_bytes()
            progress = root / "progress.json"
            payload = json.loads(progress.read_text(encoding="utf-8"))
            payload["completed_datasets"] = 0
            payload["completed_record_digests"] = {}
            progress.write_bytes(canonical_json_bytes(payload) + b"\n")

            with (
                patch.object(
                    checkpoints,
                    "_atomic_write_exclusive_at",
                    wraps=checkpoints._atomic_write_exclusive_at,
                ) as publish,
                self.assertRaisesRegex(ValueError, "quarantined"),
            ):
                _write(survivor, _result(context, "synthetic-0001"))

            publish.assert_not_called()
            self.assertFalse(root.exists())
            quarantines = tuple(parent.glob(".checkpoint.quarantine.*"))
            self.assertEqual(len(quarantines), 1)
            records = tuple((quarantines[0] / "datasets").glob("*.json"))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].read_bytes(), expected_record)

    def test_writer_reconciles_metadata_without_reloading_prior_results(self) -> None:
        """Sequential writes must not perform quadratic prior-record JSON I/O."""
        context = _context()
        expected = ("synthetic-0000", "synthetic-0001", "synthetic-0002")
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), expected)
            _write(store, _result(context, expected[0]))
            prior_name = store._dataset_path(expected[0]).name
            read_regular = checkpoints._read_regular_at

            with (
                patch.object(
                    store,
                    "_load_records_from_descriptor",
                    side_effect=AssertionError(
                        "writer reconstructed all prior results"
                    ),
                ),
                patch.object(
                    checkpoints,
                    "_record_from_bytes",
                    side_effect=AssertionError(
                        "writer parsed a prior record payload"
                    ),
                ),
                patch.object(
                    checkpoints,
                    "_read_regular_at",
                    wraps=read_regular,
                ) as reads,
            ):
                _write(store, _result(context, expected[1]))

            self.assertNotIn(
                prior_name,
                tuple(
                    call.args[1]
                    for call in reads.call_args_list
                    if len(call.args) >= 2
                ),
            )

            progress = root / "progress.json"
            payload = json.loads(progress.read_text(encoding="utf-8"))
            payload["completed_datasets"] = 1
            payload["completed_record_digests"].pop(expected[1])
            progress.write_bytes(canonical_json_bytes(payload) + b"\n")

            with (
                patch.object(
                    store,
                    "_load_records_from_descriptor",
                    side_effect=AssertionError(
                        "writer reconstructed all prior results"
                    ),
                ),
                patch.object(
                    checkpoints,
                    "_record_from_bytes",
                    side_effect=AssertionError(
                        "writer parsed a prior record payload"
                    ),
                ),
                patch.object(
                    checkpoints,
                    "_atomic_write_exclusive_at",
                    wraps=checkpoints._atomic_write_exclusive_at,
                ) as publish,
                self.assertRaisesRegex(ValueError, "quarantined"),
            ):
                _write(store, _result(context, expected[2]))

            publish.assert_not_called()
            self.assertFalse(root.exists())
            self.assertEqual(
                len(tuple(parent.glob(".checkpoint.quarantine.*"))),
                1,
            )

    def test_writer_quarantines_foreign_progress_key_before_publish(self) -> None:
        """Metadata reconciliation rejects foreign progress keys pre-mutation."""
        context = _context()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkpoint"
            store = Exp417CheckpointStore(
                root, _provenance(), ("synthetic-0000",)
            )
            foreign_dataset = "foreign-dataset"
            progress = root / "progress.json"
            payload = json.loads(progress.read_text(encoding="utf-8"))
            payload["completed_datasets"] = 1
            payload["completed_record_digests"] = {
                foreign_dataset: "f" * 64
            }
            progress.write_bytes(canonical_json_bytes(payload) + b"\n")
            foreign_record = (
                root
                / "datasets"
                / f"{sha256(foreign_dataset.encode('utf-8')).hexdigest()}.json"
            )
            foreign_record.write_bytes(b"foreign-metadata-record")

            with (
                patch.object(
                    checkpoints,
                    "_atomic_write_exclusive_at",
                    wraps=checkpoints._atomic_write_exclusive_at,
                ) as publish,
                self.assertRaisesRegex(ValueError, "quarantined"),
            ):
                _write(store, _result(context))

            publish.assert_not_called()
            self.assertFalse(root.exists())
            self.assertEqual(
                len(tuple(parent.glob(".checkpoint.quarantine.*"))),
                1,
            )

    def test_completed_record_digests_is_validated_sorted_and_read_only(self) -> None:
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            _write(store, _result(context, "synthetic-0001"))
            _write(store, _result(context, "synthetic-0000"))
            records = tuple((root / "datasets").glob("*.json"))
            expected = {
                payload["dataset"]: payload["payload_sha256"]
                for payload in (
                    json.loads(path.read_text(encoding="utf-8")) for path in records
                )
            }
            method = getattr(store, "completed_record_digests", None)
            self.assertIsNotNone(method)
            digests = method()
            self.assertEqual(tuple(digests), ("synthetic-0000", "synthetic-0001"))
            self.assertEqual(dict(digests), expected)
            with self.assertRaises(TypeError):
                digests["synthetic-0002"] = "f" * 64

    def test_canonical_result_digest_matches_the_full_persisted_result_payload(self) -> None:
        context = _context()
        result = _result(context)
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            _write(store, result)
            record = json.loads(
                next((root / "datasets").glob("*.json")).read_text(
                    encoding="utf-8"
                )
            )
            expected = sha256(
                canonical_json_bytes(record["qualification"])
            ).hexdigest()
            function = getattr(checkpoints, "canonical_result_sha256", None)
            self.assertIsNotNone(function)
            self.assertEqual(function(result), expected)

    def test_mixed_diagnostic_checkpoint_accepts_oof_without_a_final_model(self) -> None:
        context = _context()
        result = _mixed_result(context)
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            try:
                store.write_dataset(
                    result,
                    raw_train_identity_sha256="a" * 64,
                    repaired_train_identity_sha256="b" * 64,
                    length_group_identity_sha256={64: "c" * 64, 100: "d" * 64},
                    canonical_model_hashes={
                        100: {
                            seed: "e" * 64
                            for seed in FROZEN_CONFIG.model_seeds
                        }
                    },
                    score_calibration_hashes={64: "f" * 64, 100: "0" * 64},
                    backend_audit=BackendAudit(
                        "cpu", True, 0.0, 1.0, 1.0, False
                    ),
                    elapsed_seconds=0.25,
                )
            except ValueError as error:
                self.fail(
                    f"diagnostic OOF evidence was incorrectly rejected: {error}"
                )

            self.assertEqual(tuple(store.load_completed()), ("synthetic-0000",))
            self.assertEqual(store.progress()["completed_datasets"], 1)

    def test_checkpoint_stage_hashes_exactly_match_each_group_state(self) -> None:
        """Undefined abstentions invent no hashes; measured failures omit none."""
        context = _context()
        undefined = {
            "level_shift": None,
            "gradual_drift": None,
            "amplitude_scaling": None,
            "period_change": None,
        }
        abstained = GroupQualification(
            length=100,
            series_count=32,
            status="abstained",
            oof_candidate_rate=None,
            wilson_upper=None,
            pooled_synthetic_sensitivity=None,
            per_type_synthetic_sensitivity=undefined,
            seed_agreement=None,
            finite=None,
            backend_fallback=None,
        )
        failed = replace(
            _group(),
            status="failed",
            oof_candidate_rate=0.02,
            wilson_upper=0.06,
        )

        def result_for(group: GroupQualification) -> DatasetResult:
            contract = FrozenTrainContract(
                "synthetic-0000",
                (group,),
                BackendAudit("cpu", True, 0.0, 1.0, 1.0, False),
                "a" * 64,
            )
            evidence = DatasetDecisionEvidence(
                "synthetic-0000",
                context,
                contract,
                context.database_sha256,
                contract.train_identity_sha256,
                (),
            )
            return DatasetResult(
                "synthetic-0000",
                2,
                1,
                (group,),
                (),
                evidence,
                context.lanes,
                context,
                "abstained",
            )

        cases = (
            (
                "abstained-invented-stage-hashes",
                result_for(abstained),
                {
                    100: {
                        seed: "d" * 64
                        for seed in FROZEN_CONFIG.model_seeds
                    }
                },
                {100: "e" * 64},
            ),
            ("failed-missing-stage-hashes", result_for(failed), {}, {}),
        )
        for name, result, models, calibrations in cases:
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                store = Exp417CheckpointStore(
                    Path(temporary) / "checkpoint",
                    _provenance(),
                    ("synthetic-0000",),
                )
                with self.assertRaisesRegex(ValueError, "length-group"):
                    store.write_dataset(
                        result,
                        raw_train_identity_sha256="a" * 64,
                        repaired_train_identity_sha256="b" * 64,
                        length_group_identity_sha256={100: "c" * 64},
                        canonical_model_hashes=models,
                        score_calibration_hashes=calibrations,
                        backend_audit=BackendAudit(
                            "cpu", True, 0.0, 1.0, 1.0, False
                        ),
                        elapsed_seconds=0.125,
                    )
                self.assertEqual(dict(store.load_completed()), {})

    def test_numeric_key_aliases_are_rejected(self) -> None:
        context = _context()
        for parent, alias in (
            ("length_group_identity_sha256", "0100"),
            ("score_calibration_hashes", "+100"),
            ("canonical_model_hashes", " 100"),
        ):
            with self.subTest(parent=parent), TemporaryDirectory() as temporary:
                root = Path(temporary) / "checkpoint"
                store = Exp417CheckpointStore(root, _provenance(), context.datasets)
                _write(store, _result(context))
                path = next((root / "datasets").glob("*.json"))
                payload = json.loads(path.read_text(encoding="utf-8"))
                source = payload[parent]
                payload[parent] = {alias: next(iter(source.values()))}
                payload["payload_sha256"] = _digest(payload)
                path.write_bytes(canonical_json_bytes(payload) + b"\n")
                with self.assertRaises(ValueError):
                    store.load_completed()
        for mutation in ("model_seed", "horizon"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as temporary:
                root = Path(temporary) / "checkpoint"
                store = Exp417CheckpointStore(root, _provenance(), context.datasets)
                _write(store, _result(context))
                path = next((root / "datasets").glob("*.json"))
                payload = json.loads(path.read_text(encoding="utf-8"))
                if mutation == "model_seed":
                    seeds = payload["canonical_model_hashes"]["100"]
                    payload["canonical_model_hashes"]["100"] = {
                        "041701": seeds["41701"], "41702": seeds["41702"], "41703": seeds["41703"],
                    }
                else:
                    scores = payload["qualification"]["decision_evidence"]["decisions"][0]["seed_decisions"][0]["scores"]
                    payload["qualification"]["decision_evidence"]["decisions"][0]["seed_decisions"][0]["scores"] = {
                        "01": scores["1"], "2": scores["2"], "10": scores["10"],
                    }
                payload["payload_sha256"] = _digest(payload)
                path.write_bytes(canonical_json_bytes(payload) + b"\n")
                with self.assertRaises(ValueError):
                    store.load_completed()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            target = Path(temporary) / "target"
            target.mkdir()
            os.symlink(target, root)
            with self.assertRaises(ValueError):
                Exp417CheckpointStore(root, _provenance(), context.datasets)
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            _write(store, _result(context))
            record_path = next((root / "datasets").glob("*.json"))
            target = Path(temporary) / "record-target.json"
            target.write_text("{}", encoding="utf-8")
            record_path.unlink()
            os.symlink(target, record_path)
            with self.assertRaises(ValueError):
                store.load_completed()


def _write_arguments(*, elapsed_seconds: float = 0.125) -> dict[str, object]:
    return {
        "raw_train_identity_sha256": "a" * 64,
        "repaired_train_identity_sha256": "b" * 64,
        "length_group_identity_sha256": {100: "c" * 64},
        "canonical_model_hashes": {100: {seed: "d" * 64 for seed in FROZEN_CONFIG.model_seeds}},
        "score_calibration_hashes": {100: "e" * 64},
        "backend_audit": BackendAudit("cpu", True, 0.0, 1.0, 1.0, False),
        "elapsed_seconds": elapsed_seconds,
    }


def _digest(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("payload_sha256", None)
    return sha256(canonical_json_bytes(body)).hexdigest()
