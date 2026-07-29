"""Synthetic-only tests for Exp417 local checkpoint safety."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
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


class Exp417CheckpointTests(unittest.TestCase):
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
                with self.assertRaises(OSError):
                    _write(store, record)
            self.assertEqual(tuple((root / "datasets").iterdir()), ())
            _write(store, record)
            path = next((root / "datasets").glob("*.json"))
            path.write_bytes(b"{")
            with self.assertRaises(ValueError):
                store.load_completed()

    def test_every_publish_boundary_failure_rolls_back_record_and_progress(self) -> None:
        context = _context()
        # Lock, marker write/fsync, marker directory fsync, record write/fsync,
        # datasets directory fsync, progress write/fsync, progress directory
        # fsync, and transaction-finalization root fsync all have rollback.
        for fail_at in range(1, 9):
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

    def test_progress_replace_failure_stale_transaction_or_orphan_record_fail_closed(self) -> None:
        context = _context()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            original_replace = os.replace

            def reject_progress(source: object, destination: object) -> None:
                if Path(destination).name == "progress.json":
                    raise OSError("injected progress replace")
                original_replace(source, destination)

            with patch.object(checkpoints.os, "replace", side_effect=reject_progress):
                with self.assertRaises(OSError):
                    _write(store, _result(context))
            self.assertEqual(tuple((root / "datasets").iterdir()), ())
            self.assertEqual(store.progress()["completed_datasets"], 0)
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            original_write = checkpoints._atomic_write
            failed = False

            def reject_progress_write(path: Path, *args: object, **kwargs: object) -> None:
                nonlocal failed
                if path.name == "progress.json" and not failed:
                    failed = True
                    raise OSError("injected progress temp/write")
                return original_write(path, *args, **kwargs)

            with patch.object(checkpoints, "_atomic_write", side_effect=reject_progress_write):
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

    def test_numeric_key_aliases_and_two_writer_race_are_rejected(self) -> None:
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
            first = Exp417CheckpointStore(root, _provenance(), context.datasets)
            second = Exp417CheckpointStore(root, _provenance(), context.datasets)
            barrier = threading.Barrier(2)
            original_write = checkpoints._atomic_write
            results: list[object] = []

            def synchronize_record(path: Path, *args: object, **kwargs: object) -> None:
                if path.parent.name == "datasets":
                    barrier.wait(timeout=2)
                return original_write(path, *args, **kwargs)

            def writer(store: Exp417CheckpointStore) -> None:
                try:
                    _write(store, _result(context))
                except BaseException as error:
                    results.append(error)
                else:
                    results.append("ok")

            with patch.object(checkpoints, "_atomic_write", side_effect=synchronize_record):
                threads = [threading.Thread(target=writer, args=(store,)) for store in (first, second)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)
            self.assertEqual(results.count("ok"), 1)
            self.assertEqual(len(first.load_completed()), 1)
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
