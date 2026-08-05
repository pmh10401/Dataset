"""Read-only Exp424 intake and TRAIN-only abstention tests for Exp426."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import Exp417CheckpointStore
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    DatasetResult,
    DatasetTerminalEvidence,
    RawSeries,
    RawTrainDataset,
)
from experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints import (
    _context as exp417_context,
    _provenance as exp417_provenance,
    _result as exp417_result,
    _write as exp417_write,
)
from experiments.exp425_selective_recovery.predecessor import (
    _fixture_contract,
    open_validated_exp424_source,
)
from experiments.exp425_selective_recovery.tests.test_eligibility import (
    _series_with_identity_classes,
)
from experiments.exp426_verified_baseline.source_antigravity import (
    classify_exp424_record_bytes,
    inspect_production_source_authority,
    read_verified_source_tasks,
    revalidate_verified_source,
    verify_train_abstention,
)


def _source_fixture(parent: Path):
    context = exp417_context()
    datasets = context.datasets[:3]
    root = parent / "source"
    store = Exp417CheckpointStore(root, exp417_provenance(), datasets)
    for dataset in datasets[:2]:
        exp417_write(store, exp417_result(context, dataset))
    terminal_evidence = DatasetTerminalEvidence(
        datasets[2], context, "oof",
        "early stopping requires at least five identities on both sides",
    )
    terminal = DatasetResult(
        datasets[2], context.test_sizes[datasets[2]], 0, (), (),
        terminal_evidence, context.lanes, context, "abstained",
    )
    store.write_dataset(
        terminal,
        raw_train_identity_sha256=None,
        repaired_train_identity_sha256=None,
        length_group_identity_sha256={},
        canonical_model_hashes={},
        score_calibration_hashes={},
        backend_audit=None,
        elapsed_seconds=0.0,
    )
    provenance_raw = (root / "checkpoint_provenance.json").read_bytes()
    progress_raw = (root / "progress.json").read_bytes()
    progress = json.loads(progress_raw)
    contract = _fixture_contract(
        root,
        root_provenance_sha256=sha256(provenance_raw).hexdigest(),
        progress_sha256=sha256(progress_raw).hexdigest(),
        record_set_sha256=sha256(
            canonical_json_bytes(sorted(progress["completed_record_digests"].items()))
        ).hexdigest(),
    )
    return root, contract, datasets


class Exp426SourceTests(unittest.TestCase):

    def test_three_record_source_is_classified_two_importable_one_terminal_without_writes(self) -> None:
        with TemporaryDirectory() as directory:
            root, contract, datasets = _source_fixture(Path(directory))
            before = {
                path.relative_to(root): (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
                for path in root.rglob("*") if path.is_file()
            }
            with open_validated_exp424_source(root, contract) as source:
                tasks = tuple(read_verified_source_tasks(source))
                revalidate_verified_source(source)
            after = {
                path.relative_to(root): (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
                for path in root.rglob("*") if path.is_file()
            }

            classified = tuple(
                classify_exp424_record_bytes(task.dataset, task.raw, task.expected_digest)
                for task in tasks
            )
            self.assertEqual(tuple(task.dataset for task in tasks), datasets)
            self.assertEqual(sum(item.imported_record is not None for item in classified), 2)
            self.assertEqual(
                tuple(item.terminal_reason for item in classified if item.terminal_reason),
                ("oof_identity_insufficient",),
            )
            self.assertTrue(all(
                item.imported_record is None or item.imported_record.status == "scored"
                for item in classified
            ))
            self.assertEqual(
                tuple(
                    (item.train_size, item.train_lengths, item.test_size, item.test_lengths)
                    for item in classified[:2]
                ),
                ((100, (100,), 2, (100,)), (100, (100,), 2, (100,))),
            )
            self.assertEqual(
                (classified[2].train_size, classified[2].train_lengths), (0, ())
            )
            self.assertEqual(before, after)

    def test_classification_rejects_dataset_digest_and_byte_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            root, contract, _datasets = _source_fixture(Path(directory))
            with open_validated_exp424_source(root, contract) as source:
                task = tuple(read_verified_source_tasks(source))[0]
            for dataset, raw, digest in (
                ("wrong", task.raw, task.expected_digest),
                (task.dataset, task.raw + b" ", task.expected_digest),
                (task.dataset, task.raw, "0" * 64),
            ):
                with self.subTest(dataset=dataset, size=len(raw), digest=digest), self.assertRaises(ValueError):
                    classify_exp424_record_bytes(dataset, raw, digest)

    def test_live_source_mutation_after_worker_classification_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root, contract, _datasets = _source_fixture(Path(directory))
            with open_validated_exp424_source(root, contract) as source:
                task = tuple(read_verified_source_tasks(source))[0]
                classify_exp424_record_bytes(
                    task.dataset, task.raw, task.expected_digest
                )
                record_path = root / "datasets" / sha256(
                    task.dataset.encode("utf-8")
                ).hexdigest()
                record_path = record_path.with_suffix(".json")
                record_path.write_bytes(task.raw + b" ")
                with self.assertRaises(ValueError):
                    revalidate_verified_source(source)

    def test_train_verification_records_full_identity_insufficiency_evidence(self) -> None:
        dataset = "B_normal_1"
        identities = _series_with_identity_classes(
            64, early_count=4, other_count=6
        )
        train = RawTrainDataset(
            dataset,
            tuple(
                RawSeries(index, identities[index % len(identities)].values)
                for index in range(100)
            ),
        )

        record = verify_train_abstention(
            dataset,
            train,
            "a" * 64,
            source_record_sha256="b" * 64,
            source_result_sha256="c" * 64,
        )

        self.assertEqual(record.origin, "verified_train_abstention")
        self.assertEqual(record.status, "abstained")
        self.assertEqual(record.candidate_coordinates, ())
        self.assertEqual(record.reason, "oof_identity_insufficient")
        self.assertEqual(len(record.train_identity_sha256 or ""), 64)
        self.assertEqual(len(record.eligibility_sha256 or ""), 64)
        self.assertEqual(
            record,
            verify_train_abstention(
                dataset, train, "a" * 64,
                source_record_sha256="b" * 64,
                source_result_sha256="c" * 64,
            ),
        )

    def test_train_verification_fails_closed_if_any_group_is_eligible(self) -> None:
        dataset = "Eligible_normal_1"
        train = RawTrainDataset(
            dataset,
            _series_with_identity_classes(64, early_count=20, other_count=100),
        )
        with self.assertRaisesRegex(ValueError, "eligible"):
            verify_train_abstention(
                dataset, train, "a" * 64,
                source_record_sha256="b" * 64,
                source_result_sha256="c" * 64,
            )

    def test_worker_source_has_no_test_value_or_label_loader_symbols(self) -> None:
        source_path = Path(__file__).parents[1] / "source_antigravity.py"
        text = source_path.read_text(encoding="utf-8")
        for forbidden in (
            "RawTestDataset", "load_raw_test_values", "load_test_labels",
            "actual_anomaly", "mean_combined_f1",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    @unittest.skipUnless(
        os.environ.get("EXP426_RUN_PRODUCTION_SOURCE_TEST") == "1",
        "production source audit is an explicit readiness check",
    )
    def test_production_source_is_exactly_1117_983_134(self) -> None:
        report = inspect_production_source_authority()
        self.assertEqual((report.records, report.importable, report.terminal), (1117, 983, 134))
        self.assertEqual(report.terminal_reasons, ("oof_identity_insufficient",))
        self.assertEqual(
            report.source_root_sha256,
            "f7526bf638f4c1051a5970271bdbbccd753986f7d067ec1260817da884c1b757",
        )
        self.assertEqual(
            report.source_progress_sha256,
            "5e9aa4a4b7240ac89c078bdc1dce4b41bda6b6dbc505dd9ad49782556da7bd84",
        )
        self.assertEqual(
            report.source_record_set_sha256,
            "d74a0def7788db89a1d3e32f46662abf07c44c7cbbce5e02612b510e2fdf4258",
        )


if __name__ == "__main__":
    unittest.main()
