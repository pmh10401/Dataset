from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp438_frequency_complexity_evidence_antigravity import inputs_antigravity


class _SyntheticConfig:
    database_sha256 = "d" * 64
    dataset_count = 2
    expected_datasets = 2
    minimum_raw_length = 16
    minimum_train_rows = 4

    def validate(self) -> None:
        return None


class _Snapshot:
    def __init__(self) -> None:
        self.closed = False
        self.train_queries = 0
        self.test_queries = 0
        self.test_authorities: list[object] = []
        self.train_blobs = tuple(
            (index, np.asarray(np.arange(16) + index, dtype="<f4").tobytes())
            for index in range(4)
        )
        self.test_blobs = (
            (4, np.asarray(np.arange(16), dtype="<f4").tobytes()),
            (5, np.asarray([np.nan] + list(range(15)), dtype="<f4").tobytes()),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.closed = True

    def freeze_context(self, lanes: object, *, verified_code_commit: str) -> object:
        return SimpleNamespace(
            datasets=tuple(lanes.population),
            code_commit=verified_code_commit,
            test_identity_manifest={"A": ((4, 16), (5, 16)), "B": ((4, 16), (5, 16))},
        )

    def load_normal_train_value_blobs(self, dataset: str, context: object):
        self.train_queries += 1
        return self.train_blobs

    def load_test_value_blobs_after_train(self, dataset: str, train: object, context: object):
        self.test_queries += 1
        self.test_authorities.append(train)
        return self.test_blobs


class Exp438InputsAntigravityTests(unittest.TestCase):
    def fixture(self):
        snapshot = _Snapshot()
        trace: list[str] = []

        @contextmanager
        def manager(*args, **kwargs):
            trace.append("open_database_snapshot")
            with snapshot:
                yield snapshot

        def freeze(*args, **kwargs):
            trace.append("freeze_lane_inputs")
            return SimpleNamespace(population=("A", "B"))

        config = _SyntheticConfig()
        patcher = patch.multiple(
            inputs_antigravity,
            FROZEN_CONFIG=config,
            EXP417_FROZEN_CONFIG=config,
            freeze_lane_inputs=freeze,
            open_database_snapshot=manager,
        )
        return patcher, snapshot, trace

    def open_session(self):
        return inputs_antigravity.open_exp438_input_session(
            database=Path("db.sqlite"),
            exp291_prediction=Path("p291"),
            exp291_manifest=Path("m291"),
            exp292_prediction=Path("p292"),
            exp292_manifest=Path("m292"),
            verified_code_commit="f" * 40,
        )

    def test_decoder_preserves_exact_little_endian_float32_and_blob_hash(self) -> None:
        blob = bytes.fromhex("0000803f0000c07f0000807f000080ff")
        row = inputs_antigravity.decode_raw_value_row_antigravity("A", "TEST", 7, blob)
        self.assertEqual(row.values_blob_sha256, sha256(blob).hexdigest())
        self.assertEqual(row.raw_length, 4)
        self.assertEqual(float(row.values[0]), 1.0)
        self.assertTrue(np.isnan(row.values[1]))
        self.assertEqual(float(row.values[2]), np.inf)
        self.assertEqual(float(row.values[3]), -np.inf)

    def test_decoder_rejects_malformed_identity_and_blob(self) -> None:
        for index, blob in ((True, b"\0" * 4), (-1, b"\0" * 4), (0, bytearray(4)), (0, b""), (0, b"\0" * 3)):
            with self.subTest(index=index, blob=type(blob).__name__), self.assertRaises(ValueError):
                inputs_antigravity.decode_raw_value_row_antigravity("A", "TRAIN", index, blob)

    def test_frozen_lanes_precede_one_snapshot_and_train_is_cached(self) -> None:
        patcher, snapshot, trace = self.fixture()
        with patcher:
            with self.open_session() as session:
                first = session.load_train("A")
                second = session.load_train("A")
                self.assertIs(first, second)
                self.assertEqual(session.label_access_count, 0)
                self.assertEqual(snapshot.train_queries, 1)
        self.assertEqual(trace, ["freeze_lane_inputs", "open_database_snapshot"])
        self.assertTrue(snapshot.closed)

    def test_test_values_require_exact_cached_train_identity(self) -> None:
        patcher, snapshot, _ = self.fixture()
        with patcher:
            with self.open_session() as session:
                train = session.load_train("A")
                with self.assertRaisesRegex(ValueError, "bound TRAIN"):
                    session.load_test_values("A", tuple(train.series))
                self.assertEqual(snapshot.test_queries, 0)
                test = session.load_test_values("A", train)
                self.assertEqual(test.identity_lengths, ((4, 16), (5, 16)))
                self.assertIs(snapshot.test_authorities[-1], snapshot.train_blobs)
                self.assertEqual(session.label_access_count, 0)

    def test_train_partition_excludes_whole_nonfinite_rows_and_keeps_exact_lengths(self) -> None:
        rows = []
        for index in range(5):
            values = np.arange(16, dtype=np.float32) + index
            if index == 4:
                values[3] = np.nan
            rows.append(
                inputs_antigravity.decode_raw_value_row_antigravity(
                    "A", "TRAIN", index, values.astype("<f4").tobytes()
                )
            )
        dataset = inputs_antigravity.Exp438RawDataset("A", "TRAIN", tuple(rows))
        prepared = inputs_antigravity.partition_train_rows_antigravity(dataset)
        self.assertEqual(tuple(group.raw_length for group in prepared.groups), (16,))
        self.assertEqual(prepared.groups[0].matrix.shape, (4, 16))
        self.assertEqual(len(prepared.exclusions), 1)
        self.assertEqual(prepared.exclusions[0].instance_index, 4)
        self.assertEqual(prepared.exclusions[0].nonfinite_positions, (3,))
        self.assertEqual(prepared.insufficient_lengths, ())

    def test_short_or_nonfinite_test_rows_abstain_without_changing_siblings(self) -> None:
        finite = inputs_antigravity.RawValueSeries("A", 0, "TEST", np.arange(16), "1" * 64)
        short = inputs_antigravity.RawValueSeries("A", 1, "TEST", np.arange(15), "2" * 64)
        nonfinite = inputs_antigravity.RawValueSeries("A", 2, "TEST", [1.0, np.inf] + list(range(14)), "3" * 64)
        self.assertIsNone(inputs_antigravity.test_row_abstention_reason_antigravity(finite))
        self.assertEqual(inputs_antigravity.test_row_abstention_reason_antigravity(short), "unsupported_length")
        self.assertEqual(inputs_antigravity.test_row_abstention_reason_antigravity(nonfinite), "nonfinite_test")
        self.assertTrue(np.isfinite(finite.values).all())

    def test_closed_session_rejects_reuse_and_discards_authority(self) -> None:
        patcher, _, _ = self.fixture()
        with patcher:
            with self.open_session() as session:
                train = session.load_train("A")
            self.assertEqual(session._train_authority_antigravity, {})
            with self.assertRaisesRegex(ValueError, "closed"):
                session.load_train("A")
            with self.assertRaisesRegex(ValueError, "closed"):
                session.load_test_values("A", train)


if __name__ == "__main__":
    unittest.main()
