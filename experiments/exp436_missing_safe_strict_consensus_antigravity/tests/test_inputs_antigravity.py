"""TDD coverage for Exp436's label-free missing-safe input session."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from hashlib import sha256
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import (
    MissingSafeRawDataset,
    RawValueSeries,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity import (
    inputs_antigravity,
)


class _SyntheticConfig:
    database_sha256 = "d" * 64
    exp291_prediction_sha256 = "1" * 64
    exp291_manifest_sha256 = "2" * 64
    exp292_prediction_sha256 = "3" * 64
    exp292_manifest_sha256 = "4" * 64

    def __init__(self, expected_datasets: int = 2) -> None:
        self.expected_datasets = expected_datasets

    def validate(self) -> None:
        return None


class _Snapshot:
    def __init__(self, *, malformed: str | None = None) -> None:
        self.closed = False
        self.opened = False
        self.boundaries: list[str] = []
        self.calls: list[tuple[str, str]] = []
        self.test_sql: list[str] = []
        self.train_queries = 0
        self.test_queries = 0
        self.identity_checks = 0
        self.test_authorities: list[object] = []
        self.malformed = malformed
        self.train_blobs = (
            (0, np.asarray([1.0, np.nan], dtype="<f4").tobytes()),
            (1, np.asarray([2.0, -np.inf], dtype="<f4").tobytes()),
        )
        self.test_blobs = ((2, np.asarray([3.0, np.inf, -np.inf], dtype="<f4").tobytes()),)
        self.snapshot_identity_manifest = ((2, 3),)

    def __enter__(self) -> "_Snapshot":
        self.opened = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.closed = True

    def freeze_context(self, lanes: object, *, verified_code_commit: str) -> object:
        return SimpleNamespace(
            datasets=tuple(lanes.population),
            code_commit=verified_code_commit,
            test_identity_manifest={"A": ((2, 3),), "B": ((2, 3),)},
        )

    def load_normal_train_value_blobs(self, dataset: str, context: object) -> tuple[tuple[int, bytes], ...]:
        self.boundaries.extend(("train:before", "train:after"))
        self.calls.append(("TRAIN", dataset))
        self.train_queries += 1
        if self.malformed == "train_decode":
            return ((-1, b"\x00\x00\x00\x00"),)
        return self.train_blobs

    def load_test_value_blobs_after_train(
        self, dataset: str, train: object, context: object
    ) -> tuple[tuple[int, bytes], ...]:
        self.boundaries.extend(("test:before", "test:after"))
        self.calls.append(("TEST", dataset))
        self.test_queries += 1
        self.test_authorities.append(train)
        self.test_sql.append(
            "SELECT i.instance_index, i.values_blob FROM instances AS i "
            "WHERE i.split = 'TEST' ORDER BY i.instance_index"
        )
        self.identity_checks += 1
        if tuple((index, len(blob) // 4) for index, blob in self.test_blobs) != self.snapshot_identity_manifest:
            raise ValueError("snapshot TEST identity mismatch")
        if self.malformed == "test_container":
            return ((2, self.test_blobs[0][1]), (2, self.test_blobs[0][1]))
        return self.test_blobs


class Exp436InputTests(unittest.TestCase):
    def fixture(self, *, malformed: str | None = None):
        snapshot = _Snapshot(malformed=malformed)
        lanes = SimpleNamespace(population=("A", "B"))
        trace: list[str] = []

        @contextmanager
        def manager(*args: object, **kwargs: object):
            trace.append("open_database_snapshot")
            with snapshot:
                yield snapshot

        def freeze(*args: object, **kwargs: object) -> object:
            trace.append("freeze_lane_inputs")
            return lanes

        config = _SyntheticConfig()
        patcher = patch.multiple(
            inputs_antigravity,
            FROZEN_CONFIG=config,
            EXP417_FROZEN_CONFIG=config,
            freeze_lane_inputs=freeze,
            open_database_snapshot=manager,
        )
        return patcher, snapshot, trace

    def open_session(self, patcher: object):
        return inputs_antigravity.open_exp436_input_session(
            database=Path("db.sqlite"),
            exp291_prediction=Path("p291"),
            exp291_manifest=Path("m291"),
            exp292_prediction=Path("p292"),
            exp292_manifest=Path("m292"),
            verified_code_commit="f" * 40,
        )

    def test_decoder_hashes_original_bytes_and_preserves_nonfinite_values(self) -> None:
        raw = bytes.fromhex("0000803f0000c07f0000807f000080ff")
        row = inputs_antigravity.decode_raw_value_row_antigravity(7, raw)
        self.assertEqual(row.raw_blob_sha256, sha256(raw).hexdigest())
        self.assertEqual(row.values[0], 1.0)
        self.assertTrue(math.isnan(row.values[1]))
        self.assertEqual(row.values[2], math.inf)
        self.assertEqual(row.values[3], -math.inf)

    def test_decoder_rejects_malformed_index_and_blob_without_conversion(self) -> None:
        for index, blob in ((True, b"\x00" * 4), (-1, b"\x00" * 4), (0, bytearray(4)), (0, b""), (0, b"\x00" * 3)):
            with self.subTest(index=index, blob_type=type(blob).__name__), self.assertRaises(ValueError):
                inputs_antigravity.decode_raw_value_row_antigravity(index, blob)  # type: ignore[arg-type]

    def test_frozen_lanes_are_checked_before_snapshot_open(self) -> None:
        patcher, snapshot, trace = self.fixture()
        with patcher:
            with self.open_session(patcher):
                pass
        self.assertEqual(trace, ["freeze_lane_inputs", "open_database_snapshot"])
        self.assertTrue(snapshot.closed)

    def test_train_is_cached_by_identity_without_a_second_query(self) -> None:
        patcher, snapshot, _trace = self.fixture()
        with patcher:
            with self.open_session(patcher) as session:
                first = session.load_train("A")
                second = session.load_train("A")
                self.assertIs(first, second)
                self.assertEqual(snapshot.train_queries, 1)

    def test_test_requires_exact_cached_train_identity_before_query(self) -> None:
        patcher, snapshot, _trace = self.fixture()
        with patcher:
            with self.open_session(patcher) as session:
                train = session.load_train("A")
                reordered = MissingSafeRawDataset(
                    "A", "TRAIN", tuple(
                        RawValueSeries(row.instance_index, tuple(reversed(row.values)), row.raw_blob_sha256)
                        for row in train.series
                    )
                )
                foreign = MissingSafeRawDataset("A", "TRAIN", train.series)
                for unauthorized in (copy.copy(train), copy.deepcopy(train), reordered, foreign):
                    with self.subTest(kind=type(unauthorized).__name__), self.assertRaisesRegex(ValueError, "bound TRAIN"):
                        session.load_test_values("A", unauthorized)
                self.assertEqual(snapshot.test_queries, 0)
                test = session.load_test_values("A", train)
                self.assertEqual(test.identity_lengths, ((2, 3),))
                self.assertIs(snapshot.test_authorities[-1], snapshot.train_blobs)

    def test_test_identity_is_checked_by_snapshot_and_session(self) -> None:
        patcher, snapshot, _trace = self.fixture()
        with patcher:
            with self.open_session(patcher) as session:
                train = session.load_train("A")
                session.context.test_identity_manifest["A"] = ((2, 4),)
                with self.assertRaisesRegex(ValueError, "identity manifest"):
                    session.load_test_values("A", train)
                self.assertEqual(snapshot.identity_checks, 1)

    def test_closed_session_rejects_cached_train_and_test_reuse(self) -> None:
        patcher, _snapshot, _trace = self.fixture()
        with patcher:
            with self.open_session(patcher) as session:
                train = session.load_train("A")
                self.assertTrue(session._train_authority_antigravity)
            self.assertEqual(session._train_authority_antigravity, {})
            session._close_antigravity()
            self.assertEqual(session._train_authority_antigravity, {})
            with self.assertRaisesRegex(ValueError, "closed"):
                session.load_train("A")
            with self.assertRaisesRegex(ValueError, "closed"):
                session.load_test_values("A", train)

    def test_exceptional_context_exit_disposes_train_authority(self) -> None:
        patcher, _snapshot, _trace = self.fixture()
        retained_session = None
        with patcher, self.assertRaisesRegex(RuntimeError, "body failure"):
            with self.open_session(patcher) as session:
                retained_session = session
                session.load_train("A")
                self.assertTrue(session._train_authority_antigravity)
                raise RuntimeError("body failure")
        assert retained_session is not None
        self.assertEqual(retained_session._train_authority_antigravity, {})
        with self.assertRaisesRegex(ValueError, "closed"):
            retained_session.load_train("A")

    def test_label_access_is_read_only_zero_and_no_label_loader_exists(self) -> None:
        patcher, _snapshot, _trace = self.fixture()
        with patcher:
            with self.open_session(patcher) as session:
                self.assertEqual(session.label_access_count, 0)
                with self.assertRaises(AttributeError):
                    session.label_access_count = 1
                self.assertFalse(any("label" in name.lower() and name != "label_access_count" for name in dir(session)))

    def test_test_query_trace_never_uses_forbidden_fields(self) -> None:
        patcher, snapshot, _trace = self.fixture()
        with patcher:
            with self.open_session(patcher) as session:
                session.load_test_values("A", session.load_train("A"))
        trace = " ".join(snapshot.test_sql).lower()
        for forbidden in ("label", "anomaly", "evaluation"):
            self.assertNotIn(forbidden, trace)

    def test_decode_failure_preserves_snapshot_after_boundary_and_closes(self) -> None:
        patcher, snapshot, _trace = self.fixture(malformed="train_decode")
        with patcher, self.assertRaises(ValueError):
            with self.open_session(patcher) as session:
                session.load_train("A")
        self.assertEqual(snapshot.boundaries, ["train:before", "train:after"])
        self.assertTrue(snapshot.closed)

    def test_container_failure_preserves_test_after_boundary_and_closes(self) -> None:
        patcher, snapshot, _trace = self.fixture(malformed="test_container")
        with patcher, self.assertRaises(ValueError):
            with self.open_session(patcher) as session:
                session.load_test_values("A", session.load_train("A"))
        self.assertEqual(snapshot.boundaries[-2:], ["test:before", "test:after"])
        self.assertTrue(snapshot.closed)

    def test_invalid_commit_rejects_before_database_open(self) -> None:
        patcher, snapshot, trace = self.fixture()
        with patcher, self.assertRaisesRegex(ValueError, "reviewed code commit"):
            with inputs_antigravity.open_exp436_input_session(
                verified_code_commit="BAD",
            ):
                pass
        self.assertEqual(trace, ["freeze_lane_inputs"])
        self.assertFalse(snapshot.opened)


if __name__ == "__main__":
    unittest.main()
