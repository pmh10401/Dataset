"""TDD tests for Exp435's descriptor-bound, label-free input session."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp435_train_only_consensus_recovery_antigravity import inputs_antigravity


class _SyntheticConfig:
    database_sha256 = "db"
    exp291_prediction_sha256 = "p291"
    exp291_manifest_sha256 = "m291"
    exp292_prediction_sha256 = "p292"
    exp292_manifest_sha256 = "m292"

    def __init__(self, expected_datasets: int = 2) -> None:
        self.expected_datasets = expected_datasets

    def validate(self) -> None:
        return None


class _Snapshot:
    def __init__(
        self,
        database_names: tuple[str, ...],
        context: object | None = None,
    ) -> None:
        self.database_names = tuple(database_names)
        self.context = context
        self.closed = False
        self.loaded: list[tuple[str, str]] = []

    def __enter__(self) -> "_Snapshot":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.closed = True

    def freeze_context(self, lanes: object, *, verified_code_commit: str) -> object:
        if self.context is None:
            frozen = tuple(lanes.population)
            database_names = tuple(sorted(self.database_names))
            database_set = set(database_names)
            missing = tuple(name for name in frozen if name not in database_set)
            if missing:
                raise ValueError(
                    "frozen lane references dataset outside modeled database"
                )
            self.context = SimpleNamespace(
                datasets=tuple(name for name in frozen if name in database_set),
                code_commit=verified_code_commit,
            )
        return self.context

    def load_train(self, dataset: str, context: object) -> object:
        self.loaded.append(("TRAIN", dataset))
        return SimpleNamespace(dataset=dataset, series=())

    def load_test(self, dataset: str, contract: object, context: object) -> object:
        self.loaded.append(("TEST", dataset))
        return SimpleNamespace(dataset=dataset, series=())


class InputBoundaryTests(unittest.TestCase):
    def fixture_session(
        self,
        *,
        database_names: tuple[str, ...],
        frozen: tuple[str, ...],
        context_datasets: tuple[str, ...] | None = None,
        expected_datasets: int = 2,
    ):
        snapshot = _Snapshot(database_names)
        lanes = SimpleNamespace(population=tuple(sorted(frozen)))

        if context_datasets is not None:
            snapshot.context = SimpleNamespace(datasets=context_datasets)

        @contextmanager
        def manager(*args: object, **kwargs: object):
            try:
                yield snapshot
            finally:
                snapshot.closed = True

        return patch.multiple(
            inputs_antigravity,
            FROZEN_CONFIG=_SyntheticConfig(expected_datasets),
            EXP417_FROZEN_CONFIG=_SyntheticConfig(expected_datasets),
            freeze_lane_inputs=lambda *args, **kwargs: lanes,
            open_database_snapshot=manager,
        ), snapshot

    def test_session_uses_frozen_1117_population_not_all_db_names(self):
        frozen = tuple(f"Dataset{i:04d}" for i in range(1117))
        database_names = frozen + ("DBOnlyExtra",)
        patcher, _snapshot = self.fixture_session(
            database_names=database_names,
            frozen=frozen,
            expected_datasets=1117,
        )
        with patcher:
            with inputs_antigravity.open_exp435_input_session(
                database=Path("db.sqlite"),
                exp291_prediction=Path("p291"),
                exp291_manifest=Path("m291"),
                exp292_prediction=Path("p292"),
                exp292_manifest=Path("m292"),
                verified_code_commit="f" * 40,
            ) as session:
                self.assertIn("DBOnlyExtra", _snapshot.database_names)
                self.assertNotIn("DBOnlyExtra", session.context.datasets)
                self.assertEqual(session.context.datasets, frozen)
                self.assertEqual(
                    session.context.datasets,
                    tuple(name for name in frozen if name in _snapshot.database_names),
                )
                self.assertEqual(len(session.context.datasets), 1117)
                self.assertEqual(
                    session.context.datasets, tuple(sorted(session.context.datasets))
                )

    def test_prediction_sql_never_selects_test_labels(self):
        source = Path(inputs_antigravity.__file__).read_text(encoding="utf-8")
        self.assertNotIn("labels_blob", source)
        self.assertNotRegex(source, r"TEST.+label")

    def test_session_delegates_values_only_and_keeps_label_access_count_zero(self):
        patcher, snapshot = self.fixture_session(
            database_names=("A", "B", "DBOnlyExtra"), frozen=("A", "B")
        )
        with patcher:
            with inputs_antigravity.open_exp435_input_session(
                database=Path("db.sqlite"),
                exp291_prediction=Path("p291"),
                exp291_manifest=Path("m291"),
                exp292_prediction=Path("p292"),
                exp292_manifest=Path("m292"),
                verified_code_commit="f" * 40,
            ) as session:
                session.load_train("A")
                session.load_test_values("A", object())
                self.assertEqual(session.label_access_count, 0)
        self.assertEqual(snapshot.loaded, [("TRAIN", "A"), ("TEST", "A")])
        self.assertTrue(snapshot.closed)

    def test_lane_population_count_mismatch_is_rejected_before_database_open(self):
        for frozen in (("A",), ("A", "B", "C")):
            with self.subTest(frozen=frozen):
                patcher, _snapshot = self.fixture_session(
                    database_names=("A", "B", "DBOnlyExtra"), frozen=frozen
                )
                with patcher, self.assertRaisesRegex(ValueError, "population"):
                    with inputs_antigravity.open_exp435_input_session(
                        database=Path("db.sqlite"),
                        exp291_prediction=Path("p291"),
                        exp291_manifest=Path("m291"),
                        exp292_prediction=Path("p292"),
                        exp292_manifest=Path("m292"),
                        verified_code_commit="f" * 40,
                    ):
                        pass

    def test_context_population_mismatch_is_rejected_and_snapshot_closes(self):
        patcher, snapshot = self.fixture_session(
            database_names=("A", "B", "DBOnlyExtra"),
            frozen=("A", "B"),
            context_datasets=("A", "DBOnlyExtra"),
        )
        with patcher, self.assertRaisesRegex(ValueError, "populations"):
            with inputs_antigravity.open_exp435_input_session(
                database=Path("db.sqlite"),
                exp291_prediction=Path("p291"),
                exp291_manifest=Path("m291"),
                exp292_prediction=Path("p292"),
                exp292_manifest=Path("m292"),
                verified_code_commit="f" * 40,
            ):
                pass
        self.assertTrue(snapshot.closed)

    def test_snapshot_closes_when_prediction_body_raises(self):
        patcher, snapshot = self.fixture_session(
            database_names=("A", "B", "DBOnlyExtra"), frozen=("A", "B")
        )
        with patcher, self.assertRaisesRegex(RuntimeError, "body"):
            with inputs_antigravity.open_exp435_input_session(
                database=Path("db.sqlite"),
                exp291_prediction=Path("p291"),
                exp291_manifest=Path("m291"),
                exp292_prediction=Path("p292"),
                exp292_manifest=Path("m292"),
                verified_code_commit="f" * 40,
            ):
                raise RuntimeError("body")
        self.assertTrue(snapshot.closed)

    def test_database_hash_mismatch_is_propagated(self):
        patcher, _snapshot = self.fixture_session(
            database_names=("A", "B"), frozen=("A", "B")
        )
        with patcher, patch.object(
            inputs_antigravity,
            "open_database_snapshot",
            side_effect=ValueError("database SHA256 does not match expected digest"),
        ), self.assertRaisesRegex(ValueError, "SHA256"):
            with inputs_antigravity.open_exp435_input_session(
                database=Path("db.sqlite"),
                exp291_prediction=Path("p291"),
                exp291_manifest=Path("m291"),
                exp292_prediction=Path("p292"),
                exp292_manifest=Path("m292"),
                verified_code_commit="f" * 40,
            ):
                pass

    def test_frozen_dataset_missing_from_modeled_database_fails_closed(self):
        patcher, snapshot = self.fixture_session(
            database_names=("A", "B"),
            frozen=("A", "B", "Missing"),
            expected_datasets=3,
        )
        with patcher, self.assertRaisesRegex(ValueError, "outside modeled database"):
            with inputs_antigravity.open_exp435_input_session(
                database=Path("db.sqlite"),
                exp291_prediction=Path("p291"),
                exp291_manifest=Path("m291"),
                exp292_prediction=Path("p292"),
                exp292_manifest=Path("m292"),
                verified_code_commit="f" * 40,
            ):
                pass
        self.assertTrue(snapshot.closed)


if __name__ == "__main__":
    unittest.main()
