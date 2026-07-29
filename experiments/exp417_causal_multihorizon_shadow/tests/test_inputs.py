"""Synthetic boundary tests for Exp417 frozen values-only inputs."""

from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp417_causal_multihorizon_shadow import inputs
from experiments.exp417_causal_multihorizon_shadow.config import (
    Exp417FrozenConfig,
    FROZEN_CONFIG,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    FrozenTrainContract,
    GroupQualification,
)


def assert_package_has_no_label_path(package_root: Path) -> None:
    forbidden = ("frozen_test_labels", "labels_blob", "load_test_labels")
    for path in package_root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        if any(token in text for token in forbidden):
            raise ValueError(f"Exp417 label isolation violation: {path.name}")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _row(dataset: str, *, hard: list[int] | None = None, normal: list[int] | None = None,
         standard: list[int] | None = None, priority: list[int] | None = None) -> dict[str, object]:
    return {
        "dataset": dataset,
        "n_test": 5,
        "automatic_hard_indices": [] if hard is None else hard,
        "automatic_normal_exact_indices": [] if normal is None else normal,
        "standard_review_indices": [] if standard is None else standard,
        "priority_review_indices": [] if priority is None else priority,
    }


class InputBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prediction_291 = self.root / "exp291.jsonl"
        self.manifest_291 = self.root / "exp291.manifest.json"
        self.prediction_292 = self.root / "exp292.jsonl"
        self.manifest_292 = self.root / "exp292.manifest.json"
        self.database = self.root / "synthetic.sqlite"
        self.rows = (
            _row("A", hard=[0], normal=[1], standard=[2], priority=[3]),
            _row("B"),
        )
        self._write_predictions(self.rows, self.rows)
        self._create_database()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @contextmanager
    def synthetic_config(self, **changes: object):
        defaults: dict[str, object] = {
            "expected_datasets": 2,
            "expected_hard": 1,
            "expected_standard": 1,
            "expected_priority": 1,
            "expected_automatic_normal": 1,
            "exp291_prediction_sha256": _sha(self.prediction_291),
            "exp291_manifest_sha256": _sha(self.manifest_291),
            "exp292_prediction_sha256": _sha(self.prediction_292),
            "exp292_manifest_sha256": _sha(self.manifest_292),
            "database_sha256": _sha(self.database),
        }
        defaults.update(changes)
        config = replace(FROZEN_CONFIG, **defaults)
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None):
            yield config

    def _write_predictions(
        self,
        rows_291: tuple[dict[str, object], ...],
        rows_292: tuple[dict[str, object], ...],
    ) -> None:
        for path, rows in (
            (self.prediction_291, rows_291),
            (self.prediction_292, rows_292),
        ):
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
        for prediction, manifest in (
            (self.prediction_291, self.manifest_291),
            (self.prediction_292, self.manifest_292),
        ):
            manifest.write_text(
                json.dumps(
                    {
                        "prediction_sha256": _sha(prediction),
                        "rows": 2,
                        "contains_test_labels": False,
                        "contains_evaluation_metrics": False,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    def _create_database(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE instances (
                    dataset_id INTEGER NOT NULL,
                    split TEXT NOT NULL,
                    instance_index INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    values_blob BLOB NOT NULL
                );
                INSERT INTO datasets(id, name) VALUES (1, 'A'), (2, 'B');
                """
            )
            values_a = np.array((1.5, -2.0, 3.25), dtype="<f4").tobytes()
            values_b = np.array((4.0, 5.5), dtype="<f4").tobytes()
            connection.executemany(
                "INSERT INTO instances VALUES (?, ?, ?, ?, ?)",
                (
                    (1, "TRAIN", 1, "0", values_a),
                    (1, "TRAIN", 0, "0", values_b),
                    (1, "TRAIN", 2, "1", values_a),
                    (1, "TEST", 4, "unseen", values_a),
                    (1, "TEST", 2, "unseen", values_b),
                    (2, "TRAIN", 0, "0", values_a),
                    (2, "TEST", 0, "unseen", values_b),
                ),
            )

    def _freeze_lanes(self, config: Exp417FrozenConfig):
        return inputs.freeze_lane_inputs(
            self.prediction_291,
            self.manifest_291,
            self.prediction_292,
            self.manifest_292,
            config=config,
        )

    def _context(self, config: Exp417FrozenConfig):
        return inputs.freeze_run_context(self.database, self._freeze_lanes(config), config=config)

    def _train_contract(self) -> FrozenTrainContract:
        return FrozenTrainContract(
            dataset="A",
            groups=(
                GroupQualification(
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
                ),
            ),
            backend=BackendAudit("cpu", True, 0.0, 1.0, 1.0, False),
            train_identity_sha256="synthetic-train-identity",
        )

    def test_freezes_equal_lanes_and_automatic_normal_coordinates(self) -> None:
        """A missing automatic-normal lane must not silently become eligible."""
        with self.synthetic_config() as config:
            lanes = self._freeze_lanes(config)
        self.assertEqual(lanes.hard, {"A": (0,)})
        self.assertEqual(lanes.automatic_normal, {"A": (1,)})
        self.assertEqual(lanes.standard, {"A": (2,)})
        self.assertEqual(lanes.priority, {"A": (3,)})

    def test_ignores_hashed_source_metadata_outside_the_frozen_lane_fields(self) -> None:
        """Provenance fields must not make a valid frozen lane unreadable."""
        enriched = (dict(self.rows[0], source_metadata="synthetic"), self.rows[1])
        self._write_predictions(enriched, enriched)
        with self.synthetic_config() as config:
            lanes = self._freeze_lanes(config)
        self.assertEqual(lanes.automatic_normal, {"A": (1,)})

    def test_rejects_preparse_mutation_of_each_frozen_artifact(self) -> None:
        """Every independently changed artifact must fail before parsing begins."""
        for artifact in (
            self.prediction_291,
            self.manifest_291,
            self.prediction_292,
            self.manifest_292,
        ):
            with self.subTest(artifact=artifact.name):
                self._write_predictions(self.rows, self.rows)
                with self.synthetic_config() as config:
                    artifact.write_text("mutated\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        self._freeze_lanes(config)

    def test_rehash_after_parse_rejects_mutation_of_each_frozen_artifact(self) -> None:
        """Every artifact is rehashed after its own successful parse."""
        prediction_paths = (self.prediction_291, self.prediction_292)
        manifest_paths = (self.manifest_291, self.manifest_292)
        for artifact in prediction_paths + manifest_paths:
            with self.subTest(artifact=artifact.name):
                self._write_predictions(self.rows, self.rows)
                with self.synthetic_config() as config:
                    if artifact in prediction_paths:
                        original = inputs._read_prediction_rows

                        def mutate_prediction(path: Path):
                            rows = original(path)
                            if Path(path) == artifact:
                                artifact.write_text("changed\n", encoding="utf-8")
                            return rows

                        replacement = mutate_prediction
                        target = "_read_prediction_rows"
                    else:
                        original = inputs._read_manifest

                        def mutate_manifest(path: Path):
                            manifest = original(path)
                            if Path(path) == artifact:
                                artifact.write_text("changed\n", encoding="utf-8")
                            return manifest

                        replacement = mutate_manifest
                        target = "_read_manifest"
                    with patch.object(inputs, target, replacement):
                        with self.assertRaisesRegex(ValueError, "changed during parsing"):
                            self._freeze_lanes(config)

    def test_rejects_duplicate_keys_in_prediction_and_manifest_json(self) -> None:
        """Duplicate JSON keys cannot choose a lane or manifest value by order."""
        duplicate_prediction = (
            '{"dataset":"A","dataset":"B","n_test":5,'
            '"automatic_hard_indices":[0],'
            '"automatic_normal_exact_indices":[1],'
            '"standard_review_indices":[2],"priority_review_indices":[3]}\n'
        )
        self.prediction_291.write_text(duplicate_prediction, encoding="utf-8")
        with self.synthetic_config() as config:
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                self._freeze_lanes(config)
        self._write_predictions(self.rows, self.rows)
        duplicate_manifest = json.dumps(
            {
                "prediction_sha256": _sha(self.prediction_291),
                "rows": 2,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            },
            separators=(",", ":"),
        )
        self.manifest_291.write_text(
            duplicate_manifest[:-1] + ',"rows":2}', encoding="utf-8"
        )
        with self.synthetic_config() as config:
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                self._freeze_lanes(config)

    def test_rejects_exp291_exp292_lane_mismatch(self) -> None:
        """Different frozen sources must not be merged into a new lane."""
        rows_292 = (_row("A", hard=[0], normal=[1], standard=[2], priority=[]), _row("B"))
        self._write_predictions(self.rows, rows_292)
        with self.synthetic_config() as config:
            with self.assertRaisesRegex(ValueError, "mismatch"):
                self._freeze_lanes(config)

    def test_rejects_duplicate_out_of_range_and_overlapping_coordinates(self) -> None:
        """Malformed coordinates must fail instead of changing lane membership."""
        cases = (
            (_row("A", hard=[0, 0], normal=[1], standard=[2], priority=[3]), "duplicate"),
            (_row("A", hard=[5], normal=[1], standard=[2], priority=[3]), "range"),
            (_row("A", hard=[0], normal=[0], standard=[2], priority=[3]), "overlap"),
        )
        for row, message in cases:
            with self.subTest(message=message):
                self._write_predictions((row, _row("B")), (row, _row("B")))
                with self.synthetic_config() as config:
                    with self.assertRaisesRegex(ValueError, message):
                        self._freeze_lanes(config)

    def test_rejects_manifest_disagreement_and_lane_total_mismatch(self) -> None:
        """Manifest counts and the immutable lane-total contract are binding."""
        self.manifest_291.write_text(
            json.dumps(
                {
                    "prediction_sha256": _sha(self.prediction_291),
                    "rows": 99,
                    "contains_test_labels": False,
                    "contains_evaluation_metrics": False,
                }
            ),
            encoding="utf-8",
        )
        with self.synthetic_config() as config:
            with self.assertRaisesRegex(ValueError, "row count"):
                self._freeze_lanes(config)
        self._write_predictions(self.rows, self.rows)
        with self.synthetic_config(expected_hard=2) as config:
            with self.assertRaisesRegex(ValueError, "total"):
                self._freeze_lanes(config)

    def test_context_lists_exact_database_population_and_rechecks_snapshot(self) -> None:
        """A changed database or unknown lane dataset cannot establish a run context."""
        with self.synthetic_config() as config:
            context = self._context(config)
            self.assertEqual(context.datasets, ("A", "B"))
            self.assertEqual(context.database_sha256, _sha(self.database))
            inputs.verify_input_snapshots(
                self.database,
                self.prediction_291,
                self.manifest_291,
                self.prediction_292,
                self.manifest_292,
                config=config,
            )
        with self.synthetic_config() as config:
            with closing(sqlite3.connect(self.database)) as connection, connection:
                connection.execute("INSERT INTO datasets(id, name) VALUES (3, 'C')")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                inputs.freeze_run_context(self.database, self._freeze_lanes(config), config=config)

    def test_rejects_frozen_lane_dataset_outside_database_population(self) -> None:
        """A lane coordinate for an absent dataset cannot enter a run context."""
        outside_rows = (
            _row("A", normal=[1], standard=[2], priority=[3]),
            _row("C", hard=[0]),
        )
        self._write_predictions(outside_rows, outside_rows)
        with self.synthetic_config() as config:
            lanes = self._freeze_lanes(config)
            with self.assertRaisesRegex(ValueError, "outside the database population"):
                inputs.list_dataset_names(self.database, lanes, config=config)

    def test_train_loader_uses_normal_train_values_and_preserves_identity_order(self) -> None:
        """Dropping the normal-only predicate would admit the synthetic abnormal row."""
        with self.synthetic_config() as config:
            context = self._context(config)
            statements: list[str] = []
            original = inputs._open_read_only

            def traced(path: Path):
                connection = original(path)
                connection.set_trace_callback(statements.append)
                return connection

            with patch.object(inputs, "_open_read_only", traced):
                bundle = inputs.load_raw_train_values(self.database, "A", context)
        self.assertEqual(tuple(series.instance_index for series in bundle.series), (0, 1))
        self.assertEqual(bundle.series[0].values, (4.0, 5.5))
        train_selects = [sql.lower() for sql in statements if "split = 'train'" in sql.lower()]
        self.assertTrue(train_selects)
        self.assertTrue(any("label = '0'" in sql for sql in train_selects))

    def test_test_loader_is_values_only_and_requires_real_train_contract_first(self) -> None:
        """Selecting a TEST label or resolving an input before the contract is a boundary breach."""
        with self.synthetic_config() as config:
            context = self._context(config)
            with self.assertRaisesRegex(ValueError, "FrozenTrainContract"):
                inputs.load_raw_test_values(Path("does-not-exist.sqlite"), "A", object(), context)
            statements: list[str] = []
            original = inputs._open_read_only

            def traced(path: Path):
                connection = original(path)
                connection.set_trace_callback(statements.append)
                return connection

            with patch.object(inputs, "_open_read_only", traced):
                bundle = inputs.load_raw_test_values(self.database, "A", self._train_contract(), context)
        self.assertEqual(tuple(series.instance_index for series in bundle.series), (2, 4))
        self.assertEqual(bundle.series[1].values, (1.5, -2.0, 3.25))
        test_selects = [sql.lower() for sql in statements if "split = 'test'" in sql.lower()]
        self.assertTrue(test_selects)
        for sql in test_selects:
            self.assertNotIn("label", sql.split("from", 1)[0])
            self.assertNotIn("labels_blob", sql.split("from", 1)[0])

    def test_rejects_non_float32_blob_width_without_transforming_values(self) -> None:
        """Accepting a partial float would implicitly pad or truncate raw input."""
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                "UPDATE instances SET values_blob = ? WHERE dataset_id = 1 AND split = 'TEST' AND instance_index = 2",
                (b"abc",),
            )
        with self.synthetic_config(database_sha256=_sha(self.database)) as config:
            context = self._context(config)
            with self.assertRaisesRegex(ValueError, "multiple of four"):
                inputs.load_raw_test_values(self.database, "A", self._train_contract(), context)

    def test_static_label_isolation_and_query_contract_hold(self) -> None:
        """Adding a forbidden input path or TEST selected field must fail preflight."""
        assert_package_has_no_label_path(Path(inputs.__file__).parent)
        inputs.assert_label_free_sql_contract()

    def test_sql_guard_rejects_label_anywhere_in_test_query_but_allows_train_predicate(self) -> None:
        """A TEST predicate using labels is forbidden even when SELECT stays values-only."""
        self.assertIn("label", inputs._TRAIN_VALUES_SQL.lower())
        mutated_test_sql = inputs._TEST_VALUES_SQL.replace(
            "WHERE d.name = ?", "WHERE i.label = '0' AND d.name = ?"
        )
        with patch.object(inputs, "_TEST_VALUES_SQL", mutated_test_sql):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                inputs.assert_label_free_sql_contract()

    def test_open_read_only_connection_rejects_writes(self) -> None:
        """Changing URI mode would permit a synthetic SQLite write and fail this test."""
        connection = inputs._open_read_only(self.database)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("INSERT INTO datasets(id, name) VALUES (99, 'write-forbidden')")
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
