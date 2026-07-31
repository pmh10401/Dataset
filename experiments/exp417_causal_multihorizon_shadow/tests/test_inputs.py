"""Synthetic boundary tests for Exp417 frozen values-only inputs."""

from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import replace
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shutil
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
        return inputs.freeze_run_context(
            self.database,
            self._freeze_lanes(config),
            verified_code_commit="f" * 40,
            config=config,
        )

    def test_verified_connection_loads_multiple_train_datasets_with_one_source_digest(self) -> None:
        """A worker session must not rehash the pinned database per dataset."""
        with self.synthetic_config() as config, patch.object(
            inputs, "_sha256_descriptor", wraps=inputs._sha256_descriptor
        ) as digest:
            with inputs.verified_read_only_connection(
                self.database, config.database_sha256
            ) as connection:
                first = inputs.load_raw_train_values_from_connection(
                    connection,
                    "A",
                    database_sha256=config.database_sha256,
                    datasets=("A", "B"),
                )
                second = inputs.load_raw_train_values_from_connection(
                    connection,
                    "B",
                    database_sha256=config.database_sha256,
                    datasets=("A", "B"),
                )
        self.assertEqual(first.dataset, "A")
        self.assertEqual(second.dataset, "B")
        self.assertEqual(digest.call_count, 1)

    def test_verified_connection_rejects_database_inode_replacement_on_close(self) -> None:
        """A query session cannot accept an ABA replacement of its source DB."""
        replacement = self.root / "replacement.sqlite"
        with self.synthetic_config() as config, self.assertRaisesRegex(
            ValueError, "inode changed"
        ):
            with inputs.verified_read_only_connection(
                self.database, config.database_sha256
            ) as connection:
                inputs.load_raw_train_values_from_connection(
                    connection,
                    "A",
                    database_sha256=config.database_sha256,
                    datasets=("A", "B"),
                )
                shutil.copyfile(self.database, replacement)
                os.replace(replacement, self.database)

    def test_lane_snapshot_verification_does_not_rehash_the_database(self) -> None:
        """A session-backed worker verifies lane artifacts without a second DB pass."""
        with self.synthetic_config() as config, patch.object(
            inputs, "_verify_hashes"
        ) as verify_hashes, patch.object(
            inputs, "_verify_database_hash"
        ) as verify_database:
            inputs.verify_lane_input_snapshots(
                self.database,
                self.database,
                self.database,
                self.database,
            )
        verify_hashes.assert_called_once()
        verify_database.assert_not_called()

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

    def test_path_mutation_after_pinned_capture_cannot_change_parsed_evidence(self) -> None:
        """Parsing captured bytes is independent of later pathname replacement."""
        prediction_paths = (self.prediction_291, self.prediction_292)
        manifest_paths = (self.manifest_291, self.manifest_292)
        for artifact in prediction_paths + manifest_paths:
            with self.subTest(artifact=artifact.name):
                self._write_predictions(self.rows, self.rows)
                with self.synthetic_config() as config:
                    if artifact in prediction_paths:
                        original = inputs._parse_prediction_rows

                        def mutate_prediction(payload: bytes, name: str):
                            if name == artifact.name:
                                artifact.write_text("changed\n", encoding="utf-8")
                            return original(payload, name)

                        replacement = mutate_prediction
                        target = "_parse_prediction_rows"
                    else:
                        original = inputs._parse_manifest

                        def mutate_manifest(payload: bytes, name: str):
                            if name == artifact.name:
                                artifact.write_text("changed\n", encoding="utf-8")
                            return original(payload, name)

                        replacement = mutate_manifest
                        target = "_parse_manifest"
                    with patch.object(inputs, target, replacement):
                        snapshot = self._freeze_lanes(config)
                    self.assertEqual(snapshot.hard["A"], (0,))

    def test_lane_hash_and_parse_share_one_pinned_descriptor_across_aba_swap(self) -> None:
        """Restoring a pathname after parsing foreign bytes must not make them trusted."""
        malicious_rows = (
            _row("A", hard=[4], normal=[1], standard=[2], priority=[3]),
            _row("B"),
        )
        malicious_paths: dict[Path, Path] = {}
        for target in (self.prediction_291, self.prediction_292):
            malicious = target.with_suffix(target.suffix + ".malicious")
            malicious.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in malicious_rows
                ),
                encoding="utf-8",
            )
            malicious_paths[target] = malicious

        with self.synthetic_config() as config:
            real_open = io.open

            def aba_open(file: object, mode: str = "r", *args: object, **kwargs: object):
                path = Path(file)
                if path in malicious_paths and "r" in mode and "b" not in mode:
                    original = path.with_suffix(path.suffix + ".trusted")
                    malicious = malicious_paths[path]
                    os.replace(path, original)
                    os.replace(malicious, path)
                    handle = real_open(path, mode, *args, **kwargs)
                    os.replace(path, malicious)
                    os.replace(original, path)
                    return handle
                return real_open(file, mode, *args, **kwargs)

            with patch("io.open", side_effect=aba_open):
                lanes = self._freeze_lanes(config)

        self.assertEqual(lanes.hard, {"A": (0,)})

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
            self.assertEqual(context.test_identity_manifest, {"A": ((2, 2), (4, 3)), "B": ((0, 2),)})
            self.assertEqual(context.test_sizes, {"A": 2, "B": 1})
            with self.assertRaises(TypeError):
                context.test_sizes["A"] = 999  # type: ignore[index]
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
                inputs.freeze_run_context(
                    self.database,
                    self._freeze_lanes(config),
                    verified_code_commit="f" * 40,
                    config=config,
                )

    def test_context_uses_the_explicitly_verified_head_without_cwd_git_lookup(self) -> None:
        """The reviewed clean HEAD, not the caller's current repository, is provenance."""
        with self.synthetic_config() as config, patch.object(
            inputs, "_git_commit", side_effect=AssertionError("CWD Git lookup reached")
        ):
            context = inputs.freeze_run_context(
                self.database,
                self._freeze_lanes(config),
                verified_code_commit="d" * 40,
                config=config,
            )
        self.assertEqual(context.code_commit, "d" * 40)

    def test_frozen_test_manifest_rejects_database_row_mutation_before_loading(self) -> None:
        """Changing a valid-width TEST row after context freeze cannot replace its identity manifest."""
        with self.synthetic_config() as config:
            context = self._context(config)
            with closing(sqlite3.connect(self.database)) as connection, connection:
                connection.execute(
                    "UPDATE instances SET values_blob = ? WHERE dataset_id = 1 AND split = 'TEST' AND instance_index = 2",
                    (np.array((1.0, 2.0, 3.0, 4.0), dtype="<f4").tobytes(),),
                )
            with self.assertRaisesRegex(ValueError, "SHA256"):
                inputs.load_raw_test_values(self.database, "A", self._train_contract(), context)

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

    def test_database_rows_outside_frozen_lane_universe_are_not_population(self) -> None:
        """A DB-only source row must not expand the hash-frozen 1,117 universe."""
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                "INSERT INTO datasets(id, name) VALUES (3, 'DB-only-source')"
            )
        with self.synthetic_config(database_sha256=_sha(self.database)) as config:
            lanes = self._freeze_lanes(config)
            self.assertEqual(
                inputs.list_dataset_names(self.database, lanes, config=config),
                ("A", "B"),
            )

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

    def test_database_query_is_bound_to_verified_inode_across_path_aba(self) -> None:
        """A foreign SQLite inode visible only during connect must never supply rows."""
        malicious = self.root / "malicious.sqlite"
        shutil.copy2(self.database, malicious)
        with closing(sqlite3.connect(malicious)) as connection, connection:
            connection.execute(
                "UPDATE instances SET values_blob = ? "
                "WHERE dataset_id = 1 AND split = 'TRAIN' AND instance_index = 0",
                (np.array((99.0, 100.0), dtype="<f4").tobytes(),),
            )

        with self.synthetic_config() as config:
            context = self._context(config)
            original_open = inputs._open_read_only

            def aba_connect(source: object):
                if isinstance(source, Path):
                    trusted = self.root / "trusted.sqlite"
                    os.replace(self.database, trusted)
                    os.replace(malicious, self.database)
                    connection = original_open(self.database)
                    os.replace(self.database, malicious)
                    os.replace(trusted, self.database)
                    return connection
                return original_open(source)

            with patch.object(inputs, "_open_read_only", side_effect=aba_connect):
                bundle = inputs.load_raw_train_values(
                    self.database, "A", context
                )

        self.assertEqual(bundle.series[0].values, (4.0, 5.5))

    def test_sqlite_queries_use_an_immutable_descriptor_uri(self) -> None:
        """A pathname mode=ro URI is not an inode-bound immutable snapshot."""
        with self.synthetic_config() as config:
            context = self._context(config)
            uris: list[str] = []
            real_connect = sqlite3.connect

            def observe(database: object, *args: object, **kwargs: object):
                uris.append(str(database))
                return real_connect(database, *args, **kwargs)

            with patch.object(inputs.sqlite3, "connect", side_effect=observe):
                inputs.load_raw_train_values(self.database, "A", context)

        self.assertTrue(
            any(
                "immutable=1" in uri
                and ("/dev/fd/" in uri or "/proc/self/fd/" in uri)
                for uri in uris
            )
        )

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
            with self.assertRaisesRegex(ValueError, "raw float32 length"):
                self._context(config)

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
