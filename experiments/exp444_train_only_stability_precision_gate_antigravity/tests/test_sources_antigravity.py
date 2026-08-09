"""Adversarial tests for Exp444's frozen-source and values-only boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp444_train_only_stability_precision_gate_antigravity import sources_antigravity as sources
from experiments.exp444_train_only_stability_precision_gate_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    CandidateInputAntigravity,
    FrozenSourceSnapshotAntigravity,
    SourceCandidateAntigravity,
    candidate_attestation_sha256_antigravity,
)


def _jsonl_bytes_antigravity(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        for row in rows
    )


def _candidate_row_antigravity(dataset: str, index: int, rank: int, *, selected: bool = True) -> dict[str, object]:
    return {
        "agreement_count": 2,
        "dataset": dataset,
        "depths": [1, 2],
        "global_rank": rank,
        "instance_index": index,
        "raw_length": 8,
        "second_empirical_rank": 0.1,
        "second_excess_ratio": 1.2,
        "selected": selected,
        "selection_rank": rank if selected else None,
    }


def _candidate_populations_antigravity() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranking = [
        _candidate_row_antigravity(f"Dataset{rank:02d}", rank - 1, rank, selected=rank <= 30)
        for rank in range(1, 50)
    ]
    return [dict(row) for row in ranking[:30]], ranking


def _remove_retained_pin_antigravity(pin: Path) -> None:
    if pin.exists():
        pin.unlink()
    if pin.parent.exists():
        for child in tuple(pin.parent.iterdir()):
            child.unlink()
        pin.parent.rmdir()


class Exp444SourceBoundaryAntigravityTests(unittest.TestCase):
    def tearDown(self) -> None:
        sources._SOURCE_PRE_OPEN_HOOK_ANTIGRAVITY = None
        sources._DATABASE_PRE_OPEN_HOOK_ANTIGRAVITY = None
        sources._CLOSE_STEP_HOOK_ANTIGRAVITY = None

    def test_real_source_snapshot_has_exact_thirty_coordinates_antigravity(self) -> None:
        snapshot = sources.verify_frozen_sources_antigravity()
        self.addCleanup(snapshot.session.close)
        self.assertEqual(len(snapshot.candidates), 30)
        self.assertEqual(len({item.dataset for item in snapshot.candidates}), 30)
        self.assertEqual(
            snapshot.selected_coordinate_identity_sha256,
            "db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048",
        )
        self.assertEqual(snapshot.label_access_count, 0)

    def test_fabricated_snapshots_are_rejected_before_loader_open_antigravity(self) -> None:
        one = SourceCandidateAntigravity("A", 0, 4, (1, 2), 1)
        with self.assertRaisesRegex(ValueError, "exactly 30"):
            FrozenSourceSnapshotAntigravity(
                candidates=(one,),
                selected_coordinate_identity_sha256="0" * 64,
                candidate_attestation_sha256="0" * 64,
                source_hashes={"database": "0" * 64},
                database_path="/tmp/does-not-open-antigravity.db",
            )

    def test_loader_rejects_altered_candidate_metadata_before_any_query_antigravity(self) -> None:
        verified = sources.verify_frozen_sources_antigravity()
        self.addCleanup(verified.session.close)
        altered = list(verified.candidates)
        first = altered[0]
        altered[0] = SourceCandidateAntigravity(
            first.dataset,
            first.instance_index,
            first.raw_length + 1,
            first.original_depths,
            first.original_rank,
        )
        forged = FrozenSourceSnapshotAntigravity(
            candidates=tuple(altered),
            selected_coordinate_identity_sha256=verified.selected_coordinate_identity_sha256,
            candidate_attestation_sha256=candidate_attestation_sha256_antigravity(altered),
            source_hashes=verified.source_hashes,
            database_path=verified.database_path,
            session=verified.session,
        )
        before = verified.session.execution_count
        with self.assertRaisesRegex(ValueError, "candidate signatures"):
            sources.load_candidate_values_antigravity(forged)
        self.assertEqual(verified.session.execution_count, before)

    def test_loader_requires_parent_owned_verified_session_before_database_open_antigravity(self) -> None:
        verified = sources.verify_frozen_sources_antigravity()
        try:
            detached = FrozenSourceSnapshotAntigravity(
                candidates=verified.candidates,
                selected_coordinate_identity_sha256=verified.selected_coordinate_identity_sha256,
                candidate_attestation_sha256=verified.candidate_attestation_sha256,
                source_hashes=verified.source_hashes,
                database_path=verified.database_path,
                session=None,
            )
        finally:
            verified.session.close()
        with self.assertRaisesRegex(ValueError, "verified source session"):
            sources.load_candidate_values_antigravity(detached)

    def test_reversed_snapshot_rank_order_is_rejected_before_query_antigravity(self) -> None:
        verified = sources.verify_frozen_sources_antigravity()
        self.addCleanup(verified.session.close)
        before = verified.session.execution_count
        with self.assertRaisesRegex(ValueError, "rank order"):
            FrozenSourceSnapshotAntigravity(
                candidates=tuple(reversed(verified.candidates)),
                selected_coordinate_identity_sha256=verified.selected_coordinate_identity_sha256,
                candidate_attestation_sha256=candidate_attestation_sha256_antigravity(tuple(reversed(verified.candidates))),
                source_hashes=verified.source_hashes,
                database_path=verified.database_path,
                session=verified.session,
            )
        self.assertEqual(verified.session.execution_count, before)

    def test_public_session_rejects_jointly_forged_depths_before_database_open_antigravity(self) -> None:
        verified = sources.verify_frozen_sources_antigravity()
        candidates = list(verified.candidates)
        first = candidates[0]
        candidates[0] = SourceCandidateAntigravity(
            first.dataset,
            first.instance_index,
            first.raw_length,
            (first.original_depths[0], first.original_depths[-1] + 1),
            first.original_rank,
        )
        verified.session.close()
        source_path = Path(FROZEN_CONFIG.database_path)
        before_links = source_path.stat().st_nlink
        escaped = None
        try:
            with self.assertRaisesRegex(ValueError, "verified replay"):
                escaped = sources.ValuesOnlyDatabaseSessionAntigravity(
                    source_path,
                    expected_sha256=FROZEN_CONFIG.database_sha256,
                    candidates=tuple(candidates),
                )
        finally:
            if escaped is not None:
                escaped.close()
        self.assertEqual(source_path.stat().st_nlink, before_links)

    def test_public_session_rejects_caller_chosen_database_and_hash_antigravity(self) -> None:
        verified = sources.verify_frozen_sources_antigravity()
        candidates = verified.candidates
        verified.session.close()
        with tempfile.TemporaryDirectory(prefix="exp444_constructor_escape_antigravity_") as directory:
            database = Path(directory) / "caller_antigravity.db"
            database.write_bytes(b"caller chosen database")
            opened: list[Path] = []
            sources._DATABASE_PRE_OPEN_HOOK_ANTIGRAVITY = opened.append
            with self.assertRaisesRegex(ValueError, "verified replay"):
                sources.ValuesOnlyDatabaseSessionAntigravity(
                    database,
                    expected_sha256="0" * 64,
                    candidates=candidates,
                )
            self.assertEqual(opened, [])
            self.assertFalse(hasattr(sources.ValuesOnlyDatabaseSessionAntigravity, "_from_test_fixture_antigravity"))
        duplicate_dataset = tuple(
            SourceCandidateAntigravity("A", rank - 1, 8, (1, 2), rank)
            for rank in range(1, 31)
        )
        with self.assertRaisesRegex(ValueError, "one candidate per dataset"):
            FrozenSourceSnapshotAntigravity(
                candidates=duplicate_dataset,
                selected_coordinate_identity_sha256=FROZEN_CONFIG.exp442_selected_coordinate_identity_sha256,
                candidate_attestation_sha256="0" * 64,
                source_hashes={"database": "0" * 64},
                database_path="/tmp/does-not-open-antigravity.db",
            )
        arbitrary = tuple(
            SourceCandidateAntigravity(f"Fake{rank:02d}", rank - 1, 8, (1, 2), rank)
            for rank in range(1, 31)
        )
        with self.assertRaisesRegex(ValueError, "coordinate identity"):
            FrozenSourceSnapshotAntigravity(
                candidates=arbitrary,
                selected_coordinate_identity_sha256=FROZEN_CONFIG.exp442_selected_coordinate_identity_sha256,
                candidate_attestation_sha256=candidate_attestation_sha256_antigravity(arbitrary),
                source_hashes={"database": "0" * 64},
                database_path="/tmp/does-not-open-antigravity.db",
            )

    def test_prediction_sql_rejects_test_labels_labels_blob_star_and_unknowns_antigravity(self) -> None:
        cases = (
            ("SELECT instance_index, label FROM instances WHERE split='TEST'", "label-capable SQL"),
            ("SELECT instance_index, labels_blob FROM instances WHERE split='TEST'", "label-capable SQL"),
            ("SELECT * FROM instances", "unregistered prediction SQL"),
            ("SELECT made_up_column FROM instances", "unregistered prediction SQL"),
            ("SELECT instance_index, values_blob FROM made_up_table", "unregistered prediction SQL"),
        )
        for sql, message in cases:
            with self.subTest(sql=sql), self.assertRaisesRegex(ValueError, message):
                sources.validate_prediction_sql_antigravity(sql)

    def test_exact_values_sql_shapes_remain_frozen_antigravity(self) -> None:
        self.assertEqual(
            sources.NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY,
            """SELECT i.instance_index, i.values_blob
FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TRAIN' AND i.label = '0'
ORDER BY i.instance_index""",
        )
        self.assertEqual(
            sources.BOUND_TEST_VALUE_SQL_ANTIGRAVITY,
            """SELECT i.instance_index, i.values_blob
FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST' AND i.instance_index = ?""",
        )

    def test_unbound_test_coordinate_is_denied_before_execution_counters_change_antigravity(self) -> None:
        snapshot = sources.verify_frozen_sources_antigravity()
        session = snapshot.session
        with session:
            candidate = snapshot.candidates[0]
            with self.assertRaisesRegex(ValueError, "unbound TEST coordinate"):
                session.query_antigravity(
                    sources.BOUND_TEST_VALUE_SQL_ANTIGRAVITY,
                    (candidate.dataset, candidate.instance_index + 1),
                )
            self.assertEqual(session.sql_trace, [])
            self.assertEqual(session.execution_count, 0)
            self.assertEqual(session.label_access_count, 0)
            self.assertEqual(session.denied_sql_attempts, 1)

    def test_unbound_train_dataset_and_forbidden_sql_have_zero_execution_and_label_access_antigravity(self) -> None:
        snapshot = sources.verify_frozen_sources_antigravity()
        session = snapshot.session
        with session:
            denied = (
                (sources.NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY, ("not-frozen-antigravity",)),
                ("SELECT instance_index, label FROM instances WHERE split='TEST'", ()),
                ("SELECT instance_index, labels_blob FROM instances WHERE split='TEST'", ()),
                ("SELECT * FROM instances", ()),
                ("SELECT made_up_column FROM instances", ()),
            )
            for sql, parameters in denied:
                with self.subTest(sql=sql), self.assertRaises(ValueError):
                    session.query_antigravity(sql, parameters)
            self.assertEqual(session.execution_count, 0)
            self.assertEqual(session.sql_trace, [])
            self.assertEqual(session.label_access_count, 0)
            self.assertEqual(session.denied_sql_attempts, len(denied))

    def test_allowed_values_queries_execute_with_no_label_access_antigravity(self) -> None:
        snapshot = sources.verify_frozen_sources_antigravity()
        session = snapshot.session
        with session:
            candidate = snapshot.candidates[0]
            train = session.query_antigravity(sources.NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY, (candidate.dataset,))
            test = session.query_antigravity(
                sources.BOUND_TEST_VALUE_SQL_ANTIGRAVITY,
                (candidate.dataset, candidate.instance_index),
            )
            self.assertGreaterEqual(len(train), 4)
            self.assertEqual(tuple(row[0] for row in test), (candidate.instance_index,))
            self.assertEqual(session.execution_count, 2)
            self.assertEqual(session.label_access_count, 0)

    def test_values_preparation_excludes_nonfinite_rows_and_returns_read_only_arrays_antigravity(self) -> None:
        candidate = SourceCandidateAntigravity("A", 0, 4, (1, 2), 1)
        train_rows = (
            (0, np.asarray([1, 2, 3, 4], dtype="<f4").tobytes()),
            (1, np.asarray([1, np.nan, 3, 4], dtype="<f4").tobytes()),
            (2, np.asarray([2, 3, 4, 5], dtype="<f4").tobytes()),
            (3, np.asarray([3, 4, 5, 6], dtype="<f4").tobytes()),
            (4, np.asarray([4, 5, 6, 7], dtype="<f4").tobytes()),
            (5, np.asarray([9, 9, 9], dtype="<f4").tobytes()),
        )
        test_rows = ((0, np.asarray([6, 7, 8, 9], dtype="<f4").tobytes()),)
        result = sources._prepare_candidate_input_antigravity(candidate, train_rows, test_rows)
        self.assertIsInstance(result, CandidateInputAntigravity)
        self.assertEqual(result.train_indices, (0, 2, 3, 4))
        self.assertFalse(result.train_values.flags.writeable)
        self.assertFalse(result.test_values.flags.writeable)

    def test_values_preparation_rejects_noncontiguous_missing_or_duplicate_test_identity_antigravity(self) -> None:
        candidate = SourceCandidateAntigravity("A", 2, 4, (1, 2), 1)
        finite = np.asarray([1, 2, 3, 4], dtype="<f4").tobytes()
        train_rows = tuple((index, finite) for index in range(4))
        for rows in ((), ((1, finite),), ((2, finite), (2, finite))):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "bound TEST values row"):
                sources._prepare_candidate_input_antigravity(candidate, train_rows, rows)

    def test_regular_source_rejects_path_replacement_symlink_fifo_and_directory_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_regular_source_antigravity_") as directory:
            root = Path(directory)
            source = root / "source_antigravity.json"
            replacement = root / "replacement_antigravity.json"
            source.write_bytes(b"{}")
            replacement.write_bytes(b'{"changed":true}')

            def replace(_path: Path) -> None:
                os.replace(replacement, source)

            sources._SOURCE_PRE_OPEN_HOOK_ANTIGRAVITY = replace
            with self.assertRaisesRegex(ValueError, "changed before read"):
                sources._read_regular_bytes_antigravity(source)
            sources._SOURCE_PRE_OPEN_HOOK_ANTIGRAVITY = None

            target = root / "target_antigravity.json"
            target.write_bytes(b"{}")
            symlink = root / "symlink_antigravity.json"
            symlink.symlink_to(target)
            fifo = root / "fifo_antigravity"
            os.mkfifo(fifo)
            for path in (symlink, fifo, root):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "regular linked file"):
                    sources._read_regular_bytes_antigravity(path)

    def test_json_firewall_rejects_truncated_jsonl_blank_rows_and_duplicate_keys_antigravity(self) -> None:
        for payload in (b'{"a":1}', b'{"a":1}\n\n'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                sources._parse_jsonl_antigravity(payload, "fixture")
        with self.assertRaises(sources.DuplicateJsonKeyError):
            sources._parse_json_object_antigravity(b'{"a":1,"a":2}', "fixture")

    def test_candidate_population_rejects_duplicates_rank_gaps_and_altered_dimensions_antigravity(self) -> None:
        prediction, ranking = _candidate_populations_antigravity()
        duplicate = [dict(row) for row in prediction]
        duplicate[1]["dataset"] = duplicate[0]["dataset"]
        duplicate[1]["instance_index"] = duplicate[0]["instance_index"]
        with self.assertRaisesRegex(ValueError, "duplicate candidates"):
            sources._parse_prediction_and_ranking_antigravity(
                _jsonl_bytes_antigravity(duplicate), _jsonl_bytes_antigravity(ranking)
            )
        rank_gap = [dict(row) for row in prediction]
        rank_gap[0]["selection_rank"] = 31
        with self.assertRaisesRegex(ValueError, "ranks are not 1-30"):
            sources._parse_prediction_and_ranking_antigravity(
                _jsonl_bytes_antigravity(rank_gap), _jsonl_bytes_antigravity(ranking)
            )
        for field, value in (("raw_length", 0), ("depths", [2, 1])):
            altered = dict(prediction[0])
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                sources._validate_candidate_row_antigravity(altered, rank_kind="fixture")

    def test_candidate_replay_rejects_selection_prediction_ranking_signature_mismatch_antigravity(self) -> None:
        source = tuple(
            SourceCandidateAntigravity(f"D{rank:02d}", rank - 1, 8, (1, 2), rank)
            for rank in range(1, 31)
        )
        altered = list(source)
        altered[0] = SourceCandidateAntigravity("D01", 0, 9, (1, 2), 1)
        with self.assertRaisesRegex(ValueError, "prediction/selection candidate mismatch"):
            sources._validate_candidate_replay_agreement_antigravity(source, source, tuple(altered))
        altered_depths = list(source)
        altered_depths[0] = SourceCandidateAntigravity("D01", 0, 8, (1, 3), 1)
        with self.assertRaisesRegex(ValueError, "ranking/selection candidate mismatch"):
            sources._validate_candidate_replay_agreement_antigravity(source, tuple(altered_depths), source)

    def test_wrong_source_hash_and_artifact_map_fail_closed_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_artifact_map_antigravity_") as directory:
            root = Path(directory)
            source = root / "source_antigravity.json"
            source.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                sources._read_regular_bytes_antigravity(source, "0" * 64)
            artifact_map = root / "artifact_sha256_antigravity.json"
            artifact_map.write_text('{"schema":"wrong"}', encoding="utf-8")
            paths = sources.FrozenSourcePathsAntigravity(
                source, source, source, source, source, artifact_map, source, source, source
            )
            with self.assertRaisesRegex(ValueError, "artifact map is invalid"):
                sources._validate_artifact_map_antigravity(
                    paths, {"prediction": "0" * 64, "ranking": "0" * 64, "selection": "0" * 64}
                )

    def test_database_replacement_and_metadata_drift_fail_before_query_antigravity(self) -> None:
        for mutation in ("replace", "metadata"):
            with self.subTest(mutation=mutation):
                snapshot = sources.verify_frozen_sources_antigravity()
                session = snapshot.session
                pin = session.pin_path
                replacement = pin.parent / "replacement_antigravity.db"
                replacement.write_bytes(b"replacement database pin")
                if mutation == "metadata":
                    before = replacement.stat()
                    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
                os.replace(replacement, pin)
                candidate = snapshot.candidates[0]
                with self.assertRaisesRegex(ValueError, "database source or pin identity changed"):
                    session.query_antigravity(
                        sources.BOUND_TEST_VALUE_SQL_ANTIGRAVITY,
                        (candidate.dataset, candidate.instance_index),
                    )
                self.assertEqual(session.execution_count, 0)
                self.assertEqual(session.label_access_count, 0)
                with self.assertRaisesRegex(ValueError, "pinned database close failed"):
                    session.close()
                self.assertTrue(pin.exists())
                self.assertTrue(pin.parent.exists())
                _remove_retained_pin_antigravity(pin)

    def test_close_failure_injections_preserve_truthful_pin_and_root_antigravity(self) -> None:
        steps = ("before_descriptor_close", "before_pin_unlink", "after_pin_unlink", "before_root_rmdir")
        for step in steps:
            with self.subTest(step=step):
                database = Path(FROZEN_CONFIG.database_path)
                original_links = database.stat().st_nlink
                snapshot = sources.verify_frozen_sources_antigravity()
                session = snapshot.session
                pin = session.pin_path

                def fail(actual: str, _pin: Path, _root: Path) -> None:
                    if actual == step:
                        raise OSError(f"injected {step}")

                sources._CLOSE_STEP_HOOK_ANTIGRAVITY = fail
                with self.assertRaisesRegex(ValueError, "pinned database close failed"):
                    session.close()
                self.assertTrue(pin.exists(), step)
                self.assertTrue(pin.parent.exists(), step)
                self.assertEqual((pin.stat().st_dev, pin.stat().st_ino), session.descriptor_inode)
                self.assertEqual(database.stat().st_nlink, original_links + 1)
                sources._CLOSE_STEP_HOOK_ANTIGRAVITY = None
                _remove_retained_pin_antigravity(pin)
                self.assertEqual(database.stat().st_nlink, original_links)

    def test_successful_close_removes_only_owned_pin_and_restores_link_count_antigravity(self) -> None:
        database = Path(FROZEN_CONFIG.database_path)
        original_links = database.stat().st_nlink
        snapshot = sources.verify_frozen_sources_antigravity()
        session = snapshot.session
        pin = session.pin_path
        root = pin.parent
        session.close()
        self.assertFalse(pin.exists())
        self.assertFalse(root.exists())
        self.assertEqual(database.stat().st_nlink, original_links)

    def test_session_exposes_no_raw_connection_or_cursor_antigravity(self) -> None:
        snapshot = sources.verify_frozen_sources_antigravity()
        session = snapshot.session
        with session:
            for name in ("connection", "cursor", "raw_connection", "raw_cursor"):
                self.assertFalse(hasattr(session, name), name)


if __name__ == "__main__":
    unittest.main()
