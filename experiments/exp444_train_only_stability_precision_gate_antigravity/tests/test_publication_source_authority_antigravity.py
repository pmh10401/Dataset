"""Real-boundary regression coverage for Exp445 publication authority repair."""
from __future__ import annotations

import inspect
import unittest
from unittest import mock

from experiments.exp444_train_only_stability_precision_gate_antigravity import artifacts_antigravity as artifacts
from experiments.exp444_train_only_stability_precision_gate_antigravity import provenance_antigravity as provenance
from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
from experiments.exp444_train_only_stability_precision_gate_antigravity import sources_antigravity as sources


class Exp445PublicationSourceAuthorityAntigravityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary_bytes = provenance.verify_exp444_provenance_antigravity().dataset_summary_bytes

    def test_real_composition_uses_pinned_exp440_bytes_and_closes_values_session_antigravity(self) -> None:
        """Replacing pinned bytes, retaining a session, or deriving Exp442 summaries must fail this contract."""
        close_states: list[bool] = []
        original_close = sources.ValuesOnlyDatabaseSessionAntigravity.close
        original_query = sources.ValuesOnlyDatabaseSessionAntigravity.query_antigravity

        def checked_query(session, sql, parameters=()):
            self.assertIn(sql, (sources.NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY, sources.BOUND_TEST_VALUE_SQL_ANTIGRAVITY))
            return original_query(session, sql, parameters)

        def checked_close(session):
            result = original_close(session)
            close_states.append(session._closed)
            return result

        with (
            mock.patch.object(sources.ValuesOnlyDatabaseSessionAntigravity, "query_antigravity", new=checked_query),
            mock.patch.object(sources.ValuesOnlyDatabaseSessionAntigravity, "close", new=checked_close),
            mock.patch.object(reporting, "_exp442_dataset_summaries_antigravity", create=True, side_effect=AssertionError("Exp442 summary path access is forbidden")),
            mock.patch.object(reporting, "_fallback_dataset_summaries_antigravity", create=True, side_effect=AssertionError("summary fallback is forbidden")),
            mock.patch.object(reporting, "_synthesize_dataset_summaries_antigravity", create=True, side_effect=AssertionError("summary synthesis is forbidden")),
        ):
            values = reporting._capture_values_only_publish_authority_antigravity()
        self.assertEqual(close_states, [True])
        self.assertFalse(hasattr(values, "session"))
        with self.assertRaises(TypeError):
            values.source_hashes["replacement"] = "0" * 64
        with self.assertRaises(TypeError):
            values.control_rows[0]["replacement"] = "forbidden"
        summaries = reporting._parse_dataset_summaries_antigravity(self.summary_bytes)
        with mock.patch.object(reporting, "Path", side_effect=AssertionError("parser/composer path access is forbidden")):
            authority = reporting._compose_publish_source_authority_antigravity(summaries, values)
        self.assertEqual(authority.dataset_summaries, summaries)
        self.assertEqual((len(authority.candidates), len(authority.dataset_summaries), len(authority.control_rows)), (30, 1117, 30))

    def test_publication_wrapper_exposes_no_source_authority_override_antigravity(self) -> None:
        """Accepting source bytes or a handoff publicly would bypass provenance capture."""
        self.assertEqual(tuple(inspect.signature(reporting.publish_prediction_antigravity).parameters), ("attempt_root",))

    def test_values_only_handoff_is_immutable_after_source_close_antigravity(self) -> None:
        """A mutable detached handoff could be altered after the SQLite boundary closes."""
        snapshot = sources.verify_frozen_sources_antigravity()
        session = snapshot.session
        try:
            handoff = reporting._ValuesOnlyPublishAuthorityAntigravity(
                snapshot.candidates,
                snapshot.source_hashes,
                tuple({"candidate": item.to_payload()} for item in snapshot.candidates),
            )
        finally:
            assert session is not None
            session.close()
        with self.assertRaises(TypeError):
            handoff.source_hashes["replacement"] = "0" * 64
        with self.assertRaises(TypeError):
            handoff.control_rows[0]["replacement"] = "forbidden"

    def test_summary_parser_rejects_noncanonical_malformed_order_count_and_label_rows_antigravity(self) -> None:
        """A parser accepting any malformed or label-bearing summary would lose Exp440 authority."""
        lines = self.summary_bytes.splitlines(keepends=True)
        first = artifacts.parse_canonical_json_bytes_antigravity(lines[0])
        self.assertIsInstance(first, dict)
        label_row = dict(first)
        label_row["label"] = 0
        cases = (
            (b'{"dataset": "not-canonical"}\n', "invalid|canonical"),
            (b'not-json\n', "invalid|canonical"),
            (b"".join((lines[1], lines[0], *lines[2:])), "canonical order"),
            (b"".join(lines[:-1]), "count"),
            (b"".join((artifacts.canonical_json_bytes_antigravity(label_row), *lines[1:])), "forbidden prediction field"),
        )
        for payload, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                reporting._parse_dataset_summaries_antigravity(payload)


if __name__ == "__main__":
    unittest.main()
