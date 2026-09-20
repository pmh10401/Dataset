"""Synthetic-only tests for Exp454's sole TEST-label session boundary."""

from __future__ import annotations

from dataclasses import replace
import os
import unittest

from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.tests.test_predecessor_antigravity import (
    SyntheticExp453PredecessorFixtureAntigravity,
)


class CompleteConnectionAntigravity:
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        self.query_calls = 0
        self.close_calls = 0
        self.parameters: list[tuple[object, ...]] = []
        self.close_error = close_error

    def execute(self, sql: str, parameters: tuple[object, ...]):  # type: ignore[no-untyped-def]
        self.query_calls += 1
        self.parameters.append(parameters)
        self.last_sql = sql
        size = 1518 if self.query_calls == 48 else 1506
        return ((index, "1" if index % 2 else "0") for index in range(size))

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class SourcesAntigravityTest(unittest.TestCase):
    def setUp(self) -> None:
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )

        self.fixture = SyntheticExp453PredecessorFixtureAntigravity()
        self.authority = predecessor._capture_predecessors_with_dependencies_antigravity(
            self.fixture.dependencies_antigravity()
        )

    def tearDown(self) -> None:
        self.fixture.close_antigravity()

    def _dependencies_antigravity(self, sources, connection, events):  # type: ignore[no-untyped-def]
        descriptor = os.open("/dev/null", os.O_RDONLY)
        return sources._LabelDependenciesAntigravity(
            verify_database=lambda: (descriptor, (1, 2, 3, 1, 5, 6, 7)),
            open_connection=lambda opened: connection,
            assert_database_unchanged=lambda opened, identity: None,
            close_descriptor=os.close,
            audit_sink=events.append,
        )

    def test_forged_and_cloned_authorities_fail_before_database_setup_antigravity(self) -> None:
        """Break caught: a caller-minted or value-equal cloned object opens the database."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            sources_antigravity as sources,
        )

        reached: list[str] = []
        dependencies = sources._LabelDependenciesAntigravity(
            verify_database=lambda: reached.append("database"),  # type: ignore[arg-type,return-value]
            open_connection=lambda _descriptor: reached.append("connection"),  # type: ignore[arg-type,return-value]
            assert_database_unchanged=lambda _descriptor, _identity: None,
            close_descriptor=lambda _descriptor: None,
            audit_sink=lambda _audit: None,
        )
        for candidate in (object(), replace(self.authority)):
            with self.subTest(candidate=type(candidate).__name__), self.assertRaisesRegex(ValueError, "authority"):
                sources._open_test_labels_once_with_dependencies_antigravity(candidate, dependencies)
        self.assertEqual(reached, [])

    def test_clean_session_is_exactly_one_setup_48_ordered_queries_and_72300_rows_antigravity(self) -> None:
        """Break caught: the label session changes order/count/type or remains reusable after close."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            sources_antigravity as sources,
        )

        events: list[object] = []
        connection = CompleteConnectionAntigravity()
        dependencies = self._dependencies_antigravity(sources, connection, events)
        session = sources._open_test_labels_once_with_dependencies_antigravity(self.authority, dependencies)
        for dataset in self.authority.dataset_order:
            labels = session.labels_for_dataset_antigravity(dataset)
            self.assertTrue(labels)
            self.assertEqual(tuple(index for index, _label in labels), tuple(range(len(labels))))
            self.assertTrue(all(type(label) is int and label in (0, 1) for _index, label in labels))
        session.close_antigravity()
        audit = session.audit_antigravity()
        self.assertEqual((audit.session_count, audit.setup_attempts, audit.setup_completions), (1, 1, 1))
        self.assertEqual((audit.query_attempts, audit.query_completions, audit.label_rows), (48, 48, 72300))
        self.assertEqual((audit.close_attempts, audit.close_completions, audit.state), (1, 1, "closed"))
        self.assertIsNone(audit.primary_error)
        self.assertIsNone(audit.cleanup_error)
        self.assertEqual(connection.query_calls, 48)
        self.assertEqual(connection.close_calls, 1)
        self.assertEqual(connection.parameters, [(dataset,) for dataset in self.authority.dataset_order])
        self.assertIn("i.label", connection.last_sql)
        with self.assertRaisesRegex(ValueError, "closed"):
            session.labels_for_dataset_antigravity(self.authority.dataset_order[0])
        session.close_antigravity()
        self.assertEqual(connection.close_calls, 1)
        with self.assertRaisesRegex(ValueError, "consumed"):
            sources._open_test_labels_once_with_dependencies_antigravity(self.authority, dependencies)
        self.assertTrue(events)
        self.assertEqual(events[-1], audit)

    def test_setup_failure_is_audited_consumed_and_never_retried_antigravity(self) -> None:
        """Break caught: failed database setup is invisible or the same capability retries."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            sources_antigravity as sources,
        )

        attempts: list[str] = []
        events: list[object] = []
        dependencies = sources._LabelDependenciesAntigravity(
            verify_database=lambda: (attempts.append("setup"), (_ for _ in ()).throw(OSError("setup boom")))[1],  # type: ignore[return-value]
            open_connection=lambda _descriptor: None,
            assert_database_unchanged=lambda _descriptor, _identity: None,
            close_descriptor=lambda _descriptor: None,
            audit_sink=events.append,
        )
        with self.assertRaisesRegex(OSError, "setup boom"):
            sources._open_test_labels_once_with_dependencies_antigravity(self.authority, dependencies)
        self.assertEqual(attempts, ["setup"])
        self.assertEqual(events[-1].state, "setup_failed")
        self.assertEqual(events[-1].setup_attempts, 1)
        self.assertEqual(events[-1].setup_completions, 0)
        self.assertEqual(events[-1].primary_error, "OSError: setup boom")
        with self.assertRaisesRegex(ValueError, "consumed"):
            sources._open_test_labels_once_with_dependencies_antigravity(self.authority, dependencies)
        self.assertEqual(attempts, ["setup"])

    def test_wrong_order_and_query_failure_are_terminal_with_one_cleanup_antigravity(self) -> None:
        """Break caught: an invalid query can continue, retry, or conceal its cleanup state."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            sources_antigravity as sources,
        )

        events: list[object] = []
        connection = CompleteConnectionAntigravity()
        session = sources._open_test_labels_once_with_dependencies_antigravity(
            self.authority,
            self._dependencies_antigravity(sources, connection, events),
        )
        with self.assertRaisesRegex(ValueError, "frozen dataset order"):
            session.labels_for_dataset_antigravity("wrong_antigravity")
        audit = session.audit_antigravity()
        self.assertEqual(audit.state, "terminal")
        self.assertEqual((audit.query_attempts, audit.query_completions, audit.label_rows), (0, 0, 0))
        self.assertEqual(connection.close_calls, 1)
        with self.assertRaisesRegex(ValueError, "frozen dataset order"):
            session.close_antigravity()
        self.assertEqual(connection.close_calls, 1)

    def test_bad_row_type_index_population_and_iterator_failure_are_terminal_antigravity(self) -> None:
        """Break caught: malformed, noncontiguous, empty, or interrupted label rows are accepted."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            sources_antigravity as sources,
        )

        cases: dict[str, object] = {
            "empty": (),
            "bool_index": ((True, "0"),),
            "gap": ((1, "0"),),
            "bad_label": ((0, 0),),
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                authority = predecessor._capture_predecessors_with_dependencies_antigravity(
                    self.fixture.dependencies_antigravity()
                )
                connection = CompleteConnectionAntigravity()
                connection.execute = lambda _sql, _parameters, rows=rows: rows  # type: ignore[method-assign]
                session = sources._open_test_labels_once_with_dependencies_antigravity(
                    authority,
                    self._dependencies_antigravity(sources, connection, []),
                )
                with self.assertRaises(ValueError):
                    session.labels_for_dataset_antigravity(authority.dataset_order[0])
                self.assertEqual(session.audit_antigravity().state, "terminal")
                self.assertEqual(connection.close_calls, 1)

        authority = predecessor._capture_predecessors_with_dependencies_antigravity(
            self.fixture.dependencies_antigravity()
        )
        connection = CompleteConnectionAntigravity()

        def interrupted(_sql: str, _parameters: tuple[object, ...]):  # type: ignore[no-untyped-def]
            def rows():
                yield (0, "0")
                raise RuntimeError("iterator boom")

            return rows()

        connection.execute = interrupted  # type: ignore[method-assign]
        session = sources._open_test_labels_once_with_dependencies_antigravity(
            authority,
            self._dependencies_antigravity(sources, connection, []),
        )
        with self.assertRaisesRegex(RuntimeError, "iterator boom"):
            session.labels_for_dataset_antigravity(authority.dataset_order[0])
        self.assertEqual(connection.close_calls, 1)

    def test_early_close_and_cleanup_failure_report_distinct_truth_antigravity(self) -> None:
        """Break caught: primary incompleteness and resource-close faults overwrite one another."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            predecessor_antigravity as predecessor,
        )
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            sources_antigravity as sources,
        )

        connection = CompleteConnectionAntigravity()
        session = sources._open_test_labels_once_with_dependencies_antigravity(
            self.authority,
            self._dependencies_antigravity(sources, connection, []),
        )
        with self.assertRaisesRegex(ValueError, "did not complete"):
            session.close_antigravity()
        audit = session.audit_antigravity()
        self.assertEqual(audit.state, "terminal")
        self.assertEqual(
            audit.primary_error,
            "ValueError: TEST label session did not complete the frozen dataset order",
        )
        self.assertIsNone(audit.cleanup_error)

        authority = predecessor._capture_predecessors_with_dependencies_antigravity(
            self.fixture.dependencies_antigravity()
        )
        close_error = OSError("close boom")
        connection = CompleteConnectionAntigravity(close_error=close_error)
        session = sources._open_test_labels_once_with_dependencies_antigravity(
            authority,
            self._dependencies_antigravity(sources, connection, []),
        )
        for dataset in authority.dataset_order:
            session.labels_for_dataset_antigravity(dataset)
        with self.assertRaisesRegex(OSError, "close boom"):
            session.close_antigravity()
        audit = session.audit_antigravity()
        self.assertEqual(audit.state, "close_failed")
        self.assertIsNone(audit.primary_error)
        self.assertEqual(audit.cleanup_error, "OSError: close boom")
        self.assertEqual(connection.close_calls, 1)
        with self.assertRaisesRegex(OSError, "close boom"):
            session.close_antigravity()
        self.assertEqual(connection.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
