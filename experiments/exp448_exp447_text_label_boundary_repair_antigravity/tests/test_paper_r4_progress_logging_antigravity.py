"""Result-neutral paper r4 progress-callback contracts."""
from __future__ import annotations

import contextlib
from inspect import signature
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

class ProgressLoggingAntigravityTests(unittest.TestCase):
    def _parallel_fixture(self):
        from paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper import (
            Task9ParallelCompositionTests,
        )

        fixture = Task9ParallelCompositionTests("run")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        return fixture

    def _compose_patched(self, fixture, manifest, *, progress=None):
        from paper.equipment_anomaly_detection import proposal_score_paper as score

        events: list[tuple[str, str]] = []

        def hard(raw):
            events.append(("task4", raw.dataset))
            return fixture.per_dataset[raw.dataset][0]

        def mini(raw):
            return object()

        def route(raw, _sources, _mini):
            events.append(("task5", raw.dataset))
            return fixture.per_dataset[raw.dataset][1]

        def low(raw, lanes, **_kwargs):
            events.append(("task6", "global"))
            return fixture.low

        class Future:
            def __init__(self, fn, args):
                try:
                    self.value, self.error = fn(*args), None
                except Exception as error:
                    self.value, self.error = None, error

            def result(self):
                if self.error is not None:
                    raise self.error
                return self.value

            def cancel(self):
                return False

        class InlineExecutor:
            def __init__(self, *, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, fn, *args):
                return Future(fn, args)

        patches = (
            patch.object(score, "score_hard_sources", side_effect=hard),
            patch.object(score, "score_minirocket_confirmations", side_effect=mini),
            patch.object(score, "replay_hard_route", side_effect=route),
            patch.object(score, "score_low_frequency", side_effect=low),
            patch.object(score, "classify_stability", return_value=fixture.stability),
            patch.object(score, "build_hard_evidence_surface", return_value=object()),
            patch.object(score, "build_policies", return_value=fixture.policies),
            patch.object(score, "ProcessPoolExecutor", InlineExecutor),
            patch.object(score, "as_completed", side_effect=lambda futures: tuple(futures)),
        )
        kwargs = {} if progress is None else {"progress": progress}
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            return score.compose_proposal_score(fixture.raw, manifest, **kwargs), events

    def test_progress_callback_default_none_preserves_typed_proposal_and_stderr_contract(self) -> None:
        fixture = self._parallel_fixture()
        manifest = fixture._acceleration(3)
        stderr = StringIO()
        with contextlib.redirect_stderr(stderr):
            first, _events = self._compose_patched(fixture, manifest)
            second, _again = self._compose_patched(fixture, fixture._acceleration(3))
        text = stderr.getvalue()
        lines = text.splitlines()
        self.assertTrue(any("Task4+Task5 start" in line for line in lines))
        self.assertTrue(any("Task4+Task5" in line and "dataset=alpha" in line for line in lines))
        self.assertTrue(any("Task4+Task5" in line and "dataset=beta" in line for line in lines))
        self.assertIn("Task6 start", lines)
        self.assertIn("Task6 finish", lines)
        self.assertIn("Task7 start", lines)
        self.assertIn("Task7 finish", lines)
        self.assertNotIn("raw_preload", text)
        self.assertNotIn("task6 0/", text)
        self.assertNotIn("task7 0/", text)
        self.assertEqual(tuple(first.sources), tuple(second.sources))
        self.assertEqual(tuple(first.routes), tuple(second.routes))
        self.assertEqual(first.low_frequency, second.low_frequency)
        self.assertEqual(tuple(first.stability.selected_coordinates), tuple(second.stability.selected_coordinates))
        self.assertEqual(
            tuple(route.name for route in first.policies.policies),
            tuple(route.name for route in second.policies.policies),
        )

    def test_raising_progress_callback_does_not_change_typed_result_or_replace_scientific_error(self) -> None:
        from paper.equipment_anomaly_detection.proposal_score_paper import compose_proposal_score

        def boom(_event):
            raise RuntimeError("callback boom")

        fixture = self._parallel_fixture()
        manifest = fixture._acceleration(3)
        baseline, _events = self._compose_patched(fixture, manifest)
        observed, _again = self._compose_patched(fixture, fixture._acceleration(3), progress=boom)
        self.assertEqual(tuple(baseline.sources), tuple(observed.sources))
        self.assertEqual(tuple(baseline.routes), tuple(observed.routes))
        self.assertEqual(baseline.low_frequency, observed.low_frequency)
        self.assertEqual(
            tuple(route.name for route in baseline.policies.policies),
            tuple(route.name for route in observed.policies.policies),
        )
        with self.assertRaises(ValueError) as raised:
            compose_proposal_score({}, object(), progress=boom)
        self.assertIsInstance(raised.exception, ValueError)
        self.assertNotIsInstance(raised.exception, RuntimeError)
    def test_rate_gate_emits_start_end_error_immediately_and_ticks_at_most_once_per_15s(self) -> None:
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            make_stderr_progress_sink_paper,
        )

        stamps = iter((0.0, 0.0, 1.0, 14.9, 15.0, 15.1, 20.0))
        stream = StringIO()
        sink = make_stderr_progress_sink_paper(clock=lambda: next(stamps), stream=stream)
        payload = {"stage": "task6", "completed": 0, "total": 3, "elapsed_s": 0.0}
        sink({**payload, "phase": "start", "completed": 0})
        sink({**payload, "phase": "tick", "completed": 0, "elapsed_s": 0.0})
        sink({**payload, "phase": "tick", "completed": 0, "elapsed_s": 1.0})
        sink({**payload, "phase": "tick", "completed": 1, "elapsed_s": 14.9})
        sink({**payload, "phase": "tick", "completed": 2, "elapsed_s": 15.0})
        sink({**payload, "phase": "end", "completed": 3, "elapsed_s": 15.1})
        sink({**payload, "phase": "error", "completed": 3, "elapsed_s": 20.0})
        lines = [line for line in stream.getvalue().splitlines() if line]
        self.assertEqual(sum(line.startswith("task6 0/3") and "0.0%" in line for line in lines[:2]), 2)
        self.assertTrue(any("task6 2/3" in line for line in lines))
        self.assertTrue(any("task6 3/3" in line and "100.0%" in line for line in lines))
        tick_lines = [line for line in lines if "task6 1/3" in line]
        self.assertEqual(tick_lines, [])
        self.assertGreaterEqual(len(lines), 4)
        self.assertLessEqual(len(lines), 5)

    def test_progress_text_firewall_rejects_label_and_score_fields(self) -> None:
        from paper.equipment_anomaly_detection.data_paper import emit_progress_paper
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            make_stderr_progress_sink_paper,
        )

        received: list[dict[str, object]] = []
        emit_progress_paper(
            received.append,
            {
                "stage": "task6",
                "phase": "tick",
                "completed": 1,
                "total": 2,
                "elapsed_s": 1.0,
                "score": 1.23,
                "pvalue": 0.01,
                "label": 1,
                "alerts": [0, 1],
                "detail": "score=1.23",
            },
        )
        self.assertEqual(len(received), 1)
        self.assertEqual(set(received[0]), {"stage", "phase", "completed", "total", "elapsed_s"})
        self.assertNotIn("detail", received[0])
        stream = StringIO()
        sink = make_stderr_progress_sink_paper(clock=lambda: 0.0, stream=stream)
        sink(received[0])
        text = stream.getvalue().lower()
        for token in ("label=", "p-value", "score=", "alerts=", "rank="):
            self.assertNotIn(token, text)

    def test_raw_preload_emits_dataset_progress_without_label_sql_or_scores(self) -> None:
        from paper.equipment_anomaly_detection import runner_paper as runner

        events: list[dict[str, object]] = []
        catalog = {"datasets": [{"dataset": "alpha"}, {"dataset": "beta"}]}
        with patch.object(runner, "load_dataset_values", side_effect=lambda db_path, name: f"raw-{name}") as loaded:
            raw = runner._load_catalog_raw(catalog, Path("fixture.sqlite"), progress=events.append)
        self.assertEqual(tuple(raw), ("alpha", "beta"))
        self.assertEqual([event["phase"] for event in events], ["start", "tick", "tick", "end"])
        self.assertEqual([event.get("dataset") for event in events], [None, "alpha", "beta", None])
        joined = " ".join(str(value) for event in events for value in event.values())
        for token in ("label", "SELECT", "pvalue", "score", "alert"):
            self.assertNotIn(token.lower(), joined.lower())
        for call in loaded.call_args_list:
            self.assertEqual(call.kwargs, {})
            self.assertEqual(len(call.args), 2)

    def test_task4_task5_existing_completion_progress_unchanged(self) -> None:
        from paper.equipment_anomaly_detection import proposal_score_paper as score

        fixture = self._parallel_fixture()
        completed: list[str] = []
        values = fixture.per_dataset

        class Future:
            def __init__(self, dataset):
                self.dataset = dataset

            def result(self):
                completed.append(self.dataset)
                return values[self.dataset]

        class CompletionExecutor:
            def __init__(self, *, max_workers):
                self.futures = {}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, _fn, raw):
                future = Future(raw.dataset)
                self.futures[raw.dataset] = future
                return future

        def completed_first(futures):
            by_dataset = {future.dataset: future for future in futures}
            return (by_dataset["beta"], by_dataset["alpha"]) if set(by_dataset) == {"alpha", "beta"} else tuple(by_dataset.values())

        class RecordingStderr(StringIO):
            def __init__(self):
                super().__init__()
                self.flush_calls = 0

            def flush(self):
                self.flush_calls += 1
                super().flush()

        stderr = RecordingStderr()
        patches = (
            patch.object(score, "ProcessPoolExecutor", CompletionExecutor),
            patch.object(score, "as_completed", side_effect=completed_first),
            patch.object(score, "score_low_frequency", return_value=fixture.low),
            patch.object(score, "classify_stability", return_value=fixture.stability),
            patch.object(score, "build_hard_evidence_surface", return_value=object()),
            patch.object(score, "build_policies", return_value=fixture.policies),
        )
        with contextlib.redirect_stderr(stderr), patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            proposal = score.compose_proposal_score(fixture.raw, fixture._acceleration(3))
        lines = stderr.getvalue().splitlines()
        self.assertEqual(completed, ["beta", "alpha"])
        self.assertEqual(tuple(proposal.sources), ("alpha", "beta"))
        self.assertGreaterEqual(stderr.flush_calls, 7)
        self.assertTrue(any("Task4+Task5 start" in line and "0/2" in line for line in lines))
        self.assertTrue(any("Task4+Task5 1/2" in line and "50.0%" in line and "dataset=beta" in line for line in lines))
        self.assertTrue(any("Task4+Task5 2/2" in line and "100.0%" in line and "dataset=alpha" in line for line in lines))
        self.assertTrue(any(line == "Task6 start" for line in lines))
        self.assertTrue(any(line == "Task6 finish" for line in lines))
        self.assertTrue(any(line == "Task7 start" for line in lines))
        self.assertTrue(any(line == "Task7 finish" for line in lines))

    def test_journal_and_exit_progress_does_not_change_stdout_json_or_exit_code(self) -> None:
        from paper.equipment_anomaly_detection import pipeline_paper as pipeline

        payload = {
            "status": "completed",
            "completed_stages": ["preflight", "prepare", "acceleration", "proposal_score"],
        }
        stdout, stderr = StringIO(), StringIO()
        with patch.object(pipeline, "run_pipeline", lambda _config: payload), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = pipeline.main([
                "run", "--db-path", "fixture.sqlite", "--run-root", "run_root_antigravity",
                "--mode", "smoke", "--smoke-dataset", "alpha",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        self.assertIn("journal_exit", stderr.getvalue())
        self.assertIn("exit=0", stderr.getvalue())
        self.assertNotIn("{", stderr.getvalue().split("journal_exit", 1)[1].splitlines()[0])

        incomplete = {"status": "proposal_score_incomplete", "completed": 1, "failed": 1, "pending": 1}
        stdout, stderr = StringIO(), StringIO()
        with patch.object(pipeline, "run_pipeline", lambda _config: incomplete), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = pipeline.main([
                "run", "--db-path", "fixture.sqlite", "--run-root", "run_root_antigravity",
                "--mode", "smoke", "--smoke-dataset", "alpha",
            ])
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), json.dumps(incomplete, sort_keys=True, separators=(",", ":")) + "\n")
        self.assertIn("journal_exit", stderr.getvalue())
        self.assertIn("exit=2", stderr.getvalue())
        self.assertNotIn("{", stderr.getvalue().split("journal_exit", 1)[1].splitlines()[0])

    def test_worker_submit_signature_unchanged_no_callback_in_process_pool(self) -> None:
        from paper.equipment_anomaly_detection import proposal_score_paper as score
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            _collect_task4_task5,
            _score_dataset_task4_task5,
        )

        self.assertEqual(list(signature(_score_dataset_task4_task5).parameters), ["raw"])
        self.assertEqual(_score_dataset_task4_task5.__code__.co_argcount, 1)

        fixture = self._parallel_fixture()
        submits: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        class Future:
            def __init__(self, dataset):
                self.dataset = dataset

            def result(self):
                return fixture.per_dataset[self.dataset]

            def cancel(self):
                return False

        class RecordingExecutor:
            def __init__(self, *, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, fn, *args, **kwargs):
                submits.append((fn, args, dict(kwargs)))
                return Future(args[0].dataset)

        backend = FrozenTask8BackendDecision.from_acceleration_manifest(fixture._acceleration(3))
        with patch.object(score, "ProcessPoolExecutor", RecordingExecutor), patch.object(score, "as_completed", side_effect=lambda futures: tuple(futures)):
            _collect_task4_task5(fixture.raw, backend, progress=lambda _event: None)
        self.assertGreaterEqual(len(submits), 1)
        for fn, args, kwargs in submits:
            self.assertIs(fn, _score_dataset_task4_task5)
            self.assertEqual(len(args), 1)
            self.assertEqual(kwargs, {})

    def test_task6_dataset_and_group_progress_without_scores(self) -> None:
        from paper.equipment_anomaly_detection.tests_paper.test_low_frequency_paper import _values
        from paper.equipment_anomaly_detection.sources_paper.low_frequency_paper import score_low_frequency

        events: list[dict[str, object]] = []
        values = _values()
        baseline = score_low_frequency(values, {values.dataset: frozenset()})
        observed = score_low_frequency(values, {values.dataset: frozenset()}, progress=events.append)
        self.assertEqual(
            tuple(candidate.coordinate for candidate in observed.selected_candidates),
            tuple(candidate.coordinate for candidate in baseline.selected_candidates),
        )
        self.assertEqual(observed.rows[1].missing_reason, baseline.rows[1].missing_reason)
        self.assertTrue(events[0]["phase"] == "start" and events[0]["completed"] == 0 and events[0]["total"] == 1)
        self.assertTrue(any(event.get("detail", "").startswith("group_len=") and event["completed"] == 0 and event["total"] == 1 for event in events))
        parent_ticks = [event for event in events if event["phase"] == "tick" and not str(event.get("detail", "")).startswith("group_len=")]
        self.assertEqual(parent_ticks[-1]["completed"], 1)
        self.assertEqual(events[-1]["phase"], "end")
        joined = json.dumps(events)
        for token in ("excess_ratio", "empirical_rank", "score", "label="):
            self.assertNotIn(token, joined)

        dual_events: list[dict[str, object]] = []
        dual = {"A": _values("A"), "B": _values("B")}
        dual_hard = {"A": frozenset(), "B": frozenset()}
        dual_baseline = score_low_frequency(dual, dual_hard)
        dual_observed = score_low_frequency(dual, dual_hard, progress=dual_events.append)
        self.assertEqual(
            tuple(candidate.coordinate for candidate in dual_observed.selected_candidates),
            tuple(candidate.coordinate for candidate in dual_baseline.selected_candidates),
        )
        self.assertTrue(dual_events and all(event["total"] == 2 for event in dual_events))
        self.assertTrue(
            any(
                str(event.get("detail", "")).startswith("group_len=")
                and event["completed"] == 0
                and event["total"] == 2
                and event.get("dataset") == "A"
                for event in dual_events
            )
        )
        dual_parent = [
            event for event in dual_events
            if event["phase"] == "tick" and not str(event.get("detail", "")).startswith("group_len=")
        ]
        self.assertTrue(any(event["completed"] == 1 and event.get("dataset") == "A" for event in dual_parent))

    def test_task7_candidate_and_view_progress_is_rate_limited_and_label_free(self) -> None:
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import classify_stability
        from paper.equipment_anomaly_detection.proposal_score_paper import make_stderr_progress_sink_paper
        from paper.equipment_anomaly_detection.tests_paper.test_stability_paper import _raw, _result

        values = _raw("fixture")
        low = _result(values)
        selected = low.selected_candidates
        self.assertGreaterEqual(len(selected), 1)
        events: list[dict[str, object]] = []
        stamps = iter([0.0] * 8 + [15.0, 30.0, 45.0])
        stream = StringIO()
        sink = make_stderr_progress_sink_paper(clock=lambda: next(stamps, 45.0), stream=stream)

        def both(event):
            events.append(dict(event))
            sink(event)

        classify_stability(low, {values.dataset: values}, progress=both)
        total = len(selected)
        self.assertTrue(all(event["total"] == total for event in events if event["stage"] == "task7"))
        self.assertTrue(any(str(event.get("detail", "")).startswith("instance_index=") for event in events))
        self.assertTrue(any(str(event.get("detail", "")).startswith("view_index=") and event["completed"] == 0 for event in events))
        self.assertEqual(events[0]["phase"], "start")
        self.assertEqual(events[-1]["phase"], "end")
        printed = stream.getvalue()
        self.assertIn("task7 0/", printed)
        tick_lines = [line for line in printed.splitlines() if "view_index=" in line]
        self.assertLessEqual(len(tick_lines), 2)
        joined = json.dumps(events)
        for token in ("survived", "agreeing_depths", "excess_ratio", "rank=", "label="):
            self.assertNotIn(token, joined)

    def test_hard_evidence_and_policies_progress_has_no_alert_or_score_payload(self) -> None:
        import numpy as np
        from paper.equipment_anomaly_detection.routing_paper import (
            POLICY_ORDER,
            build_hard_evidence_surface,
            build_policies,
        )
        from paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper import (
            Task4Task5SuccessRecordTests,
        )

        fixture = Task4Task5SuccessRecordTests("run")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        raw, sources, route, low, stability, _policies, _backend = fixture._task4_task5_values()
        raw_map = {raw.dataset: raw}
        sources_map = {raw.dataset: sources}
        routes_map = {raw.dataset: route}
        events: list[dict[str, object]] = []
        surface = build_hard_evidence_surface(sources_map, routes_map, raw_map, progress=events.append)
        build_policies(surface, low, stability, raw_map, routes_map, progress=events.append)
        self.assertEqual({event["stage"] for event in events}, {"hard_evidence", "policies"})
        policy_events = [event for event in events if event["stage"] == "policies"]
        self.assertTrue(policy_events and all(event["total"] == 4 for event in policy_events))
        self.assertEqual(
            tuple(event.get("detail") for event in policy_events if event["phase"] == "tick"),
            POLICY_ORDER,
        )
        for event in events:
            self.assertNotIn("alerts", event)
            self.assertNotIn("pvalue", event)
            self.assertNotIn("strength", event)
            for value in event.values():
                self.assertFalse(isinstance(value, np.ndarray))
                self.assertFalse(isinstance(value, frozenset) and all(type(item) is int for item in value))

    def test_shard_write_and_freeze_verify_progress(self) -> None:
        import numpy as np
        from paper.equipment_anomaly_detection import proposal_score_paper as score
        from paper.equipment_anomaly_detection.artifacts_paper import (
            freeze_score_root,
            verify_score_root,
            write_typed_proposal_shard,
        )
        from paper.equipment_anomaly_detection.contracts_paper import LowFrequencyResult
        from paper.equipment_anomaly_detection.proposal_score_paper import ProposalScore, write_and_freeze_proposal_score
        from paper.equipment_anomaly_detection.routing_paper import PolicyBundle, PolicyRoute
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import StabilityPartition
        from paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper import Task8LifecycleTests

        fixture = Task8LifecycleTests("run")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        first, second = fixture._typed("alpha"), fixture._typed("beta")
        raw = {"alpha": first[0], "beta": second[0]}
        sources = {"alpha": first[1], "beta": second[1]}
        routes = {"alpha": first[2], "beta": second[2]}
        low = LowFrequencyResult(
            ("alpha", "beta"),
            tuple(row for values in (first[3], second[3]) for row in values.rows),
            (), (), 0, {"alpha": frozenset(), "beta": frozenset()},
        )
        stability = StabilityPartition((), (), (), {})
        policies = PolicyBundle(tuple(
            PolicyRoute(
                name,
                {dataset: frozenset() for dataset in raw},
                {dataset: np.zeros(2) for dataset in raw},
                {dataset: np.zeros(2) for dataset in raw},
            )
            for name in ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")
        ))
        proposal = ProposalScore(
            raw, sources, routes, low, stability, policies, first[6],
            {"task4_task5": 0.0, "task6": 0.0, "task7": 0.0, "total": 0.0},
        )
        events: list[dict[str, object]] = []
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        original = score.write_typed_proposal_shard

        def spy(*args, **kwargs):
            calls.append((args, dict(kwargs)))
            return original(*args, **kwargs)

        with patch.object(score, "write_typed_proposal_shard", side_effect=spy):
            write_and_freeze_proposal_score(fixture.root, fixture.catalog, proposal, progress=events.append)
        shard_events = [event for event in events if event["stage"] == "shard_write"]
        freeze_events = [event for event in events if event["stage"] == "freeze_verify"]
        self.assertEqual([event["phase"] for event in shard_events], ["start", "tick", "tick", "end"])
        self.assertEqual([event["completed"] for event in shard_events if event["phase"] == "tick"], [1, 2])
        self.assertTrue(all(event["total"] == 2 for event in shard_events))
        self.assertEqual(len(calls), 2)
        for args, kwargs in calls:
            self.assertNotIn("progress", kwargs)
            self.assertEqual(len(args), 9)
        self.assertEqual(freeze_events[0]["phase"], "start")
        self.assertEqual(freeze_events[-1]["phase"], "end")
        self.assertEqual([event["completed"] for event in freeze_events if event["phase"] == "tick"], [1, 2])

        direct = Task8LifecycleTests("run")
        direct.setUp()
        self.addCleanup(direct.tearDown)
        direct_events: list[dict[str, object]] = []
        write_typed_proposal_shard(direct.root, direct.catalog, *direct._typed("alpha"), progress=direct_events.append)
        self.assertEqual([event["phase"] for event in direct_events], ["tick"])
        self.assertTrue(all(event["stage"] == "shard_write" for event in direct_events))

        write_typed_proposal_shard(direct.root, direct.catalog, *direct._typed("beta"))
        freeze_only: list[dict[str, object]] = []
        freeze_score_root(direct.root, direct.catalog, progress=freeze_only.append)
        self.assertEqual(freeze_only[0]["phase"], "start")
        self.assertEqual(freeze_only[-1]["phase"], "end")
        self.assertTrue(all(event["stage"] == "freeze_verify" for event in freeze_only))
        self.assertEqual([event["completed"] for event in freeze_only if event["phase"] == "tick"], [1, 2])

        verify_events: list[dict[str, object]] = []
        verify_score_root(fixture.root, fixture.catalog, progress=verify_events.append)
        self.assertEqual(verify_events[0]["phase"], "start")
        self.assertEqual(verify_events[-1]["phase"], "end")
        self.assertTrue(all(event["stage"] == "freeze_verify" for event in verify_events))
        self.assertEqual([event["completed"] for event in verify_events if event["phase"] == "tick"], [1, 2])

        for event in events + direct_events + freeze_only + verify_events:
            detail = event.get("detail")
            if detail is not None:
                self.assertTrue(str(detail).endswith("_paper.json"))
            joined = json.dumps(event)
            self.assertNotIn("shard_hash", joined)
            self.assertNotIn("sha256", joined)

    def test_identical_candidate_indices_and_shard_bytes_with_and_without_callback(self) -> None:
        from paper.equipment_anomaly_detection import proposal_score_paper as score
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.artifacts_paper import write_typed_proposal_shard
        from paper.equipment_anomaly_detection.data_paper import emit_progress_paper
        from paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper import (
            Task4Task5SuccessRecordTests,
        )

        fixture = Task4Task5SuccessRecordTests("run")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        raw, sources, route, low, stability, policies, backend = fixture._task4_task5_values()
        raw_map = {raw.dataset: raw}
        manifest = build_acceleration_manifest("f" * 64, logical_cpu_count=3)
        received: list[dict[str, object]] = []

        def recording(event):
            received.append(event)
            event["stage"] = "mutated"

        class Future:
            def __init__(self, fn, args):
                try:
                    self.value, self.error = fn(*args), None
                except Exception as error:
                    self.value, self.error = None, error

            def result(self):
                if self.error is not None:
                    raise self.error
                return self.value

            def cancel(self):
                return False

        class InlineExecutor:
            def __init__(self, *, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, fn, *args):
                return Future(fn, args)

        patches = (
            patch.object(score, "score_hard_sources", side_effect=lambda _raw: sources),
            patch.object(score, "score_minirocket_confirmations", return_value=object()),
            patch.object(score, "replay_hard_route", side_effect=lambda _raw, _sources, _mini: route),
            patch.object(score, "score_low_frequency", return_value=low),
            patch.object(score, "classify_stability", return_value=stability),
            patch.object(score, "ProcessPoolExecutor", InlineExecutor),
            patch.object(score, "as_completed", side_effect=lambda futures: tuple(futures)),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            baseline = score.compose_proposal_score(raw_map, manifest)
            observed = score.compose_proposal_score(raw_map, manifest, progress=recording)
        self.assertEqual(
            tuple(candidate.coordinate for candidate in observed.low_frequency.selected_candidates),
            tuple(candidate.coordinate for candidate in baseline.low_frequency.selected_candidates),
        )
        self.assertEqual(tuple(observed.stability.selected_coordinates), tuple(baseline.stability.selected_coordinates))
        self.assertEqual(tuple(observed.stability.stable_coordinates), tuple(baseline.stability.stable_coordinates))
        self.assertEqual(tuple(observed.stability.unstable_coordinates), tuple(baseline.stability.unstable_coordinates))
        self.assertEqual(
            tuple(candidate.coordinate for candidate in observed.low_frequency.selected_candidates),
            tuple(candidate.coordinate for candidate in low.selected_candidates),
        )
        self.assertEqual(tuple(observed.stability.selected_coordinates), tuple(stability.selected_coordinates))

        left = fixture.fixture.base / "left_antigravity"
        right = fixture.fixture.base / "right_antigravity"
        left.mkdir()
        right.mkdir()
        write_typed_proposal_shard(left, fixture.catalog, raw, sources, route, low, stability, policies, backend)
        write_typed_proposal_shard(
            right, fixture.catalog, raw, sources, route, low, stability, policies, backend, progress=recording,
        )
        left_bytes = (left / "stage_artifacts_paper" / "proposal_freeze" / "shards" / f"{raw.dataset}_paper.json").read_bytes()
        right_bytes = (right / "stage_artifacts_paper" / "proposal_freeze" / "shards" / f"{raw.dataset}_paper.json").read_bytes()
        self.assertEqual(left_bytes, right_bytes)
        self.assertGreater(len(left_bytes), 0)
        self.assertTrue(received)
        self.assertTrue(any(event.get("stage") == "mutated" for event in received))
        emit_progress_paper(recording, {"stage": "shard_write", "phase": "tick", "completed": 1, "total": 1, "elapsed_s": 0.0})


if __name__ == "__main__":
    unittest.main()
