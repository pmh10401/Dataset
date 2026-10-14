from contextlib import closing, contextmanager
from http import HTTPStatus
import errno
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread, Timer
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import urllib.error
import urllib.request

from services.time_series_anomaly_diagnosis import application_server
from services.time_series_anomaly_diagnosis.analysis_history import (
    create_analysis_run,
    get_analysis_run,
    mark_analysis_running,
    open_history_database,
    save_completed_analysis,
)
from services.time_series_anomaly_diagnosis.application_server import (
    create_application_server,
    execute_analysis_run,
    main,
)
from services.time_series_anomaly_diagnosis.anomaly_detection import calculate_evaluation_metrics


@contextmanager
def temporary_databases():
    with TemporaryDirectory() as directory:
        history_conn = open_history_database(Path(directory) / "history.sqlite3")
        source_conn = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            yield history_conn, source_conn
        finally:
            for conn in (history_conn, source_conn):
                try:
                    conn.close()
                except Exception:
                    pass


def make_test_dependencies(
    source_connection=None,
    history_connection=None,
    catalog=None,
    prepare_values=None,
    detect=None,
    load_labels=None,
    evaluate=None,
    load_series_for_chart=None,
    review_evidence=None,
    rank_review=None,
):
    return SimpleNamespace(
        source_connection=source_connection or object(),
        source_lock=Lock(),
        history_connection=history_connection or object(),
        history_lock=Lock(),
        catalog=catalog or [],
        prepare_values=prepare_values or (lambda *a: SimpleNamespace(
            dataset_name="sample", test_indices=(0, 1), train_values=[None] * 199
        )),
        detect=detect or (lambda *a: SimpleNamespace(
            method_name="minirocket_local_gap",
            rows=(
                {"series_index": 0, "score": 0.1, "automatic_anomaly": False},
                {"series_index": 1, "score": 0.9, "automatic_anomaly": True},
            ),
            automatic_anomaly_indices=(1,),
            train_reference={"median": [0.0] * 10, "q10": [-1.0] * 10, "q90": [1.0] * 10},
            target_length=10,
        )),
        load_labels=load_labels or (lambda *a: {0: 0, 1: 1}),
        evaluate=evaluate or (lambda *a: {"auc": 1.0, "false_negatives": 0, "false_positives": 0}),
        load_series_for_chart=load_series_for_chart or (lambda *a: [0.5] * 10),
        review_evidence=review_evidence or {},
        rank_review=rank_review or (lambda *a: {
            "evidence_available": False,
            "automatic_promotion": False,
            "candidates": [],
            "unavailable_candidate_indices": [],
        }),
    )


@contextmanager
def running_test_server(dependencies=None, catalog=None, **dep_kwargs):
    with temporary_databases() as (history_conn, source_conn):
        deps = dependencies or make_test_dependencies(
            source_connection=source_conn,
            history_connection=history_conn,
            catalog=catalog,
            **dep_kwargs,
        )
        server = create_application_server("127.0.0.1", 0, source_conn, history_conn, deps)
        host, port = server.server_address
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(path: str, data: bytes | None = None, method: str = "GET", headers: dict | None = None):
            req = urllib.request.Request(
                f"http://{host}:{port}{path}",
                data=data,
                headers=headers or ({"Content-Type": "application/json"} if data else {}),
                method=method,
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    return resp.status, json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                try:
                    return err.code, json.loads(err.read().decode("utf-8"))
                finally:
                    err.close()

        ctx = SimpleNamespace(
            server=server,
            host=host,
            port=port,
            history_conn=history_conn,
            source_conn=source_conn,
            dependencies=deps,
            request=request,
        )
        try:
            yield ctx
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3.0)


class ApplicationServerTest(unittest.TestCase):
    def test_status_reports_ready_without_cuda(self) -> None:
        with running_test_server() as ctx:
            status, body = ctx.request("/api/status")
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(body["status"], "ready")
        self.assertEqual(body["backend"], "cpu_numba")

    def test_current_result_promotes_resolution_limited_candidate_with_warning(self) -> None:
        detection = SimpleNamespace(
            method_name="minirocket_local_gap",
            rows=(
                {
                    "series_index": 0,
                    "score": 0.1,
                    "p_value": 0.5,
                    "automatic_anomaly": False,
                },
                {
                    "series_index": 49,
                    "score": 0.9,
                    "p_value": 1 / 41,
                    "automatic_anomaly": True,
                },
            ),
            automatic_anomaly_indices=(49,),
            resolution_limited_warning_indices=(49,),
            train_reference={"median": [0.0], "q10": [-1.0], "q90": [1.0]},
            target_length=1,
        )
        dependencies = make_test_dependencies(
            prepare_values=lambda *_: SimpleNamespace(
                dataset_name="sample",
                test_indices=(0, 49),
                train_values=[None] * 40,
            ),
            detect=lambda *_: detection,
            load_labels=lambda *_: {0: 0, 49: 1},
            evaluate=calculate_evaluation_metrics,
        )
        entry = {
            "dataset_name": "sample",
            "method_name": "minirocket_local_gap",
            "availability": "standard",
        }
        module = "services.time_series_anomaly_diagnosis.application_server"
        with patch(f"{module}.mark_analysis_running"), patch(
            f"{module}.save_completed_analysis"
        ) as saved:
            execute_analysis_run("run-current", entry, dependencies)

        result = saved.call_args[0][2]
        self.assertEqual(result["schema_version"], "live_analysis_result_v3")
        self.assertEqual(result["analysis_policy"], "registered_dataset_best_method")
        self.assertEqual(result["resolution_limited_warning_indices"], [49])
        self.assertEqual(result["automatic_anomaly_indices"], [49])
        self.assertTrue(result["automatic_promotion"])
        self.assertEqual(result["calibration"]["reference_count"], 40)
        self.assertAlmostEqual(result["calibration"]["minimum_p_value"], 1 / 41)
        self.assertEqual(result["calibration"]["automatic_threshold"], 0.005)
        self.assertFalse(result["calibration"]["automatic_threshold_attainable"])
        for historical_field in (
            "frozen_hard_confirmed_review_indices",
            "small_sample_review_indices",
            "resolution_limited_review_indices",
            "review",
        ):
            self.assertNotIn(historical_field, result)
        row_49 = next(row for row in result["rows"] if row["series_index"] == 49)
        self.assertTrue(row_49["automatic_anomaly"])
        self.assertEqual(result["metrics"]["true_positives"], 1)
        self.assertEqual(result["metrics"]["false_negatives"], 0)

    def test_review_candidates_are_ranked_before_labels_and_remain_review_only(self) -> None:
        events = []
        detection = SimpleNamespace(
            method_name="minirocket_local_gap",
            rows=(
                {"series_index": 0, "score": 0.4, "p_value": 0.5, "automatic_anomaly": False},
                {"series_index": 1, "score": 0.9, "p_value": 0.005, "automatic_anomaly": True},
            ),
            automatic_anomaly_indices=(1,),
            resolution_limited_warning_indices=(),
            train_reference={"median": [0.0], "q10": [-1.0], "q90": [1.0]},
            target_length=1,
        )

        def detect(*_):
            events.append("detect")
            return detection

        def rank_review(dataset_name, rows, evidence):
            self.assertEqual(dataset_name, "sample")
            self.assertIs(rows, detection.rows)
            self.assertEqual(evidence, {"sample": "evidence"})
            events.append("rank")
            return {
                "evidence_available": True,
                "automatic_promotion": False,
                "candidates": [{
                    "series_index": 0,
                    "priority": "P1",
                    "support_count": 2,
                    "source_families": ["official_multirocket", "structural"],
                    "score": 0.4,
                    "p_value": 0.5,
                }],
                "unavailable_candidate_indices": [],
            }

        def load_labels(*_):
            self.assertEqual(events[-1], "rank")
            events.append("labels")
            return {0: 0, 1: 1}

        dependencies = make_test_dependencies(
            prepare_values=lambda *_: SimpleNamespace(
                dataset_name="sample", test_indices=(0, 1), train_values=[None] * 199
            ),
            detect=detect,
            load_labels=load_labels,
            review_evidence={"sample": "evidence"},
            rank_review=rank_review,
        )
        entry = {
            "dataset_name": "sample",
            "method_name": "minirocket_local_gap",
            "availability": "standard",
        }
        module = "services.time_series_anomaly_diagnosis.application_server"
        with patch(f"{module}.mark_analysis_running"), patch(
            f"{module}.save_completed_analysis"
        ) as saved:
            execute_analysis_run("run-review", entry, dependencies)

        result = saved.call_args.args[2]
        self.assertEqual(events, ["detect", "rank", "labels"])
        self.assertEqual(result["schema_version"], "live_analysis_result_v3")
        self.assertEqual(result["automatic_anomaly_indices"], [1])
        self.assertEqual(result["metrics"]["false_positives"], 0)
        self.assertEqual(result["metrics"]["false_negatives"], 0)
        self.assertFalse(result["multiple_evidence_review"]["automatic_promotion"])
        self.assertEqual(result["multiple_evidence_review"]["candidates"][0]["priority"], "P1")

    def test_labels_open_only_after_current_detection(self) -> None:
        events = []
        prepared = SimpleNamespace(
            dataset_name="sample", test_indices=(0, 1), train_values=[None] * 199
        )
        detection = SimpleNamespace(
            method_name="minirocket_local_gap",
            rows=(
                {"series_index": 0, "score": 0.1, "automatic_anomaly": False},
                {"series_index": 1, "score": 0.9, "automatic_anomaly": True},
            ),
            automatic_anomaly_indices=(1,),
            resolution_limited_warning_indices=(1,),
            train_reference={"median": [0.0], "q10": [-1.0], "q90": [1.0]},
            target_length=1,
        )

        def prepare(_connection, _dataset_name, _availability):
            events.append("prepare")
            return prepared

        def detect(_prepared, _method_name):
            events.append("detect")
            return detection

        def labels(_connection, _dataset_name, indices):
            self.assertEqual(events[-1], "detect")
            events.append("labels")
            return {index: index for index in indices}

        def evaluate(_rows, _labels):
            events.append("evaluate")
            return {"auc": 1.0, "false_negatives": 0, "false_positives": 0}

        dependencies = make_test_dependencies(
            prepare_values=prepare,
            detect=detect,
            load_labels=labels,
            evaluate=evaluate,
        )
        entry = {
            "dataset_name": "sample",
            "method_name": "minirocket_local_gap",
            "availability": "standard",
        }
        module = "services.time_series_anomaly_diagnosis.application_server"
        with patch(f"{module}.mark_analysis_running") as marked, patch(
            f"{module}.save_completed_analysis"
        ) as saved:
            execute_analysis_run("run-1", entry, dependencies)
        self.assertEqual(events, ["prepare", "detect", "labels", "evaluate"])
        marked.assert_called_once_with(dependencies.history_connection, "run-1")
        saved.assert_called_once()
        _, _, saved_result = saved.call_args[0]
        self.assertEqual(saved_result["dataset_name"], "sample")
        self.assertEqual(saved_result["method_name"], "minirocket_local_gap")
        self.assertEqual(saved_result["availability"], "standard")
        self.assertEqual(saved_result["automatic_anomaly_indices"], [1])
        self.assertEqual(saved_result["resolution_limited_warning_indices"], [1])
        self.assertTrue(saved_result["automatic_promotion"])
        self.assertIn("rows", saved_result)
        self.assertIn("train_reference", saved_result)
        self.assertIn("multiple_evidence_review", saved_result)
        self.assertIn("metrics", saved_result)

    def test_legacy_completed_result_requires_current_reanalysis(self) -> None:
        with running_test_server() as ctx:
            run = create_analysis_run(
                ctx.history_conn, "LegacyDataset", "minirocket_local_gap", "standard"
            )
            save_completed_analysis(
                ctx.history_conn,
                run["run_id"],
                {"dataset_name": "LegacyDataset", "rows": [{"series_index": 0}]},
            )

            status, detail = ctx.request(f"/api/analysis-runs/{run['run_id']}")
            self.assertEqual(status, HTTPStatus.OK)
            self.assertIsNone(detail["result"])
            self.assertTrue(detail["requires_reanalysis"])

            status, body = ctx.request(f"/api/analysis-runs/{run['run_id']}/series/0")
            self.assertEqual(status, HTTPStatus.CONFLICT)
            self.assertIn("reanalysis", body["error"])

    def test_analysis_run_failure_saves_error_and_allows_next_run(self) -> None:
        with temporary_databases() as (history_conn, source_conn):
            def prepare(_conn, dataset_name, _avail):
                if dataset_name == "fail_sample":
                    raise RuntimeError("Source data extraction error")
                return SimpleNamespace(
                    dataset_name=dataset_name,
                    test_indices=(0, 1),
                    train_values=[None] * 199,
                )

            deps = make_test_dependencies(
                source_connection=source_conn,
                history_connection=history_conn,
                prepare_values=prepare,
            )
            server = create_application_server("127.0.0.1", 0, source_conn, history_conn, deps)
            try:
                fail_run = create_analysis_run(history_conn, "fail_sample", "minirocket_local_gap", "standard")
                fut_fail = server.analysis_executor.submit(
                    execute_analysis_run,
                    fail_run["run_id"],
                    {"dataset_name": "fail_sample", "method_name": "minirocket_local_gap", "availability": "standard"},
                    deps,
                )
                fut_fail.result(timeout=3.0)

                failed_record = get_analysis_run(history_conn, fail_run["run_id"])
                self.assertEqual(failed_record["status"], "failed")
                self.assertEqual(failed_record["error_message"], "Source data extraction error")
                self.assertIsNone(failed_record["result"])

                success_run = create_analysis_run(history_conn, "success_sample", "minirocket_local_gap", "standard")
                fut_success = server.analysis_executor.submit(
                    execute_analysis_run,
                    success_run["run_id"],
                    {"dataset_name": "success_sample", "method_name": "minirocket_local_gap", "availability": "standard"},
                    deps,
                )
                fut_success.result(timeout=3.0)

                completed_record = get_analysis_run(history_conn, success_run["run_id"])
                self.assertEqual(completed_record["status"], "completed")
                self.assertIsNone(completed_record["error_message"])
                self.assertIsNotNone(completed_record["result"])
                self.assertEqual(completed_record["result"]["dataset_name"], "success_sample")
                self.assertEqual(completed_record["result"]["method_name"], "minirocket_local_gap")
            finally:
                server.server_close()

    def test_server_queue_isolation_and_sequential_execution(self) -> None:
        first_run_started, first_run_blocker, concurrency_lock = Event(), Event(), Lock()
        active_workers, max_active_workers = 0, 0

        def slow_prepare(_conn, dataset_name, _avail):
            nonlocal active_workers, max_active_workers
            with concurrency_lock:
                active_workers += 1
                max_active_workers = max(max_active_workers, active_workers)
            if dataset_name == "Dataset_A":
                first_run_started.set()
                first_run_blocker.wait(timeout=5.0)
            return SimpleNamespace(
                dataset_name=dataset_name,
                test_indices=(0, 1),
                train_values=[None] * 199,
            )

        def detect(prepared, method_name):
            try:
                return SimpleNamespace(
                    method_name=method_name,
                    rows=(
                        {"series_index": 0, "score": 0.2, "automatic_anomaly": False},
                        {"series_index": 1, "score": 0.8, "automatic_anomaly": True},
                    ),
                    automatic_anomaly_indices=(1,),
                    train_reference={"median": [0.0] * 50, "q10": [-1.0] * 50, "q90": [1.0] * 50},
                    target_length=50,
                )
            finally:
                nonlocal active_workers
                with concurrency_lock:
                    active_workers -= 1

        catalog = [
            {"dataset_name": "Dataset_A", "method_name": "minirocket_local_gap", "train_count": 10, "test_count": 2, "series_length": 50, "availability": "standard", "analysis_available": True},
            {"dataset_name": "Dataset_B", "method_name": "hydra_local_gap", "train_count": 10, "test_count": 2, "series_length": 50, "availability": "standard", "analysis_available": True},
        ]

        with running_test_server(catalog=catalog, prepare_values=slow_prepare, detect=detect) as ctx:
            status1, run1 = ctx.request("/api/analysis-runs", data=json.dumps({"dataset_name": "Dataset_A"}).encode("utf-8"), method="POST")
            self.assertEqual(status1, HTTPStatus.ACCEPTED)
            self.assertTrue(first_run_started.wait(timeout=3.0))

            status2, run2 = ctx.request("/api/analysis-runs", data=json.dumps({"dataset_name": "Dataset_B"}).encode("utf-8"), method="POST")
            self.assertEqual(status2, HTTPStatus.ACCEPTED)

            status_q2, detail2 = ctx.request(f"/api/analysis-runs/{run2['run_id']}")
            self.assertEqual(status_q2, HTTPStatus.OK)
            self.assertEqual(detail2["status"], "queued")

            first_run_blocker.set()

            for _ in range(50):
                _, detail2 = ctx.request(f"/api/analysis-runs/{run2['run_id']}")
                if detail2["status"] == "completed":
                    break
                Event().wait(0.05)

            self.assertEqual(detail2["status"], "completed")
            self.assertEqual(max_active_workers, 1)

    def test_http_api_validation_and_status_codes(self) -> None:
        catalog = [
            {"dataset_name": "ValidDataset", "method_name": "minirocket_local_gap", "train_count": 20, "test_count": 2, "series_length": 10, "availability": "standard", "analysis_available": True},
            {"dataset_name": "UnavailableDataset", "method_name": "minirocket_local_gap", "train_count": 2, "test_count": 2, "series_length": 10, "availability": "insufficient_training_data", "analysis_available": False},
        ]
        with running_test_server(catalog=catalog) as ctx:
            req = ctx.request

            # 1. GET /api/datasets
            status, body = req("/api/datasets")
            self.assertEqual(status, HTTPStatus.OK)
            self.assertEqual(len(body), 2)
            self.assertEqual(body[0]["dataset_name"], "ValidDataset")

            # 2. GET /api/analysis-runs
            status, body = req("/api/analysis-runs")
            self.assertEqual(status, HTTPStatus.OK)
            self.assertEqual(body, [])

            # 3. POST validation errors
            status, body = req("/api/analysis-runs", data=b"{malformed json", method="POST")
            self.assertEqual(status, HTTPStatus.BAD_REQUEST)
            self.assertIn("error", body)

            status, body = req("/api/analysis-runs", data=json.dumps({}).encode("utf-8"), method="POST")
            self.assertEqual(status, HTTPStatus.BAD_REQUEST)

            status, body = req("/api/analysis-runs", data=json.dumps({"dataset_name": ""}).encode("utf-8"), method="POST")
            self.assertEqual(status, HTTPStatus.BAD_REQUEST)

            status, body = req("/api/analysis-runs", data=json.dumps({"dataset_name": 12345}).encode("utf-8"), method="POST")
            self.assertEqual(status, HTTPStatus.BAD_REQUEST)

            status, body = req("/api/analysis-runs", data=json.dumps({"dataset_name": "NonExistent"}).encode("utf-8"), method="POST")
            self.assertEqual(status, HTTPStatus.NOT_FOUND)

            status, body = req("/api/analysis-runs", data=json.dumps({"dataset_name": "UnavailableDataset"}).encode("utf-8"), method="POST")
            self.assertEqual(status, HTTPStatus.CONFLICT)
            self.assertEqual(body["error"], "insufficient_training_data")

            # 4. Unknown run query -> 404
            status, body = req("/api/analysis-runs/non-existent-run-id")
            self.assertEqual(status, HTTPStatus.NOT_FOUND)

            # 5. Successful POST -> 202
            status, run_info = req("/api/analysis-runs", data=json.dumps({"dataset_name": "ValidDataset"}).encode("utf-8"), method="POST")
            self.assertEqual(status, HTTPStatus.ACCEPTED)
            run_id = run_info["run_id"]
            self.assertEqual(run_info["status"], "queued")

            for _ in range(50):
                status, detail = req(f"/api/analysis-runs/{run_id}")
                if detail.get("status") == "completed":
                    break
                Event().wait(0.05)

            self.assertEqual(detail["status"], "completed")
            self.assertIsNotNone(detail["result"])

            # 6. GET series
            status, body = req(f"/api/analysis-runs/{run_id}/series/abc")
            self.assertEqual(status, HTTPStatus.NOT_FOUND)

            status, body = req(f"/api/analysis-runs/{run_id}/series/999")
            self.assertEqual(status, HTTPStatus.NOT_FOUND)

            status, series_data = req(f"/api/analysis-runs/{run_id}/series/0")
            self.assertEqual(status, HTTPStatus.OK)
            self.assertEqual(
                set(series_data.keys()),
                {"series_index", "test_values", "train_median", "train_q10", "train_q90"},
            )
            self.assertEqual(series_data["series_index"], 0)
            self.assertEqual(len(series_data["test_values"]), 10)
            self.assertEqual(len(series_data["train_median"]), 10)

            # 7. Disallowed routes
            status, body = req("/unknown_file.html")
            self.assertEqual(status, HTTPStatus.NOT_FOUND)

            status, body = req("/api/unknown")
            self.assertEqual(status, HTTPStatus.NOT_FOUND)

    def test_series_request_on_incomplete_run_returns_409(self) -> None:
        with running_test_server() as ctx:
            run = create_analysis_run(ctx.history_conn, "Sample", "method_m", "standard")
            mark_analysis_running(ctx.history_conn, run["run_id"])
            status, _ = ctx.request(f"/api/analysis-runs/{run['run_id']}/series/0")
            self.assertEqual(status, HTTPStatus.CONFLICT)

    def test_server_creation_recovers_interrupted_runs_and_server_close_closes_connections(self) -> None:
        with temporary_databases() as (history_conn, source_conn):
            history_path = history_conn.execute("PRAGMA database_list").fetchone()[2]

            run1 = create_analysis_run(history_conn, "A", "method_1", "standard")
            run2 = create_analysis_run(history_conn, "B", "method_2", "standard")
            mark_analysis_running(history_conn, run2["run_id"])

            deps = make_test_dependencies(source_connection=source_conn, history_connection=history_conn)
            server = create_application_server("127.0.0.1", 0, source_conn, history_conn, deps)

            self.assertEqual(get_analysis_run(history_conn, run1["run_id"])["status"], "interrupted")
            self.assertEqual(get_analysis_run(history_conn, run2["run_id"])["status"], "interrupted")

            active_run = create_analysis_run(history_conn, "Active", "method_a", "standard")
            pending_run = create_analysis_run(history_conn, "Pending", "method_p", "standard")

            release_active = Event()
            active_saved = Event()
            pending_ran = False

            def active_job():
                release_active.wait(timeout=5.0)
                with deps.history_lock:
                    save_completed_analysis(history_conn, active_run["run_id"], {"status": "ok"})
                active_saved.set()

            def pending_job():
                nonlocal pending_ran
                pending_ran = True

            active_future = server.analysis_executor.submit(active_job)
            pending_future = server.analysis_executor.submit(pending_job)

            timer = Timer(0.05, release_active.set)
            timer.start()
            server.server_close()
            timer.join()

            self.assertTrue(active_future.done())
            self.assertTrue(active_saved.is_set())
            self.assertTrue(pending_future.cancelled())
            self.assertFalse(pending_ran)

            with self.assertRaises(sqlite3.ProgrammingError):
                history_conn.execute("SELECT 1")
            with self.assertRaises(sqlite3.ProgrammingError):
                source_conn.execute("SELECT 1")

            reopened_history = open_history_database(history_path)
            try:
                active_record = get_analysis_run(reopened_history, active_run["run_id"])
                self.assertEqual(active_record["status"], "completed")
                self.assertEqual(active_record["result"], {"status": "ok"})

                pending_record = get_analysis_run(reopened_history, pending_run["run_id"])
                self.assertEqual(pending_record["status"], "queued")
                self.assertIsNone(pending_record["result"])
            finally:
                reopened_history.close()

    def test_bind_failure_preserves_oserror_closes_new_databases_and_keeps_first_server(self) -> None:
        with running_test_server() as first:
            before = first.request("/api/status")
            history_path = first.history_conn.execute("PRAGMA database_list").fetchone()[2]
            queued = create_analysis_run(first.history_conn, "Queued", "method", "standard")
            running = create_analysis_run(first.history_conn, "Running", "method", "standard")
            mark_analysis_running(first.history_conn, running["run_id"])
            records_before = [
                get_analysis_run(first.history_conn, run["run_id"]) for run in (queued, running)
            ]
            with (
                closing(open_history_database(history_path)) as history,
                closing(sqlite3.connect(":memory:", check_same_thread=False)) as source,
            ):
                dependencies = make_test_dependencies(
                    source_connection=source, history_connection=history,
                )
                with self.assertRaises(OSError) as raised:
                    create_application_server(first.host, first.port, source, history, dependencies)
                self.assertEqual(raised.exception.errno, errno.EADDRINUSE)
                for connection in (source, history):
                    with self.assertRaises(sqlite3.ProgrammingError):
                        connection.execute("SELECT 1")
            self.assertEqual(first.request("/api/status"), before)
            self.assertEqual(before[0], HTTPStatus.OK)
            for connection in (first.source_conn, first.history_conn):
                self.assertEqual(connection.execute("SELECT 1").fetchone()[0], 1)
            self.assertEqual(
                [get_analysis_run(first.history_conn, run["run_id"]) for run in (queued, running)],
                records_before,
            )

    def test_recovery_failure_closes_bound_server_and_databases(self) -> None:
        with temporary_databases() as (history, source):
            history.execute("DROP TABLE analysis_runs")
            history.commit()
            dependencies = make_test_dependencies(
                source_connection=source, history_connection=history,
            )
            with patch.object(
                application_server.ApplicationServer, "server_close", autospec=True,
                side_effect=application_server.ApplicationServer.server_close,
            ) as close:
                with self.assertRaisesRegex(sqlite3.OperationalError, "no such table"):
                    create_application_server("127.0.0.1", 0, source, history, dependencies)
            close.assert_called_once()
            server = close.call_args.args[0]
            self.assertEqual(server.socket.fileno(), -1)
            for connection in (source, history):
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

    def test_main_cli_argument_parsing(self) -> None:
        with patch.object(application_server, "create_runtime_server") as create:
            server = create.return_value
            main(
                [
                    "--host", "0.0.0.0",
                    "--port", "8999",
                    "--public-host", "diagnosis.internal",
                    "--source-database", "source.db",
                    "--history-database", "history.db",
                    "--resources-directory", "resources",
                ]
            )

        create.assert_called_once_with(
            "0.0.0.0",
            8999,
            "diagnosis.internal",
            Path("source.db"),
            Path("history.db"),
            Path("resources"),
        )
        server.serve_forever.assert_called_once_with()
        server.server_close.assert_called_once_with()

    def test_main_defaults_to_team_network_binding(self) -> None:
        with patch.object(application_server, "create_runtime_server") as create:
            main(
                [
                    "--public-host", "diagnosis.internal",
                    "--source-database", "source.db",
                    "--history-database", "history.db",
                    "--resources-directory", "resources",
                ]
            )

        self.assertEqual(create.call_args.args[0], "0.0.0.0")

    def test_runtime_server_uses_explicit_paths_and_team_address(self) -> None:
        resources = SimpleNamespace(review_evidence={})
        source = MagicMock()
        history = MagicMock()
        server = SimpleNamespace(server_port=4321)
        with (
            patch.object(application_server, "load_model_resources", return_value=resources),
            patch.object(application_server, "open_source_database", return_value=source),
            patch.object(application_server, "open_history_database", return_value=history),
            patch.object(application_server, "build_runtime_dependencies", return_value="dependencies") as build,
            patch.object(application_server, "create_application_server", return_value=server) as create,
        ):
            actual = application_server.create_runtime_server(
                "0.0.0.0",
                0,
                "diagnosis.internal",
                Path("source.db"),
                Path("history.db"),
                Path("resources"),
            )

        self.assertIs(actual, server)
        build.assert_called_once_with(source, history, resources)
        create.assert_called_once_with("0.0.0.0", 0, source, history, "dependencies")
        self.assertEqual(
            server.service_status,
            {
                "status": "ready",
                "backend": "cpu_numba",
                "team_url": "http://diagnosis.internal:4321/",
            },
        )

    def test_runtime_dependencies_preserve_current_analysis_behavior(self) -> None:
        source = object()
        history = object()
        resources = SimpleNamespace(review_evidence={})
        catalog = [{"dataset_name": "Sample"}]
        with patch.object(application_server, "get_dataset_catalog", return_value=catalog) as get_catalog:
            dependencies = application_server.build_runtime_dependencies(source, history, resources)

        get_catalog.assert_called_once_with(source, resources)
        self.assertIs(dependencies.source_connection, source)
        self.assertIs(dependencies.history_connection, history)
        self.assertIsNot(dependencies.source_lock, dependencies.history_lock)
        self.assertEqual(dependencies.catalog, catalog)
        self.assertIs(dependencies.prepare_values, application_server.load_dataset_values)
        self.assertIs(dependencies.detect, application_server.run_anomaly_detection)
        self.assertIs(dependencies.load_labels, application_server.load_test_labels)
        self.assertIs(dependencies.evaluate, application_server.calculate_evaluation_metrics)
        self.assertIs(dependencies.load_series_for_chart, application_server.load_series_for_chart)
        self.assertIs(dependencies.review_evidence, resources.review_evidence)
        self.assertIs(dependencies.rank_review, application_server.rank_review_candidates)


if __name__ == "__main__":
    unittest.main()
