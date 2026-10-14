from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import socket
import sqlite3
from threading import Lock
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/application.js": ("application.js", "text/javascript; charset=utf-8"),
    "/service_styles.css": ("service_styles.css", "text/css; charset=utf-8"),
}

from .analysis_history import (
    create_analysis_run,
    get_analysis_run,
    list_analysis_runs,
    mark_analysis_running,
    mark_interrupted_runs,
    open_history_database,
    save_completed_analysis,
    save_failed_analysis,
)
from .anomaly_detection import (
    ALPHA,
    calculate_evaluation_metrics,
    run_anomaly_detection,
)
from .dataset_repository import (
    get_dataset_catalog,
    load_dataset_values,
    load_series_for_chart,
    load_test_labels,
    open_source_database,
)
from .model_resources import load_model_resources
from .multi_evidence_review import rank_review_candidates

RUN_SERIES_PATH = re.compile(r"^/api/analysis-runs/([^/]+)/series/([^/]+)$")
RUN_PATH = re.compile(r"^/api/analysis-runs/([^/]+)$")
LIVE_RESULT_SCHEMA = "live_analysis_result_v3"


def execute_analysis_run(
    run_id: str,
    dataset_entry: dict[str, Any],
    dependencies: Any,
) -> None:
    with dependencies.history_lock:
        mark_analysis_running(dependencies.history_connection, run_id)
    try:
        with dependencies.source_lock:
            prepared = dependencies.prepare_values(
                dependencies.source_connection,
                dataset_entry["dataset_name"],
                dataset_entry["availability"],
            )
        detection = dependencies.detect(prepared, dataset_entry["method_name"])
        multiple_evidence_review = dependencies.rank_review(
            dataset_entry["dataset_name"],
            detection.rows,
            dependencies.review_evidence,
        )
        resolution_limited_warning_indices = tuple(
            getattr(detection, "resolution_limited_warning_indices", ())
        )
        reference_count = len(prepared.train_values)
        minimum_p_value = 1.0 / (reference_count + 1.0)
        with dependencies.source_lock:
            labels = dependencies.load_labels(
                dependencies.source_connection,
                dataset_entry["dataset_name"],
                prepared.test_indices,
            )
        metrics = dependencies.evaluate(detection.rows, labels)
        result = {
            "schema_version": LIVE_RESULT_SCHEMA,
            "analysis_policy": "registered_dataset_best_method",
            "dataset_name": dataset_entry["dataset_name"],
            "method_name": detection.method_name,
            "selected_method": detection.method_name,
            "availability": dataset_entry["availability"],
            "rows": list(detection.rows),
            "automatic_anomaly_indices": list(detection.automatic_anomaly_indices),
            "resolution_limited_warning_indices": list(
                resolution_limited_warning_indices
            ),
            "automatic_promotion": bool(resolution_limited_warning_indices),
            "multiple_evidence_review": multiple_evidence_review,
            "calibration": {
                "reference_count": reference_count,
                "minimum_p_value": minimum_p_value,
                "automatic_threshold": ALPHA,
                "automatic_threshold_attainable": minimum_p_value <= ALPHA + 1e-15,
            },
            "train_reference": detection.train_reference,
            "target_length": detection.target_length,
            "metrics": metrics,
        }
        with dependencies.history_lock:
            save_completed_analysis(dependencies.history_connection, run_id, result)
    except Exception as error:
        with dependencies.history_lock:
            save_failed_analysis(dependencies.history_connection, run_id, str(error))


class ApplicationRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _send_static_file(self, filename: str, content_type: str) -> None:
        target = WEB_DIR / filename
        if not target.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "static file not found")
            return
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            self._send_static_file(filename, content_type)
            return

        if path == "/api/status":
            self._send_json(HTTPStatus.OK, self.server.service_status)
            return

        if path == "/api/datasets":
            self._send_json(HTTPStatus.OK, self.server.catalog)
            return

        if path == "/api/analysis-runs":
            with self.server.dependencies.history_lock:
                runs = list_analysis_runs(self.server.history_connection)
            self._send_json(HTTPStatus.OK, runs)
            return

        series_match = RUN_SERIES_PATH.match(path)
        if series_match:
            run_id = series_match.group(1)
            series_index_str = series_match.group(2)
            self._handle_get_series(run_id, series_index_str)
            return

        run_match = RUN_PATH.match(path)
        if run_match:
            run_id = run_match.group(1)
            self._handle_get_run(run_id)
            return

        self._send_error(HTTPStatus.NOT_FOUND, "route not found")

    def _handle_get_run(self, run_id: str) -> None:
        with self.server.dependencies.history_lock:
            record = get_analysis_run(self.server.history_connection, run_id)
        if record is None:
            self._send_error(HTTPStatus.NOT_FOUND, "analysis run not found")
            return
        result = record.get("result")
        record["requires_reanalysis"] = bool(
            result is not None and result.get("schema_version") != LIVE_RESULT_SCHEMA
        )
        if record["requires_reanalysis"]:
            record["result"] = None
        self._send_json(HTTPStatus.OK, record)

    def _handle_get_series(self, run_id: str, series_index_str: str) -> None:
        with self.server.dependencies.history_lock:
            record = get_analysis_run(self.server.history_connection, run_id)
        if record is None:
            self._send_error(HTTPStatus.NOT_FOUND, "analysis run not found")
            return

        result = record.get("result")
        if result is not None and result.get("schema_version") != LIVE_RESULT_SCHEMA:
            self._send_error(
                HTTPStatus.CONFLICT,
                "analysis result requires reanalysis with the current database pipeline",
            )
            return
        if result is None:
            self._send_error(
                HTTPStatus.CONFLICT,
                f"analysis run has no completed result (status: {record.get('status')})",
            )
            return

        try:
            series_index = int(series_index_str)
        except ValueError:
            self._send_error(HTTPStatus.NOT_FOUND, "series index must be an integer")
            return

        valid_indices = {row["series_index"] for row in result.get("rows", [])}
        if series_index not in valid_indices:
            self._send_error(HTTPStatus.NOT_FOUND, f"series index {series_index} is out of range")
            return

        try:
            with self.server.dependencies.source_lock:
                test_values = self.server.dependencies.load_series_for_chart(
                    self.server.source_connection,
                    result["dataset_name"],
                    series_index,
                    result["availability"],
                    result["target_length"],
                )
        except Exception as error:
            self._send_error(HTTPStatus.NOT_FOUND, str(error))
            return

        train_ref = result.get("train_reference", {})
        payload = {
            "series_index": series_index,
            "test_values": test_values,
            "train_median": train_ref.get("median", []),
            "train_q10": train_ref.get("q10", []),
            "train_q90": train_ref.get("q90", []),
        }
        self._send_json(HTTPStatus.OK, payload)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/analysis-runs":
            self._send_error(HTTPStatus.NOT_FOUND, "route not found")
            return

        content_length_header = self.headers.get("Content-Length")
        if content_length_header is None:
            self._send_error(HTTPStatus.BAD_REQUEST, "Content-Length header required")
            return

        try:
            content_length = int(content_length_header)
            if content_length < 0:
                raise ValueError
        except ValueError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            return

        raw_body = self.rfile.read(content_length)
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._send_error(HTTPStatus.BAD_REQUEST, "request body must be valid JSON")
            return

        if not isinstance(data, dict) or "dataset_name" not in data:
            self._send_error(HTTPStatus.BAD_REQUEST, "request body must be an object with dataset_name")
            return

        dataset_name = data["dataset_name"]
        if type(dataset_name) is not str or not dataset_name:
            self._send_error(HTTPStatus.BAD_REQUEST, "dataset_name must be a non-empty string")
            return

        dataset_entry = self.server.catalog_by_name.get(dataset_name)
        if dataset_entry is None:
            self._send_error(HTTPStatus.NOT_FOUND, "dataset is unknown")
            return

        if not dataset_entry.get("analysis_available"):
            self._send_error(
                HTTPStatus.CONFLICT,
                dataset_entry.get("availability", "dataset is not available for analysis"),
            )
            return

        with self.server.dependencies.history_lock:
            run = create_analysis_run(
                self.server.history_connection,
                dataset_name,
                dataset_entry["method_name"],
                dataset_entry["availability"],
            )
            self.server.analysis_executor.submit(
                execute_analysis_run,
                run["run_id"],
                dataset_entry,
                self.server.dependencies,
            )

        self._send_json(HTTPStatus.ACCEPTED, run)


class ApplicationServer(ThreadingHTTPServer):
    request_queue_size = 32

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        source_connection: sqlite3.Connection,
        history_connection: sqlite3.Connection,
        dependencies: Any,
    ):
        self.source_connection = source_connection
        self.history_connection = history_connection
        self.dependencies = dependencies
        self.analysis_executor = ThreadPoolExecutor(max_workers=1)
        self.catalog = list(dependencies.catalog)
        self.catalog_by_name = {entry["dataset_name"]: entry for entry in self.catalog}
        self.service_status = {
            "status": "ready",
            "backend": "cpu_numba",
            "team_url": "",
        }
        self._closed = False
        super().__init__(server_address, RequestHandlerClass)

    def server_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.analysis_executor.shutdown(wait=True, cancel_futures=True)
        super().server_close()
        self.source_connection.close()
        self.history_connection.close()


def create_application_server(
    host: str,
    port: int,
    source_connection: sqlite3.Connection,
    history_connection: sqlite3.Connection,
    dependencies: Any,
) -> ApplicationServer:
    server = ApplicationServer(
        (host, port),
        ApplicationRequestHandler,
        source_connection,
        history_connection,
        dependencies,
    )
    try:
        with dependencies.history_lock:
            mark_interrupted_runs(history_connection)
    except BaseException:
        server.server_close()
        raise
    return server


def build_runtime_dependencies(
    source: sqlite3.Connection,
    history: sqlite3.Connection,
    resources: Any,
) -> SimpleNamespace:
    source_lock = Lock()
    history_lock = Lock()
    with source_lock:
        catalog = get_dataset_catalog(source, resources)
    return SimpleNamespace(
        source_connection=source,
        source_lock=source_lock,
        history_connection=history,
        history_lock=history_lock,
        catalog=catalog,
        prepare_values=load_dataset_values,
        detect=run_anomaly_detection,
        load_labels=load_test_labels,
        evaluate=calculate_evaluation_metrics,
        load_series_for_chart=load_series_for_chart,
        review_evidence=resources.review_evidence,
        rank_review=rank_review_candidates,
    )


def create_runtime_server(
    host: str,
    port: int,
    public_host: str,
    source_database: Path,
    history_database: Path,
    resources_directory: Path,
) -> ApplicationServer:
    resources = load_model_resources(resources_directory)
    source = open_source_database(source_database)
    history = open_history_database(history_database)
    dependencies = build_runtime_dependencies(source, history, resources)
    server = create_application_server(host, port, source, history, dependencies)
    server.service_status = {
        "status": "ready",
        "backend": "cpu_numba",
        "team_url": f"http://{public_host}:{server.server_port}/",
    }
    return server


def main(arguments: list[str] | None = None) -> None:
    parser = ArgumentParser(description="Time Series Anomaly Diagnosis Application Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8485)
    parser.add_argument("--public-host", default=socket.gethostname())
    parser.add_argument("--source-database", type=Path, default=Path("univariate_ts.db"))
    parser.add_argument(
        "--history-database",
        type=Path,
        default=Path("services/time_series_anomaly_diagnosis/runtime/analysis_history.sqlite3"),
    )
    parser.add_argument(
        "--resources-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "resources",
    )
    args = parser.parse_args(arguments)

    server = create_runtime_server(
        args.host,
        args.port,
        args.public_host,
        args.source_database,
        args.history_database,
        args.resources_directory,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
