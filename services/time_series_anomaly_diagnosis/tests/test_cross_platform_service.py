"""Portable runtime contract using synthetic data, not real Trace performance."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from http import HTTPStatus
import json
import math
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from services.time_series_anomaly_diagnosis.application_server import create_runtime_server
from services.time_series_anomaly_diagnosis.dataset_importer import install_database
from services.time_series_anomaly_diagnosis.tests.test_dataset_importer import (
    add_source_dataset,
    make_source_database,
    values_blob,
)

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
REGISTERED = "Trace_normal_2"
UNKNOWN = "SyntheticUnregistered_normal_7"


@contextmanager
def running_runtime(source: Path, history: Path):
    server = create_runtime_server("127.0.0.1", 0, "127.0.0.1", source, history, RESOURCES)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def request_json(server, path: str, payload: dict | None = None):
    request = Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urlopen(request, timeout=10)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.loads(response.read())


class CrossPlatformServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.raw = root / "synthetic_raw.sqlite3"
        self.data = root / "data"
        self.installed = self.data / "univariate_ts.db"
        self.history = self.data / "analysis_history.sqlite3"
        # Borrow only the registered route name; every waveform and label is synthetic.
        make_source_database(
            self.raw, name=REGISTERED,
            train_labels=("7",) * 9 + ("0",) * 3,
            test_labels=("7", "0", "7", "0"),
            blobs=tuple(
                values_blob(*(
                    math.sin(position * (0.12 + index * 0.013))
                    + 0.15 * math.cos(position * 0.37 + index * 0.23)
                    for position in range(64)
                ))
                for index in range(16)
            ),
        )
        with closing(sqlite3.connect(self.raw)) as connection:
            connection.execute("UPDATE datasets SET series_length=64 WHERE id=1")
            add_source_dataset(connection, 2, UNKNOWN, ("7",) * 5 + ("0",), ("7", "0"))
            connection.commit()
        self.raw_bytes = self.raw.read_bytes()
        self.summary = install_database(self.raw, self.data, root / "import_report.html")

    def test_ten_clients_read_nonempty_catalog_and_upload_is_absent(self) -> None:
        with running_runtime(self.installed, self.history) as server:
            barrier = Barrier(10)

            def read_catalog(_client):
                barrier.wait(timeout=10)
                return request_json(server, "/api/datasets")

            with ThreadPoolExecutor(max_workers=10) as clients:
                responses = list(clients.map(read_catalog, range(10)))
            expected = [
                {
                    "dataset_name": UNKNOWN, "method_name": None,
                    "train_count": 6, "test_count": 2, "series_length": 2,
                    "availability": "missing_verified_route", "analysis_available": False,
                },
                {
                    "dataset_name": REGISTERED, "method_name": "minirocket_local_gap",
                    "train_count": 12, "test_count": 4, "series_length": 64,
                    "availability": "standard", "analysis_available": True,
                },
            ]
            for status, catalog in responses:
                self.assertEqual(status, HTTPStatus.OK)
                self.assertEqual(catalog, expected)
            self.assertEqual(
                request_json(server, "/api/upload", {"filename": "unused.db"}),
                (HTTPStatus.NOT_FOUND, {"error": "route not found"}),
            )

    def test_registered_majority_labels_and_unknown_route_rejection(self) -> None:
        self.assertEqual({row.normal_source_label for row in self.summary.datasets}, {"7"})
        with running_runtime(self.installed, self.history) as server:
            status, catalog = request_json(server, "/api/datasets")
            self.assertEqual(status, HTTPStatus.OK)
            unknown = next(row for row in catalog if row["dataset_name"] == UNKNOWN)
            self.assertIsNone(unknown["method_name"])
            self.assertFalse(unknown["analysis_available"])
            self.assertEqual(unknown["availability"], "missing_verified_route")
            self.assertEqual(
                request_json(server, "/api/analysis-runs", {"dataset_name": UNKNOWN}),
                (HTTPStatus.CONFLICT, {"error": "missing_verified_route"}),
            )
            self.assertEqual(request_json(server, "/api/analysis-runs"), (HTTPStatus.OK, []))
            with server.dependencies.source_lock:
                rows = server.source_connection.execute(
                    "SELECT i.split, i.label FROM instances i JOIN datasets d ON d.id=i.dataset_id "
                    "WHERE d.name=? ORDER BY CASE i.split WHEN 'TRAIN' THEN 0 ELSE 1 END, i.instance_index",
                    (REGISTERED,),
                ).fetchall()
                self.assertEqual(
                    [tuple(row) for row in rows],
                    [("TRAIN", "0")] * 9 + [("TRAIN", "1")] * 3
                    + [("TEST", "0"), ("TEST", "1"), ("TEST", "0"), ("TEST", "1")],
                )
                with self.assertRaisesRegex(sqlite3.OperationalError, "readonly"):
                    server.source_connection.execute("UPDATE datasets SET name='changed'")
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)

    def test_real_analysis_result_survives_runtime_restart(self) -> None:
        installed_bytes = self.installed.read_bytes()
        with running_runtime(self.installed, self.history) as server:
            status, submitted = request_json(server, "/api/analysis-runs", {"dataset_name": REGISTERED})
            self.assertEqual(status, HTTPStatus.ACCEPTED)
            path = f"/api/analysis-runs/{submitted['run_id']}"
            deadline = time.monotonic() + 120
            while True:
                status, completed = request_json(server, path)
                self.assertEqual(status, HTTPStatus.OK)
                if completed["status"] not in {"queued", "running"} or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            self.assertEqual(completed["status"], "completed", completed)
            self.assertIsNone(completed["error_message"])
            self.assertFalse(completed["requires_reanalysis"])
            result = completed["result"]
            self.assertEqual(result["dataset_name"], REGISTERED)
            self.assertEqual(result["method_name"], "minirocket_local_gap")
            self.assertEqual(result["target_length"], 64)
            self.assertEqual(sorted(row["series_index"] for row in result["rows"]), list(range(4)))
            scores = [row["score"] for row in result["rows"]]
            self.assertTrue(all(math.isfinite(score) for score in scores))
            self.assertGreater(len(set(scores)), 1)
            self.assertTrue(all(0 < row["p_value"] <= 1 for row in result["rows"]))
            self.assertEqual(
                result["automatic_anomaly_indices"],
                [row["series_index"] for row in result["rows"] if row["automatic_anomaly"]],
            )
            metrics = result["metrics"]
            self.assertTrue(math.isfinite(metrics["auc"]))
            self.assertEqual(metrics["true_positives"] + metrics["false_negatives"], 2)
            self.assertEqual(metrics["true_negatives"] + metrics["false_positives"], 2)
            chart_status, chart = request_json(server, path + "/series/0")
            self.assertEqual(chart_status, HTTPStatus.OK)
            for field in ("test_values", "train_median", "train_q10", "train_q90"):
                self.assertEqual(len(chart[field]), 64)
                self.assertTrue(all(math.isfinite(value) for value in chart[field]))

        with running_runtime(self.installed, self.history) as restarted:
            self.assertEqual(request_json(restarted, path), (HTTPStatus.OK, completed))
            self.assertEqual(request_json(restarted, path + "/series/0"), (HTTPStatus.OK, chart))
            status, runs = request_json(restarted, "/api/analysis-runs")
            self.assertEqual(status, HTTPStatus.OK)
            self.assertEqual([(run["run_id"], run["status"]) for run in runs], [(submitted["run_id"], "completed")])
        self.assertEqual(self.raw.read_bytes(), self.raw_bytes)
        self.assertEqual(self.installed.read_bytes(), installed_bytes)


if __name__ == "__main__":
    unittest.main()
