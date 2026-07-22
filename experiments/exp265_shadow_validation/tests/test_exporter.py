from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from experiments.exp265_shadow_validation.exporter import PermanentExportError, RetryableExportError, push_prediction
from experiments.exp265_shadow_validation.tests.test_spool import payload


class _Handler(BaseHTTPRequestHandler):
    status = 200
    received_headers: dict[str, str] = {}
    received_body = b""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        type(self).received_headers = dict(self.headers.items())
        type(self).received_body = self.rfile.read(length)
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": type(self).status < 300}).encode())

    def log_message(self, *_args: object) -> None:
        return


class ExporterTest(unittest.TestCase):
    def setUp(self) -> None:
        _Handler.status = 200
        _Handler.received_headers = {}
        _Handler.received_body = b""
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/api/shadow/ingest"

    def test_push_sends_only_required_auth_headers(self) -> None:
        receipt = push_prediction(self.url, "ingest", "session", payload())
        self.assertTrue(receipt["ok"])
        self.assertEqual(_Handler.received_headers["Authorization"], "Bearer ingest")
        self.assertEqual(_Handler.received_headers["X-Supabase-Access-Token"], "session")
        self.assertNotIn("y_test", _Handler.received_body.decode())

    def test_conflict_is_permanent(self) -> None:
        _Handler.status = 409
        with self.assertRaises(PermanentExportError):
            push_prediction(self.url, "ingest", "session", payload())

    def test_server_failure_is_retryable(self) -> None:
        _Handler.status = 503
        with self.assertRaises(RetryableExportError):
            push_prediction(self.url, "ingest", "session", payload())


if __name__ == "__main__":
    unittest.main()
