from http import HTTPStatus
import json
from pathlib import Path
import subprocess
import unittest
import urllib.request

from services.time_series_anomaly_diagnosis.tests.test_application_server import running_test_server

WEB = Path(__file__).resolve().parents[1] / "web"


class WebInterfaceTest(unittest.TestCase):
    def test_catalog_filter_combines_name_and_availability(self) -> None:
        script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const context = {document: {addEventListener() {}}, console};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const rows = [{dataset_name:'Trace', analysis_available:true},
              {dataset_name:'Trace_small', analysis_available:false},
              {dataset_name:'ECG', analysis_available:true}];
assert.equal(typeof context.filterCatalog, 'function', 'catalog availability filter is missing');
assert.equal(JSON.stringify(context.filterCatalog(rows, ' TRACE ', true)), JSON.stringify([rows[0]]));
assert.equal(JSON.stringify(context.filterCatalog(rows, '', false)), JSON.stringify(rows));
assert.equal(context.filterCatalog(rows, 'unknown', true).length, 0);
'''
        subprocess.run(['node', '-e', script, str(WEB / 'application.js')], check=True)

    def test_request_errors_preserve_connection_and_unrelated_alerts(self) -> None:
        node_script = r'''
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const context = {
  console,
  document: { addEventListener() {} },
  requestAnimationFrame() {},
  setTimeout,
  clearTimeout,
};
vm.createContext(context);
vm.runInContext(source + "\nthis.callRequestJson = requestJson;", context);

async function scenario(response) {
  const events = [];
  context.updateConnectionState = (connected) => events.push(["connection", connected]);
  context.showAlert = (message) => events.push(["alert", message]);
  context.clearAlert = () => events.push(["clear"]);
  context.fetch = response.fetch;
  let result = null;
  let error = null;
  try {
    result = await context.callRequestJson("/api/test");
  } catch (caught) {
    error = caught.message;
  }
  return { events, result, error };
}

(async () => {
  const http = await scenario({
    fetch: async () => ({
      ok: false,
      status: 409,
      json: async () => ({ error: "분석할 수 없는 데이터셋" }),
    }),
  });
  const network = await scenario({
    fetch: async () => { throw new TypeError("Failed to fetch"); },
  });
  const success = await scenario({
    fetch: async () => ({ ok: true, status: 200, json: async () => ({ ok: true }) }),
  });
  console.log(JSON.stringify({ http, network, success }));
})().catch((error) => { console.error(error); process.exit(1); });
'''
        completed = subprocess.run(
            ["node", "-e", node_script, str(WEB / "application.js")],
            check=True,
            capture_output=True,
            text=True,
        )
        actual = json.loads(completed.stdout)

        self.assertEqual(
            actual["http"]["events"],
            [
                ["connection", True],
                ["alert", "요청 실패: 분석할 수 없는 데이터셋"],
            ],
        )
        self.assertEqual(actual["http"]["error"], "분석할 수 없는 데이터셋")
        self.assertEqual(
            actual["network"]["events"],
            [
                ["connection", False],
                ["alert", "네트워크 오류: Failed to fetch"],
            ],
        )
        self.assertEqual(actual["success"]["events"], [["connection", True]])
        self.assertEqual(actual["success"]["result"], {"ok": True})

    def test_workspace_uses_approved_structure_without_sample_rows(self) -> None:
        html = (WEB / "index.html").read_text(encoding="utf-8")
        script = (WEB / "application.js").read_text(encoding="utf-8")

        for element_id in (
            "analysis-view",
            "queue-view",
            "history-view",
            "dataset-search",
            "dataset-list",
            "selected-dataset-info",
            "start-analysis",
            "result-workspace",
            "series-chart",
            "global-alert",
            "connection-state",
            "queue-list",
            "history-list",
            "detection-summary",
            "automatic-detections",
            "resolution-limited-warning",
            "multiple-evidence-review",
            "model-evaluation",
        ):
            self.assertIn(f'id="{element_id}"', html)

        self.assertNotIn("Trace_normal_2", html)
        self.assertIn("/api/status", script)
        self.assertNotIn("/api/upload", html + script)

    def test_workspace_navigation_and_catalog_are_accessible_real_api_controls(self) -> None:
        html = (WEB / "index.html").read_text(encoding="utf-8")
        script = (WEB / "application.js").read_text(encoding="utf-8")

        self.assertIn('data-view="analysis-view" aria-current="page"', html)
        self.assertIn('id="queue-view"', html)
        self.assertIn('id="history-view"', html)
        self.assertIn('role="listbox"', html)
        self.assertIn('aria-live="polite"', html)
        self.assertIn('role="alert"', html)
        self.assertIn('e.key === "Enter"', script)
        self.assertIn('item.dataset_name.toLowerCase().includes(query)', script)
        self.assertIn("목록을 불러오지 못했습니다.", script)

    def test_served_workspace_has_resolution_limited_warning_region(self) -> None:
        with running_test_server() as ctx:
            with urllib.request.urlopen(f"http://{ctx.host}:{ctx.port}/") as response:
                html = response.read().decode("utf-8")

        self.assertIn('id="detection-summary"', html)
        self.assertIn('id="resolution-limited-warning"', html)
        self.assertIn('id="multiple-evidence-review"', html)
        self.assertIn("표본 수 제한 자동 확정 경고", html)
        self.assertNotIn("과거 동결 Hard", html)
        self.assertLess(
            html.index('id="resolution-limited-warning"'),
            html.index('id="multiple-evidence-review"'),
        )
        self.assertLess(
            html.index('id="multiple-evidence-review"'),
            html.index('id="model-evaluation"'),
        )

    def test_workspace_has_required_controls_and_no_internal_experiment_names(self) -> None:
        html = (WEB / "index.html").read_text(encoding="utf-8")
        script = (WEB / "application.js").read_text(encoding="utf-8")
        styles = (WEB / "service_styles.css").read_text(encoding="utf-8")

        for text in (html, script, styles):
            self.assertNotRegex(text.lower(), r"antigravity|exp[0-9]+")

        for element_id in (
            "dataset-search",
            "dataset-list",
            "start-analysis",
            "queue-list",
            "history-list",
            "result-workspace",
            "series-chart",
            "detection-summary",
            "automatic-detections",
            "resolution-limited-warning",
            "multiple-evidence-review",
            "model-evaluation",
        ):
            self.assertIn(f'id="{element_id}"', html)

        self.assertIn("시계열 이상 진단", html)
        self.assertIn("새 분석", html)
        self.assertIn("작업 대기열", html)
        self.assertIn("분석 기록", html)
        self.assertIn('role="alert"', html)

        for fn_name in (
            "loadDatasetCatalog",
            "submitAnalysis",
            "refreshAnalysisRuns",
            "openAnalysisRun",
            "drawSeriesChart",
        ):
            self.assertIn(fn_name, script)

        self.assertIn("/api/analysis-runs", script)
        self.assertIn("/api/datasets", script)
        self.assertIn("requestAnimationFrame", script)
        self.assertIn("@media", styles)
        for stale_text in (
            "frozen_hard_confirmed_review_indices",
            "small_sample_review_indices",
            "resolution_limited_review_indices",
            "과거 Hard",
            "priority-one",
            "priority-two",
            "priority-three",
        ):
            self.assertNotIn(stale_text, html + script)

        self.assertIn("resolution_limited_warning_indices", script)
        self.assertIn("표본 수 제한으로 자동 확정", script)
        self.assertIn("multiple_evidence_review", script)
        self.assertIn("renderMultipleEvidenceReview", script)

    def test_series_request_is_bound_to_the_open_analysis_run(self) -> None:
        script = (WEB / "application.js").read_text(encoding="utf-8")
        render_result = script[script.index("function renderAnalysisResult"):]
        self.assertLess(
            render_result.index("clearActiveSeries();"),
            render_result.index("if (!result)"),
        )

        open_series = script[script.index("async function openSeries"):]
        self.assertIn("const requestedRunId = activeRunRecord.run_id;", open_series)
        self.assertIn("activeRunRecord.run_id !== requestedRunId", open_series)

    def test_static_file_http_routes_and_exact_mime_types(self) -> None:
        index_bytes = (WEB / "index.html").read_bytes()
        script_bytes = (WEB / "application.js").read_bytes()
        styles_bytes = (WEB / "service_styles.css").read_bytes()

        with running_test_server() as ctx:
            for path, expected_type, expected_body in (
                ("/", "text/html; charset=utf-8", index_bytes),
                ("/application.js", "text/javascript; charset=utf-8", script_bytes),
                ("/service_styles.css", "text/css; charset=utf-8", styles_bytes),
            ):
                req = urllib.request.Request(f"http://{ctx.host}:{ctx.port}{path}", method="GET")
                with urllib.request.urlopen(req) as resp:
                    self.assertEqual(resp.status, HTTPStatus.OK)
                    self.assertEqual(resp.headers.get("Content-Type"), expected_type)
                    self.assertEqual(resp.read(), expected_body)

            status, body = ctx.request("/unknown_file.html")
            self.assertEqual(status, HTTPStatus.NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
