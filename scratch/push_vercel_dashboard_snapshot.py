#!/usr/bin/env python3
"""Push a strict, sanitized dashboard status snapshot to the Vercel ingest API."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import importlib.util
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
MAX_PAYLOAD_BYTES = 256 * 1024
DEFAULT_SOURCE_URL = None
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from operational.alpha_setting import (  # noqa: E402
    DEFAULT_ALPHA_CONFIG_PATH,
    DEFAULT_ALPHA_SOURCE_ID,
    OperationalAlphaClient,
    apply_pending_alpha_setting,
)

DEFAULT_HEARTBEAT = Path("/Users/minho/Documents/Dataset/vercel_dashboard_exporter_heartbeat.json")
QUEUE_FIELDS = {
    "id", "experiment_id", "label", "status", "phase", "scope", "source",
    "datasets_done", "expected_datasets", "done", "expected", "progress_percent",
    "errors", "rows", "started_at", "updated_at", "blocked_reason",
}
RESULT_FIELDS = {
    "id", "label", "best_strategy", "mean_f1", "median_f1", "mean_auc_roc",
    "mean_auc_pr", "mean_oracle_f1", "zero_f1_count", "mean_predicted_count",
    "mean_tp", "mean_fp", "mean_fn", "alert_precision", "mean_combined_f1",
    "standard_review_candidate_count", "priority_review_candidate_count",
    "total_review_candidate_count", "review_candidate_count", "selected_alpha",
    "autonomous_metric_scope", "review_metric_scope", "history_status",
    "comparison_eligible", "scope", "status_reason", "mean_train_exceed_rate",
    "operational_tier", "operational_excluded", "operational_default",
    "exclusion_reason", "datasets", "rows", "elapsed_minutes", "finished_at",
}
SUMMARY_FIELDS = {
    "coverage": {"datasets_done", "expected_datasets", "error_datasets", "remaining_or_missing", "coverage_percent"},
    "throughput": {"recent_datasets_per_minute", "average_datasets_per_minute", "recent_eta_seconds", "eta_basis", "progress_log_gap"},
    "runtime_health": {"started_at", "uptime_seconds", "code_mtime", "heartbeat"},
    "warning_summary": {"total_warnings"},
    "error_summary": {"total_errors", "error_datasets", "by_reason", "by_family"},
}


class SupabaseSession:
    def __init__(self, url, publishable_key, email, password):
        self.url = url.rstrip("/")
        self.publishable_key = publishable_key
        self.email = email
        self.password = password
        self.access_token = None
        self.refresh_token = None
        self.expires_at = 0.0

    def update_tokens(self, response):
        self.access_token = response["access_token"]
        self.refresh_token = response.get("refresh_token")
        self.expires_at = time.time() + max(0, int(response.get("expires_in", 3600)) - 60)

    def _request_token(self, grant_type, payload):
        request = urllib.request.Request(
            f"{self.url}/auth/v1/token?grant_type={grant_type}",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"apikey": self.publishable_key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.load(response)
        self.update_tokens(result)

    def current_access_token(self):
        if self.access_token and time.time() < self.expires_at:
            return self.access_token
        if self.refresh_token:
            try:
                self._request_token("refresh_token", {"refresh_token": self.refresh_token})
                return self.access_token
            except urllib.error.HTTPError:
                self.refresh_token = None
        self._request_token("password", {"email": self.email, "password": self.password})
        return self.access_token


def _safe_value(value, depth=0):
    if depth > 3:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_safe_value(item, depth + 1) for item in value[:50]]
    if isinstance(value, dict):
        return {str(key): _safe_value(item, depth + 1) for key, item in list(value.items())[:50]}
    return str(value)


def _select(record, fields):
    if not isinstance(record, dict):
        return {}
    return {key: _safe_value(record[key]) for key in fields if key in record}


def _number(value, default=0):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _nullable_number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def sanitize_status(status, source_instance_id):
    queue = [_select(row, QUEUE_FIELDS) for row in status.get("queue_items", [])[:100] if isinstance(row, dict)]
    recent_results = status.get("completed_results", [])[-50:]
    results = [_select(row, RESULT_FIELDS) for row in reversed(recent_results) if isinstance(row, dict)]
    summaries = {key: _select(status.get(key), fields) for key, fields in SUMMARY_FIELDS.items()}
    runtime_source = status.get("runtime_health") if isinstance(status.get("runtime_health"), dict) else {}
    summaries["runtime_health"]["heartbeat"] = _select(
        runtime_source.get("heartbeat"),
        {"age_seconds", "status", "ok", "updated_at", "stalled", "progress_percent", "datasets_done"},
    )
    system = status.get("system") if isinstance(status.get("system"), dict) else {}
    safe_system = {
        "cpu": _select(system.get("cpu"), {"percent", "cpu_percent", "cpu_user_percent", "cpu_system_percent", "cpu_idle_percent", "load_percent", "count", "available"}),
        "gpu": _select(system.get("gpu"), {"percent", "gpu_percent", "utilization", "memory_percent", "device", "available", "note"}),
    }
    process_summary = _select(status.get("process"), {"process_count", "cpu", "mem", "etime"})
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_instance_id": source_instance_id,
        "active_experiment": status.get("active_experiment") if isinstance(status.get("active_experiment"), str) else None,
        "active_label": status.get("active_label") if isinstance(status.get("active_label"), str) else None,
        "running": bool(status.get("running")),
        "datasets_done": _number(status.get("datasets_done")),
        "expected_datasets": _number(status.get("expected_datasets")),
        "progress_percent": _number(status.get("progress_percent")),
        "elapsed_seconds": _nullable_number(status.get("elapsed_seconds")),
        "eta_seconds": _nullable_number(status.get("eta_seconds")),
        "current_dataset": status.get("current_dataset") if isinstance(status.get("current_dataset"), str) else None,
        "queue_items": queue,
        "coverage": summaries["coverage"],
        "throughput": summaries["throughput"],
        "runtime_health": summaries["runtime_health"],
        "system": safe_system,
        "process_summary": process_summary,
        "warning_summary": summaries["warning_summary"],
        "error_summary": summaries["error_summary"],
        "last_completed_experiment": status.get("last_completed_experiment") if isinstance(status.get("last_completed_experiment"), str) else None,
        "last_completed_label": status.get("last_completed_label") if isinstance(status.get("last_completed_label"), str) else None,
        "completed_results": results,
    }


def default_source_id():
    return "mac-" + hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]


def fetch_local_status(url, username=None, password=None, timeout=10):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if username and password:
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def build_direct_status():
    """Read experiment state directly without running the local web dashboard."""
    module_path = REPO_ROOT / "serve_rank_dashboard.py"
    spec = importlib.util.spec_from_file_location("rank_dashboard_status_source", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dashboard status source: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_status(include_details=True)


def read_status(source_url=None, username=None, password=None):
    if source_url:
        return fetch_local_status(source_url, username, password)
    return build_direct_status()


def push_snapshot(url, token, supabase_access_token, snapshot, timeout=15):
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode()
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"sanitized snapshot is too large: {len(payload)} bytes")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Supabase-Access-Token": supabase_access_token,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def write_heartbeat(path, **payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def sync_operational_alpha(
    access_token,
    source_id,
    config_path,
    supabase_url,
    publishable_key,
):
    client = OperationalAlphaClient(supabase_url, publishable_key)
    return apply_pending_alpha_setting(
        client,
        access_token,
        source_id=source_id,
        config_path=config_path,
    )


def run_once(args, supabase_session):
    status = read_status(args.source_url, args.local_user, args.local_password)
    snapshot = sanitize_status(status, args.source_instance_id)
    access_token = supabase_session.current_access_token()
    alpha_setting = {"status": "skipped"}
    if not args.skip_alpha_sync:
        try:
            alpha_setting = sync_operational_alpha(
                access_token,
                args.alpha_source_id,
                args.alpha_config,
                args.supabase_url,
                args.supabase_publishable_key,
            )
        except Exception as error:
            alpha_setting = {
                "status": "sync_error",
                "error": str(error)[:500],
            }
    result = push_snapshot(
        args.target_url,
        args.ingest_token,
        access_token,
        snapshot,
    )
    write_heartbeat(
        args.heartbeat,
        ok=True,
        pushed_at=datetime.now(timezone.utc).isoformat(),
        active_experiment=snapshot["active_experiment"],
        progress_percent=snapshot["progress_percent"],
        target_url=args.target_url,
        alpha_setting=alpha_setting,
        response=result,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-url",
        default=os.environ.get("DASHBOARD_SOURCE_URL", DEFAULT_SOURCE_URL),
        help="Optional diagnostic HTTP source. Omit to read experiment state directly.",
    )
    parser.add_argument("--target-url", default=os.environ.get("DASHBOARD_INGEST_URL"))
    parser.add_argument("--ingest-token", default=os.environ.get("DASHBOARD_INGEST_TOKEN"))
    parser.add_argument("--supabase-url", default=os.environ.get("NEXT_PUBLIC_SUPABASE_URL", os.environ.get("SUPABASE_URL")))
    parser.add_argument("--supabase-publishable-key", default=os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", os.environ.get("SUPABASE_PUBLISHABLE_KEY")))
    parser.add_argument("--supabase-email", default=os.environ.get("DASHBOARD_LOGIN_EMAIL"))
    parser.add_argument("--supabase-password", default=os.environ.get("DASHBOARD_LOGIN_PASSWORD"))
    parser.add_argument("--local-user", default=os.environ.get("RANK_DASHBOARD_USER"))
    parser.add_argument("--local-password", default=os.environ.get("RANK_DASHBOARD_PASSWORD"))
    parser.add_argument("--source-instance-id", default=os.environ.get("DASHBOARD_SOURCE_INSTANCE_ID", default_source_id()))
    parser.add_argument("--heartbeat", type=Path, default=DEFAULT_HEARTBEAT)
    parser.add_argument(
        "--alpha-source-id",
        default=os.environ.get("DASHBOARD_ALPHA_SOURCE_ID", DEFAULT_ALPHA_SOURCE_ID),
    )
    parser.add_argument(
        "--alpha-config",
        type=Path,
        default=Path(os.environ.get("DASHBOARD_ALPHA_CONFIG", DEFAULT_ALPHA_CONFIG_PATH)),
    )
    parser.add_argument("--skip-alpha-sync", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    required = {
        "target URL": args.target_url,
        "ingest token": args.ingest_token,
        "Supabase URL": args.supabase_url,
        "Supabase publishable key": args.supabase_publishable_key,
        "Supabase email": args.supabase_email,
        "Supabase password": args.supabase_password,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("missing required configuration: " + ", ".join(missing))
    supabase_session = SupabaseSession(
        args.supabase_url,
        args.supabase_publishable_key,
        args.supabase_email,
        args.supabase_password,
    )

    while True:
        try:
            result = run_once(args, supabase_session)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except (OSError, ValueError, urllib.error.URLError) as error:
            write_heartbeat(
                args.heartbeat,
                ok=False,
                failed_at=datetime.now(timezone.utc).isoformat(),
                error=str(error),
                target_url=args.target_url,
            )
            print(f"dashboard exporter error: {error}", flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
