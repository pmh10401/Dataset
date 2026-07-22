#!/usr/bin/env python3
"""Single-process controller for the read-only rank experiment dashboard."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/minho/Documents/Dataset")
PID_PATH = DATA_DIR / "rank_dashboard_server.pid"
LOG_PATH = DATA_DIR / "rank_dashboard_server.log"
SERVER_SCRIPT = ROOT / "serve_rank_dashboard.py"
PORT = int(os.environ.get("RANK_DASHBOARD_PORT", "8765"))


def read_pid_metadata(path=PID_PATH):
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            return {"pid": int(raw), "legacy": True}
        except ValueError:
            return {}
    if isinstance(value, int):
        return {"pid": value, "legacy": True}
    if not isinstance(value, dict):
        return {}
    try:
        value["pid"] = int(value["pid"])
    except (KeyError, TypeError, ValueError):
        return {}
    return value


def read_pid():
    return read_pid_metadata().get("pid")


def health_url(port=PORT):
    return f"http://127.0.0.1:{int(port)}/api/healthz"


def health_matches_process(payload, pid):
    try:
        return int((payload or {}).get("server", {}).get("pid")) == int(pid)
    except (TypeError, ValueError):
        return False


def request_json(url, timeout=10):
    request = Request(url)
    user = os.environ.get("RANK_DASHBOARD_USER")
    password = os.environ.get("RANK_DASHBOARD_PASSWORD")
    if user and password:
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def command_for_pid(pid):
    try:
        return subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True).strip()
    except subprocess.CalledProcessError:
        return ""


def dashboard_pid():
    pid = read_pid()
    if not pid:
        return None
    command = command_for_pid(pid)
    return pid if "serve_rank_dashboard.py" in command else None


def port_listener_pid(port=PORT):
    try:
        output = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{int(port)}", "-sTCP:LISTEN", "-t"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return int(output.splitlines()[0]) if output else None
    except ValueError:
        return None


def listener_dashboard_pid(port=PORT):
    """Return the dashboard listener PID only when the port owner is this dashboard."""

    pid = port_listener_pid(port)
    if pid and "serve_rank_dashboard.py" in command_for_pid(pid):
        return pid
    return None


def controlled_dashboard_pid():
    """Prefer the recorded PID, but recover safely from a stale PID file."""

    return dashboard_pid() or listener_dashboard_pid(PORT)


def status():
    pid = controlled_dashboard_pid()
    payload = {"pid": pid, "port": PORT, "reachable": False}
    try:
        body = request_json(f"http://127.0.0.1:{PORT}/api/status")
        health = request_json(health_url(PORT))
        health_pid = (health.get("server") or {}).get("pid")
        listener_pid = port_listener_pid(PORT)
        pid_matches = bool(health_pid) and (pid is None or health_matches_process(health, pid))
        listener_matches = listener_pid is None or int(listener_pid) == int(health_pid)
        payload.update(
            {
                "reachable": pid_matches and listener_matches,
                "running": body.get("running"),
                "active_experiment": body.get("active_experiment"),
                "last_completed_experiment": body.get("last_completed_experiment"),
                "runtime_health": body.get("runtime_health"),
                "server": health.get("server"),
                "listener_pid": listener_pid,
                "pid_mismatch": not (pid_matches and listener_matches),
            }
        )
    except Exception as exc:
        payload["error"] = str(exc)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["reachable"] else 1


def stop():
    pid = controlled_dashboard_pid()
    if not pid:
        print("dashboard not running")
        return 0
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not controlled_dashboard_pid():
            break
        time.sleep(0.1)
    if controlled_dashboard_pid():
        raise SystemExit(f"dashboard pid {pid} did not stop")
    listener = port_listener_pid(PORT)
    if listener:
        raise SystemExit(f"dashboard port {PORT} is still owned by pid {listener}")
    print(f"dashboard stopped pid={pid}")
    return 0


def start():
    pid = controlled_dashboard_pid()
    if pid:
        try:
            health = request_json(health_url(PORT))
        except Exception as exc:
            raise SystemExit(f"dashboard pid {pid} exists but health check failed: {exc}")
        if health_matches_process(health, pid):
            print(f"dashboard already running pid={pid}")
            return 0
        raise SystemExit(f"dashboard PID/health mismatch for pid {pid}")
    listener = port_listener_pid(PORT)
    if listener:
        command = command_for_pid(listener)
        raise SystemExit(f"port {PORT} is already in use by pid {listener}: {command}")
    deadline = time.time() + 15
    while time.time() < deadline:
        with LOG_PATH.open("a") as log_file:
            process = subprocess.Popen(
                [sys.executable, str(SERVER_SCRIPT)],
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(10):
            pid = controlled_dashboard_pid()
            if pid:
                try:
                    health = request_json(health_url(PORT), timeout=1)
                except Exception:
                    health = {}
                if health_matches_process(health, pid):
                    print(f"dashboard started pid={pid} port={PORT}")
                    return 0
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if process.poll() is None:
            process.terminate()
        time.sleep(0.5)
    raise SystemExit("dashboard failed to start; inspect rank_dashboard_server.log")


def main():
    parser = argparse.ArgumentParser(description="Control the local rank experiment dashboard")
    parser.add_argument("command", choices=("start", "stop", "restart", "status"))
    args = parser.parse_args()
    if args.command == "start":
        return start()
    if args.command == "stop":
        return stop()
    if args.command == "restart":
        stop()
        return start()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
