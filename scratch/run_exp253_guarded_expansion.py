from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp253_guarded_standard_to_hard_expansion.config import EXPERIMENT_ID, OUTPUT_ROOT

CANONICAL_STATE_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_sequential_state.json")
QUEUE_LOCK_PATH = Path("/Users/minho/Documents/Dataset/exp253_guarded_standard_to_hard_expansion.lock")


def _state() -> dict[str, object]:
    state = json.loads(CANONICAL_STATE_PATH.read_text()) if CANONICAL_STATE_PATH.exists() else {}
    state.setdefault("completed", [])
    state.setdefault("history", [])
    state.setdefault("queue", [])
    state.setdefault("running", None)
    return state


def _write_state(payload: object) -> None:
    CANONICAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CANONICAL_STATE_PATH.with_suffix(".json.exp253.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, CANONICAL_STATE_PATH)


def _register_queued() -> None:
    state = _state()
    if EXPERIMENT_ID not in state["completed"] and EXPERIMENT_ID not in state["queue"]:
        state["queue"].append(EXPERIMENT_ID)
    _write_state(state)


def main() -> None:
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--dataset-limit", type=int)
    args_parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = args_parser.parse_args()
    QUEUE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Exp253 deferred runner is already active")
        _register_queued()
        while True:
            state = _state()
            if EXPERIMENT_ID in state["completed"]:
                return
            if not state.get("running"):
                break
            time.sleep(max(1.0, args.poll_seconds))
        command = [
            sys.executable, "-m",
            "experiments.exp253_guarded_standard_to_hard_expansion.run_pipeline",
            "--stage", "all", "--output-root", str(OUTPUT_ROOT),
        ]
        if args.dataset_limit:
            command.extend(["--dataset-limit", str(args.dataset_limit)])
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        log_path = OUTPUT_ROOT / "full.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            state = _state()
            state["queue"] = [value for value in state["queue"] if value != EXPERIMENT_ID]
            state["running"] = {
                "id": EXPERIMENT_ID,
                "label": "Exp253 - guarded Standard-to-Hard expansion",
                "pid": os.getpid(),
                "child_pid": process.pid,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "expected_datasets": args.dataset_limit or 1117,
                "stdout_log": str(log_path),
            }
            _write_state(state)
            started = time.time()
            returncode = process.wait()
        state = _state()
        state["running"] = None
        state["history"].append({
            "id": EXPERIMENT_ID,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "return_code": returncode,
            "elapsed_minutes": round((time.time() - started) / 60.0, 2),
        })
        if returncode == 0:
            if EXPERIMENT_ID not in state["completed"]:
                state["completed"].append(EXPERIMENT_ID)
        elif EXPERIMENT_ID not in state["queue"]:
            state["queue"].append(EXPERIMENT_ID)
        _write_state(state)
        if returncode:
            raise SystemExit(returncode)


if __name__ == "__main__":
    main()
