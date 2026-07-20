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
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from experiments.exp254_sktime_structural_features.config import EXPERIMENT_ID, OUTPUT_ROOT, VENV_PYTHON

STATE_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_sequential_state.json")
LOCK_PATH = Path("/Users/minho/Documents/Dataset/exp254_sktime_structural_features.lock")


def state() -> dict[str, object]:
    value = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    value.setdefault("completed", []); value.setdefault("history", []); value.setdefault("queue", []); value.setdefault("running", None)
    return value


def write_state(value: object) -> None:
    temporary = STATE_PATH.with_suffix(".json.exp254.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n"); os.replace(temporary, STATE_PATH)


def main() -> None:
    arg_parser = argparse.ArgumentParser(); arg_parser.add_argument("--dataset-limit", type=int); arg_parser.add_argument("--reset-cache", action="store_true")
    args = arg_parser.parse_args(); LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit("Exp254 runner already active")
        current = state()
        if EXPERIMENT_ID not in current["completed"] and EXPERIMENT_ID not in current["queue"]: current["queue"].append(EXPERIMENT_ID)
        write_state(current)
        while state().get("running"): time.sleep(10)
        command = [str(VENV_PYTHON), "-m", "experiments.exp254_sktime_structural_features.run_pipeline", "--stage", "all", "--workers", "7", "--output-root", str(OUTPUT_ROOT)]
        if args.dataset_limit: command.extend(["--dataset-limit", str(args.dataset_limit)])
        if args.reset_cache: command.append("--reset-cache")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True); log_path = OUTPUT_ROOT / "full.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            current = state(); current["queue"] = [x for x in current["queue"] if x != EXPERIMENT_ID]
            current["running"] = {"id": EXPERIMENT_ID, "label": "Exp254 - sktime structural features", "pid": os.getpid(), "child_pid": process.pid, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "expected_datasets": args.dataset_limit or 1117, "stdout_log": str(log_path)}
            write_state(current); started = time.time(); code = process.wait()
        current = state(); current["running"] = None
        current["history"].append({"id": EXPERIMENT_ID, "return_code": code, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_minutes": round((time.time()-started)/60, 2)})
        if code == 0:
            if EXPERIMENT_ID not in current["completed"]: current["completed"].append(EXPERIMENT_ID)
        elif EXPERIMENT_ID not in current["queue"]: current["queue"].append(EXPERIMENT_ID)
        write_state(current)
        if code: raise SystemExit(code)


if __name__ == "__main__": main()
