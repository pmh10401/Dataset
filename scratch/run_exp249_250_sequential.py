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

from experiments.exp249_250_official_mini_multirocket.cache_pipeline import source_root
from experiments.exp249_250_official_mini_multirocket.config import MODEL_CONFIGS, OUTPUT_ROOT


CANONICAL_STATE_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_sequential_state.json")
QUEUE_LOCK_PATH = Path("/Users/minho/Documents/Dataset/exp249_250_official_mini_multirocket.lock")


def experiment_id(experiment: int) -> str:
    return f"experiment_{experiment}_{MODEL_CONFIGS[experiment].source}"


def build_queue() -> list[dict[str, object]]:
    return [
        {
            "experiment": experiment,
            "source": MODEL_CONFIGS[experiment].source,
            "status": "pending",
            "workers": 7,
        }
        for experiment in (249, 250)
    ]


def _state() -> dict[str, object]:
    state = json.loads(CANONICAL_STATE_PATH.read_text()) if CANONICAL_STATE_PATH.exists() else {}
    state.setdefault("completed", [])
    state.setdefault("history", [])
    state.setdefault("queue", [])
    state.setdefault("running", None)
    return state


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".exp249_250.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _completed(root: Path, experiment: int) -> bool:
    base = source_root(root, experiment)
    return (base / "MANIFEST.json").exists() and (base / "summary.csv").exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--reset-cache", action="store_true")
    args = parser.parse_args()
    queue = build_queue()
    queue_status = args.output_root / "queue_status.json"
    QUEUE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Exp249-250 sequential queue is already running")
        canonical = _state()
        if canonical.get("running"):
            raise SystemExit(f"another experiment is recorded as running: {canonical['running']}")
        identifiers = [experiment_id(int(item["experiment"])) for item in queue]
        completed = set(canonical["completed"])
        canonical["queue"] = [
            value for value in canonical["queue"] if value not in identifiers
        ] + [value for value in identifiers if value not in completed]
        _write(CANONICAL_STATE_PATH, canonical)

        for item in queue:
            experiment = int(item["experiment"])
            identifier = experiment_id(experiment)
            if _completed(args.output_root, experiment):
                canonical = _state()
                if identifier not in canonical["completed"]:
                    canonical["completed"].append(identifier)
                canonical["queue"] = [value for value in canonical["queue"] if value != identifier]
                _write(CANONICAL_STATE_PATH, canonical)
                item["status"] = "completed"
                _write(queue_status, {"updated_at": time.time(), "queue": queue})
                continue

            item["status"] = "running"
            _write(queue_status, {"updated_at": time.time(), "queue": queue})
            command = [
                sys.executable,
                "-m",
                "experiments.exp249_250_official_mini_multirocket.run_pipeline",
                "--experiment",
                str(experiment),
                "--stage",
                "all",
                "--output-root",
                str(args.output_root),
                "--workers",
                str(item["workers"]),
            ]
            if args.dataset_limit:
                command.extend(["--dataset-limit", str(args.dataset_limit)])
            if args.reset_cache:
                command.append("--reset-cache")
            started = time.time()
            log_path = source_root(args.output_root, experiment) / "full.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                canonical = _state()
                canonical["running"] = {
                    "id": identifier,
                    "label": f"Exp{experiment} - {item['source'].replace('_', ' ')}",
                    "pid": os.getpid(),
                    "child_pid": process.pid,
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "expected_datasets": args.dataset_limit or 1117,
                    "stdout_log": str(log_path),
                }
                _write(CANONICAL_STATE_PATH, canonical)
                returncode = process.wait()
            canonical = _state()
            canonical["running"] = None
            canonical["history"].append(
                {
                    "id": identifier,
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "return_code": returncode,
                    "elapsed_minutes": round((time.time() - started) / 60.0, 2),
                }
            )
            if returncode == 0:
                if identifier not in canonical["completed"]:
                    canonical["completed"].append(identifier)
                canonical["queue"] = [value for value in canonical["queue"] if value != identifier]
            _write(CANONICAL_STATE_PATH, canonical)
            if returncode:
                item["status"] = "failed"
                item["returncode"] = returncode
                _write(queue_status, {"updated_at": time.time(), "queue": queue})
                raise SystemExit(returncode)
            item["status"] = "completed"
            _write(queue_status, {"updated_at": time.time(), "queue": queue})
        _write(queue_status, {"updated_at": time.time(), "status": "completed", "queue": queue})


if __name__ == "__main__":
    main()
