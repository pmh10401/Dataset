from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP263_ID = "experiment_263_full_missing_aware_integration"
EXP264_ID = "experiment_264_full_missing_aware_posthoc_evaluation"
EXPERIMENT_IDS = (EXP263_ID, EXP264_ID)
PHASES = (
    ("audit", EXP263_ID),
    ("predict", EXP263_ID),
    ("evaluate", EXP264_ID),
    ("report", EXP264_ID),
)
TERMINAL_PHASE = {EXP263_ID: "predict", EXP264_ID: "report"}
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration")
STATE_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_sequential_state.json")


def default_state() -> dict[str, object]:
    return {"completed": [], "history": [], "queue": [], "running": None}


def load_state(path: Path = STATE_PATH) -> dict[str, object]:
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        state = {}
    if not isinstance(state, dict):
        raise ValueError("queue state is not an object")
    defaults = default_state()
    for key, value in defaults.items():
        state.setdefault(key, value)
    return state


def write_state(payload: object, path: Path = STATE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write((json.dumps(payload, sort_keys=True, indent=2) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def prepare_queue(state: dict[str, object]) -> dict[str, object]:
    prepared = dict(state)
    completed = list(prepared.get("completed") or [])
    history = list(prepared.get("history") or [])
    queue = [item for item in prepared.get("queue") or [] if item not in EXPERIMENT_IDS]
    for experiment_id in EXPERIMENT_IDS:
        if experiment_id not in completed:
            queue.append(experiment_id)
    prepared.update(
        {"completed": completed, "history": history, "queue": queue, "running": prepared.get("running")}
    )
    return prepared


def finish_phase(
    state: dict[str, object],
    *,
    experiment_id: str,
    phase: str,
    return_code: int,
    started_unix: float,
    finished_unix: float,
) -> dict[str, object]:
    finished = dict(state)
    finished["running"] = None
    finished["queue"] = [
        item for item in finished.get("queue") or [] if item != experiment_id
    ]
    history = list(finished.get("history") or [])
    history.append(
        {
            "id": experiment_id,
            "phase": phase,
            "return_code": int(return_code),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finished_unix)),
            "elapsed_minutes": round((finished_unix - started_unix) / 60.0, 2),
        }
    )
    finished["history"] = history
    completed = list(finished.get("completed") or [])
    if return_code == 0 and TERMINAL_PHASE[experiment_id] == phase:
        if experiment_id not in completed:
            completed.append(experiment_id)
    elif return_code == 0 and experiment_id not in completed:
        finished["queue"].append(experiment_id)
    if return_code != 0:
        finished["queue"] = [
            item for item in finished["queue"] if item not in EXPERIMENT_IDS
        ]
    finished["completed"] = completed
    return finished


def _wait_for_open_slot(state_path: Path, poll_seconds: float) -> None:
    while load_state(state_path).get("running"):
        time.sleep(max(0.1, poll_seconds))


def _command(args: argparse.Namespace, phase: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.exp263_264_full_missing_aware_integration.run_pipeline",
        "--phase",
        phase,
        "--output-root",
        str(args.output_root),
    ]
    if args.dataset_limit is not None:
        command.extend(["--dataset-limit", str(args.dataset_limit)])
    if phase == "evaluate" and args.expected_total_anomalies is not None:
        command.extend(
            ["--expected-total-anomalies", str(args.expected_total_anomalies)]
        )
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--expected-total-anomalies", type=int)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.state_path = args.state_path.expanduser().resolve()
    if args.dataset_limit is not None and args.expected_total_anomalies is None:
        raise SystemExit("smoke runner requires --expected-total-anomalies")
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / ".runner.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Exp263/264 runner is already active")
        write_state(prepare_queue(load_state(args.state_path)), args.state_path)
        for phase, experiment_id in PHASES:
            _wait_for_open_slot(args.state_path, args.poll_seconds)
            command = _command(args, phase)
            log_path = args.output_root / "full.log"
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                state = load_state(args.state_path)
                state["queue"] = [
                    item for item in state["queue"] if item != experiment_id
                ]
                state["running"] = {
                    "id": experiment_id,
                    "label": "Exp263/264 - full missing-aware integration",
                    "phase": phase,
                    "pid": os.getpid(),
                    "child_pid": process.pid,
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "expected_datasets": args.dataset_limit or 1117,
                    "status_json": str(args.output_root / "status.json"),
                    "detail_csv": str(args.output_root / "results.csv"),
                    "stdout_log": str(log_path),
                }
                write_state(state, args.state_path)
                started = time.time()
                return_code = process.wait()
            state = finish_phase(
                load_state(args.state_path),
                experiment_id=experiment_id,
                phase=phase,
                return_code=return_code,
                started_unix=started,
                finished_unix=time.time(),
            )
            write_state(state, args.state_path)
            if return_code:
                return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
