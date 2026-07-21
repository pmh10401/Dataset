from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp266_fn_recovery_shadow.config import EXPECTED_DATASETS, EXPERIMENT_ID, FULL_RUN_WORKERS, OUTPUT_ROOT
from experiments.exp266_fn_recovery_shadow.run_pipeline import completed_output_is_valid, write_status

STATE_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_sequential_state.json")
GLOBAL_STATE_LOCK_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_sequential.lock")


def _read_state(path: Path) -> dict[str, object]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        state = {}
    if not isinstance(state, dict):
        raise ValueError("queue state is not an object")
    for key, value in {"completed": [], "history": [], "queue": [], "running": None}.items():
        state.setdefault(key, value)
    return state


def _write_state(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write((json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _queued(state: dict[str, object]) -> dict[str, object]:
    updated = dict(state)
    queue = [item for item in updated.get("queue") or [] if item != EXPERIMENT_ID]
    if EXPERIMENT_ID not in (updated.get("completed") or []):
        queue.append(EXPERIMENT_ID)
    updated["queue"] = queue
    return updated


def _claim(state: dict[str, object], owner_token: str, output_root: Path, expected: int) -> dict[str, object] | None:
    if state.get("running"):
        return None
    claimed = _queued(state)
    claimed["queue"] = [item for item in claimed["queue"] if item != EXPERIMENT_ID]
    claimed["running"] = {
        "id": EXPERIMENT_ID,
        "label": "Exp266 - lane 밖 FN 복구 Shadow",
        "phase": "all",
        "owner_token": owner_token,
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_datasets": expected,
        "status_json": str(output_root / "status.json"),
        "detail_csv": str(output_root / "05_dataset_results.csv"),
        "stdout_log": str(output_root / "full.log"),
    }
    return claimed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue the resumable Exp266 full run")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    state_path = args.state_path.expanduser().resolve()
    expected = args.dataset_limit or EXPECTED_DATASETS
    if args.dataset_limit is not None and not 0 < args.dataset_limit < EXPECTED_DATASETS:
        raise SystemExit("dataset-limit must be a strict positive subset of 1,117")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status_path = output_root / "status.json"
    log_path = output_root / "full.log"
    write_status(status_path, status="queued", stage="all", datasets_done=0, expected_datasets=expected,
                 current_dataset=None, workers=FULL_RUN_WORKERS, output_root=output_root, stdout_log=log_path,
                 started_unix=started, warnings=[], errors=[])

    with (output_root / ".runner.lock").open("w", encoding="utf-8") as runner_lock:
        try:
            fcntl.flock(runner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Exp266 runner is already active")
        with _locked(GLOBAL_STATE_LOCK_PATH):
            _write_state(_queued(_read_state(state_path)), state_path)
        owner_token = uuid.uuid4().hex
        while True:
            with _locked(GLOBAL_STATE_LOCK_PATH):
                claimed = _claim(_read_state(state_path), owner_token, output_root, expected)
                if claimed is not None:
                    _write_state(claimed, state_path)
                    break
            time.sleep(max(0.1, args.poll_seconds))

        command = [sys.executable, "-m", "experiments.exp266_fn_recovery_shadow.run_pipeline", "--stage", "all",
                   "--workers", str(FULL_RUN_WORKERS), "--output-root", str(output_root)]
        if args.dataset_limit is not None:
            command.extend(["--dataset-limit", str(args.dataset_limit)])
        child: subprocess.Popen[bytes] | None = None

        def forward_sigterm(_signum, _frame):
            if child is not None and child.poll() is None:
                child.terminate()

        previous_handler = signal.signal(signal.SIGTERM, forward_sigterm)
        try:
            with log_path.open("a", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
                with _locked(GLOBAL_STATE_LOCK_PATH):
                    state = _read_state(state_path)
                    running = dict(state.get("running") or {})
                    if running.get("owner_token") != owner_token:
                        child.terminate()
                        child.wait()
                        return 1
                    running["child_pid"] = child.pid
                    state["running"] = running
                    _write_state(state, state_path)
                return_code = child.wait()
        finally:
            signal.signal(signal.SIGTERM, previous_handler)

        succeeded = return_code == 0 and completed_output_is_valid(output_root, args.dataset_limit)
        if not succeeded:
            write_status(status_path, status="failed", stage="all", datasets_done=0, expected_datasets=expected,
                         current_dataset=None, workers=FULL_RUN_WORKERS, output_root=output_root,
                         stdout_log=log_path, started_unix=started, warnings=[],
                         errors=["pipeline exited without complete outputs"])
        with _locked(GLOBAL_STATE_LOCK_PATH):
            state = _read_state(state_path)
            running = dict(state.get("running") or {})
            if running.get("owner_token") == owner_token:
                state["running"] = None
                state["queue"] = [item for item in state.get("queue") or [] if item != EXPERIMENT_ID]
                state["history"] = list(state.get("history") or []) + [{
                    "id": EXPERIMENT_ID,
                    "return_code": 0 if succeeded else return_code or 1,
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_minutes": round((time.time() - started) / 60.0, 2),
                }]
                if succeeded and EXPERIMENT_ID not in state.get("completed", []):
                    state["completed"] = list(state.get("completed") or []) + [EXPERIMENT_ID]
                _write_state(state, state_path)
        return 0 if succeeded else return_code or 1


if __name__ == "__main__":
    raise SystemExit(main())
