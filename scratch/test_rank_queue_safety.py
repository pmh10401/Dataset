import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_rank_experiments_sequential as queue


def test_reconcile_running_state_clears_dead_pid_without_touching_queue():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        old_state_path = queue.STATE_PATH
        old_log_path = queue.LOG_PATH
        old_get_experiment = queue.get_experiment
        try:
            queue.STATE_PATH = tmp_path / "state.json"
            queue.LOG_PATH = tmp_path / "queue.log"
            queue.get_experiment = lambda _exp_id: (_ for _ in ()).throw(KeyError(_exp_id))
            state = {
                "completed": [],
                "queue": ["experiment_44_classical_embedding_baselines"],
                "history": [],
                "running": {
                    "id": "experiment_44_classical_embedding_baselines",
                    "pid": 99999998,
                    "child_pid": 99999999,
                    "started_at": "2026-07-07 00:00:00",
                },
            }
            queue.save_state(state)

            reconciled = queue.reconcile_running_state(queue.load_state())

            assert reconciled["running"] is None
            assert reconciled["queue"] == ["experiment_44_classical_embedding_baselines"]
            assert "STALE running state cleared" in queue.LOG_PATH.read_text()
        finally:
            queue.STATE_PATH = old_state_path
            queue.LOG_PATH = old_log_path
            queue.get_experiment = old_get_experiment


def test_reconcile_running_state_keeps_live_pid():
    state = {
        "completed": [],
        "queue": [],
        "history": [],
        "running": {"id": "x", "pid": os.getpid(), "started_at": "2026-07-07 00:00:00"},
    }

    reconciled = queue.reconcile_running_state(dict(state))

    assert reconciled["running"] == state["running"]


def test_reconcile_running_state_recovers_complete_expected_dataset_output():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        detail = tmp_path / "detail.csv"
        summary = tmp_path / "summary.csv"
        log_path = tmp_path / "run.log"
        detail.write_text("dataset_name,value\nA,1\nB,2\n", encoding="utf-8")
        summary.write_text("source_or_variant,rate\na,0.1\n", encoding="utf-8")
        log_path.write_text("completed artifacts\n", encoding="utf-8")
        old_state_path = queue.STATE_PATH
        old_log_path = queue.LOG_PATH
        old_get_experiment = queue.get_experiment
        try:
            queue.STATE_PATH = tmp_path / "state.json"
            queue.LOG_PATH = tmp_path / "queue.log"
            queue.get_experiment = lambda _exp_id: {
                "id": "complete",
                "detail_csv": detail,
                "summary_csv": summary,
                "stdout_log": log_path,
                "expected_datasets": 2,
            }
            state = {
                "completed": [],
                "queue": ["complete"],
                "history": [],
                "running": {"id": "complete", "pid": 99999998, "child_pid": 99999999},
            }

            reconciled = queue.reconcile_running_state(state)

            assert reconciled["running"] is None
            assert reconciled["queue"] == []
            assert reconciled["completed"] == ["complete"]
            assert reconciled["history"][-1]["recovered_state"] is True
        finally:
            queue.STATE_PATH = old_state_path
            queue.LOG_PATH = old_log_path
            queue.get_experiment = old_get_experiment


def test_process_inspection_failure_defers_queue_work():
    original_active = queue.active_rank_processes
    try:
        def unavailable():
            raise BlockingIOError(35, "Resource temporarily unavailable")

        queue.active_rank_processes = unavailable
        assert queue.active_rank_processes_or_none() is None
    finally:
        queue.active_rank_processes = original_active


def test_queue_creates_output_directories_before_opening_logs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "new" / "experiment"
        exp = {
            "detail_csv": root / "detail.csv",
            "summary_csv": root / "summary.csv",
            "stdout_log": root / "full_run.log",
        }

        queue.ensure_output_directories(exp)

        assert root.is_dir()
        with exp["stdout_log"].open("w", encoding="utf-8") as handle:
            handle.write("ready\n")


if __name__ == "__main__":
    test_reconcile_running_state_clears_dead_pid_without_touching_queue()
    test_reconcile_running_state_keeps_live_pid()
    test_reconcile_running_state_recovers_complete_expected_dataset_output()
    test_process_inspection_failure_defers_queue_work()
    test_queue_creates_output_directories_before_opening_logs()
    print("rank queue safety tests passed")
