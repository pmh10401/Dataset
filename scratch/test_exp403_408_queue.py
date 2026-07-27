#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "run_rank_experiments_sequential.py"
    spec = importlib.util.spec_from_file_location("rank_runner_exp403_408_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    runner = load_runner()
    experiment = runner.get_experiment("experiment_403_408_feature_ml_validation")
    assert experiment["expected_datasets"] == 1117
    assert experiment["archive_outputs"] is False
    assert experiment["automatic_policy_changed"] is False
    assert experiment["review_only"] is True
    assert experiment["status_json"].name == "status.json"
    assert experiment["script"].name == "run_exp403_408_feature_ml_validation.py"
    assert experiment["args"][:2] == ["--stage", "all"]

    env = runner.build_child_environment({}, worker_count="7")
    assert env["RANK_EXPERIMENT_WORKERS"] == "7"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT)

    saved = {}
    runner.load_state = lambda: {"completed": [], "queue": [], "history": [], "running": None}
    runner.reconcile_running_state = lambda state: state
    runner.save_state = lambda state: saved.update(state)
    runner.log = lambda _message: None
    runner.queue_exp403_408_batch()
    assert saved["queue"] == ["experiment_403_408_feature_ml_validation"]
    assert saved["exp403_408_batch"]["execution_mode"] == "sequential_one_experiment_at_a_time"
    assert saved["exp403_408_batch"]["classical_outer_workers"] == 7
    assert saved["exp403_408_batch"]["learned_model_processes"] == 1

    ps_output = "PID PPID STAT ELAPSED COMMAND\n900 1 S 00:01 /opt/homebrew/bin/python3 " + (
        f"{ROOT / 'scratch' / 'run_exp403_408_feature_ml_validation.py'} --stage all\n"
    )
    assert runner.active_rank_processes(ps_output=ps_output)

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scratch" / "run_exp403_408_feature_ml_validation.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "audit-features" in completed.stdout

    dashboard_source = (ROOT / "serve_rank_dashboard.py").read_text(encoding="utf-8")
    assert "experiment_403_408_feature_ml_validation" in dashboard_source
    assert "exp403_408_feature_ml_validation" in dashboard_source
    print("Exp403-408 queue and dashboard registration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
