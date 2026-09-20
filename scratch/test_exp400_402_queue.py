#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "run_rank_experiments_sequential.py"
    spec = importlib.util.spec_from_file_location("rank_runner_exp400_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    runner = load_runner()
    experiment = runner.get_experiment(
        "experiment_400_402_exp397_shadow_validation"
    )
    assert experiment["expected_datasets"] == 1117
    assert experiment["archive_outputs"] is False
    assert experiment["automatic_policy_changed"] is False
    assert experiment["review_only"] is True
    assert experiment["status_json"].name == "status.json"
    assert experiment["script"].name == "run_exp400_402_shadow_validation.py"

    dashboard_source = (ROOT / "serve_rank_dashboard.py").read_text(
        encoding="utf-8"
    )
    assert "experiment_400_402_exp397_shadow_validation" in dashboard_source
    assert "retrospective_shadow_replay" in dashboard_source
    print("Exp400-402 queue and dashboard registration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
