#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import serve_rank_dashboard as dashboard

    experiment_id = "experiment_403_408_feature_ml_validation"
    info = dashboard.EXPERIMENTS[experiment_id]
    assert info["status_json"].name == "status.json"

    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "status.json"
        status_path.write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "status": "running",
                    "updated_unix": 1000.0,
                    "stage": "ssl-predict",
                    "current_dataset": "Dataset-7",
                    "completed": 7,
                    "total": 1117,
                    "progress_percent": 0.6267,
                    "eta_seconds": 120.0,
                    "throughput_per_minute": 3.5,
                    "errors": 1,
                    "device": "cpu",
                    "checkpoint_sha256": "a" * 64,
                }
            ),
            encoding="utf-8",
        )
        live = dashboard.read_experiment_status(
            experiment_id, {"status_json": status_path}, now=1005.0
        )

    assert live["stage"] == "ssl-predict"
    assert live["current_dataset"] == "Dataset-7"
    assert live["checkpoint_sha256"] == "a" * 64
    assert "jsonl" not in dashboard.read_experiment_status.__code__.co_names

    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "status.json"
        status_path.write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "status": "running",
                    "updated_unix": time.time(),
                    "completed": 7,
                    "total": 1117,
                    "progress_percent": 0.6267,
                }
            ),
            encoding="utf-8",
        )
        original_read_csv_rows = dashboard.read_csv_rows
        original_experiment = dashboard.EXPERIMENTS[experiment_id]
        try:
            dashboard.EXPERIMENTS[experiment_id] = {
                **original_experiment,
                "status_json": status_path,
                "live_status_only": True,
            }
            dashboard.read_csv_rows = lambda _path: (_ for _ in ()).throw(
                AssertionError("live status must not scan result artifacts")
            )
            rows = dashboard.experiment_queue_snapshot(
                {"completed": [], "queue": [experiment_id], "history": [], "running": {"id": experiment_id}},
                1117,
                experiment_ids=[experiment_id],
            )
        finally:
            dashboard.read_csv_rows = original_read_csv_rows
            dashboard.EXPERIMENTS[experiment_id] = original_experiment

    assert rows[0]["datasets_done"] == 7
    assert rows[0]["progress_percent"] == 0.6267
    print("Exp403-408 dashboard status contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
