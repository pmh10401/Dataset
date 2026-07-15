from __future__ import annotations

import json
import os
import time
from pathlib import Path


DEFAULT_STATE_PATH = Path("/Users/minho/Documents/Dataset/rank_dashboard_external_live_run.json")


class ExperimentProgress:
    def __init__(
        self,
        *,
        experiment_id: str,
        label: str,
        expected_datasets: int,
        detail_csv: str,
        stdout_log: str,
        state_path: Path = DEFAULT_STATE_PATH,
        scope: str = "retrospective research",
        autonomous_metric: str = "not_evaluated",
        review_metric: str = "not_evaluated",
    ) -> None:
        self.experiment_id = experiment_id
        self.label = label
        self.expected_datasets = int(expected_datasets)
        self.detail_csv = str(detail_csv)
        self.stdout_log = str(stdout_log)
        self.state_path = Path(state_path)
        self.scope = scope
        self.autonomous_metric = autonomous_metric
        self.review_metric = review_metric
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._state: dict = {}

    def _write(self, update: dict) -> None:
        self._state.update(update)
        self._state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._state["updated_unix"] = time.time()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)

    def start(self, *, phase: str) -> None:
        self._state = {
            "status": "running",
            "id": self.experiment_id,
            "label": self.label,
            "pid": os.getpid(),
            "child_pid": os.getpid(),
            "expected_datasets": self.expected_datasets,
            "datasets_done": 0,
            "progress_percent": 0.0,
            "rows": 0,
            "started_at": self.started_at,
            "current_dataset": None,
            "errors": 0,
            "phase": phase,
            "detail_csv": self.detail_csv,
            "stdout_log": self.stdout_log,
            "scope": self.scope,
            "autonomous_metric": self.autonomous_metric,
            "review_metric": self.review_metric,
            "error": None,
        }
        self._write({})

    def update(
        self,
        *,
        done: int,
        current_dataset: str | None,
        rows: int,
        errors: int,
        phase: str,
    ) -> None:
        done = int(done)
        self._write(
            {
                "status": "running",
                "datasets_done": done,
                "progress_percent": done / self.expected_datasets * 100 if self.expected_datasets else 0.0,
                "current_dataset": current_dataset,
                "rows": int(rows),
                "errors": int(errors),
                "phase": phase,
            }
        )

    def finish(self, *, rows: int, result_summary: dict) -> None:
        self._write(
            {
                "status": "completed",
                "datasets_done": self.expected_datasets,
                "progress_percent": 100.0,
                "current_dataset": None,
                "rows": int(rows),
                "errors": int(result_summary.get("errors", 0)),
                "phase": "completed",
                "result_summary": result_summary,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def fail(self, *, error: str, done: int, rows: int, errors: int) -> None:
        self._write(
            {
                "status": "failed",
                "datasets_done": int(done),
                "progress_percent": int(done) / self.expected_datasets * 100 if self.expected_datasets else 0.0,
                "rows": int(rows),
                "errors": int(errors),
                "phase": "failed",
                "error": str(error),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
