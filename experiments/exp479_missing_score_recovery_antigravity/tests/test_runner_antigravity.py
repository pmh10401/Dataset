"""Task 5 tests for resumable Exp479 execution."""

from __future__ import annotations

import importlib
import importlib.util
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys


def _runner_antigravity():
    return importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity.run_antigravity"
    )


class _CheckboxParserAntigravity(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.checkboxes: list[dict[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if tag == "input" and values.get("type") == "checkbox":
            self.checkboxes.append(values)


def test_runner_module_exposes_the_task5_api_antigravity() -> None:
    """Deleting the runner or a required execution boundary must fail."""
    module_name = (
        "experiments.exp479_missing_score_recovery_antigravity."
        "run_antigravity"
    )
    specification = importlib.util.find_spec(module_name)
    assert specification is not None, "Exp479 runner module is missing"
    module = importlib.import_module(module_name)
    assert hasattr(module, "write_json_atomic_antigravity")
    assert hasattr(module, "resume_rows_antigravity")
    assert hasattr(module, "evaluate_dataset_antigravity")
    assert hasattr(module, "run_smoke_antigravity")


def test_runner_module_exposes_the_approved_task6_api_antigravity() -> None:
    """The approved official execution and its aggregation boundary must exist."""
    module = _runner_antigravity()
    assert hasattr(module, "summarise_performance_antigravity")
    assert hasattr(module, "render_official_dashboard_antigravity")
    assert hasattr(module, "run_official_antigravity")


def test_task6_summary_uses_exact_rows_instead_of_rounded_cohort_means_antigravity() -> None:
    """Combined metrics must be calculated from dataset rows, not rounded means."""
    module = _runner_antigravity()
    rows = [
        {
            "dataset": "A_normal_1",
            "method": "minirocket_local_gap",
            "auc": 0.80,
            "zf": 1,
            "baseline_auc": 0.80,
            "baseline_zf": 1,
            "oracle_auc": 0.90,
        },
        {
            "dataset": "B_normal_1",
            "method": "hydra_local_gap",
            "auc": 1.00,
            "zf": 3,
            "baseline_auc": 0.70,
            "baseline_zf": 0,
            "oracle_auc": 1.00,
        },
        {
            "dataset": "C_normal_1",
            "method": "multirocket_local_gap",
            "auc": 0.95,
            "zf": 2,
            "baseline_auc": 0.90,
            "baseline_zf": 1,
            "oracle_auc": 0.97,
        },
    ]
    summary = module.summarise_performance_antigravity(rows)
    assert summary["dataset_count"] == 3
    assert summary["average_auc"] == (0.80 + 1.00 + 0.95) / 3
    assert summary["average_zf_tp"] == 2.0
    assert summary["baseline_average_auc"] == (0.80 + 0.70 + 0.90) / 3
    assert summary["baseline_average_zf_tp"] == 2 / 3
    assert summary["average_oracle_auc_loss"] == (
        (0.90 - 0.80) + (1.00 - 1.00) + (0.97 - 0.95)
    ) / 3
    assert summary["severe_auc_loss_over_0_05"] == 1
    assert summary["very_severe_auc_loss_over_0_10"] == 0
    assert summary["method_counts"] == {
        "hydra_local_gap": 1,
        "minirocket_local_gap": 1,
        "multirocket_local_gap": 1,
    }


def test_task6_dashboard_checks_completion_only_after_all_155_succeed_antigravity() -> None:
    """A partial official run must stay visibly incomplete in the HTML dashboard."""
    module = _runner_antigravity()
    running = module.render_official_dashboard_antigravity(
        {
            "state": "running",
            "completed": 7,
            "success": 7,
            "failures": 0,
            "latest_dataset": "Demo_normal_1",
            "elapsed_seconds": 12.5,
        }
    )
    complete = module.render_official_dashboard_antigravity(
        {
            "state": "complete",
            "completed": 155,
            "success": 155,
            "failures": 0,
            "latest_dataset": "Last_normal_1",
            "elapsed_seconds": 99.0,
            "recovery_summary": {
                "average_auc": 0.9,
                "average_zf_tp": 1.2,
                "baseline_average_auc": 0.89,
                "baseline_average_zf_tp": 1.1,
                "severe_auc_loss_over_0_05": 4,
            },
            "combined_summary": {
                "average_auc": 0.98,
                "average_zf_tp": 2.0,
                "baseline_average_auc": 0.97,
                "baseline_average_zf_tp": 1.9,
                "severe_auc_loss_over_0_05": 45,
            },
        }
    )
    running_parser = _CheckboxParserAntigravity()
    running_parser.feed(running)
    complete_parser = _CheckboxParserAntigravity()
    complete_parser.feed(complete)
    assert "7 / 155" in running
    assert "공식 실행 중" in running
    assert sum("checked" in item for item in running_parser.checkboxes) == 9
    assert "155 / 155" in complete
    assert "공식 실행 완료" in complete
    assert sum("checked" in item for item in complete_parser.checkboxes) == 10


def test_atomic_json_write_leaves_only_the_complete_target_antigravity(tmp_path: Path) -> None:
    """Writing invalid JSON or leaving the temporary file behind must fail."""
    module = _runner_antigravity()
    target = tmp_path / "result_antigravity.json"
    payload = {"schema": "demo", "value": 7}
    module.write_json_atomic_antigravity(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not tuple(tmp_path.glob("*temporary_antigravity*"))


def test_resume_rejects_changed_sources_population_or_execution_mode_antigravity() -> None:
    """Reusing stale smoke rows in another source state or official mode must fail."""
    module = _runner_antigravity()
    checkpoint = {
        "schema": "exp479_checkpoint_v1",
        "mode": "smoke",
        "source_fingerprint": "source-a",
        "population_sha256": "population-a",
        "rows": [
            {"dataset": "A_normal_0", "status": "ok"},
            {"dataset": "Outside_normal_0", "status": "ok"},
        ],
    }
    expected = {"A_normal_0": {"dataset": "A_normal_0", "status": "ok"}}
    assert module.resume_rows_antigravity(
        checkpoint,
        source_fingerprint="source-a",
        population_sha256="population-a",
        mode="smoke",
        population=("A_normal_0", "B_normal_0"),
    ) == expected
    assert module.resume_rows_antigravity(
        checkpoint,
        source_fingerprint="changed",
        population_sha256="population-a",
        mode="smoke",
        population=("A_normal_0",),
    ) == {}
    assert module.resume_rows_antigravity(
        checkpoint,
        source_fingerprint="source-a",
        population_sha256="population-a",
        mode="official",
        population=("A_normal_0",),
    ) == {}


def test_direct_cli_validate_only_finds_the_repository_packages_antigravity() -> None:
    """Removing repository-root path setup from the direct script must fail."""
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scratch/run_exp479_missing_score_recovery_antigravity.py"),
            "--validate-only",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"recovery": 155, "status": "ready"}


def test_completed_smoke_and_dashboard_preserve_the_task6_state_antigravity() -> None:
    """The dashboard must distinguish smoke, running, and completed official states."""
    root = Path(__file__).resolve().parents[3]
    result = json.loads(
        (
            root
            / "experiments/exp479_missing_score_recovery_antigravity/"
            "smoke_results_antigravity.json"
        ).read_text(encoding="utf-8")
    )
    assert result["requested"] == result["completed"] == result["success"] == 2
    assert result["failures"] == 0
    assert all(tuple(row["scores"]) == (
        "minirocket_local_gap",
        "multirocket_local_gap",
        "hydra_local_gap",
        "hydra_multirocket_local_gap",
    ) for row in result["rows"])
    dashboard = (root / "docs/EXP479_DASHBOARD_antigravity.html").read_text(
        encoding="utf-8"
    )
    parser = _CheckboxParserAntigravity()
    parser.feed(dashboard)
    official_path = (
        root
        / "experiments/exp479_missing_score_recovery_antigravity/"
        "results_antigravity.json"
    )
    if official_path.exists() and json.loads(
        official_path.read_text(encoding="utf-8")
    ).get("state") == "complete":
        assert "공식 실행 완료" in dashboard
        assert sum("checked" in item for item in parser.checkboxes) == 10
    else:
        assert "smoke 2/2 성공" in dashboard or "공식 실행 중" in dashboard
        assert sum("checked" in item for item in parser.checkboxes) == 9
