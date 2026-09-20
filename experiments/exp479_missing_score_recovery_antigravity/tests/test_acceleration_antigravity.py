"""Task 4 tests for the Exp479 acceleration decision."""

from __future__ import annotations

import importlib
import importlib.util
from html.parser import HTMLParser
import json
from pathlib import Path


def test_acceleration_module_exposes_the_task4_api_antigravity() -> None:
    """Deleting the acceleration assessment or its public API must fail."""
    module_name = (
        "experiments.exp479_missing_score_recovery_antigravity."
        "acceleration_antigravity"
    )
    specification = importlib.util.find_spec(module_name)
    assert specification is not None, "Exp479 acceleration module is missing"
    module = importlib.import_module(module_name)
    assert hasattr(module, "assess_mlx_acceleration_antigravity")


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


def test_missing_mlx_selects_numba_and_records_the_exact_reason_antigravity() -> None:
    """Pretending MLX passed when unavailable or changing the Numba default must fail."""
    assert importlib.util.find_spec("mlx") is None
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "acceleration_antigravity"
    )
    assessment = module.assess_mlx_acceleration_antigravity()
    assert assessment["schema"] == "exp479_mlx_assessment_v1"
    assert assessment["stage"] == "ragged_linear_resampling"
    assert assessment["mlx_available"] is False
    assert assessment["eligible"] is False
    assert assessment["parity"] == "not_run"
    assert assessment["speedup"] is None
    assert assessment["selected_backend"] == "numba_cpu"
    assert assessment["fallback_reason"] == "mlx_not_installed_in_fixed_interpreter"
    assert assessment["versions"]["numba"]
    assert assessment["versions"]["numpy"]
    assert assessment["versions"]["mlx"] is None
    assert assessment["versions"]["mlx_metal"] is None


def test_task4_artifact_and_dashboard_match_the_numba_decision_antigravity() -> None:
    """A stale acceleration artifact or unchecked completed Task 4 must fail."""
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity."
        "acceleration_antigravity"
    )
    assessment = module.assess_mlx_acceleration_antigravity()
    root = Path(__file__).resolve().parents[3]
    artifact = root / "scratch/exp479_mlx_assessment_antigravity.json"
    dashboard = root / "docs/EXP479_DASHBOARD_antigravity.html"
    assert json.loads(artifact.read_text(encoding="utf-8")) == assessment
    html = dashboard.read_text(encoding="utf-8")
    parser = _CheckboxParserAntigravity()
    parser.feed(html)
    assert "MLX 미설치" in html and "Numba CPU 유지" in html
    assert all("checked" in item for item in parser.checkboxes[:8])
    result_path = (
        root
        / "experiments/exp479_missing_score_recovery_antigravity/results_antigravity.json"
    )
    complete = result_path.exists() and json.loads(
        result_path.read_text(encoding="utf-8")
    ).get("state") == "complete"
    assert ("checked" in parser.checkboxes[-1]) is complete
