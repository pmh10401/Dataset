"""Self-written atomic JSON and tail-friendly text progress."""
from __future__ import annotations
import os, time
from dataclasses import dataclass
from pathlib import Path
from .artifacts_antigravity import append_text_antigravity, canonical_json_bytes_antigravity, write_atomic_antigravity

PROGRESS_KEYS_ANTIGRAVITY = {"pid","dataset","family","train_rows","test_rows","series_length","seed","fold","stage","backend","completed","total","success","error","reused","recomputed","datasets_per_second","elapsed_seconds","eta_seconds","checkpoint","latest_error"}
PROGRESS_STAGES_ANTIGRAVITY = {"starting", "active", "interrupted", "complete", "failed", "setup_failed"}

def render_progress_line_antigravity(payload: dict[str, object]) -> bytes:
    return (" ".join(f"{key}={payload[key]}" for key in sorted(payload)) + "\n").encode()

def validate_progress_antigravity(payload: object) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != PROGRESS_KEYS_ANTIGRAVITY:
        raise ValueError("progress keys differ")
    for key in ("pid","train_rows","test_rows","series_length","completed","total","success","error","reused","recomputed"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError(f"progress {key} differs")
    stage = payload["stage"]
    if stage not in PROGRESS_STAGES_ANTIGRAVITY:
        raise ValueError("progress stage differs")
    terminal = stage in {"complete", "failed", "setup_failed"}
    if payload["pid"] < 1:
        raise ValueError("progress PID differs")
    if terminal:
        if payload["dataset"] is not None or payload["family"] is not None or any(payload[key] != 0 for key in ("train_rows", "test_rows", "series_length")):
            raise ValueError("terminal progress dimensions differ")
    elif any(type(payload[key]) is not str or not payload[key] for key in ("dataset", "family")) or min(payload["train_rows"], payload["test_rows"], payload["series_length"]) < 1:
        raise ValueError("active progress dimensions differ")
    for key in ("datasets_per_second","elapsed_seconds","eta_seconds"):
        if type(payload[key]) is not float or not __import__("math").isfinite(payload[key]) or payload[key] < 0.0:
            raise ValueError(f"progress {key} differs")
    if payload["seed"] is not None or payload["fold"] is not None or payload["checkpoint"] != "checkpoint_antigravity.json":
        raise ValueError("progress worker authority differs")
    if payload["total"] != 48 or payload["success"] != payload["completed"] or payload["completed"] != payload["reused"] + payload["recomputed"]:
        raise ValueError("progress counters differ")
    if payload["backend"] not in {"cpu","mlx","mps"}:
        raise ValueError("progress identity differs")
    if payload["latest_error"] is not None and type(payload["latest_error"]) is not str:
        raise ValueError("progress error differs")
    if stage == "complete" and (payload["completed"] != 48 or payload["error"] != 0 or payload["latest_error"] is not None or payload["eta_seconds"] != 0.0):
        raise ValueError("complete progress differs")
    if stage in {"failed", "setup_failed"} and (payload["error"] != 1 or not payload["latest_error"]):
        raise ValueError("failed progress differs")
    if stage == "interrupted" and (payload["error"] != 1 or not payload["latest_error"]):
        raise ValueError("interrupted progress differs")
    if stage not in {"failed", "setup_failed", "interrupted"} and (payload["error"] != 0 or payload["latest_error"] is not None):
        raise ValueError("nonerror progress differs")
    return payload

def select_outer_workers_antigravity(logical_cpus: int, backend: str) -> int:
    if type(logical_cpus) is not int or logical_cpus < 1 or backend not in {"cpu", "mlx", "mps"}: raise ValueError("invalid scheduler inputs")
    return max(1, logical_cpus - 1) if backend == "cpu" else 1

@dataclass(slots=True)
class ProgressWriterAntigravity:
    root: Path
    fault_hook: object = None
    def write_antigravity(self, payload: dict[str, object], sequence: int = 0) -> None:
        validate_progress_antigravity(payload)
        write_atomic_antigravity(self.root / "progress_antigravity.json", canonical_json_bytes_antigravity(payload), fault_hook=self.fault_hook, sequence=sequence)
        if self.fault_hook: self.fault_hook("progress_text", sequence)
        line = render_progress_line_antigravity(payload)
        text_path = self.root / "progress_antigravity.txt"
        try:
            append_text_antigravity(text_path, line)
        except FileNotFoundError:
            from .artifacts_antigravity import write_create_only_antigravity
            write_create_only_antigravity(text_path, b"")
            append_text_antigravity(text_path, line)

__all__ = ["PROGRESS_KEYS_ANTIGRAVITY", "PROGRESS_STAGES_ANTIGRAVITY", "ProgressWriterAntigravity", "render_progress_line_antigravity", "select_outer_workers_antigravity", "validate_progress_antigravity"]
