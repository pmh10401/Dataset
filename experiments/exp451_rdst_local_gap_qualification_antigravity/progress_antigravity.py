"""Self-written atomic JSON and tail-friendly text progress."""
from __future__ import annotations
import os, time
from dataclasses import dataclass
from pathlib import Path
from .artifacts_antigravity import _write_all_antigravity, canonical_json_bytes_antigravity, write_atomic_antigravity

PROGRESS_KEYS_ANTIGRAVITY = {"pid","dataset","family","train_rows","test_rows","series_length","seed","fold","stage","backend","completed","total","success","error","reused","recomputed","datasets_per_second","elapsed_seconds","eta_seconds","checkpoint","latest_error"}

def validate_progress_antigravity(payload: object) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != PROGRESS_KEYS_ANTIGRAVITY:
        raise ValueError("progress keys differ")
    for key in ("pid","train_rows","test_rows","series_length","completed","total","success","error","reused","recomputed"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError(f"progress {key} differs")
    for key in ("datasets_per_second","elapsed_seconds","eta_seconds"):
        if type(payload[key]) is not float or not __import__("math").isfinite(payload[key]) or payload[key] < 0.0:
            raise ValueError(f"progress {key} differs")
    if payload["seed"] is not None or payload["fold"] is not None or payload["checkpoint"] != "checkpoint_antigravity.json":
        raise ValueError("progress worker authority differs")
    if payload["total"] != 48 or payload["success"] != payload["completed"] or payload["completed"] != payload["reused"] + payload["recomputed"]:
        raise ValueError("progress counters differ")
    if type(payload["dataset"]) is not str or type(payload["family"]) is not str or type(payload["stage"]) is not str or payload["backend"] not in {"cpu","mlx","mps"}:
        raise ValueError("progress identity differs")
    if payload["latest_error"] is not None and type(payload["latest_error"]) is not str:
        raise ValueError("progress error differs")
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
        line = " ".join(f"{key}={payload[key]}" for key in sorted(payload)) + "\n"
        fd = os.open(self.root / "progress_antigravity.txt", os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try: _write_all_antigravity(fd, line.encode()); os.fsync(fd)
        finally: os.close(fd)

__all__ = ["PROGRESS_KEYS_ANTIGRAVITY", "ProgressWriterAntigravity", "select_outer_workers_antigravity", "validate_progress_antigravity"]
