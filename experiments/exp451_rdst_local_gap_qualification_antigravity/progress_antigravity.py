"""Self-written atomic JSON and tail-friendly text progress."""
from __future__ import annotations
import os, time
from dataclasses import dataclass
from pathlib import Path
from .artifacts_antigravity import _write_all_antigravity, canonical_json_bytes_antigravity, write_atomic_antigravity

PROGRESS_KEYS_ANTIGRAVITY = {"pid","dataset","family","train_rows","test_rows","series_length","seed","fold","stage","backend","completed","total","success","error","reused","datasets_per_second","elapsed_seconds","eta_seconds","checkpoint","latest_error"}

def select_outer_workers_antigravity(logical_cpus: int, backend: str) -> int:
    if type(logical_cpus) is not int or logical_cpus < 1 or backend not in {"cpu", "mlx", "mps"}: raise ValueError("invalid scheduler inputs")
    return max(1, logical_cpus - 1) if backend == "cpu" else 1

@dataclass(slots=True)
class ProgressWriterAntigravity:
    root: Path
    fault_hook: object = None
    def write_antigravity(self, payload: dict[str, object], sequence: int = 0) -> None:
        if type(payload) is not dict or set(payload) != PROGRESS_KEYS_ANTIGRAVITY:
            raise ValueError("progress keys differ")
        write_atomic_antigravity(self.root / "progress_antigravity.json", canonical_json_bytes_antigravity(payload), fault_hook=self.fault_hook, sequence=sequence)
        if self.fault_hook: self.fault_hook("progress_text", sequence)
        line = " ".join(f"{key}={payload[key]}" for key in sorted(payload)) + "\n"
        fd = os.open(self.root / "progress_antigravity.txt", os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try: _write_all_antigravity(fd, line.encode()); os.fsync(fd)
        finally: os.close(fd)

__all__ = ["PROGRESS_KEYS_ANTIGRAVITY", "ProgressWriterAntigravity", "select_outer_workers_antigravity"]
