"""Task 13 command-wrapper cleanup for an explicitly reset output directory."""

from __future__ import annotations

from pathlib import Path
import shutil


def reset_checkpoint_state(output_dir: Path) -> None:
    """Remove resumable state while retaining generated immutable publications."""
    output = Path(output_dir)
    shutil.rmtree(output / "checkpoints", ignore_errors=True)
    try:
        (output / "status.json").unlink()
    except FileNotFoundError:
        pass
