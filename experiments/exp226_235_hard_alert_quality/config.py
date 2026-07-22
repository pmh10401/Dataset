from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
OUTPUT_ROOT = DATA_ROOT / "exp226_235_hard_alert_quality"
REPO_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "exp226_235_hard_alert_quality"
SEED = 20260719


@dataclass(frozen=True)
class FrozenConfig:
    database_path: Path = DB_PATH
    output_root: Path = OUTPUT_ROOT
    seed: int = SEED
    target_length_min: int = 8
    target_length_max: int = 2048

