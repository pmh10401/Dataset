from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RocketModelConfig:
    experiment: int
    source: str
    transformer: str
    requested_kernels: tuple[int, ...]
    max_dilations_per_kernel: int = 32
    n_features_per_kernel: int = 1
    normalise: bool = False


ALPHAS = (0.005, 0.01, 0.02, 0.05)
SEEDS = (20260721, 20261721)
MINIMUM_TRAIN = 5
CROSSFIT_FOLDS = 5
KNN_NEIGHBORS = 3
MAX_TRAIN_REFERENCE = 1000
QUERY_CHUNK_SIZE = 128
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp249_250_official_mini_multirocket")
REPO_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "exp249_250_official_mini_multirocket"
)
HEARTBEAT_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_dashboard_heartbeat.json")
EXTERNAL_STATUS_PATH = Path("/Users/minho/Documents/Dataset/rank_dashboard_external_live_run.json")

MODEL_CONFIGS = {
    249: RocketModelConfig(
        experiment=249,
        source="official_minirocket",
        transformer="minirocket",
        requested_kernels=(1024, 4096),
    ),
    250: RocketModelConfig(
        experiment=250,
        source="official_multirocket",
        transformer="multirocket",
        requested_kernels=(512, 1024),
        n_features_per_kernel=4,
        normalise=False,
    ),
}
