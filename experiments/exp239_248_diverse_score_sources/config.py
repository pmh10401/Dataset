from __future__ import annotations

from pathlib import Path


SEED = 20260720
SECOND_SEED = 20261720
ALPHAS = (0.005, 0.01, 0.02, 0.05)
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp239_248_diverse_score_sources")
REPO_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "exp239_248_diverse_score_sources"
)
HEARTBEAT_PATH = Path(
    "/Users/minho/Documents/Dataset/rank_experiments_dashboard_heartbeat.json"
)
EXTERNAL_STATUS_PATH = Path(
    "/Users/minho/Documents/Dataset/rank_dashboard_external_live_run.json"
)

SOURCE_EXPERIMENTS = {
    239: "multiscale_wavelet_energy_deviation",
    240: "ordinal_pattern_distribution_shift",
    241: "symbolic_motif_rarity",
    242: "complexity_signature_deviation",
    243: "recurrence_dynamics_deviation",
    244: "robust_state_space_innovation",
    245: "smooth_fourier_normal_residual",
    246: "transformation_consistency",
    247: "masked_patch_context_error",
    248: "nonlinear_normal_density",
}

