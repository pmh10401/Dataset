from __future__ import annotations

ALPHAS = (0.005, 0.01, 0.02, 0.05)
WORKLOADS = (25, 50, 100, 200)
VARIANTS = (
    "exp292_baseline",
    "tier_preserving_grammar",
    "additive_grammar_family",
    "grammar_only_control",
)
PRIMITIVE_SCORES = ("transition", "dwell", "recovery")
EXPECTED_DATASETS = 1117
EXPECTED_STANDARD_CANDIDATES = 806

