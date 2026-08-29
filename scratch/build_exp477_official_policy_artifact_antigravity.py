"""Build the anonymous frozen training artifact for official Exp476 routing."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

from experiments.exp474_conservative_fallback_routing_antigravity.evaluation_antigravity import (
    load_checkpoint_records_antigravity,
)
from experiments.exp474_conservative_fallback_routing_antigravity.policy_antigravity import (
    BASELINE_METHOD_ANTIGRAVITY,
    FEATURE_KEYS_ANTIGRAVITY,
)
from experiments.exp474_conservative_fallback_routing_antigravity.run_antigravity import (
    CHECKPOINT_DIR_ANTIGRAVITY,
    FEATURE_CACHE_ANTIGRAVITY,
)
from experiments.exp_routing_label_free_antigravity.routing_rules_antigravity import (
    route_dataset_antigravity,
)


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
EXP476_RESULT_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp476_minimum_zf_fallback_routing_antigravity/results_antigravity.json"
)
ARTIFACT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp_routing_label_free_antigravity/official_exp476_policy_artifact_antigravity.json"
)
ROUTING_RULES_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp_routing_label_free_antigravity/routing_rules_antigravity.py"
)
EXPECTED_EXP476_SHA256_ANTIGRAVITY = (
    "8b6f881d25d0d2865e05c5a5b9ca204f78641e1364bb60d536e7dcc8ec21733f"
)
POLICY_CONFIG_ANTIGRAVITY = {
    "k": 31,
    "min_support": 20,
    "positive_rate_threshold": 0.90,
    "lower_quantile": 0.20,
    "min_auc_gain": 0.0,
    "min_zf_gain": 0.0,
    "min_lower_zf_gain": 0.0,
}


def build_artifact_antigravity() -> dict[str, object]:
    result_bytes = EXP476_RESULT_ANTIGRAVITY.read_bytes()
    result_hash = sha256(result_bytes).hexdigest()
    if result_hash != EXPECTED_EXP476_SHA256_ANTIGRAVITY:
        raise RuntimeError("Exp476 result hash changed; refusing to build official artifact")
    result = json.loads(result_bytes)
    if result.get("dataset_count") != 962 or not result.get("gates", {}).get("overall_pass"):
        raise RuntimeError("Exp476 qualification evidence is incomplete or did not pass")

    feature_bytes = FEATURE_CACHE_ANTIGRAVITY.read_bytes()
    features_by_name = json.loads(feature_bytes)
    records = load_checkpoint_records_antigravity(CHECKPOINT_DIR_ANTIGRAVITY)
    if len(records) != 962 or set(features_by_name) != {row["dataset"] for row in records}:
        raise RuntimeError("frozen checkpoint and feature-cache populations do not match")

    training_rows: list[dict[str, object]] = []
    proposal_alignment_change_count = 0
    for record in records:
        name = str(record["dataset"])
        scores = record["scores"]
        if not isinstance(scores, dict):
            raise TypeError("checkpoint scores must be an object")
        baseline = scores[BASELINE_METHOD_ANTIGRAVITY]
        features = {
            key: float(features_by_name[name][key]) for key in FEATURE_KEYS_ANTIGRAVITY
        }
        proposal = route_dataset_antigravity(features)
        candidate = scores[proposal]
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            raise TypeError("checkpoint method scores must be objects")
        proposal_alignment_change_count += int(proposal != str(record["proposal"]))
        auc_gain = float(candidate["auc"]) - float(baseline["auc"])
        zf_gain = float(candidate["zf"]) - float(baseline["zf"])
        if not all(math.isfinite(value) for value in (*features.values(), auc_gain, zf_gain)):
            raise ValueError("official policy artifact requires finite numeric values")
        training_rows.append(
            {
                "features": features,
                "proposal": proposal,
                "auc_gain": auc_gain,
                "zf_gain": zf_gain,
            }
        )

    artifact: dict[str, object] = {
        "schema": "official_exp476_minimum_zf_policy_artifact_v1",
        "source_exp476_result_sha256": result_hash,
        "source_feature_cache_sha256": sha256(feature_bytes).hexdigest(),
        "source_routing_rules_sha256": sha256(
            ROUTING_RULES_PATH_ANTIGRAVITY.read_bytes()
        ).hexdigest(),
        "proposal_alignment_change_count": proposal_alignment_change_count,
        "row_count": len(training_rows),
        "feature_keys": list(FEATURE_KEYS_ANTIGRAVITY),
        "policy_config": POLICY_CONFIG_ANTIGRAVITY,
        "training_rows": training_rows,
    }
    ARTIFACT_PATH_ANTIGRAVITY.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact


if __name__ == "__main__":
    built = build_artifact_antigravity()
    print(json.dumps({"path": str(ARTIFACT_PATH_ANTIGRAVITY), "rows": built["row_count"]}))
