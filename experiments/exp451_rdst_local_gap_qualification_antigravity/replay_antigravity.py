"""Basic source-free publication replay; scientific validation lives in the pure codec."""
from __future__ import annotations
import hashlib
from pathlib import Path
from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity
from .evaluation_codec_antigravity import reconstruct_and_validate_antigravity

NAMES=("context_antigravity.json","datasets_antigravity.json","result_antigravity.json","dashboard_antigravity.json","artifact_map_antigravity.json","manifest_antigravity.json")

def inspect_replay_antigravity(evaluation_root:Path,*,expected_prediction_authority,expected_predecessor_authority):
    root=Path(evaluation_root)
    if set(path.name for path in root.iterdir())!=set(NAMES): raise ValueError("evaluation inventory differs")
    raw={name:(root/name).read_bytes() for name in NAMES}; values={name:strict_json_bytes_antigravity(payload) for name,payload in raw.items()}
    if any(canonical_json_bytes_antigravity(values[name])!=raw[name] for name in NAMES): raise ValueError("noncanonical leaf")
    manifest=values["manifest_antigravity.json"]; artifact=values["artifact_map_antigravity.json"]
    if type(manifest) is not dict or set(manifest)!={"schema","inventory","artifact_map_sha256","result_sha256","dashboard_sha256"} or manifest["inventory"]!=list(NAMES): raise ValueError("manifest differs")
    leaves=set(NAMES)-{"manifest_antigravity.json","artifact_map_antigravity.json"}
    if type(artifact) is not dict or set(artifact)!={"schema","leaf_sha256"} or set(artifact["leaf_sha256"])!=leaves: raise ValueError("artifact map differs")
    if hashlib.sha256(raw["artifact_map_antigravity.json"]).hexdigest()!=manifest["artifact_map_sha256"] or any(hashlib.sha256(raw[name]).hexdigest()!=artifact["leaf_sha256"][name] for name in leaves): raise ValueError("artifact hash differs")
    return reconstruct_and_validate_antigravity(raw["context_antigravity.json"],raw["datasets_antigravity.json"],raw["result_antigravity.json"],raw["dashboard_antigravity.json"],expected_prediction=expected_prediction_authority,expected_predecessor=expected_predecessor_authority)
