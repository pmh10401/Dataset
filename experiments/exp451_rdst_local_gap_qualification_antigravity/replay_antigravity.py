"""Basic source-free publication replay; scientific validation lives in the pure codec."""
from __future__ import annotations
import hashlib
from pathlib import Path
from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity
from .evaluation_codec_antigravity import DatasetEvaluationAntigravity, derive_evaluation_antigravity, reconstruct_and_validate_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity

NAMES=("context_antigravity.json","datasets_antigravity.json","result_antigravity.json","dashboard_antigravity.json","artifact_map_antigravity.json","manifest_antigravity.json")

def _exact_equal(left,right):
    if type(left) is not type(right): return False
    if type(left) is dict: return set(left)==set(right) and all(_exact_equal(left[k],right[k]) for k in left)
    if type(left) is list: return len(left)==len(right) and all(_exact_equal(a,b) for a,b in zip(left,right,strict=True))
    return left==right

def _conf(value):
    if type(value) is not dict or set(value)!={"tp","fp","fn","tn"} or any(type(value[k]) is not int or value[k]<0 for k in value): raise ValueError("confusion schema differs")
    return ConfusionAntigravity(value["tp"],value["fp"],value["fn"],value["tn"])

def _reconstruct_codec_payloads_antigravity(context,datasets,result,dashboard,*,expected_prediction,expected_predecessor):
    from .contracts_antigravity import FrozenPredictionAuthorityAntigravity
    from .predecessor_antigravity import Exp448PredecessorAuthorityAntigravity
    from .evaluation_antigravity import _result_payload_antigravity,_baseline_authority_sha256_antigravity,EXP448_HASHES_ANTIGRAVITY
    from .dashboard_export_antigravity import public_result_antigravity
    if type(expected_prediction) is not FrozenPredictionAuthorityAntigravity or type(expected_predecessor) is not Exp448PredecessorAuthorityAntigravity: raise ValueError("independent authorities required")
    if type(context) is not dict or set(context)!={"schema","reviewed_commit","prediction_authority","predecessor_hashes","exp448_baseline_authority_sha256"} or context["schema"]!="exp451.evaluation.context.v1" or tuple(context["predecessor_hashes"])!=EXP448_HASHES_ANTIGRAVITY or context["exp448_baseline_authority_sha256"]!=_baseline_authority_sha256_antigravity(expected_predecessor): raise ValueError("context differs")
    if FrozenPredictionAuthorityAntigravity.from_payload_antigravity(context["prediction_authority"])!=expected_prediction: raise ValueError("prediction authority differs")
    if type(datasets) is not dict or set(datasets)!={"schema","rows"} or datasets["schema"]!="exp451.evaluation.datasets.v1" or type(datasets["rows"]) is not list or len(datasets["rows"])!=48: raise ValueError("dataset schema differs")
    keys={"sequence","identity","target_length","baseline","candidate","new_detections","new_false_alarms","detection_losses","false_alarm_losses","stable_candidate_count","stable_candidate_sha256"}; rows=[]
    for sequence,value in enumerate(datasets["rows"]):
        if type(value) is not dict or set(value)!=keys or type(value["sequence"]) is not int or value["sequence"]!=sequence or type(value["identity"]) is not list or len(value["identity"])!=3 or type(value["identity"][0]) is not int or any(type(x) is not str for x in value["identity"][1:]) or type(value["target_length"]) is not int or any(type(value[k]) is not int for k in ("new_detections","new_false_alarms","detection_losses","false_alarm_losses","stable_candidate_count")) or type(value["stable_candidate_sha256"]) is not str: raise ValueError("dataset row types differ")
        rows.append(DatasetEvaluationAntigravity(sequence,tuple(value["identity"]),value["target_length"],_conf(value["baseline"]),_conf(value["candidate"]),value["new_detections"],value["new_false_alarms"],value["detection_losses"],value["false_alarm_losses"],value["stable_candidate_count"],value["stable_candidate_sha256"]))
    expected=[]
    for predecessor_row,name,length in zip(expected_predecessor.target_baseline_rows,expected_predecessor.target_dataset_names,expected_predecessor.target_test_lengths,strict=True): expected.append((predecessor_row.dataset_order,predecessor_row.family,name,length,predecessor_row.confusion,len(predecessor_row.stable_candidates),hashlib.sha256(canonical_json_bytes_antigravity(sorted(predecessor_row.stable_candidates))).hexdigest()))
    if tuple((r.identity[0],r.identity[1],r.identity[2],r.target_length,r.baseline,r.stable_candidate_count,r.stable_candidate_sha256) for r in rows)!=tuple(expected): raise ValueError("baseline authority differs")
    derived=derive_evaluation_antigravity(tuple(rows),expected_prediction.reviewed_commit,expected_prediction.canonical_sha256_antigravity,EXP448_HASHES_ANTIGRAVITY)
    if not _exact_equal(result,_result_payload_antigravity(derived)): raise ValueError("result differs from shared derivation")
    if not _exact_equal(dashboard,public_result_antigravity(derived)): raise ValueError("dashboard differs from shared derivation")
    return derived

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
