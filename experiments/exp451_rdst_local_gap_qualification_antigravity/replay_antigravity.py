"""Descriptor-pinned, source-free Exp451 evaluation replay."""
from __future__ import annotations
import hashlib, os, stat
from pathlib import Path
from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.bootstrap_antigravity import paired_delta_antigravity
from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity

NAMES=("context_antigravity.json","datasets_antigravity.json","result_antigravity.json","dashboard_antigravity.json","manifest_antigravity.json")
def _conf(v): return ConfusionAntigravity(v["tp"],v["fp"],v["fn"],v["tn"])

def inspect_replay_antigravity(evaluation_root: Path):
    from .evaluation_antigravity import DatasetEvaluationAntigravity,FrozenEvaluationAntigravity,_f1_antigravity,qualifies_antigravity
    from .dashboard_export_antigravity import public_result_antigravity
    root=Path(evaluation_root)
    info=os.lstat(root)
    if not stat.S_ISDIR(info.st_mode) or info.st_nlink < 1 or set(os.listdir(root)) != set(NAMES): raise ValueError("evaluation inventory differs")
    payloads={}
    for name in NAMES:
        item=os.lstat(root/name)
        if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1: raise ValueError("evaluation leaf identity differs")
        data=(root/name).read_bytes(); payloads[name]=strict_json_bytes_antigravity(data)
        if canonical_json_bytes_antigravity(payloads[name]) != data: raise ValueError("evaluation leaf noncanonical")
    manifest=payloads["manifest_antigravity.json"]
    if type(manifest) is not dict or manifest.get("inventory") != list(NAMES): raise ValueError("manifest inventory differs")
    for name,digest in manifest.get("leaf_sha256",{}).items():
        if name=="manifest_antigravity.json" or name not in payloads or hashlib.sha256(canonical_json_bytes_antigravity(payloads[name])).hexdigest()!=digest: raise ValueError("leaf hash differs")
    rows=[]
    for value in payloads["datasets_antigravity.json"]["rows"]:
        rows.append(DatasetEvaluationAntigravity(value["sequence"],tuple(value["identity"]),value["target_length"],_conf(value["baseline"]),_conf(value["candidate"]),value["new_detections"],value["new_false_alarms"],value["detection_losses"]))
    value=payloads["result_antigravity.json"]
    result=FrozenEvaluationAntigravity(value["identity"],value["status"],value["reviewed_commit"],value["prediction_authority_sha256"],tuple(value["predecessor_hashes"]),value["replaced_dataset_count"],value["reused_dataset_count"],value["global_dataset_count"],value["label_session_count"],value["label_query_count"],value["label_row_count"],value["bootstrap_dataset_pairs"],tuple(tuple(x) for x in value["target_order"]),_conf(value["baseline_global"]),_conf(value["candidate_global"]),value["baseline_f1"],value["candidate_f1"],value["target_baseline_fn"],value["target_candidate_fn"],value["target_fn_delta"],tuple(value["transition_totals"]),tuple(value["transition_dataset_counts"]),tuple(tuple(x) for x in value["family_fn_deltas"]),tuple(tuple(x) for x in value["gates"]),value["qualified"],value["bootstrap"],tuple(tuple(x) for x in value["baseline_matrix"]),tuple(tuple(x) for x in value["candidate_matrix"]),tuple(rows))
    base=tuple((x.baseline.tp,x.baseline.fp,x.baseline.fn,x.baseline.tn) for x in rows); candidate=tuple((x.candidate.tp,x.candidate.fp,x.candidate.fn,x.candidate.tn) for x in rows)
    old=tuple(sum(x[i] for x in base) for i in range(4)); new=tuple(sum(x[i] for x in candidate) for i in range(4)); expected_global=ConfusionAntigravity(*(getattr(result.baseline_global,f)-old[i]+new[i] for i,f in enumerate(("tp","fp","fn","tn"))))
    transitions=(sum(x.new_detections for x in rows),sum(x.new_false_alarms for x in rows),sum(x.detection_losses for x in rows))
    if len(rows)!=48 or tuple(x.identity for x in rows)!=result.target_order or sum(x.target_length for x in rows)!=72300 or any(sum((x.candidate.tp,x.candidate.fp,x.candidate.fn,x.candidate.tn))!=x.target_length for x in rows) or base!=result.baseline_matrix or candidate!=result.candidate_matrix or paired_delta_antigravity(base,candidate)!=result.bootstrap or expected_global!=result.candidate_global or transitions!=result.transition_totals or result.target_fn_delta!=new[2]-old[2] or result.candidate_f1!=_f1_antigravity(result.candidate_global) or result.qualified!=qualifies_antigravity(result.candidate_global,result.target_fn_delta): raise ValueError("replay arithmetic differs")
    if public_result_antigravity(result)!=payloads["dashboard_antigravity.json"]: raise ValueError("dashboard replay differs")
    return result
