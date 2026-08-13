"""Descriptor-pinned, source-free Exp451 evaluation replay."""
from __future__ import annotations
import hashlib, os, stat
from pathlib import Path
from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.bootstrap_antigravity import paired_delta_antigravity
from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity

NAMES=("context_antigravity.json","datasets_antigravity.json","result_antigravity.json","dashboard_antigravity.json","artifact_map_antigravity.json","manifest_antigravity.json")
def _conf(v): return ConfusionAntigravity(v["tp"],v["fp"],v["fn"],v["tn"])

def inspect_replay_antigravity(evaluation_root: Path, *, expected_prediction_authority):
    from .evaluation_antigravity import DatasetEvaluationAntigravity,FrozenEvaluationAntigravity,_f1_antigravity,qualifies_antigravity
    from .dashboard_export_antigravity import public_result_antigravity
    from .contracts_antigravity import FrozenPredictionAuthorityAntigravity
    if type(expected_prediction_authority) is not FrozenPredictionAuthorityAntigravity: raise ValueError("independent prediction authority required")
    root=Path(evaluation_root)
    info=os.lstat(root)
    if not stat.S_ISDIR(info.st_mode) or info.st_nlink < 1 or set(os.listdir(root)) != set(NAMES): raise ValueError("evaluation inventory differs")
    root_fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); retained=os.fstat(root_fd); payloads={}; raw={}
    try:
      if (retained.st_dev,retained.st_ino,retained.st_mode)!=(info.st_dev,info.st_ino,info.st_mode): raise ValueError("evaluation root namespace changed")
      if set(os.listdir(root_fd))!=set(NAMES): raise ValueError("evaluation inventory differs")
      for name in NAMES:
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=root_fd)
        try:
          item=os.fstat(fd)
          if not stat.S_ISREG(item.st_mode) or item.st_nlink!=1: raise ValueError("evaluation leaf identity differs")
          chunks=[]
          while chunk:=os.read(fd,65536): chunks.append(chunk)
          data=b"".join(chunks); raw[name]=data; payloads[name]=strict_json_bytes_antigravity(data)
          if canonical_json_bytes_antigravity(payloads[name])!=data: raise ValueError("evaluation leaf noncanonical")
          after=os.stat(name,dir_fd=root_fd,follow_symlinks=False)
          if (after.st_dev,after.st_ino,after.st_mode,after.st_nlink)!=(item.st_dev,item.st_ino,item.st_mode,item.st_nlink): raise ValueError("evaluation leaf namespace changed")
        finally: os.close(fd)
      final=os.fstat(root_fd)
      if (final.st_dev,final.st_ino,final.st_mode)!=(retained.st_dev,retained.st_ino,retained.st_mode) or set(os.listdir(root_fd))!=set(NAMES): raise ValueError("evaluation root changed")
      namespace=os.stat(root,follow_symlinks=False)
      if (namespace.st_dev,namespace.st_ino,namespace.st_mode)!=(retained.st_dev,retained.st_ino,retained.st_mode): raise ValueError("evaluation root namespace changed")
    finally: os.close(root_fd)
    manifest=payloads["manifest_antigravity.json"]
    if type(manifest) is not dict or set(manifest)!={"schema","inventory","artifact_map_sha256","result_sha256","dashboard_sha256"} or manifest["schema"]!="exp451.evaluation.manifest.v1" or manifest["inventory"]!=list(NAMES): raise ValueError("manifest inventory differs")
    artifact=payloads["artifact_map_antigravity.json"]; expected_leaves=set(NAMES)-{"artifact_map_antigravity.json","manifest_antigravity.json"}
    if type(artifact) is not dict or set(artifact)!={"schema","leaf_sha256"} or artifact["schema"]!="exp451.evaluation.artifact_map.v1" or type(artifact["leaf_sha256"]) is not dict or set(artifact["leaf_sha256"])!=expected_leaves: raise ValueError("artifact map differs")
    if hashlib.sha256(raw["artifact_map_antigravity.json"]).hexdigest()!=manifest["artifact_map_sha256"]: raise ValueError("artifact map hash differs")
    for name,digest in artifact["leaf_sha256"].items():
        if type(digest) is not str or hashlib.sha256(raw[name]).hexdigest()!=digest: raise ValueError("leaf hash differs")
    if manifest["result_sha256"]!=artifact["leaf_sha256"]["result_antigravity.json"] or manifest["dashboard_sha256"]!=artifact["leaf_sha256"]["dashboard_antigravity.json"]: raise ValueError("manifest result hash differs")
    context=payloads["context_antigravity.json"]
    if type(context) is not dict or set(context)!={"schema","reviewed_commit","prediction_authority","predecessor_hashes"} or context["schema"]!="exp451.evaluation.context.v1": raise ValueError("context differs")
    restored=FrozenPredictionAuthorityAntigravity.from_payload_antigravity(context["prediction_authority"])
    if restored!=expected_prediction_authority or context["reviewed_commit"]!=expected_prediction_authority.reviewed_commit: raise ValueError("independent prediction authority differs")
    rows=[]
    for value in payloads["datasets_antigravity.json"]["rows"]:
        rows.append(DatasetEvaluationAntigravity(value["sequence"],tuple(value["identity"]),value["target_length"],_conf(value["baseline"]),_conf(value["candidate"]),value["new_detections"],value["new_false_alarms"],value["detection_losses"]))
    value=payloads["result_antigravity.json"]
    result=FrozenEvaluationAntigravity(value["identity"],value["status"],value["reviewed_commit"],value["prediction_authority_sha256"],tuple(value["predecessor_hashes"]),value["replaced_dataset_count"],value["reused_dataset_count"],value["global_dataset_count"],value["label_session_count"],value["label_query_count"],value["label_row_count"],value["bootstrap_dataset_pairs"],tuple(tuple(x) for x in value["target_order"]),_conf(value["baseline_global"]),_conf(value["candidate_global"]),value["baseline_f1"],value["candidate_f1"],value["target_baseline_fn"],value["target_candidate_fn"],value["target_fn_delta"],tuple(value["transition_totals"]),tuple(value["transition_dataset_counts"]),tuple(tuple(x) for x in value["family_fn_deltas"]),tuple(tuple(x) for x in value["gates"]),value["qualified"],value["bootstrap"],tuple(tuple(x) for x in value["baseline_matrix"]),tuple(tuple(x) for x in value["candidate_matrix"]),tuple(rows))
    base=tuple((x.baseline.tp,x.baseline.fp,x.baseline.fn,x.baseline.tn) for x in rows); candidate=tuple((x.candidate.tp,x.candidate.fp,x.candidate.fn,x.candidate.tn) for x in rows)
    old=tuple(sum(x[i] for x in base) for i in range(4)); new=tuple(sum(x[i] for x in candidate) for i in range(4)); expected_global=ConfusionAntigravity(*(getattr(result.baseline_global,f)-old[i]+new[i] for i,f in enumerate(("tp","fp","fn","tn"))))
    transitions=(sum(x.new_detections for x in rows),sum(x.new_false_alarms for x in rows),sum(x.detection_losses for x in rows))
    transition_counts=(sum(x.new_detections>0 for x in rows),sum(x.new_false_alarms>0 for x in rows),sum(x.detection_losses>0 for x in rows))
    families=tuple((family,sum(x.candidate.fn-x.baseline.fn for x in rows if x.identity[1]==family)) for family in dict.fromkeys(x.identity[1] for x in rows))
    gates=(("global_fn",result.candidate_global.fn<1988),("global_fp",result.candidate_global.fp<=314),("global_f1",_f1_antigravity(result.candidate_global)>=0.6264199935),("target_fn_delta",result.target_fn_delta<0)); qualified=all(x[1] for x in gates)
    if len(rows)!=48 or tuple(x.identity for x in rows)!=result.target_order or result.target_order!=expected_prediction_authority.dataset_identities or tuple(x.target_length for x in rows)!=expected_prediction_authority.target_lengths or sum(x.target_length for x in rows)!=72300 or any(sum((x.baseline.tp,x.baseline.fp,x.baseline.fn,x.baseline.tn))!=x.target_length or sum((x.candidate.tp,x.candidate.fp,x.candidate.fn,x.candidate.tn))!=x.target_length for x in rows) or result.predecessor_hashes!=tuple(context["predecessor_hashes"]) or result.reviewed_commit!=expected_prediction_authority.reviewed_commit or result.prediction_authority_sha256!=expected_prediction_authority.canonical_sha256_antigravity or result.baseline_global!=ConfusionAntigravity(1930,314,1988,191668) or result.baseline_f1!=_f1_antigravity(result.baseline_global) or base!=result.baseline_matrix or candidate!=result.candidate_matrix or paired_delta_antigravity(base,candidate)!=result.bootstrap or expected_global!=result.candidate_global or transitions!=result.transition_totals or transition_counts!=result.transition_dataset_counts or result.target_baseline_fn!=old[2] or result.target_candidate_fn!=new[2] or result.target_fn_delta!=new[2]-old[2] or result.family_fn_deltas!=families or result.gates!=gates or result.candidate_f1!=_f1_antigravity(result.candidate_global) or result.qualified!=qualified or result.qualified!=qualifies_antigravity(result.candidate_global,result.target_fn_delta) or result.status!=("qualified" if qualified else "not_qualified") or (result.replaced_dataset_count,result.reused_dataset_count,result.global_dataset_count,result.label_session_count,result.label_query_count,result.label_row_count,result.bootstrap_dataset_pairs)!=(48,1069,1117,1,48,72300,48): raise ValueError("replay arithmetic differs")
    if public_result_antigravity(result)!=payloads["dashboard_antigravity.json"]: raise ValueError("dashboard replay differs")
    return result
