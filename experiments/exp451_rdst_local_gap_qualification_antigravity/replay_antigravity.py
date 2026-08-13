"""Descriptor-pinned, source-free Exp451 evaluation replay."""
from __future__ import annotations
import hashlib, os, stat
from pathlib import Path
from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.bootstrap_antigravity import paired_delta_antigravity
from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity

NAMES=("context_antigravity.json","datasets_antigravity.json","result_antigravity.json","dashboard_antigravity.json","artifact_map_antigravity.json","manifest_antigravity.json")
def _conf(v):
    if type(v) is not dict or set(v)!={"tp","fp","fn","tn"} or any(type(v[k]) is not int or v[k]<0 for k in v): raise ValueError("confusion schema differs")
    return ConfusionAntigravity(v["tp"],v["fp"],v["fn"],v["tn"])
def _identity(value): return (value.st_dev,value.st_ino,value.st_mode,value.st_nlink,value.st_size,value.st_mtime_ns,value.st_ctime_ns)

class _RetainedPinsAntigravity:
    def __init__(self,root_fd): self.root_fd=root_fd; self.leaf_fds=[]
    def close(self):
        errors=[]
        for fd in reversed(self.leaf_fds):
            try: os.close(fd)
            except OSError as error: errors.append(error)
        self.leaf_fds=[]
        if self.root_fd>=0:
            try: os.close(self.root_fd)
            except OSError as error: errors.append(error)
            self.root_fd=-1
        if errors: raise ExceptionGroup("replay descriptor close failures",errors)
    def __del__(self): self.close()

def inspect_replay_antigravity(evaluation_root: Path, *, expected_prediction_authority, expected_predecessor_authority, _retained_root_fd: int | None = None, _expected_named_identity: tuple[int, ...] | None = None):
    from .evaluation_antigravity import DatasetEvaluationAntigravity,FrozenEvaluationAntigravity,_f1_antigravity,qualifies_antigravity,EXP448_HASHES_ANTIGRAVITY,_baseline_authority_sha256_antigravity
    from .dashboard_export_antigravity import public_result_antigravity
    from .contracts_antigravity import FrozenPredictionAuthorityAntigravity
    from .predecessor_antigravity import Exp448PredecessorAuthorityAntigravity
    if type(expected_prediction_authority) is not FrozenPredictionAuthorityAntigravity or type(expected_predecessor_authority) is not Exp448PredecessorAuthorityAntigravity: raise ValueError("independent authorities required")
    root=Path(evaluation_root)
    info=os.lstat(root) if _retained_root_fd is None else os.fstat(_retained_root_fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_nlink < 1 or set(os.listdir(root)) != set(NAMES): raise ValueError("evaluation inventory differs")
    root_fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW) if _retained_root_fd is None else os.dup(_retained_root_fd); pins=_RetainedPinsAntigravity(root_fd); retained=os.fstat(root_fd); payloads={}; raw={}; identities={}
    if _identity(retained)!=_identity(info): raise ValueError("evaluation root namespace changed")
    if _expected_named_identity is not None and _identity(retained)!=_expected_named_identity: raise ValueError("retained evaluation identity differs")
    try:
      if set(os.listdir(root_fd))!=set(NAMES): raise ValueError("evaluation inventory differs")
      for name in NAMES:
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=root_fd)
        pins.leaf_fds.append(fd)
        try:
          item=os.fstat(fd)
          if not stat.S_ISREG(item.st_mode) or item.st_nlink!=1: raise ValueError("evaluation leaf identity differs")
          identities[name]=_identity(item)
          chunks=[]
          while chunk:=os.read(fd,65536): chunks.append(chunk)
          data=b"".join(chunks); raw[name]=data; payloads[name]=strict_json_bytes_antigravity(data)
          if canonical_json_bytes_antigravity(payloads[name])!=data: raise ValueError("evaluation leaf noncanonical")
          after=os.stat(name,dir_fd=root_fd,follow_symlinks=False)
          if _identity(after)!=identities[name] or _identity(os.fstat(fd))!=identities[name]: raise ValueError("evaluation leaf namespace changed")
        except BaseException: raise
      final=os.fstat(root_fd)
      if _identity(final)!=_identity(retained) or set(os.listdir(root_fd))!=set(NAMES): raise ValueError("evaluation root changed")
      namespace=os.stat(root,follow_symlinks=False)
      if _identity(namespace)!=_identity(retained): raise ValueError("evaluation root namespace changed")
    except BaseException:
      pins.close(); raise
    manifest=payloads["manifest_antigravity.json"]
    if type(manifest) is not dict or set(manifest)!={"schema","inventory","artifact_map_sha256","result_sha256","dashboard_sha256"} or manifest["schema"]!="exp451.evaluation.manifest.v1" or manifest["inventory"]!=list(NAMES): raise ValueError("manifest inventory differs")
    artifact=payloads["artifact_map_antigravity.json"]; expected_leaves=set(NAMES)-{"artifact_map_antigravity.json","manifest_antigravity.json"}
    if type(artifact) is not dict or set(artifact)!={"schema","leaf_sha256"} or artifact["schema"]!="exp451.evaluation.artifact_map.v1" or type(artifact["leaf_sha256"]) is not dict or set(artifact["leaf_sha256"])!=expected_leaves: raise ValueError("artifact map differs")
    if hashlib.sha256(raw["artifact_map_antigravity.json"]).hexdigest()!=manifest["artifact_map_sha256"]: raise ValueError("artifact map hash differs")
    for name,digest in artifact["leaf_sha256"].items():
        if type(digest) is not str or hashlib.sha256(raw[name]).hexdigest()!=digest: raise ValueError("leaf hash differs")
    if manifest["result_sha256"]!=artifact["leaf_sha256"]["result_antigravity.json"] or manifest["dashboard_sha256"]!=artifact["leaf_sha256"]["dashboard_antigravity.json"]: raise ValueError("manifest result hash differs")
    context=payloads["context_antigravity.json"]
    if type(context) is not dict or set(context)!={"schema","reviewed_commit","prediction_authority","predecessor_hashes","exp448_baseline_authority_sha256"} or context["schema"]!="exp451.evaluation.context.v1" or tuple(context["predecessor_hashes"])!=EXP448_HASHES_ANTIGRAVITY or context["exp448_baseline_authority_sha256"]!=_baseline_authority_sha256_antigravity(expected_predecessor_authority): raise ValueError("context differs")
    restored=FrozenPredictionAuthorityAntigravity.from_payload_antigravity(context["prediction_authority"])
    if restored!=expected_prediction_authority or context["reviewed_commit"]!=expected_prediction_authority.reviewed_commit: raise ValueError("independent prediction authority differs")
    datasets_payload=payloads["datasets_antigravity.json"]
    if type(datasets_payload) is not dict or set(datasets_payload)!={"schema","rows"} or datasets_payload["schema"]!="exp451.evaluation.datasets.v1" or type(datasets_payload["rows"]) is not list or len(datasets_payload["rows"])!=48: pins.close(); raise ValueError("dataset evidence schema differs")
    row_keys={"sequence","identity","target_length","baseline","candidate","new_detections","new_false_alarms","detection_losses","false_alarm_losses","stable_candidate_count","stable_candidate_sha256"}
    rows=[]
    for sequence,value in enumerate(datasets_payload["rows"]):
        if type(value) is not dict or set(value)!=row_keys or type(value["sequence"]) is not int or value["sequence"]!=sequence or type(value["identity"]) is not list or len(value["identity"])!=3 or type(value["target_length"]) is not int or any(type(value[key]) is not int or value[key]<0 for key in ("new_detections","new_false_alarms","detection_losses","false_alarm_losses","stable_candidate_count")) or type(value["stable_candidate_sha256"]) is not str or len(value["stable_candidate_sha256"])!=64: pins.close(); raise ValueError("dataset evidence row differs")
        rows.append(DatasetEvaluationAntigravity(value["sequence"],tuple(value["identity"]),value["target_length"],_conf(value["baseline"]),_conf(value["candidate"]),value["new_detections"],value["new_false_alarms"],value["detection_losses"],value["false_alarm_losses"],value["stable_candidate_count"],value["stable_candidate_sha256"]))
    expected_rows=[]
    for predecessor_row,name,length in zip(expected_predecessor_authority.target_baseline_rows,expected_predecessor_authority.target_dataset_names,expected_predecessor_authority.target_test_lengths,strict=True):
        expected_rows.append((predecessor_row.dataset_order,predecessor_row.family,name,length,predecessor_row.confusion,len(predecessor_row.stable_candidates),hashlib.sha256(canonical_json_bytes_antigravity(sorted(predecessor_row.stable_candidates))).hexdigest()))
    if tuple((row.identity[0],row.identity[1],row.identity[2],row.target_length,row.baseline,row.stable_candidate_count,row.stable_candidate_sha256) for row in rows)!=tuple(expected_rows): pins.close(); raise ValueError("published Exp448 baseline evidence differs")
    value=payloads["result_antigravity.json"]
    result_keys={"schema","identity","status","reviewed_commit","prediction_authority_sha256","predecessor_hashes","replaced_dataset_count","reused_dataset_count","global_dataset_count","label_session_count","label_query_count","label_row_count","bootstrap_dataset_pairs","target_order","baseline_global","candidate_global","baseline_f1","candidate_f1","target_baseline_fn","target_candidate_fn","target_fn_delta","transition_totals","transition_dataset_counts","family_fn_deltas","gates","qualified","bootstrap","baseline_matrix","candidate_matrix"}
    if type(value) is not dict or set(value)!=result_keys or value["schema"]!="exp451.evaluation.result.v1": pins.close(); raise ValueError("result schema differs")
    integer_fields={"replaced_dataset_count","reused_dataset_count","global_dataset_count","label_session_count","label_query_count","label_row_count","bootstrap_dataset_pairs","target_baseline_fn","target_candidate_fn","target_fn_delta"}
    if any(type(value[key]) is not int for key in integer_fields) or type(value["baseline_f1"]) is not float or type(value["candidate_f1"]) is not float or type(value["qualified"]) is not bool or type(value["identity"]) is not str or type(value["status"]) is not str or type(value["reviewed_commit"]) is not str or type(value["prediction_authority_sha256"]) is not str: pins.close(); raise ValueError("result primitive types differ")
    result=FrozenEvaluationAntigravity(value["identity"],value["status"],value["reviewed_commit"],value["prediction_authority_sha256"],tuple(value["predecessor_hashes"]),value["replaced_dataset_count"],value["reused_dataset_count"],value["global_dataset_count"],value["label_session_count"],value["label_query_count"],value["label_row_count"],value["bootstrap_dataset_pairs"],tuple(tuple(x) for x in value["target_order"]),_conf(value["baseline_global"]),_conf(value["candidate_global"]),value["baseline_f1"],value["candidate_f1"],value["target_baseline_fn"],value["target_candidate_fn"],value["target_fn_delta"],tuple(value["transition_totals"]),tuple(value["transition_dataset_counts"]),tuple(tuple(x) for x in value["family_fn_deltas"]),tuple(tuple(x) for x in value["gates"]),value["qualified"],value["bootstrap"],tuple(tuple(x) for x in value["baseline_matrix"]),tuple(tuple(x) for x in value["candidate_matrix"]),tuple(rows))
    base=tuple((x.baseline.tp,x.baseline.fp,x.baseline.fn,x.baseline.tn) for x in rows); candidate=tuple((x.candidate.tp,x.candidate.fp,x.candidate.fn,x.candidate.tn) for x in rows)
    old=tuple(sum(x[i] for x in base) for i in range(4)); new=tuple(sum(x[i] for x in candidate) for i in range(4)); expected_global=ConfusionAntigravity(*(getattr(result.baseline_global,f)-old[i]+new[i] for i,f in enumerate(("tp","fp","fn","tn"))))
    transitions=(sum(x.new_detections for x in rows),sum(x.new_false_alarms for x in rows),sum(x.detection_losses for x in rows),sum(x.false_alarm_losses for x in rows))
    transition_counts=(sum(x.new_detections>0 for x in rows),sum(x.new_false_alarms>0 for x in rows),sum(x.detection_losses>0 for x in rows),sum(x.false_alarm_losses>0 for x in rows))
    families=tuple((family,sum(x.candidate.fn-x.baseline.fn for x in rows if x.identity[1]==family)) for family in dict.fromkeys(x.identity[1] for x in rows))
    gates=(("global_fn",result.candidate_global.fn<1988),("global_fp",result.candidate_global.fp<=314),("global_f1",_f1_antigravity(result.candidate_global)>=0.6264199935),("target_fn_delta",result.target_fn_delta<0)); qualified=all(x[1] for x in gates)
    if len(rows)!=48 or tuple(x.identity for x in rows)!=result.target_order or result.target_order!=expected_prediction_authority.dataset_identities or tuple(x.target_length for x in rows)!=expected_prediction_authority.target_lengths or sum(x.target_length for x in rows)!=72300 or any(sum((x.baseline.tp,x.baseline.fp,x.baseline.fn,x.baseline.tn))!=x.target_length or sum((x.candidate.tp,x.candidate.fp,x.candidate.fn,x.candidate.tn))!=x.target_length or x.candidate.tp!=x.baseline.tp+x.new_detections-x.detection_losses or x.candidate.fn!=x.baseline.fn-x.new_detections+x.detection_losses or x.candidate.fp!=x.baseline.fp+x.new_false_alarms-x.false_alarm_losses or x.candidate.tn!=x.baseline.tn-x.new_false_alarms+x.false_alarm_losses for x in rows) or result.predecessor_hashes!=tuple(context["predecessor_hashes"]) or result.reviewed_commit!=expected_prediction_authority.reviewed_commit or result.prediction_authority_sha256!=expected_prediction_authority.canonical_sha256_antigravity or result.baseline_global!=ConfusionAntigravity(1930,314,1988,191668) or result.baseline_f1!=_f1_antigravity(result.baseline_global) or base!=result.baseline_matrix or candidate!=result.candidate_matrix or paired_delta_antigravity(base,candidate)!=result.bootstrap or expected_global!=result.candidate_global or transitions!=result.transition_totals or transition_counts!=result.transition_dataset_counts or result.target_baseline_fn!=old[2] or result.target_candidate_fn!=new[2] or result.target_fn_delta!=new[2]-old[2] or result.family_fn_deltas!=families or result.gates!=gates or result.candidate_f1!=_f1_antigravity(result.candidate_global) or result.qualified!=qualified or result.qualified!=qualifies_antigravity(result.candidate_global,result.target_fn_delta) or result.status!=("qualified" if qualified else "not_qualified") or (result.replaced_dataset_count,result.reused_dataset_count,result.global_dataset_count,result.label_session_count,result.label_query_count,result.label_row_count,result.bootstrap_dataset_pairs)!=(48,1069,1117,1,48,72300,48): pins.close(); raise ValueError("replay arithmetic differs")
    if public_result_antigravity(result)!=payloads["dashboard_antigravity.json"]: raise ValueError("dashboard replay differs")
    for name,identity in identities.items():
        if _identity(os.fstat(pins.leaf_fds[NAMES.index(name)]))!=identity or _identity(os.stat(name,dir_fd=root_fd,follow_symlinks=False))!=identity: pins.close(); raise ValueError("evaluation leaf changed after semantics")
    if _identity(os.fstat(root_fd))!=_identity(retained) or _identity(os.stat(root,follow_symlinks=False))!=_identity(retained) or set(os.listdir(root_fd))!=set(NAMES): pins.close(); raise ValueError("evaluation root changed after semantics")
    pins.close(); return result
