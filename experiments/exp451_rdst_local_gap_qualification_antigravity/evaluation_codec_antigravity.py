"""Pure exact scientific codec for Exp451 evaluation; no paths, descriptors, or labels."""
from __future__ import annotations
import dataclasses, hashlib, json
from dataclasses import dataclass
from experiments.exp449_residual_fn_method_qualification_antigravity.bootstrap_antigravity import paired_delta_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity

BASELINE_GLOBAL_ANTIGRAVITY=ConfusionAntigravity(1930,314,1988,191668)
FAMILIES_ANTIGRAVITY=("Crop","ElectricDevices","FordA","FordB","Yoga","ECG5000","ChlorineConcentration","StarLightCurves")

@dataclass(frozen=True,slots=True)
class DatasetEvaluationAntigravity:
    sequence:int; identity:tuple[int,str,str]; target_length:int
    baseline:ConfusionAntigravity; candidate:ConfusionAntigravity
    new_detections:int; new_false_alarms:int; detection_losses:int; false_alarm_losses:int
    stable_candidate_count:int; stable_candidate_sha256:str

@dataclass(frozen=True,slots=True)
class FrozenEvaluationAntigravity:
    identity:str; status:str; reviewed_commit:str; prediction_authority_sha256:str
    predecessor_hashes:tuple[str,str,str]; replaced_dataset_count:int; reused_dataset_count:int
    global_dataset_count:int; label_session_count:int; label_query_count:int; label_row_count:int
    bootstrap_dataset_pairs:int; target_order:tuple[tuple[int,str,str],...]
    baseline_global:ConfusionAntigravity; candidate_global:ConfusionAntigravity
    baseline_f1:float; candidate_f1:float; target_baseline_fn:int; target_candidate_fn:int
    target_fn_delta:int; transition_totals:tuple[int,...]; transition_dataset_counts:tuple[int,...]
    family_fn_deltas:tuple[tuple[str,int],...]; gates:tuple[tuple[str,bool],...]; qualified:bool
    bootstrap:dict[str,object]; baseline_matrix:tuple[tuple[int,int,int,int],...]
    candidate_matrix:tuple[tuple[int,int,int,int],...]; dataset_rows:tuple[DatasetEvaluationAntigravity,...]

def f1_antigravity(value:ConfusionAntigravity)->float:
    denominator=2*value.tp+value.fp+value.fn
    return 0.0 if denominator==0 else 2.0*value.tp/denominator

def derive_evaluation_antigravity(rows:tuple[DatasetEvaluationAntigravity,...],reviewed_commit:str,prediction_sha:str,predecessor_hashes:tuple[str,str,str])->FrozenEvaluationAntigravity:
    if type(rows) is not tuple or len(rows)!=48 or any(type(row) is not DatasetEvaluationAntigravity for row in rows): raise ValueError("exact 48 dataset rows required")
    if type(reviewed_commit) is not str or type(prediction_sha) is not str or type(predecessor_hashes) is not tuple or len(predecessor_hashes)!=3 or any(type(x) is not str for x in predecessor_hashes): raise ValueError("authority types differ")
    for sequence,row in enumerate(rows):
        if row.sequence!=sequence or type(row.identity) is not tuple or len(row.identity)!=3 or type(row.identity[0]) is not int or any(type(x) is not str for x in row.identity[1:]) or row.identity[0]!=sequence or row.identity[1] not in FAMILIES_ANTIGRAVITY: raise ValueError("target order differs")
        if type(row.target_length) is not int or row.target_length<0 or any(type(getattr(row,name)) is not int or getattr(row,name)<0 for name in ("new_detections","new_false_alarms","detection_losses","false_alarm_losses","stable_candidate_count")) or type(row.stable_candidate_sha256) is not str: raise ValueError("dataset exact types differ")
        if sum(dataclasses.astuple(row.baseline))!=row.target_length or sum(dataclasses.astuple(row.candidate))!=row.target_length: raise ValueError("dataset conservation differs")
        expected=(row.baseline.tp+row.new_detections-row.detection_losses,row.baseline.fp+row.new_false_alarms-row.false_alarm_losses,row.baseline.fn-row.new_detections+row.detection_losses,row.baseline.tn-row.new_false_alarms+row.false_alarm_losses)
        if dataclasses.astuple(row.candidate)!=expected: raise ValueError("transition contingency differs")
    matrix=lambda field:tuple(dataclasses.astuple(getattr(row,field)) for row in rows)
    baseline,candidate=matrix("baseline"),matrix("candidate")
    sums=lambda values:tuple(sum(row[i] for row in values) for i in range(4))
    old,new=sums(baseline),sums(candidate)
    global_candidate=ConfusionAntigravity(*(getattr(BASELINE_GLOBAL_ANTIGRAVITY,key)-old[i]+new[i] for i,key in enumerate(("tp","fp","fn","tn"))))
    delta=new[2]-old[2]; gates=(("global_fn",global_candidate.fn<1988),("global_fp",global_candidate.fp<=314),("global_f1",f1_antigravity(global_candidate)>=0.6264199935),("target_fn_delta",delta<0))
    families=tuple((family,sum(row.candidate.fn-row.baseline.fn for row in rows if row.identity[1]==family)) for family in FAMILIES_ANTIGRAVITY)
    names=("new_detections","new_false_alarms","detection_losses","false_alarm_losses")
    totals=tuple(sum(getattr(row,name) for row in rows) for name in names); affected=tuple(sum(getattr(row,name)>0 for row in rows) for name in names)
    bootstrap=paired_delta_antigravity(baseline,candidate); qualified=all(value for _,value in gates)
    return FrozenEvaluationAntigravity("Exp451","qualified" if qualified else "not_qualified",reviewed_commit,prediction_sha,predecessor_hashes,48,1069,1117,1,48,72300,48,tuple(row.identity for row in rows),BASELINE_GLOBAL_ANTIGRAVITY,global_candidate,f1_antigravity(BASELINE_GLOBAL_ANTIGRAVITY),f1_antigravity(global_candidate),old[2],new[2],delta,totals,affected,families,gates,qualified,bootstrap,baseline,candidate,rows)

def _strict(payload:bytes):
    value=json.loads(payload)
    if json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()+b"\n"!=payload: raise ValueError("noncanonical codec bytes")
    return value

def reconstruct_and_validate_antigravity(context_bytes:bytes,datasets_bytes:bytes,result_bytes:bytes,dashboard_bytes:bytes,*,expected_prediction,expected_predecessor)->FrozenEvaluationAntigravity:
    """Decode canonical leaves and reconstruct science through the shared derivation."""
    context,datasets,result,dashboard=map(_strict,(context_bytes,datasets_bytes,result_bytes,dashboard_bytes))
    if type(context) is not dict or type(datasets) is not dict or type(result) is not dict or type(dashboard) is not dict: raise ValueError("codec leaf type differs")
    from .replay_antigravity import _reconstruct_codec_payloads_antigravity
    return _reconstruct_codec_payloads_antigravity(context,datasets,result,dashboard,expected_prediction=expected_prediction,expected_predecessor=expected_predecessor)
