"""Pure exact scientific codec for Exp451 evaluation; no paths, descriptors, or labels."""
from __future__ import annotations
import dataclasses, hashlib, json
from dataclasses import dataclass
from experiments.exp449_residual_fn_method_qualification_antigravity.bootstrap_antigravity import paired_delta_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity
from .artifacts_antigravity import canonical_json_bytes_antigravity
from .contracts_antigravity import FrozenPredictionAuthorityAntigravity
from .predecessor_antigravity import Exp448PredecessorAuthorityAntigravity

BASELINE_GLOBAL_ANTIGRAVITY=ConfusionAntigravity(1930,314,1988,191668)
FAMILIES_ANTIGRAVITY=("Crop","ElectricDevices","FordA","FordB","Yoga","ECG5000","ChlorineConcentration","StarLightCurves")
EXP448_HASHES_ANTIGRAVITY=("ea9856c7de039697ed4f352864def0fde1c0d336714e68c9b0cd586d2c796f05","a58a74f69dcc13a64efb2a679ee7e93eca40ac8c5db87b29b12ed11ea068b21c","54ac58d89367ec5396dafdb180777f787384ffbd21ead1a96764bfe627bbddd5")

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
        if row.sequence!=sequence or type(row.identity) is not tuple or len(row.identity)!=3 or type(row.identity[0]) is not int or any(type(x) is not str for x in row.identity[1:]) or row.identity[1] not in FAMILIES_ANTIGRAVITY: raise ValueError("target order differs")
        if type(row.target_length) is not int or row.target_length<0 or any(type(getattr(row,name)) is not int or getattr(row,name)<0 for name in ("new_detections","new_false_alarms","detection_losses","false_alarm_losses","stable_candidate_count")) or type(row.stable_candidate_sha256) is not str: raise ValueError("dataset exact types differ")
        if sum(dataclasses.astuple(row.baseline))!=row.target_length or sum(dataclasses.astuple(row.candidate))!=row.target_length: raise ValueError("dataset conservation differs")
        expected=(row.baseline.tp+row.new_detections-row.detection_losses,row.baseline.fp+row.new_false_alarms-row.false_alarm_losses,row.baseline.fn-row.new_detections+row.detection_losses,row.baseline.tn-row.new_false_alarms+row.false_alarm_losses)
        if dataclasses.astuple(row.candidate)!=expected: raise ValueError("transition contingency differs")
    if sum(row.target_length for row in rows)!=72300: raise ValueError("target row count differs")
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

def confusion_payload_antigravity(value): return dataclasses.asdict(value)
def dataset_payload_antigravity(row):
    return {"sequence":row.sequence,"identity":list(row.identity),"target_length":row.target_length,"baseline":confusion_payload_antigravity(row.baseline),"candidate":confusion_payload_antigravity(row.candidate),"new_detections":row.new_detections,"new_false_alarms":row.new_false_alarms,"detection_losses":row.detection_losses,"false_alarm_losses":row.false_alarm_losses,"stable_candidate_count":row.stable_candidate_count,"stable_candidate_sha256":row.stable_candidate_sha256}
def baseline_authority_sha256_antigravity(predecessor):
    rows=[]
    for row,name,length in zip(predecessor.target_baseline_rows,predecessor.target_dataset_names,predecessor.target_test_lengths,strict=True):
        stable=canonical_json_bytes_antigravity(sorted(row.stable_candidates)); rows.append({"identity":[row.dataset_order,row.family,name],"target_length":length,"baseline":confusion_payload_antigravity(row.confusion),"stable_candidate_count":len(row.stable_candidates),"stable_candidate_sha256":hashlib.sha256(stable).hexdigest()})
    return hashlib.sha256(canonical_json_bytes_antigravity({"rows":rows})).hexdigest()
def result_payload_antigravity(result):
    return {"schema":"exp451.evaluation.result.v1","identity":result.identity,"status":result.status,"reviewed_commit":result.reviewed_commit,"prediction_authority_sha256":result.prediction_authority_sha256,"predecessor_hashes":list(result.predecessor_hashes),"replaced_dataset_count":48,"reused_dataset_count":1069,"global_dataset_count":1117,"label_session_count":1,"label_query_count":48,"label_row_count":72300,"bootstrap_dataset_pairs":48,"target_order":[list(x) for x in result.target_order],"baseline_global":confusion_payload_antigravity(result.baseline_global),"candidate_global":confusion_payload_antigravity(result.candidate_global),"baseline_f1":result.baseline_f1,"candidate_f1":result.candidate_f1,"target_baseline_fn":result.target_baseline_fn,"target_candidate_fn":result.target_candidate_fn,"target_fn_delta":result.target_fn_delta,"transition_totals":list(result.transition_totals),"transition_dataset_counts":list(result.transition_dataset_counts),"family_fn_deltas":[list(x) for x in result.family_fn_deltas],"gates":[list(x) for x in result.gates],"qualified":result.qualified,"bootstrap":result.bootstrap,"baseline_matrix":[list(x) for x in result.baseline_matrix],"candidate_matrix":[list(x) for x in result.candidate_matrix]}
def public_projection_antigravity(result):
    conf=lambda v:{"tp":v.tp,"fp":v.fp,"fn":v.fn,"tn":v.tn,"f1":f1_antigravity(v)}
    return {"identity":"Exp451","status":result.status,"global":{"baseline":conf(result.baseline_global),"candidate":conf(result.candidate_global)},"qualification":{"qualified":result.qualified,"target_fn_delta":result.target_fn_delta,"gates":dict(result.gates)},"counts":{"replaced":48,"reused":1069,"global":1117},"transitions":{"events":dict(zip(("new_detections","new_false_alarms","detection_losses","false_alarm_losses"),result.transition_totals,strict=True)),"affected_counts":dict(zip(("new_detections","new_false_alarms","detection_losses","false_alarm_losses"),result.transition_dataset_counts,strict=True))},"family_fn_deltas":dict(result.family_fn_deltas),"bootstrap":{"scope":"target_48_paired",**{key:result.bootstrap[key] for key in ("seed","samples","population","dtype","endpoint","percentile_method","fn_delta","fp_delta","f1_delta","fn_interval","fp_interval","f1_interval")}},"runtime":{"sessions":1,"queries":48,"rows":72300}}

def _conf(value):
    if type(value) is not dict or set(value)!={"tp","fp","fn","tn"} or any(type(value[k]) is not int or value[k]<0 for k in value): raise ValueError("confusion schema differs")
    return ConfusionAntigravity(value["tp"],value["fp"],value["fn"],value["tn"])
def _exact(left,right):
    if type(left) is not type(right): return False
    if type(left) is dict: return set(left)==set(right) and all(_exact(left[k],right[k]) for k in left)
    if type(left) is list: return len(left)==len(right) and all(_exact(a,b) for a,b in zip(left,right,strict=True))
    return left==right

def reconstruct_and_validate_antigravity(context_bytes:bytes,datasets_bytes:bytes,result_bytes:bytes,dashboard_bytes:bytes,*,expected_prediction,expected_predecessor)->FrozenEvaluationAntigravity:
    """Decode canonical leaves and reconstruct science through the shared derivation."""
    context,datasets,result,dashboard=map(_strict,(context_bytes,datasets_bytes,result_bytes,dashboard_bytes))
    if type(expected_prediction) is not FrozenPredictionAuthorityAntigravity or type(expected_predecessor) is not Exp448PredecessorAuthorityAntigravity: raise ValueError("independent authorities required")
    if type(context) is not dict or set(context)!={"schema","reviewed_commit","prediction_authority","predecessor_hashes","exp448_baseline_authority_sha256"} or context["schema"]!="exp451.evaluation.context.v1" or tuple(context["predecessor_hashes"])!=EXP448_HASHES_ANTIGRAVITY or context["exp448_baseline_authority_sha256"]!=baseline_authority_sha256_antigravity(expected_predecessor): raise ValueError("context differs")
    if FrozenPredictionAuthorityAntigravity.from_payload_antigravity(context["prediction_authority"])!=expected_prediction: raise ValueError("prediction authority differs")
    if type(datasets) is not dict or set(datasets)!={"schema","rows"} or datasets["schema"]!="exp451.evaluation.datasets.v1" or type(datasets["rows"]) is not list or len(datasets["rows"])!=48: raise ValueError("dataset schema differs")
    keys={"sequence","identity","target_length","baseline","candidate","new_detections","new_false_alarms","detection_losses","false_alarm_losses","stable_candidate_count","stable_candidate_sha256"}; rows=[]
    for sequence,value in enumerate(datasets["rows"]):
        if type(value) is not dict or set(value)!=keys or type(value["sequence"]) is not int or value["sequence"]!=sequence or type(value["identity"]) is not list or len(value["identity"])!=3 or type(value["identity"][0]) is not int or any(type(x) is not str for x in value["identity"][1:]) or type(value["target_length"]) is not int or any(type(value[k]) is not int for k in ("new_detections","new_false_alarms","detection_losses","false_alarm_losses","stable_candidate_count")) or type(value["stable_candidate_sha256"]) is not str: raise ValueError("dataset row types differ")
        rows.append(DatasetEvaluationAntigravity(sequence,tuple(value["identity"]),value["target_length"],_conf(value["baseline"]),_conf(value["candidate"]),value["new_detections"],value["new_false_alarms"],value["detection_losses"],value["false_alarm_losses"],value["stable_candidate_count"],value["stable_candidate_sha256"]))
    expected=[]
    for row,name,length in zip(expected_predecessor.target_baseline_rows,expected_predecessor.target_dataset_names,expected_predecessor.target_test_lengths,strict=True): expected.append((row.dataset_order,row.family,name,length,row.confusion,len(row.stable_candidates),hashlib.sha256(canonical_json_bytes_antigravity(sorted(row.stable_candidates))).hexdigest()))
    if tuple((r.identity[0],r.identity[1],r.identity[2],r.target_length,r.baseline,r.stable_candidate_count,r.stable_candidate_sha256) for r in rows)!=tuple(expected): raise ValueError("baseline authority differs")
    derived=derive_evaluation_antigravity(tuple(rows),expected_prediction.reviewed_commit,expected_prediction.canonical_sha256_antigravity,EXP448_HASHES_ANTIGRAVITY)
    if not _exact(result,result_payload_antigravity(derived)) or not _exact(dashboard,public_projection_antigravity(derived)): raise ValueError("published science differs")
    return derived
