"""Aggregate-only projection of a frozen Exp451 evaluation."""
from __future__ import annotations
import math

FORBIDDEN=("label","dataset","indices","index","path","database","sql","log","checkpoint","audit","sha","hash","truth","actual")

def _conf(value): return {"tp":value.tp,"fp":value.fp,"fn":value.fn,"tn":value.tn,"f1":0.0 if 2*value.tp+value.fp+value.fn==0 else 2*value.tp/(2*value.tp+value.fp+value.fn)}
def public_result_antigravity(result):
    from .evaluation_antigravity import FrozenEvaluationAntigravity
    if type(result) is not FrozenEvaluationAntigravity: raise ValueError("evaluation result type differs")
    value={"identity":"Exp451","status":result.status,"global":{"baseline":_conf(result.baseline_global),"candidate":_conf(result.candidate_global)},"qualification":{"qualified":result.qualified,"gates":dict(result.gates)},"counts":{"replaced":48,"reused":1069,"global":1117},"transitions":{"new_detections":result.transition_totals[0],"new_false_alarms":result.transition_totals[1],"detection_losses":result.transition_totals[2]},"family_fn_deltas":{family:delta for family,delta in result.family_fn_deltas},"bootstrap":{"fn_delta":result.bootstrap["fn_delta"],"fp_delta":result.bootstrap["fp_delta"],"f1_delta":result.bootstrap["f1_delta"],"fn_interval":result.bootstrap["fn_interval"],"fp_interval":result.bootstrap["fp_interval"],"f1_interval":result.bootstrap["f1_interval"]},"runtime":{"sessions":1,"queries":48,"rows":72300}}
    validate_public_result_antigravity(value); return value

def validate_public_result_antigravity(value):
    if type(value) is not dict or set(value)!={"identity","status","global","qualification","counts","transitions","family_fn_deltas","bootstrap","runtime"}: raise ValueError("public schema differs")
    def walk(item):
        if type(item) is dict:
            for key,child in item.items():
                if type(key) is not str or any(word in key.lower() for word in FORBIDDEN): raise ValueError("sensitive public key")
                walk(child)
        elif type(item) is list:
            for child in item: walk(child)
        elif type(item) not in (str,int,float,bool) or type(item) is float and not math.isfinite(item): raise ValueError("public value type differs")
    walk(value)
    if value["identity"]!="Exp451" or value["runtime"]!={"sessions":1,"queries":48,"rows":72300} or value["counts"]!={"replaced":48,"reused":1069,"global":1117}: raise ValueError("public constants differ")
    return value

