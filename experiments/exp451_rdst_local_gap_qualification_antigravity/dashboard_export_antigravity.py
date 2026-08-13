"""Aggregate-only projection of a frozen Exp451 evaluation."""
from __future__ import annotations
import math

FORBIDDEN=("label","dataset","indices","index","path","database","sql","log","checkpoint","audit","sha","hash","truth","actual")
FAMILIES=("Crop","ElectricDevices","FordA","FordB","Yoga","ECG5000","ChlorineConcentration","StarLightCurves")

def _conf(value): return {"tp":value.tp,"fp":value.fp,"fn":value.fn,"tn":value.tn,"f1":0.0 if 2*value.tp+value.fp+value.fn==0 else 2*value.tp/(2*value.tp+value.fp+value.fn)}
def public_result_antigravity(result):
    from .evaluation_codec_antigravity import FrozenEvaluationAntigravity, public_projection_antigravity
    if type(result) is not FrozenEvaluationAntigravity: raise ValueError("evaluation result type differs")
    value=public_projection_antigravity(result)
    _validate_shape_antigravity(value)
    return value

def _validate_shape_antigravity(value):
    if type(value) is not dict or set(value)!={"identity","status","global","qualification","counts","transitions","family_fn_deltas","bootstrap","runtime"}: raise ValueError("public schema differs")
    def walk(item):
        if type(item) is dict:
            for key,child in item.items():
                if type(key) is not str or any(word in key.lower() for word in FORBIDDEN): raise ValueError("sensitive public key")
                walk(child)
        elif type(item) is list:
            for child in item: walk(child)
        elif type(item) not in (str,int,float,bool) or type(item) is float and not math.isfinite(item): raise ValueError("public value type differs")
        elif type(item) is str and any(word in item.lower() for word in FORBIDDEN): raise ValueError("sensitive public value")
    walk(value)
    if value["identity"]!="Exp451" or value["status"] not in {"qualified","not_qualified"} or value["runtime"]!={"sessions":1,"queries":48,"rows":72300} or any(type(x) is not int for x in value["runtime"].values()) or value["counts"]!={"replaced":48,"reused":1069,"global":1117} or any(type(x) is not int for x in value["counts"].values()): raise ValueError("public constants differ")
    if type(value["global"]) is not dict or set(value["global"])!={"baseline","candidate"}: raise ValueError("global schema differs")
    for name,confusion in value["global"].items():
        if type(confusion) is not dict or set(confusion)!={"tp","fp","fn","tn","f1"} or any(type(confusion[k]) is not int or confusion[k]<0 for k in ("tp","fp","fn","tn")) or type(confusion["f1"]) is not float: raise ValueError("confusion schema differs")
        denominator=2*confusion["tp"]+confusion["fp"]+confusion["fn"]; expected=0.0 if denominator==0 else 2*confusion["tp"]/denominator
        if confusion["f1"]!=expected: raise ValueError("confusion F1 differs")
    baseline=value["global"]["baseline"]; candidate=value["global"]["candidate"]
    if tuple(baseline[k] for k in ("tp","fp","fn","tn"))!=(1930,314,1988,191668) or sum(baseline[k] for k in ("tp","fp","fn","tn"))!=sum(candidate[k] for k in ("tp","fp","fn","tn")): raise ValueError("global conservation differs")
    qualification=value["qualification"]
    if type(qualification) is not dict or set(qualification)!={"qualified","target_fn_delta","gates"} or type(qualification["qualified"]) is not bool or type(qualification["target_fn_delta"]) is not int or type(qualification["gates"]) is not dict or set(qualification["gates"])!={"global_fn","global_fp","global_f1","target_fn_delta"} or any(type(x) is not bool for x in qualification["gates"].values()): raise ValueError("qualification schema differs")
    expected_gates={"global_fn":candidate["fn"]<1988,"global_fp":candidate["fp"]<=314,"global_f1":candidate["f1"]>=0.6264199935,"target_fn_delta":qualification["target_fn_delta"]<0}
    if qualification["gates"]!=expected_gates or qualification["qualified"]!=all(expected_gates.values()) or (value["status"]=="qualified")!=qualification["qualified"]: raise ValueError("qualification arithmetic differs")
    transitions=value["transitions"]
    if type(transitions) is not dict or set(transitions)!={"events","affected_counts"}: raise ValueError("transition schema differs")
    transition_keys={"new_detections","new_false_alarms","detection_losses","false_alarm_losses"}
    for section in transitions.values():
        if type(section) is not dict or set(section)!=transition_keys or any(type(x) is not int or x<0 for x in section.values()): raise ValueError("transition value differs")
    for key in transition_keys:
        events,affected=transitions["events"][key],transitions["affected_counts"][key]
        if affected>48 or (events==0)!=(affected==0) or affected>events: raise ValueError("transition bounds differ")
    if transitions["events"]["new_detections"]-transitions["events"]["detection_losses"]!=-qualification["target_fn_delta"]: raise ValueError("transition FN identity differs")
    if type(value["family_fn_deltas"]) is not dict or set(value["family_fn_deltas"])!=set(FAMILIES) or any(type(v) is not int for v in value["family_fn_deltas"].values()) or sum(value["family_fn_deltas"].values())!=qualification["target_fn_delta"]: raise ValueError("family aggregate differs")
    bootstrap=value["bootstrap"]
    if type(bootstrap) is not dict or set(bootstrap)!={"scope","seed","samples","population","dtype","endpoint","percentile_method","fn_delta","fp_delta","f1_delta","fn_interval","fp_interval","f1_interval"} or (bootstrap["scope"],bootstrap["seed"],bootstrap["samples"],bootstrap["population"],bootstrap["dtype"],bootstrap["endpoint"],bootstrap["percentile_method"])!=("target_48_paired",44920260812,20000,48,"int64",False,"linear") or type(bootstrap["fn_delta"]) is not int or type(bootstrap["fp_delta"]) is not int or type(bootstrap["f1_delta"]) is not float: raise ValueError("bootstrap schema differs")
    for key in ("fn_interval","fp_interval","f1_interval"):
        interval=bootstrap[key]
        if type(interval) is not list or len(interval)!=2 or any(type(x) is not float or not math.isfinite(x) for x in interval) or interval[0]>interval[1]: raise ValueError("bootstrap interval differs")
    if bootstrap["fn_delta"]!=qualification["target_fn_delta"]: raise ValueError("bootstrap target FN differs")
    if bootstrap["fp_delta"]!=transitions["events"]["new_false_alarms"]-transitions["events"]["false_alarm_losses"]: raise ValueError("bootstrap target FP differs")
    for key in ("fn","fp","f1"):
        if not bootstrap[f"{key}_interval"][0]<=bootstrap[f"{key}_delta"]<=bootstrap[f"{key}_interval"][1]: raise ValueError("bootstrap point outside interval")
    return value

def _exact_equal_antigravity(left, right):
    if type(left) is not type(right): return False
    if type(left) is dict:
        return list(left)==list(right) and all(_exact_equal_antigravity(left[key],right[key]) for key in left)
    if type(left) is list:
        return len(left)==len(right) and all(_exact_equal_antigravity(a,b) for a,b in zip(left,right,strict=True))
    return left==right

def validate_public_result_antigravity(value, *, expected_result):
    from .evaluation_codec_antigravity import FrozenEvaluationAntigravity
    if type(expected_result) is not FrozenEvaluationAntigravity: raise ValueError("trusted evaluation authority required")
    expected=public_result_antigravity(expected_result)
    if not _exact_equal_antigravity(value,expected): raise ValueError("public projection differs from trusted evaluation")
    return value
