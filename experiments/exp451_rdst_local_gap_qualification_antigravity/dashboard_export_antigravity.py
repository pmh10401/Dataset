"""Aggregate-only projection of a frozen Exp451 evaluation."""
from __future__ import annotations
import math

FORBIDDEN=("label","dataset","indices","index","path","database","sql","log","checkpoint","audit","sha","hash","truth","actual")
FAMILIES=("Crop","ElectricDevices","FordA","FordB","Yoga","ECG5000","ChlorineConcentration","StarLightCurves")

def _conf(value): return {"tp":value.tp,"fp":value.fp,"fn":value.fn,"tn":value.tn,"f1":0.0 if 2*value.tp+value.fp+value.fn==0 else 2*value.tp/(2*value.tp+value.fp+value.fn)}
def public_result_antigravity(result):
    from .evaluation_antigravity import FrozenEvaluationAntigravity
    if type(result) is not FrozenEvaluationAntigravity: raise ValueError("evaluation result type differs")
    baseline_public=_conf(result.baseline_global); candidate_public=_conf(result.candidate_global)
    value={"identity":"Exp451","status":result.status,"global":{"baseline":baseline_public,"candidate":candidate_public},"qualification":{"qualified":result.qualified,"target_fn_delta":result.target_fn_delta,"gates":dict(result.gates)},"counts":{"replaced":48,"reused":1069,"global":1117},"transitions":{"events":{"new_detections":result.transition_totals[0],"new_false_alarms":result.transition_totals[1],"detection_losses":result.transition_totals[2],"false_alarm_losses":result.transition_totals[3]},"affected_counts":{"new_detections":result.transition_dataset_counts[0],"new_false_alarms":result.transition_dataset_counts[1],"detection_losses":result.transition_dataset_counts[2],"false_alarm_losses":result.transition_dataset_counts[3]}},"family_fn_deltas":{family:delta for family,delta in result.family_fn_deltas},"bootstrap":{"scope":"target_48_paired","seed":result.bootstrap["seed"],"samples":result.bootstrap["samples"],"population":result.bootstrap["population"],"dtype":result.bootstrap["dtype"],"endpoint":result.bootstrap["endpoint"],"percentile_method":result.bootstrap["percentile_method"],"fn_delta":result.bootstrap["fn_delta"],"fp_delta":result.bootstrap["fp_delta"],"f1_delta":result.bootstrap["f1_delta"],"fn_interval":result.bootstrap["fn_interval"],"fp_interval":result.bootstrap["fp_interval"],"f1_interval":result.bootstrap["f1_interval"]},"runtime":{"sessions":1,"queries":48,"rows":72300}}
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
