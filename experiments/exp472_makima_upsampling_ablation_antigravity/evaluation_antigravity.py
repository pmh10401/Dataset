"""Path-only, one-label-read Exp472 E-versus-D evaluation."""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from .authority_antigravity import Exp472ReuseAuthorityAntigravity, Exp472VerifiedExp471BundleAntigravity, bind_e_to_exp471_exp472_antigravity, verify_exp471_reuse_exp472_antigravity, _coordinate
from .contracts_antigravity import canonical_json_bytes_exp472_antigravity

METHOD_ORDER_EXP472_ANTIGRAVITY=("A","B","C","D","E","knn_distance","lof_novelty","ocsvm_rbf","isolation_forest","deep_svdd_1d","drocc_1d")
_DRAWS=20_000; _SEED=47220260828; _TIE=1e-12

def _vector(value: object, name: str) -> np.ndarray:
    result=np.asarray(value,dtype=np.float64)
    if result.ndim!=1 or result.size<1 or not np.isfinite(result).all(): raise ValueError(f"Exp472 {name} differs")
    return result

@dataclass(frozen=True,slots=True)
class Exp472VerifiedEBundleAntigravity:
    root: Path; database_sha256: str; dataset_order: tuple[str,...]; coordinates: Mapping[str,Mapping[str,object]]; scores: Mapping[str,Sequence[object]]; upsampled_rows: Mapping[str,int]; path_sha256: str
    def __post_init__(self)->None:
        if not isinstance(self.root,Path) or not self.root.is_absolute() or len(self.dataset_order)!=1117 or len(set(self.dataset_order))!=1117: raise ValueError("Exp472 E inventory differs")
        if tuple(self.coordinates)!=self.dataset_order or tuple(self.scores)!=self.dataset_order or tuple(self.upsampled_rows)!=self.dataset_order: raise ValueError("Exp472 E coverage differs")
        for d in self.dataset_order:
            _coordinate(self.coordinates[d]); values=_vector(self.scores[d],"E scores")
            if values.size != self.coordinates[d]["count"] or type(self.upsampled_rows[d]) is not int or self.upsampled_rows[d]<0: raise ValueError("Exp472 E shard differs")

def _read_verified_e_path_antigravity(root: Path, exp471: Exp472VerifiedExp471BundleAntigravity)->Exp472VerifiedEBundleAntigravity:
    """Task 5 is intentionally not an official-run reader; Task 6 supplies its frozen context/catalog.
    Rejecting an unbound Path is safer than accepting an unverified manifest mapping.
    """
    raise ValueError("Exp472 E Path requires the Task 6 verified manifest adapter")

def paired_e_minus_d_exp472_antigravity(e_values:Sequence[object],d_values:Sequence[object],*,seed:int=_SEED)->Mapping[str,object]:
    e=_vector(e_values,"paired E"); d=_vector(d_values,"paired D")
    if e.shape!=d.shape or type(seed)is not int: raise ValueError("Exp472 paired values differ")
    delta=e-d; means=np.empty(_DRAWS,dtype=np.float64); rng=np.random.default_rng(seed)
    for start in range(0,_DRAWS,512):
        end=min(_DRAWS,start+512); means[start:end]=delta[rng.integers(0,delta.size,size=(end-start,delta.size),dtype=np.intp)].mean(axis=1)
    lo,hi=np.quantile(means,(.025,.975))
    return MappingProxyType({"mean_delta":float(delta.mean()),"ci_lower":float(lo),"ci_upper":float(hi),"wins":int(np.count_nonzero(delta>_TIE)),"ties":int(np.count_nonzero(np.abs(delta)<=_TIE)),"losses":int(np.count_nonzero(delta<-_TIE)),"seed":seed,"draw_count":_DRAWS,"percentile":95.0,"tie_tolerance":_TIE})

def _labels(value:object,order:tuple[str,...])->Mapping[str,np.ndarray]:
    if not isinstance(value,Mapping) or tuple(value)!=order: raise ValueError("Exp472 TEST labels differ")
    out={}
    for d in order:
        y=np.asarray(value[d],dtype=np.int8)
        if y.ndim!=1 or not np.isin(y,(0,1)).all(): raise ValueError("Exp472 TEST labels differ")
        out[d]=y
    return MappingProxyType(out)

def _summary(y_by_d:Mapping[str,np.ndarray],s_by_d:Mapping[str,np.ndarray|None],order:tuple[str,...],subset:tuple[str,...],method:str)->tuple[list[Mapping[str,object]],Mapping[str,object],Mapping[str,object],Mapping[str,object],dict[str,float]]:
    rows=[]; values={}; metric_values={"average_precision":[],"auroc":[]}; joined_y=[];joined_s=[]
    for d in order:
        y=y_by_d[d]; s=s_by_d[d]; row={"method":method,"dataset":d,"average_precision":None,"average_precision_exclusion":None,"auroc":None,"auroc_exclusion":None}
        if s is None: row.update(average_precision_exclusion="typed_abstention",auroc_exclusion="typed_abstention")
        else:
            if s.shape!=y.shape: raise ValueError("Exp472 score TEST identity differs")
            joined_y.append(y);joined_s.append(s)
            if y.sum(): row["average_precision"]=float(average_precision_score(y,s));metric_values["average_precision"].append(row["average_precision"]);values[(d,"average_precision")]=row["average_precision"]
            else: row["average_precision_exclusion"]="no_positive_test_labels"
            if np.unique(y).size==2: row["auroc"]=float(roc_auc_score(y,s));metric_values["auroc"].append(row["auroc"]);values[(d,"auroc")]=row["auroc"]
            else: row["auroc_exclusion"]="single_class_test_labels"
        rows.append(MappingProxyType(row))
    macro=MappingProxyType({m:(float(np.mean(metric_values[m])) if metric_values[m] else None) for m in metric_values}|{f"{m}_denominator":len(metric_values[m]) for m in metric_values}|{f"{m}_excluded":len(order)-len(metric_values[m]) for m in metric_values})
    if joined_y and np.unique(np.concatenate(joined_y)).size==2:
        yy=np.concatenate(joined_y);ss=np.concatenate(joined_s); micro=MappingProxyType({"average_precision":float(average_precision_score(yy,ss)),"auroc":float(roc_auc_score(yy,ss)),"dataset_denominator":len(joined_y),"observation_denominator":int(yy.size)})
    else: micro=MappingProxyType({"average_precision":None,"auroc":None,"dataset_denominator":len(joined_y),"observation_denominator":sum(x.size for x in joined_y)})
    sub_values={m:[values[(d,m)] for d in subset if (d,m) in values] for m in metric_values}
    subset_summary=MappingProxyType({f"{m}_denominator":len(v) for m,v in sub_values.items()}|{f"{m}_macro":float(np.mean(v)) if v else None for m,v in sub_values.items()})
    return rows,macro,micro,subset_summary,values

def _curves(labels:Mapping[str,np.ndarray],scores:Mapping[str,Mapping[str,np.ndarray|None]],order:tuple[str,...])->Mapping[str,Mapping[str,object]]:
    roc_rows=[];pr_rows=[]
    for method in METHOD_ORDER_EXP472_ANTIGRAVITY:
        present=[d for d in order if scores[method][d] is not None]
        if not present: continue
        y=np.concatenate([labels[d] for d in present]);s=np.concatenate([scores[method][d] for d in present])
        if np.unique(y).size!=2: continue
        fpr,tpr,th=roc_curve(y,s);precision,recall,_=precision_recall_curve(y,s)
        if not(np.all((0<=fpr)&(fpr<=1)) and np.all((0<=tpr)&(tpr<=1)) and np.all(np.diff(fpr)>=0) and np.all(np.diff(tpr)>=0) and (fpr[0],tpr[0])==(0,0) and (fpr[-1],tpr[-1])==(1,1)): raise ValueError("Exp472 ROC table differs")
        if not(np.all((0<=recall)&(recall<=1)) and np.all((0<=precision)&(precision<=1)) and np.all(np.diff(recall)<=0)): raise ValueError("Exp472 PR table differs")
        roc_rows.extend(MappingProxyType({"method":method,"false_positive_rate":float(x),"true_positive_rate":float(z),"threshold":float(t)}) for x,z,t in zip(fpr,tpr,th,strict=True));pr_rows.extend(MappingProxyType({"method":method,"recall":float(x),"precision":float(z)}) for x,z in zip(recall,precision,strict=True))
    return MappingProxyType({"roc":MappingProxyType({"x_name":"false_positive_rate","y_name":"true_positive_rate","rows":tuple(roc_rows)}),"pr":MappingProxyType({"x_name":"recall","y_name":"precision","rows":tuple(pr_rows)})})

@dataclass(frozen=True,slots=True)
class Exp472EvaluationAntigravity:
    authority:Exp472ReuseAuthorityAntigravity; method_order:tuple[str,...]; per_dataset:tuple[Mapping[str,object],...]; macro:Mapping[str,Mapping[str,object]]; micro:Mapping[str,Mapping[str,object]]; paired_statistics:tuple[Mapping[str,object],...]; denominators:Mapping[str,Mapping[str,object]]; curves:Mapping[str,Mapping[str,object]]
    def artifact_record(self)->dict[str,object]:
        bare={"schema":"exp472-evaluation-antigravity.v1","authority":self.authority.__dict__ if hasattr(self.authority,"__dict__") else {name:getattr(self.authority,name) for name in self.authority.__dataclass_fields__},"method_order":list(self.method_order),"denominators":{k:dict(v) for k,v in self.denominators.items()},"paired_statistics":[dict(x) for x in self.paired_statistics]}
        return {**bare,"evaluation_sha256":sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()}

def evaluate_exp472_antigravity(e_root:Path,exp471_root:Path,label_loader:Callable[[tuple[str,...]],Mapping[str,Sequence[object]]])->Exp472EvaluationAntigravity:
    if not isinstance(e_root,Path) or not isinstance(exp471_root,Path) or not callable(label_loader): raise ValueError("Exp472 evaluation requires Path authorities and label loader")
    exp471=verify_exp471_reuse_exp472_antigravity(exp471_root); e=_read_verified_e_path_antigravity(e_root,exp471); order=bind_e_to_exp471_exp472_antigravity(e,exp471)
    e_scores={d:_vector(e.scores[d],"E scores") for d in order}; subset=tuple(d for d in order if e.upsampled_rows[d]>0)
    scores={m:{d:None if exp471.statuses[m][d]=="abstained" else _vector(exp471.scores[m][d],"D scores") for d in order} for m in METHOD_ORDER_EXP472_ANTIGRAVITY if m!="E"};scores["E"]=e_scores
    labels=_labels(label_loader(order),order)
    rows=[];macro={};micro={};value_by_method={};subset_audit={}
    for m in METHOD_ORDER_EXP472_ANTIGRAVITY:
        r,ma,mi,sub,v=_summary(labels,scores[m],order,subset,m);rows.extend(r);macro[m]=ma;micro[m]=mi;value_by_method[m]=v;subset_audit[m]=sub
    paired=[]
    for metric in ("average_precision","auroc"):
        common=tuple(d for d in order if (d,metric) in value_by_method["E"] and (d,metric) in value_by_method["D"]); stat=paired_e_minus_d_exp472_antigravity([value_by_method["E"][(d,metric)] for d in common],[value_by_method["D"][(d,metric)] for d in common]) if common else {"mean_delta":None,"ci_lower":None,"ci_upper":None,"wins":0,"ties":0,"losses":0,"seed":_SEED,"draw_count":0,"percentile":95.0,"tie_tolerance":_TIE};paired.append(MappingProxyType({"comparison":"E-D","metric":metric,"dataset_order":common,**stat}))
    denominators=MappingProxyType({"full_population":MappingProxyType({"dataset_count":len(order),"observation_count":sum(x.size for x in labels.values())}),"upsampled_rows_gt_zero":MappingProxyType({"dataset_count":len(subset),"dataset_order":subset,"metrics":subset_audit})})
    return Exp472EvaluationAntigravity(exp471.authority,METHOD_ORDER_EXP472_ANTIGRAVITY,tuple(rows),MappingProxyType(macro),MappingProxyType(micro),tuple(paired),denominators,_curves(labels,scores,order))

__all__=["Exp472EvaluationAntigravity","Exp472VerifiedEBundleAntigravity","evaluate_exp472_antigravity","paired_e_minus_d_exp472_antigravity"]
