"""Path-only, one-label-read Exp472 E-versus-D evaluation."""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from .authority_antigravity import Exp472ReuseAuthorityAntigravity, Exp472VerifiedExp471BundleAntigravity, bind_e_to_exp471_exp472_antigravity, verify_exp471_reuse_exp472_antigravity, _coordinate
from .contracts_antigravity import Exp472RunContextAntigravity, canonical_json_bytes_exp472_antigravity
from .artifacts_antigravity import Exp472CatalogEntryAntigravity, validate_diagnostic_exp472_antigravity, verify_manifest_exp472_antigravity, verify_score_shard_exp472_antigravity

METHOD_ORDER_EXP472_ANTIGRAVITY=("A","B","C","D","E","knn_distance","lof_novelty","ocsvm_rbf","isolation_forest","deep_svdd_1d","drocc_1d")
_DRAWS=20_000; _SEED=47220260828; _TIE=1e-12
EVALUATION_NAME_ANTIGRAVITY="exp472_evaluation_antigravity.json"
_EVALUATION_SCHEMA_ANTIGRAVITY="exp472-evaluation-antigravity.v2"

def _vector(value: object, name: str) -> np.ndarray:
    result=np.asarray(value,dtype=np.float64)
    if result.ndim!=1 or result.size<1 or not np.isfinite(result).all(): raise ValueError(f"Exp472 {name} differs")
    return result

@dataclass(frozen=True,slots=True)
class Exp472VerifiedEBundleAntigravity:
    root: Path; database_sha256: str; dataset_order: tuple[str,...]; coordinates: Mapping[str,Mapping[str,object]]; scores: Mapping[str,Sequence[object]]; upsampled_rows: Mapping[str,int]; makima_row_invocation_count: Mapping[str,int]; diagnostic_sha256: Mapping[str,str]; path_sha256: str
    def __post_init__(self)->None:
        if not isinstance(self.root,Path) or not self.root.is_absolute() or len(self.dataset_order)!=1117 or len(set(self.dataset_order))!=1117: raise ValueError("Exp472 E inventory differs")
        if tuple(self.coordinates)!=self.dataset_order or tuple(self.scores)!=self.dataset_order or tuple(self.upsampled_rows)!=self.dataset_order or tuple(self.makima_row_invocation_count)!=self.dataset_order or tuple(self.diagnostic_sha256)!=self.dataset_order: raise ValueError("Exp472 E coverage differs")
        coordinates={};scores={};upsampled={};makima={};diagnostic_sha={}
        for d in self.dataset_order:
            coordinates[d]=_coordinate(self.coordinates[d]); values=_vector(self.scores[d],"E scores")
            if values.size != self.coordinates[d]["count"] or type(self.upsampled_rows[d]) is not int or self.upsampled_rows[d]<0 or type(self.makima_row_invocation_count[d]) is not int or self.makima_row_invocation_count[d] < self.upsampled_rows[d] or type(self.diagnostic_sha256[d]) is not str or len(self.diagnostic_sha256[d]) != 64 or any(character not in "0123456789abcdef" for character in self.diagnostic_sha256[d]): raise ValueError("Exp472 E shard differs")
            scores[d]=tuple(float(value) for value in values);upsampled[d]=self.upsampled_rows[d];makima[d]=self.makima_row_invocation_count[d];diagnostic_sha[d]=self.diagnostic_sha256[d]
        object.__setattr__(self,"coordinates",MappingProxyType(coordinates));object.__setattr__(self,"scores",MappingProxyType(scores));object.__setattr__(self,"upsampled_rows",MappingProxyType(upsampled));object.__setattr__(self,"makima_row_invocation_count",MappingProxyType(makima));object.__setattr__(self,"diagnostic_sha256",MappingProxyType(diagnostic_sha))

def _read_verified_e_path_antigravity(root: Path, exp471: Exp472VerifiedExp471BundleAntigravity, context: Exp472RunContextAntigravity, catalog: Sequence[Exp472CatalogEntryAntigravity])->Exp472VerifiedEBundleAntigravity:
    """Reopen Task 4's terminal manifest and every bound E shard before labels."""
    if not isinstance(root,Path) or not isinstance(context,Exp472RunContextAntigravity): raise ValueError("Exp472 E Path context differs")
    entries=tuple(catalog)
    if len(entries)!=1117 or tuple(entry.dataset for entry in entries)!=exp471.dataset_order: raise ValueError("Exp472 E catalog order differs")
    manifest=verify_manifest_exp472_antigravity(root,context,entries)
    if manifest["database_sha256"]!=exp471.authority.database_sha256 or manifest.get("catalog_sha256")!=context.catalog_sha256 or tuple(manifest["dataset_order"])!=exp471.dataset_order: raise ValueError("Exp472 E manifest identity differs")
    coordinates={};scores={};upsampled={};makima={};diagnostic_sha={}
    for entry,descriptor in zip(entries,manifest["descriptors"],strict=True):
        shard=verify_score_shard_exp472_antigravity(root,context,entry)
        if descriptor.get("dataset")!=entry.dataset or descriptor.get("shard_sha256")!=shard.get("shard_sha256"): raise ValueError("Exp472 E manifest shard differs")
        diagnostic=shard.get("diagnostic")
        try:
            validate_diagnostic_exp472_antigravity(diagnostic,dataset=entry.dataset,upsampled_rows=shard["upsampled_rows"])
        except (TypeError, ValueError) as error:
            raise ValueError("Exp472 E diagnostic differs") from error
        coordinates[entry.dataset]=_coordinate(shard["test_identity"]);scores[entry.dataset]=tuple(shard["scores"]);upsampled[entry.dataset]=diagnostic["upsampled_rows"];makima[entry.dataset]=diagnostic["makima_row_invocation_count"];diagnostic_sha[entry.dataset]=diagnostic["diagnostic_sha256"]
    return Exp472VerifiedEBundleAntigravity(root,context.database_sha256,exp471.dataset_order,coordinates,scores,upsampled,makima,diagnostic_sha,sha256(canonical_json_bytes_exp472_antigravity(manifest)).hexdigest())

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
        roc_rows.extend(MappingProxyType({"method":method,"false_positive_rate":float(x),"true_positive_rate":float(z),"threshold":float(t) if np.isfinite(t) else None}) for x,z,t in zip(fpr,tpr,th,strict=True));pr_rows.extend(MappingProxyType({"method":method,"recall":float(x),"precision":float(z)}) for x,z in zip(recall,precision,strict=True))
    return MappingProxyType({"roc":MappingProxyType({"x_name":"false_positive_rate","y_name":"true_positive_rate","rows":tuple(roc_rows)}),"pr":MappingProxyType({"x_name":"recall","y_name":"precision","rows":tuple(pr_rows)})})

@dataclass(frozen=True,slots=True)
class Exp472EvaluationAntigravity:
    authority:Exp472ReuseAuthorityAntigravity; method_order:tuple[str,...]; per_dataset:tuple[Mapping[str,object],...]; subset_per_dataset:tuple[Mapping[str,object],...]; macro:Mapping[str,Mapping[str,object]]; micro:Mapping[str,Mapping[str,object]]; subset_macro:Mapping[str,Mapping[str,object]]; subset_micro:Mapping[str,Mapping[str,object]]; paired_statistics:tuple[Mapping[str,object],...]; denominators:Mapping[str,Mapping[str,object]]; diagnostic_summary:Mapping[str,Mapping[str,object]]; curves:Mapping[str,Mapping[str,object]]; e_manifest_sha256:str; context_sha256:str
    def artifact_record(self)->dict[str,object]:
        authority=self.authority.__dict__ if hasattr(self.authority,"__dict__") else {name:getattr(self.authority,name) for name in self.authority.__dataclass_fields__}
        bare={"schema":_EVALUATION_SCHEMA_ANTIGRAVITY,"authority":json.loads(canonical_json_bytes_exp472_antigravity(authority).decode("utf-8")),"method_order":list(self.method_order),"per_dataset":[dict(x) for x in self.per_dataset],"subset_per_dataset":[dict(x) for x in self.subset_per_dataset],"macro":{k:dict(v) for k,v in self.macro.items()},"micro":{k:dict(v) for k,v in self.micro.items()},"subset_macro":{k:dict(v) for k,v in self.subset_macro.items()},"subset_micro":{k:dict(v) for k,v in self.subset_micro.items()},"denominators":{k:dict(v) for k,v in self.denominators.items()},"diagnostic_summary":{k:dict(v) for k,v in self.diagnostic_summary.items()},"paired_statistics":[dict(x) for x in self.paired_statistics],"curves":{k:{**{a:b for a,b in v.items() if a!="rows"},"rows":[dict(row) for row in v["rows"]]} for k,v in self.curves.items()},"e_manifest_sha256":self.e_manifest_sha256,"context_sha256":self.context_sha256}
        return {**bare,"evaluation_sha256":sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()}

def evaluate_exp472_antigravity(e_root:Path,exp471_root:Path,label_loader:Callable[[tuple[str,...]],Mapping[str,Sequence[object]]], *, context: Exp472RunContextAntigravity, catalog: Sequence[Exp472CatalogEntryAntigravity])->Exp472EvaluationAntigravity:
    if not isinstance(e_root,Path) or not isinstance(exp471_root,Path) or not callable(label_loader): raise ValueError("Exp472 evaluation requires Path authorities and label loader")
    if not isinstance(context,Exp472RunContextAntigravity) or not isinstance(catalog,(tuple,list)) or not all(isinstance(entry,Exp472CatalogEntryAntigravity) for entry in catalog): raise ValueError("Exp472 evaluation requires typed E context and catalog")
    exp471=verify_exp471_reuse_exp472_antigravity(exp471_root); e=_read_verified_e_path_antigravity(e_root,exp471,context,catalog); order=bind_e_to_exp471_exp472_antigravity(e,exp471)
    e_scores={d:_vector(e.scores[d],"E scores") for d in order}; subset=tuple(d for d in order if e.upsampled_rows[d]>0)
    scores={m:{d:None if exp471.statuses[m][d]=="abstained" else _vector(exp471.scores[m][d],"D scores") for d in order} for m in METHOD_ORDER_EXP472_ANTIGRAVITY if m!="E"};scores["E"]=e_scores
    labels=_labels(label_loader(order),order)
    rows=[];subset_rows=[];macro={};micro={};subset_macro={};subset_micro={};value_by_method={};subset_value_by_method={}
    for m in METHOD_ORDER_EXP472_ANTIGRAVITY:
        r,ma,mi,_unused,v=_summary(labels,scores[m],order,subset,m);sr,sma,smi,_unused,sv=_summary(labels,scores[m],subset,(),m);rows.extend(r);subset_rows.extend(sr);macro[m]=ma;micro[m]=mi;subset_macro[m]=sma;subset_micro[m]=smi;value_by_method[m]=v;subset_value_by_method[m]=sv
    paired=[]
    for population,population_order,values in (("full",order,value_by_method),("upsampled_rows_gt_zero",subset,subset_value_by_method)):
        for metric in ("average_precision","auroc"):
            common=tuple(d for d in population_order if (d,metric) in values["E"] and (d,metric) in values["D"]); stat=paired_e_minus_d_exp472_antigravity([values["E"][(d,metric)] for d in common],[values["D"][(d,metric)] for d in common]) if common else {"mean_delta":None,"ci_lower":None,"ci_upper":None,"wins":0,"ties":0,"losses":0,"seed":_SEED,"draw_count":0,"percentile":95.0,"tie_tolerance":_TIE};paired.append(MappingProxyType({"comparison":"E-D","population":population,"metric":metric,"dataset_order":common,**stat}))
    denominators=MappingProxyType({"full_population":MappingProxyType({"dataset_count":len(order),"observation_count":sum(x.size for x in labels.values())}),"upsampled_rows_gt_zero":MappingProxyType({"dataset_count":len(subset),"observation_count":sum(labels[d].size for d in subset),"dataset_order":subset})})
    diagnostic_summary=MappingProxyType({d:MappingProxyType({"upsampled_rows":e.upsampled_rows[d],"makima_row_invocation_count":e.makima_row_invocation_count[d],"diagnostic_sha256":e.diagnostic_sha256[d]}) for d in order})
    return Exp472EvaluationAntigravity(exp471.authority,METHOD_ORDER_EXP472_ANTIGRAVITY,tuple(rows),tuple(subset_rows),MappingProxyType(macro),MappingProxyType(micro),MappingProxyType(subset_macro),MappingProxyType(subset_micro),tuple(paired),denominators,diagnostic_summary,_curves(labels,scores,order),e.path_sha256,context.sha256)

def evaluation_path_exp472_antigravity(root: Path) -> Path:
    target=Path(root)
    if not target.is_absolute() or "antigravity" not in target.name:
        raise ValueError("Exp472 evaluation root differs")
    return target / EVALUATION_NAME_ANTIGRAVITY

def _safe_evaluation_root_antigravity(root: Path, *, create: bool) -> Path:
    target=Path(root)
    if not target.is_absolute() or "antigravity" not in target.name:
        raise ValueError("Exp472 evaluation root differs")
    if create: target.mkdir(parents=True,exist_ok=True)
    try: details=target.lstat()
    except FileNotFoundError as error: raise FileNotFoundError("Exp472 evaluation artifact is missing") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode): raise ValueError("Exp472 evaluation root is unsafe")
    return target

def _read_regular_evaluation_antigravity(path: Path) -> bytes:
    try: linked=path.lstat()
    except FileNotFoundError as error: raise FileNotFoundError("Exp472 evaluation artifact is missing") from error
    if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode): raise ValueError("Exp472 evaluation artifact is unsafe")
    try: descriptor=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    except OSError as error: raise ValueError("Exp472 evaluation artifact is unsafe") from error
    try:
        opened=os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_dev!=linked.st_dev or opened.st_ino!=linked.st_ino or opened.st_nlink!=1: raise ValueError("Exp472 evaluation artifact is unsafe")
        chunks=[]
        while True:
            block=os.read(descriptor,1<<20)
            if not block: return b"".join(chunks)
            chunks.append(block)
    finally: os.close(descriptor)

def _write_create_or_identical_evaluation_antigravity(path: Path, payload: bytes) -> None:
    try: descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        if _read_regular_evaluation_antigravity(path)!=payload: raise ValueError("Exp472 evaluation artifact already differs")
        return
    try:
        view=memoryview(payload)
        while view:
            written=os.write(descriptor,view)
            if written<=0: raise OSError("Exp472 evaluation write made no progress")
            view=view[written:]
        os.fsync(descriptor)
    finally: os.close(descriptor)
    directory=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    if _read_regular_evaluation_antigravity(path)!=payload: raise ValueError("Exp472 evaluation artifact changed after publication")

def _digest_evaluation_antigravity(value: object, name: str) -> str:
    if type(value) is not str or len(value)!=64 or any(character not in "0123456789abcdef" for character in value): raise ValueError(f"Exp472 {name} differs")
    return value

def _read_evaluation_record_antigravity(path: Path) -> Mapping[str, object]:
    raw=_read_regular_evaluation_antigravity(path)
    try: record=json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError,json.JSONDecodeError) as error: raise ValueError("Exp472 evaluation artifact is invalid") from error
    if not isinstance(record,dict) or raw!=canonical_json_bytes_exp472_antigravity(record): raise ValueError("Exp472 evaluation artifact is noncanonical")
    expected={"schema","authority","method_order","per_dataset","subset_per_dataset","macro","micro","subset_macro","subset_micro","paired_statistics","denominators","diagnostic_summary","curves","e_manifest_sha256","context_sha256","evaluation_sha256"}
    if set(record)!=expected or record.get("schema")!=_EVALUATION_SCHEMA_ANTIGRAVITY: raise ValueError("Exp472 evaluation artifact schema differs")
    bare={key:value for key,value in record.items() if key!="evaluation_sha256"}
    if _digest_evaluation_antigravity(record["evaluation_sha256"],"evaluation digest")!=sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest(): raise ValueError("Exp472 evaluation artifact digest differs")
    return MappingProxyType(record)

def read_exp472_evaluation_antigravity(root: Path, *, authority: Exp472ReuseAuthorityAntigravity, e_manifest_sha256: str, context_sha256: str, dataset_order: Sequence[str]) -> Exp472EvaluationAntigravity:
    """Reopen only a canonical, authority-bound evaluation; never labels."""
    path=evaluation_path_exp472_antigravity(_safe_evaluation_root_antigravity(root,create=False))
    record=_read_evaluation_record_antigravity(path)
    expected_authority=json.loads(canonical_json_bytes_exp472_antigravity({name:getattr(authority,name) for name in authority.__dataclass_fields__}).decode("utf-8"))
    if record["authority"]!=expected_authority or _digest_evaluation_antigravity(record["e_manifest_sha256"],"E manifest digest")!=e_manifest_sha256 or _digest_evaluation_antigravity(record["context_sha256"],"context digest")!=context_sha256: raise ValueError("Exp472 persisted evaluation authority differs")
    order=tuple(dataset_order)
    summary=record["diagnostic_summary"]
    if tuple(record["method_order"])!=METHOD_ORDER_EXP472_ANTIGRAVITY or not isinstance(summary,dict) or set(summary)!=set(order): raise ValueError("Exp472 persisted evaluation inventory differs")
    for dataset in order:
        item=summary[dataset]
        if not isinstance(item,dict) or set(item)!={"upsampled_rows","makima_row_invocation_count","diagnostic_sha256"} or type(item["upsampled_rows"]) is not int or item["upsampled_rows"]<0 or type(item["makima_row_invocation_count"]) is not int or item["makima_row_invocation_count"]<item["upsampled_rows"]: raise ValueError("Exp472 persisted evaluation diagnostic differs")
        _digest_evaluation_antigravity(item["diagnostic_sha256"],"diagnostic digest")
    try:
        parsed_authority=Exp472ReuseAuthorityAntigravity(**record["authority"])
        return Exp472EvaluationAntigravity(parsed_authority,tuple(record["method_order"]),tuple(record["per_dataset"]),tuple(record["subset_per_dataset"]),MappingProxyType(record["macro"]),MappingProxyType(record["micro"]),MappingProxyType(record["subset_macro"]),MappingProxyType(record["subset_micro"]),tuple(record["paired_statistics"]),MappingProxyType(record["denominators"]),MappingProxyType(summary),MappingProxyType(record["curves"]),record["e_manifest_sha256"],record["context_sha256"])
    except (TypeError,ValueError) as error: raise ValueError("Exp472 persisted evaluation differs") from error

def write_exp472_evaluation_antigravity(evaluation: Exp472EvaluationAntigravity, root: Path) -> Exp472EvaluationAntigravity:
    """Create one canonical evaluation record, or reuse exactly its bytes."""
    if not isinstance(evaluation,Exp472EvaluationAntigravity): raise ValueError("Exp472 evaluation publication requires typed evaluation")
    target=_safe_evaluation_root_antigravity(root,create=True); path=evaluation_path_exp472_antigravity(target); payload=canonical_json_bytes_exp472_antigravity(evaluation.artifact_record())
    _write_create_or_identical_evaluation_antigravity(path,payload)
    return read_exp472_evaluation_antigravity(target,authority=evaluation.authority,e_manifest_sha256=evaluation.e_manifest_sha256,context_sha256=evaluation.context_sha256,dataset_order=tuple(evaluation.diagnostic_summary))

__all__=["EVALUATION_NAME_ANTIGRAVITY","Exp472EvaluationAntigravity","Exp472VerifiedEBundleAntigravity","evaluate_exp472_antigravity","evaluation_path_exp472_antigravity","paired_e_minus_d_exp472_antigravity","read_exp472_evaluation_antigravity","write_exp472_evaluation_antigravity"]
