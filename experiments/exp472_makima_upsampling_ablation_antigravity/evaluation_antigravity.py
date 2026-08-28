"""Replay sealed Exp471/E scores plus private labels into capability-gated Exp472 evaluation."""
from __future__ import annotations
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import fcntl
from types import MappingProxyType
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from .authority_antigravity import Exp472ReuseAuthorityAntigravity, Exp472VerifiedExp471BundleAntigravity, bind_e_to_exp471_exp472_antigravity, verify_exp471_reuse_exp472_antigravity, _coordinate
from .contracts_antigravity import Exp472RunContextAntigravity, canonical_json_bytes_exp472_antigravity
from .artifacts_antigravity import Exp472CatalogEntryAntigravity, validate_diagnostic_exp472_antigravity, verify_manifest_exp472_antigravity, verify_score_shard_exp472_antigravity

METHOD_ORDER_EXP472_ANTIGRAVITY=("A","B","C","D","E","knn_distance","lof_novelty","ocsvm_rbf","isolation_forest","deep_svdd_1d","drocc_1d")
_DRAWS=20_000; _SEED=47220260828; _TIE=1e-12
EVALUATION_NAME_ANTIGRAVITY="exp472_evaluation_antigravity.json"
EVALUATION_COMPLETE_NAME_ANTIGRAVITY=".exp472_evaluation_complete_antigravity.json"
PRIVATE_LABEL_EVIDENCE_NAME_ANTIGRAVITY="exp472_private_label_evidence_antigravity.json"
_EVALUATION_LOCK_NAME_ANTIGRAVITY=".exp472_evaluation_lock_antigravity"
_EVALUATION_SCHEMA_ANTIGRAVITY="exp472-evaluation-antigravity.v3"
_EVALUATION_COMPLETE_SCHEMA_ANTIGRAVITY="exp472-evaluation-complete-antigravity.v1"
_PRIVATE_LABEL_EVIDENCE_SCHEMA_ANTIGRAVITY="exp472-private-label-evidence-antigravity.v1"

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
    return Exp472VerifiedEBundleAntigravity(root,context.database_sha256,exp471.dataset_order,coordinates,scores,upsampled,makima,diagnostic_sha,manifest["manifest_sha256"])

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
        raw=np.asarray(value[d],dtype=np.float64)
        if raw.ndim!=1 or raw.size<1 or not np.isfinite(raw).all() or not np.isin(raw,(0.0,1.0)).all(): raise ValueError("Exp472 TEST labels differ")
        out[d]=raw.astype(np.int8,copy=False)
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
    authority:Exp472ReuseAuthorityAntigravity; method_order:tuple[str,...]; per_dataset:tuple[Mapping[str,object],...]; subset_per_dataset:tuple[Mapping[str,object],...]; macro:Mapping[str,Mapping[str,object]]; micro:Mapping[str,Mapping[str,object]]; subset_macro:Mapping[str,Mapping[str,object]]; subset_micro:Mapping[str,Mapping[str,object]]; paired_statistics:tuple[Mapping[str,object],...]; denominators:Mapping[str,Mapping[str,object]]; diagnostic_summary:Mapping[str,Mapping[str,object]]; curves:Mapping[str,Mapping[str,object]]; private_labels:Mapping[str,Sequence[object]]; e_manifest_sha256:str; context_sha256:str
    def artifact_record(self)->dict[str,object]:
        authority=self.authority.__dict__ if hasattr(self.authority,"__dict__") else {name:getattr(self.authority,name) for name in self.authority.__dataclass_fields__}
        bare={"schema":_EVALUATION_SCHEMA_ANTIGRAVITY,"authority":json.loads(canonical_json_bytes_exp472_antigravity(authority).decode("utf-8")),"method_order":list(self.method_order),"per_dataset":[dict(x) for x in self.per_dataset],"subset_per_dataset":[dict(x) for x in self.subset_per_dataset],"macro":{k:dict(v) for k,v in self.macro.items()},"micro":{k:dict(v) for k,v in self.micro.items()},"subset_macro":{k:dict(v) for k,v in self.subset_macro.items()},"subset_micro":{k:dict(v) for k,v in self.subset_micro.items()},"denominators":{k:dict(v) for k,v in self.denominators.items()},"diagnostic_summary":{k:dict(v) for k,v in self.diagnostic_summary.items()},"paired_statistics":[dict(x) for x in self.paired_statistics],"curves":{k:{**{a:b for a,b in v.items() if a!="rows"},"rows":[dict(row) for row in v["rows"]]} for k,v in self.curves.items()},"private_label_evidence_sha256":_private_label_evidence_record_antigravity(self)["private_label_evidence_sha256"],"e_manifest_sha256":self.e_manifest_sha256,"context_sha256":self.context_sha256}
        canonical_bare=json.loads(canonical_json_bytes_exp472_antigravity(bare).decode("utf-8"))
        return {**canonical_bare,"evaluation_sha256":sha256(canonical_json_bytes_exp472_antigravity(canonical_bare)).hexdigest()}

_VERIFIED_PERSISTED_TOKEN_ANTIGRAVITY=object()

class _VerifiedPersistedEvaluationAntigravity:
    """Python-internal capability issued only after trusted terminal-score replay."""
    __slots__=("_record_bytes","_evaluation_sha256","_authority_sha256","_context_sha256","_private_label_evidence_sha256","_e_manifest_sha256","_token","_sealed")
    def __init__(self,evaluation: Exp472EvaluationAntigravity,token: object)->None:
        if token is not _VERIFIED_PERSISTED_TOKEN_ANTIGRAVITY or not isinstance(evaluation,Exp472EvaluationAntigravity): raise ValueError("Exp472 verified persisted evaluation capability differs")
        record=evaluation.artifact_record(); payload=canonical_json_bytes_exp472_antigravity(record)
        object.__setattr__(self,"_record_bytes",payload);object.__setattr__(self,"_evaluation_sha256",record["evaluation_sha256"]);object.__setattr__(self,"_authority_sha256",sha256(canonical_json_bytes_exp472_antigravity(record["authority"])).hexdigest());object.__setattr__(self,"_context_sha256",record["context_sha256"]);object.__setattr__(self,"_private_label_evidence_sha256",record["private_label_evidence_sha256"]);object.__setattr__(self,"_e_manifest_sha256",record["e_manifest_sha256"]);object.__setattr__(self,"_token",token);object.__setattr__(self,"_sealed",True)
    def __setattr__(self,name: str,value: object)->None: raise AttributeError("Exp472 verified persisted evaluation capability is immutable")
    def artifact_record(self)->dict[str,object]: return json.loads(self._record_bytes.decode("utf-8"))

def _verified_persisted_evaluation_antigravity(evaluation: Exp472EvaluationAntigravity)->_VerifiedPersistedEvaluationAntigravity:
    return _VerifiedPersistedEvaluationAntigravity(evaluation,_VERIFIED_PERSISTED_TOKEN_ANTIGRAVITY)

def _require_verified_persisted_evaluation_antigravity(value: object) -> Mapping[str,object]:
    """Unwrap the non-public report capability issued by the trusted reader.

    This is a Python module-boundary capability, not a replacement for an
    external signature or WORM store.  It rejects normal public objects and
    look-alikes; code with arbitrary access to this private module remains out
    of scope for this in-process boundary.
    """
    if type(value) is not _VerifiedPersistedEvaluationAntigravity or value._token is not _VERIFIED_PERSISTED_TOKEN_ANTIGRAVITY:
        raise ValueError("Exp472 reporting requires a trusted persisted evaluation capability")
    try: record=json.loads(value._record_bytes.decode("utf-8"))
    except (AttributeError,UnicodeDecodeError,json.JSONDecodeError) as error: raise ValueError("Exp472 verified persisted evaluation capability differs") from error
    if canonical_json_bytes_exp472_antigravity(record)!=value._record_bytes or record.get("evaluation_sha256")!=value._evaluation_sha256 or _digest_evaluation_antigravity(value._evaluation_sha256,"evaluation digest")!=sha256(canonical_json_bytes_exp472_antigravity({key:item for key,item in record.items() if key!="evaluation_sha256"})).hexdigest() or record.get("e_manifest_sha256")!=value._e_manifest_sha256 or record.get("context_sha256")!=value._context_sha256 or record.get("private_label_evidence_sha256")!=value._private_label_evidence_sha256 or sha256(canonical_json_bytes_exp472_antigravity(record.get("authority"))).hexdigest()!=value._authority_sha256: raise ValueError("Exp472 verified persisted evaluation capability differs")
    return MappingProxyType(record)

def evaluate_exp472_antigravity(e_root:Path,exp471_root:Path,label_loader:Callable[[tuple[str,...]],Mapping[str,Sequence[object]]], *, context: Exp472RunContextAntigravity, catalog: Sequence[Exp472CatalogEntryAntigravity])->Exp472EvaluationAntigravity:
    if not isinstance(e_root,Path) or not isinstance(exp471_root,Path) or not callable(label_loader): raise ValueError("Exp472 evaluation requires Path authorities and label loader")
    if not isinstance(context,Exp472RunContextAntigravity) or not isinstance(catalog,(tuple,list)) or not all(isinstance(entry,Exp472CatalogEntryAntigravity) for entry in catalog): raise ValueError("Exp472 evaluation requires typed E context and catalog")
    exp471=verify_exp471_reuse_exp472_antigravity(exp471_root); e=_read_verified_e_path_antigravity(e_root,exp471,context,catalog); order=bind_e_to_exp471_exp472_antigravity(e,exp471)
    return _evaluate_from_verified_antigravity(exp471,e,_labels(label_loader(order),order),context)

def _evaluate_from_verified_antigravity(exp471: Exp472VerifiedExp471BundleAntigravity, e: Exp472VerifiedEBundleAntigravity, labels: Mapping[str,np.ndarray], context: Exp472RunContextAntigravity) -> Exp472EvaluationAntigravity:
    """Recompute every public scientific section from sealed terminal scores and labels."""
    order=bind_e_to_exp471_exp472_antigravity(e,exp471)
    labels=_labels(labels,order)
    e_scores={d:_vector(e.scores[d],"E scores") for d in order}; subset=tuple(d for d in order if e.upsampled_rows[d]>0)
    scores={m:{d:None if exp471.statuses[m][d]=="abstained" else _vector(exp471.scores[m][d],"D scores") for d in order} for m in METHOD_ORDER_EXP472_ANTIGRAVITY if m!="E"};scores["E"]=e_scores
    rows=[];subset_rows=[];macro={};micro={};subset_macro={};subset_micro={};value_by_method={};subset_value_by_method={}
    for m in METHOD_ORDER_EXP472_ANTIGRAVITY:
        r,ma,mi,_unused,v=_summary(labels,scores[m],order,subset,m);sr,sma,smi,_unused,sv=_summary(labels,scores[m],subset,(),m);rows.extend(r);subset_rows.extend(sr);macro[m]=ma;micro[m]=mi;subset_macro[m]=sma;subset_micro[m]=smi;value_by_method[m]=v;subset_value_by_method[m]=sv
    paired=[]
    for population,population_order,values in (("full",order,value_by_method),("upsampled_rows_gt_zero",subset,subset_value_by_method)):
        for metric in ("average_precision","auroc"):
            common=tuple(d for d in population_order if (d,metric) in values["E"] and (d,metric) in values["D"]); stat=paired_e_minus_d_exp472_antigravity([values["E"][(d,metric)] for d in common],[values["D"][(d,metric)] for d in common]) if common else {"mean_delta":None,"ci_lower":None,"ci_upper":None,"wins":0,"ties":0,"losses":0,"seed":_SEED,"draw_count":0,"percentile":95.0,"tie_tolerance":_TIE};paired.append(MappingProxyType({"comparison":"E-D","population":population,"metric":metric,"dataset_order":common,**stat}))
    denominators=MappingProxyType({"full_population":MappingProxyType({"dataset_count":len(order),"observation_count":sum(x.size for x in labels.values())}),"upsampled_rows_gt_zero":MappingProxyType({"dataset_count":len(subset),"observation_count":sum(labels[d].size for d in subset),"dataset_order":subset})})
    diagnostic_summary=MappingProxyType({d:MappingProxyType({"upsampled_rows":e.upsampled_rows[d],"makima_row_invocation_count":e.makima_row_invocation_count[d],"diagnostic_sha256":e.diagnostic_sha256[d]}) for d in order})
    return Exp472EvaluationAntigravity(exp471.authority,METHOD_ORDER_EXP472_ANTIGRAVITY,tuple(rows),tuple(subset_rows),MappingProxyType(macro),MappingProxyType(micro),MappingProxyType(subset_macro),MappingProxyType(subset_micro),tuple(paired),denominators,diagnostic_summary,_curves(labels,scores,order),labels,e.path_sha256,context.sha256)

def evaluation_path_exp472_antigravity(root: Path) -> Path:
    target=Path(root)
    if not target.is_absolute() or "antigravity" not in target.name:
        raise ValueError("Exp472 evaluation root differs")
    return target / EVALUATION_NAME_ANTIGRAVITY

def _evaluation_marker_path_antigravity(root: Path) -> Path:
    return Path(root) / EVALUATION_COMPLETE_NAME_ANTIGRAVITY

def _private_label_evidence_path_antigravity(root: Path) -> Path:
    return Path(root) / PRIVATE_LABEL_EVIDENCE_NAME_ANTIGRAVITY

def _authority_record_antigravity(authority: Exp472ReuseAuthorityAntigravity) -> dict[str, object]:
    return json.loads(canonical_json_bytes_exp472_antigravity({name:getattr(authority,name) for name in authority.__dataclass_fields__}).decode("utf-8"))

def _private_label_evidence_record_antigravity(evaluation: Exp472EvaluationAntigravity) -> dict[str, object]:
    order=tuple(evaluation.private_labels)
    labels=_labels(evaluation.private_labels,order)
    bare={"schema":_PRIVATE_LABEL_EVIDENCE_SCHEMA_ANTIGRAVITY,"authority":_authority_record_antigravity(evaluation.authority),"e_manifest_sha256":evaluation.e_manifest_sha256,"context_sha256":evaluation.context_sha256,"dataset_order":list(order),"labels":[{"dataset":dataset,"values":[int(value) for value in labels[dataset]]} for dataset in order]}
    canonical=json.loads(canonical_json_bytes_exp472_antigravity(bare).decode("utf-8"))
    return {**canonical,"private_label_evidence_sha256":sha256(canonical_json_bytes_exp472_antigravity(canonical)).hexdigest()}

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

@contextmanager
def _evaluation_claim_antigravity(root: Path) -> Iterator[None]:
    """Serialize the one label-opening writer; flock releases after a crash."""
    lock=root / _EVALUATION_LOCK_NAME_ANTIGRAVITY
    flags=os.O_CREAT|os.O_RDWR|getattr(os,"O_NOFOLLOW",0)
    try: descriptor=os.open(lock,flags,0o600)
    except OSError as error: raise ValueError("Exp472 evaluation lock is unsafe") from error
    try:
        details=os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink!=1: raise ValueError("Exp472 evaluation lock is unsafe")
        fcntl.flock(descriptor,fcntl.LOCK_EX)
        yield
    finally:
        try: fcntl.flock(descriptor,fcntl.LOCK_UN)
        finally: os.close(descriptor)

def _write_create_or_identical_evaluation_antigravity(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    descriptor, temporary=tempfile.mkstemp(prefix=".exp472_evaluation_publish_antigravity_",dir=path.parent)
    try:
        os.fchmod(descriptor,mode)
        view=memoryview(payload)
        while view:
            written=os.write(descriptor,view)
            if written<=0: raise OSError("Exp472 evaluation write made no progress")
            view=view[written:]
        os.fsync(descriptor)
    finally: os.close(descriptor)
    try:
        try: os.link(temporary,path)
        except FileExistsError:
            if _read_regular_evaluation_antigravity(path)!=payload: raise ValueError("Exp472 evaluation artifact already differs")
        directory=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
    if _read_regular_evaluation_antigravity(path)!=payload: raise ValueError("Exp472 evaluation artifact changed after publication")

def _digest_evaluation_antigravity(value: object, name: str) -> str:
    if type(value) is not str or len(value)!=64 or any(character not in "0123456789abcdef" for character in value): raise ValueError(f"Exp472 {name} differs")
    return value

def _read_evaluation_record_antigravity(path: Path) -> Mapping[str, object]:
    raw=_read_regular_evaluation_antigravity(path)
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object]={}
        for key, value in pairs:
            if key in result: raise ValueError("Exp472 evaluation artifact has duplicate JSON keys")
            result[key]=value
        return result
    try: record=json.loads(raw.decode("utf-8"),object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError,json.JSONDecodeError,ValueError) as error: raise ValueError("Exp472 evaluation artifact is invalid") from error
    if not isinstance(record,dict) or raw!=canonical_json_bytes_exp472_antigravity(record): raise ValueError("Exp472 evaluation artifact is noncanonical")
    expected={"schema","authority","method_order","per_dataset","subset_per_dataset","macro","micro","subset_macro","subset_micro","paired_statistics","denominators","diagnostic_summary","curves","private_label_evidence_sha256","e_manifest_sha256","context_sha256","evaluation_sha256"}
    if set(record)!=expected or record.get("schema")!=_EVALUATION_SCHEMA_ANTIGRAVITY: raise ValueError("Exp472 evaluation artifact schema differs")
    bare={key:value for key,value in record.items() if key!="evaluation_sha256"}
    if _digest_evaluation_antigravity(record["evaluation_sha256"],"evaluation digest")!=sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest(): raise ValueError("Exp472 evaluation artifact digest differs")
    return MappingProxyType(record)

def _read_private_label_evidence_antigravity(root: Path, *, authority: Exp472ReuseAuthorityAntigravity, e_manifest_sha256: str, context_sha256: str, dataset_order: Sequence[str]) -> tuple[Mapping[str, np.ndarray], str]:
    path=_private_label_evidence_path_antigravity(root)
    try: details=path.lstat()
    except FileNotFoundError as error: raise FileNotFoundError("Exp472 private label evidence is missing") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode) or details.st_mode & 0o077: raise ValueError("Exp472 private label evidence is unsafe")
    raw=_read_regular_evaluation_antigravity(path)
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object]={}
        for key,value in pairs:
            if key in result: raise ValueError("Exp472 private label evidence has duplicate JSON keys")
            result[key]=value
        return result
    try: record=json.loads(raw.decode("utf-8"),object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError,json.JSONDecodeError,ValueError) as error: raise ValueError("Exp472 private label evidence is invalid") from error
    if not isinstance(record,dict) or raw!=canonical_json_bytes_exp472_antigravity(record): raise ValueError("Exp472 private label evidence is noncanonical")
    expected={"schema","authority","e_manifest_sha256","context_sha256","dataset_order","labels","private_label_evidence_sha256"}
    if set(record)!=expected or record.get("schema")!=_PRIVATE_LABEL_EVIDENCE_SCHEMA_ANTIGRAVITY: raise ValueError("Exp472 private label evidence schema differs")
    bare={key:value for key,value in record.items() if key!="private_label_evidence_sha256"}
    if _digest_evaluation_antigravity(record["private_label_evidence_sha256"],"private label evidence digest")!=sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest(): raise ValueError("Exp472 private label evidence digest differs")
    order=tuple(dataset_order)
    if record["authority"]!=_authority_record_antigravity(authority) or record["e_manifest_sha256"]!=e_manifest_sha256 or record["context_sha256"]!=context_sha256 or tuple(record["dataset_order"])!=order or len(order)!=1117 or len(set(order))!=1117 or not isinstance(record["labels"],list) or len(record["labels"])!=len(order): raise ValueError("Exp472 private label evidence authority differs")
    labels: dict[str, Sequence[object]]={}
    for expected_dataset,row in zip(order,record["labels"],strict=True):
        if not isinstance(row,dict) or set(row)!={"dataset","values"} or row["dataset"]!=expected_dataset or expected_dataset in labels or not isinstance(row["values"],list): raise ValueError("Exp472 private label evidence labels differ")
        labels[expected_dataset]=row["values"]
    return _labels(labels,order),str(record["private_label_evidence_sha256"])

def _evaluation_marker_record_antigravity(record: Mapping[str,object]) -> dict[str,object]:
    return {"schema":_EVALUATION_COMPLETE_SCHEMA_ANTIGRAVITY,"evaluation":EVALUATION_NAME_ANTIGRAVITY,"evaluation_sha256":record["evaluation_sha256"],"private_label_evidence":PRIVATE_LABEL_EVIDENCE_NAME_ANTIGRAVITY,"private_label_evidence_sha256":record["private_label_evidence_sha256"],"authority":record["authority"],"e_manifest_sha256":record["e_manifest_sha256"],"context_sha256":record["context_sha256"]}

def _read_evaluation_marker_antigravity(root: Path, record: Mapping[str,object]) -> None:
    raw=_read_regular_evaluation_antigravity(_evaluation_marker_path_antigravity(root))
    try: marker=json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError,json.JSONDecodeError) as error: raise ValueError("Exp472 evaluation completion marker is invalid") from error
    if raw!=canonical_json_bytes_exp472_antigravity(marker) or marker!=_evaluation_marker_record_antigravity(record): raise ValueError("Exp472 evaluation completion marker differs")

def _finite_probability_antigravity(value: object) -> bool:
    return type(value) in (int,float) and math.isfinite(float(value)) and 0.0<=float(value)<=1.0

def _validate_metric_rows_antigravity(rows: object, order: tuple[str,...], methods: tuple[str,...], name: str) -> Mapping[tuple[str,str], Mapping[str,object]]:
    if not isinstance(rows,list) or len(rows)!=len(order)*len(methods): raise ValueError(f"Exp472 persisted {name} inventory differs")
    expected=tuple((method,dataset) for method in methods for dataset in order); found=[]; result={}
    for row in rows:
        if not isinstance(row,dict) or set(row)!={"method","dataset","average_precision","average_precision_exclusion","auroc","auroc_exclusion"}: raise ValueError(f"Exp472 persisted {name} row differs")
        pair=(row["method"],row["dataset"]); found.append(pair)
        for metric in ("average_precision","auroc"):
            score, exclusion=row[metric],row[f"{metric}_exclusion"]
            if score is None:
                if exclusion not in {"typed_abstention","no_positive_test_labels","single_class_test_labels"}: raise ValueError(f"Exp472 persisted {name} exclusion differs")
            elif not _finite_probability_antigravity(score) or exclusion is not None: raise ValueError(f"Exp472 persisted {name} metric differs")
        result[pair]=row
    if tuple(found)!=expected or len(result)!=len(expected): raise ValueError(f"Exp472 persisted {name} coverage differs")
    return MappingProxyType(result)

def _validate_evaluation_science_antigravity(record: Mapping[str,object], order: tuple[str,...]) -> None:
    methods=METHOD_ORDER_EXP472_ANTIGRAVITY
    if len(order)!=1117 or len(set(order))!=1117 or tuple(record["method_order"])!=methods: raise ValueError("Exp472 persisted evaluation inventory differs")
    summary=record["diagnostic_summary"]
    if not isinstance(summary,dict) or set(summary)!=set(order): raise ValueError("Exp472 persisted evaluation diagnostic inventory differs")
    subset=[]
    for dataset in order:
        item=summary[dataset]
        if not isinstance(item,dict) or set(item)!={"upsampled_rows","makima_row_invocation_count","diagnostic_sha256"} or type(item["upsampled_rows"]) is not int or item["upsampled_rows"]<0 or type(item["makima_row_invocation_count"]) is not int or item["makima_row_invocation_count"]<item["upsampled_rows"]: raise ValueError("Exp472 persisted evaluation diagnostic differs")
        _digest_evaluation_antigravity(item["diagnostic_sha256"],"diagnostic digest")
        if item["upsampled_rows"]>0: subset.append(dataset)
    subset_order=tuple(subset); full=_validate_metric_rows_antigravity(record["per_dataset"],order,methods,"per_dataset"); sub=_validate_metric_rows_antigravity(record["subset_per_dataset"],subset_order,methods,"subset_per_dataset")
    for key, rows, expected_order in (("macro",full,order),("subset_macro",sub,subset_order)):
        table=record[key]
        if not isinstance(table,dict) or set(table)!=set(methods): raise ValueError(f"Exp472 persisted {key} differs")
        for method in methods:
            value=table[method]
            required={"average_precision","auroc","average_precision_denominator","auroc_denominator","average_precision_excluded","auroc_excluded"}
            if not isinstance(value,dict) or set(value)!=required: raise ValueError(f"Exp472 persisted {key} row differs")
            for metric in ("average_precision","auroc"):
                scores=[float(rows[(method,d)][metric]) for d in expected_order if rows[(method,d)][metric] is not None]
                if value[f"{metric}_denominator"]!=len(scores) or (value[metric] is None)!=(not scores) or (scores and (not _finite_probability_antigravity(value[metric]) or not math.isclose(float(value[metric]),float(np.mean(scores)),rel_tol=0,abs_tol=1e-12))): raise ValueError(f"Exp472 persisted {key} metric differs")
                if value[f"{metric}_excluded"]!=len(expected_order)-len(scores): raise ValueError(f"Exp472 persisted {key} denominator differs")
    for key, rows in (("micro",full),("subset_micro",sub)):
        table=record[key]
        if not isinstance(table,dict) or set(table)!=set(methods): raise ValueError(f"Exp472 persisted {key} differs")
        for method in methods:
            value=table[method]
            if not isinstance(value,dict) or set(value)!={"average_precision","auroc","dataset_denominator","observation_denominator"} or type(value["dataset_denominator"]) is not int or type(value["observation_denominator"]) is not int or value["dataset_denominator"]<0 or value["observation_denominator"]<0: raise ValueError(f"Exp472 persisted {key} row differs")
            scored=sum(rows[(method,d)]["average_precision_exclusion"] != "typed_abstention" for d in (order if key=="micro" else subset_order))
            if value["dataset_denominator"]!=scored or any(x is not None and not _finite_probability_antigravity(x) for x in (value["average_precision"],value["auroc"])): raise ValueError(f"Exp472 persisted {key} metric differs")
    den=record["denominators"]
    if not isinstance(den,dict) or set(den)!={"full_population","upsampled_rows_gt_zero"} or den["full_population"]!={"dataset_count":1117,"observation_count":record["micro"]["E"]["observation_denominator"]} or not isinstance(den["upsampled_rows_gt_zero"],dict) or set(den["upsampled_rows_gt_zero"])!={"dataset_count","observation_count","dataset_order"} or den["upsampled_rows_gt_zero"]["dataset_count"]!=len(subset_order) or tuple(den["upsampled_rows_gt_zero"]["dataset_order"])!=subset_order or den["upsampled_rows_gt_zero"]["observation_count"]!=record["subset_micro"]["E"]["observation_denominator"]: raise ValueError("Exp472 persisted denominators differ")
    pairs=record["paired_statistics"]
    if not isinstance(pairs,list) or len(pairs)!=4: raise ValueError("Exp472 persisted paired statistics differ")
    expected_pairs=tuple((p,m) for p in ("full","upsampled_rows_gt_zero") for m in ("average_precision","auroc"))
    for row,(population,metric) in zip(pairs,expected_pairs,strict=True):
        population_order=order if population=="full" else subset_order; rows=full if population=="full" else sub
        support=tuple(d for d in population_order if rows[("E",d)][metric] is not None and rows[("D",d)][metric] is not None)
        required={"comparison","population","metric","dataset_order","mean_delta","ci_lower","ci_upper","wins","ties","losses","seed","draw_count","percentile","tie_tolerance"}
        if not isinstance(row,dict) or set(row)!=required or row["comparison"]!="E-D" or row["population"]!=population or row["metric"]!=metric or tuple(row["dataset_order"])!=support or any(type(row[k]) is not int or row[k]<0 for k in ("wins","ties","losses","draw_count")) or row["wins"]+row["ties"]+row["losses"]!=len(support): raise ValueError("Exp472 persisted paired statistics differs")
        expected_stat=paired_e_minus_d_exp472_antigravity([rows[("E",dataset)][metric] for dataset in support],[rows[("D",dataset)][metric] for dataset in support]) if support else {"mean_delta":None,"ci_lower":None,"ci_upper":None,"wins":0,"ties":0,"losses":0,"seed":_SEED,"draw_count":0,"percentile":95.0,"tie_tolerance":_TIE}
        for stat_name, expected_value in expected_stat.items():
            actual=row[stat_name]
            if isinstance(expected_value,float):
                if type(actual) not in (int,float) or not math.isfinite(float(actual)) or not math.isclose(float(actual),expected_value,rel_tol=0,abs_tol=1e-12): raise ValueError("Exp472 persisted paired statistics differs")
            elif actual!=expected_value: raise ValueError("Exp472 persisted paired statistics differs")
    curves=record["curves"]
    if not isinstance(curves,dict) or set(curves)!={"roc","pr"}: raise ValueError("Exp472 persisted curves differs")
    for name,x,y,ascending in (("roc","false_positive_rate","true_positive_rate",True),("pr","recall","precision",False)):
        table=curves[name]
        if not isinstance(table,dict) or set(table)!={"x_name","y_name","rows"} or table["x_name"]!=x or table["y_name"]!=y or not isinstance(table["rows"],list): raise ValueError("Exp472 persisted curve schema differs")
        grouped={m:[] for m in methods}
        for row in table["rows"]:
            required={"method",x,y}|({"threshold"} if name=="roc" else set())
            if not isinstance(row,dict) or set(row)!=required or row["method"] not in grouped or not _finite_probability_antigravity(row[x]) or not _finite_probability_antigravity(row[y]) or (name=="roc" and row["threshold"] is not None and not (type(row["threshold"]) in (int,float) and math.isfinite(float(row["threshold"])) )): raise ValueError("Exp472 persisted curve row differs")
            grouped[row["method"]].append(row)
        expected_curve_methods={method for method in methods if record["micro"][method]["average_precision"] is not None and record["micro"][method]["auroc"] is not None}
        if set(method for method, values in grouped.items() if values)!=expected_curve_methods: raise ValueError("Exp472 persisted curve membership differs")
        for method, values in grouped.items():
            if not values: continue
            xx=[float(row[x]) for row in values]; yy=[float(row[y]) for row in values]
            if len(values)<2 or (name=="roc" and (any(a>b for a,b in zip(xx,xx[1:])) or any(a>b for a,b in zip(yy,yy[1:])) or (xx[0],yy[0])!=(0.0,0.0) or (xx[-1],yy[-1])!=(1.0,1.0))) or (name=="pr" and any(a<b for a,b in zip(xx,xx[1:]))): raise ValueError("Exp472 persisted curve monotonicity differs")
            if name=="roc":
                thresholds=[row["threshold"] for row in values]
                if thresholds[0] is not None or any(threshold is None for threshold in thresholds[1:]) or any(float(left)<float(right) for left,right in zip(thresholds[1:],thresholds[2:])): raise ValueError("Exp472 persisted ROC thresholds differ")
    # The subset table is a projection, not an independently supplied result.
    # This catches a self-digested mutation of one subset cell without labels.
    if any(dict(sub[(method,dataset)]) != dict(full[(method,dataset)]) for method in methods for dataset in subset_order): raise ValueError("Exp472 persisted subset projection differs")
    # Labels and raw score vectors are intentionally not persisted.  The sealed
    # completion marker below anchors the generated science; these two exact
    # producer identities additionally cross-check the persisted metric/curve
    # sections without reopening labels.
    for method in methods:
        roc=[row for row in curves["roc"]["rows"] if row["method"]==method]
        pr=[row for row in curves["pr"]["rows"] if row["method"]==method]
        micro=record["micro"][method]
        if micro["auroc"] is None:
            if roc: raise ValueError("Exp472 persisted ROC membership differs")
        elif not math.isclose(float(micro["auroc"]),float(np.trapezoid([float(row["true_positive_rate"]) for row in roc],[float(row["false_positive_rate"]) for row in roc])),rel_tol=0,abs_tol=1e-12): raise ValueError("Exp472 persisted AUROC curve differs")
        if micro["average_precision"] is None:
            if pr: raise ValueError("Exp472 persisted PR membership differs")
        elif not math.isclose(float(micro["average_precision"]),float(-np.sum(np.diff([float(row["recall"]) for row in pr])*np.asarray([float(row["precision"]) for row in pr[:-1]]))),rel_tol=0,abs_tol=1e-12): raise ValueError("Exp472 persisted AP curve differs")

def _validate_evaluation_record_antigravity(record: Mapping[str,object], *, authority: Exp472ReuseAuthorityAntigravity, e_manifest_sha256: str, context_sha256: str, dataset_order: Sequence[str]) -> None:
    expected_authority=json.loads(canonical_json_bytes_exp472_antigravity({name:getattr(authority,name) for name in authority.__dataclass_fields__}).decode("utf-8"))
    if record["authority"]!=expected_authority or _digest_evaluation_antigravity(record["e_manifest_sha256"],"E manifest digest")!=e_manifest_sha256 or _digest_evaluation_antigravity(record["context_sha256"],"context digest")!=context_sha256: raise ValueError("Exp472 persisted evaluation authority differs")
    _validate_evaluation_science_antigravity(record,tuple(dataset_order))

def _rebuild_from_trusted_evidence_antigravity(labels: Mapping[str,np.ndarray], *, authority: Exp472ReuseAuthorityAntigravity, e_manifest_sha256: str, context_sha256: str, dataset_order: Sequence[str], e_root: Path | None, exp471_root: Path | None, context: Exp472RunContextAntigravity | None, catalog: Sequence[Exp472CatalogEntryAntigravity] | None) -> Exp472EvaluationAntigravity:
    if not isinstance(e_root,Path) or not isinstance(exp471_root,Path) or not isinstance(context,Exp472RunContextAntigravity) or catalog is None: raise ValueError("Exp472 persisted evaluation recomputation context differs")
    exp471=verify_exp471_reuse_exp472_antigravity(exp471_root)
    e=_read_verified_e_path_antigravity(e_root,exp471,context,catalog)
    if exp471.authority!=authority or e.path_sha256!=e_manifest_sha256 or context.sha256!=context_sha256 or exp471.dataset_order!=tuple(dataset_order): raise ValueError("Exp472 persisted evaluation trusted authority differs")
    return _evaluate_from_verified_antigravity(exp471,e,labels,context)

def read_exp472_evaluation_antigravity(root: Path, *, authority: Exp472ReuseAuthorityAntigravity, e_manifest_sha256: str, context_sha256: str, dataset_order: Sequence[str], e_root: Path | None = None, exp471_root: Path | None = None, context: Exp472RunContextAntigravity | None = None, catalog: Sequence[Exp472CatalogEntryAntigravity] | None = None) -> Exp472EvaluationAntigravity | _VerifiedPersistedEvaluationAntigravity:
    """Reopen only a canonical, authority-bound evaluation; never labels."""
    path=evaluation_path_exp472_antigravity(_safe_evaluation_root_antigravity(root,create=False))
    record=_read_evaluation_record_antigravity(path)
    target=path.parent
    _validate_evaluation_record_antigravity(record,authority=authority,e_manifest_sha256=e_manifest_sha256,context_sha256=context_sha256,dataset_order=dataset_order)
    _read_evaluation_marker_antigravity(target,record)
    try:
        labels,evidence_sha256=_read_private_label_evidence_antigravity(target,authority=authority,e_manifest_sha256=e_manifest_sha256,context_sha256=context_sha256,dataset_order=dataset_order)
        if evidence_sha256!=record["private_label_evidence_sha256"]: raise ValueError("Exp472 private label evidence binding differs")
        parsed_authority=Exp472ReuseAuthorityAntigravity(**{**record["authority"],"method_order":tuple(record["authority"]["method_order"])})
        persisted=Exp472EvaluationAntigravity(parsed_authority,tuple(record["method_order"]),tuple(record["per_dataset"]),tuple(record["subset_per_dataset"]),MappingProxyType(record["macro"]),MappingProxyType(record["micro"]),MappingProxyType(record["subset_macro"]),MappingProxyType(record["subset_micro"]),tuple(record["paired_statistics"]),MappingProxyType(record["denominators"]),MappingProxyType(record["diagnostic_summary"]),MappingProxyType(record["curves"]),labels,record["e_manifest_sha256"],record["context_sha256"])
        if any(value is not None for value in (e_root,exp471_root,context,catalog)):
            rebuilt=_rebuild_from_trusted_evidence_antigravity(labels,authority=authority,e_manifest_sha256=e_manifest_sha256,context_sha256=context_sha256,dataset_order=dataset_order,e_root=e_root,exp471_root=exp471_root,context=context,catalog=catalog)
            if rebuilt.artifact_record()!=persisted.artifact_record(): raise ValueError("Exp472 persisted evaluation science differs from trusted evidence")
            return _verified_persisted_evaluation_antigravity(persisted)
        return persisted
    except (TypeError,ValueError) as error: raise ValueError("Exp472 persisted evaluation differs") from error

def _publish_verified_evaluation_antigravity(evaluation: Exp472EvaluationAntigravity, root: Path) -> None:
    """Private no-clobber publication; callers must have completed trusted replay."""
    if not isinstance(evaluation,Exp472EvaluationAntigravity): raise ValueError("Exp472 evaluation publication requires typed evaluation")
    record=evaluation.artifact_record()
    _validate_evaluation_record_antigravity(record,authority=evaluation.authority,e_manifest_sha256=evaluation.e_manifest_sha256,context_sha256=evaluation.context_sha256,dataset_order=tuple(evaluation.diagnostic_summary))
    target=_safe_evaluation_root_antigravity(root,create=True); path=evaluation_path_exp472_antigravity(target); payload=canonical_json_bytes_exp472_antigravity(record)
    evidence=_private_label_evidence_record_antigravity(evaluation)
    if evidence["private_label_evidence_sha256"]!=record["private_label_evidence_sha256"]: raise ValueError("Exp472 private label evidence binding differs")
    _write_create_or_identical_evaluation_antigravity(_private_label_evidence_path_antigravity(target),canonical_json_bytes_exp472_antigravity(evidence),mode=0o600)
    _write_create_or_identical_evaluation_antigravity(path,payload)
    _write_create_or_identical_evaluation_antigravity(_evaluation_marker_path_antigravity(target),canonical_json_bytes_exp472_antigravity(_evaluation_marker_record_antigravity(record)))

def read_or_evaluate_exp472_antigravity(root: Path, *, authority: Exp472ReuseAuthorityAntigravity, e_manifest_sha256: str, context_sha256: str, dataset_order: Sequence[str], compute: Callable[[], Exp472EvaluationAntigravity], e_root: Path | None = None, exp471_root: Path | None = None, context: Exp472RunContextAntigravity | None = None, catalog: Sequence[Exp472CatalogEntryAntigravity] | None = None) -> _VerifiedPersistedEvaluationAntigravity:
    """Read the sealed result, or let exactly one process compute and seal it."""
    if not callable(compute): raise ValueError("Exp472 evaluation compute differs")
    if not all(value is not None for value in (e_root,exp471_root,context,catalog)):
        raise ValueError("Exp472 evaluation publication requires trusted replay inputs")
    target=_safe_evaluation_root_antigravity(root,create=True)
    with _evaluation_claim_antigravity(target):
        try:
            return read_exp472_evaluation_antigravity(target,authority=authority,e_manifest_sha256=e_manifest_sha256,context_sha256=context_sha256,dataset_order=dataset_order,e_root=e_root,exp471_root=exp471_root,context=context,catalog=catalog)
        except FileNotFoundError:
            if evaluation_path_exp472_antigravity(target).exists(): raise
            try:
                labels,_evidence_sha256=_read_private_label_evidence_antigravity(target,authority=authority,e_manifest_sha256=e_manifest_sha256,context_sha256=context_sha256,dataset_order=dataset_order)
            except FileNotFoundError:
                evaluation=compute()
            else:
                evaluation=_rebuild_from_trusted_evidence_antigravity(labels,authority=authority,e_manifest_sha256=e_manifest_sha256,context_sha256=context_sha256,dataset_order=dataset_order,e_root=e_root,exp471_root=exp471_root,context=context,catalog=catalog)
            if not isinstance(evaluation,Exp472EvaluationAntigravity): raise ValueError("Exp472 evaluation compute differs")
            _validate_evaluation_record_antigravity(evaluation.artifact_record(),authority=authority,e_manifest_sha256=e_manifest_sha256,context_sha256=context_sha256,dataset_order=dataset_order)
            if any(value is not None for value in (e_root,exp471_root,context,catalog)):
                rebuilt=_rebuild_from_trusted_evidence_antigravity(_labels(evaluation.private_labels,tuple(dataset_order)),authority=authority,e_manifest_sha256=e_manifest_sha256,context_sha256=context_sha256,dataset_order=dataset_order,e_root=e_root,exp471_root=exp471_root,context=context,catalog=catalog)
                if rebuilt.artifact_record()!=evaluation.artifact_record(): raise ValueError("Exp472 evaluation compute science differs from trusted evidence")
            _publish_verified_evaluation_antigravity(rebuilt,target)
            return read_exp472_evaluation_antigravity(target,authority=authority,e_manifest_sha256=e_manifest_sha256,context_sha256=context_sha256,dataset_order=dataset_order,e_root=e_root,exp471_root=exp471_root,context=context,catalog=catalog)

__all__=["EVALUATION_COMPLETE_NAME_ANTIGRAVITY","EVALUATION_NAME_ANTIGRAVITY","Exp472EvaluationAntigravity","Exp472VerifiedEBundleAntigravity","evaluate_exp472_antigravity","evaluation_path_exp472_antigravity","paired_e_minus_d_exp472_antigravity","read_exp472_evaluation_antigravity","read_or_evaluate_exp472_antigravity"]
