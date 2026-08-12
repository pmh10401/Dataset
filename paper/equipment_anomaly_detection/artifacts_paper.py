"""Create-only Task 8 artifacts with one typed Task1--7 shard writer."""
from __future__ import annotations

from hashlib import sha256
import json, math, os, stat
from pathlib import Path

import numpy as np

from .contracts_paper import FrozenTask8BackendDecision, LowFrequencyResult, RawDataset, Task8FrozenManifest
from .hard_route_paper import HardRouteResult
from .lineage_paper import read_strict_json
from .routing_paper import PolicyBundle
from .sources_paper.hard_sources_paper import EXPECTED_HARD_SOURCES, HardSourceBundle
from .sources_paper.stability_paper import StabilityPartition

SHARD_SCHEMA="paper-task8-proposal-shard.v1"; MANIFEST_SCHEMA="paper-task8-proposal-manifest.v1"; PARITY_SCHEMA="paper-task8-parity.v1"
ARTIFACT_DIRECTORY="stage_artifacts_paper"; FREEZE_STAGE="proposal_freeze"; SHARD_DIRECTORY="shards"; MANIFEST_NAME="proposal_manifest_paper.json"
POLICIES=("hard_only","hard_plus_source","hard_plus_stable","hard_plus_unstable")

def canonical_json_bytes(value: object) -> bytes:
    try:return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()+b"\n"
    except (TypeError,ValueError) as error:raise ValueError("artifact must be canonical finite JSON") from error
def sha256_bytes(data: bytes) -> str:return sha256(data).hexdigest()
def _lstat(path:Path):
    try:return path.lstat()
    except FileNotFoundError:return None
def safe_root(root:Path)->Path:
    raw=Path(root)
    if ".." in raw.parts:raise ValueError("root traversal is forbidden")
    current=Path(raw.anchor) if raw.is_absolute() else Path.cwd()
    for part in raw.parts[1:] if raw.is_absolute() else raw.parts:
        if part in {"",".",".."}:raise ValueError("root traversal is forbidden")
        current/=part; details=_lstat(current)
        if details is not None and stat.S_ISLNK(details.st_mode):raise ValueError("symlinked root component is forbidden")
    return current
def safe_child(root:Path,*parts:str)->Path:
    target=safe_root(root)
    for part in parts:
        if type(part)is not str or not part or "/" in part or Path(part).is_absolute() or part in {".",".."}:raise ValueError("artifact path traversal is forbidden")
        target/=part; details=_lstat(target)
        if details is not None and stat.S_ISLNK(details.st_mode):raise ValueError("symlinked artifact component is forbidden")
    return target
def _mkdirs(path:Path)->None:
    pending=[]; current=path
    while _lstat(current) is None:pending.append(current);current=current.parent
    if stat.S_ISLNK(_lstat(current).st_mode) or not stat.S_ISDIR(_lstat(current).st_mode):raise ValueError("artifact parent is unsafe")
    for directory in reversed(pending):os.mkdir(directory)
def write_create_only(path:Path,data:bytes)->None:
    _mkdirs(path.parent)
    if _lstat(path) is not None:raise FileExistsError(f"create-only artifact already exists: {path}")
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
    try:os.write(descriptor,data);os.fsync(descriptor)
    finally:os.close(descriptor)
def read_canonical(path:Path)->object:
    details=_lstat(path)
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):raise ValueError("artifact missing or symlinked")
    raw=path.read_bytes()
    try:value=read_strict_json(path)
    except Exception as error:raise ValueError("artifact JSON invalid") from error
    if raw!=canonical_json_bytes(value):raise ValueError("artifact noncanonical")
    return value
def _finite(value:object)->None:
    if isinstance(value,dict):
        for key,child in value.items():
            skeleton="".join(character for character in str(key).casefold() if character.isascii() and character.isalnum())
            if skeleton in {"label","tp","fp","fn","f1","precision","recall","auroc","averageprecision"}:raise ValueError("forbidden scientific alias")
            _finite(child)
    elif isinstance(value,list):
        for child in value:_finite(child)
    elif isinstance(value,float) and not math.isfinite(value):raise ValueError("nonfinite artifact value")
def _freeze_root(root:Path)->Path:return safe_child(root,ARTIFACT_DIRECTORY,FREEZE_STAGE)
def _shard_path(root:Path,dataset:str)->Path:return safe_child(root,ARTIFACT_DIRECTORY,FREEZE_STAGE,SHARD_DIRECTORY,f"{dataset}_paper.json")
def _catalog_entry(catalog:dict[str,object],dataset:str)->dict[str,object]:
    entries=[entry for entry in catalog["datasets"] if entry["dataset"]==dataset]
    if len(entries)!=1:raise ValueError("catalog dataset differs")
    return entries[0]
def _raw_identity(raw:RawDataset,entry:dict[str,object])->None:
    if not isinstance(raw,RawDataset) or raw.dataset!=entry["dataset"] or raw.metadata_length!=entry["series_length"]:raise ValueError("typed raw dataset differs")
    for indices,blobs,identity in ((raw.train_indices,raw.train_blobs,entry["train"]),(raw.test_indices,raw.test_blobs,entry["test"])):
        if tuple(indices)!=tuple(identity["indices"]) or len(blobs)!=identity["count"] or [sha256(bytes(blob)).hexdigest() for blob in blobs]!=identity["values_blob_sha256"]:raise ValueError("typed raw identity differs")
    if len(raw.train_rows)!=len(raw.train_blobs) or len(raw.test_rows)!=len(raw.test_blobs):raise ValueError("typed raw rows differ")
def _array(values:object,n:int,name:str)->list[float]:
    value=np.asarray(values)
    if value.ndim!=1 or value.size!=n or value.dtype.kind not in "iuf" or not np.all(np.isfinite(value)):raise ValueError(f"{name} differs")
    return [float(item) for item in value]
def _bools(values:object,n:int,name:str)->list[bool]:
    value=np.asarray(values)
    if value.ndim!=1 or value.size!=n or value.dtype!=np.dtype(bool):raise ValueError(f"{name} differs")
    return [bool(item) for item in value]
def _indices(values:object,n:int,name:str)->list[int]:
    if not isinstance(values,frozenset) or any(type(item)is not int or item<0 or item>=n for item in values):raise ValueError(f"{name} differs")
    return sorted(values)
def _lanes(value,n):return {"hard":_indices(value.hard,n,"hard lanes"),"standard":_indices(value.standard,n,"standard lanes"),"priority":_indices(value.priority,n,"priority lanes")}
def _typed_payload(raw:RawDataset,sources:HardSourceBundle,hard:HardRouteResult,low:LowFrequencyResult,stability:StabilityPartition,policies:PolicyBundle,backend:FrozenTask8BackendDecision,entry:dict[str,object])->dict[str,object]:
    _raw_identity(raw,entry); n=entry["test"]["count"]
    if not isinstance(sources,HardSourceBundle) or tuple(sources.source_names)!=EXPECTED_HARD_SOURCES or set(sources.by_name)!=set(EXPECTED_HARD_SOURCES):raise TypeError("typed Task4 source bundle is invalid")
    source_rows=[]
    for order,name in enumerate(EXPECTED_HARD_SOURCES):
        result=sources.by_name[name]
        if result.name!=name or type(result.eligible)is not bool or type(result.eligibility_reason)is not str:raise ValueError("typed Task4 source result differs")
        source_rows.append({"source_id":name,"order":order,"test_score":_array(result.test_score,n,"source score"),"pvalue":_array(result.test_pvalue,n,"source pvalue"),"native_pass":_bools(result.test_native_pass,n,"source pass"),"eligible":result.eligible,"reason":result.eligibility_reason})
    if not isinstance(hard,HardRouteResult) or hard.n_test!=n:raise TypeError("typed Task5 route result is invalid")
    hard_stages=[{"stage_id":"exp74d","order":0,"lanes":_lanes(hard.exp74d,n)},{"stage_id":"exp89","order":1,"indices":_indices(hard.exp89,n,"exp89")},{"stage_id":"exp90","order":2,"indices":_indices(hard.exp90,n,"exp90")},{"stage_id":"exp93","order":3,"indices":_indices(hard.exp93,n,"exp93")},{"stage_id":"exp137","order":4,"lanes":_lanes(hard.exp137,n)},{"stage_id":"exp234","order":5,"minimum_two":_indices(hard.exp234.minimum_two,n,"exp234"),"minimum_three":_indices(hard.exp234.minimum_three,n,"exp234")},{"stage_id":"exp236","order":6,"lanes":_lanes(hard.exp236,n)},{"stage_id":"exp253","order":7,"lanes":_lanes(hard.exp253,n)}]
    if not isinstance(low,LowFrequencyResult) or low.datasets!=(raw.dataset,) or tuple(row.instance_index for row in low.rows)!=tuple(range(n)) or low.hard_lanes.get(raw.dataset)!=hard.exp253.hard:raise TypeError("typed Task6 result is invalid")
    low_rows=[{"index":row.instance_index,"evidence_missing":row.evidence_missing,"strict_pass":row.strict_pass,"provisional":row.provisional,"depths":[{"depth":item.depth,"score":item.score,"threshold":item.threshold,"strict_pass":item.strict_pass,"excess_ratio":item.excess_ratio,"empirical_rank":item.empirical_rank} for item in row.depth_evidence]} for row in low.rows]
    if not isinstance(stability,StabilityPartition) or tuple(stability.selected_coordinates)!=tuple(candidate.coordinate for candidate in low.selected_candidates):raise TypeError("typed Task7 stability differs")
    stability_record={"selected":[list(item) for item in stability.selected_coordinates],"stable":[list(item) for item in stability.stable_coordinates],"unstable":[list(item) for item in stability.unstable_coordinates]}
    if not isinstance(policies,PolicyBundle):raise TypeError("typed Task7 policy bundle is invalid")
    expected=(hard.automatic_hard,hard.automatic_hard|{index for dataset,index in stability.selected_coordinates if dataset==raw.dataset},hard.automatic_hard|{index for dataset,index in stability.stable_coordinates if dataset==raw.dataset},hard.automatic_hard|{index for dataset,index in stability.unstable_coordinates if dataset==raw.dataset})
    policy_rows=[]
    for order,(route,want) in enumerate(zip(policies.policies,expected,strict=True)):
        if route.name!=POLICIES[order] or set(route.alerts)!={raw.dataset} or route.alerts[raw.dataset]!=want:raise ValueError("typed Task7 policy route differs")
        scores=_array(route.scores[raw.dataset],n,"policy scores"); policy_rows.append({"policy_id":route.name,"order":order,"alerts":sorted(want),"scores":scores,"score_mask":[score>=.5 for score in scores]})
    if not isinstance(backend,FrozenTask8BackendDecision):raise TypeError("typed backend decision is invalid")
    return {"schema":SHARD_SCHEMA,"dataset":raw.dataset,"train_identity":entry["train"],"test_identity":entry["test"],"source_rows":source_rows,"hard_stages":hard_stages,"low_frequency_rows":low_rows,"stability":stability_record,"policies":policy_rows,"backend":{"selected_backend":backend.selected_backend,"reason":backend.reason,"verdict":backend.verdict}}
def _validate_shard(value:object,entry:dict[str,object])->dict[str,object]:
    _finite(value)
    required={"schema","dataset","train_identity","test_identity","source_rows","hard_stages","low_frequency_rows","stability","policies","backend"}
    if not isinstance(value,dict) or set(value)!=required or value.get("schema")!=SHARD_SCHEMA or value.get("dataset")!=entry["dataset"] or value["train_identity"]!=entry["train"] or value["test_identity"]!=entry["test"]:raise ValueError("proposal shard identity differs")
    n=entry["test"]["count"]
    if not isinstance(value["source_rows"],list) or [row.get("source_id") if isinstance(row,dict) else None for row in value["source_rows"]]!=list(EXPECTED_HARD_SOURCES):raise ValueError("proposal source order differs")
    for order,row in enumerate(value["source_rows"]):
        if set(row)!={"source_id","order","test_score","pvalue","native_pass","eligible","reason"} or row["order"]!=order:_raise=ValueError("proposal source schema differs");raise _raise
        _array(row["test_score"],n,"source score");_array(row["pvalue"],n,"source pvalue");_bools(row["native_pass"],n,"source pass")
    if not isinstance(value["low_frequency_rows"],list) or [row.get("index") if isinstance(row,dict) else None for row in value["low_frequency_rows"]]!=list(range(n)):raise ValueError("proposal Task6 rows differ")
    stability=value["stability"]
    if not isinstance(stability,dict) or set(stability)!={"selected","stable","unstable"} or any(not isinstance(stability[key],list) for key in stability):raise ValueError("proposal stability schema differs")
    selected={tuple(item) for item in stability["selected"]}; stable={tuple(item) for item in stability["stable"]}; unstable={tuple(item) for item in stability["unstable"]}
    if stable&unstable or stable|unstable!=selected or any(item[0]!=entry["dataset"] or type(item[1])is not int or not 0<=item[1]<n for item in selected):raise ValueError("proposal stability identity differs")
    if not isinstance(value["policies"],list) or [row.get("policy_id") if isinstance(row,dict) else None for row in value["policies"]]!=list(POLICIES):raise ValueError("proposal policy order differs")
    hard_rows=value["hard_stages"]
    if not isinstance(hard_rows,list) or len(hard_rows)!=8 or hard_rows[-1].get("stage_id")!="exp253":raise ValueError("proposal Task5 stages differ")
    automatic=set(hard_rows[-1]["lanes"]["hard"]); expected=(automatic,automatic|{index for _,index in selected},automatic|{index for _,index in stable},automatic|{index for _,index in unstable})
    for order,(row,want) in enumerate(zip(value["policies"],expected,strict=True)):
        if set(row)!={"policy_id","order","alerts","scores","score_mask"} or row["order"]!=order or row["alerts"]!=sorted(want) or row["score_mask"]!=[score>=.5 for score in _array(row["scores"],n,"policy scores")]:raise ValueError("proposal policy relation differs")
    if value["backend"]!={"selected_backend":"cpu","reason":"task9_pending","verdict":"cpu_fallback"}:raise ValueError("proposal backend differs")
    return value
def write_typed_proposal_shard(root:Path,catalog:dict[str,object],raw:RawDataset,sources:HardSourceBundle,hard:HardRouteResult,low:LowFrequencyResult,stability:StabilityPartition,policies:PolicyBundle,backend:FrozenTask8BackendDecision)->Path:
    if not isinstance(raw,RawDataset):raise TypeError("typed RawDataset is required")
    entry=_catalog_entry(catalog,raw.dataset); payload=_typed_payload(raw,sources,hard,low,stability,policies,backend,entry); path=_shard_path(root,raw.dataset);write_create_only(path,canonical_json_bytes(payload));return path
def _acceptance(path:Path,catalog:dict[str,object]):
    from .catalog_paper import load_acceptance
    accepted=load_acceptance(Path(path),catalog); raw=Path(path).read_bytes(); return accepted,raw
def freeze_score_root(root:Path,catalog:dict[str,object],acceptance_path:Path)->Task8FrozenManifest:
    root=safe_root(root); accepted,raw=_acceptance(acceptance_path,catalog); names=[entry["dataset"] for entry in catalog["datasets"]]; hashes={}; total=0
    for entry in catalog["datasets"]:
        path=_shard_path(root,entry["dataset"]);_validate_shard(read_canonical(path),entry);hashes[entry["dataset"]]=sha256_bytes(path.read_bytes());total+=entry["test"]["count"]
    shard_dir=safe_child(root,ARTIFACT_DIRECTORY,FREEZE_STAGE,SHARD_DIRECTORY)
    if {path.name for path in shard_dir.iterdir()}!={f"{name}_paper.json" for name in names}:raise ValueError("proposal shard namespace differs")
    if catalog["mode"]=="full" and (len(names)!=1117 or total!=195900):raise ValueError("full proposal inventory differs")
    payload={"schema":MANIFEST_SCHEMA,"catalog_sha256":catalog["catalog_sha256"],"datasets":names,"test_row_count":total,"shard_hashes":hashes,"acceptance_path":str(Path(acceptance_path)),"acceptance_sha256":accepted["acceptance_sha256"],"acceptance_file_sha256":sha256_bytes(raw)};write_create_only(safe_child(root,ARTIFACT_DIRECTORY,FREEZE_STAGE,MANIFEST_NAME),canonical_json_bytes(payload));return Task8FrozenManifest(**{**payload,"datasets":tuple(names)})
def verify_score_root(root:Path,catalog:dict[str,object])->Task8FrozenManifest:
    root=safe_root(root); path=safe_child(root,ARTIFACT_DIRECTORY,FREEZE_STAGE,MANIFEST_NAME); value=read_canonical(path); manifest=Task8FrozenManifest(**{**value,"datasets":tuple(value["datasets"])})
    accepted,raw=_acceptance(Path(manifest.acceptance_path),catalog)
    if manifest.catalog_sha256!=catalog["catalog_sha256"] or manifest.acceptance_sha256!=accepted["acceptance_sha256"] or manifest.acceptance_file_sha256!=sha256_bytes(raw):raise ValueError("proposal manifest authority differs")
    names=[entry["dataset"] for entry in catalog["datasets"]]; total=0
    for entry in catalog["datasets"]:
        shard_path=_shard_path(root,entry["dataset"]);_validate_shard(read_canonical(shard_path),entry)
        if manifest.shard_hashes[entry["dataset"]]!=sha256_bytes(shard_path.read_bytes()):raise ValueError("proposal shard hash differs")
        total+=entry["test"]["count"]
    if manifest.datasets!=tuple(names) or manifest.test_row_count!=total:raise ValueError("proposal manifest total differs")
    directory=_freeze_root(root); shards=safe_child(root,ARTIFACT_DIRECTORY,FREEZE_STAGE,SHARD_DIRECTORY)
    if {item.name for item in directory.iterdir()}!={MANIFEST_NAME,SHARD_DIRECTORY} or {item.name for item in shards.iterdir()}!={f"{name}_paper.json" for name in names}:raise ValueError("proposal freeze namespace differs")
    return manifest
def verify_parity(root:Path,catalog:dict[str,object])->dict[str,object]:
    manifest=verify_score_root(root,catalog); accepted,raw=_acceptance(Path(manifest.acceptance_path),catalog)
    if sha256_bytes(raw)!=manifest.acceptance_file_sha256:raise ValueError("parity acceptance hash differs")
    oracle=safe_child(Path(manifest.acceptance_path).parent,accepted["oracle"]["relative_path"])
    if sha256_bytes(oracle.read_bytes())!=accepted["oracle"]["sha256"]:raise ValueError("parity oracle hash differs")
    oracle_value=read_canonical(oracle); observed=[]
    for policy in POLICIES:
        pairs=[]
        for entry in catalog["datasets"]:
            shard=_validate_shard(read_canonical(_shard_path(root,entry["dataset"])),entry);row=next(item for item in shard["policies"] if item["policy_id"]==policy);pairs.extend((entry["dataset"],index) for index in row["alerts"])
        observed.append(tuple(sorted(pairs)))
    expected=accepted["policies"]
    if not isinstance(oracle_value,dict) or oracle_value!={"schema":"paper-task8-coordinate-oracle.v1","policies":[{"policy_id":item["policy_id"],"coordinates":item["coordinates"]} for item in expected]}:raise ValueError("parity oracle content differs")
    for got,want in zip(observed,expected,strict=True):
        if [[dataset,index] for dataset,index in got]!=want["coordinates"] or len(got)!=want["coordinate_count"] or sha256_bytes(canonical_json_bytes([[dataset,index] for dataset,index in got]))!=want["coordinate_sha256"]:raise ValueError("parity coordinate differs")
    return {"schema":PARITY_SCHEMA,"manifest_sha256":sha256_bytes(safe_child(root,ARTIFACT_DIRECTORY,FREEZE_STAGE,MANIFEST_NAME).read_bytes()),"acceptance_sha256":manifest.acceptance_sha256,"oracle_sha256":accepted["oracle"]["sha256"],"result":"exact"}
