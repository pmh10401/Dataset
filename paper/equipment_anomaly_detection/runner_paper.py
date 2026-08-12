"""Closed Task 8 pipeline: only context and independently recomputed prepare exist."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata, os, platform
from pathlib import Path

from .artifacts_paper import canonical_json_bytes, read_canonical, safe_child, safe_root, sha256_bytes, verify_score_root, write_create_only
from .catalog_paper import build_catalog, load_acceptance, verify_catalog
from .config_paper import LINEAGE_LOCK_PATH, PAPER_ROOT
from .lineage_paper import read_strict_json, sha256_file

PIPELINE_STAGE_ORDER=("preflight","prepare","acceleration","proposal_score","proposal_freeze","proposal_parity","public_baselines","baseline_bridge","evaluate","report","independent_verify")
JOURNAL_DIRECTORY="pipeline_journal_paper"; ARTIFACT_DIRECTORY="stage_artifacts_paper"
class PipelineStateError(ValueError): pass
class PipelineNotImplementedError(NotImplementedError): pass
@dataclass(frozen=True,slots=True)
class PipelineConfig:
    db_path: Path; run_root: Path; mode: str; smoke_datasets: tuple[str,...]; acceptance_path: Path
    def __post_init__(self):
        if self.mode not in {"smoke","full"}: raise ValueError("mode invalid")
        datasets=tuple(self.smoke_datasets)
        if self.mode=="smoke" and (not datasets or datasets!=tuple(sorted(set(datasets)))): raise ValueError("smoke subset invalid")
        if self.mode=="full" and datasets: raise ValueError("full mode has no smoke subset")
        for field in ("db_path","run_root","acceptance_path"): object.__setattr__(self,field,Path(getattr(self,field)))
        object.__setattr__(self,"smoke_datasets",datasets)
def _now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds")
def _sha(value): return sha256(canonical_json_bytes(value)).hexdigest()
def _closure():
    lock=read_strict_json(PAPER_ROOT/"codemap_paper.lock")
    if not isinstance(lock,dict) or set(lock)!={"schema","hashes"} or lock["schema"]!="paper-codemap-lock.v1" or not isinstance(lock["hashes"],dict): raise PipelineStateError("codemap closure lock invalid")
    hashes=lock["hashes"]
    required={"artifacts_paper.py","catalog_paper.py","runner_paper.py","pipeline_paper.py","contracts_paper.py"}
    if not required <= set(hashes) or any(not isinstance(path,str) or not isinstance(digest,str) or len(digest)!=64 for path,digest in hashes.items()): raise PipelineStateError("codemap closure missing executor")
    if any(not (PAPER_ROOT/path).is_file() or sha256_file(PAPER_ROOT/path)!=digest for path,digest in hashes.items()): raise PipelineStateError("codemap closure differs")
    return hashes
def _environment():
    def version(name):
        try:return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:return None
    cpu=os.cpu_count() or 1
    return {"python":platform.python_version(),"platform":platform.platform(),"cpu_count":cpu,"dataset_workers":min(max(1,cpu-1),7),"dependencies":{"numpy":version("numpy"),"torch":version("torch"),"mlx":version("mlx")},"acceleration_assessment":{"tensor_eligible":False,"pytorch":version("torch") is not None,"mlx":version("mlx") is not None,"mps":"not_assessed_until_task9","canonical":"cpu","benchmark_required":"deterministic parity and >=1.1x including transfer"}}
def _paths(config):
    def field(path):
        raw=str(path); resolved=str(safe_root(path)); return {"unresolved_path":raw,"resolved_path":resolved}
    return {"db_path":field(config.db_path),"run_root":field(config.run_root),"acceptance_path":field(config.acceptance_path)}
def context_payload(config: PipelineConfig):
    root=safe_root(config.run_root)
    try:
        catalog=build_catalog(config.db_path,mode=config.mode,smoke_datasets=config.smoke_datasets)
        acceptance=load_acceptance(config.acceptance_path,catalog)
    except ValueError as error:
        raise PipelineStateError(f"catalog or acceptance context differs: {error}") from error
    base={"schema":"paper-task8-context.v1","mode":config.mode,"smoke_datasets":list(config.smoke_datasets),"paths":_paths(config),"database":catalog["database"],"catalog_sha256":catalog["catalog_sha256"],"acceptance_sha256":acceptance["acceptance_sha256"],"acceptance_path_sha256":sha256_file(config.acceptance_path),"lineage_sha256":sha256_file(LINEAGE_LOCK_PATH),"codemap_lock_sha256":sha256_file(PAPER_ROOT/"codemap_paper.lock"),"code_closure":_closure(),"environment":_environment(),"constants":{"stage_order":list(PIPELINE_STAGE_ORDER),"registry":{"preflight":"paper-task8-preflight.v1","prepare":"paper-task8-prepare.v1"}}}
    return {**base,"context_sha256":_sha(base)}
def _record(payload): return {**payload,"record_sha256":_sha(payload)}
def _journal_path(root,index,name): return safe_child(root,JOURNAL_DIRECTORY,f"{index:06d}_{name}_paper.json")
def _artifact_path(root,stage): return safe_child(root,ARTIFACT_DIRECTORY,stage,f"{stage}_paper.json")
def _legacy(root):
    old={"pipeline_state_paper.json","proposal_inventory_paper.json","proposal_freeze_manifest_paper.json"}
    if any((root/name).exists() for name in old): return True
    journal=root/JOURNAL_DIRECTORY
    return journal.exists() and any(path.name=="000000_context_paper.json" and b"paper-task8-journal-context.v1" not in path.read_bytes() for path in journal.iterdir() if path.is_file())
def _records(root):
    journal=safe_child(root,JOURNAL_DIRECTORY)
    if not journal.is_dir(): raise PipelineStateError("journal missing")
    files=sorted(journal.iterdir())
    if not files: raise PipelineStateError("journal missing")
    names=("context","preflight","prepare","acceleration")
    result=[]; prior="genesis"
    for index,path in enumerate(files):
        if index>=len(names) or path.name!=f"{index:06d}_{names[index]}_paper.json": raise PipelineStateError("journal gap, reorder, or truncation")
        record=read_canonical(path)
        if not isinstance(record,dict) or record.get("record_sha256")!=_sha({key:value for key,value in record.items() if key!="record_sha256"}) or record.get("previous_record_sha256")!=prior: raise PipelineStateError("forged journal record")
        if index==0 and (record.get("schema")!="paper-task8-journal-context.v1" or set(record)!={"schema","context","previous_record_sha256","record_sha256"}): raise PipelineStateError("journal genesis invalid")
        if index in (1,2) and record.get("schema")!="paper-task8-journal-complete.v1": raise PipelineStateError("journal completion invalid")
        if index==3 and (record.get("schema")!="paper-task8-journal-failure.v1" or record.get("reason")!="stage_not_implemented"): raise PipelineStateError("journal failure invalid")
        prior=record["record_sha256"]; result.append(record)
    return result
def _artifact_record(root, stage, context, prior):
    path=_artifact_path(root,stage); raw=path.read_bytes(); value=read_canonical(path)
    expected={"schema":f"paper-task8-{stage}.v1","stage":stage,"context_sha256":context["context_sha256"]}
    if stage=="prepare":
        catalog=build_catalog(Path(context["paths"]["db_path"]["unresolved_path"]),mode=context["mode"],smoke_datasets=tuple(context["smoke_datasets"]))
        acceptance=load_acceptance(Path(context["paths"]["acceptance_path"]["unresolved_path"]),catalog)
        expected["catalog"]=catalog; expected["acceptance"]={key:value for key,value in acceptance.items() if key!="_path"}; expected["acceptance_sha256"]=context["acceptance_sha256"]
    if value!=expected: raise PipelineStateError("stage semantic verifier failed")
    return {"relative_path":str(path.relative_to(root)),"size":len(raw),"sha256":sha256_bytes(raw),"schema":expected["schema"]}
def verify_journal(config):
    root=safe_root(config.run_root)
    if _legacy(root): raise PipelineStateError("legacy_root_requires_new_root")
    records=_records(root); context=context_payload(config)
    if records[0]["context"]!=context: raise PipelineStateError("context recomputation differs")
    allowed={JOURNAL_DIRECTORY,ARTIFACT_DIRECTORY}
    if {item.name for item in root.iterdir()}-allowed: raise PipelineStateError("root pollution")
    complete=[]
    for record in records[1:]:
        if record["schema"]=="paper-task8-journal-complete.v1":
            stage=record.get("stage");
            if stage not in {"preflight","prepare"} or record.get("artifact")!=_artifact_record(root,stage,context,record["previous_record_sha256"]): raise PipelineStateError("journal artifact differs")
            namespace=safe_child(root,ARTIFACT_DIRECTORY,stage)
            if {item.relative_to(root).as_posix() for item in namespace.iterdir() if item.is_file()}!={record["artifact"]["relative_path"]}: raise PipelineStateError("namespace pollution")
            complete.append(stage)
    artifact_root=safe_child(root,ARTIFACT_DIRECTORY)
    actual_names={item.name for item in artifact_root.iterdir()} if artifact_root.is_dir() else set()
    if actual_names - (set(complete)|{"proposal_freeze"}):
        raise PipelineStateError("namespace pollution")
    if "proposal_freeze" in actual_names:
        try:
            catalog=build_catalog(config.db_path,mode=config.mode,smoke_datasets=config.smoke_datasets)
            verify_score_root(root,catalog)
        except ValueError as error:
            raise PipelineStateError(f"proposal freeze verification differs: {error}") from error
    return context,records,complete
def _write_complete(root,stage,context,prior):
    expected={"schema":f"paper-task8-{stage}.v1","stage":stage,"context_sha256":context["context_sha256"]}
    if stage=="prepare":
        catalog=build_catalog(Path(context["paths"]["db_path"]["unresolved_path"]),mode=context["mode"],smoke_datasets=tuple(context["smoke_datasets"]))
        acceptance=load_acceptance(Path(context["paths"]["acceptance_path"]["unresolved_path"]),catalog)
        expected["catalog"]=catalog; expected["acceptance"]={key:value for key,value in acceptance.items() if key!="_path"}; expected["acceptance_sha256"]=context["acceptance_sha256"]
    path=_artifact_path(root,stage); write_create_only(path,canonical_json_bytes(expected)); artifact=_artifact_record(root,stage,context,prior)
    write_create_only(_journal_path(root,1 if stage=="preflight" else 2,stage),canonical_json_bytes(_record({"schema":"paper-task8-journal-complete.v1","stage":stage,"previous_record_sha256":prior,"artifact":artifact,"finished_at":_now()})))
def run_pipeline(config: PipelineConfig):
    root=safe_root(config.run_root)
    if _legacy(root): raise PipelineStateError("legacy_root_requires_new_root")
    if not root.exists(): _mkdir_parent=root.parent; _mkdir_parent.mkdir(parents=True,exist_ok=True); os.mkdir(root)
    if not root.is_dir() or root.is_symlink(): raise PipelineStateError("run root invalid")
    if not (root/JOURNAL_DIRECTORY).exists():
        if any(root.iterdir()): raise PipelineStateError("root pollution")
        context=context_payload(config); write_create_only(_journal_path(root,0,"context"),canonical_json_bytes(_record({"schema":"paper-task8-journal-context.v1","context":context,"previous_record_sha256":"genesis"})))
    context,records,complete=verify_journal(config)
    if records[-1].get("schema")=="paper-task8-journal-failure.v1": raise PipelineNotImplementedError("acceleration: stage_not_implemented")
    if len(complete)==0: _write_complete(root,"preflight",context,records[-1]["record_sha256"]); return run_pipeline(config)
    if len(complete)==1: _write_complete(root,"prepare",context,records[-1]["record_sha256"]); return run_pipeline(config)
    failure=_record({"schema":"paper-task8-journal-failure.v1","stage":"acceleration","reason":"stage_not_implemented","previous_record_sha256":records[-1]["record_sha256"],"finished_at":_now()}); write_create_only(_journal_path(root,3,"acceleration"),canonical_json_bytes(failure)); raise PipelineNotImplementedError("acceleration: stage_not_implemented")
def verify_completed(config):
    _,records,complete=verify_journal(config); failed=records[-1].get("schema")=="paper-task8-journal-failure.v1"
    return {"status":"failed" if failed else "completed_prefix","completed_stages":complete,"failed_stage":"acceleration" if failed else None,"next_stage":"acceleration"}
def evaluate_stage(*_args,**_kwargs): raise PipelineNotImplementedError("evaluate is unavailable until Task 11 and verified parity")
__all__=["ARTIFACT_DIRECTORY","JOURNAL_DIRECTORY","PIPELINE_STAGE_ORDER","PipelineConfig","PipelineNotImplementedError","PipelineStateError","context_payload","evaluate_stage","run_pipeline","verify_completed","verify_journal"]
