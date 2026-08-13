"""Closed Task8/9 proposal pipeline through acceleration and proposal-score only."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path

from .acceleration_paper import ACCELERATION_SCHEMA, build_acceleration_manifest, verify_acceleration_manifest
from .artifacts_paper import Task4Task5SuccessAuthority, canonical_json_bytes, read_canonical, safe_child, safe_root, sha256_bytes, verify_score_root, verify_task4_task5_attempt_namespace, verify_task4_task5_success_namespace, write_create_only
from .catalog_paper import build_catalog, load_acceptance
from .config_paper import LINEAGE_LOCK_PATH, PAPER_ROOT
from .contracts_paper import FrozenTask8BackendDecision
from .data_paper import load_dataset_values
from .lineage_paper import read_strict_json, sha256_file
from .proposal_score_paper import ProposalScoreIncomplete, compose_proposal_score, write_and_freeze_proposal_score

PIPELINE_STAGE_ORDER=("preflight","prepare","acceleration","proposal_score","proposal_freeze","proposal_parity","public_baselines","baseline_bridge","evaluate","report","independent_verify")
JOURNAL_DIRECTORY="pipeline_journal_paper"; ARTIFACT_DIRECTORY="stage_artifacts_paper"
class PipelineStateError(ValueError): pass
class PipelineNotImplementedError(NotImplementedError): pass

@dataclass(frozen=True,slots=True)
class PipelineConfig:
    db_path: Path; run_root: Path; mode: str; smoke_datasets: tuple[str,...]; acceptance_path: Path|None=None
    def __post_init__(self):
        if self.mode not in {"smoke","full"}: raise ValueError("mode invalid")
        datasets=tuple(self.smoke_datasets)
        if self.mode=="smoke" and (not datasets or datasets!=tuple(sorted(set(datasets)))): raise ValueError("smoke subset invalid")
        if self.mode=="full" and datasets: raise ValueError("full mode has no smoke subset")
        for field in ("db_path","run_root"): object.__setattr__(self,field,Path(getattr(self,field)))
        if self.acceptance_path is not None: object.__setattr__(self,"acceptance_path",Path(self.acceptance_path))
        object.__setattr__(self,"smoke_datasets",datasets)

def _now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds")
def _sha(value): return sha256(canonical_json_bytes(value)).hexdigest()
def _closure():
    lock=read_strict_json(PAPER_ROOT/"codemap_paper.lock")
    if not isinstance(lock,dict) or set(lock)!={"schema","hashes"} or lock["schema"]!="paper-codemap-lock.v1" or not isinstance(lock["hashes"],dict): raise PipelineStateError("codemap closure lock invalid")
    hashes=lock["hashes"]
    required={"acceleration_paper.py","artifacts_paper.py","catalog_paper.py","data_paper.py","proposal_score_paper.py","runner_paper.py","pipeline_paper.py","contracts_paper.py"}
    if not required <= set(hashes) or any(not isinstance(path,str) or not isinstance(digest,str) or len(digest)!=64 for path,digest in hashes.items()): raise PipelineStateError("codemap closure missing executor")
    if any(not (PAPER_ROOT/path).is_file() or sha256_file(PAPER_ROOT/path)!=digest for path,digest in hashes.items()): raise PipelineStateError("codemap closure differs")
    return hashes

def _paths(config):
    def field(path):
        if path is None: return {"state":"acceptance_absent"}
        raw=str(path); resolved=str(safe_root(path)); return {"unresolved_path":raw,"resolved_path":resolved}
    return {"db_path":field(config.db_path),"run_root":field(config.run_root),"acceptance_path":field(config.acceptance_path)}

def _acceptance_context(config,catalog):
    if config.acceptance_path is None: return {"status":"acceptance_absent"}
    acceptance=load_acceptance(config.acceptance_path,catalog)
    return {"status":"bound","acceptance_sha256":acceptance["acceptance_sha256"],"acceptance_path_sha256":sha256_file(config.acceptance_path)}

def context_payload(config: PipelineConfig):
    safe_root(config.run_root)
    try:
        catalog=build_catalog(config.db_path,mode=config.mode,smoke_datasets=config.smoke_datasets)
        acceptance=_acceptance_context(config,catalog)
    except ValueError as error:
        raise PipelineStateError(f"catalog or acceptance context differs: {error}") from error
    base={"schema":"paper-task9-context.v1","mode":config.mode,"smoke_datasets":list(config.smoke_datasets),"paths":_paths(config),"database":catalog["database"],"catalog_sha256":catalog["catalog_sha256"],"parity_authority":acceptance,"lineage_sha256":sha256_file(LINEAGE_LOCK_PATH),"codemap_lock_sha256":sha256_file(PAPER_ROOT/"codemap_paper.lock"),"code_closure":_closure(),"constants":{"stage_order":list(PIPELINE_STAGE_ORDER),"registry":{"preflight":"paper-task8-preflight.v1","prepare":"paper-task8-prepare.v1","acceleration":ACCELERATION_SCHEMA,"proposal_score":"paper-task9-proposal-score.v1"}}}
    return {**base,"context_sha256":_sha(base)}

def _record(payload): return {**payload,"record_sha256":_sha(payload)}
def _journal_path(root,index,name): return safe_child(root,JOURNAL_DIRECTORY,f"{index:06d}_{name}_paper.json")
def _artifact_path(root,stage): return safe_child(root,ARTIFACT_DIRECTORY,stage,f"{stage}_paper.json")
def _legacy(root):
    old={"pipeline_state_paper.json","proposal_inventory_paper.json","proposal_freeze_manifest_paper.json"}
    return any((root/name).exists() for name in old)

def _records(root):
    journal=safe_child(root,JOURNAL_DIRECTORY)
    if not journal.is_dir(): raise PipelineStateError("journal missing")
    files=sorted(journal.iterdir()); names=("context","preflight","prepare","acceleration","proposal_score")
    if not files or len(files)>len(names): raise PipelineStateError("journal missing")
    result=[]; prior="genesis"
    for index,path in enumerate(files):
        if path.name!=f"{index:06d}_{names[index]}_paper.json": raise PipelineStateError("journal gap, reorder, or truncation")
        record=read_canonical(path)
        if not isinstance(record,dict) or record.get("record_sha256")!=_sha({key:value for key,value in record.items() if key!="record_sha256"}) or record.get("previous_record_sha256")!=prior: raise PipelineStateError("forged journal record")
        if index==0 and (record.get("schema")!="paper-task8-journal-context.v1" or set(record)!={"schema","context","previous_record_sha256","record_sha256"}): raise PipelineStateError("journal genesis invalid")
        if index and record.get("schema")!="paper-task8-journal-complete.v1": raise PipelineStateError("journal completion invalid")
        prior=record["record_sha256"]; result.append(record)
    return result

def _catalog_from_context(context):
    return build_catalog(Path(context["paths"]["db_path"]["unresolved_path"]),mode=context["mode"],smoke_datasets=tuple(context["smoke_datasets"]))

def _proposal_artifact(root,context):
    acceleration_path=_artifact_path(root,"acceleration")
    acceleration=verify_acceleration_manifest(read_canonical(acceleration_path),context["context_sha256"])
    catalog=_catalog_from_context(context); manifest=verify_score_root(root,catalog,FrozenTask8BackendDecision.from_acceleration_manifest(acceleration))
    return {"schema":"paper-task9-proposal-score.v1","stage":"proposal_score","context_sha256":context["context_sha256"],"acceleration_sha256":sha256_bytes(acceleration_path.read_bytes()),"catalog_sha256":catalog["catalog_sha256"],"proposal_manifest_sha256":sha256_bytes(safe_child(root,ARTIFACT_DIRECTORY,"proposal_freeze","proposal_manifest_paper.json").read_bytes()),"parity_status":manifest.parity_status}

def _artifact_record(root,stage,context):
    path=_artifact_path(root,stage); raw=path.read_bytes(); value=read_canonical(path)
    if stage=="preflight": expected={"schema":"paper-task8-preflight.v1","stage":stage,"context_sha256":context["context_sha256"]}
    elif stage=="prepare": expected={"schema":"paper-task8-prepare.v1","stage":stage,"context_sha256":context["context_sha256"],"catalog":_catalog_from_context(context),"parity_authority":context["parity_authority"]}
    elif stage=="acceleration": expected=verify_acceleration_manifest(value,context["context_sha256"])
    elif stage=="proposal_score": expected=_proposal_artifact(root,context)
    else: raise PipelineStateError("unregistered pipeline stage")
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
        stage=record.get("stage")
        if stage not in {"preflight","prepare","acceleration","proposal_score"} or record.get("artifact")!=_artifact_record(root,stage,context): raise PipelineStateError("journal artifact differs")
        complete.append(stage)
    artifact_root=safe_child(root,ARTIFACT_DIRECTORY)
    actual_names={item.name for item in artifact_root.iterdir()} if artifact_root.is_dir() else set()
    expected_names=set(complete)|({"proposal_freeze"} if "proposal_score" in complete else set())
    if "proposal_work" in actual_names:
        if "acceleration" not in complete:
            raise PipelineStateError("namespace pollution")
        try:
            acceleration=verify_acceleration_manifest(read_canonical(_artifact_path(root,"acceleration")),context["context_sha256"])
            catalog=_catalog_from_context(context)
            raw={entry["dataset"]:load_dataset_values(Path(context["paths"]["db_path"]["unresolved_path"]),entry["dataset"]) for entry in catalog["datasets"]}
            authority=Task4Task5SuccessAuthority.from_verified_context(context,catalog,acceleration)
            verify_task4_task5_success_namespace(root,catalog,authority,raw)
            verify_task4_task5_attempt_namespace(root,catalog,authority,raw)
        except ValueError as error:
            raise PipelineStateError("namespace pollution") from error
        expected_names.add("proposal_work")
    if actual_names != expected_names: raise PipelineStateError("namespace pollution")
    return context,records,complete

def _incomplete_status(root, context):
    acceleration=verify_acceleration_manifest(read_canonical(_artifact_path(root,"acceleration")),context["context_sha256"])
    catalog=_catalog_from_context(context); raw={entry["dataset"]:load_dataset_values(Path(context["paths"]["db_path"]["unresolved_path"]),entry["dataset"]) for entry in catalog["datasets"]}
    authority=Task4Task5SuccessAuthority.from_verified_context(context,catalog,acceleration)
    completed=verify_task4_task5_success_namespace(root,catalog,authority,raw)
    attempts=verify_task4_task5_attempt_namespace(root,catalog,authority,raw)
    pending=tuple(dataset for dataset in raw if dataset not in completed)
    failed=tuple(sorted({record.dataset for record in attempts if record.dataset in pending}))
    attempt_path=str(safe_child(root,ARTIFACT_DIRECTORY,"proposal_work","task4_task5_attempts"))
    return {"status":"proposal_score_incomplete","completed":len(completed),"failed":len(failed),"pending":len(pending),"pending_datasets":list(pending),"failed_datasets":list(failed),"attempts_path":attempt_path,"completed_stages":["preflight","prepare","acceleration"],"next_stage":"proposal_score"}

def _write_complete(root,stage,context,prior):
    if stage=="preflight": payload={"schema":"paper-task8-preflight.v1","stage":stage,"context_sha256":context["context_sha256"]}
    elif stage=="prepare": payload={"schema":"paper-task8-prepare.v1","stage":stage,"context_sha256":context["context_sha256"],"catalog":_catalog_from_context(context),"parity_authority":context["parity_authority"]}
    elif stage=="acceleration": payload=build_acceleration_manifest(context["context_sha256"])
    elif stage=="proposal_score":
        catalog=_catalog_from_context(context); raw={entry["dataset"]:load_dataset_values(Path(context["paths"]["db_path"]["unresolved_path"]),entry["dataset"]) for entry in catalog["datasets"]}
        acceleration=verify_acceleration_manifest(read_canonical(_artifact_path(root,"acceleration")),context["context_sha256"])
        proposal=compose_proposal_score(raw,acceleration,root=root,context=context,catalog=catalog); write_and_freeze_proposal_score(root,catalog,proposal); payload=_proposal_artifact(root,context)
    else: raise PipelineStateError("unregistered pipeline stage")
    path=_artifact_path(root,stage); write_create_only(path,canonical_json_bytes(payload)); artifact=_artifact_record(root,stage,context)
    index=("preflight","prepare","acceleration","proposal_score").index(stage)+1
    write_create_only(_journal_path(root,index,stage),canonical_json_bytes(_record({"schema":"paper-task8-journal-complete.v1","stage":stage,"previous_record_sha256":prior,"artifact":artifact,"finished_at":_now()})))

def run_pipeline(config: PipelineConfig):
    root=safe_root(config.run_root)
    if _legacy(root): raise PipelineStateError("legacy_root_requires_new_root")
    if not root.exists(): root.parent.mkdir(parents=True,exist_ok=True); os.mkdir(root)
    if not root.is_dir() or root.is_symlink(): raise PipelineStateError("run root invalid")
    if not (root/JOURNAL_DIRECTORY).exists():
        if any(root.iterdir()): raise PipelineStateError("root pollution")
        context=context_payload(config); write_create_only(_journal_path(root,0,"context"),canonical_json_bytes(_record({"schema":"paper-task8-journal-context.v1","context":context,"previous_record_sha256":"genesis"})))
    context,records,complete=verify_journal(config)
    next_stage=next((stage for stage in ("preflight","prepare","acceleration","proposal_score") if stage not in complete),None)
    if next_stage is None: return verify_completed(config)
    try:
        _write_complete(root,next_stage,context,records[-1]["record_sha256"])
    except ProposalScoreIncomplete:
        return _incomplete_status(root,context)
    return run_pipeline(config)

def verify_completed(config):
    context,_records_value,complete=verify_journal(config)
    if complete == ["preflight","prepare","acceleration"] and safe_child(config.run_root,ARTIFACT_DIRECTORY, "proposal_work").exists():
        return _incomplete_status(safe_root(config.run_root),context)
    return {"status":"completed_prefix" if complete != ["preflight","prepare","acceleration","proposal_score"] else "completed","completed_stages":complete,"failed_stage":None,"next_stage":"proposal_freeze" if len(complete)==4 else ("preflight","prepare","acceleration","proposal_score")[len(complete)]}
def evaluate_stage(*_args,**_kwargs): raise PipelineNotImplementedError("evaluate is unavailable until Task 11")
__all__=["ARTIFACT_DIRECTORY","JOURNAL_DIRECTORY","PIPELINE_STAGE_ORDER","PipelineConfig","PipelineNotImplementedError","PipelineStateError","context_payload","evaluate_stage","run_pipeline","verify_completed","verify_journal"]
