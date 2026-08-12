"""Closed v2 pipeline authority: context recomputation and fixed stage registry."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timezone
from hashlib import sha256
import importlib.metadata
import os
from pathlib import Path
import platform
import re
from typing import Callable,Mapping
from .artifacts_paper import canonical_json_bytes,policy_coordinates,read_canonical,safe_child,safe_root,sha256_bytes,verify_score_root,write_create_only
from .catalog_paper import load_acceptance,load_catalog
from .config_paper import LINEAGE_LOCK_PATH,PAPER_ROOT
from .lineage_paper import read_strict_json,sha256_file

PIPELINE_STAGE_ORDER=("preflight","prepare","acceleration","proposal_score","proposal_freeze","proposal_parity","public_baselines","baseline_bridge","evaluate","report","independent_verify")
JOURNAL_DIRECTORY="pipeline_journal_paper";ARTIFACT_DIRECTORY="stage_artifacts_paper";_HEX=re.compile(r"[0-9a-f]{64}")
class PipelineStateError(ValueError):pass
class PipelineFailedError(RuntimeError):pass
class PipelineNotImplementedError(NotImplementedError):pass
def _sha(value:object)->str:return sha256(canonical_json_bytes(value)).hexdigest()
def _now()->str:return datetime.now(timezone.utc).isoformat(timespec="microseconds")
@dataclass(frozen=True,slots=True)
class StageSpec:
    stage:str;verifier_id:str;version:str;predecessor:str|None;filename:str;schema:str;evidence_key:str
    @property
    def relative_path(self)->str:return f"{ARTIFACT_DIRECTORY}/{self.stage}/{self.filename}"
STAGE_REGISTRY={stage:StageSpec(stage,f"{stage}-verifier","v2",None if index==0 else PIPELINE_STAGE_ORDER[index-1],f"{stage}_paper.json",f"paper-stage-{stage}.v2",f"{stage}_evidence") for index,stage in enumerate(PIPELINE_STAGE_ORDER)}
@dataclass(frozen=True,slots=True)
class PipelineConfig:
    db_path:Path;run_root:Path;mode:str;catalog_path:Path;acceptance_path:Path
    def __post_init__(self):
        if self.mode not in {"smoke","full"}:raise ValueError("mode invalid")
        object.__setattr__(self,"db_path",Path(self.db_path));object.__setattr__(self,"run_root",Path(self.run_root));object.__setattr__(self,"catalog_path",Path(self.catalog_path));object.__setattr__(self,"acceptance_path",Path(self.acceptance_path))
    def artifact_payload(self)->dict[str,object]:
        catalog=load_catalog(self.catalog_path,self.db_path,self.mode);entries=catalog["datasets"]
        bare={"mode":self.mode,"datasets":[entry["dataset"] for entry in entries],"test_rows":{entry["dataset"]:entry["test"]["count"] for entry in entries},"train_identity_digests":{entry["dataset"]:entry["train"]["row_digest"] for entry in entries},"test_identity_digests":{entry["dataset"]:entry["test"]["row_digest"] for entry in entries}}
        return {**bare,"context_sha256":_sha(bare)}
def _environment()->dict[str,object]:
    def version(name:str):
        try:return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:return None
    torch=version("torch");mlx=version("mlx");cpu=os.cpu_count() or 1
    return {"python":platform.python_version(),"platform":platform.platform(),"cpu_count":cpu,"dataset_workers":min(max(1,cpu-1),7),"dependencies":{"numpy":version("numpy"),"torch":torch,"mlx":mlx},"accelerators":{"pytorch_available":torch is not None,"mlx_available":mlx is not None,"mps_assessment":"unavailable_until_task9"}}
def _closure()->dict[str,str]:
    """External codemap lock binds every listed source, including executing modules."""
    lock=read_strict_json(PAPER_ROOT/"codemap_paper.lock");hashes=lock.get("hashes") if isinstance(lock,dict) else None
    if not isinstance(hashes,dict):raise PipelineStateError("codemap closure lock invalid")
    actual={name:sha256_file(PAPER_ROOT/name) for name in hashes if (PAPER_ROOT/name).is_file()}
    if actual!=hashes:raise PipelineStateError("codemap closure differs")
    return actual
def _path_fields(path:Path)->dict[str,str]:
    raw=str(path);resolved=str(safe_root(path));return {"unresolved_path":raw,"resolved_path":resolved}
def context_payload(config:PipelineConfig)->dict[str,object]:
    try:catalog=load_catalog(config.catalog_path,config.db_path,config.mode);acceptance=load_acceptance(config.acceptance_path,catalog)
    except ValueError as error:raise PipelineStateError(f"catalog or acceptance context differs: {error}") from error
    base={"schema":"paper-pipeline-context.v3","mode":config.mode,"db_path":_path_fields(config.db_path),"run_root":_path_fields(config.run_root),"catalog_path":_path_fields(config.catalog_path),"acceptance_path":_path_fields(config.acceptance_path),"catalog_sha256":catalog["catalog_sha256"],"acceptance_sha256":acceptance["acceptance_sha256"],"lineage_sha256":sha256_file(LINEAGE_LOCK_PATH),"codemap_lock_sha256":sha256_file(PAPER_ROOT/"codemap_paper.lock"),"code_closure":_closure(),"environment":_environment(),"constants":{"source_order":["rocket_exp40","exp55_best","exp56_best","exp84","rocket_block_b","canonical_normal_state"],"policy_order":["hard_only","hard_plus_source","hard_plus_stable","hard_plus_unstable"],"shard_schema":"paper-proposal-shard.v3","registry":{k:[v.verifier_id,v.version,v.schema] for k,v in STAGE_REGISTRY.items()}}}
    return {**base,"context_sha256":_sha(base)}
def _path(root:Path,index:int,stage:str)->Path:return safe_child(root,JOURNAL_DIRECTORY,f"{index:06d}_{stage}_paper.json")
def _record(bare:dict[str,object])->dict[str,object]:return {**bare,"record_sha256":_sha(bare)}
def _read_record(path:Path)->dict[str,object]:
    value=read_canonical(path)
    if not isinstance(value,dict) or value.get("record_sha256")!=_sha({k:v for k,v in value.items() if k!="record_sha256"}):raise PipelineStateError("forged journal record")
    return value
def _stage_payload(spec:StageSpec,context:dict[str,object],predecessor:str)->dict[str,object]:return {"schema":spec.schema,"stage":spec.stage,"verifier_id":spec.verifier_id,"verifier_version":spec.version,"context_sha256":context["context_sha256"],"predecessor_record_sha256":predecessor,"evidence":{spec.evidence_key:_sha([spec.stage,context["context_sha256"],predecessor])}}
def stage_file(config:PipelineConfig,spec:StageSpec)->Path:return safe_child(safe_root(config.run_root),ARTIFACT_DIRECTORY,spec.stage,spec.filename)
def _verify_stage(root:Path,spec:StageSpec,record:dict[str,object],context:dict[str,object])->None:
    artifact=record.get("artifact_set")
    if not isinstance(artifact,dict) or set(artifact)!={"relative_path","size","sha256","schema","verifier_id","verifier_version"} or artifact["relative_path"]!=spec.relative_path or artifact["schema"]!=spec.schema or artifact["verifier_id"]!=spec.verifier_id or artifact["verifier_version"]!=spec.version:raise PipelineStateError("closed stage registry artifact mismatch")
    namespace=safe_child(root,ARTIFACT_DIRECTORY,spec.stage);actual={p.relative_to(root).as_posix() for p in namespace.rglob("*") if p.is_file()} if namespace.is_dir() and not namespace.is_symlink() else set()
    if actual!={spec.relative_path}:raise PipelineStateError("stage namespace pollution or missing artifact")
    path=root/spec.relative_path;value=read_canonical(path)
    if path.stat().st_size!=artifact["size"] or sha256_bytes(path.read_bytes())!=artifact["sha256"] or value!=_stage_payload(spec,context,record["previous_record_sha256"]):raise PipelineStateError("stage semantic verifier failed")
def _files(root:Path)->list[Path]:
    d=safe_child(root,JOURNAL_DIRECTORY)
    if not d.is_dir() or d.is_symlink():raise PipelineStateError("journal missing")
    files=sorted(d.iterdir())
    if not files:raise PipelineStateError("journal missing genesis")
    for index,path in enumerate(files):
        expected="context" if index==0 else PIPELINE_STAGE_ORDER[index-1]
        if path.name!=f"{index:06d}_{expected}_paper.json":raise PipelineStateError("journal gap, reorder, duplicate, or truncation evidence invalid")
    return files
def _legacy_root(root:Path)->bool:
    if (root/"pipeline_state_paper.json").exists():return True
    manifest=root/"proposal_freeze_manifest_paper.json"
    if manifest.is_file() and b"paper-proposal-manifest.v3" not in manifest.read_bytes():return True
    genesis=root/JOURNAL_DIRECTORY/"000000_context_paper.json"
    return genesis.is_file() and b"paper-journal-context.v3" not in genesis.read_bytes()
def verify_journal(config:PipelineConfig)->tuple[dict[str,object],list[dict[str,object]]]:
    root=safe_root(config.run_root)
    if _legacy_root(root):raise PipelineStateError("legacy_root_requires_new_root")
    files=_files(root);records=[];prior="genesis"
    for index,path in enumerate(files):
        record=_read_record(path)
        if record.get("previous_record_sha256")!=prior:raise PipelineStateError("journal chain differs")
        if index==0:
            if record.get("schema")!="paper-journal-context.v3" or not isinstance(record.get("context"),dict):raise PipelineStateError("journal genesis invalid")
        else:
            spec=STAGE_REGISTRY[PIPELINE_STAGE_ORDER[index-1]]
            if record.get("stage")!=spec.stage:
                raise PipelineStateError("journal stage differs")
            if record.get("schema")=="paper-journal-failure.v3":
                if index!=len(files)-1 or not isinstance(record.get("failure_evidence"),dict):raise PipelineStateError("failure journal invalid")
            elif record.get("schema")=="paper-journal-stage.v3":_verify_stage(root,spec,record,records[0]["context"])
            else:raise PipelineStateError("journal schema invalid")
        prior=record["record_sha256"];records.append(record)
    current=context_payload(config)
    if records[0]["context"]!=current:raise PipelineStateError("context recomputation differs")
    allowed={JOURNAL_DIRECTORY,ARTIFACT_DIRECTORY}
    if {p.name for p in root.iterdir()}-allowed:raise PipelineStateError("root pollution")
    return current,records
def _write_stage(config:PipelineConfig,spec:StageSpec,context:dict[str,object],prior:str)->None:
    write_create_only(stage_file(config,spec),canonical_json_bytes(_stage_payload(spec,context,prior)))
def _preflight(config:PipelineConfig,context:dict[str,object],prior:str)->None:_write_stage(config,STAGE_REGISTRY["preflight"],context,prior)
def run_pipeline(config:PipelineConfig,*,test_stage_builders:Mapping[str,Callable[[PipelineConfig,StageSpec,dict[str,object],str],None]]|None=None):
    root=safe_root(config.run_root)
    if not root.exists():root.mkdir(parents=True)
    if not root.is_dir() or root.is_symlink():raise PipelineStateError("run root invalid")
    if _legacy_root(root):raise PipelineStateError("legacy_root_requires_new_root")
    if not (root/JOURNAL_DIRECTORY).exists():
        if any(root.iterdir()):raise PipelineStateError("run root pollution")
        (root/JOURNAL_DIRECTORY).mkdir();context=context_payload(config);write_create_only(_path(root,0,"context"),canonical_json_bytes(_record({"schema":"paper-journal-context.v3","context":context,"previous_record_sha256":"genesis"})))
    context,records=verify_journal(config)
    if records[-1].get("schema")=="paper-journal-failure.v3":raise PipelineStateError("failed root requires clean root")
    if len(records)-1==len(PIPELINE_STAGE_ORDER):return records
    spec=STAGE_REGISTRY[PIPELINE_STAGE_ORDER[len(records)-1]];prior=records[-1]["record_sha256"]
    try:
        if test_stage_builders is not None and spec.stage in test_stage_builders:test_stage_builders[spec.stage](config,spec,context,prior)
        elif spec.stage=="preflight":_preflight(config,context,prior)
        else:raise PipelineNotImplementedError(f"{spec.stage} is not implemented at Task 8")
        path=stage_file(config,spec);value=read_canonical(path);data=path.read_bytes()
        record=_record({"schema":"paper-journal-stage.v3","stage":spec.stage,"previous_record_sha256":prior,"started_at":_now(),"finished_at":_now(),"artifact_set":{"relative_path":spec.relative_path,"size":len(data),"sha256":sha256_bytes(data),"schema":spec.schema,"verifier_id":spec.verifier_id,"verifier_version":spec.version}})
        _verify_stage(root,spec,record,context);write_create_only(_path(root,len(records),spec.stage),canonical_json_bytes(record))
    except Exception as error:
        failure=_record({"schema":"paper-journal-failure.v3","stage":spec.stage,"previous_record_sha256":prior,"started_at":_now(),"finished_at":_now(),"failure_evidence":{"type":type(error).__name__,"message":str(error)}});write_create_only(_path(root,len(records),spec.stage),canonical_json_bytes(failure));raise
    return run_pipeline(config,test_stage_builders=test_stage_builders)
def verify_completed(config:PipelineConfig):
    context,records=verify_journal(config)
    names=[r.get("stage") for r in records[1:] if r.get("schema")=="paper-journal-stage.v3"]
    if "proposal_freeze" in names:verify_score_root(config.run_root)
    if "proposal_parity" in names:verify_parity(config)
    return {"status":"failed" if records[-1].get("schema")=="paper-journal-failure.v3" else "completed_prefix","completed_stages":names,"next_stage":None if len(names)==len(PIPELINE_STAGE_ORDER) else PIPELINE_STAGE_ORDER[len(records)-1]}
def verify_parity(config:PipelineConfig)->dict[str,object]:
    root=safe_root(config.run_root);catalog=load_catalog(config.catalog_path,config.db_path,config.mode);acceptance=load_acceptance(config.acceptance_path,catalog);manifest=verify_score_root(root);report=read_canonical(safe_child(root,ARTIFACT_DIRECTORY,"proposal_parity","proposal_parity_paper.json"))
    required={"schema","proposal_manifest_sha256","acceptance_sha256","policy_counts","policy_coordinate_digests","oracle_identifier","oracle_version","oracle_relative_path","oracle_sha256","oracle_result_sha256","comparison_result"}
    if not isinstance(report,dict) or set(report)!=required or report.get("schema")!="paper-proposal-parity.v3" or report.get("proposal_manifest_sha256")!=sha256_file(root/"proposal_freeze_manifest_paper.json") or report.get("acceptance_sha256")!=acceptance["acceptance_sha256"] or report.get("comparison_result")!="exact" or report.get("oracle_identifier")!="historical-coordinate-oracle" or report.get("oracle_version")!="v1":raise ValueError("parity report invalid")
    rel=report["oracle_relative_path"]
    if type(rel)is not str or Path(rel).is_absolute() or ".." in Path(rel).parts:raise ValueError("parity oracle path invalid")
    oracle=root/rel
    if not oracle.is_file() or sha256_file(oracle)!=report.get("oracle_sha256"):raise ValueError("parity oracle invalid")
    oracle_value=read_canonical(oracle);observed=policy_coordinates(root,manifest);expected=tuple(tuple((str(d),int(i)) for d,i in policy) for policy in acceptance["policies"])
    digests=[_sha([[d,i] for d,i in values]) for values in observed];counts=[len(values) for values in observed]
    if observed!=expected or report["policy_counts"]!=counts or report["policy_coordinate_digests"]!=digests or report["oracle_result_sha256"]!=sha256_bytes(canonical_json_bytes(oracle_value)):raise ValueError("parity coordinate binding differs")
    return report
def evaluate_stage(config:PipelineConfig):
    verify_parity(config)
    from . import label_access_paper
    raise PipelineNotImplementedError("evaluation remains Task 11")
__all__=["ARTIFACT_DIRECTORY","JOURNAL_DIRECTORY","PIPELINE_STAGE_ORDER","PipelineConfig","PipelineFailedError","PipelineNotImplementedError","PipelineStateError","STAGE_REGISTRY","StageSpec","context_payload","evaluate_stage","run_pipeline","stage_file","verify_completed","verify_journal","verify_parity"]
