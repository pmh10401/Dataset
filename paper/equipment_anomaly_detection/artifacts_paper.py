"""Canonical, create-only proposal artifacts and strict v2 shard validation."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Mapping

from .contracts_paper import FrozenProposalManifest
from .lineage_paper import read_strict_json

SHARD_SCHEMA = "paper-proposal-shard.v3"
INVENTORY_SCHEMA = "paper-proposal-inventory.v2"
MANIFEST_SCHEMA = "paper-proposal-manifest.v2"
SHARD_DIRECTORY = "proposal_shards_paper"
INVENTORY_NAME = "proposal_inventory_paper.json"
MANIFEST_NAME = "proposal_freeze_manifest_paper.json"
PARITY_NAME = "proposal_parity_paper.json"
POLICIES = ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")
EXPECTED_SOURCES=("rocket_exp40", "exp55_best", "exp56_best", "exp84", "rocket_block_b", "canonical_normal_state")
EXPECTED_HARD_STAGES=("exp74d","exp89","exp90","exp93","exp137","exp234","exp236","exp253")
FORBIDDEN = frozenset({"label", "tp", "fp", "fn", "precision", "recall", "f1", "auroc", "averageprecision"})
_NAME = re.compile(r"[A-Za-z0-9_.-]+")

class CreateOnlyError(FileExistsError): pass

def canonical_json_bytes(value: object) -> bytes:
    try: return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    except (TypeError, ValueError) as error: raise ValueError("artifact must be canonical finite JSON") from error

def sha256_bytes(value: bytes) -> str: return sha256(value).hexdigest()

def safe_root(root: Path) -> Path:
    """Reject a raw root or any existing component symlink before resolving it."""
    raw = Path(root)
    current = Path(raw.anchor) if raw.is_absolute() else Path.cwd()
    for part in raw.parts[1:] if raw.is_absolute() else raw.parts:
        if part in {"", ".", ".."}: raise ValueError("root traversal is forbidden")
        current /= part
        # macOS commonly aliases /tmp to /private/tmp.  It is an OS mount
        # alias, not a caller-controlled component; all later symlinks are
        # rejected before canonicalization.
        if current.exists() and current.is_symlink() and str(current) not in {"/tmp", "/var", "/private"}: raise ValueError("symlinked root component is forbidden")
    return raw.resolve(strict=False)

def safe_child(root: Path, *parts: str) -> Path:
    if any(not p or Path(p).is_absolute() or "/" in p or p in {".", ".."} for p in parts): raise ValueError("artifact path traversal is forbidden")
    base=safe_root(root); target=base
    for part in parts:
        target/=part
        if target.exists() and target.is_symlink():raise ValueError("symlinked artifact component is forbidden")
    try:target.resolve(strict=False).relative_to(base)
    except ValueError as error:raise ValueError("artifact path traversal is forbidden") from error
    return target

def write_create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent=path.parent
    while parent!=parent.parent:
        if parent.exists() and parent.is_symlink() and str(parent) not in {"/tmp","/var","/private"}:raise ValueError("symlinked artifact component is forbidden")
        parent=parent.parent
    if path.is_symlink(): raise ValueError("symlinked artifact is forbidden")
    fd, name = tempfile.mkstemp(prefix=".tmp_paper_", dir=path.parent); tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as out: out.write(data); out.flush(); os.fsync(out.fileno())
        try: os.link(tmp, path)
        except FileExistsError as error: raise CreateOnlyError(f"create-only artifact already exists: {path}") from error
    finally:
        if tmp.exists(): tmp.unlink()

def read_canonical(path: Path) -> object:
    if path.is_symlink() or not path.is_file(): raise ValueError(f"artifact missing or symlinked: {path.name}")
    raw = path.read_bytes()
    try: value = read_strict_json(path)
    except Exception as error: raise ValueError(f"artifact JSON invalid: {path.name}") from error
    if raw != canonical_json_bytes(value): raise ValueError(f"artifact noncanonical: {path.name}")
    return value

def _forbidden(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            norm = "".join(c for c in str(key).casefold() if c.isalnum())
            long_tokens=("label","precision","recall","auroc","averageprecision")
            if any(token in norm for token in long_tokens) or norm in {"tp","fp","fn","f1"}: raise ValueError(f"forbidden field: {key}")
            _forbidden(child)
    elif isinstance(value, list):
        for child in value: _forbidden(child)
    elif isinstance(value, float) and not math.isfinite(value): raise ValueError("nonfinite artifact value")

def _digest(value: object, name: str) -> str:
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value): raise ValueError(f"{name} digest invalid")
    return value

def _indices(value: object, count: int, name: str, contiguous: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(v) is not int or not 0 <= v < count for v in value) or len(set(value)) != len(value): raise ValueError(f"{name} identity invalid")
    result=tuple(value)
    if contiguous and result != tuple(range(count)): raise ValueError(f"{name} identity invalid")
    return result

def _vector(value: object, count: int, name: str, low: float=0., high: float=1., boolean: bool=False) -> None:
    if not isinstance(value, list) or len(value) != count: raise ValueError(f"{name} alignment invalid")
    if boolean:
        if any(type(v) is not bool for v in value): raise ValueError(f"{name} boolean invalid")
    elif any(type(v) not in (int,float) or not math.isfinite(float(v)) or not low <= float(v) <= high for v in value): raise ValueError(f"{name} domain invalid")

def validate_proposal_shard(value: object) -> dict[str, object]:
    _forbidden(value)
    required={"schema","dataset","train_identity","test_identity","sources","hard_route","low_frequency","stability","policies","backend_decision"}
    if not isinstance(value,dict) or set(value)!=required or value.get("schema")!=SHARD_SCHEMA: raise ValueError("proposal shard schema invalid")
    if type(value["dataset"]) is not str or not _NAME.fullmatch(value["dataset"]): raise ValueError("dataset identity invalid")
    train,test=value["train_identity"],value["test_identity"]
    if not isinstance(train,dict) or set(train)!={"indices","digest"} or not isinstance(test,dict) or set(test)!={"indices","digest"}: raise ValueError("identity schema invalid")
    _digest(train["digest"],"TRAIN"); _digest(test["digest"],"TEST")
    n=len(test["indices"]); _indices(test["indices"],n,"TEST",True)
    if not isinstance(train["indices"],list) or any(type(i)is not int or i<0 for i in train["indices"]) or len(set(train["indices"]))!=len(train["indices"]): raise ValueError("TRAIN identity invalid")
    sources=value["sources"]
    if not isinstance(sources,dict) or tuple(sources)!=EXPECTED_SOURCES: raise ValueError("source order/schema invalid")
    for source in sources.values():
        if not isinstance(source,dict) or set(source)!={"score","pvalue","strength","native_pass"}: raise ValueError("source schema invalid")
        _vector(source["score"],n,"score"); _vector(source["pvalue"],n,"pvalue",low=float.fromhex('0x1.0p-1074')); _vector(source["strength"],n,"strength"); _vector(source["native_pass"],n,"native pass",boolean=True)
    hard=value["hard_route"]
    if not isinstance(hard,dict) or set(hard)!={"stages"} or not isinstance(hard["stages"],dict) or tuple(hard["stages"])!=EXPECTED_HARD_STAGES: raise ValueError("Hard route schema invalid")
    for lane in hard["stages"].values():
        if not isinstance(lane,dict) or set(lane)!={"hard","standard","priority"}: raise ValueError("Hard lane schema invalid")
        groups=[set(_indices(lane[x],n,"Hard lane")) for x in ("hard","standard","priority")]
        if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]: raise ValueError("Hard lane overlap")
    low=value["low_frequency"]
    if not isinstance(low,dict) or set(low)!={"rows"} or not isinstance(low["rows"],list) or len(low["rows"])!=n: raise ValueError("low-frequency schema invalid")
    seen=[]
    for row in low["rows"]:
        if not isinstance(row,dict) or set(row)!={"index","evidence_missing","strict_pass","depths"} or type(row["evidence_missing"]) is not bool or type(row["strict_pass"]) is not bool or not isinstance(row["depths"],list): raise ValueError("low-frequency row invalid")
        seen.append(row["index"])
    _indices(seen,n,"low-frequency",True)
    stability=value["stability"]
    if not isinstance(stability,dict) or set(stability)!={"selected","stable","unstable","views"} or not isinstance(stability["views"],list): raise ValueError("stability schema invalid")
    selected=set(_indices(stability["selected"],n,"stability")); stable=set(_indices(stability["stable"],n,"stability")); unstable=set(_indices(stability["unstable"],n,"stability"))
    if stable&unstable or stable|unstable!=selected: raise ValueError("stability partition invalid")
    if any(not isinstance(view,dict) or set(view)!={"view","survives"} or type(view["view"]) is not int or type(view["survives"]) is not bool for view in stability["views"]): raise ValueError("stability views invalid")
    policies=value["policies"]
    if not isinstance(policies,dict) or tuple(policies)!=POLICIES: raise ValueError("policy order invalid")
    for policy in policies.values():
        if not isinstance(policy,dict) or set(policy)!={"alerts","scores"}: raise ValueError("policy schema invalid")
        alerts=set(_indices(policy["alerts"],n,"policy")); _vector(policy["scores"],n,"policy scores")
        if any((float(score)>=.5)!=(index in alerts) for index,score in enumerate(policy["scores"])): raise ValueError("policy score route invalid")
    backend=value["backend_decision"]
    if not isinstance(backend,dict) or set(backend)!={"selected_backend","reason","verdict"} or backend!={"selected_backend":"cpu","reason":"task9_pending","verdict":"cpu_fallback"}: raise ValueError("backend schema invalid")
    return value

def initialize_score_root(root: Path, config: object) -> None:
    dest=safe_root(root); dest.mkdir(parents=True,exist_ok=True)
    if set(p.name for p in dest.iterdir()) - {"pipeline_journal_paper","stage_artifacts_paper"}: raise ValueError("run root has unexpected content")
    payload={"schema":INVENTORY_SCHEMA,"context":config.artifact_payload()}
    write_create_only(safe_child(dest,INVENTORY_NAME),canonical_json_bytes(payload))

def _inventory(root: Path) -> dict[str,object]:
    value=read_canonical(safe_child(root,INVENTORY_NAME))
    if not isinstance(value,dict) or set(value)!={"schema","context"} or value["schema"]!=INVENTORY_SCHEMA or not isinstance(value["context"],dict): raise ValueError("proposal inventory schema invalid")
    return value

def write_proposal_shard(root: Path,value: object) -> Path:
    dest=safe_root(root); inv=_inventory(dest); shard=validate_proposal_shard(value); ctx=inv["context"]
    dataset=shard["dataset"]
    if dataset not in ctx["datasets"] or len(shard["test_identity"]["indices"])!=ctx["test_rows"][dataset] or shard["train_identity"]["digest"]!=ctx["train_identity_digests"][dataset] or shard["test_identity"]["digest"]!=ctx["test_identity_digests"][dataset]: raise ValueError("proposal shard inventory identity differs")
    path=safe_child(dest,SHARD_DIRECTORY,f"{dataset}_paper.json"); write_create_only(path,canonical_json_bytes(shard)); return path

def _all(root: Path):
    inv=_inventory(root); ctx=inv["context"]; datasets=tuple(ctx["datasets"]); directory=safe_child(root,SHARD_DIRECTORY)
    actual={p.name for p in directory.iterdir()} if directory.is_dir() and not directory.is_symlink() else set(); expected={f"{d}_paper.json" for d in datasets}
    if actual-expected: raise ValueError("extra proposal shards")
    if expected-actual: raise ValueError("proposal shard coverage incomplete")
    hashes={}; total=0
    for dataset in datasets:
        path=safe_child(root,SHARD_DIRECTORY,f"{dataset}_paper.json"); shard=validate_proposal_shard(read_canonical(path));
        if shard["dataset"]!=dataset: raise ValueError("proposal shard identity differs")
        hashes[dataset]=sha256_bytes(path.read_bytes()); total+=len(shard["test_identity"]["indices"])
    return inv,hashes,total

def freeze_score_root(root: Path) -> FrozenProposalManifest:
    dest=safe_root(root); inv,hashes,total=_all(dest); ctx=inv["context"]
    if ctx["mode"]=="full" and (len(ctx["datasets"])!=1117 or total!=195900): raise ValueError("full inventory counts differ")
    payload={"schema":MANIFEST_SCHEMA,"limited_for_smoke":ctx["mode"]=="smoke","datasets":ctx["datasets"],"test_row_count":total,"shard_hashes":hashes,"inventory_sha256":sha256_bytes(canonical_json_bytes(inv)),"config_sha256":ctx["context_sha256"]}
    write_create_only(safe_child(dest,MANIFEST_NAME),canonical_json_bytes(payload)); return FrozenProposalManifest(**{**payload,"datasets":tuple(payload["datasets"])})

def verify_score_root(root: Path) -> FrozenProposalManifest:
    dest=safe_root(root); payload=read_canonical(safe_child(dest,MANIFEST_NAME)); manifest=FrozenProposalManifest(**{**payload,"datasets":tuple(payload["datasets"])})
    try: inv,hashes,total=_all(dest)
    except ValueError as error: raise ValueError(f"proposal artifact hash verification failed: {error}") from error
    if manifest.datasets!=tuple(inv["context"]["datasets"]) or dict(manifest.shard_hashes)!=hashes or manifest.test_row_count!=total or manifest.inventory_sha256!=sha256_bytes(canonical_json_bytes(inv)): raise ValueError("proposal manifest hash differs")
    return manifest

def policy_coordinates(root: Path, manifest: FrozenProposalManifest) -> tuple[tuple[tuple[str,int],...],...]:
    result=[]
    for policy in POLICIES:
        pairs=[]
        for dataset in manifest.datasets:
            shard=validate_proposal_shard(read_canonical(safe_child(safe_root(root),SHARD_DIRECTORY,f"{dataset}_paper.json"))); pairs.extend((dataset,index) for index in shard["policies"][policy]["alerts"])
        result.append(tuple(sorted(pairs)))
    return tuple(result)

__all__=["CreateOnlyError","POLICIES","canonical_json_bytes","freeze_score_root","initialize_score_root","policy_coordinates","read_canonical","safe_child","safe_root","sha256_bytes","validate_proposal_shard","verify_score_root","write_create_only","write_proposal_shard"]
