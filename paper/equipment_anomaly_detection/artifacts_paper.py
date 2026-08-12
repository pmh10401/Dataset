"""Create-only Task 8 artifacts, standalone proposal freeze, and parity verification."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json, math, os, stat
from pathlib import Path
from typing import Iterable

from .contracts_paper import RawDataset, Task8FrozenManifest
from .lineage_paper import read_strict_json

SHARD_SCHEMA = "paper-task8-proposal-shard.v1"; MANIFEST_SCHEMA = "paper-task8-proposal-manifest.v1"; PARITY_SCHEMA = "paper-task8-parity.v1"
SHARD_DIRECTORY = "proposal_shards_paper"; MANIFEST_NAME = "proposal_manifest_paper.json"; PARITY_NAME = "proposal_parity_paper.json"
POLICIES = ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")

def canonical_json_bytes(value: object) -> bytes:
    try: return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    except (TypeError, ValueError) as error: raise ValueError("artifact must be canonical finite JSON") from error
def sha256_bytes(data: bytes) -> str: return sha256(data).hexdigest()
def _lstat(path: Path):
    try: return path.lstat()
    except FileNotFoundError: return None
def safe_root(root: Path) -> Path:
    raw = Path(root)
    if ".." in raw.parts: raise ValueError("root traversal is forbidden")
    current = Path(raw.anchor) if raw.is_absolute() else Path.cwd()
    parts = raw.parts[1:] if raw.is_absolute() else raw.parts
    for part in parts:
        if part in {"", ".", ".."}: raise ValueError("root traversal is forbidden")
        current /= part; details = _lstat(current)
        if details is not None and stat.S_ISLNK(details.st_mode): raise ValueError("symlinked root component is forbidden")
    return current
def safe_child(root: Path, *parts: str) -> Path:
    base = safe_root(root); target = base
    for part in parts:
        if type(part) is not str or not part or "/" in part or Path(part).is_absolute() or part in {".", ".."}: raise ValueError("artifact path traversal is forbidden")
        target /= part; details = _lstat(target)
        if details is not None and stat.S_ISLNK(details.st_mode): raise ValueError("symlinked artifact component is forbidden")
    return target
def _mkdirs(path: Path) -> None:
    todo=[]; current=path
    while _lstat(current) is None: todo.append(current); current=current.parent
    if stat.S_ISLNK(_lstat(current).st_mode) or not stat.S_ISDIR(_lstat(current).st_mode): raise ValueError("artifact parent is unsafe")
    for directory in reversed(todo): os.mkdir(directory)
def write_create_only(path: Path, data: bytes) -> None:
    _mkdirs(path.parent); details=_lstat(path)
    if details is not None: raise FileExistsError(f"create-only artifact already exists: {path}")
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0)
    descriptor=os.open(path,flags,0o600)
    try: os.write(descriptor,data); os.fsync(descriptor)
    finally: os.close(descriptor)
def read_canonical(path: Path) -> object:
    details=_lstat(path)
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode): raise ValueError("artifact missing or symlinked")
    raw=path.read_bytes()
    try: value=read_strict_json(path)
    except Exception as error: raise ValueError("artifact JSON invalid") from error
    if raw != canonical_json_bytes(value): raise ValueError("artifact noncanonical")
    return value

def _finite(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            skeleton="".join(character for character in str(key).casefold() if character.isascii() and character.isalnum())
            if skeleton in {"label","tp","fp","fn","f1","precision","recall","auroc","averageprecision"}: raise ValueError("forbidden scientific alias")
            _finite(child)
    elif isinstance(value,list):
        for child in value:_finite(child)
    elif isinstance(value,float) and not math.isfinite(value): raise ValueError("nonfinite artifact value")
def _digest(value: object) -> bool:return type(value) is str and len(value)==64 and all(c in "0123456789abcdef" for c in value)
def validate_proposal_shard(value: object, catalog_entry: dict[str, object]) -> dict[str, object]:
    _finite(value)
    required={"schema","dataset","train_identity","test_identity","source_rows","hard_stages","low_frequency_rows","stability","policies","backend"}
    if not isinstance(value,dict) or set(value)!=required or value.get("schema")!=SHARD_SCHEMA or value.get("dataset")!=catalog_entry["dataset"]: raise ValueError("proposal shard schema invalid")
    if value["train_identity"]!=catalog_entry["train"] or value["test_identity"]!=catalog_entry["test"]: raise ValueError("proposal shard catalog identity differs")
    n=catalog_entry["test"]["count"]
    for key in ("source_rows","hard_stages","low_frequency_rows"):
        if not isinstance(value[key],list): raise ValueError("proposal shard ordered evidence is invalid")
    if not isinstance(value["policies"],list) or [record.get("policy_id") if isinstance(record,dict) else None for record in value["policies"]]!=list(POLICIES): raise ValueError("proposal policy order invalid")
    for policy in value["policies"]:
        if set(policy)!={"policy_id","order","alerts","score_mask"} or policy["order"]!=POLICIES.index(policy["policy_id"]): raise ValueError("proposal policy schema invalid")
        alerts=policy["alerts"]; mask=policy["score_mask"]
        if not isinstance(alerts,list) or alerts!=sorted(set(alerts)) or any(type(i)is not int or i<0 or i>=n for i in alerts) or not isinstance(mask,list) or len(mask)!=n or any(type(item)is not bool for item in mask) or {i for i,item in enumerate(mask) if item}!=set(alerts): raise ValueError("proposal policy route invalid")
    if value["backend"]!={"selected_backend":"cpu","reason":"task9_pending","verdict":"cpu_fallback"}: raise ValueError("proposal backend schema invalid")
    return value

@dataclass(frozen=True, slots=True)
class SerializedProposalShard:
    payload: dict[str, object]

def serialize_proposal_shard(raw: RawDataset, *, source_rows: Iterable[object], hard_stages: Iterable[object], low_frequency_rows: Iterable[object], stability: object, policy_alerts: dict[str, Iterable[int]], catalog_entry: dict[str, object]) -> SerializedProposalShard:
    """Narrow typed boundary: raw Task2 identity plus Task4--7 results only."""
    if not isinstance(raw, RawDataset) or raw.dataset != catalog_entry["dataset"]: raise ValueError("typed proposal dataset differs")
    n=len(raw.test_indices)
    payload={"schema":SHARD_SCHEMA,"dataset":raw.dataset,"train_identity":catalog_entry["train"],"test_identity":catalog_entry["test"],
      "source_rows":list(source_rows),"hard_stages":list(hard_stages),"low_frequency_rows":list(low_frequency_rows),"stability":stability,
      "policies":[{"policy_id":name,"order":order,"alerts":sorted(set(policy_alerts[name])),"score_mask":[index in set(policy_alerts[name]) for index in range(n)]} for order,name in enumerate(POLICIES)],
      "backend":{"selected_backend":"cpu","reason":"task9_pending","verdict":"cpu_fallback"}}
    return SerializedProposalShard(validate_proposal_shard(payload,catalog_entry))

def write_proposal_shard(root: Path, serialized: SerializedProposalShard, catalog_entry: dict[str, object]) -> Path:
    if not isinstance(serialized,SerializedProposalShard): raise TypeError("proposal writer requires typed serialized shard")
    shard=validate_proposal_shard(serialized.payload,catalog_entry); path=safe_child(root,SHARD_DIRECTORY,f"{shard['dataset']}_paper.json"); write_create_only(path,canonical_json_bytes(shard)); return path
def freeze_score_root(root: Path, catalog: dict[str, object], shards: Iterable[SerializedProposalShard]) -> Task8FrozenManifest:
    root=safe_root(root); _mkdirs(root); expected={entry["dataset"]:entry for entry in catalog["datasets"]}; supplied=tuple(shards)
    if len(supplied)!=len(expected): raise ValueError("proposal shard coverage incomplete")
    hashes={}
    for shard in supplied:
        if not isinstance(shard,SerializedProposalShard): raise TypeError("freeze requires typed serialized shards")
        name=shard.payload.get("dataset"); entry=expected.get(name)
        if entry is None: raise ValueError("proposal shard dataset differs")
        path=write_proposal_shard(root,shard,entry); hashes[name]=sha256_bytes(path.read_bytes())
    total=sum(entry["test"]["count"] for entry in expected.values())
    if catalog["mode"]=="full" and (len(expected)!=1117 or total!=195900): raise ValueError("full proposal inventory differs")
    payload={"schema":MANIFEST_SCHEMA,"catalog_sha256":catalog["catalog_sha256"],"datasets":sorted(expected),"test_row_count":total,"shard_hashes":hashes}; write_create_only(safe_child(root,MANIFEST_NAME),canonical_json_bytes(payload)); return Task8FrozenManifest(**{**payload,"datasets":tuple(payload["datasets"])})
def verify_score_root(root: Path, catalog: dict[str, object]) -> Task8FrozenManifest:
    root=safe_root(root); value=read_canonical(safe_child(root,MANIFEST_NAME)); manifest=Task8FrozenManifest(**{**value,"datasets":tuple(value["datasets"])})
    if manifest.catalog_sha256!=catalog["catalog_sha256"] or set(manifest.datasets)!={entry["dataset"] for entry in catalog["datasets"]}: raise ValueError("proposal manifest catalog differs")
    for entry in catalog["datasets"]:
        path=safe_child(root,SHARD_DIRECTORY,f"{entry['dataset']}_paper.json"); shard=validate_proposal_shard(read_canonical(path),entry)
        if manifest.shard_hashes[entry["dataset"]]!=sha256_bytes(path.read_bytes()): raise ValueError("proposal shard hash differs")
    return manifest
def verify_parity(root: Path, catalog: dict[str, object], acceptance: dict[str, object]) -> dict[str, object]:
    manifest=verify_score_root(root,catalog); oracle_path=safe_child(Path(acceptance.get("_path", ".")).parent,acceptance["oracle"]["relative_path"])
    if sha256_bytes(oracle_path.read_bytes())!=acceptance["oracle"]["sha256"]: raise ValueError("parity oracle hash differs")
    oracle=read_canonical(oracle_path); expected=acceptance["policies"]
    observed=[]
    for policy in POLICIES:
        coordinates=[]
        for entry in catalog["datasets"]:
            shard=read_canonical(safe_child(root,SHARD_DIRECTORY,f"{entry['dataset']}_paper.json")); row=next(item for item in shard["policies"] if item["policy_id"]==policy); coordinates.extend((entry["dataset"],index) for index in row["alerts"])
        observed.append(tuple(sorted(coordinates)))
    if not isinstance(oracle,dict) or oracle.get("schema")!="paper-task8-coordinate-oracle.v1" or oracle.get("policies")!=[{"policy_id":policy["policy_id"],"coordinates":policy["coordinates"]} for policy in expected]: raise ValueError("parity oracle content differs")
    for got,want in zip(observed,expected,strict=True):
        if len(got)!=want["coordinate_count"] or sha256_bytes(canonical_json_bytes([[d,i] for d,i in got]))!=want["coordinate_sha256"] or [[d,i] for d,i in got]!=want["coordinates"]: raise ValueError("parity coordinate differs")
    report={"schema":PARITY_SCHEMA,"manifest_sha256":sha256_bytes(safe_child(root,MANIFEST_NAME).read_bytes()),"acceptance_sha256":acceptance["acceptance_sha256"],"oracle_sha256":acceptance["oracle"]["sha256"],"result":"exact"}; return report
