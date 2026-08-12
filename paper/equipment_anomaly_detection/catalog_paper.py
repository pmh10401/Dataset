"""Read-only, label-free population catalog authority for pipeline v2."""
from __future__ import annotations
from hashlib import sha256
import os
from pathlib import Path
import sqlite3
from typing import Mapping
from .artifacts_paper import canonical_json_bytes, read_canonical, safe_root, sha256_bytes, write_create_only

CATALOG_SCHEMA="paper-population-catalog.v2"
ACCEPTANCE_SCHEMA="paper-smoke-acceptance.v2"
def _db(path:Path)->dict[str,object]:
    raw=Path(path)
    if raw.is_symlink() or not raw.is_file():raise ValueError("database must be a regular nonsymlink file")
    stat=raw.stat();return {"unresolved_path":str(raw),"resolved_path":str(raw.resolve()),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns,"sha256":sha256(raw.read_bytes()).hexdigest()}
def _identity(dataset:str,rows:list[tuple[int,bytes]])->dict[str,object]:
    indices=[row[0] for row in rows]
    if not rows or indices!=list(range(len(rows))) or any(type(blob)is not bytes or not blob for _,blob in rows):raise ValueError("catalog row identity is invalid")
    return {"count":len(indices),"indices":indices,"row_digest":sha256_bytes(canonical_json_bytes({"dataset":dataset,"indices":indices,"blobs_sha256":[sha256(blob).hexdigest() for _,blob in rows]}))}
def build_catalog(db_path:Path,*,datasets:tuple[str,...]|None=None)->dict[str,object]:
    """Uses Task2's values-only SQL shapes: TRAIN normal values and TEST values."""
    identity=_db(db_path); uri=f"file:{Path(db_path).resolve()}?mode=ro"; connection=sqlite3.connect(uri,uri=True)
    try:
        names=tuple(row[0] for row in connection.execute("SELECT name FROM datasets ORDER BY name"))
        selected=names if datasets is None else tuple(datasets)
        if selected!=tuple(sorted(set(selected))) or not selected or any(d not in names for d in selected):raise ValueError("catalog dataset subset is invalid")
        entries=[]
        for dataset in selected:
            train=connection.execute("SELECT i.instance_index,i.values_blob FROM instances i JOIN datasets d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TRAIN' AND i.label='0' ORDER BY i.instance_index",(dataset,)).fetchall()
            test=connection.execute("SELECT i.instance_index,i.values_blob FROM instances i JOIN datasets d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index",(dataset,)).fetchall()
            entries.append({"dataset":dataset,"train":_identity(dataset,train),"test":_identity(dataset,test)})
    finally:connection.close()
    bare={"schema":CATALOG_SCHEMA,"database":identity,"datasets":entries};return {**bare,"catalog_sha256":sha256_bytes(canonical_json_bytes(bare))}
def write_catalog(path:Path,db_path:Path,*,datasets:tuple[str,...]|None=None)->dict[str,object]:
    value=build_catalog(db_path,datasets=datasets);write_create_only(Path(path),canonical_json_bytes(value));return value
def load_catalog(path:Path,db_path:Path,mode:str)->dict[str,object]:
    value=read_canonical(Path(path));bare={k:v for k,v in value.items() if k!="catalog_sha256"}
    if not isinstance(value,dict) or value.get("schema")!=CATALOG_SCHEMA or value.get("catalog_sha256")!=sha256_bytes(canonical_json_bytes(bare)) or value.get("database")!=_db(db_path):raise ValueError("catalog authority differs")
    entries=value.get("datasets")
    if not isinstance(entries,list) or not entries or tuple(e.get("dataset") for e in entries)!=tuple(sorted(e.get("dataset") for e in entries)):raise ValueError("catalog inventory invalid")
    if mode=="full" and (len(entries)!=1117 or sum(e["test"]["count"] for e in entries)!=195900):raise ValueError("full catalog is unavailable or not official exact inventory")
    return value
def write_acceptance(path:Path,catalog:Mapping[str,object],coordinates:tuple[tuple[tuple[str,int],...],...])->dict[str,object]:
    if len(coordinates)!=4 or any(not policy for policy in coordinates):raise ValueError("acceptance policies must be nonempty")
    entries={e["dataset"]:e for e in catalog["datasets"]}
    normalized=[]
    for policy in coordinates:
        pairs=tuple(sorted(policy))
        if len(set(pairs))!=len(pairs) or any(d not in entries or not isinstance(i,int) or i<0 or i>=entries[d]["test"]["count"] for d,i in pairs):raise ValueError("acceptance coordinate invalid")
        normalized.append([[d,i] for d,i in pairs])
    bare={"schema":ACCEPTANCE_SCHEMA,"catalog_sha256":catalog["catalog_sha256"],"policies":normalized};value={**bare,"acceptance_sha256":sha256_bytes(canonical_json_bytes(bare))};write_create_only(Path(path),canonical_json_bytes(value));return value
def load_acceptance(path:Path,catalog:Mapping[str,object])->dict[str,object]:
    value=read_canonical(Path(path));bare={k:v for k,v in value.items() if k!="acceptance_sha256"}
    if not isinstance(value,dict) or value.get("schema")!=ACCEPTANCE_SCHEMA or value.get("catalog_sha256")!=catalog["catalog_sha256"] or value.get("acceptance_sha256")!=sha256_bytes(canonical_json_bytes(bare)) or not isinstance(value.get("policies"),list) or len(value["policies"])!=4 or any(not policy for policy in value["policies"]):raise ValueError("acceptance authority differs")
    return value
