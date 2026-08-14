"""Pure Exp449 manifest verification primitives; intentionally paper-writer free."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import stat

import numpy as np

from .config_antigravity import FROZEN_EXP449_CONFIG, FROZEN_EXP449_POLICY_KEYS
from .contracts_antigravity import Exp449Context, Exp449ProposalManifest, Exp449ShardDescriptor


_ARTIFACT_DIRECTORY = "stage_artifacts_antigravity"
_FREEZE_DIRECTORY = "proposal_freeze_antigravity"
_SHARDS_DIRECTORY = "shards_antigravity"
_MANIFEST_NAME = "proposal_manifest_antigravity.json"
_SHARD_SCHEMA = "exp449-proposal-shard-antigravity.v1"


def canonical_json_bytes_antigravity(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("Exp449 canonical JSON differs") from error


def canonical_float64_bytes_antigravity(values: object) -> bytes:
    array = np.asarray(values, dtype=np.dtype("<f8"))
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("Exp449 score vector differs")
    return np.ascontiguousarray(array).tobytes(order="C")


def _plain(value: object) -> object:
    if isinstance(value, Mapping): return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, tuple): return [_plain(child) for child in value]
    if isinstance(value, Path): return str(value)
    return value


def _lstat(path: Path):
    try: return path.lstat()
    except FileNotFoundError: return None


def _safe_root(root: Path) -> Path:
    root = Path(root)
    if not root.is_absolute() or ".." in root.parts or "exp448" in root.as_posix().casefold():
        raise ValueError("Exp449 root authority differs")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part; details = _lstat(current)
        if details is not None and stat.S_ISLNK(details.st_mode): raise ValueError("Exp449 symlinked root component is forbidden")
    return root


def _context_sha256(context: Exp449Context) -> str:
    if not isinstance(context, Exp449Context) or context.experiment_id != "Exp449" or context.historical_parity_status != "not_used":
        raise ValueError("Exp449 context differs")
    value = {"experiment_id":context.experiment_id,"mode":context.mode,"run_root":str(context.run_root),"database_sha256":context.database_sha256,"catalog_sha256":context.catalog_sha256,"code_closure":_plain(context.code_closure),"lineage_sha256":context.lineage_sha256,"codemap_sha256":context.codemap_sha256,"interpreter":context.interpreter,"environment":_plain(context.environment),"backend_record":_plain(context.backend_record),"seeds_and_constants":_plain(context.seeds_and_constants),"historical_parity_status":context.historical_parity_status}
    return sha256(canonical_json_bytes_antigravity(value)).hexdigest()


def _catalog_names(catalog: object, mode: str, manifest: Exp449ProposalManifest | None = None) -> tuple[str, ...]:
    if mode not in {"smoke", "official"} or not isinstance(catalog, Mapping) or type(catalog.get("catalog_sha256")) is not str:
        raise ValueError("Exp449 catalog differs")
    bare={key:value for key,value in catalog.items() if key != "catalog_sha256"}
    if sha256(canonical_json_bytes_antigravity(bare)).hexdigest() != catalog["catalog_sha256"]:
        raise ValueError("Exp449 catalog self-hash differs")
    entries=catalog.get("datasets")
    if not isinstance(entries,list): raise ValueError("Exp449 catalog differs")
    names=tuple(item.get("dataset") if isinstance(item,Mapping) else None for item in entries)
    if any(type(name) is not str or not name or "/" in name or "\\" in name for name in names) or names != tuple(sorted(names)) or len(set(names)) != len(names): raise ValueError("Exp449 catalog differs")
    if mode == "smoke":
        if catalog.get("mode") != "smoke" or names != FROZEN_EXP449_CONFIG.smoke_datasets: raise ValueError("Exp449 smoke catalog differs")
    else:
        counts=[item.get("test",{}).get("count") if isinstance(item,Mapping) and isinstance(item.get("test"),Mapping) else None for item in entries]
        if catalog.get("mode") != "full" or len(names) != 1117 or any(type(n) is not int or n < 0 for n in counts) or sum(counts) != 195900: raise ValueError("Exp449 full catalog differs")
    if manifest is not None and tuple(item.dataset for item in manifest.entries) != names: raise ValueError("Exp449 manifest catalog inventory differs")
    return names


def validate_exp449_catalog_antigravity(catalog: object, *, mode: str, manifest: Exp449ProposalManifest | None = None) -> tuple[str, ...]:
    return _catalog_names(catalog, mode, manifest)


def _entry(catalog: Mapping[str, object], dataset: str) -> dict[str, object]:
    rows=[item for item in catalog["datasets"] if isinstance(item,dict) and item.get("dataset")==dataset]
    if len(rows)!=1: raise ValueError("Exp449 catalog shard identity differs")
    return rows[0]


def _relative(dataset: str) -> str:
    return f"{_SHARDS_DIRECTORY}/{sha256(dataset.encode()).hexdigest()}_antigravity/{dataset}_proposal_shard_antigravity.json"


def _integer_list(value: object, n: int, name: str) -> list[int]:
    if not isinstance(value,list) or any(type(x) is not int or x < 0 or x >= n for x in value) or value != sorted(set(value)): raise ValueError(f"Exp449 {name} differs")
    return value


def _scores(value: object, n: int) -> list[float]:
    if not isinstance(value,list) or len(value)!=n or any(type(x) not in (int,float) or not isfinite(float(x)) for x in value): raise ValueError("Exp449 policy scores differ")
    return [float(x) for x in value]


def _scientific(value: object, entry: Mapping[str, object]) -> dict[str, object]:
    required={"schema","dataset","train_identity","test_identity","source_rows","hard_stages","low_frequency_rows","stability","policies","backend"}
    if not isinstance(value,dict) or set(value)!=required or value["schema"]!="paper-task8-proposal-shard.v3" or value["dataset"]!=entry.get("dataset") or value["train_identity"]!=entry.get("train") or value["test_identity"]!=entry.get("test"): raise ValueError("Exp449 scientific payload differs")
    test=entry.get("test"); n=test.get("count") if isinstance(test,Mapping) else None
    if type(n) is not int or n < 0: raise ValueError("Exp449 scientific test count differs")
    stages=value["hard_stages"]
    if not isinstance(stages,list) or len(stages)!=8 or not isinstance(stages[-1],dict) or stages[-1].get("stage_id")!="exp253" or not isinstance(stages[-1].get("lanes"),dict): raise ValueError("Exp449 hard stages differ")
    automatic=_integer_list(stages[-1]["lanes"].get("hard"),n,"hard alerts")
    stability=value["stability"]
    if not isinstance(stability,dict) or set(stability)!={"selected","stable","unstable"}: raise ValueError("Exp449 stability differs")
    def coordinates(name: str) -> list[int]:
        rows=stability[name]
        if not isinstance(rows,list) or any(not isinstance(row,list) or len(row)!=2 or row[0]!=entry["dataset"] for row in rows): raise ValueError("Exp449 stability differs")
        return _integer_list([row[1] for row in rows],n,"stability coordinates")
    selected,stable,unstable=coordinates("selected"),coordinates("stable"),coordinates("unstable")
    if set(stable)&set(unstable) or set(stable)|set(unstable)!=set(selected): raise ValueError("Exp449 stability partition differs")
    policies=value["policies"]
    if not isinstance(policies,list) or [row.get("policy_id") if isinstance(row,dict) else None for row in policies] != list(FROZEN_EXP449_POLICY_KEYS): raise ValueError("Exp449 policy closure differs")
    expected=(set(automatic),set(automatic)|set(selected),set(automatic)|set(stable),set(automatic)|set(unstable))
    for order,(row,want) in enumerate(zip(policies,expected,strict=True)):
        if not isinstance(row,dict) or set(row)!={"policy_id","order","alerts","scores","score_mask"} or row["order"]!=order or _integer_list(row["alerts"],n,"policy alerts") != sorted(want): raise ValueError("Exp449 policy alerts differ")
        scores=_scores(row["scores"],n)
        if not isinstance(row["score_mask"],list) or len(row["score_mask"])!=n or any(type(x) is not bool for x in row["score_mask"]) or row["score_mask"] != [score >= .5 for score in scores]: raise ValueError("Exp449 policy score mask differs")
        if set(row["alerts"]) != {index for index,score in enumerate(scores) if score >= .5}: raise ValueError("Exp449 policy alerts and scores differ")
    return value


def _digests(scientific: Mapping[str, object], dataset: str) -> tuple[dict[str,str],str]:
    result={}
    candidate=None
    for row in scientific["policies"]:
        result[row["policy_id"]]=sha256(canonical_json_bytes_antigravity([[dataset,index] for index in row["alerts"]])).hexdigest()
        if row["policy_id"]==FROZEN_EXP449_CONFIG.policy_id: candidate=sha256(canonical_float64_bytes_antigravity(row["scores"])).hexdigest()
    assert candidate is not None
    return result,candidate


def validate_exp449_proposal_shard_antigravity(value: object, *, entry: dict[str, object], context: Exp449Context, catalog: Mapping[str, object]) -> tuple[dict[str,str],str]:
    if not isinstance(value,dict) or set(value)!={"schema","catalog_sha256","backend_record","scientific_payload"} or value["schema"]!=_SHARD_SCHEMA or value["catalog_sha256"]!=context.catalog_sha256 or value["catalog_sha256"]!=catalog.get("catalog_sha256") or value["backend_record"]!=_plain(context.backend_record): raise ValueError("Exp449 proposal shard authority differs")
    scientific=_scientific(value["scientific_payload"],entry)
    if scientific["backend"] != value["backend_record"]: raise ValueError("Exp449 proposal shard backend differs")
    return _digests(scientific,entry["dataset"])


def _strict(path: Path) -> object:
    details=_lstat(path)
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode): raise ValueError("Exp449 artifact is missing or symlinked")
    raw=path.read_bytes()
    try: value=json.loads(raw)
    except (UnicodeDecodeError,json.JSONDecodeError) as error: raise ValueError("Exp449 artifact JSON differs") from error
    if raw != canonical_json_bytes_antigravity(value): raise ValueError("Exp449 artifact noncanonical")
    return value


def _manifest(value: object) -> Exp449ProposalManifest:
    if not isinstance(value,dict) or set(value)!={"schema","context_sha256","catalog_sha256","backend_record","entries","historical_parity_status"} or not isinstance(value.get("entries"),list): raise ValueError("Exp449 proposal manifest schema differs")
    return Exp449ProposalManifest(value["schema"],value["context_sha256"],value["catalog_sha256"],value["backend_record"],tuple(Exp449ShardDescriptor(**item) for item in value["entries"]),value["historical_parity_status"])


def verify_exp449_proposal_root_antigravity(root: Path, *, context: Exp449Context, catalog: dict[str,object]) -> Exp449ProposalManifest:
    root=_safe_root(root)
    if "antigravity" not in root.name or Path(context.run_root).resolve(strict=False)!=root.resolve(strict=False): raise ValueError("Exp449 root authority differs")
    freeze=root/_ARTIFACT_DIRECTORY/_FREEZE_DIRECTORY
    manifest=_manifest(_strict(freeze/_MANIFEST_NAME)); names=_catalog_names(catalog,context.mode,manifest)
    if manifest.context_sha256!=_context_sha256(context) or manifest.catalog_sha256!=context.catalog_sha256 or manifest.catalog_sha256!=catalog.get("catalog_sha256") or _plain(manifest.backend_record)!=_plain(context.backend_record): raise ValueError("Exp449 proposal manifest authority differs")
    declared={f"{_ARTIFACT_DIRECTORY}/{_FREEZE_DIRECTORY}/{_MANIFEST_NAME}"}
    for descriptor in manifest.entries:
        if descriptor.relative_path != _relative(descriptor.dataset): raise ValueError("Exp449 shard path differs")
        rel=f"{_ARTIFACT_DIRECTORY}/{_FREEZE_DIRECTORY}/{descriptor.relative_path}"; declared.add(rel); path=root/rel; details=_lstat(path)
        if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode) or details.st_size!=descriptor.size or sha256(path.read_bytes()).hexdigest()!=descriptor.sha256: raise ValueError("Exp449 shard hash differs")
        coords,candidate=validate_exp449_proposal_shard_antigravity(_strict(path),entry=_entry(catalog,descriptor.dataset),context=context,catalog=catalog)
        if dict(descriptor.policy_coordinate_sha256)!=coords or descriptor.candidate_c_score_sha256!=candidate: raise ValueError("Exp449 shard digest differs")
    declared_nodes=set(declared)
    for item in tuple(declared):
        parent=PurePosixPath(item).parent
        while parent != PurePosixPath("."):
            declared_nodes.add(parent.as_posix()); parent=parent.parent
    found={item.relative_to(root).as_posix() for item in root.rglob("*")}
    if found!=declared_nodes or any("antigravity" not in part for item in found for part in item.split("/")): raise ValueError("Exp449 root inventory differs")
    return manifest


__all__=["canonical_float64_bytes_antigravity","canonical_json_bytes_antigravity","validate_exp449_catalog_antigravity","validate_exp449_proposal_shard_antigravity","verify_exp449_proposal_root_antigravity"]
