"""Create-only Task 8 artifacts with one typed Task1--7 shard writer."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json, math, os, stat, time
from pathlib import Path

import numpy as np

from .contracts_paper import FrozenTask8BackendDecision, LowFrequencyResult, RawDataset, Task8FrozenManifest
from .calibration_paper import upper_tail_pvalues
from .data_paper import ProgressCallback, emit_progress_paper
from .hard_route_paper import EXP234_SOURCES_PAPER, MAX_TOP_PAPER, HardRouteResult, RouteLanes, SourceExpansion
from .lineage_paper import LINEAGE_NODES, read_strict_json
from .routing_paper import PolicyBundle
from .sources_paper.hard_sources_paper import EXPECTED_HARD_SOURCES, SOURCE_ALPHA_PAPER, HardSourceBundle, HardSourceResult
from .sources_paper.stability_paper import StabilityPartition

SHARD_SCHEMA="paper-task8-proposal-shard.v3"; MANIFEST_SCHEMA="paper-task8-proposal-manifest.v1"; PARITY_SCHEMA="paper-task8-parity.v1"
ARTIFACT_DIRECTORY="stage_artifacts_paper"; FREEZE_STAGE="proposal_freeze"; SHARD_DIRECTORY="shards"; MANIFEST_NAME="proposal_manifest_paper.json"
TASK4_TASK5_SUCCESS_SCHEMA="paper-task4-task5-success.v1"; TASK4_TASK5_ATTEMPT_SCHEMA="paper-task4-task5-attempt.v1"; PROPOSAL_WORK_DIRECTORY="proposal_work"; TASK4_TASK5_SUCCESS_DIRECTORY="task4_task5_success"; TASK4_TASK5_ATTEMPT_DIRECTORY="task4_task5_attempts"; TASK4_TASK5_SUCCESS_NAME="task4_task5_success_antigravity.json"
TASK4_TASK5_SOURCE_ORIGIN_IDS={"rocket_exp40":"rocket_exp40","exp55_best":"exp55_best","exp56_best":"exp56_best","exp84":"exp84_family_guard","rocket_block_b":"rocket_block_b","canonical_normal_state":"canonical_normal_state"}
POLICIES=("hard_only","hard_plus_source","hard_plus_stable","hard_plus_unstable")

@dataclass(frozen=True, slots=True, init=False)
class Task4Task5SuccessAuthority:
    """Run-bound identities required to reuse one Task4+5 success record."""
    context_sha256: str
    catalog_sha256: str
    backend: FrozenTask8BackendDecision
    scientific_config_sha256: str
    code_closure_sha256: str

    def __init__(self, *_args, **_kwargs) -> None:
        raise TypeError("Task4+5 success authority must be derived from verified context")

    @classmethod
    def from_verified_context(cls, context: object, catalog: object, acceleration_manifest: object) -> "Task4Task5SuccessAuthority":
        """Derive the only reusable-record authority from runner-verified inputs."""
        if not isinstance(context, dict) or not isinstance(catalog, dict):
            raise ValueError("Task4+5 verified context is invalid")
        context_sha256 = context.get("context_sha256")
        base = {key: value for key, value in context.items() if key != "context_sha256"}
        if type(context_sha256) is not str or context_sha256 != sha256_bytes(canonical_json_bytes(base)) or context.get("catalog_sha256") != catalog.get("catalog_sha256"):
            raise ValueError("Task4+5 verified context differs")
        closure = context.get("code_closure")
        if not isinstance(closure, dict) or not closure or any(type(path) is not str or type(digest) is not str or len(digest) != 64 for path, digest in closure.items()):
            raise ValueError("Task4+5 code closure differs")
        from .acceleration_paper import verify_acceleration_manifest
        verified = verify_acceleration_manifest(acceleration_manifest, context_sha256)
        instance = object.__new__(cls)
        object.__setattr__(instance, "context_sha256", context_sha256)
        object.__setattr__(instance, "catalog_sha256", catalog["catalog_sha256"])
        object.__setattr__(instance, "backend", FrozenTask8BackendDecision.from_acceleration_manifest(verified))
        object.__setattr__(instance, "scientific_config_sha256", _task4_task5_scientific_config_sha256())
        object.__setattr__(instance, "code_closure_sha256", sha256_bytes(canonical_json_bytes(closure)))
        return instance

@dataclass(frozen=True, slots=True)
class Task4Task5SuccessRecord:
    """Allowlisted typed reconstruction of one cached per-dataset Task4+5 result."""
    sources: HardSourceBundle
    route: HardRouteResult

@dataclass(frozen=True, slots=True)
class Task4Task5AttemptRecord:
    """Operational-only parent evidence for a failed Task4+5 coordinate attempt."""
    dataset: str
    wave: int
    attempt: int
    exception_type: str
    exception_message: str

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
def _fsync_directory(path:Path)->None:
    descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY|getattr(os,"O_CLOEXEC",0))
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
def _stale_tmp_candidates(destination:Path)->list[Path]:
    parent=destination.parent; prefix=f".{destination.name}."; suffix=".tmp_antigravity"
    if _lstat(parent) is None:return []
    return [item for item in parent.iterdir() if item.name.startswith(prefix) and item.name.endswith(suffix)]
def _recover_matching_stale_tmp(destination:Path,data:bytes)->None:
    candidates=_stale_tmp_candidates(destination)
    if not candidates:return
    if len(candidates)>1:raise ValueError("stale create-only temp is ambiguous")
    temporary=candidates[0]; details=_lstat(temporary)
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):raise ValueError("stale create-only temp is not a regular file")
    published=_lstat(destination)
    if published is None or stat.S_ISLNK(published.st_mode) or not stat.S_ISREG(published.st_mode):raise ValueError("create-only destination is not a regular file")
    if (details.st_dev,details.st_ino)!=(published.st_dev,published.st_ino):raise ValueError("stale create-only temp is not the published inode")
    if temporary.read_bytes()!=data:raise ValueError("stale create-only temp differs")
    temporary.unlink(); _fsync_directory(destination.parent)
def write_create_only(path:Path,data:bytes,*,reuse_identical:bool=False)->None:
    destination=Path(path)
    details=_lstat(destination)
    if details is not None:
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):raise ValueError("create-only destination is not a regular file")
        if not reuse_identical:raise FileExistsError(f"create-only artifact already exists: {destination}")
        if destination.read_bytes()!=data:raise ValueError("existing destination differs")
        _recover_matching_stale_tmp(destination,data);return
    _mkdirs(destination.parent)
    temporary=destination.with_name(f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp_antigravity")
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0),0o600)
    try:
        view=memoryview(data); offset=0
        while offset<len(data):
            written=os.write(descriptor,view[offset:])
            if written<=0:raise OSError("create-only write returned no bytes")
            offset+=written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor); temporary.unlink(missing_ok=True); raise
    os.close(descriptor)
    try:os.link(temporary,destination)
    except FileExistsError as error:
        existing=_lstat(destination)
        if existing is None or stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):raise ValueError("create-only destination is not a regular file") from error
        if destination.read_bytes()!=data:raise ValueError("existing destination differs") from error
        temporary.unlink(missing_ok=True); _fsync_directory(destination.parent)
        if not reuse_identical:raise FileExistsError(f"create-only artifact already exists: {destination}") from error
        _recover_matching_stale_tmp(destination,data);return
    except BaseException:
        if _lstat(destination) is None:temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(destination.parent); temporary.unlink(missing_ok=True); _fsync_directory(destination.parent)
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
    for blobs,rows in ((raw.train_blobs,raw.train_rows),(raw.test_blobs,raw.test_rows)):
        for blob,row in zip(blobs,rows,strict=True):
            decoded=np.frombuffer(blob,dtype="<f4")
            array=np.asarray(row)
            if array.dtype!=np.dtype("<f4") or array.ndim!=1 or array.shape!=decoded.shape or array.tobytes()!=blob:
                raise ValueError("typed decoded raw rows differ")
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

def _task4_task5_dataset_directory(root: Path, dataset: str) -> Path:
    if type(dataset) is not str or not dataset:
        raise ValueError("Task4+5 success dataset is invalid")
    token = sha256(dataset.encode("utf-8")).hexdigest()
    return safe_child(root, ARTIFACT_DIRECTORY, PROPOSAL_WORK_DIRECTORY, TASK4_TASK5_SUCCESS_DIRECTORY, f"{token}_antigravity")


def _task4_task5_success_path(root: Path, dataset: str) -> Path:
    return safe_child(_task4_task5_dataset_directory(root, dataset), TASK4_TASK5_SUCCESS_NAME)


def _task4_task5_scientific_config_sha256() -> str:
    """Bind the frozen Task4 source identities and Task5 route constants, not caller text."""
    origins = {node.node_id: _lineage_record(node) for node in LINEAGE_NODES}
    return sha256_bytes(canonical_json_bytes({"source_names": list(EXPECTED_HARD_SOURCES), "source_alpha": SOURCE_ALPHA_PAPER,
                                               "source_origins": {name: origins[TASK4_TASK5_SOURCE_ORIGIN_IDS[name]] for name in EXPECTED_HARD_SOURCES},
                                               "route": {"exp234_sources": list(EXP234_SOURCES_PAPER), "max_top": MAX_TOP_PAPER}}))


def _lineage_record(origin: object) -> dict[str, object]:
    if origin not in LINEAGE_NODES:
        raise ValueError("Task4+5 source origin is not allowlisted")
    return {"node_id": origin.node_id, "stage": origin.stage, "input_kind": origin.input_kind, "module": origin.module,
            "symbols": list(origin.symbols), "source_sha256": origin.source_sha256, "paper_module": origin.paper_module,
            "label_access": origin.label_access, "parity_oracle": origin.parity_oracle}


def _lineage_from_record(value: object):
    required = {"node_id", "stage", "input_kind", "module", "symbols", "source_sha256", "paper_module", "label_access", "parity_oracle"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Task4+5 source origin schema differs")
    for origin in LINEAGE_NODES:
        if value == _lineage_record(origin):
            return origin
    raise ValueError("Task4+5 source origin is not allowlisted")


def _task4_task5_float_array(values: object, n: int, name: str, *, unit_interval: bool = False) -> list[float]:
    array = np.asarray(values)
    if array.dtype != np.dtype("float64"):
        raise ValueError(f"Task4+5 {name} must be float64")
    result = _array(values, n, name)
    if unit_interval and any(value <= 0.0 or value > 1.0 for value in result):
        raise ValueError(f"Task4+5 {name} domain differs")
    return result


def _task4_task5_readonly_float(values: object, n: int, name: str, *, unit_interval: bool = False) -> np.ndarray:
    result = np.array(_task4_task5_float_array(values, n, name, unit_interval=unit_interval), dtype=np.float64)
    result.setflags(write=False)
    return result


def _task4_task5_readonly_bool(values: object, n: int, name: str) -> np.ndarray:
    result = np.array(_bools(values, n, name), dtype=bool)
    result.setflags(write=False)
    return result


def _task4_task5_source_payload(sources: HardSourceBundle, n_train: int, n_test: int) -> list[dict[str, object]]:
    if not isinstance(sources, HardSourceBundle) or tuple(sources.source_names) != EXPECTED_HARD_SOURCES or set(sources.by_name) != set(EXPECTED_HARD_SOURCES):
        raise TypeError("typed Task4 source bundle is invalid")
    rows: list[dict[str, object]] = []
    for order, name in enumerate(EXPECTED_HARD_SOURCES):
        result = sources.by_name[name]
        if not isinstance(result, HardSourceResult) or result.name != name or result.origin.node_id != TASK4_TASK5_SOURCE_ORIGIN_IDS[name] or type(result.eligible) is not bool or type(result.eligibility_reason) is not str or not result.eligibility_reason:
            raise ValueError("typed Task4 source result differs")
        threshold = float(result.threshold)
        if type(result.threshold) not in (int, float) or not math.isfinite(threshold) or threshold != SOURCE_ALPHA_PAPER[name]:
            raise ValueError("typed Task4 source threshold differs")
        pvalue = _task4_task5_float_array(result.test_pvalue, n_test, "source pvalue", unit_interval=True)
        native = _bools(result.test_native_pass, n_test, "source pass")
        if result.eligible:
            train_oof = _task4_task5_float_array(result.train_oof_score, n_train, "source TRAIN OOF score")
            test_score = _task4_task5_float_array(result.test_score, n_test, "source TEST score")
            if result.eligibility_reason != "eligible" or native != [value <= threshold for value in pvalue] or not np.array_equal(np.asarray(pvalue), upper_tail_pvalues(np.asarray(train_oof), np.asarray(test_score))):
                raise ValueError("typed Task4 eligible source relation differs")
        else:
            _task4_task5_float_array(result.train_oof_score, 0, "source abstained TRAIN OOF score")
            _task4_task5_float_array(result.test_score, 0, "source abstained TEST score")
            if np.asarray(result.train_oof_score).size or np.asarray(result.test_score).size or pvalue != [1.0] * n_test or any(native):
                raise ValueError("typed Task4 abstained source relation differs")
            train_oof, test_score = [], []
        rows.append({"source_id": name, "order": order, "origin": _lineage_record(result.origin), "train_oof_score": train_oof,
                     "test_score": test_score, "test_pvalue": pvalue, "threshold": threshold, "test_native_pass": native,
                     "eligible": result.eligible, "eligibility_reason": result.eligibility_reason})
    return rows


def _task4_task5_sources_from_payload(value: object, n_train: int, n_test: int) -> HardSourceBundle:
    if not isinstance(value, list) or [row.get("source_id") if isinstance(row, dict) else None for row in value] != list(EXPECTED_HARD_SOURCES):
        raise ValueError("Task4+5 source order differs")
    results: dict[str, HardSourceResult] = {}
    required = {"source_id", "order", "origin", "train_oof_score", "test_score", "test_pvalue", "threshold", "test_native_pass", "eligible", "eligibility_reason"}
    for order, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != required or row["order"] != order or type(row["eligible"]) is not bool or type(row["eligibility_reason"]) is not str or not row["eligibility_reason"]:
            raise ValueError("Task4+5 source schema differs")
        threshold = row["threshold"]
        if type(threshold) not in (int, float) or not math.isfinite(float(threshold)) or float(threshold) != SOURCE_ALPHA_PAPER[row["source_id"]]:
            raise ValueError("Task4+5 source threshold differs")
        pvalue = _task4_task5_readonly_float(row["test_pvalue"], n_test, "source pvalue", unit_interval=True)
        native = _task4_task5_readonly_bool(row["test_native_pass"], n_test, "source pass")
        if row["eligible"]:
            train_oof = _task4_task5_readonly_float(row["train_oof_score"], n_train, "source TRAIN OOF score")
            test_score = _task4_task5_readonly_float(row["test_score"], n_test, "source TEST score")
            if row["eligibility_reason"] != "eligible" or not np.array_equal(native, pvalue <= float(threshold)) or not np.array_equal(pvalue, upper_tail_pvalues(train_oof, test_score)):
                raise ValueError("Task4+5 eligible source relation differs")
        else:
            if row["train_oof_score"] != [] or row["test_score"] != [] or not np.array_equal(pvalue, np.ones(n_test)) or np.any(native):
                raise ValueError("Task4+5 abstained source relation differs")
            train_oof = np.empty(0, dtype=np.float64); test_score = np.empty(0, dtype=np.float64)
            train_oof.setflags(write=False); test_score.setflags(write=False)
        origin = _lineage_from_record(row["origin"])
        if origin.node_id != TASK4_TASK5_SOURCE_ORIGIN_IDS[row["source_id"]]:
            raise ValueError("Task4+5 source origin differs")
        results[row["source_id"]] = HardSourceResult(row["source_id"], origin, train_oof, test_score, pvalue, float(threshold), native, row["eligible"], row["eligibility_reason"])
    return HardSourceBundle(EXPECTED_HARD_SOURCES, results)


def _task4_task5_lanes(value: object, n: int, name: str) -> dict[str, list[int]]:
    if not isinstance(value, RouteLanes):
        raise TypeError(f"Task4+5 {name} lanes are invalid")
    return _lanes(value, n)


def _task4_task5_route_payload(route: HardRouteResult, n_test: int) -> dict[str, object]:
    if not isinstance(route, HardRouteResult) or route.n_test != n_test:
        raise TypeError("typed Task5 route result is invalid")
    return {"n_test": route.n_test, "exp74d": _task4_task5_lanes(route.exp74d, n_test, "exp74d"), "exp89": _indices(route.exp89, n_test, "exp89"),
            "exp90": _indices(route.exp90, n_test, "exp90"), "exp93": _indices(route.exp93, n_test, "exp93"), "exp137": _task4_task5_lanes(route.exp137, n_test, "exp137"),
            "exp234": {"minimum_two": _indices(route.exp234.minimum_two, n_test, "exp234"), "minimum_three": _indices(route.exp234.minimum_three, n_test, "exp234")},
            "exp236": _task4_task5_lanes(route.exp236, n_test, "exp236"), "promoted": _indices(route.promoted, n_test, "promoted"),
            "exp253": _task4_task5_lanes(route.exp253, n_test, "exp253"), "exact_normal_removed": _indices(route.exact_normal_removed, n_test, "exact normal removed"),
            "automatic_hard": _indices(route.automatic_hard, n_test, "automatic hard")}


def _task4_task5_indices_from_payload(value: object, n: int, name: str) -> frozenset[int]:
    if not isinstance(value, list) or value != sorted(set(value)) or any(type(item) is not int or item < 0 or item >= n for item in value):
        raise ValueError(f"Task4+5 {name} differs")
    return frozenset(value)


def _task4_task5_lanes_from_payload(value: object, n: int, name: str) -> RouteLanes:
    if not isinstance(value, dict) or set(value) != {"hard", "standard", "priority"}:
        raise ValueError(f"Task4+5 {name} schema differs")
    return RouteLanes(_task4_task5_indices_from_payload(value["hard"], n, name), _task4_task5_indices_from_payload(value["standard"], n, name), _task4_task5_indices_from_payload(value["priority"], n, name))


def _task4_task5_route_from_payload(value: object, n_test: int) -> HardRouteResult:
    required = {"n_test", "exp74d", "exp89", "exp90", "exp93", "exp137", "exp234", "exp236", "promoted", "exp253", "exact_normal_removed", "automatic_hard"}
    if not isinstance(value, dict) or set(value) != required or value["n_test"] != n_test:
        raise ValueError("Task4+5 route schema differs")
    exp234 = value["exp234"]
    if not isinstance(exp234, dict) or set(exp234) != {"minimum_two", "minimum_three"}:
        raise ValueError("Task4+5 exp234 schema differs")
    return HardRouteResult(_task4_task5_lanes_from_payload(value["exp74d"], n_test, "exp74d"), _task4_task5_indices_from_payload(value["exp89"], n_test, "exp89"),
                           _task4_task5_indices_from_payload(value["exp90"], n_test, "exp90"), _task4_task5_indices_from_payload(value["exp93"], n_test, "exp93"),
                           _task4_task5_lanes_from_payload(value["exp137"], n_test, "exp137"), SourceExpansion(_task4_task5_indices_from_payload(exp234["minimum_two"], n_test, "exp234"), _task4_task5_indices_from_payload(exp234["minimum_three"], n_test, "exp234")),
                           _task4_task5_lanes_from_payload(value["exp236"], n_test, "exp236"), _task4_task5_indices_from_payload(value["promoted"], n_test, "promoted"),
                           _task4_task5_lanes_from_payload(value["exp253"], n_test, "exp253"), _task4_task5_indices_from_payload(value["exact_normal_removed"], n_test, "exact normal removed"),
                           _task4_task5_indices_from_payload(value["automatic_hard"], n_test, "automatic hard"), n_test)


def _task4_task5_payload(raw: RawDataset, sources: HardSourceBundle, route: HardRouteResult, entry: dict[str, object]) -> dict[str, object]:
    _raw_identity(raw, entry)
    return {"sources": _task4_task5_source_payload(sources, entry["train"]["count"], entry["test"]["count"]), "route": _task4_task5_route_payload(route, entry["test"]["count"])}


def _task4_task5_success_envelope(catalog: dict[str, object], authority: Task4Task5SuccessAuthority, raw: RawDataset, sources: HardSourceBundle, route: HardRouteResult) -> dict[str, object]:
    if not isinstance(catalog, dict) or catalog.get("catalog_sha256") != authority.catalog_sha256 or not isinstance(raw, RawDataset):
        raise ValueError("Task4+5 record catalog authority differs")
    entry = _catalog_entry(catalog, raw.dataset)
    payload = _task4_task5_payload(raw, sources, route, entry)
    return {"schema": TASK4_TASK5_SUCCESS_SCHEMA, "context_sha256": authority.context_sha256, "catalog_sha256": authority.catalog_sha256, "raw_identity": entry,
            "backend": authority.backend.artifact_record(), "scientific_config_sha256": authority.scientific_config_sha256, "code_closure_sha256": authority.code_closure_sha256,
            "dataset": raw.dataset, "payload": payload, "payload_sha256": sha256_bytes(canonical_json_bytes(payload))}


def _task4_task5_namespace(path: Path, catalog: dict[str, object]) -> None:
    details = _lstat(path.parent)
    success_root = path.parent.parent
    expected = {f"{sha256(entry['dataset'].encode('utf-8')).hexdigest()}_antigravity" for entry in catalog["datasets"]}
    actual = {item.name for item in success_root.iterdir()} if success_root.is_dir() else set()
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode) or {item.name for item in path.parent.iterdir()} != {TASK4_TASK5_SUCCESS_NAME} or not actual <= expected:
        raise ValueError("Task4+5 success namespace differs")
    for child in success_root.iterdir():
        child_details = _lstat(child)
        if child_details is None or stat.S_ISLNK(child_details.st_mode) or not stat.S_ISDIR(child_details.st_mode) or {item.name for item in child.iterdir()} != {TASK4_TASK5_SUCCESS_NAME}:
            raise ValueError("Task4+5 success namespace differs")


def write_task4_task5_success_record(root: Path, catalog: dict[str, object], authority: Task4Task5SuccessAuthority, raw: RawDataset, sources: HardSourceBundle, route: HardRouteResult) -> Path:
    """Create one context-bound, canonical Task4+5 success record after typed round-trip validation."""
    if not isinstance(authority, Task4Task5SuccessAuthority):
        raise TypeError("Task4+5 success authority is required")
    envelope = _task4_task5_success_envelope(catalog, authority, raw, sources, route)
    entry = _catalog_entry(catalog, raw.dataset)
    reread_sources = _task4_task5_sources_from_payload(envelope["payload"]["sources"], entry["train"]["count"], entry["test"]["count"])
    reread_route = _task4_task5_route_from_payload(envelope["payload"]["route"], entry["test"]["count"])
    if canonical_json_bytes(_task4_task5_payload(raw, reread_sources, reread_route, entry)) != canonical_json_bytes(envelope["payload"]):
        raise ValueError("Task4+5 success typed round-trip differs")
    path = _task4_task5_success_path(root, raw.dataset)
    write_create_only(path, canonical_json_bytes(envelope), reuse_identical=True)
    return path


def read_task4_task5_success_record(root: Path, catalog: dict[str, object], authority: Task4Task5SuccessAuthority, raw: RawDataset) -> Task4Task5SuccessRecord:
    """Read exactly one allowlisted Task4+5 success record under the supplied run authority."""
    if not isinstance(authority, Task4Task5SuccessAuthority) or not isinstance(raw, RawDataset) or not isinstance(catalog, dict) or catalog.get("catalog_sha256") != authority.catalog_sha256:
        raise ValueError("Task4+5 record authority differs")
    entry = _catalog_entry(catalog, raw.dataset)
    _raw_identity(raw, entry)
    path = _task4_task5_success_path(root, raw.dataset)
    _task4_task5_namespace(path, catalog)
    value = read_canonical(path)
    _finite(value)
    required = {"schema", "context_sha256", "catalog_sha256", "raw_identity", "backend", "scientific_config_sha256", "code_closure_sha256", "dataset", "payload", "payload_sha256"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != TASK4_TASK5_SUCCESS_SCHEMA or value["dataset"] != raw.dataset:
        raise ValueError("Task4+5 success record schema differs")
    if value["context_sha256"] != authority.context_sha256 or value["catalog_sha256"] != authority.catalog_sha256 or value["scientific_config_sha256"] != authority.scientific_config_sha256 or value["code_closure_sha256"] != authority.code_closure_sha256:
        raise ValueError("Task4+5 success record authority differs")
    if value["raw_identity"] != entry:
        raise ValueError("Task4+5 success raw identity differs")
    try:
        backend = FrozenTask8BackendDecision.from_artifact_record(value["backend"])
    except ValueError as error:
        raise ValueError("Task4+5 success backend differs") from error
    if backend != authority.backend:
        raise ValueError("Task4+5 success backend differs")
    if type(value["payload_sha256"]) is not str or value["payload_sha256"] != sha256_bytes(canonical_json_bytes(value["payload"])):
        raise ValueError("Task4+5 success payload hash differs")
    sources = _task4_task5_sources_from_payload(value["payload"].get("sources") if isinstance(value["payload"], dict) else None, entry["train"]["count"], entry["test"]["count"])
    route = _task4_task5_route_from_payload(value["payload"].get("route") if isinstance(value["payload"], dict) else None, entry["test"]["count"])
    if set(value["payload"]) != {"sources", "route"} or canonical_json_bytes(_task4_task5_payload(raw, sources, route, entry)) != canonical_json_bytes(value["payload"]):
        raise ValueError("Task4+5 success typed round-trip differs")
    return Task4Task5SuccessRecord(sources, route)


def _task4_task5_attempt_root(root: Path) -> Path:
    return safe_child(root, ARTIFACT_DIRECTORY, PROPOSAL_WORK_DIRECTORY, TASK4_TASK5_ATTEMPT_DIRECTORY)


def _task4_task5_attempt_path(root: Path, attempt: int) -> Path:
    return safe_child(_task4_task5_attempt_root(root), f"attempt_{attempt:06d}_antigravity.json")


def _task4_task5_attempt_envelope(catalog: dict[str, object], authority: Task4Task5SuccessAuthority, raw: RawDataset, wave: int, attempt: int, error: Exception) -> dict[str, object]:
    if type(wave) is not int or wave < 1 or type(attempt) is not int or attempt < 1 or not isinstance(error, Exception):
        raise ValueError("Task4+5 attempt identity is invalid")
    entry = _catalog_entry(catalog, raw.dataset)
    _raw_identity(raw, entry)
    payload = {"schema": TASK4_TASK5_ATTEMPT_SCHEMA, "context_sha256": authority.context_sha256, "catalog_sha256": authority.catalog_sha256,
               "raw_identity": entry, "backend": authority.backend.artifact_record(), "scientific_config_sha256": authority.scientific_config_sha256,
               "code_closure_sha256": authority.code_closure_sha256, "dataset": raw.dataset, "wave": wave, "attempt": attempt,
               "exception_type": type(error).__name__, "exception_message": str(error)}
    return {**payload, "record_sha256": sha256_bytes(canonical_json_bytes(payload))}


def append_task4_task5_attempt_record(root: Path, catalog: dict[str, object], authority: Task4Task5SuccessAuthority, raw: RawDataset, wave: int, error: Exception) -> Path:
    """Append one parent-written failure record; attempts never authorize scientific success."""
    if not isinstance(authority, Task4Task5SuccessAuthority):
        raise TypeError("Task4+5 attempt authority is required")
    directory = _task4_task5_attempt_root(root)
    details = _lstat(directory)
    if details is not None and (stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode)):
        raise ValueError("Task4+5 attempt namespace differs")
    existing = sorted(directory.iterdir()) if details is not None else []
    names = [item.name for item in existing]
    expected = [f"attempt_{index:06d}_antigravity.json" for index in range(1, len(names) + 1)]
    if names != expected:
        raise ValueError("Task4+5 attempt namespace differs")
    attempt = len(names) + 1
    path = _task4_task5_attempt_path(root, attempt)
    write_create_only(path, canonical_json_bytes(_task4_task5_attempt_envelope(catalog, authority, raw, wave, attempt, error)))
    return path


def verify_task4_task5_attempt_namespace(root: Path, catalog: dict[str, object], authority: Task4Task5SuccessAuthority, raw_by_dataset: dict[str, RawDataset]) -> tuple[Task4Task5AttemptRecord, ...]:
    """Verify the registered append-only failure ledger without mistaking it for success."""
    directory = _task4_task5_attempt_root(root)
    details = _lstat(directory)
    if details is None:
        return ()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("Task4+5 attempt namespace differs")
    paths = sorted(directory.iterdir())
    if [path.name for path in paths] != [f"attempt_{index:06d}_antigravity.json" for index in range(1, len(paths) + 1)]:
        raise ValueError("Task4+5 attempt namespace differs")
    records: list[Task4Task5AttemptRecord] = []
    required = {"schema", "context_sha256", "catalog_sha256", "raw_identity", "backend", "scientific_config_sha256", "code_closure_sha256", "dataset", "wave", "attempt", "exception_type", "exception_message", "record_sha256"}
    for index, path in enumerate(paths, 1):
        value = read_canonical(path)
        if not isinstance(value, dict) or set(value) != required or value.get("schema") != TASK4_TASK5_ATTEMPT_SCHEMA or value.get("attempt") != index:
            raise ValueError("Task4+5 attempt record differs")
        payload = {key: child for key, child in value.items() if key != "record_sha256"}
        if value["record_sha256"] != sha256_bytes(canonical_json_bytes(payload)) or value["context_sha256"] != authority.context_sha256 or value["catalog_sha256"] != authority.catalog_sha256 or value["scientific_config_sha256"] != authority.scientific_config_sha256 or value["code_closure_sha256"] != authority.code_closure_sha256:
            raise ValueError("Task4+5 attempt authority differs")
        dataset = value["dataset"]
        if dataset not in raw_by_dataset or type(value["wave"]) is not int or value["wave"] < 1 or type(value["exception_type"]) is not str or not value["exception_type"] or type(value["exception_message"]) is not str:
            raise ValueError("Task4+5 attempt record differs")
        entry = _catalog_entry(catalog, dataset)
        _raw_identity(raw_by_dataset[dataset], entry)
        if value["raw_identity"] != entry:
            raise ValueError("Task4+5 attempt raw identity differs")
        try:
            backend = FrozenTask8BackendDecision.from_artifact_record(value["backend"])
        except ValueError as error:
            raise ValueError("Task4+5 attempt backend differs") from error
        if backend != authority.backend:
            raise ValueError("Task4+5 attempt backend differs")
        records.append(Task4Task5AttemptRecord(dataset, value["wave"], index, value["exception_type"], value["exception_message"]))
    return tuple(records)


def verify_task4_task5_success_namespace(root: Path, catalog: dict[str, object], authority: Task4Task5SuccessAuthority, raw_by_dataset: dict[str, RawDataset]) -> tuple[str, ...]:
    """Accept only an empty or fully readable parent-published Task4+5 work inventory."""
    if not isinstance(authority, Task4Task5SuccessAuthority) or not isinstance(catalog, dict) or catalog.get("catalog_sha256") != authority.catalog_sha256 or not isinstance(raw_by_dataset, dict):
        raise ValueError("Task4+5 success namespace authority differs")
    datasets = {entry["dataset"] for entry in catalog.get("datasets", ()) if isinstance(entry, dict) and type(entry.get("dataset")) is str}
    if not datasets or set(raw_by_dataset) != datasets:
        raise ValueError("Task4+5 success namespace raw inventory differs")
    work = safe_child(root, ARTIFACT_DIRECTORY, PROPOSAL_WORK_DIRECTORY)
    details = _lstat(work)
    if details is None:
        return ()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("Task4+5 success namespace differs")
    children = {item.name for item in work.iterdir()}
    if not children:
        return ()
    if not children <= {TASK4_TASK5_SUCCESS_DIRECTORY, TASK4_TASK5_ATTEMPT_DIRECTORY}:
        raise ValueError("Task4+5 success namespace differs")
    if TASK4_TASK5_ATTEMPT_DIRECTORY in children:
        verify_task4_task5_attempt_namespace(root, catalog, authority, raw_by_dataset)
    if TASK4_TASK5_SUCCESS_DIRECTORY not in children:
        return ()
    success_root = safe_child(work, TASK4_TASK5_SUCCESS_DIRECTORY)
    details = _lstat(success_root)
    if details is None:
        return ()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("Task4+5 success namespace differs")
    tokens = {f"{sha256(dataset.encode('utf-8')).hexdigest()}_antigravity": dataset for dataset in datasets}
    present = {item.name for item in success_root.iterdir()}
    if not present <= set(tokens):
        raise ValueError("Task4+5 success namespace differs")
    for token in present:
        directory = safe_child(success_root, token)
        details = _lstat(directory)
        if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode) or {item.name for item in directory.iterdir()} != {TASK4_TASK5_SUCCESS_NAME}:
            raise ValueError("Task4+5 success namespace differs")
        read_task4_task5_success_record(root, catalog, authority, raw_by_dataset[tokens[token]])
    return tuple(sorted(tokens[token] for token in present))

def _typed_payload(raw:RawDataset,sources:HardSourceBundle,hard:HardRouteResult,low:LowFrequencyResult,stability:StabilityPartition,policies:PolicyBundle,backend:FrozenTask8BackendDecision,entry:dict[str,object])->dict[str,object]:
    _raw_identity(raw,entry); n=entry["test"]["count"]
    if not isinstance(sources,HardSourceBundle) or tuple(sources.source_names)!=EXPECTED_HARD_SOURCES or set(sources.by_name)!=set(EXPECTED_HARD_SOURCES):raise TypeError("typed Task4 source bundle is invalid")
    source_rows=[]
    for order,name in enumerate(EXPECTED_HARD_SOURCES):
        result=sources.by_name[name]
        if not isinstance(result,HardSourceResult):raise TypeError("typed Task4 source child is invalid")
        if result.name!=name or type(result.eligible)is not bool or type(result.eligibility_reason)is not str:raise ValueError("typed Task4 source result differs")
        score = _array(result.test_score,n,"source score") if result.eligible else ([] if np.asarray(result.test_score).size == 0 else (_ for _ in ()).throw(ValueError("source abstained score differs")))
        source_rows.append({"source_id":name,"order":order,"test_score":score,"pvalue":_array(result.test_pvalue,n,"source pvalue"),"native_pass":_bools(result.test_native_pass,n,"source pass"),"eligible":result.eligible,"reason":result.eligibility_reason})
    if not isinstance(hard,HardRouteResult) or hard.n_test!=n:raise TypeError("typed Task5 route result is invalid")
    hard_stages=[{"stage_id":"exp74d","order":0,"lanes":_lanes(hard.exp74d,n)},{"stage_id":"exp89","order":1,"indices":_indices(hard.exp89,n,"exp89")},{"stage_id":"exp90","order":2,"indices":_indices(hard.exp90,n,"exp90")},{"stage_id":"exp93","order":3,"indices":_indices(hard.exp93,n,"exp93")},{"stage_id":"exp137","order":4,"lanes":_lanes(hard.exp137,n)},{"stage_id":"exp234","order":5,"minimum_two":_indices(hard.exp234.minimum_two,n,"exp234"),"minimum_three":_indices(hard.exp234.minimum_three,n,"exp234")},{"stage_id":"exp236","order":6,"lanes":_lanes(hard.exp236,n)},{"stage_id":"exp253","order":7,"lanes":_lanes(hard.exp253,n)}]
    dataset_rows=tuple(row for row in low.rows if row.dataset==raw.dataset) if isinstance(low,LowFrequencyResult) else ()
    if not isinstance(low,LowFrequencyResult) or raw.dataset not in low.datasets or tuple(row.instance_index for row in dataset_rows)!=tuple(range(n)) or low.hard_lanes.get(raw.dataset)!=hard.exp253.hard:raise TypeError("typed Task6 result is invalid")
    low_rows=[{"index":row.instance_index,"evidence_missing":row.evidence_missing,"strict_pass":row.strict_pass,"provisional":row.provisional,"depths":[{"depth":item.depth,"score":item.score,"threshold":item.threshold,"strict_pass":item.strict_pass,"excess_ratio":item.excess_ratio,"empirical_rank":item.empirical_rank} for item in row.depth_evidence]} for row in dataset_rows]
    if not isinstance(stability,StabilityPartition) or tuple(stability.selected_coordinates)!=tuple(candidate.coordinate for candidate in low.selected_candidates):raise TypeError("typed Task7 stability differs")
    stability_record={"selected":[list(item) for item in stability.selected_coordinates if item[0]==raw.dataset],"stable":[list(item) for item in stability.stable_coordinates if item[0]==raw.dataset],"unstable":[list(item) for item in stability.unstable_coordinates if item[0]==raw.dataset]}
    if not isinstance(policies,PolicyBundle):raise TypeError("typed Task7 policy bundle is invalid")
    expected=(hard.automatic_hard,hard.automatic_hard|{index for dataset,index in stability.selected_coordinates if dataset==raw.dataset},hard.automatic_hard|{index for dataset,index in stability.stable_coordinates if dataset==raw.dataset},hard.automatic_hard|{index for dataset,index in stability.unstable_coordinates if dataset==raw.dataset})
    policy_rows=[]
    for order,(route,want) in enumerate(zip(policies.policies,expected,strict=True)):
        if route.name!=POLICIES[order] or set(route.alerts)!=set(low.datasets) or route.alerts[raw.dataset]!=want:raise ValueError("typed Task7 policy route differs")
        scores=_array(route.scores[raw.dataset],n,"policy scores"); policy_rows.append({"policy_id":route.name,"order":order,"alerts":sorted(want),"scores":scores,"score_mask":[score>=.5 for score in scores]})
    if not isinstance(backend,FrozenTask8BackendDecision):raise TypeError("typed backend decision is invalid")
    return {"schema":SHARD_SCHEMA,"dataset":raw.dataset,"train_identity":entry["train"],"test_identity":entry["test"],"source_rows":source_rows,"hard_stages":hard_stages,"low_frequency_rows":low_rows,"stability":stability_record,"policies":policy_rows,"backend":backend.artifact_record()}
def _validate_shard(value:object,entry:dict[str,object])->dict[str,object]:
    _finite(value)
    required={"schema","dataset","train_identity","test_identity","source_rows","hard_stages","low_frequency_rows","stability","policies","backend"}
    if not isinstance(value,dict) or set(value)!=required or value.get("schema")!=SHARD_SCHEMA or value.get("dataset")!=entry["dataset"] or value["train_identity"]!=entry["train"] or value["test_identity"]!=entry["test"]:raise ValueError("proposal shard identity differs")
    n=entry["test"]["count"]
    if not isinstance(value["source_rows"],list) or [row.get("source_id") if isinstance(row,dict) else None for row in value["source_rows"]]!=list(EXPECTED_HARD_SOURCES):raise ValueError("proposal source order differs")
    for order,row in enumerate(value["source_rows"]):
        if set(row)!={"source_id","order","test_score","pvalue","native_pass","eligible","reason"} or row["order"]!=order:_raise=ValueError("proposal source schema differs");raise _raise
        if row["eligible"] is True:_array(row["test_score"],n,"source score")
        elif row["eligible"] is False:
            if row["test_score"] != []:raise ValueError("proposal abstained source score differs")
        else:raise ValueError("proposal source eligibility differs")
        _array(row["pvalue"],n,"source pvalue");_bools(row["native_pass"],n,"source pass")
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
    try: FrozenTask8BackendDecision.from_artifact_record(value["backend"])
    except ValueError as error: raise ValueError("proposal backend differs") from error
    return value
def write_typed_proposal_shard(root:Path,catalog:dict[str,object],raw:RawDataset,sources:HardSourceBundle,hard:HardRouteResult,low:LowFrequencyResult,stability:StabilityPartition,policies:PolicyBundle,backend:FrozenTask8BackendDecision,*,progress:ProgressCallback|None=None)->Path:
    if not isinstance(raw,RawDataset):raise TypeError("typed RawDataset is required")
    entry=_catalog_entry(catalog,raw.dataset); payload=_typed_payload(raw,sources,hard,low,stability,policies,backend,entry); path=_shard_path(root,raw.dataset);write_create_only(path,canonical_json_bytes(payload))
    if progress is not None:
        emit_progress_paper(progress, {
            "stage": "shard_write", "phase": "tick",
            "completed": 1, "total": 1,
            "dataset": raw.dataset, "detail": f"{raw.dataset}_paper.json",
            "elapsed_s": 0.0,
        })
    return path
def _acceptance(path:Path,catalog:dict[str,object]):
    from .catalog_paper import load_acceptance
    accepted=load_acceptance(Path(path),catalog); raw=Path(path).read_bytes(); return accepted,raw
def freeze_score_root(root:Path,catalog:dict[str,object],acceptance_path:Path|None=None,*,progress:ProgressCallback|None=None)->Task8FrozenManifest:
    """Freeze typed score shards before, and independently from, historical parity."""
    root=safe_root(root); names=[entry["dataset"] for entry in catalog["datasets"]]; hashes={}; total=0
    started=time.monotonic(); dataset_total=len(names); finished=0
    emit_progress_paper(progress, {"stage": "freeze_verify", "phase": "start", "completed": 0, "total": dataset_total, "elapsed_s": 0.0})
    try:
        for entry in catalog["datasets"]:
            path=_shard_path(root,entry["dataset"]);_validate_shard(read_canonical(path),entry);hashes[entry["dataset"]]=sha256_bytes(path.read_bytes());total+=entry["test"]["count"]
            finished+=1
            emit_progress_paper(progress, {
                "stage": "freeze_verify", "phase": "tick",
                "completed": finished, "total": dataset_total,
                "dataset": entry["dataset"], "elapsed_s": float(time.monotonic() - started),
            })
        shard_dir=safe_child(root,ARTIFACT_DIRECTORY,FREEZE_STAGE,SHARD_DIRECTORY)
        if {path.name for path in shard_dir.iterdir()}!={f"{name}_paper.json" for name in names}:raise ValueError("proposal shard namespace differs")
        if catalog["mode"]=="full" and (len(names)!=1117 or total!=195900):raise ValueError("full proposal inventory differs")
        payload={"schema":MANIFEST_SCHEMA,"catalog_sha256":catalog["catalog_sha256"],"datasets":names,"test_row_count":total,"shard_hashes":hashes}
        if acceptance_path is None:
            payload.update({"parity_status":"not_available","acceptance_path":None,"acceptance_sha256":None,"acceptance_file_sha256":None})
        else:
            accepted,raw=_acceptance(acceptance_path,catalog)
            payload.update({"parity_status":"available_not_run","acceptance_path":str(Path(acceptance_path)),"acceptance_sha256":accepted["acceptance_sha256"],"acceptance_file_sha256":sha256_bytes(raw)})
        write_create_only(safe_child(root,ARTIFACT_DIRECTORY,FREEZE_STAGE,MANIFEST_NAME),canonical_json_bytes(payload))
        emit_progress_paper(progress, {"stage": "freeze_verify", "phase": "end", "completed": dataset_total, "total": dataset_total, "elapsed_s": float(time.monotonic() - started)})
    except Exception:
        emit_progress_paper(progress, {"stage": "freeze_verify", "phase": "error", "completed": finished, "total": dataset_total, "elapsed_s": float(time.monotonic() - started)})
        raise
    return Task8FrozenManifest(**{**payload,"datasets":tuple(names)})
def verify_score_root(root:Path,catalog:dict[str,object], expected_backend:FrozenTask8BackendDecision|None=None,*,progress:ProgressCallback|None=None)->Task8FrozenManifest:
    root=safe_root(root); path=safe_child(root,ARTIFACT_DIRECTORY,FREEZE_STAGE,MANIFEST_NAME); value=read_canonical(path); manifest=Task8FrozenManifest(**{**value,"datasets":tuple(value["datasets"])})
    if manifest.catalog_sha256!=catalog["catalog_sha256"]:raise ValueError("proposal manifest authority differs")
    if manifest.parity_status == "available_not_run":
        accepted,raw=_acceptance(Path(manifest.acceptance_path),catalog)
        if manifest.acceptance_sha256!=accepted["acceptance_sha256"] or manifest.acceptance_file_sha256!=sha256_bytes(raw):raise ValueError("proposal manifest authority differs")
    names=[entry["dataset"] for entry in catalog["datasets"]]; total=0
    started=time.monotonic(); dataset_total=len(names); finished=0
    emit_progress_paper(progress, {"stage": "freeze_verify", "phase": "start", "completed": 0, "total": dataset_total, "elapsed_s": 0.0})
    try:
        for entry in catalog["datasets"]:
            shard_path=_shard_path(root,entry["dataset"]); shard=_validate_shard(read_canonical(shard_path),entry)
            if expected_backend is not None and shard["backend"] != expected_backend.artifact_record():raise ValueError("proposal backend differs from verified acceleration assessment")
            if manifest.shard_hashes[entry["dataset"]]!=sha256_bytes(shard_path.read_bytes()):raise ValueError("proposal shard hash differs")
            total+=entry["test"]["count"]
            finished+=1
            emit_progress_paper(progress, {
                "stage": "freeze_verify", "phase": "tick",
                "completed": finished, "total": dataset_total,
                "dataset": entry["dataset"], "elapsed_s": float(time.monotonic() - started),
            })
        if manifest.datasets!=tuple(names) or manifest.test_row_count!=total:raise ValueError("proposal manifest total differs")
        directory=_freeze_root(root); shards=safe_child(root,ARTIFACT_DIRECTORY,FREEZE_STAGE,SHARD_DIRECTORY)
        if {item.name for item in directory.iterdir()}!={MANIFEST_NAME,SHARD_DIRECTORY} or {item.name for item in shards.iterdir()}!={f"{name}_paper.json" for name in names}:raise ValueError("proposal freeze namespace differs")
        emit_progress_paper(progress, {"stage": "freeze_verify", "phase": "end", "completed": dataset_total, "total": dataset_total, "elapsed_s": float(time.monotonic() - started)})
    except Exception:
        emit_progress_paper(progress, {"stage": "freeze_verify", "phase": "error", "completed": finished, "total": dataset_total, "elapsed_s": float(time.monotonic() - started)})
        raise
    return manifest
def verify_parity(root:Path,catalog:dict[str,object])->dict[str,object]:
    manifest=verify_score_root(root,catalog)
    if manifest.parity_status != "available_not_run":raise ValueError("parity unavailable without an authenticated acceptance oracle")
    accepted,raw=_acceptance(Path(manifest.acceptance_path),catalog)
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
